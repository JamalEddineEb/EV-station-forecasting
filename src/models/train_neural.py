import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import profiling
from models.profiling import profile


class DLinear(nn.Module):
    def __init__(self, seq_len, horizon, cal_dim, kernel=25):
        super().__init__()
        self.seasonal = nn.Linear(seq_len, horizon)
        self.trend = nn.Linear(seq_len, horizon)
        self.calendar = nn.Linear(cal_dim, 1)
        self.kernel = kernel
    def forward(self, hist, cal):
        x = hist.squeeze(-1)
        p = (self.kernel - 1) // 2
        xp = nn.functional.pad(x.unsqueeze(1), (p,p), mode="replicate")
        trend = nn.functional.avg_pool1d(xp, self.kernel, stride=1).squeeze(1)
        return self.seasonal(x-trend) + self.trend(trend) + self.calendar(cal).squeeze(-1)

class GRUForecast(nn.Module):
    def __init__(self, hidden, cal_dim, horizon):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden+cal_dim,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, hist, cal):
        _, h = self.gru(hist)
        context = h[-1].unsqueeze(1).expand(-1, cal.shape[1], -1)
        return self.head(torch.cat([context,cal],dim=-1)).squeeze(-1)

def make_tensors(hist, cal, y, mean, std):
    h = torch.from_numpy(((hist-mean)/std).astype(np.float32)).unsqueeze(-1)
    c = torch.from_numpy(cal.astype(np.float32))
    if y is None: return h,c
    return h,c,torch.from_numpy(((y-mean)/std).astype(np.float32))


@profile("train_epoch")
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    for h,c,y in loader:
        h,c,y = h.to(device),c.to(device),y.to(device)
        opt.zero_grad(); loss=loss_fn(model(h,c),y); loss.backward(); opt.step()


@profile("validate")
@torch.no_grad()
def validate(model, val_data, device, loss_fn, batch=2048):
    """Mean validation loss, computed in batches so large val sets don't OOM."""
    model.eval(); h,c,y = val_data; total = 0.0
    for i in range(0, len(h), batch):
        hb,cb,yb = h[i:i+batch].to(device), c[i:i+batch].to(device), y[i:i+batch].to(device)
        total += loss_fn(model(hb,cb),yb).item() * len(hb)
    return total / len(h)


@profile("fit")
def fit(model, loader, val_data, device, epochs, patience, lr, name):
    """Train with early stopping. Call profiling.tag(name) first to label the measurements."""
    model.to(device); opt=torch.optim.Adam(model.parameters(),lr=lr); loss_fn=nn.MSELoss()
    val_data=[x.to(device) for x in val_data]
    best=float("inf"); best_state=None; stale=0
    for epoch in range(epochs):
        train_epoch(model, loader, opt, loss_fn, device)
        vl = validate(model, val_data, device, loss_fn)
        print("{} epoch {:02d}: val_mse={:.6f}".format(name,epoch+1,vl),flush=True)
        if vl<best:
            best=vl; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=patience: break
    if best_state is None:
        raise RuntimeError("{}: validation loss never improved (NaN/inf?)".format(name))
    model.load_state_dict(best_state)
    return model


@profile("predict")
@torch.no_grad()
def predict(model, data, device, mean, std, batch=512):
    model.eval(); h,c=data; out=[]
    for i in range(0,len(h),batch):
        out.append(model(h[i:i+batch].to(device),c[i:i+batch].to(device)).cpu().numpy())
    return np.concatenate(out)*std+mean


def save_checkpoint(model, name, outdir, mean, std, meta):
    """Save everything needed to rebuild the model, so benchmarks never need to retrain."""
    path = os.path.join(outdir, name.lower()+".pt")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"name": name, "state_dict": state, "mean": mean, "std": std, **meta}, path)
    return path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="outputs/caltech_occupancy_data.npz")
    p.add_argument("--outdir",default="outputs")
    p.add_argument("--metrics-dir",default="outputs/metrics")
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--patience",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--hidden",type=int,default=64)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--energy",action="store_true",help="estimate energy with codecarbon (optional)")
    a=p.parse_args()
    profiling.ENERGY = a.energy
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu=torch.cuda.get_device_name(0) if device.type=="cuda" else None
    print("Device:",device,flush=True)
    if gpu: print("GPU:",gpu,flush=True)
    with np.load(a.data,allow_pickle=False) as d:
        th,tc,ty=d["train_X_hist"],d["train_X_cal"],d["train_y"]
        vh,vc,vy=d["val_X_hist"],d["val_X_cal"],d["val_y"]
        sh,sc,sy=d["test_X_hist"],d["test_X_cal"],d["test_y"]
    mean=float(th.mean()); std=max(float(th.std()),1e-6)
    tr=make_tensors(th,tc,ty,mean,std); va=make_tensors(vh,vc,vy,mean,std); te=make_tensors(sh,sc,None,mean,std)
    loader=DataLoader(TensorDataset(*tr),batch_size=a.batch_size,shuffle=True,num_workers=0)
    os.makedirs(a.outdir,exist_ok=True)
    L,H,C = th.shape[1], ty.shape[1], tc.shape[2]
    models=[("DLinear",DLinear(L,H,C)),("GRU",GRUForecast(a.hidden,C,H))]
    for name,model in models:
        profiling.tag(name)
        model=fit(model,loader,va,device,a.epochs,a.patience,a.lr,name)
        vp=predict(model,(va[0],va[1]),device,mean,std); sp=predict(model,te,device,mean,std)
        out=os.path.join(a.outdir,name.lower()+"_predictions.npz")
        np.savez_compressed(out,val_pred=vp.astype(np.float32),test_pred=sp.astype(np.float32),y_val=vy,y_test=sy)
        ckpt=save_checkpoint(model,name,a.outdir,mean,std,
                             {"hidden":a.hidden,"seq_len":L,"horizon":H,"calendar_dim":C})
        profiling.dump(os.path.join(a.metrics_dir,name.lower()+"_train.json"),prefix=name,extra={
            "model":name,"parameters":sum(q.numel() for q in model.parameters()),
            "checkpoint_bytes":os.path.getsize(ckpt),"batch_size":a.batch_size,
            "epochs_run":len(profiling.records("train_epoch")),"device":str(device),"gpu":gpu,
        })
        print("Saved {} ({:.1f}s fit): {}".format(name,profiling.total("fit"),out),flush=True)

if __name__=="__main__": main()
