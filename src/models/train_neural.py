import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import profiling
from models.profiling import profile

# Each model outputs three quantiles per horizon step: the 10th-90th range is the 80% interval,
# the median is the point forecast. Output shape: (batch, horizon, 3), sorted along the last axis.
QUANTILES = (0.1, 0.5, 0.9)
LO, MID, HI = 0, 1, 2
N_Q = len(QUANTILES)

# Per-window normalization: every input window is divided by its own recent occupancy level, the model
# works on relative occupancy, and outputs are multiplied back. Intervals therefore scale with the level.
SCALE_WINDOW = 96  # last day at 15-min resolution
SCALE_EPS = 0.5    # keeps empty windows from blowing up the division (same unit as occupancy)


def window_scale(hist):
    """Per-window scale, shape (N, 1): mean occupancy over the last SCALE_WINDOW steps + SCALE_EPS."""
    return hist[:, -SCALE_WINDOW:].mean(axis=1, keepdims=True) + SCALE_EPS


class DLinear(nn.Module):
    def __init__(self, seq_len, horizon, cal_dim, kernel=25, n_q=N_Q):
        super().__init__()
        self.seasonal = nn.Linear(seq_len, horizon*n_q)
        self.trend = nn.Linear(seq_len, horizon*n_q)
        self.calendar = nn.Linear(cal_dim, n_q)
        self.kernel, self.horizon, self.n_q = kernel, horizon, n_q
    def forward(self, hist, cal):
        x = hist.squeeze(-1)
        p = (self.kernel - 1) // 2
        xp = nn.functional.pad(x.unsqueeze(1), (p,p), mode="replicate")
        trend = nn.functional.avg_pool1d(xp, self.kernel, stride=1).squeeze(1)
        out = (self.seasonal(x-trend) + self.trend(trend)).reshape(-1, self.horizon, self.n_q)
        return torch.sort(out + self.calendar(cal), dim=-1).values  # sorted: no crossing quantiles

class GRUForecast(nn.Module):
    def __init__(self, hidden, cal_dim, horizon, n_q=N_Q):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden+cal_dim,64), nn.ReLU(), nn.Linear(64,n_q))
    def forward(self, hist, cal):
        _, h = self.gru(hist)
        context = h[-1].unsqueeze(1).expand(-1, cal.shape[1], -1)
        return torch.sort(self.head(torch.cat([context,cal],dim=-1)), dim=-1).values

class PinballLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles))
    def forward(self, pred, y):
        err = y.unsqueeze(-1) - pred
        return torch.maximum(self.q*err, (self.q-1)*err).mean()

def make_tensors(hist, cal, y):
    s = window_scale(hist)
    h = torch.from_numpy((hist/s).astype(np.float32)).unsqueeze(-1)
    c = torch.from_numpy(cal.astype(np.float32))
    if y is None: return h,c
    return h,c,torch.from_numpy((y/s).astype(np.float32))


@profile("train_epoch")
def train_epoch(model, loader, opt, loss_fn, device):
    """One pass over the training data; returns the mean training loss (running average over the epoch)."""
    model.train(); total = torch.zeros((), device=device)
    for h,c,y in loader:
        h,c,y = h.to(device),c.to(device),y.to(device)
        opt.zero_grad(); loss=loss_fn(model(h,c),y); loss.backward(); opt.step()
        total += loss.detach() * len(h)  # stays on device: no per-step sync
    return total.item() / len(loader.dataset)


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
    """Train with pinball loss and early stopping; returns (model, history).

    history has one {epoch, train_loss, val_loss} record per epoch (pinball loss on the normalized scale).
    Call profiling.tag(name) first to label measurements.
    """
    model.to(device); opt=torch.optim.Adam(model.parameters(),lr=lr); loss_fn=PinballLoss(QUANTILES).to(device)
    val_data=[x.to(device) for x in val_data]
    best=float("inf"); best_state=None; stale=0; history=[]
    for epoch in range(epochs):
        tl = train_epoch(model, loader, opt, loss_fn, device)
        vl = validate(model, val_data, device, loss_fn)
        history.append({"epoch": epoch+1, "train_loss": tl, "val_loss": vl})
        print("{} epoch {:02d}: train_pinball={:.6f} val_pinball={:.6f}".format(name,epoch+1,tl,vl),flush=True)
        if vl<best:
            best=vl; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=patience: break
    if best_state is None:
        raise RuntimeError("{}: validation loss never improved (NaN/inf?)".format(name))
    model.load_state_dict(best_state)
    return model, history


@profile("predict")
@torch.no_grad()
def predict(model, data, device, scale, batch=512):
    """Quantile forecasts in original units, shape (N, horizon, 3). `scale` = window_scale(history)."""
    model.eval(); h,c=data; out=[]
    for i in range(0,len(h),batch):
        out.append(model(h[i:i+batch].to(device),c[i:i+batch].to(device)).cpu().numpy())
    return np.concatenate(out)*scale[:, :, None]


def save_checkpoint(model, name, outdir, meta):
    """Save everything needed to rebuild the model, so benchmarks never need to retrain."""
    path = os.path.join(outdir, name.lower()+".pt")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"name": name, "state_dict": state, "quantiles": QUANTILES,
                "scale_window": SCALE_WINDOW, "scale_eps": SCALE_EPS, **meta}, path)
    return path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="outputs/caltech_occupancy_data.npz")
    p.add_argument("--outdir",default="outputs")
    p.add_argument("--metrics-dir",default="outputs/metrics")
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--patience",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--lr",type=float,default=1e-4)
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
    tr=make_tensors(th,tc,ty); va=make_tensors(vh,vc,vy); te=make_tensors(sh,sc,None)
    loader=DataLoader(TensorDataset(*tr),batch_size=a.batch_size,shuffle=True,num_workers=0)
    os.makedirs(a.outdir,exist_ok=True)
    L,H,C = th.shape[1], ty.shape[1], tc.shape[2]
    models=[("DLinear",DLinear(L,H,C)),("GRU",GRUForecast(a.hidden,C,H))]
    for name,model in models:
        profiling.tag(name)
        model,history=fit(model,loader,va,device,a.epochs,a.patience,a.lr,name)
        vp=predict(model,(va[0],va[1]),device,window_scale(vh)); sp=predict(model,te,device,window_scale(sh))
        out=os.path.join(a.outdir,name.lower()+"_predictions.npz")
        arrays={}
        for split,q in (("val",vp),("test",sp)):
            arrays.update({split+"_lower":q[...,LO], split+"_pred":q[...,MID], split+"_upper":q[...,HI]})
        np.savez_compressed(out,y_val=vy,y_test=sy,**{k:v.astype(np.float32) for k,v in arrays.items()})
        ckpt=save_checkpoint(model,name,a.outdir,
                             {"hidden":a.hidden,"seq_len":L,"horizon":H,"calendar_dim":C})
        verr=vy-vp[...,MID]
        profiling.dump(os.path.join(a.metrics_dir,name.lower()+"_train.json"),prefix=name,extra={
            "model":name,"parameters":sum(q.numel() for q in model.parameters()),
            "checkpoint_bytes":os.path.getsize(ckpt),"batch_size":a.batch_size,"lr":a.lr,
            "epochs_run":len(history),"best_val_pinball":min(r["val_loss"] for r in history),
            "val_mae":float(np.abs(verr).mean()),"val_rmse":float(np.sqrt((verr**2).mean())),
            "history":history,"device":str(device),"gpu":gpu,
        })
        print("Saved {} ({:.1f}s fit): {}".format(name,profiling.total("fit"),out),flush=True)

if __name__=="__main__": main()