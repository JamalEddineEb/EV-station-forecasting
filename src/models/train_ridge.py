import argparse, os
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from models import profiling
from models.profiling import profile


def make_features(hist, cal):
    return np.concatenate([hist, cal.reshape(len(cal), -1)], axis=1)


@profile("features")
def prepare(th, tc, vh, vc, sh, sc):
    scaler = StandardScaler()
    xt = scaler.fit_transform(make_features(th, tc))
    xv = scaler.transform(make_features(vh, vc))
    xs = scaler.transform(make_features(sh, sc))
    return scaler, xt, xv, xs


@profile("fit")
def fit_ridge(xt, ty, alpha, solver, threads):
    model = Ridge(alpha=alpha, solver=solver)
    with threadpool_limits(limits=threads):
        model.fit(xt, ty)
    return model


@profile("predict")
def predict_ridge(model, x, threads):
    with threadpool_limits(limits=threads):
        return model.predict(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/caltech_occupancy_data.npz")
    p.add_argument("--out", default="outputs/ridge_predictions.npz")
    p.add_argument("--model-out", default="outputs/ridge.joblib")
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--solver", default="lsqr", choices=["auto","svd","cholesky","lsqr","sparse_cg","sag","saga","lbfgs"])
    p.add_argument("--threads", type=int, default=1, help="BLAS threads (benchmark default 1 to match torch)")
    p.add_argument("--energy", action="store_true", help="estimate energy with codecarbon (optional)")
    a = p.parse_args()
    profiling.ENERGY = a.energy
    profiling.tag("Ridge")
    with np.load(a.data, allow_pickle=False) as d:
        th, tc, ty = d["train_X_hist"], d["train_X_cal"], d["train_y"]
        vh, vc, vy = d["val_X_hist"], d["val_X_cal"], d["val_y"]
        sh, sc, sy = d["test_X_hist"], d["test_X_cal"], d["test_y"]
    scaler, xt, xv, xs = prepare(th, tc, vh, vc, sh, sc)
    print("Feature preparation: {:.2f}s".format(profiling.total("features")), flush=True)
    model = fit_ridge(xt, ty, a.alpha, a.solver, a.threads)
    print("Fit: {:.2f}s".format(profiling.total("fit")), flush=True)
    vp, sp = predict_ridge(model, xv, a.threads), predict_ridge(model, xs, a.threads)
    print("Prediction: {:.3f}s".format(profiling.total("predict")), flush=True)
    for path in (a.out, a.model_out):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(a.out, val_pred=vp.astype(np.float32), test_pred=sp.astype(np.float32), y_val=vy, y_test=sy)
    joblib.dump({"scaler": scaler, "model": model}, a.model_out)
    profiling.dump(os.path.join(a.metrics_dir, "ridge_train.json"), prefix="Ridge", extra={
        "model": "Ridge", "parameters": int(model.coef_.size + model.intercept_.size),
        "checkpoint_bytes": os.path.getsize(a.model_out), "threads": a.threads,
        "train_samples": len(th),
    })
    print("Saved", a.out, flush=True)

if __name__ == "__main__":
    main()
