"""Benchmark training efficiency and inference performance of the forecasting models.

Each model runs in its own subprocess, so peak-RSS numbers are not contaminated by
earlier models. Reuses fit/DLinear/GRUForecast, make_features and the baselines.

    python -m src.benchmark_models --data outputs/caltech_occupancy_data.npz
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import copy
import io
import json
import platform
import subprocess
import sys
import time

import joblib
import numpy as np
import psutil
import sklearn
import torch
import torch.nn as nn
from threadpoolctl import threadpool_limits
from torch.ao.quantization import quantize_dynamic
from torch.utils.data import DataLoader, TensorDataset

from model_selection.baselines import persistence, seasonal_naive
from models import profiling
from models.train_neural import DLinear, GRUForecast, fit, make_tensors
from models.train_ridge import fit_ridge, make_features, prepare

WORKERS = ("dlinear", "gru", "ridge", "baselines")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="outputs/caltech_occupancy_data.npz")
    p.add_argument("--outdir", default="outputs/benchmark")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inference-batch-size", type=int, default=64)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--reuse-raw", action="store_true",
                   help="skip running workers; rebuild results/report from existing _raw_*.json")
    p.add_argument("--worker", choices=WORKERS, help=argparse.SUPPRESS)
    return p.parse_args()


# ----------------------------------------------------------------- helpers
def pick_device(a):
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    use_cuda = a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())
    return torch.device("cuda", torch.cuda.current_device()) if use_cuda else torch.device("cpu")


def rss_mib():
    return psutil.Process().memory_info().rss / 2**20


def serialized_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def load(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def hardware(device):
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor() or platform.machine()
    hw = {
        "cpu": cpu,
        "logical_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        hw["gpu"] = torch.cuda.get_device_name(device)
        hw["gpu_total_gib"] = round(torch.cuda.mem_get_info(device)[1] / 2**30, 2)
    return hw


def analytic_flops(name, L, H, C, hidden):
    """Hand-derived forward FLOPs/sample (2 per multiply-add; ignores activations)."""
    if name == "dlinear":
        return 4 * L * H + 2 * H * C + 25 * L  # two Linear(L,H), calendar, avg-pool
    gru = 6 * hidden * (1 + hidden) * L  # 3 gates x 2 x (in*h + h*h) per step
    head = H * (2 * (hidden + C) * 64 + 2 * 64)  # Linear(h+C,64) + Linear(64,1) per step
    return gru + head


# ------------------------------------------------------- inference benchmark
def torch_fn(model, device, mean, std):
    def f(hist, cal):
        x = torch.from_numpy(((hist - mean) / std).astype(np.float32)).unsqueeze(-1).to(device)
        k = torch.from_numpy(cal.astype(np.float32)).to(device)
        with torch.inference_mode():
            y = model(x, k)
        return y.cpu().numpy() * std + mean

    return f


def nop():
    pass


def bench(fn, sync, is_cuda, d, a):
    """Latency/throughput of fn(hist, cal) including NumPy preprocessing (and H2D for CUDA)."""
    hist, cal = d["val_X_hist"], d["val_X_cal"]
    out = {}
    if is_cuda:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    for label, bs in (("single", 1), ("batch", min(a.inference_batch_size, len(hist)))):
        span = len(hist) - bs + 1
        for i in range(a.warmup):
            s = (i * bs) % span
            fn(hist[s:s + bs], cal[s:s + bs])
            sync()
        lat = []
        wall0 = time.perf_counter()
        for i in range(a.iterations):
            s = (i * bs) % span
            t = time.perf_counter()
            fn(hist[s:s + bs], cal[s:s + bs])
            sync()
            lat.append(time.perf_counter() - t)
        wall = time.perf_counter() - wall0
        out[label] = {
            "batch_size": bs,
            "p50_ms": float(np.percentile(lat, 50) * 1000),
            "p90_ms": float(np.percentile(lat, 90) * 1000),
            "throughput_sps": bs * a.iterations / wall,
        }
    out["rss_mib"] = rss_mib()
    if is_cuda:
        out["gpu_peak_incremental_mib"] = (torch.cuda.max_memory_allocated() - before) / 2**20
    return out


def evaluate(fn, d, chunk=512):
    hist, cal, y = d["test_X_hist"], d["test_X_cal"], d["test_y"]
    pred = np.concatenate([fn(hist[i:i + chunk], cal[i:i + chunk]) for i in range(0, len(hist), chunk)])
    err = y - pred
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err**2).mean()))}


def measure(runtimes, d, a):
    out = {}
    for name, (fn, sync, is_cuda, size) in runtimes.items():
        r = bench(fn, sync, is_cuda, d, a)
        r["accuracy"] = evaluate(fn, d)
        if size is not None:
            r["size_kib"] = size / 1024
        out[name] = r
    return out


# ---------------------------------------------------------------- workers
def run_neural(name, a, d, device):
    th, tc, ty = d["train_X_hist"], d["train_X_cal"], d["train_y"]
    L, H, C = th.shape[1], ty.shape[1], tc.shape[2]
    mean, std = float(th.mean()), max(float(th.std()), 1e-6)
    train = make_tensors(th, tc, ty, mean, std)
    val = make_tensors(d["val_X_hist"], d["val_X_cal"], d["val_y"], mean, std)
    loader = DataLoader(TensorDataset(*train), batch_size=a.batch_size, shuffle=True, num_workers=0)
    model = DLinear(L, H, C) if name == "dlinear" else GRUForecast(a.hidden, C, H)

    profiling.reset()
    profiling.tag(name)
    model = fit(model, loader, val, device, a.epochs, a.patience, a.lr, name)  # instrumented via @profile
    fit_rec = profiling.records("fit")[0]
    epoch_s = [t["wall_s"] + v["wall_s"] for t, v in zip(profiling.records("train_epoch"), profiling.records("validate"))]
    training = {
        "total_s": fit_rec["wall_s"],
        "epochs": len(epoch_s),
        "s_per_epoch": float(np.mean(epoch_s)),
        "samples_per_s": len(th) * len(epoch_s) / sum(epoch_s),
        "rss_growth_mib": fit_rec["rss_delta_mib"],
        "peak_gpu_mib": fit_rec.get("gpu_peak_mib"),
        "energy_kwh": fit_rec.get("energy_kwh"),
    }

    model = model.cpu().eval()
    errors, runtimes = {}, {}
    runtimes["cpu_fp32"] = (torch_fn(model, torch.device("cpu"), mean, std), nop, False, serialized_bytes(model))
    try:
        q = quantize_dynamic(copy.deepcopy(model), {nn.Linear, nn.GRU}, dtype=torch.qint8).eval()
        runtimes["cpu_int8"] = (torch_fn(q, torch.device("cpu"), mean, std), nop, False, serialized_bytes(q))
    except Exception as e:  # noqa: BLE001
        errors["cpu_int8"] = repr(e)[:200]
    if device.type == "cuda":
        gpu = copy.deepcopy(model).to(device).eval()
        runtimes["cuda_fp32"] = (torch_fn(gpu, device, mean, std), lambda: torch.cuda.synchronize(), True, None)

    return {name: {
        "parameters": sum(p.numel() for p in model.parameters()),
        "disk_mib": serialized_bytes(model) / 2**20,
        "flops_analytic": analytic_flops(name, L, H, C, a.hidden),
        "training": training,
        "runtimes": measure(runtimes, d, a),
        "errors": errors,
    }}


def run_ridge(a, d):
    th, tc, ty = d["train_X_hist"], d["train_X_cal"], d["train_y"]
    profiling.reset()
    profiling.tag("ridge")
    scaler, xt, _, _ = prepare(th, tc, d["val_X_hist"], d["val_X_cal"], d["test_X_hist"], d["test_X_cal"])
    model = fit_ridge(xt, ty, a.alpha, "lsqr", a.cpu_threads)
    prep_rec, fit_rec = profiling.records("features")[0], profiling.records("fit")[0]
    path = os.path.join(a.outdir, "ridge.joblib")
    os.makedirs(a.outdir, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, path)

    def fn(hist, cal):
        return model.predict(scaler.transform(make_features(hist, cal)))

    with threadpool_limits(limits=a.cpu_threads):
        runtimes = measure({"cpu_fp32": (fn, nop, False, None)}, d, a)
    return {"ridge": {
        "parameters": int(model.coef_.size + model.intercept_.size),
        "disk_mib": os.path.getsize(path) / 2**20,
        "flops_analytic": 2 * xt.shape[1] * ty.shape[1],
        "training": {
            "total_s": fit_rec["wall_s"],  # solver only; feature prep reported separately
            "feature_prep_s": prep_rec["wall_s"],
            "samples_per_s": len(th) / fit_rec["wall_s"],
            "rss_growth_mib": fit_rec["rss_delta_mib"],
            "energy_kwh": fit_rec.get("energy_kwh"),
        },
        "runtimes": runtimes,
        "errors": {},
    }}


def run_baselines(a, d):
    H = d["test_y"].shape[1]
    out = {}
    for key, func in (("persistence", persistence), ("seasonal_naive", seasonal_naive)):
        fn = lambda hist, cal, f=func: f(hist, H)  # noqa: E731
        out[key] = {
            "parameters": 0, "disk_mib": 0.0, "flops_analytic": 0,
            "training": {"total_s": 0.0}, "errors": {},
            "runtimes": measure({"cpu_numpy": (fn, nop, False, None)}, d, a),
        }
    return out


def worker_main(a):
    torch.set_num_threads(a.cpu_threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a)
    hw = hardware(device)  # fail fast, before any long-running work
    print("[{}] device={} threads={}".format(a.worker, device, a.cpu_threads), flush=True)
    d = load(a.data)
    print("[{}] data loaded: {} train examples".format(a.worker, len(d["train_X_hist"])), flush=True)
    if a.worker in ("dlinear", "gru"):
        models = run_neural(a.worker, a, d, device)
    elif a.worker == "ridge":
        models = run_ridge(a, d)
    else:
        models = run_baselines(a, d)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "_raw_{}.json".format(a.worker)), "w") as f:
        json.dump({"models": models, "hardware": hw}, f)


# ----------------------------------------------------------------- report
def fmt(x, spec):
    return "n/a" if x is None else format(x, spec)


def render(res):
    hw, ds, st = res["hardware"], res["dataset"], res["settings"]
    L = [
        "# Forecast model benchmark", "",
        "Generated: " + res["generated_at"], "",
        "## Setup", "",
        "- CPU: {} ({} logical CPUs available, {} thread(s) used).".format(
            hw["cpu"], hw["logical_cpus"], st["cpu_threads"]),
        "- Training device: {}{}.".format(
            hw["device"], " ({}, {} GiB)".format(hw["gpu"], hw["gpu_total_gib"]) if "gpu" in hw else ""),
        "- Data: {} train / {} val / {} test; input length {}, horizon {}.".format(
            ds["train"], ds["val"], ds["test"], ds["input_length"], ds["horizon"]),
        "- Batch size: training {}, inference {}. Latency = {} timed calls after {} warmup calls, "
        "including NumPy preprocessing (and H2D/D2H for CUDA, with synchronization).".format(
            st["batch_size"], st["inference_batch_size"], st["iterations"], st["warmup"]),
        "- Versions: Python {}, PyTorch {}, scikit-learn {}.".format(hw["python"], hw["torch"], hw["scikit_learn"]),
        "", "## Training efficiency", "",
        "| Model | Total fit (s) | Epochs | s/epoch | Samples/s | Peak GPU (MiB) | RSS growth (MiB) | Params | Size (MiB) | GFLOPs/sample (analytic) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in res["models"].items():
        t = m["training"]
        L.append("| {} | {} | {} | {} | {} | {} | {} | {:,} | {:.4f} | {:.6f} |".format(
            name, fmt(t.get("total_s"), ".2f"), fmt(t.get("epochs"), "d"), fmt(t.get("s_per_epoch"), ".3f"),
            fmt(t.get("samples_per_s"), ".0f"), fmt(t.get("peak_gpu_mib"), ".1f"),
            fmt(t.get("rss_growth_mib"), ".1f"), m["parameters"], m["disk_mib"],
            m["flops_analytic"] / 1e9))
    L += [
        "",
        "Neural fit time includes per-epoch validation and early stopping (first CUDA epoch includes "
        "warmup). Ridge time is the solver only; feature preparation is in the JSON. RSS growth is the "
        "process peak minus RSS after data loading (each model runs in its own process). FLOPs are hand-derived "
        "forward-pass estimates (2 per multiply-add, activations ignored).",
        "", "## Inference", "",
        "| Model / runtime | Single p50 (ms) | Single p90 (ms) | Batch | Batch p50 (ms) | Batch p90 (ms) | Samples/s (batch) | RSS (MiB) | GPU peak (+MiB) | Weights (KiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in res["models"].items():
        for rt, r in m["runtimes"].items():
            s, b = r["single"], r["batch"]
            L.append("| {}/{} | {:.4f} | {:.4f} | {} | {:.4f} | {:.4f} | {:.0f} | {:.0f} | {} | {} |".format(
                name, rt, s["p50_ms"], s["p90_ms"], b["batch_size"], b["p50_ms"], b["p90_ms"],
                b["throughput_sps"], r["rss_mib"], fmt(r.get("gpu_peak_incremental_mib"), ".1f"),
                fmt(r.get("size_kib"), ".1f")))
    L += ["", "## Test accuracy and effect of optimizations", "",
          "| Model / runtime | MAE | RMSE | dMAE vs first runtime | dRMSE vs first runtime |", "|---|---:|---:|---:|---:|"]
    for name, m in res["models"].items():
        ref = next(iter(m["runtimes"].values()))["accuracy"]
        for rt, r in m["runtimes"].items():
            acc = r["accuracy"]
            L.append("| {}/{} | {:.5f} | {:.5f} | {:+.5f} | {:+.5f} |".format(
                name, rt, acc["mae"], acc["rmse"], acc["mae"] - ref["mae"], acc["rmse"] - ref["rmse"]))
    L += ["", "## Optimizations attempted", ""]
    for name, m in res["models"].items():
        if name in ("dlinear", "gru"):
            status = "measured" if "cpu_int8" in m["runtimes"] else "failed: " + m["errors"].get("cpu_int8", "unknown")
            L.append("- {} dynamic INT8: {}".format(name, status))
    L += ["- ONNX export and distillation: not attempted. Energy/cost: not measured (no power telemetry).", "",
          "Numbers are specific to this machine and software stack; rerun on the deployment target.", ""]
    return "\n".join(L)


def driver_main(a):
    os.makedirs(a.outdir, exist_ok=True)
    argv = []
    for k, v in vars(a).items():
        if k not in ("worker", "reuse_raw"):
            argv += ["--" + k.replace("_", "-"), str(v)]
    launcher = [sys.executable, "-m", __spec__.name] if __spec__ else [sys.executable, os.path.abspath(__file__)]
    models, hw = {}, None
    for w in WORKERS:
        if not a.reuse_raw:
            print("== {} ==".format(w), flush=True)
            subprocess.run(launcher + argv + ["--worker", w], check=True)
        with open(os.path.join(a.outdir, "_raw_{}.json".format(w))) as f:
            raw = json.load(f)
        models.update(raw["models"])
        hw = hw or raw["hardware"]
    with np.load(a.data, allow_pickle=False) as d:
        ds = {"train": len(d["train_X_hist"]), "val": len(d["val_X_hist"]), "test": len(d["test_X_hist"]),
              "input_length": d["train_X_hist"].shape[1], "horizon": d["train_y"].shape[1]}
    res = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "hardware": hw, "dataset": ds, "models": models,
        "settings": {k: v for k, v in vars(a).items() if k not in ("worker", "reuse_raw")},
    }
    with open(os.path.join(a.outdir, "benchmark_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(a.outdir, "benchmark_report.md"), "w") as f:
        f.write(render(res))
    print("Saved report to", os.path.join(a.outdir, "benchmark_report.md"))


if __name__ == "__main__":
    args = parse_args()
    worker_main(args) if args.worker else driver_main(args)