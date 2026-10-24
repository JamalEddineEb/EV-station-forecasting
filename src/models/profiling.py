"""Tiny profiling helpers: wall/CPU time, per-call peak RSS, CUDA peak memory, optional energy.

Use as a decorator or as a context manager:

    @profile("fit")
    def fit(...): ...

    with profile("epoch"):
        ...

Records accumulate in RESULTS under "<tag>/<label>" (set the tag with tag("GRU")
before the call) and can be written to JSON with dump().

Peak RSS is sampled from a background thread, so it is a per-call peak (unlike
ru_maxrss, a process-lifetime high-water mark). Spikes shorter than the sampling
interval can be missed. A profile object is not re-entrant (don't recurse with
the same label). Energy needs `pip install codecarbon`; on machines without
power counters it falls back to a TDP-based estimate, so treat it as an estimate.
"""
import json
import os
import threading
import time
from contextlib import ContextDecorator

import psutil

try:
    import torch
except ImportError:  # profiling still works without torch
    torch = None

ENERGY = False  # set True (e.g. from --energy) to measure energy with codecarbon
RESULTS = {}
_TAG = ""


def tag(name):
    global _TAG
    _TAG = name


def _key(label):
    return "{}/{}".format(_TAG, label) if _TAG else label


def reset():
    RESULTS.clear()


def records(label):
    """All records for `label` under the current tag, in call order."""
    return RESULTS.get(_key(label), [])


def total(label, field="wall_s"):
    return sum(r[field] for r in records(label))


def dump(path, prefix=None, extra=None):
    """Write records (optionally only those tagged `prefix`) plus `extra` metadata to JSON."""
    data = {k: v for k, v in RESULTS.items() if prefix is None or k.startswith(prefix + "/")}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"profile": data, **(extra or {})}, f, indent=2)


class profile(ContextDecorator):
    def __init__(self, label, energy=None, sample_s=0.01):
        self.label = label
        self.energy = energy
        self.sample_s = sample_s

    def _sample(self):
        while not self._stop.wait(self.sample_s):
            self._peak = max(self._peak, self._proc.memory_info().rss)

    def __enter__(self):
        self._key = _key(self.label)
        self._cuda = torch is not None and torch.cuda.is_available()
        if self._cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._proc = psutil.Process()
        self._rss0 = self._peak = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._sampler = threading.Thread(target=self._sample, daemon=True)
        self._sampler.start()
        self._tracker = None
        if self.energy if self.energy is not None else ENERGY:
            try:
                from codecarbon import EmissionsTracker

                self._tracker = EmissionsTracker(save_to_file=False, log_level="error")
                self._tracker.start()
            except Exception:  # noqa: BLE001 - energy is strictly optional
                self._tracker = None
        self._wall0 = time.perf_counter()
        self._cpu0 = time.process_time()
        return self

    def __exit__(self, *exc):
        if self._cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - self._wall0
        cpu = time.process_time() - self._cpu0
        self._stop.set()
        self._sampler.join()
        self._peak = max(self._peak, self._proc.memory_info().rss)
        rec = {
            "wall_s": wall,
            "cpu_s": cpu,  # all threads of this process
            "rss_peak_mib": self._peak / 2**20,
            "rss_delta_mib": (self._peak - self._rss0) / 2**20,
        }
        if self._cuda:
            rec["gpu_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
        if self._tracker is not None:
            try:
                rec["co2_kg"] = self._tracker.stop()
                rec["energy_kwh"] = getattr(self._tracker.final_emissions_data, "energy_consumed", None)
            except Exception:  # noqa: BLE001
                pass
        RESULTS.setdefault(self._key, []).append(rec)
        return False
