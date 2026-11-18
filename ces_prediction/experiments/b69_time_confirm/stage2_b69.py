"""B.69 second stage (PREREGISTRATION_B69.md).  CPU only; no network is trained here.

The second stage is B.64's, imported, not copied: `stage2_b64.fit_fold` runs unchanged.  It reaches the pooled split
only through five module-level names of stage2_b64 (`Paths`, `manifest`, `P`, `B39`, `B54`); for the call they are bound
to the temporal split of B.5 and restored afterwards (`bound`).  Per seed s the B.64 code then
  - fits B.38's model (`run_b37.gbr` on `run_b37.inputs`, the network on fifteen inputs, the twin on the seven
    history-side ones) on the out-of-fold rows of the seed's training discharges (the inner networks of run_b69.py)
    together with its validation discharges (predicted by the seed's outer network);
  - applies it once to the scored rows of the test block (the rows of the outer run's comparison_errors_test.npz);
  - keeps the first stage in BOTH arms on test rows where the network's second-stage inputs are missing (B.64,
    addendum 1).
The fold key of B.64's file names is (r, k) = (seed, "camp"): data/.b69/stage2_r{s}_fcamp.npz.

Stages (repo root; each caches under data/.b69/ and is resumable):
    py ces_prediction/experiments/b69_time_confirm/stage2_b69.py base     # per-file base rows (stage2_b64.stage_base)
    py ces_prediction/experiments/b69_time_confirm/stage2_b69.py rows     # row table of each seed's outer runs (B.54)
    py ces_prediction/experiments/b69_time_confirm/stage2_b69.py fit      # stage2_b64.fit_fold per seed
    py ces_prediction/experiments/b69_time_confirm/stage2_b69.py all      # the three in order
Then: py ces_prediction/experiments/b69_time_confirm/analyze_b69.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the GPU belongs to the training batches
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b64_two_stage"))
sys.path.insert(0, str(HERE))
import stage2_b64 as S64  # noqa: E402  (B.64's second stage; puts b37/b38/b39/b54/b60/b61 on sys.path)
import run_b54 as B54  # noqa: E402
import run_b69 as P69  # noqa: E402

DATA = P69.DATA
CACHE = DATA / ".b69"
TAG = "camp"                       # the outer-fold label k of B.64's (r, k) for the temporal test block
Y_TOL = B54.Y_TOL                  # eV: value located in the file vs the stored value (run_b54 check (d))
BOUND_NAMES = ("Paths", "manifest", "P", "B39", "B54")


def log(m):
    S64.log(m)


# ------------------------------------------------------------------------------------------ layouts
class Layout:
    """Where the B.64 code finds a split's artifacts: the cache (base.npz, rows_s{s}.npz, stage2_r{r}_f{k}.npz), the
    outer and inner run directories, the manifest, the inner folds and the row table, all keyed by B.64's (r, k)."""

    def __init__(self, cache, outer_dir, inner_dir, manifest, inner_folds, rows, n_inner=P69.N_INNER):
        self.cache, self.outer_dir, self.inner_dir = Path(cache), outer_dir, inner_dir
        self.manifest, self.inner_folds, self.rows, self.n_inner = manifest, inner_folds, rows, n_inner


class _Paths:
    """Stands in for stage2_b64.Paths: the cache and the inner run directories."""

    def __init__(self, layout):
        self.plumb, self.cache, self._inner = False, layout.cache, layout.inner_dir

    def inner(self, arm, r, k, j):
        return self._inner(arm, r, k, j)


def b69_layout(cache=CACHE):
    """The registered layout: outer `.b69_{arm}_camp_cut_s{s}`, inner `..._i{j}`, the temporal manifest of seed s."""
    return Layout(cache,
                  outer_dir=lambda arm, r, k: P69.run_dir(arm, r),
                  inner_dir=lambda arm, r, k, j: P69.run_dir(arm, r, j),
                  manifest=lambda r, k: P69.outer_manifest(r),
                  inner_folds=lambda r, k: P69.inner_folds(r),
                  rows=lambda r, k: load_rows(cache, r))


@contextmanager
def bound(layout):
    """Bind stage2_b64's module-level names to `layout` for the duration of the block; restore them afterwards."""
    saved = {n: getattr(S64, n) for n in BOUND_NAMES}
    S64.Paths = lambda plumb: _Paths(layout)
    S64.manifest = layout.manifest
    S64.P = SimpleNamespace(inner_folds=layout.inner_folds, N_INNER=layout.n_inner)
    S64.B39 = SimpleNamespace(pooled_run_dir=layout.outer_dir)
    S64.B54 = SimpleNamespace(load_rows=layout.rows)
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(S64, n, v)


# ------------------------------------------------------------------------------------------ base
def stage_base(layout):
    """stage2_b64.stage_base into the layout's cache (scored T_i rows of every file: value, hold, causal GP, age)."""
    S64.stage_base(_Paths(layout))


# ------------------------------------------------------------------------------------------ rows
def load_rows(cache, s):
    z = np.load(Path(cache) / f"rows_s{s}.npz")
    return {kk: z[kk] for kk in z.files}


def stage_rows(layout, seeds):
    """The row table of B.54 for each seed's outer runs (run_b54.load_run / locate, imported; the saved fields are
    those of run_b54.stage_rows): file, grid row and value of every scored row of the test block, with the stored
    squared errors of the network, the twin, hold, causal GP, offline GP and PCHIP."""
    C, valid, F = None, None, None
    for s in seeds:
        out = layout.cache / f"rows_s{s}.npz"
        if out.exists():
            continue
        if C is None:
            C = B54.ctx41()
            valid, F = B54.valid_names(C), B54.load_files()
        c = B54.load_run(layout.outer_dir("v2", s, TAG))
        t = B54.load_run(layout.outer_dir("v2h", s, TAG))
        same = (len(c["shot"]) == len(t["shot"]) and np.array_equal(c["shot"], t["shot"])
                and np.array_equal(c["row"], t["row"]))
        if not same:
            raise SystemExit(f"FATAL: s{s}: the network and the twin were not scored on the same rows")
        refs_agree = all(np.allclose(c[q], t[q], rtol=1e-6, atol=1e-6, equal_nan=True)
                         for q in ("y", "hold", "gp", "off", "pchip"))
        loc = B54.locate(C, valid, F, c)
        dy = float(np.nanmax(np.abs(loc["y_file"] - c["y"])))
        if not np.isfinite(loc["y_file"]).all() or dy > Y_TOL:
            raise SystemExit(f"FATAL: s{s}: stored values do not match the located file values (max {dy} eV)")
        man = layout.manifest(s, TAG)
        test_names = set(man["test_files"])
        if not {C.names[int(i)] for i in np.unique(loc["fidx"])} <= test_names:
            raise SystemExit(f"FATAL: s{s}: a scored row lies outside the test block")
        layout.cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, shot=c["shot"], row=c["row"], se_c=c["se"], se_t=t["se"], y=c["y"],
                            se_hold=c["hold"], se_gp=c["gp"], se_off=c["off"], se_pchip=c["pchip"],
                            dt_ms=c["dt_ms"], same_rows=np.array(same), refs_agree=np.array(refs_agree),
                            test_files=np.array(man["test_files"]), **loc)
        log(f"rows s{s}: {len(c['shot'])} rows on {len(np.unique(loc['fidx']))} discharges, refs agree {refs_agree}, "
            f"max |y_file - y| {dy:.2g} eV")


# ------------------------------------------------------------------------------------------ fit
def stage2_path(layout, s):
    return layout.cache / f"stage2_r{s}_f{TAG}.npz"


def missing_runs(layout, seeds):
    miss = []
    for s in seeds:
        for arm in P69.MODELS:
            if not (layout.outer_dir(arm, s, TAG) / "comparison_errors_test.npz").exists():
                miss.append(layout.outer_dir(arm, s, TAG).name)
            for j in range(layout.n_inner):
                d = layout.inner_dir(arm, s, TAG, j)
                if not ((d / "weights" / "seq_lstm.pth").exists() and (d / "metrics.json").exists()):
                    miss.append(d.name)
    return miss


def fit_one(layout, r, k):
    """stage2_b64.fit_fold((r, k, False)) with its module-level names bound to `layout`."""
    with bound(layout):
        return S64.fit_fold((r, k, False))


def stage_fit(layout, seeds):
    miss = missing_runs(layout, seeds)
    if miss:
        raise SystemExit(f"FATAL: {len(miss)} runs are missing (first: {miss[:3]}); run run_b69.py first")
    for s in seeds:
        log(f"s{s}: " + fit_one(layout, s, TAG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["base", "rows", "fit", "all"])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(P69.SEEDS), choices=list(P69.SEEDS))
    args = ap.parse_args()
    layout = b69_layout()
    stages = ["base", "rows", "fit"] if args.stage == "all" else [args.stage]
    if "rows" in stages or "fit" in stages:
        miss = missing_runs(layout, args.seeds)
        if miss:
            raise SystemExit(f"FATAL: {len(miss)} runs are missing (first: {miss[:3]}); run run_b69.py first")
    for st in stages:
        if st == "base":
            stage_base(layout)
        elif st == "rows":
            stage_rows(layout, args.seeds)
        else:
            stage_fit(layout, args.seeds)
    log("done")


if __name__ == "__main__":
    main()
