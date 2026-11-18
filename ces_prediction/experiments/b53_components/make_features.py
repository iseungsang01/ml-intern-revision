"""B.53 arm C (PREREGISTRATION_B53.md section 3): per-split feature tables of the BES-array components.

For every split seed and every combination (C = c; D = g1, g2, h1, h2; CD = all five) one folder
data/.b53_feat/{C|D|CD}_s{seed}/ with, for each of the 641 shot files, mus_features_s{shot}.npz
(time (n,) = the grid times of seq_data.load_grid_files, feat (n, K) float32) and a feature_meta.json.

Components: b51_bes_modes/run_b51.py::components on the grid rows (logarithms of the nine BES channels;
c = their mean, g1/g2 = differences of column means, h1/h2 = differences of row means).  Each component
is standardized with the mean and standard deviation (ddof 0) over the TRAINING discharges of the split
(manifest train_files; validation and test files never enter), over the grid rows without the first
and the last row of each file (docs/paper/file_rules.md) and without rows with a non-positive BES
channel.  Rows with a non-positive BES channel carry zeros.  Every channel is listed as z-score exempt,
so seq_data passes the table through unscaled (its exempt-free path would z-score within the shot).

CPU only.  Run from the repo root:
    py ces_prediction/experiments/b53_components/make_features.py [--seeds 42 1 7 123]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True          # importing run_b51 must write nothing into tracked folders
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
DATA = REPO / "data"
FEAT_ROOT = DATA / ".b53_feat"
SEEDS = (42, 1, 7, 123)
COMBOS = {"C": ("c",), "D": ("g1", "g2", "h1", "h2"), "CD": ("c", "g1", "g2", "h1", "h2")}
ALL = ("c", "g1", "g2", "h1", "h2")
REG = "ces_prediction/experiments/PREREGISTRATION_B53.md"
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def split_dir(seed):
    return DATA / f".b1_w2cut_split_s{seed}"


def feat_dir(combo, seed):
    return FEAT_ROOT / f"{combo}_s{seed}"


def load_grid():
    """The grid of the B.39 cut runs (load_grid_files with the spike cut and held removal of GATE_ENV;
    neither touches the BES columns or the rows).  No extra table is set while it loads."""
    for k in [k for k in os.environ if k.startswith("CES_")]:
        del os.environ[k]
    os.environ["CES_TI_SPIKE_CUT_EV"] = "3000"
    sys.path.insert(0, str(EXP / "seq"))
    from seq_data import load_grid_files
    grid, dims = load_grid_files(DATA, True)
    if dims["bes"] != 9 or dims["extra_ti"] or dims["extra_vt"]:
        raise SystemExit(f"FATAL: unexpected grid dims {dims}")
    return grid


def components_fn():
    sys.path.insert(0, str(EXP / "b51_bes_modes"))
    import run_b51
    return run_b51.components


def raw_components(grid, comp):
    """name -> ((n, 5) components in ALL order, NaN on bad rows), (n,) bad mask)."""
    out = {}
    for nm in sorted(grid):
        o, bad = comp(grid[nm])
        X = np.column_stack([o[k] for k in ALL])
        if not np.isfinite(X[~bad]).all():
            raise SystemExit(f"FATAL: {nm}: a non-finite component on a row with all BES channels positive")
        out[nm] = (X, bad)
    return out


def constants(raw, names):
    """Mean and std (ddof 0) of each component over the rows of `names`, first and last row of each file
    excluded, bad rows excluded."""
    rows = []
    for nm in names:
        X, bad = raw[nm]
        keep = ~bad.copy()
        keep[0] = keep[-1] = False
        rows.append(X[keep])
    A = np.concatenate(rows)
    return A.mean(0), A.std(0), int(len(A))


def manifest(seed):
    return json.loads((split_dir(seed) / "split_manifest.json").read_text(encoding="utf-8"))


def write_split(seed, grid, raw):
    man = manifest(seed)
    train = list(man["train_files"])
    missing = [n for n in train if n not in raw]
    if missing:
        raise SystemExit(f"FATAL: split {seed}: train files not on disk: {missing[:5]}")
    mean, std, n_rows = constants(raw, train)
    if (std < 1e-9).any():
        raise SystemExit(f"FATAL: split {seed}: a component with zero spread {std}")
    zero_rows = sum(int(raw[nm][1].sum()) for nm in raw)
    for combo, comps in COMBOS.items():
        idx = [ALL.index(k) for k in comps]
        d = feat_dir(combo, seed)
        d.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        for nm in sorted(raw):
            X, bad = raw[nm]
            Z = (X[:, idx] - mean[idx]) / std[idx]
            Z[bad] = 0.0
            Z = Z.astype(np.float32)
            t = grid[nm][:, 0].astype(np.float64)
            np.savez(d / f"mus_features_s{nm[1:].split('.')[0]}.npz", time=t, feat=Z)
            h.update(nm.encode()); h.update(Z.tobytes())
        meta = {"k": len(comps), "components": list(comps), "z_exempt_channels": list(range(len(comps))),
                "split_seed": seed, "split_dir": str(split_dir(seed)), "split_manifest_train_files": len(train),
                "constants_from": "training discharges of the split only (manifest train_files); grid rows "
                                  "without the first and the last row of each file and without rows with a "
                                  "non-positive BES channel; std with ddof 0",
                "constants_rows": n_rows,
                "mean": {k: float(mean[ALL.index(k)]) for k in comps},
                "std": {k: float(std[ALL.index(k)]) for k in comps},
                "files": len(raw), "rows": int(sum(len(raw[nm][1]) for nm in raw)),
                "zero_rows_nonpositive_bes": zero_rows,
                "time": "grid times of seq_data.load_grid_files (float32 CSV time as float64)",
                "component_definition": "ces_prediction/experiments/b51_bes_modes/run_b51.py::components",
                "sha256_of_tables": h.hexdigest(), "registration": REG,
                "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        (d / "feature_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        log(f"split {seed} {combo}: {len(raw)} tables, K={len(comps)}, constants from {n_rows:,} rows of {len(train)} train files")
    return {"mean": mean.tolist(), "std": std.tolist(), "rows": n_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = ap.parse_args()
    grid = load_grid()
    log(f"grid: {len(grid)} files, {sum(len(a) for a in grid.values()):,} rows")
    raw = raw_components(grid, components_fn())
    for s in args.seeds:
        c = write_split(s, grid, raw)
        log(f"split {s}: mean {np.round(c['mean'], 4).tolist()} std {np.round(c['std'], 4).tolist()}")


if __name__ == "__main__":
    main()
