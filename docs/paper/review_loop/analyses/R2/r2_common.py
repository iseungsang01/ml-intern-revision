"""Shared helpers for the review-loop round 2 analyses (REGISTRATION.md, 2026-11-15).

CPU only, real data and frozen artifacts only. ASCII prints (cp949 console).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
DATA = REPO / "data"
HERE = Path(__file__).resolve().parent
CES = REPO / "ces_prediction"
for p in (CES, CES / "experiments" / "seq", CES / "experiments" / "b27_kfold", CES / "experiments" / "b1_gate"):
    sys.path.insert(0, str(p))

SEEDS = (42, 1, 7, 123)
POPS = {
    "cut": {"cut": "3000", "run": ".b1_seqv2_s{s}_i{s}", "control": ".b1_w2cut_s{s}",
            "split": ".b1_w2cut_split_s{s}"},
    "incl": {"cut": "0", "run": ".b5i_seqv2_s{s}", "control": ".wsweep_hf_w2_s{s}",
             "split": ".b1_manifest_s{s}"},
}
REFS = ("persistence", "pchip", "gp_causal", "gp")
B_BOOT, SEED_BOOT = 10_000, 12345
GAP = 0.5


def pin_confirmed_env(pop):
    """The confirmed protocol's data treatment, pinned explicitly (never a default)."""
    os.environ["CES_DROP_STUCK_TARGETS"] = "1"
    os.environ["CES_TI_SPIKE_CUT_EV"] = POPS[pop]["cut"]
    os.environ["CES_PER_SHOT_NORM"] = "1"
    os.environ["CES_TEMPORAL_SUBSETS"] = "0"
    os.environ["CES_WINDOW_SIZE"] = "2"


def build_dataset(pop):
    pin_confirmed_env(pop)
    from dataset import KSTAR_CES_Dataset  # noqa: PLC0415
    ds = KSTAR_CES_Dataset(data_dir=DATA, window_size=2, temporal_subset_augmentation=False,
                           drop_stuck_targets=True, per_shot_input_norm=True,
                           ti_spike_cut_ev=float(POPS[pop]["cut"]))
    if len(ds.valid_files) != 641:
        raise SystemExit(f"FATAL: expected 641 valid files, got {len(ds.valid_files)}")
    return ds


def canonical(pop, seed, target):
    """Backbone errors of one canonical cell + row index from the stage-4 surrogate file,
    accepted only if shot ids, dt_ms and the PCHIP squared errors are bit-identical."""
    bb = np.load(DATA / POPS[pop]["run"].format(s=seed) / "comparison_errors_test.npz")
    su = np.load(DATA / f".b28_b28h_{pop}_s{seed}" / "comparison_errors_test.npz")
    t = target
    ok = (len(bb[f"{t}_shot"]) == len(su[f"{t}_shot"])
          and np.array_equal(bb[f"{t}_shot"], su[f"{t}_shot"])
          and np.array_equal(bb[f"{t}_dt_ms"], su[f"{t}_dt_ms"])
          and np.array_equal(bb[f"{t}_se_pchip"], su[f"{t}_se_pchip"]))
    if not ok:
        raise SystemExit(f"FATAL: {pop} s{seed} {t}: surrogate rows are not the backbone rows")
    d = {"shot": bb[f"{t}_shot"], "row": su[f"{t}_row"], "y": su[f"{t}_y_true"].astype(np.float64),
         "model": bb[f"{t}_se_model"].astype(np.float64),
         "surrogate": su[f"{t}_se_model"].astype(np.float64),
         "linear": bb[f"{t}_se_linear"].astype(np.float64)}
    for r in REFS:
        d[r] = bb[f"{t}_se_{r}"].astype(np.float64)
    return d


def manifest(pop, seed):
    return json.loads((DATA / POPS[pop]["split"].format(s=seed) / "split_manifest.json").read_text(encoding="utf-8"))


def shot_sums(shot, *vs):
    u, inv = np.unique(shot, return_inverse=True)
    return [np.bincount(inv, weights=v, minlength=len(u)) for v in vs]


def skill(shot, se_a, se_ref, B=B_BOOT, seed=SEED_BOOT):
    """1 - sum SE_a / sum SE_ref, shot-cluster percentile bootstrap (fresh generator)."""
    a, r = shot_sums(shot, np.asarray(se_a, np.float64), np.asarray(se_ref, np.float64))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    sk = 1.0 - a[idx].sum(1) / r[idx].sum(1)
    sk = sk[np.isfinite(sk)]
    lo, hi = np.percentile(sk, [2.5, 97.5])
    return {"point": float(1.0 - a.sum() / r.sum()), "ci95": [float(lo), float(hi)],
            "excludes_0": bool(lo > 0 or hi < 0), "n_shots": int(len(a)), "n_rows": int(len(shot))}


def fmt(s):
    return f"{s['point']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]{'*' if s['excludes_0'] else ' '}"
