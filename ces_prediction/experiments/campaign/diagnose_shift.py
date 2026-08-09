"""Why does campaign shift cost skill? Measure the drift instead of asserting it.

The campaign-shift batch shows the model's advantage over *future-using* interpolation
disappearing on a strictly temporal split while its advantage over persistence largely
survives. That pattern has a specific candidate explanation: the part of the model that
does not transfer is the learned mapping from the FAST DIAGNOSTICS, because BES/ECEI/MC
calibration, gain and hardware change between campaigns, whereas the CES-history pathway
sees the same physical quantity in the same units at all times.

Reporting the degradation without testing that would break this project's own rule --
never report a negative result without naming the measurement that would overturn it.
So: quantify the train->test distribution shift per input group and for the targets,
using only the split this experiment already defines.

Two shift statistics per channel, both computed on the train period and applied to test:
  z          |mean_test - mean_train| / std_train   -- location drift in train units
  std_ratio  std_test / std_train                   -- scale drift

If the fast diagnostics drift substantially more than the CES targets do, the observed
skill pattern is explained by input drift, and the fix is a measurement/normalization
question (per-campaign standardisation, or campaign-adversarial training) rather than a
model-capacity one.

Usage (repo root):  py ces_prediction/experiments/campaign/diagnose_shift.py
Writes: data/.campaign_shift_diagnosis.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ces_prediction"))
from dataset import (  # noqa: E402
    STUCK_GAP_SECONDS,
    TARGET_COLUMNS,
    TIME_COLUMN,
    KSTAR_CES_Dataset,
)

DATA_DIR = Path(os.getenv("CES_DATA_DIR", REPO_ROOT / "data"))
MANIFEST = REPO_ROOT / "data" / ".campaign_split_manifest.json"


def group_of(col):
    if col.startswith("BES"):
        return "BES"
    if col.startswith("ECEI"):
        return "ECEI"
    if col.startswith("MC"):
        return "MC"
    if col in TARGET_COLUMNS:
        return "CES(target)"
    return None


def accumulate(files):
    """Streaming mean/var per column over a file list, held-free for the targets."""
    n = {}
    s1 = {}
    s2 = {}
    for name in files:
        p = DATA_DIR / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        time = df[TIME_COLUMN].to_numpy(dtype=np.float64)
        if len(time) >= 2:
            block = np.concatenate(([0], np.cumsum(np.diff(time) >= STUCK_GAP_SECONDS)))
            for col in TARGET_COLUMNS:
                v = df[col].to_numpy(dtype=np.float64, copy=True)
                v[KSTAR_CES_Dataset._stuck_repeat_mask(v, block)] = np.nan
                df[col] = v
        for col in df.columns:
            if group_of(col) is None:
                continue
            v = df[col].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            n[col] = n.get(col, 0) + v.size
            s1[col] = s1.get(col, 0.0) + float(v.sum())
            s2[col] = s2.get(col, 0.0) + float((v * v).sum())
    stats = {}
    for col in n:
        mean = s1[col] / n[col]
        var = max(s2[col] / n[col] - mean * mean, 0.0)
        stats[col] = {"n": n[col], "mean": mean, "std": float(np.sqrt(var))}
    return stats


def main():
    if not MANIFEST.exists():
        raise SystemExit(f"FATAL: {MANIFEST} missing -- run run_campaign_shift.py first.")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"[shift] train {man['counts']['train']} shots {man['shot_ranges']['train']} -> "
          f"test {man['counts']['test']} shots {man['shot_ranges']['test']}")

    train = accumulate(man["train_files"])
    test = accumulate(man["test_files"])

    per_col = {}
    for col in sorted(set(train) & set(test)):
        sd = train[col]["std"]
        if sd <= 0:
            continue
        per_col[col] = {
            "group": group_of(col),
            "z_location_shift": abs(test[col]["mean"] - train[col]["mean"]) / sd,
            "std_ratio": (test[col]["std"] / sd),
            "train_mean": train[col]["mean"], "test_mean": test[col]["mean"],
        }

    groups = {}
    for col, e in per_col.items():
        groups.setdefault(e["group"], []).append(e)
    summary = {}
    for g, items in groups.items():
        z = np.array([i["z_location_shift"] for i in items])
        r = np.array([i["std_ratio"] for i in items])
        summary[g] = {
            "n_channels": len(items),
            "z_median": float(np.median(z)), "z_max": float(z.max()),
            "std_ratio_median": float(np.median(r)),
            "std_ratio_min": float(r.min()), "std_ratio_max": float(r.max()),
        }

    out = {"split": man["shot_ranges"], "counts": man["counts"],
           "per_group": summary, "per_channel": per_col,
           "note": ("z = |mean_test - mean_train| / std_train, i.e. location drift measured "
                    "in the units the model was normalized in")}
    path = REPO_ROOT / "data" / ".campaign_shift_diagnosis.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[shift] wrote {path}\n")

    print(f"{'group':>12} {'chans':>6} {'z median':>9} {'z max':>8} "
          f"{'std ratio (med)':>16} {'std ratio range':>20}")
    for g in ("BES", "ECEI", "MC", "CES(target)"):
        if g not in summary:
            continue
        e = summary[g]
        print(f"{g:>12} {e['n_channels']:>6} {e['z_median']:>9.3f} {e['z_max']:>8.3f} "
              f"{e['std_ratio_median']:>16.3f} "
              f"{e['std_ratio_min']:.2f}..{e['std_ratio_max']:.2f}".rjust(20))


if __name__ == "__main__":
    main()
