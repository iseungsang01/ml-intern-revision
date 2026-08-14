"""B.7 protocol-audit re-verification stats (PROTOCOL_AUDIT.md section 1).

Recomputes, over every ``<data>/*.csv`` shot file, the numbers that justify the
pre-registered constraints of the W = 2 re-experiment:

  - observed ``CES_TI`` distribution: n, p99, p99.9, max, and row counts above
    the 2,089 / 2,500 / 3,000 / 4,000 eV spike-cut candidates;
  - inter-row time-delta histogram (segment-separation threshold sensitivity);
  - ``CES_VT`` value precision: decimal places and minimum nonzero spacing
    between distinct observed values (held-rule false-positive bound);
  - NaN / held ledger cross-check against ``analyze_data_evidence.py``.

Run from the repo root:  py ces_prediction/experiments/protocol_audit_stats.py
Writes ``data/.protocol_audit_stats.json`` and prints the same JSON.
"""

import csv
import glob
import json
import os

import numpy as np

DATA = os.environ.get("CES_DATA_DIR", "data")
OUT = os.path.join(DATA, ".protocol_audit_stats.json")


def main() -> None:
    files = sorted(glob.glob(os.path.join(DATA, "*.csv")))
    if not files:
        raise SystemExit(f"no shot CSVs under {DATA!r} — set CES_DATA_DIR")
    ti_parts = []
    delta_parts = []
    vt_strings = []
    ledger = {"CES_TI": {"nan": 0, "held": 0}, "CES_VT": {"nan": 0, "held": 0}}
    n_rows = 0
    for path in files:
        with open(path, newline="") as fh:
            time, cols = [], {"CES_TI": [], "CES_VT": []}
            for row in csv.DictReader(fh):
                time.append(float(row["time"]))
                for col in cols:
                    raw = row.get(col, "")
                    if raw in ("", "nan"):
                        cols[col].append(np.nan)
                    else:
                        cols[col].append(float(raw))
                        if col == "CES_VT":
                            vt_strings.append(raw)
        n_rows += len(time)
        delta_parts.append(np.diff(np.asarray(time)))
        for col, vals in cols.items():
            arr = np.asarray(vals)
            ledger[col]["nan"] += int(np.isnan(arr).sum())
            prev = None
            for x in arr:
                if np.isnan(x):
                    continue
                if prev is not None and x == prev:
                    ledger[col]["held"] += 1
                prev = x
            if col == "CES_TI":
                ti_parts.append(arr[~np.isnan(arr)])

    ti = np.concatenate(ti_parts)
    deltas = np.concatenate(delta_parts)
    vt = np.unique([float(s) for s in vt_strings])
    spacing = np.diff(vt)
    decimals = [len(s.split(".")[1]) if "." in s else 0 for s in vt_strings]
    result = {
        "n_files": len(files),
        "n_rows": n_rows,
        "ces_ti_observed": {
            "n": int(ti.size),
            "p99_eV": float(np.percentile(ti, 99)),
            "p999_eV": float(np.percentile(ti, 99.9)),
            "max_eV": float(ti.max()),
            "n_gt_2089": int((ti > 2089).sum()),
            "n_gt_2500": int((ti > 2500).sum()),
            "n_gt_3000": int((ti > 3000).sum()),
            "n_gt_4000": int((ti > 4000).sum()),
            "pct_gt_3000": float((ti > 3000).mean() * 100),
        },
        "time_deltas": {
            "<=0.011s_grid": int((deltas <= 0.011).sum()),
            "(0.011,0.1]": int(((deltas > 0.011) & (deltas <= 0.1)).sum()),
            "(0.1,0.5)": int(((deltas > 0.1) & (deltas < 0.5)).sum()),
            ">=0.5": int((deltas >= 0.5).sum()),
        },
        "ces_vt_precision": {
            "decimal_places_min": int(min(decimals)),
            "decimal_places_max": int(max(decimals)),
            "min_nonzero_spacing": float(spacing[spacing > 0].min()),
        },
        "ledger_crosscheck": ledger,
    }
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
