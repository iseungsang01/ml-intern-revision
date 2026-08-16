"""B.7 follow-up audit: the temporal structure of CES fit-failure spikes (THESIS_RESULTS.md
sec. 8ab memo). Descriptive only -- no protocol change; the V_rot cut / sensitivity rule stays
a decision for the researcher.

Over every ``<data>/*.csv`` shot file, on the raw observed (non-NaN) target sequence in time
order, split into contiguous blocks at time gaps >= STUCK_GAP_SECONDS (0.5 s):

CES_TI (the pre-registered 3 keV cut, ``CES_TI_SPIKE_CUT_EV``):
  * rows > 3 keV, grouped into runs of consecutive observed rows above the cut: run-length
    distribution, share of single-row runs, share of "isolated" runs (both observed neighbours
    < 2 keV), median jump ratio value / mean(neighbours), number of shots involved;
  * single-sample outliers in BOTH directions, independent of the cut: an observed row with
    both neighbours available is an UP outlier if v >= 2 x max(prev, next) and a DOWN dip if
    v <= 0.5 x min(prev, next) (neighbours > 0); the fraction of UP outliers the 3 keV cut
    actually removes says how partial / asymmetric a value cut is.
CES_VT (no cut in the protocol; sec. 8z found spiked anchors of thousands of km/s):
  * |v| tail (p99 / p99.9 / max, counts above 300 / 500 / 1000), runs above 1000, and
    single-sample jumps |v - mean(neighbours)| above 100 / 300 with both neighbours inside
    +-300 -- the analogue of the T_i table so a future rule can be chosen on numbers.

Run from the repo root:  py ces_prediction/experiments/b5_rescore/spike_structure_audit.py
Writes ``data/.b5_spike_structure.json`` and prints it.
"""

import csv
import glob
import json
import os

import numpy as np

DATA = os.environ.get("CES_DATA_DIR", "data")
OUT = os.path.join(DATA, ".b5_spike_structure.json")
GAP_S = 0.5
TI_CUT = 3000.0
TI_ISOLATED_NBR = 2000.0
VT_SPIKE = 1000.0
VT_QUIET_NBR = 300.0


def load(path):
    t, ti, vt = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["time"]))
            for col, dst in (("CES_TI", ti), ("CES_VT", vt)):
                raw = row.get(col, "")
                dst.append(np.nan if raw in ("", "nan") else float(raw))
    return np.asarray(t), np.asarray(ti), np.asarray(vt)


def blocks_of_observed(t, v):
    """Yield (values,) arrays of consecutive observed rows, split at time gaps >= GAP_S."""
    m = ~np.isnan(v)
    tt, vv = t[m], v[m]
    if tt.size == 0:
        return
    cut = np.flatnonzero(np.diff(tt) >= GAP_S) + 1
    for seg in np.split(vv, cut):
        yield seg


def runs_above(seg, thr):
    """Start/stop indices of maximal runs with seg > thr."""
    above = seg > thr
    if not above.any():
        return []
    d = np.diff(np.concatenate(([0], above.astype(int), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))  # [start, stop)


def main():
    files = sorted(glob.glob(os.path.join(DATA, "*.csv")))
    if not files:
        raise SystemExit(f"no shot CSVs under {DATA!r} -- set CES_DATA_DIR")

    ti_all, vt_all = [], []
    ti_runs = []          # (length, isolated: bool, jump_ratio or nan)
    ti_shots = set()
    ti_up, ti_up_gt_cut, ti_down = 0, 0, 0
    ti_up_values = []
    ti_nbr_rows = 0
    vt_runs = []
    vt_shots = set()
    vt_jump100, vt_jump300, vt_nbr_rows = 0, 0, 0
    vt_jump300_abs_gt_1000 = 0
    vt_per_shot = {}

    for path in files:
        shot = os.path.basename(path)
        t, ti, vt = load(path)
        ti_all.append(ti[~np.isnan(ti)])
        vt_all.append(vt[~np.isnan(vt)])

        for seg in blocks_of_observed(t, ti):
            n = seg.size
            for a, b in runs_above(seg, TI_CUT):
                ti_shots.add(shot)
                prev = seg[a - 1] if a > 0 else np.nan
                nxt = seg[b] if b < n else np.nan
                nbrs = [x for x in (prev, nxt) if not np.isnan(x)]
                isolated = (b - a == 1) and len(nbrs) == 2 and all(x < TI_ISOLATED_NBR for x in nbrs)
                ratio = float(seg[a:b].max() / np.mean(nbrs)) if nbrs and np.mean(nbrs) > 0 else np.nan
                ti_runs.append((int(b - a), bool(isolated), ratio))
            if n >= 3:
                v, p, q = seg[1:-1], seg[:-2], seg[2:]
                ok = (p > 0) & (q > 0)
                ti_nbr_rows += int(ok.sum())
                up = ok & (v >= 2.0 * np.maximum(p, q))
                down = ok & (v <= 0.5 * np.minimum(p, q))
                ti_up += int(up.sum()); ti_down += int(down.sum())
                ti_up_gt_cut += int((up & (v > TI_CUT)).sum())
                ti_up_values.append(v[up])

        for seg in blocks_of_observed(t, vt):
            n = seg.size
            for a, b in runs_above(np.abs(seg), VT_SPIKE):
                vt_shots.add(shot)
                vt_runs.append(int(b - a))
                vt_per_shot[shot] = vt_per_shot.get(shot, 0) + int(b - a)
            if n >= 3:
                v, p, q = seg[1:-1], seg[:-2], seg[2:]
                quiet = (np.abs(p) <= VT_QUIET_NBR) & (np.abs(q) <= VT_QUIET_NBR)
                vt_nbr_rows += int(quiet.sum())
                jump = np.abs(v - 0.5 * (p + q))
                vt_jump100 += int((quiet & (jump > 100.0)).sum())
                j300 = quiet & (jump > 300.0)
                vt_jump300 += int(j300.sum())
                vt_jump300_abs_gt_1000 += int((j300 & (np.abs(v) > VT_SPIKE)).sum())

    ti = np.concatenate(ti_all)
    vt = np.concatenate(vt_all)
    lengths = np.asarray([r[0] for r in ti_runs])
    isolated = np.asarray([r[1] for r in ti_runs])
    ratios = np.asarray([r[2] for r in ti_runs])
    ti_up_values = np.concatenate(ti_up_values) if ti_up_values else np.asarray([])
    vt_lengths = np.asarray(vt_runs)

    result = {
        "n_files": len(files),
        "definitions": {
            "block": f"observed rows in time order, split at gaps >= {GAP_S} s",
            "ti_run": f"maximal run of consecutive observed CES_TI > {TI_CUT:.0f} eV",
            "ti_isolated": f"single-row run with both observed neighbours < {TI_ISOLATED_NBR:.0f} eV",
            "ti_jump_ratio": "max(run) / mean(available neighbours)",
            "ti_up_outlier": "v >= 2 x max(prev, next), neighbours > 0",
            "ti_down_dip": "v <= 0.5 x min(prev, next), neighbours > 0",
            "vt_run": f"maximal run of consecutive observed |CES_VT| > {VT_SPIKE:.0f}",
            "vt_jump": f"|v - mean(prev, next)| with both neighbours inside +-{VT_QUIET_NBR:.0f}",
        },
        "CES_TI": {
            "n_observed": int(ti.size),
            "n_gt_cut": int((ti > TI_CUT).sum()),
            "n_runs": int(lengths.size),
            "n_shots_with_runs": len(ti_shots),
            "run_length": {"1": int((lengths == 1).sum()), "2-4": int(((lengths >= 2) & (lengths <= 4)).sum()),
                           ">=5": int((lengths >= 5).sum()), "max": int(lengths.max()) if lengths.size else 0},
            "single_row_run_share": float((lengths == 1).mean()) if lengths.size else None,
            "isolated_run_share": float(isolated.mean()) if lengths.size else None,
            "n_isolated_runs": int(isolated.sum()),
            "jump_ratio_median": float(np.nanmedian(ratios)) if lengths.size else None,
            "jump_ratio_p25_p75": [float(np.nanpercentile(ratios, 25)), float(np.nanpercentile(ratios, 75))] if lengths.size else None,
            "n_rows_with_both_neighbours": ti_nbr_rows,
            "n_up_outliers_ge2x": ti_up,
            "n_down_dips_ge2x": ti_down,
            "n_up_outliers_removed_by_cut": ti_up_gt_cut,
            "up_outlier_share_removed_by_cut": (ti_up_gt_cut / ti_up) if ti_up else None,
            "up_outlier_value_median_eV": float(np.median(ti_up_values)) if ti_up_values.size else None,
        },
        "CES_VT": {
            "n_observed": int(vt.size),
            "abs_p99": float(np.percentile(np.abs(vt), 99)),
            "abs_p999": float(np.percentile(np.abs(vt), 99.9)),
            "abs_max": float(np.abs(vt).max()),
            "n_abs_gt_300": int((np.abs(vt) > 300).sum()),
            "n_abs_gt_500": int((np.abs(vt) > 500).sum()),
            "n_abs_gt_1000": int((np.abs(vt) > VT_SPIKE).sum()),
            "n_runs_gt_1000": int(vt_lengths.size),
            "n_shots_with_runs_gt_1000": len(vt_shots),
            "run_length_gt_1000": {"1": int((vt_lengths == 1).sum()), "2-4": int(((vt_lengths >= 2) & (vt_lengths <= 4)).sum()),
                                   ">=5": int((vt_lengths >= 5).sum()), "max": int(vt_lengths.max()) if vt_lengths.size else 0},
            "n_rows_with_quiet_neighbours": vt_nbr_rows,
            "n_jump_gt_100": vt_jump100,
            "n_jump_gt_300": vt_jump300,
            "n_jump_gt_300_and_abs_gt_1000": vt_jump300_abs_gt_1000,
            "rows_gt_1000_per_shot_top": sorted(vt_per_shot.items(), key=lambda kv: -kv[1])[:5],
        },
    }
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
