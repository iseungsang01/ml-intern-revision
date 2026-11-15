"""Example time traces through CES gaps (KSTAR, T_i), with a row-level check.

Selection rule, fixed before any trace was drawn: the two discharges of this data set whose
edge has been analysed in the literature (31921, 31923) and the file-level split whose TEST
set holds both (split seed 1); in each discharge the 0.6 s window with the most grid rows
that lack a T_i value between the first and the last observed T_i of a contiguous segment.

Third panel, added after the first two had been seen (post hoc): the measured T_i of those
two discharges moves by more than 1 keV between adjacent rows, so a discharge with typical
scatter is shown beside them -- among the TEST discharges of the same split with at least
200 observed T_i rows and at least one such window, the one whose row-to-row scatter
(1.4826 x MAD of the differences between observed T_i on adjacent grid rows) is the median
(lower middle for an even count); same window rule.

Estimates on every grid row of the window:
  recurrent estimator -- inference with the frozen checkpoint data/.b1_seqv2_s1_i1 (GPU, the
                         device of the frozen evaluation; CES_TRACE_DEVICE=cpu also runs),
  persistence         -- last observed T_i before the row,
  causal GP           -- baselines_interpolation.predict_gp_causal on the observed T_i of the
                         segment, excluding the row itself.
Check: on the rows the frozen evaluation scored for these discharges, the squared errors of
the three recomputed estimates must equal the stored ones
(data/.b1_seqv2_s1_i1/comparison_errors_test.npz). If the check fails nothing is drawn.

Run from the repo root:  py docs/paper/ppcf/fig_trace.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "data" / ".b1_seqv2_s1_i1"
SPLIT = ROOT / "data" / ".b1_w2cut_split_s1"

# the frozen protocol of the backbone evaluation (cut population)
os.environ["CES_DROP_STUCK_TARGETS"] = "1"
os.environ["CES_TI_SPIKE_CUT_EV"] = "3000"
os.environ["CES_PER_SHOT_NORM"] = "1"
os.environ["CES_SEQ_MODEL"] = "v2"
os.environ["CES_OUTPUT_DIR"] = str(RUN)
for k in ("CES_SEQ_NORM_EXPANDING", "CES_SEQ_EXTRA_VT", "CES_SEQ_EXTRA_TI", "CES_SEQ_ZERO_MC",
          "CES_SEQ_ENSEMBLE_DIRS", "CES_SEQ_CONTEXTS", "CES_SEQ_EVAL_CTX"):
    os.environ.pop(k, None)
DEVICE = os.getenv("CES_TRACE_DEVICE", "cuda")

sys.path.insert(0, str(ROOT / "ces_prediction" / "experiments" / "seq"))
sys.path.insert(1, str(ROOT / "ces_prediction"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import baselines_interpolation as B  # noqa: E402
from evaluate import _load_stats  # noqa: E402
from eval_seq import build_pred_lookup  # noqa: E402
from seq_data import load_grid_files  # noqa: E402

SHOTS = ("s31921.csv", "s31923.csv")
WINDOW_S = 0.6
GAP = B.GAP_SECONDS


def segments(time):
    cut = np.flatnonzero(np.diff(time) >= GAP) + 1
    return np.split(np.arange(len(time)), cut)


def estimates(arr, pred_table):
    """Per grid row: model, persistence, causal GP (NaN where no past observation)."""
    time = arr[:, 0].astype(np.float64)
    ti = arr[:, 1].astype(np.float64)
    n = len(time)
    model = np.array([pred_table[float(t)][0] if float(t) in pred_table else np.nan
                      for t in arr[:, 0]], dtype=np.float64)
    pers = np.full(n, np.nan)
    gp = np.full(n, np.nan)
    seg_id = np.zeros(n, dtype=int)
    for s, idx in enumerate(segments(time)):
        seg_id[idx] = s
        obs = idx[~np.isnan(ti[idx])]
        for r in idx:
            keep = obs[obs != r]
            if keep.size == 0:
                continue
            tt, vv = time[keep], ti[keep]
            pers[r] = B.predict_persistence(tt, vv, time[r])
            gp[r] = B.predict_gp_causal(tt, vv, time[r])
    return time, ti, model, pers, gp, seg_id


def _inside(time, ti, seg_id):
    inside = np.zeros(len(time), dtype=bool)
    for s in np.unique(seg_id):
        idx = np.flatnonzero(seg_id == s)
        obs = idx[~np.isnan(ti[idx])]
        if obs.size >= 2:
            span = idx[(idx > obs[0]) & (idx < obs[-1])]
            inside[span] = np.isnan(ti[span])
    return inside


def scatter(time, ti):
    """1.4826 x MAD of the differences between observed T_i on adjacent grid rows."""
    ok = ~np.isnan(ti[1:]) & ~np.isnan(ti[:-1]) & (np.diff(time) < GAP)
    d = (ti[1:] - ti[:-1])[ok]
    if d.size < 50:
        return np.nan
    return 1.4826 * float(np.median(np.abs(d - np.median(d))))


def pick_window(time, ti, seg_id):
    """0.6 s window with the most unlabelled rows lying between observed rows of a segment."""
    inside = _inside(time, ti, seg_id)
    best, best_n = None, -1
    for i in np.flatnonzero(inside):
        j = np.searchsorted(time, time[i] + WINDOW_S)
        if seg_id[i] != seg_id[min(j, len(time) - 1)]:
            continue
        c = int(inside[i:j].sum())
        if c > best_n:
            best, best_n = (i, j), c
    if best is None:
        return None
    lo = max(best[0] - 5, 0)
    return lo, best[1], best_n


def typical_shot(grid, test_files):
    """TEST discharge with the median row-to-row scatter of the measured T_i."""
    rows = []
    for name in sorted(test_files):
        arr = grid[name]
        time, ti = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
        if int((~np.isnan(ti)).sum()) < 200:
            continue
        seg_id = np.zeros(len(time), dtype=int)
        for k, idx in enumerate(segments(time)):
            seg_id[idx] = k
        sc = scatter(time, ti)
        if not np.isfinite(sc) or pick_window(time, ti, seg_id) is None:
            continue
        rows.append((sc, name))
    rows.sort()
    return rows[(len(rows) - 1) // 2], len(rows), {n: round(v, 1) for v, n in rows}


def check(name, time, ti, model, pers, gp, npz, valid_names):
    """Match the stored scored rows of this discharge and compare squared errors."""
    k = valid_names.index(name)
    sel = npz["CES_TI_shot"] == k
    y = npz["CES_TI_y_true"][sel].astype(np.float64)
    obs = np.flatnonzero(~np.isnan(ti))
    order = np.argsort(ti[obs])
    pos = np.searchsorted(ti[obs][order], y)
    pos = np.clip(pos, 1, len(obs) - 1)
    left, right = obs[order][pos - 1], obs[order][pos]
    row = np.where(np.abs(ti[left] - y) <= np.abs(ti[right] - y), left, right)
    dv = np.abs(ti[row] - y)
    out = {"scored_rows": int(sel.sum()), "max_value_mismatch_eV": float(dv.max())}
    for label, est, key in (("model", model, "CES_TI_se_model"),
                            ("persistence", pers, "CES_TI_se_persistence"),
                            ("causal_gp", gp, "CES_TI_se_gp_causal")):
        mine = (est[row] - y) ** 2
        ref = npz[key][sel].astype(np.float64)
        rel = np.abs(mine - ref) / np.maximum(ref, 1.0)
        dev_ev = np.abs(np.abs(est[row] - y) - np.sqrt(ref))
        out[label] = {"max_abs_dev_of_error_eV": float(np.nanmax(dev_ev)),
                      "max_rel_dev": float(np.nanmax(rel)),
                      "share_within_1e-3": float(np.mean(rel <= 1e-3)),
                      "rmse_mine": float(np.sqrt(np.nanmean(mine))),
                      "rmse_stored": float(np.sqrt(ref.mean()))}
    return out


def main():
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    stats = _load_stats(metrics)
    manifest = json.loads((SPLIT / "split_manifest.json").read_text(encoding="utf-8"))
    for s in SHOTS:
        if s not in manifest["test_files"]:
            raise SystemExit(f"FATAL: {s} is not a TEST file of split 1")
    grid, _ = load_grid_files(ROOT / "data", True)
    valid_names = sorted(grid)
    (typ_sc, typ_name), n_typ, sc_all = typical_shot(grid, manifest["test_files"])
    shots = list(SHOTS) + [typ_name]
    with torch.no_grad():
        lookup = build_pred_lookup(ROOT / "data", shots, stats, torch.device(DEVICE))["full"]
    npz = np.load(RUN / "comparison_errors_test.npz")

    report, traces = {"device": DEVICE, "rule": __doc__.split("Estimates")[0].strip()}, {}
    ok = True
    report["typical_discharge"] = {"file": typ_name, "scatter_eV": round(typ_sc, 1),
                                   "candidates": n_typ,
                                   "scatter_eV_of_literature_discharges":
                                       {n: sc_all.get(n) for n in SHOTS}}
    for name in shots:
        time, ti, model, pers, gp, seg_id = estimates(grid[name], lookup[name])
        c = check(name, time, ti, model, pers, gp, npz, valid_names)
        report[name] = c
        # persistence and the causal GP are deterministic CPU arithmetic: squared errors must
        # agree to 1e-3. The network runs on a different device than the frozen evaluation
        # (float32 LSTM arithmetic differs between CPU and GPU), so its absolute error must
        # agree with the stored one to within 0.5 eV on every scored row.
        for label in ("persistence", "causal_gp"):
            ok &= c[label]["share_within_1e-3"] >= 0.999
        ok &= c["model"]["max_abs_dev_of_error_eV"] <= 0.5
        ok &= c["max_value_mismatch_eV"] < 0.05
        traces[name] = (time, ti, model, pers, gp, seg_id)
    report["check_passed"] = bool(ok)
    if not ok:
        (HERE / "fig_trace.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report, indent=1))
        raise SystemExit("FATAL: recomputed estimates do not match the stored errors; no figure.")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(shots), 1, figsize=(7.0, 5.3))
    for k, (ax, name) in enumerate(zip(axes, shots)):
        time, ti, model, pers, gp, seg_id = traces[name]
        lo, hi, n_gap = pick_window(time, ti, seg_id)
        sl = slice(lo, hi)
        t = time[sl]
        obs = ~np.isnan(ti[sl])
        ax.plot(t, pers[sl], color="0.55", lw=1.0, drawstyle="steps-post", label="persistence")
        ax.plot(t, gp[sl], color="#1f77b4", lw=1.1, ls="--", label="causal GP (CES only)")
        ax.plot(t, model[sl], color="#d62728", lw=1.4, label="recurrent estimator")
        ax.plot(t[obs], ti[sl][obs], "o", ms=3.6, mfc="white", mec="k", mew=0.8,
                label=r"CES $T_i$ measured")
        for r in np.flatnonzero(~obs):
            ax.axvspan(t[r] - 0.005, t[r] + 0.005, color="0.92", lw=0)
        ax.set_ylabel(r"$T_i$ (eV)")
        sc = scatter(time, ti)
        vv = ti[sl][obs]
        ax.set_title(f"({'abc'[k]}) KSTAR #{name[1:6]}: {int((~obs).sum())} of {len(t)} rows "
                     f"without $T_i$; measured {vv.min():.0f} to {vv.max():.0f} eV",
                     loc="left", fontsize=9)
        ax.margins(x=0.01)
        report[name]["window_s"] = [float(t[0]), float(t[-1])]
        report[name]["rows_without_ti_in_window"] = int((~obs).sum())
        report[name]["rows_in_window"] = int(len(t))
        report[name]["scatter_eV"] = round(float(sc), 1)
        report[name]["measured_min_max_eV_in_window"] = [round(float(vv.min()), 1),
                                                         round(float(vv.max()), 1)]
    axes[-1].set_xlabel("time (s)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, fontsize=7.5, frameon=False, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "fig_trace.pdf")
    fig.savefig(HERE / "fig_trace.png", dpi=170)
    (HERE / "fig_trace.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
