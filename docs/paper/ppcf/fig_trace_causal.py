"""Example time traces through CES gaps for the main text: the causal recurrent estimator of B.39.

Same discharges, windows and row-level check as fig_trace.py (whose selection rule was fixed
before any trace was drawn and does not depend on the estimator); the estimator drawn is the
causal variant trained in B.39 on split seed 1 with initialisation 1
(data/.b39_v2_cut_s1_i1: fast channels standardized with the statistics of the training
discharges, 80-epoch cap).  Inference on the CPU; the absolute error of the network must agree
with the stored one (comparison_errors_test.npz of that run) within 0.5 eV on every scored row
of the three discharges, hold and the causal GP to 1e-3 in squared error; otherwise nothing is
drawn.  Rows that follow an absent frame (a beam blip) are marked.

Run from the repo root:  py docs/paper/ppcf/fig_trace_causal.py
Output: docs/paper/ppcf/fig_trace_causal.{pdf,png,json}
"""
import json
import os
import sys
from pathlib import Path

os.environ["CES_TRACE_DEVICE"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import fig_trace as FT  # noqa: E402  (sets the protocol env of the earlier figure; overridden below)

RUN = ROOT / "data" / ".b39_v2_cut_s1_i1"
os.environ["CES_PER_SHOT_NORM"] = "0"
os.environ["CES_OUTPUT_DIR"] = str(RUN)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = HERE / "fig_trace_causal"


def main():
    torch.set_num_threads(4)
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    if metrics["seq_model"] != "v2" or bool(metrics["per_shot_input_norm"]):
        raise SystemExit("FATAL: the run is not the causal variant")
    stats = FT._load_stats(metrics)
    manifest = json.loads((FT.SPLIT / "split_manifest.json").read_text(encoding="utf-8"))
    for s in FT.SHOTS:
        if s not in manifest["test_files"]:
            raise SystemExit(f"FATAL: {s} is not a TEST file of split 1")
    grid, _ = FT.load_grid_files(ROOT / "data", True)
    valid_names = sorted(grid)
    (typ_sc, typ_name), n_typ, sc_all = FT.typical_shot(grid, manifest["test_files"])
    shots = list(FT.SHOTS) + [typ_name]
    with torch.no_grad():
        lookup = FT.build_pred_lookup(ROOT / "data", shots, stats, torch.device("cpu"))["full"]
    npz = np.load(RUN / "comparison_errors_test.npz")

    report = {"device": "cpu", "run": RUN.name,
              "typical_discharge": {"file": typ_name, "scatter_eV": round(typ_sc, 1), "candidates": n_typ}}
    traces, ok = {}, True
    for name in shots:
        time, ti, model, pers, gp, seg_id = FT.estimates(grid[name], lookup[name])
        c = FT.check(name, time, ti, model, pers, gp, npz, valid_names)
        report[name] = c
        for label in ("persistence", "causal_gp"):
            ok &= c[label]["share_within_1e-3"] >= 0.999
        ok &= c["model"]["max_abs_dev_of_error_eV"] <= 0.5
        ok &= c["max_value_mismatch_eV"] < 0.05
        traces[name] = (time, ti, model, pers, gp, seg_id)
    report["check_passed"] = bool(ok)
    if not ok:
        OUT.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report, indent=1))
        raise SystemExit("FATAL: recomputed estimates do not match the stored errors; no figure.")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(shots), 1, figsize=(7.0, 5.3))
    for k, (ax, name) in enumerate(zip(axes, shots)):
        time, ti, model, pers, gp, seg_id = traces[name]
        lo, hi, n_gap = FT.pick_window(time, ti, seg_id)
        sl = slice(lo, hi)
        t = time[sl]
        obs = ~np.isnan(ti[sl])
        ax.plot(t, pers[sl], color="0.55", lw=1.0, drawstyle="steps-post", label="hold")
        ax.plot(t, gp[sl], color="#1f77b4", lw=1.1, ls="--", label="causal GP (CES only)")
        ax.plot(t, model[sl], color="#d62728", lw=1.4, label="recurrent network (first stage)")
        ax.plot(t[obs], ti[sl][obs], "o", ms=3.6, mfc="white", mec="k", mew=0.8, label=r"reported $T_i$")
        for r in np.flatnonzero(~obs):
            ax.axvspan(t[r] - 0.005, t[r] + 0.005, color="0.92", lw=0)
        # absent frames (beam blips): a step of 20 to 40 ms between consecutive rows of the window
        dt = np.diff(t)
        blips = np.flatnonzero((dt > 0.015) & (dt < 0.045))
        for j, b in enumerate(blips):
            ax.axvline(0.5 * (t[b] + t[b + 1]), color="#2ca02c", lw=0.9, ls=":",
                       label="absent frame (beam blip)" if (k == 0 and j == 0) else None)
        ax.set_ylabel(r"$T_i$ (eV)")
        vv = ti[sl][obs]
        ax.set_title(f"({'abc'[k]}) KSTAR #{name[1:6]}: {int((~obs).sum())} of {len(t)} rows "
                     f"without $T_i$; reported {vv.min():.0f} to {vv.max():.0f} eV", loc="left", fontsize=9)
        ax.margins(x=0.01)
        m = model[sl]
        below = np.flatnonzero(m < vv.min())
        report[name].update({
            "window_s": [float(t[0]), float(t[-1])],
            "rows_without_ti_in_window": int((~obs).sum()), "rows_in_window": int(len(t)),
            "measured_min_max_eV_in_window": [round(float(vv.min()), 1), round(float(vv.max()), 1)],
            "absent_frame_steps_in_window": [round(float(t[b + 1]), 3) for b in blips],
            "estimate_min_max_eV_in_window": [round(float(np.nanmin(m)), 1), round(float(np.nanmax(m)), 1)],
            "rows_with_estimate_below_measured_range": [round(float(t[r]), 3) for r in below],
        })
    axes[-1].set_xlabel("time (s)")
    hs, ls = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls:
                hs.append(h); ls.append(l)
    fig.legend(hs, ls, ncol=5, fontsize=7.2, frameon=False, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=170)
    OUT.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in ("check_passed", "typical_discharge")}, indent=1))
    for name in shots:
        r = report[name]
        print(name, "rows", r["scored_rows"], "rmse model/gp/hold",
              round(r["model"]["rmse_stored"], 1), round(r["causal_gp"]["rmse_stored"], 1),
              round(r["persistence"]["rmse_stored"], 1), "dev", round(r["model"]["max_abs_dev_of_error_eV"], 3),
              "blips", r["absent_frame_steps_in_window"], "below", r["rows_with_estimate_below_measured_range"])


if __name__ == "__main__":
    main()
