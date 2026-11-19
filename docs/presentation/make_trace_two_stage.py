# -*- coding: utf-8 -*-
"""Example traces of the talk deck with the paper's estimator: the two-stage estimator of B.64.

    py docs/presentation/make_trace_two_stage.py              # draw (computes the estimates once on the CPU, then cached)
    py docs/presentation/make_trace_two_stage.py --recompute  # recompute the estimates and the checks
    -> docs/presentation/figures/deck_ppcf/trace_two_stage_ko.png

Windows.  The three windows of the poster's example traces (figures/poster_ppcf/poster_trace_cache.npz of
make_figures_poster_ppcf.py: #31921, #31923, #31343, about 0.65 s each, picked by docs/paper/ppcf/fig_trace.py).
Hold and the causal GP drawn are the ones in that cache (the code of fig_trace.py).

Estimator.  For each discharge, the fold k of re-partition r2 of the pooled cross-validation (B.39 / B.64) in which it
is a test discharge.  First stage: that outer fold's network (data/.b39p_v2_cut_r2_f{k}, one initialisation), CPU
inference on every grid row.  Second stage: B.64 does not save the boosted model, so it is refitted here for arm v2
with the functions of stage2_b64.py (manifest, idx_of, load_net, rows_for, COLS, gbr; inner networks
data/.b64_v2_cut_r2_f{k}_i{j}), step for step as its fit_fold.

Check before anything is drawn: on the fold's scored test rows, the refit applied as in fit_fold reproduces the cached
two-stage estimates (data/.b64/stage2_r2_f{k}.npz, est_c2) to REPRO_TOL_EV.

Every row of the window.  The second-stage inputs hold, causal GP and CES age are those of stage2_b64.base_rows_of,
taken at every row (the observed T_i before the row in its block, the GP on at most GP_SIDE of them); on the window's
rows that carry T_i they must equal base.npz.  Where the fifteen inputs are not all present (the BES / ECEI window
inputs on the first ten rows of a segment, no T_i yet in the block) the first stage is kept (B.64 addendum 1).

CPU only (CUDA hidden).  Cache and check report: data/.deck_trace_two_stage/ (not committed).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # the GPU belongs to the training batches

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "ces_prediction" / "experiments" / "b64_two_stage"))
import stage2_b64 as S2  # noqa: E402  (pins the CES_* env through run_b37.Ctx, CPU)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

POSTER = HERE / "figures" / "poster_ppcf"
POSTER_CACHE, POSTER_CHECK = POSTER / "poster_trace_cache.npz", POSTER / "poster_trace_check.json"
OUT = HERE / "figures" / "deck_ppcf" / "trace_two_stage_ko.png"
WORK = ROOT / "data" / ".deck_trace_two_stage"
CACHE, REPORT = WORK / "traces.npz", WORK / "check.json"
R2, ARM = 2, "v2"
REPRO_TOL_EV = 1e-6

FIG_IN = (6.95, 3.95)        # inches: about 1000 x 570 px on the 1920 x 1080 deck canvas (144 px per inch)
DPI = 200
MIN_PT = 13.0
INK, MUTED, SHADE = "#15181e", "#5e6878", "#e9ebef"
C_TWO, C_NET, C_GP = "#0b2a6f", "#c62828", "#6f6f6f"
WORDS = {"rep": r"보고된 $T_i$", "two": "두 단계 estimator", "net": "신경망 (첫 단계)", "gp": "인과 GP (CES 이력만)",
         "title": "({p}) #{shot}: {n}행 가운데 {miss}행에 $T_i$ 없음, 보고값 {lo:.0f}–{hi:.0f} eV",
         "ylabel": r"$T_i$ (eV)", "xlabel": "시간 (s)"}


# ------------------------------------------------------------------------------------------ estimates
def fold_of(name):
    ks = [k for k in range(S2.K) if name in S2.manifest(R2, k)["test_files"]]
    if len(ks) != 1:
        raise SystemExit(f"FATAL: {name} is a test discharge of {len(ks)} folds of r{R2}")
    return ks[0]


def refit(C, base, k):
    """stage2_b64.fit_fold, arm v2, fit part: out-of-fold rows of the four inner networks plus the validation rows
    under the outer network, boosted model on the residual of the first stage."""
    paths = S2.Paths(False)
    man = S2.manifest(R2, k)
    test, val = S2.idx_of(C, man["test_files"]), S2.idx_of(C, man["val_files"])
    inner = [S2.idx_of(C, f) for f in S2.P.inner_folds(R2, k)]
    mu, sd = C.fast_stats(set(test))
    outer_net, om = S2.load_net(C, S2.B39.pooled_run_dir(ARM, R2, k), ARM)
    parts = []
    for j in range(S2.P.N_INNER):
        net, _ = S2.load_net(C, paths.inner(ARM, R2, k, j), ARM)
        parts.append(S2.rows_for(C, base, inner[j], net, mu, sd, S2.COLS[ARM]))
    pv = S2.rows_for(C, base, val, outer_net, mu, sd, S2.COLS[ARM])
    Xf = np.concatenate([q[0][q[3]] for q in parts + [pv]])
    rf = np.concatenate([(q[1] - q[2])[q[3]] for q in parts + [pv]])
    model = S2.gbr().fit(Xf[:, list(S2.COLS[ARM])], rf)
    return model, outer_net, om, mu, sd


def reproduce(C, base, pos, k, model, outer_net, mu, sd):
    """stage2_b64.fit_fold, arm v2, test part, on the fold's scored rows; compared with the cached est_c / est_c2."""
    z = S2.B54.load_rows(R2, k)
    jj = np.array([pos.get((int(a), int(b)), -1) for a, b in zip(z["fidx"], z["grow"])])
    if (jj < 0).any():
        raise SystemExit(f"FATAL: r{R2} f{k}: {(jj < 0).sum()} scored rows are not base rows")
    y = z["y"].astype(np.float64)
    c_cpu, fr = np.full(len(y), np.nan), np.full((len(y), 9), np.nan)
    for i in np.unique(z["fidx"]):
        sel = np.flatnonzero(z["fidx"] == i)
        c_cpu[sel] = C.predict(C.grid[C.names[int(i)]], outer_net)[0][z["grow"][sel]]
        fr[sel] = C.feats(int(i), mu, sd)[z["grow"][sel]]
    c = y + np.sign(c_cpu - y) * np.sqrt(z["se_c"])
    X15 = S2.inputs(fr, base["age"][jj], base["hold"][jj], c, base["gp"][jj])
    okt = np.isfinite(X15).all(1)
    est2 = c.copy()
    est2[okt] = c[okt] + model.predict(X15[:, list(S2.COLS[ARM])][okt])
    cz = np.load(S2.Paths(False).cache / f"stage2_r{R2}_f{k}.npz")
    if len(cz["est_c2"]) != len(y) or np.max(np.abs(cz["y"] - y)) > 1e-6:
        raise SystemExit(f"FATAL: r{R2} f{k}: the cached rows are not the scored rows")
    return {"fold": k, "scored_test_rows": int(len(y)),
            "second_stage_applied_rows": int(okt.sum()), "first_stage_kept_rows": int((~okt).sum()),
            "max_abs_diff_est_c2_eV": float(np.max(np.abs(est2 - cz["est_c2"]))),
            "max_abs_diff_est_c_eV": float(np.max(np.abs(c - cz["est_c"]))),
            "fallback_equal": bool(np.array_equal(~okt, cz["fallback_c2"])),
            "max_abs_diff_cpu_first_stage_vs_stored_eV": float(np.max(np.abs(c_cpu - c))),
            "test_rows_table": z}


def history_inputs(C, i, rows):
    """hold, causal GP and CES age (ms) of stage2_b64.base_rows_of at the given grid rows of file i, also on rows
    without T_i: the observed T_i before the row in its block (none: NaN), the GP on at most GP_SIDE of them."""
    arr = C.grid[C.names[i]]
    tm, y = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
    h, g, age = (np.full(len(rows), np.nan) for _ in range(3))
    spans = S2.R.block_spans(tm, C.GAP)
    for n, r in enumerate(rows):
        a = next((a for a, b in spans if a <= r < b), None)
        if a is None:
            continue
        obs = np.flatnonzero(~np.isnan(y[a:r])) + a
        if not len(obs):
            continue
        lo = max(0, len(obs) - S2.R.GP_SIDE)
        h[n] = y[obs[-1]]
        g[n] = C.B.predict_gp_causal(tm[obs[lo:]], y[obs[lo:]], tm[r])
        age[n] = (tm[r] - tm[obs[-1]]) * 1000.0
    return h, g, age


def compute():
    if not json.loads(POSTER_CHECK.read_text(encoding="utf-8"))["check_passed"]:
        raise SystemExit("FATAL: the poster's trace check failed")
    D = np.load(POSTER_CACHE)
    C = S2.ctx()
    paths = S2.Paths(False)
    base = S2.load_base(paths)
    pos = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(base["fidx"], base["row"]))}
    names = [str(D[f"{p}_name"]) for p in range(int(D["n"]))]
    folds = {nm: fold_of(nm) for nm in names}
    report, arrays, fitted = {"re_partition": R2, "arm": ARM, "folds": {}}, {"n": len(names)}, {}
    for k in sorted(set(folds.values())):
        S2.log(f"r{R2} f{k}: refitting the second stage")
        model, outer_net, om, mu, sd = refit(C, base, k)
        rep = reproduce(C, base, pos, k, model, outer_net, mu, sd)
        z = rep.pop("test_rows_table")
        rep.update({"outer_network": S2.B39.pooled_run_dir(ARM, R2, k).name, "init_seed": om.get("init_seed"),
                    "inner_networks": [paths.inner(ARM, R2, k, j).name for j in range(S2.P.N_INNER)]})
        report["folds"][str(k)] = rep
        S2.log(f"r{R2} f{k}: est_c2 max abs diff {rep['max_abs_diff_est_c2_eV']:.3g} eV, "
               f"est_c {rep['max_abs_diff_est_c_eV']:.3g} eV, fallback equal {rep['fallback_equal']}")
        if rep["max_abs_diff_est_c2_eV"] > REPRO_TOL_EV or not rep["fallback_equal"]:
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
            raise SystemExit(f"FATAL: r{R2} f{k}: the refit does not reproduce the cached two-stage estimates")
        fitted[k] = (model, outer_net, mu, sd, z)
    for p, nm in enumerate(names):
        k = folds[nm]
        model, outer_net, mu, sd, z = fitted[k]
        i = C.names.index(nm)
        arr = C.grid[nm]
        tm, y = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
        t, ti = D[f"{p}_time"], D[f"{p}_ti"]
        rows = np.searchsorted(tm, t)
        if not (np.array_equal(tm[rows], t) and np.array_equal(np.isnan(y[rows]), np.isnan(ti))
                and np.nanmax(np.abs(y[rows] - ti)) < 1e-6):
            raise SystemExit(f"FATAL: {nm}: the window rows of the poster cache are not grid rows of the cut grid")
        h, g, age = history_inputs(C, i, rows)
        sel = base["by_file"][i]
        lut = {int(r): j for j, r in zip(sel, base["row"][sel])}
        at = [(n, lut[int(r)]) for n, r in enumerate(rows) if int(r) in lut]
        nb, jb = np.array([a for a, _ in at]), np.array([b for _, b in at])
        dev_base = max(float(np.max(np.abs(h[nb] - base["hold"][jb]))), float(np.max(np.abs(g[nb] - base["gp"][jb]))),
                       float(np.max(np.abs(age[nb] - base["age"][jb]))))
        if dev_base > 0:
            raise SystemExit(f"FATAL: {nm}: history inputs differ from base.npz by {dev_base}")
        c = C.predict(arr, outer_net)[0][rows]
        X15 = S2.inputs(C.feats(i, mu, sd)[rows], age, h, c, g)
        ok = np.isfinite(X15).all(1)
        two = c.copy()
        two[ok] = c[ok] + model.predict(X15[:, list(S2.COLS[ARM])][ok])
        # the window's scored rows against the cached two-stage estimates (first stage: CPU here, stored B.39 there)
        cz = np.load(paths.cache / f"stage2_r{R2}_f{k}.npz")
        zr = {int(r): j for j, r in zip(np.flatnonzero(z["fidx"] == i), z["grow"][z["fidx"] == i])}
        ws = [(n, zr[int(r)]) for n, r in enumerate(rows) if int(r) in zr]
        wn, wz = np.array([a for a, _ in ws]), np.array([b for _, b in ws])
        gp_p, hold_p = D[f"{p}_gp"], D[f"{p}_pers"]
        obs = ~np.isnan(ti)
        m = obs & np.isfinite(c) & np.isfinite(two) & np.isfinite(gp_p) & np.isfinite(hold_p)
        rmse = {e: float(np.sqrt(np.mean((v[m] - ti[m]) ** 2)))
                for e, v in (("two_stage", two), ("first_stage", c), ("causal_gp", gp_p), ("hold", hold_p))}
        report[nm] = {"fold": k, "rows_in_window": int(len(t)), "rows_with_ti": int(obs.sum()),
                      "rows_first_stage_kept": int((~ok).sum()), "first_stage_kept_at_s": [round(float(x), 3) for x in t[~ok]],
                      "window_scored_rows": int(len(ws)),
                      "max_abs_diff_two_stage_vs_cached_on_window_scored_rows_eV": float(np.max(np.abs(two[wn] - cz["est_c2"][wz]))),
                      "max_abs_diff_first_stage_vs_cached_on_window_scored_rows_eV": float(np.max(np.abs(c[wn] - cz["est_c"][wz]))),
                      "max_abs_diff_history_gp_vs_poster_gp_eV": float(np.nanmax(np.abs(g - gp_p))),
                      "max_abs_diff_history_hold_vs_poster_hold_eV": float(np.nanmax(np.abs(h - hold_p))),
                      "rmse_rows": int(m.sum()), "rmse_eV": rmse}
        S2.log(f"{nm} (f{k}): rmse two-stage {rmse['two_stage']:.1f}, first stage {rmse['first_stage']:.1f}, "
               f"causal GP {rmse['causal_gp']:.1f}, hold {rmse['hold']:.1f} eV on {int(m.sum())} rows")
        for key, v in (("time", t), ("ti", ti), ("first", c), ("two", two), ("gp", gp_p), ("hold", hold_p), ("kept", ~ok)):
            arrays[f"{p}_{key}"] = v
        arrays[f"{p}_name"] = np.array(nm)
        arrays[f"{p}_fold"] = np.array(k)
    WORK.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    np.savez(CACHE, **arrays)
    S2.log(f"written {CACHE.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}")


# ------------------------------------------------------------------------------------------ figure
def set_font():
    user = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
    for f in ("Pretendard-Regular.otf", "Pretendard-SemiBold.otf", "Pretendard-Bold.otf"):
        if (user / f).exists():
            fm.fontManager.addfont(str(user / f))
    family = "Pretendard" if "Pretendard" in {f.name for f in fm.fontManager.ttflist} else "Malgun Gothic"
    plt.rcParams.update({
        "font.family": family, "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 13, "axes.titlesize": 13.5, "axes.titlelocation": "left",
        "mathtext.fontset": "custom", "mathtext.rm": "Segoe UI", "mathtext.it": "Segoe UI:italic",
        "mathtext.bf": "Segoe UI:bold", "mathtext.sf": "Segoe UI",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9, "axes.edgecolor": INK,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
        "legend.frameon": False, "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.04,
        "figure.constrained_layout.hspace": 0.02})


def draw(narrow=False):
    """narrow: the variant for the right column of the deck's result slide (about 590 x 600 px)."""
    fig_in, out = ((4.1, 4.2), OUT.with_name("trace_two_stage_narrow_ko.png")) if narrow else (FIG_IN, OUT)
    D = np.load(CACHE)
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    set_font()
    fig, axes = plt.subplots(int(D["n"]), 1, figsize=fig_in, layout="constrained")
    for p, ax in enumerate(axes):
        t, ti, first, two, gp = (D[f"{p}_{x}"] for x in ("time", "ti", "first", "two", "gp"))
        name = str(D[f"{p}_name"])
        obs = ~np.isnan(ti)
        for r in np.flatnonzero(~obs):
            ax.axvspan(t[r] - 0.005, t[r] + 0.005, color=SHADE, lw=0, zorder=0)
        ax.plot(t, gp, color=C_GP, lw=1.3, ls=(0, (4, 2.5)), label=WORDS["gp"], zorder=2)
        ax.plot(t, first, color=C_NET, lw=1.2, label=WORDS["net"], zorder=3)
        ax.plot(t, two, color=C_TWO, lw=2.3, label=WORDS["two"], zorder=4)
        ax.plot(t[obs], ti[obs], "o", ms=4.6, mfc="white", mec="k", mew=0.9, label=WORDS["rep"], zorder=5)
        vv = ti[obs]
        if narrow:
            ax.set_title(f"({'abc'[p]}) #{name[1:6]}: {int((~obs).sum())}/{len(t)}행 Ti 없음", pad=3)
        else:
            ax.set_title(WORDS["title"].format(p="abc"[p], shot=name[1:6], n=len(t), miss=int((~obs).sum()),
                                               lo=vv.min(), hi=vv.max()), pad=3)
        ax.set_ylabel(WORDS["ylabel"])
        ax.margins(x=0.01, y=0.12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(axis="both", pad=2)
        if rep[name]["fold"] != int(D[f"{p}_fold"]):
            raise SystemExit("FATAL: cache and report disagree")
    axes[-1].set_xlabel(WORDS["xlabel"])
    hs, ls = axes[0].get_legend_handles_labels()
    order = [ls.index(WORDS[x]) for x in ("rep", "two", "net", "gp")]
    leg = fig.legend([hs[i] for i in order], [ls[i] for i in order], ncol=2 if narrow else 4, loc="outside upper center",
                     handlelength=1.3, handletextpad=0.4, columnspacing=0.9, borderaxespad=0.1)
    fig.canvas.draw()
    small = [(x.get_text()[:25], x.get_fontsize()) for x in fig.findobj(matplotlib.text.Text)
             if x.get_visible() and x.get_text().strip() and x.get_fontsize() < MIN_PT - 1e-6]
    if small:
        raise SystemExit(f"FATAL: text below {MIN_PT} pt: {small}")
    lb = leg.get_window_extent()
    if lb.x0 < 0 or lb.x1 > fig.bbox.width:
        raise SystemExit("FATAL: the legend is wider than the figure")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("written %s  %d x %d px" % (out.relative_to(ROOT), round(fig_in[0] * DPI), round(fig_in[1] * DPI)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="recompute the estimates and the checks (CPU)")
    a = ap.parse_args()
    if a.recompute or not CACHE.exists():
        compute()
    draw()
    draw(narrow=True)


if __name__ == "__main__":
    main()
