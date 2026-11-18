"""Figure: error against the time since the last CES observation (KSTAR).

Descriptive, not registered.  Rows are the frozen test rows of the repeated 5-fold
cross-validation (repeats 2-9, five folds concatenated per repeat); each row carries the
time since the last observed value of its target (dt_ms) and the squared error of every
estimator.  Skill per bin: two-level bootstrap (draw a repeat, resample its discharges),
the reader used for the pooled readout of the paper.  Rotation rows are under the
1,000 km/s scoring rule.  The unbinned values are checked against the published ones.
Run from the repo root:  py docs/paper/ppcf/fig_gap.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "docs" / "paper" / "review_loop" / "analyses" / "R2"))
import r2_common as C  # noqa: E402
import analyze_b27c as A  # noqa: E402
import r2ac_vrot_running_mean as R  # noqa: E402

PRIMARY = tuple(range(2, 10))
ARMS = ("model", "gp_causal", "persistence", "pchip")
REFS = ("gp_causal", "persistence", "pchip")
TI_BINS = (("10", 0, 15), ("20", 15, 25), ("30-50", 25, 55), (">50", 55, 1e9))
VT_BINS = (("10", 0, 15), (">=20", 15, 1e9))
EXPECT = {("cut", "CES_TI", "gp_causal"): (0.113, 0.085, 0.132),
          ("cut", "CES_TI", "persistence"): (0.391, 0.369, 0.408),
          ("cut", "CES_VT", "persistence"): (0.265, 0.198, 0.326),
          ("cut", "CES_VT", "pchip"): (0.092, 0.0003, 0.170)}
VR = "$V_" + chr(92) + "mathrm{rot}$"


def load(pop, target):
    P = {}
    for r in PRIMARY:
        parts = []
        for k in range(5):
            z = np.load(C.DATA / f".b27_seqv2_{pop}_r{r}_f{k}" / "comparison_errors_test.npz")
            d = {"shot": z[f"{target}_shot"], "row": z[f"{target}_row"], "dt": z[f"{target}_dt_ms"]}
            for a in ARMS:
                d[a] = z[f"{target}_se_{a}"].astype(np.float64)
            parts.append(d)
        P[r] = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
    return P


def hier(P, keep, a, b):
    sums = {r: tuple(C.shot_sums(P[r]["shot"][keep[r]], P[r][a][keep[r]], P[r][b][keep[r]])) for r in PRIMARY}
    h = A.hier(sums, PRIMARY, np.random.default_rng(A.SEED))
    return {"median": h["median"], "ci95": h["ci95"],
            "excludes_0": bool(h["ci95"][0] > 0 or h["ci95"][1] < 0)}


def analyse(P, base, bins):
    out = {}
    cells = (("all", 0, 1e9),) + tuple(bins)
    for name, lo, hi in cells:
        keep = {r: base[r] & (P[r]["dt"] >= lo) & (P[r]["dt"] < hi) for r in PRIMARY}
        n = [int(keep[r].sum()) for r in PRIMARY]
        ns = [int(len(np.unique(P[r]["shot"][keep[r]]))) for r in PRIMARY]
        out[name] = {"rows_per_repeat": [min(n), max(n)], "discharges_per_repeat": [min(ns), max(ns)],
                     "rmse": {a: float(np.median([np.sqrt(P[r][a][keep[r]].mean()) for r in PRIMARY]))
                              for a in ARMS},
                     "skill": {b: hier(P, keep, "model", b) for b in REFS}}
    return out


def main():
    if "--redraw" in sys.argv:
        draw(json.loads((HERE / "fig_gap.json").read_text(encoding="utf-8")))
        return
    res = {"note": "descriptive; repeats 2-9 of the 5-fold cross-validation; two-level bootstrap "
                   "(analyze_b27c.hier); rotation under the 1,000 km/s rule; dt = time since the last "
                   "observed value of the target", "check": {}}
    for pop in ("cut", "incl"):
        res[pop] = {}
        P = load(pop, "CES_TI")
        base = {r: np.ones(len(P[r]["shot"]), bool) for r in PRIMARY}
        res[pop]["CES_TI"] = analyse(P, base, TI_BINS)
        ds = C.build_dataset(pop)
        tc = int(ds._column_slices["time"])
        vc = int(ds._column_slices["target"][1])
        tabs = [R.file_tables(fa, tc, vc) for fa in ds.file_arrays]
        P = load(pop, "CES_VT")
        base = {}
        for r in PRIMARY:
            amax = np.array([tabs[int(s)]["amax"][int(i)] for s, i in zip(P[r]["shot"], P[r]["row"])])
            base[r] = amax <= R.T_RULE
        res[pop]["CES_VT"] = analyse(P, base, VT_BINS)
        res[pop]["CES_VT_rows_dropped_per_repeat"] = [int((~base[r]).sum()) for r in PRIMARY]
        for t in ("CES_TI", "CES_VT"):
            for name, c in res[pop][t].items():
                print(f"[{pop}] {t} dt {name:6s} rows {c['rows_per_repeat']} shots {c['discharges_per_repeat']} "
                      + " ".join(f"{a} {c['rmse'][a]:.2f}" for a in ARMS) + " | "
                      + " ".join(f"vs {b} {c['skill'][b]['median']:+.4f} [{c['skill'][b]['ci95'][0]:+.4f}, "
                                 f"{c['skill'][b]['ci95'][1]:+.4f}]" for b in REFS), flush=True)
    bad = 0
    for (pop, t, b), (m, lo, hi) in EXPECT.items():
        s = res[pop][t]["all"]["skill"][b]
        ok = abs(s["median"] - m) < 6e-4 and abs(s["ci95"][0] - lo) < 6e-4 and abs(s["ci95"][1] - hi) < 6e-4
        res["check"][f"{pop}/{t}/{b}"] = {"published": [m, lo, hi], "here": [s["median"]] + s["ci95"],
                                          "match": bool(ok)}
        bad += not ok
    print("published-value check mismatches:", bad)
    (HERE / "fig_gap.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if bad:
        raise SystemExit("FATAL: unbinned skills do not reproduce the published values")
    draw(res)


def draw(res):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "stix",
                         "axes.spines.top": False, "axes.spines.right": False})
    lab = {"model": "network", "gp_causal": "causal GP", "persistence": "hold",
           "pchip": "PCHIP (next sample)"}
    col = {"model": "#222222", "gp_causal": "#4c72b0", "persistence": "#c44e52", "pchip": "#55a868"}
    mk = {"model": "o", "gp_causal": "s", "persistence": "^", "pchip": "D"}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.15, 1.15, 0.75]})
    T = res["cut"]["CES_TI"]
    names = [n for n, _, _ in TI_BINS]
    x = np.arange(len(names))
    for arm in ARMS:
        a.plot(x, [T[n]["rmse"][arm] for n in names], marker=mk[arm], ms=4, lw=1.1, color=col[arm],
               label=lab[arm])
    a.set_xticks(x)
    a.set_xticklabels(names)
    a.set_xlabel("time since last $T_i$ sample (ms)")
    a.set_ylabel("$T_i$ RMSE (eV)")
    a.legend(frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
             columnspacing=0.9, handletextpad=0.3)
    a.text(0.04, 0.9, "(a)", transform=a.transAxes, fontsize=10)

    def skill_panel(ax, cells, names, off=0.2):
        xx = np.arange(len(names))
        for j, ref in enumerate(REFS):
            m = np.array([cells[n]["skill"][ref]["median"] for n in names])
            lo = np.array([cells[n]["skill"][ref]["ci95"][0] for n in names])
            hi = np.array([cells[n]["skill"][ref]["ci95"][1] for n in names])
            ax.errorbar(xx + (j - 1) * off, m, yerr=[m - lo, hi - m], fmt=mk[ref], ms=4, lw=1.0, capsize=2,
                        color=col[ref], label="vs " + lab[ref].split(" (")[0])
        ax.axhline(0, color="#888888", lw=0.7)
        ax.set_xticks(xx)
        ax.set_xticklabels(names)

    skill_panel(b, T, names)
    b.set_xlabel("time since last $T_i$ sample (ms)")
    b.set_ylabel("MSE skill")
    b.legend(frameon=False, fontsize=6.8, loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=3,
             columnspacing=0.9, handletextpad=0.2)
    b.text(0.04, 0.9, "(b)", transform=b.transAxes, fontsize=10)
    b.set_ylim(-0.04, 0.62)
    V = res["cut"]["CES_VT"]
    skill_panel(c, V, [n for n, _, _ in VT_BINS], off=0.22)
    c.set_xticklabels(["10", r"$\geq$20"])
    c.set_xlim(-0.6, 1.6)
    c.set_xlabel("time since last\n" + VR + " sample (ms)")
    c.text(0.06, 0.9, "(c)", transform=c.transAxes, fontsize=10)
    c.set_ylim(-0.06, 0.8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_gap.pdf")
    fig.savefig(HERE / "fig_gap_preview.png", dpi=170)


if __name__ == "__main__":
    main()
