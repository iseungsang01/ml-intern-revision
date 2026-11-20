"""B.79 readout (registration sections 5-6, Addendum 1 sections 6-7): every table and statement in one pass into the
JSON, then the markdown.  The progress log carries row counts only.

The interval code is B.75's (b75_availability/readout.py: Group, subs_of, describe, expectation_met), imported by path,
with its rng_for replaced by this batch's (seed 20261179).

    py ces_prediction/experiments/b79_beam_inputs/readout79.py [--out PATH]          # main run (columns as delivered)
    B79_VARIANT=refit py .../readout79.py [--out PATH]                              # refit without the thirteen files
    py ces_prediction/experiments/b79_beam_inputs/readout79.py merge                # final JSON and markdown
The main run's readout was computed at 17:04 on 2026-11-20 and written to docs/paper/b79_beam_inputs.{json,md}; it is kept
as data/.b79/readout_main.json.  The refit's readout is data/.b79/refit_valid/readout_refit.json (Addendum 1 sections 11-13).
`merge` writes docs/paper/b79_beam_inputs.json = the main readout plus the key "refit_without_13_files", and the markdown =
the main readout as rendered at 17:04 plus the refit section.
"""
from __future__ import annotations

import importlib.util
import json
import time
import types
import warnings

import common79 as M

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = M.C


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R75 = load_by_path("readout75", M.K75.HERE / "readout.py")
R75.K = types.SimpleNamespace(**{**vars(M.K75), "rng_for": M.rng_for, "SEED": M.SEED})   # B.75's draws, seed of B.79
Group, subs_of, describe, expectation_met = R75.Group, R75.subs_of, R75.describe, R75.expectation_met
N = R75.N
FORMS = R75.FORMS
DESIGNS = ("V1", "V2", "V3", "V4a", "V4b")
HALF_DESIGNS = ("V1", "V4a", "V4b")
RULES_OF = {"common": ("D10", "D650"), "S": ("S",)}
NOTCH_RULE = {"common": "D650", "S": "S"}
LABS = ("G", "T0", "TN", "TX", "TXN", "TEN", "TBN", "TNp", "TXNp")
SENS = "outside the 13 files"


def load_est(units, L, r, X, n):
    a = np.full(n, np.nan)
    for u in units:
        z = np.load(M.FITS / f"{u['name']}__{L}__{r}__{X}.npz")
        a[z["idx"]] = z["est"]
    return a


def estimator_list(kind, X, d):
    out = []
    for r in RULES_OF[kind]:
        out.append((f"G|{r}", None, r))
        out += [(f"{L}|{r}", L, r) for L in ("T0", "TN", "TX", "TXN")]
        if d in HALF_DESIGNS and r in ("D650", "S"):
            out += [(f"{L}|{r}", L, r) for L in ("TEN", "TBN")]
    if d == "V1" and kind == "common" and X == "L5":
        out += [("TNp|D650", "TNp", "D650"), ("TXNp|D650", "TXNp", "D650")]
    if X == "L5":
        out.append(("O", None, None))
    return out


def build_group(name, rows, strata, subs, kind, X, d, R, Gbase, unit_d, unit_c):
    g = Group(name, rows, strata, unit_d, unit_c, len(subs))
    y = R[f"tg_{X}"][rows]
    n = len(R["y"])
    for e, L, r in estimator_list(kind, X, d):
        if L is None:
            v = R["y"][rows] if e == "O" else Gbase[r][rows]
            se = (v - y) ** 2
            for si in range(len(subs)):
                g.add(e, si, se)
            continue
        for si, units in enumerate(subs):
            a = load_est(units, L, r, X, n)
            g.add(e, si, (a[rows] - y) ** 2)
    return g


def ratio_pack(g, s, fn, es):
    """g._f plus the number of draws (per form) whose value is not finite."""
    v = g._f(s, fn, *es)
    v["draws_not_finite"] = {}
    for form in FORMS:
        t = g.tot[(s, form)]
        with np.errstate(invalid="ignore", divide="ignore"):
            x = fn(*[t[e][1] for e in es])
        v["draws_not_finite"][form] = int((~np.isfinite(x)).sum())
    return v


def rule_block(g, s, r, X):
    e = lambda lab: f"{lab}|{r}"   # noqa: E731
    o = {"rmse_eV": {lab: g.rmse(s, e(lab)) for lab in LABS if g.has(s, e(lab))}}
    if X == "L5":
        o["rmse_eV"]["O"] = g.rmse(s, "O")
    o["TN/T0"] = g.skill(s, e("TN"), e("T0"))
    o["TXN/TN"] = g.skill(s, e("TXN"), e("TN"))
    o["TX/T0"] = g.skill(s, e("TX"), e("T0"))
    o["TXN/TX"] = g.skill(s, e("TXN"), e("TX"))
    o["TXN/T0"] = g.skill(s, e("TXN"), e("T0"))
    with np.errstate(invalid="ignore", divide="ignore"):
        o["ratio of skills (TXN/TN)/(TX/T0)"] = ratio_pack(
            g, s, lambda A, B, Cc, D: (1 - A / B) / (1 - Cc / D), (e("TXN"), e("TN"), e("TX"), e("T0")))
        o["ratio of absolute reductions"] = ratio_pack(
            g, s, lambda A, B, Cc, D: (B - A) / (D - Cc), (e("TXN"), e("TN"), e("TX"), e("T0")))
    for lab in ("TEN", "TBN"):
        if g.has(s, e(lab)):
            o[f"{lab}/TN"] = g.skill(s, e(lab), e("TN"))
    if g.has(s, e("TNp")):
        o["TNp/T0"] = g.skill(s, e("TNp"), e("T0"))
        o["TN/TNp"] = g.skill(s, e("TN"), e("TNp"))
    if g.has(s, e("TXNp")):
        o["TXNp/TN"] = g.skill(s, e("TXNp"), e("TN"))
        o["TXN/TXNp"] = g.skill(s, e("TXN"), e("TXNp"))
    return o


# ------------------------------------------------------------------------------------------ statements
def statements(res):
    S = {}

    def put(k, text, v, holds, exp, note=""):
        S[k] = {"text": text, "value": {kk: v[kk] for kk in ("point", "discharge", "cluster")}, "holds": bool(holds),
                "expected": exp, "note": note}

    def ivs(v):
        return f"{v['point']:+.3f}; c [{v['cluster'][0]:+.3f}, {v['cluster'][1]:+.3f}]"

    c = lambda d: res["main"][f"{d}|L5|common"]["all"]   # noqa: E731
    s_ = lambda d: res["main"][f"{d}|L5|S"]["all"]       # noqa: E731
    v = c("V1")["D650"]["TN/T0"]
    put("N1", "V1, D650: TN over T0 lies above zero (the beam waveforms add to the history)", v, v["cluster"][0] > 0,
        "holds (+0.03 to +0.15)")
    v = c("V1")["D650"]["TXN/TN"]
    put("N2", "V1, D650: TXN over TN lies above zero (the fast diagnostics add beyond the beam waveforms)", v,
        v["cluster"][0] > 0, "holds (+0.20 to +0.35)", f"TX over T0: {ivs(c('V1')['D650']['TX/T0'])}")
    v = c("V2")["D650"]["TXN/TN"]
    put("N3", "V2, D650: TXN over TN lies above zero", v, v["cluster"][0] > 0, "holds")
    v = c("V3")["D650"]["TXN/TN"]
    put("N4", "V3, D650: TXN over TN lies above zero", v, v["cluster"][0] > 0, "holds")
    v = c("V4a")["D650"]["TXN/TN"]
    put("N5", "V4 (fit to 31991), D650: TXN over TN lies above zero", v, v["cluster"][0] > 0, "uncertain, leaning to holds",
        f"V4b (fit to 32310): {ivs(c('V4b')['D650']['TXN/TN'])}")
    v = s_("V1")["S"]["TXN/TN"]
    put("N6", "V1, S: TXN over TN lies above zero", v, v["cluster"][0] > 0, "holds")
    v = s_("V4a")["S"]["TXN/TN"]
    put("N7", "V4 (fit to 31991), S: TXN over TN lies above zero", v, v["cluster"][0] > 0, "uncertain",
        f"V4b: {ivs(s_('V4b')['S']['TXN/TN'])}")
    v = s_("V4a")["S"]["TN/T0"]
    put("N8", "V4 (fit to 31991), S: TN over T0 lies above zero", v, v["cluster"][0] > 0, "uncertain",
        f"V4b: {ivs(s_('V4b')['S']['TN/T0'])}")
    v = c("V1")["D650"]["TXN/TX"]
    put("N9", "V1, D650: TXN over TX lies above zero (the beam waveforms add on top of the fast diagnostics)", v,
        v["cluster"][0] > 0, "uncertain (0.00 to +0.05)")
    v = c("V1")["D10"]["TXN/TN"]
    put("N10", "V1, D10: TXN over TN lies above zero", v, v["cluster"][0] > 0, "holds (+0.05 to +0.10)")
    v = c("V1")["D650"]["TNp/T0"]
    put("N11", "V1, D650, beam scalars permuted: the interval of TN over T0 contains zero or lies below it", v,
        v["cluster"][0] <= 0, "holds", f"fast scalars permuted, TXNp over TN: {ivs(c('V1')['D650']['TXNp/TN'])}")
    return S


STATEMENT_CELLS = {"N1": ("V1", "common", "D650", "TN/T0"), "N2": ("V1", "common", "D650", "TXN/TN"),
                   "N3": ("V2", "common", "D650", "TXN/TN"), "N4": ("V3", "common", "D650", "TXN/TN"),
                   "N5": ("V4a", "common", "D650", "TXN/TN"), "N6": ("V1", "S", "S", "TXN/TN"),
                   "N7": ("V4a", "S", "S", "TXN/TN"), "N8": ("V4a", "S", "S", "TN/T0"),
                   "N9": ("V1", "common", "D650", "TXN/TX"), "N10": ("V1", "common", "D10", "TXN/TN"),
                   "N11": ("V1", "common", "D650", "TNp/T0")}


def sensitivity(res):
    """Main run only (Addendum 1 section 7): the statements' quantities on the scored rows outside the thirteen files."""
    out = {}
    for k, (d, kind, r, q) in STATEMENT_CELLS.items():
        G = res["main"][f"{d}|L5|{kind}"]
        if SENS in G:
            out[k] = {"value": G[SENS][r][q], "counts": G[SENS]["counts"], "scored_rows_of_the_13_files": True}
        else:
            out[k] = {"value": G["all"][r][q], "counts": G["all"]["counts"], "scored_rows_of_the_13_files": False}
    return out


def unit_counts(R, sets, us):
    """Per unit and set: fit and test discharges with at least one row in the set, and their rows (Addendum 1 sec. 11)."""
    out = {}
    for u in us:
        fd, td = np.isin(R["fidx"], u["fit"]), np.isin(R["fidx"], u["test"])
        o = {"fit_discharges_of_unit": int(len(u["fit"])), "test_discharges_of_unit": int(len(u["test"]))}
        for k in ("common_L5", "S_L5", "common_F", "S_F"):
            m = sets[k]
            o[k] = {"fit_discharges": int(np.unique(R["fidx"][fd & m]).size), "fit_rows": int((fd & m).sum()),
                    "test_discharges": int(np.unique(R["fidx"][td & m]).size), "test_rows": int((td & m).sum())}
        out[u["name"]] = o
    return out


# ------------------------------------------------------------------------------------------ main
K_ROWS = ("fidx", "grow", "gi", "y", "shot", "clus", "blk", "tstep", "tg_F", "tg_L5", "r2_k", "r3_k", "r4_k")


def main(out_path=None):
    """The readout of the run selected by B79_VARIANT: main (with the scoring-only stratum outside the thirteen files,
    Addendum 1 section 7) or refit (without it; with the counts per unit)."""
    t_start = time.time()
    refit = M.VARIANT == "refit"
    R = M.load_keys(C.CACHE / "rows.npz", K_ROWS)
    sets = M.load_sets()
    us, tab, fifth = M.units(R)
    _, unit_d = np.unique(R["fidx"], return_inverse=True)
    _, unit_c = np.unique(R["clus"], return_inverse=True)
    Gbase = {r: M.K75.load_inputs(r)["G"] for r in M.RULES}
    Gd = M.grid_lean(("time", "seg"))
    tm, seg = Gd["time"], Gd["seg"]
    jS = M.K75.load_inputs("S")["jS_g"]
    ncls = M.notch_class(tm, R["gi"], jS)
    # blocks with an absent step anywhere (for the count of the third stratum's rows that lie before a later step)
    dt = np.diff(tm)
    ab = (seg[1:] == seg[:-1]) & (dt > M.K75.AB_LO) & (dt < M.K75.AB_HI)
    blk_has = np.zeros(int(seg.max()) + 1, bool)
    blk_has[seg[:-1][ab]] = True
    files13 = None if refit else sets["files13"]
    res = {"registration": M.REG, "addendum": M.ADD, "n_boot": N, "seed": M.SEED, "beam_scalars": list(M.BEAM_NAMES),
           "sets": json.loads((M.VDIR / "sets.json").read_text(encoding="utf-8")), "designs": describe(us, tab, fifth),
           "main": {}, "counts": {}, "notch_counts": {}}
    for d in DESIGNS:
        subs = subs_of(us, d)
        scored = np.isin(R["fidx"], np.concatenate([u["test"] for u in subs[0]]))
        for X in M.TARGETS:
            for kind in ("common", "S"):
                rows = np.flatnonzero(scored & sets[f"{kind}_{X}"])
                strata = {"all": np.ones(len(rows), bool)}
                notch = []
                if d == "V1":
                    for b in range(3):
                        strata[f"block {C.BLOCKS[b][0]}"] = R["blk"][rows] == b
                    cl = ncls[rows]
                    for ci, nm in enumerate(M.NOTCH_NAMES):
                        if (cl == ci).any():
                            strata[nm] = cl == ci
                            notch.append(nm)
                    m2 = cl == 2
                    res["notch_counts"][f"V1|{X}|{kind}"] = {
                        nm: int((cl == ci).sum()) for ci, nm in enumerate(M.NOTCH_NAMES)}
                    res["notch_counts"][f"V1|{X}|{kind}"]["no absent step earlier, but one later in the block"] = int(
                        (m2 & blk_has[seg[R["gi"][rows]]]).sum())
                if files13 is not None:
                    f13 = files13[rows]
                    if f13.any():
                        strata[SENS] = ~f13
                name = f"{d}|{X}|{kind}"
                g = build_group(name, rows, strata, subs, kind, X, d, R, Gbase, unit_d, unit_c)
                out = {}
                for s in strata:
                    g.finish(s)
                    o = {"counts": g.counts(s)}
                    rules = RULES_OF[kind] if s not in notch else (NOTCH_RULE[kind],)
                    for r in rules:
                        o[r] = rule_block(g, s, r, X)
                    out[s] = o
                    g.tot.clear()
                res["main"][name] = out
                M.log(f"{name}: rows {len(rows):,}, strata {len(strata)} done")
    # ---- counts per design and fit sizes
    for d in DESIGNS:
        subs = subs_of(us, d)
        c_ = {f"{kind}|{X}": res["main"][f"{d}|{X}|{kind}"]["all"]["counts"] for X in M.TARGETS for kind in ("common", "S")}
        fr = {}
        for units in subs:
            for u in units:
                for r in M.RULES:
                    z = np.load(M.FITS / f"{u['name']}__TXN__{r}__L5.npz")
                    fr.setdefault(f"TXN|{r}|L5", []).append(int(z["fit_rows"]))
        c_["fit_rows_per_unit_L5"] = {k: {"min": min(v), "max": max(v)} for k, v in fr.items()}
        res["counts"][d] = c_
    res["statements"] = statements(res)
    res["expectations"] = expectation_met(res["statements"])
    if refit:
        res["unit_counts"] = unit_counts(R, sets, us)
        res["checks"] = {n: json.loads((M.CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("join", "validity")}
        for n in ("repro", "causality", "scalars_vs_as_delivered"):
            res["checks"][n] = json.loads((M.VCHK / f"{n}.json").read_text(encoding="utf-8"))
        fv = json.loads((M.VCHK / "fits_vs_b75.json").read_text(encoding="utf-8"))
        res["checks"]["fits_vs_b75"] = {k: v for k, v in fv.items() if k != "per_file"}
    else:
        res["sensitivity_outside_13_files"] = sensitivity(res)
        res["checks"] = {n: json.loads((M.CHK / f"{n}.json").read_text(encoding="utf-8"))
                         for n in ("join", "repro", "causality", "fits_vs_b75")}
    res["runtime_readout_s"] = round(time.time() - t_start, 1)
    out_path = out_path or (M.REFIT_DIR / "readout_refit.json" if refit else M.CACHE / "readout_main.json")
    M.save_json(out_path, res)
    M.log(f"written {out_path}")


# ------------------------------------------------------------------------------------------ merge
ORDER_OF_EVENTS = (
    "The session lead's instruction to leave the thirteen files out of every fit and score was sent before the fits and did "
    "not reach the implementing agent before the main readout had been run (17:04) and read. The main readout uses the beam "
    "columns as delivered, as the registration reads; its statement table is the judged one. The refit was run afterwards "
    "under the validity rule, with every number of the main readout known (Addendum 1 sections 11 to 13).")


def merge():
    """docs/paper/b79_beam_inputs.json = the main readout (unchanged) plus the key 'refit_without_13_files';
    docs/paper/b79_beam_inputs.md = the main readout as rendered at 17:04 plus the refit section."""
    main_ = json.loads((M.CACHE / "readout_main.json").read_text(encoding="utf-8"))
    rf = json.loads((M.REFIT_DIR / "readout_refit.json").read_text(encoding="utf-8"))
    va = json.loads((M.CHK / "validity.json").read_text(encoding="utf-8"))
    R = M.load_keys(C.CACHE / "rows.npz", ("fidx", "shot", "clus", "r2_k", "r3_k", "r4_k"))
    us, _, _ = M.units(R)
    s_main, s_ref = M.load_npz(M.CACHE / "sets.npz"), M.load_npz(M.REFIT_DIR / "sets.npz")
    per_set = {}
    for X in M.TARGETS:
        for nm in ("common", "S"):
            k = f"{nm}_{X}"
            a, b = s_main[k], s_ref[k]
            if np.any(b & ~a):
                raise SystemExit(f"FATAL: {k}: a refit row outside the main set")
            per_set[k] = {"main_rows": int(a.sum()), "main_discharges": int(np.unique(R["fidx"][a]).size),
                          "refit_rows": int(b.sum()), "refit_discharges": int(np.unique(R["fidx"][b]).size),
                          "rows_removed": int((a & ~b).sum()),
                          "discharges_removed": int(np.unique(R["fidx"][a]).size - np.unique(R["fidx"][b]).size)}
    um, ur = unit_counts(R, s_main, us), unit_counts(R, s_ref, us)
    per_unit = {}
    for u in us:
        o = {}
        for k in ("common_L5", "S_L5", "common_F", "S_F"):
            m_, r_ = um[u["name"]][k], ur[u["name"]][k]
            o[k] = {}
            for part in ("fit", "test"):
                o[k][f"{part}_discharges_main"] = m_[f"{part}_discharges"]
                o[k][f"{part}_rows_main"] = m_[f"{part}_rows"]
                o[k][f"{part}_discharges_removed"] = m_[f"{part}_discharges"] - r_[f"{part}_discharges"]
                o[k][f"{part}_rows_removed"] = m_[f"{part}_rows"] - r_[f"{part}_rows"]
        per_unit[u["name"]] = o
    beside = {k: {"main": {"value": main_["statements"][k]["value"], "holds": main_["statements"][k]["holds"]},
                  "refit": {"value": rf["statements"][k]["value"],
                            "cluster_interval_rule_met": rf["statements"][k]["holds"]}} for k in main_["statements"]}
    out = dict(main_)
    out["refit_without_13_files"] = {
        "status": "sensitivity, not judged as the registered statements", "order_of_events": ORDER_OF_EVENTS,
        "rule": va["rule"], "invalid_files": va["invalid_files"], "invalid_pairs": va["invalid_pairs"],
        "selects_exactly_the_expected_files": va["selects_exactly_the_expected_files"],
        "estimates": {"main": "data/.b79/fits/", "refit": "data/.b79/refit_valid/fits/"},
        "statements_beside_main": beside, "removed_per_set": per_set, "removed_per_unit": per_unit, "readout": rf}
    M.save_json(M.OUT_JSON, out)
    RR = load_by_path("render79", M.HERE / "render79.py")
    M.save_text(M.OUT_MD, RR.render(main_) + RR.render_refit(out))
    M.log(f"written {M.OUT_JSON.name}, {M.OUT_MD.name}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge()
    else:
        op = None
        if "--out" in sys.argv:
            from pathlib import Path
            op = Path(sys.argv[sys.argv.index("--out") + 1])
        main(op)
