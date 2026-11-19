"""B.73 readout (registration sections 6-7, Addendum 1 sections 7-9): statements Q1-Q10, the design-by-target table of
fast-channel increments, skills over G and H, RMSE and median absolute error, V1 strata by chronological block and BES
level fifth, fitted gains, the usability classifier.

    py ces_prediction/experiments/b73_targets/readout.py
Output: docs/paper/b73_targets.json / .md
"""
from __future__ import annotations

import json
import time

import common as C
import designs as DS

import numpy as np  # noqa: E402

TREES, FILT, CLF = C.CACHE / "trees", C.CACHE / "filters", C.CACHE / "clf"
FORMS = ("discharge", "cluster")
LEARN_FAST = {"TX": "T0", "TB": "T0", "TE": "T0", "TXlag": "T0"}
NET = {"c": "t", "c2": "t2"}
DESIGN_LABEL = {"V1": "V1 random (r2-r4)", "V2": "V2 cluster-blocked (seeds 1-3)", "V3": "V3 rolling origin (fifths 2-5)",
                "V4a": "V4 chronological, fit to 31991", "V4b": "V4 chronological, fit to 32310"}
DESIGNS = ("V1", "V2", "V3", "V4a", "V4b")


# ------------------------------------------------------------------------------------------ assembly
def assemble_design(R, us, d):
    """-> {"subs": [{"name", "est": {name: full-length array}}], "mask": scored rows (common to the subs)}."""
    n = len(R["y"])
    if d in ("V1", "V2"):
        subs = C.REPEATS if d == "V1" else C.V2_SEEDS
        groups = {s: [u for u in us if u["design"] == d and u["sub"] == s] for s in subs}
    elif d == "V3":
        groups = {"pooled": [u for u in us if u["design"] == "V3"]}
    else:
        groups = {d: [u for u in us if u["name"] == d]}
    spec = DS.TREE_SPEC["V4" if d.startswith("V4") else d]
    out, mask = [], None
    for s, ul in groups.items():
        est = {}
        m = np.zeros(n, bool)
        for L, tg in spec.items():
            for X in tg:
                a = np.full(n, np.nan)
                for u in ul:
                    z = C.load_npz(TREES / f"{u['name']}__{L}__{X}.npz")
                    a[z["idx"]] = z["est"]
                    m[z["idx"]] = True
                est[f"{L}[{X}]"] = a
        for X in DS.FILTER_TARGETS:
            a0, ax = np.full(n, np.nan), np.full(n, np.nan)
            for u in ul:
                z = C.load_npz(FILT / f"{u['name']}__{X}.npz")
                a0[z["idx"]], ax[z["idx"]] = z["k0"], z["kx"]
            est[f"K0[{X}]"], est[f"KX[{X}]"] = a0, ax
        est["G"], est["H"], est["M5"] = R["gp"], R["hold"], R["m5"]
        if d == "V1":
            r = s
            for e in ("c", "t", "c2", "t2"):
                est[e] = R[f"r{r}_{e}"]
        out.append({"name": str(s), "est": est})
        mask = m if mask is None else mask
        if not np.array_equal(mask, m):
            raise SystemExit(f"FATAL: {d}: the subs score different rows")
    return {"subs": out, "mask": mask}


def gain_fit(R, D):
    """lambda per re-partition, fold, target and pair (Addendum 1 section 6); adds cL[X], c2L[X] to the V1 subs."""
    lam = {}
    for sub in D["subs"]:
        r = int(sub["name"])
        kk = R[f"r{r}_k"]
        for X in C.TARGETS:
            tg = R[f"tg_{X}"]
            for a, b in NET.items():
                est = np.full(len(tg), np.nan)
                corr = R[f"r{r}_{a}"] - R[f"r{r}_{b}"]
                for k in range(C.K):
                    oth = (kk != k) & np.isfinite(tg)
                    e = tg[oth] - R[f"r{r}_{b}"][oth]
                    dd = corr[oth]
                    lm = float(np.sum(dd * e) / np.sum(dd * dd))
                    lam[f"r{r}|k{k}|{X}|{a}"] = lm
                    te = kk == k
                    est[te] = R[f"r{r}_{b}"][te] + lm * corr[te]
                sub["est"][f"{a}L[{X}]"] = est
    return lam


# ------------------------------------------------------------------------------------------ bootstrap cells
class Cell:
    """One design x row set x form: the units with rows in the set, one set of draws (a sub drawn, then its units
    resampled with replacement); every quantity of the cell on the same draws."""

    def __init__(self, D, R, design, rowset, rows, form, unit_d, unit_c):
        self.D, self.rows, self.R = D, rows, R
        uu = (unit_d if form == "discharge" else unit_c)[rows]
        u, inv = np.unique(uu, return_inverse=True)
        self.inv, self.nu = inv, len(u)
        rng = C.rng_for(f"{design}|{rowset}|{form}")
        ns = len(D["subs"])
        pick = rng.integers(0, ns, C.N_BOOT) if ns > 1 else np.zeros(C.N_BOOT, np.int64)
        self.W = []
        for si in range(ns):
            sel = np.flatnonzero(pick == si)
            self.W.append((sel, rng.multinomial(self.nu, np.full(self.nu, 1.0 / self.nu), size=len(sel)).astype(np.float64)))
        self.cache = {}

    def sums(self, e, X):
        key = (e, X)
        if key not in self.cache:
            y = self.R[f"tg_{X}"][self.rows]
            pts, dr = [], np.empty(C.N_BOOT)
            for si, sub in enumerate(self.D["subs"]):
                se = (sub["est"][e][self.rows] - y) ** 2
                if not np.all(np.isfinite(se)):
                    raise SystemExit(f"FATAL: {e} against {X}: {int((~np.isfinite(se)).sum())} rows without a value")
                S = np.bincount(self.inv, se, self.nu)
                pts.append(S.sum())
                sel, W = self.W[si]
                dr[sel] = W @ S
            self.cache[key] = (np.array(pts), dr)
        return self.cache[key]

    def sums_masked(self, e, X, m, tag):
        """Per-unit sums of the squared errors of the rows of the cell selected by m (bool over the cell's rows)."""
        key = (e, X, tag)
        if key not in self.cache:
            y = self.R[f"tg_{X}"][self.rows]
            pts, dr = [], np.empty(C.N_BOOT)
            for si, sub in enumerate(self.D["subs"]):
                se = (sub["est"][e][self.rows] - y) ** 2 * m
                S = np.bincount(self.inv, se, self.nu)
                pts.append(S.sum())
                sel, W = self.W[si]
                dr[sel] = W @ S
            self.cache[key] = (np.array(pts), dr)
        return self.cache[key]

    def share(self, a, b, X, m, tag):
        """Share of the reduction SSE(b) - SSE(a) that the rows m hold."""
        pa, da = self.sums(a, X)
        pb, db = self.sums(b, X)
        qa, ea = self.sums_masked(a, X, m, tag)
        qb, eb = self.sums_masked(b, X, m, tag)
        return float(np.mean((qb - qa) / (pb - pa))), (eb - ea) / (db - da)

    def skill(self, a, b, X):
        pa, da = self.sums(a, X)
        pb, db = self.sums(b, X)
        return float(np.mean(1 - pa / pb)), 1 - da / db


def q(draws):
    d = draws[np.isfinite(draws)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def pack(pt, dd, dc):
    return {"point": pt, "discharge": q(dd), "cluster": q(dc)}


# ------------------------------------------------------------------------------------------ AUC
def auc_matrix(score, label, unit, nu):
    """A[a, b] = concordant (positive of unit a, negative of unit b) pairs, ties one half; P, N per unit."""
    pos, neg = label == 1, label == 0
    sp, up = score[pos], unit[pos]
    A = np.zeros((nu, nu))
    for b in range(nu):
        nb = np.sort(score[neg & (unit == b)])
        if len(nb) == 0:
            continue
        lo = np.searchsorted(nb, sp, "left")
        hi = np.searchsorted(nb, sp, "right")
        A[:, b] = np.bincount(up, lo + 0.5 * (hi - lo), nu)
    return A, np.bincount(unit[pos], None, nu).astype(float), np.bincount(unit[neg], None, nu).astype(float)


def auc_cell(Q, subs, design, form):
    """subs: list of {"idx": classifier rows, "H": scores, "HX": scores}; same rows in every sub."""
    unit_all = Q["fidx"] if form == "discharge" else Q["clus"]
    idx = subs[0]["idx"]
    u, inv = np.unique(unit_all[idx], return_inverse=True)
    nu = len(u)
    lab = Q["label"][idx].astype(np.int64)
    rng = C.rng_for(f"{design}|usability|{form}")
    ns = len(subs)
    pick = rng.integers(0, ns, C.N_BOOT) if ns > 1 else np.zeros(C.N_BOOT, np.int64)
    out = {m: np.empty(C.N_BOOT) for m in ("H", "HX")}
    pts = {m: [] for m in ("H", "HX")}
    for si, s in enumerate(subs):
        sel = np.flatnonzero(pick == si)
        W = rng.multinomial(nu, np.full(nu, 1.0 / nu), size=len(sel)).astype(np.float64)
        for m in ("H", "HX"):
            A, P, N = auc_matrix(s[m], lab, inv, nu)
            pts[m].append(float(A.sum() / (P.sum() * N.sum())))
            out[m][sel] = ((W @ A) * W).sum(1) / ((W @ P) * (W @ N))
    res = {m: {"point": float(np.mean(pts[m])), form: q(out[m]), "per_sub": pts[m]} for m in ("H", "HX")}
    res["diff"] = {"point": float(np.mean(np.array(pts["HX"]) - np.array(pts["H"]))), form: q(out["HX"] - out["H"])}
    return res


def classifier(Q, us):
    res = {}
    for d in ("V1", "V4a", "V4b"):
        if d == "V1":
            subs = []
            for r in C.REPEATS:
                parts = [{m: C.load_npz(CLF / f"V1_r{r}_k{k}__{m}.npz") for m in ("H", "HX")} for k in range(C.K)]
                idx = np.concatenate([p["H"]["idx"] for p in parts])
                o = np.argsort(idx)
                s = {"idx": idx[o]}
                for m in ("H", "HX"):
                    s[m] = np.concatenate([p[m]["p"] for p in parts])[o]
                    if not np.array_equal(np.concatenate([p[m]["idx"] for p in parts])[o], s["idx"]):
                        raise SystemExit("FATAL: classifier rows differ")
                subs.append(s)
            for s in subs[1:]:
                if not np.array_equal(s["idx"], subs[0]["idx"]):
                    raise SystemExit("FATAL: classifier rows differ between re-partitions")
        else:
            z = {m: C.load_npz(CLF / f"{d}__{m}.npz") for m in ("H", "HX")}
            subs = [{"idx": z["H"]["idx"], "H": z["H"]["p"], "HX": z["HX"]["p"]}]
        r = {"rows": int(len(subs[0]["idx"])), "positives": int(Q["label"][subs[0]["idx"]].sum())}
        for form in FORMS:
            c = auc_cell(Q, subs, d, form)
            for k_ in ("H", "HX", "diff"):
                r.setdefault(k_, {"point": c[k_]["point"]})[form] = c[k_][form]
                if "per_sub" in c[k_]:
                    r[k_]["per_sub"] = c[k_]["per_sub"]
        res[d] = r
        C.log(f"classifier {d}: AUC H {r['H']['point']:.4f}, HX {r['HX']['point']:.4f}, diff {r['diff']['point']:+.4f} "
              f"c{r['diff']['cluster']}")
    return res


# ------------------------------------------------------------------------------------------ main readout
def estimators_for(D, d, X):
    names = [k for k in D["subs"][0]["est"] if k.endswith(f"[{X}]")]
    base = ["G", "H", "M5"] + (["c", "t", "c2", "t2"] if d == "V1" else [])
    return names + base


def increments_for(D, d, X):
    """(label, A, B) pairs of the fast-channel increment for target X in design d."""
    est = D["subs"][0]["est"]
    out = []
    for L, B in LEARN_FAST.items():
        if f"{L}[{X}]" in est:
            out.append((f"{L}/{B}", f"{L}[{X}]", f"{B}[{X}]"))
    if f"KX[{X}]" in est:
        out.append(("KX/K0", f"KX[{X}]", f"K0[{X}]"))
    if d == "V1":
        out += [("c/t as built", "c", "t"), ("c2/t2 as built", "c2", "t2"),
                ("c/t gain-fitted", f"cL[{X}]", "t"), ("c2/t2 gain-fitted", f"c2L[{X}]", "t2")]
    return out


def main():
    t_start = time.time()
    R = C.rows()
    Q = C.clf_rows()
    us, tab, fifth = DS.units(R)
    _, unit_d = np.unique(R["fidx"], return_inverse=True)
    _, unit_c = np.unique(R["clus"], return_inverse=True)
    n = len(R["y"])
    Ds = {d: assemble_design(R, us, d) for d in DESIGNS}
    lam = gain_fit(R, Ds["V1"])
    C.log("assembled")
    res = {"registration": C.REG, "addendum": "ces_prediction/experiments/PREREGISTRATION_B73_ADDENDUM1.md (session lead) and _ADDENDUM2.md (implementation)",
           "n_boot": C.N_BOOT, "seed": C.SEED, "rows_info": json.loads((C.CACHE / "rows_info.json").read_text(encoding="utf-8")),
           "designs": DS.describe(), "increments": {}, "per_estimator": {}, "cross_target_L5_rows": {}, "strata": {},
           "extra": {}}
    # share of rows without a complete input set (the same in every unit)
    z = C.load_npz(TREES / "V1_r2_k0__T0__F.npz")
    res["rows_info"]["incomplete_share_V1_r2_k0_test"] = float(1 - z["complete"].mean())
    allc = []
    for k in range(C.K):
        allc.append(C.load_npz(TREES / f"V1_r2_k{k}__T0__F.npz")["complete"])
    res["rows_info"]["incomplete_share_all_d1_rows"] = float(1 - np.concatenate(allc).mean())
    cells = {}

    def cell(d, rowset, rows, form):
        key = (d, rowset, form)
        if key not in cells:
            cells[key] = Cell(Ds[d], R, d, rowset, rows, form, unit_d, unit_c)
        return cells[key]

    def compare(d, rowset, rows, a, b, X):
        pts, dd = cell(d, rowset, rows, "discharge").skill(a, b, X)
        _, dc = cell(d, rowset, rows, "cluster").skill(a, b, X)
        return pack(pts, dd, dc), (dd, dc)

    targets_of = {d: (C.TARGETS if d == "V1" else ("F", "L5")) for d in DESIGNS}
    # ---- increments and per-estimator skills
    for d in DESIGNS:
        D = Ds[d]
        for X in targets_of[d]:
            rows = np.flatnonzero(D["mask"] & np.isfinite(R[f"tg_{X}"]))
            inc = {}
            for lab, a, b in increments_for(D, d, X):
                inc[lab], _ = compare(d, X, rows, a, b, X)
                inc[lab]["reduction_pct"] = 100 * inc[lab]["point"]
            res["increments"][f"{d}|{X}"] = {"rows": int(len(rows)), "discharges": int(len(np.unique(unit_d[rows]))),
                                              "clusters": int(len(np.unique(unit_c[rows]))), "cells": inc}
            pe = {}
            y = R[f"tg_{X}"][rows]
            for e in estimators_for(D, d, X):
                o = {}
                if e != "G":
                    o["over_G"], _ = compare(d, X, rows, e, "G", X)
                if e != "H":
                    o["over_H"], _ = compare(d, X, rows, e, "H", X)
                ae = [np.abs(s["est"][e][rows] - y) for s in D["subs"]]
                o["rmse_eV"] = float(np.mean([np.sqrt(np.mean(a_ ** 2)) for a_ in ae]))
                o["median_abs_eV"] = float(np.mean([np.median(a_) for a_ in ae]))
                pe[e] = o
            if d == "V1":
                pe["TX vs c2 gain-fitted"], _ = compare(d, X, rows, f"TX[{X}]", f"c2L[{X}]", X)
                pe["TX vs c2 as built"], _ = compare(d, X, rows, f"TX[{X}]", "c2", X)
                pe["T0 vs t2 as built"], _ = compare(d, X, rows, f"T0[{X}]", "t2", X)
            res["per_estimator"][f"{d}|{X}"] = pe
            if X == "L5":
                res["cross_target_L5_rows"][d] = cross_target(d, D, rows, cell)
            C.log(f"{d} {X}: rows {len(rows):,}; TX/T0 {inc['TX/T0']['point']:+.4f} c{inc['TX/T0']['cluster']} "
                  f"KX/K0 {inc['KX/K0']['point'] if 'KX/K0' in inc else float('nan'):+.4f}")
            cells.clear()
        # V3 per fifth (points)
        if d == "V3":
            per = {}
            for u in [x for x in us if x["design"] == "V3"]:
                tr = np.isin(R["fidx"], u["test"])
                for X in ("F", "L5"):
                    m = tr & np.isfinite(R[f"tg_{X}"])
                    y = R[f"tg_{X}"][m]
                    se = {e: (D["subs"][0]["est"][e][m] - y) ** 2 for e in (f"TX[{X}]", f"T0[{X}]", f"KX[{X}]", f"K0[{X}]", "G")}
                    per[f"{u['name']}|{X}"] = {"TX/T0": C.skill(se[f"TX[{X}]"], se[f"T0[{X}]"]),
                                                "KX/K0": C.skill(se[f"KX[{X}]"], se[f"K0[{X}]"]),
                                                "T0/G": C.skill(se[f"T0[{X}]"], se["G"]), "rows": int(m.sum())}
            res["extra"]["V3_per_fifth"] = per
    # ---- V1 strata: chronological block and BES level fifth
    D = Ds["V1"]
    qs = np.quantile(R["bes_level"], [0.2, 0.4, 0.6, 0.8])
    bes5 = np.searchsorted(qs, R["bes_level"], "right")
    strata = {f"block {C.BLOCKS[b][0]}": R["blk"] == b for b in range(3)}
    strata.update({f"BES level fifth {j + 1}": bes5 == j for j in range(5)})
    res["extra"]["bes_level_quintile_edges"] = qs.tolist()
    for sname, sm in strata.items():
        for X in C.TARGETS:
            rows = np.flatnonzero(D["mask"] & sm & np.isfinite(R[f"tg_{X}"]))
            inc = {}
            for lab, a, b in increments_for(D, "V1", X):
                inc[lab], _ = compare("V1", f"{X}|{sname}", rows, a, b, X)
            inc["T0/G"], _ = compare("V1", f"{X}|{sname}", rows, f"T0[{X}]", "G", X)
            res["strata"][f"{sname}|{X}"] = {"rows": int(len(rows)), "discharges": int(len(np.unique(unit_d[rows]))),
                                             "clusters": int(len(np.unique(unit_c[rows]))), "cells": inc}
            cells.clear()
        C.log(f"stratum {sname} done")
    # ---- concentration of the V4 increments over discharges (points; descriptive)
    conc = {}
    for d in ("V4a", "V4b"):
        for X in ("F", "L5"):
            D = Ds[d]
            rows = np.flatnonzero(D["mask"] & np.isfinite(R[f"tg_{X}"]))
            y = R[f"tg_{X}"][rows]
            red = (D["subs"][0]["est"][f"T0[{X}]"][rows] - y) ** 2 - (D["subs"][0]["est"][f"TX[{X}]"][rows] - y) ** 2
            u, inv = np.unique(R["shot"][rows], return_inverse=True)
            sr = np.bincount(inv, red)
            o_ = np.argsort(-sr)
            conc[f"{d}|{X}"] = {"discharges": int(len(u)), "share_top5": float(sr[o_[:5]].sum() / sr.sum()),
                                "top5_shots": u[o_[:5]].tolist(), "discharges_with_negative_reduction": int((sr < 0).sum())}
    res["extra"]["V4_concentration_TX_over_T0"] = conc
    # ---- the session lead's Addendum 1: strata on the inputs (A1, A2), A3, A4
    lead_addendum(res, Ds, R, unit_d, unit_c, cell, compare, cells)
    # ---- gains
    lam_s = {}
    for X in C.TARGETS:
        for a in NET:
            v = [lam[f"r{r}|k{k}|{X}|{a}"] for r in C.REPEATS for k in range(C.K)]
            lam_s[f"{a}|{X}"] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
    res["lambda"] = {"per_fold": lam, "summary": lam_s}
    fg = {}
    for u in us:
        for X in DS.FILTER_TARGETS:
            p = json.loads((FILT / f"{u['name']}__{X}.json").read_text(encoding="utf-8"))
            fg[f"{u['name']}|{X}"] = {"theta_k0": p["theta_k0"], "b_c": p["b_kx"][0], "b_g": p["b_kx"][1],
                                      "fit_mse_k0": p["fit_mse_k0"], "fit_mse_kx": p["fit_mse_kx"],
                                      "test_rows_without_output": p["test_rows_without_output"]}
    fsum = {}
    for d in ("V1", "V2", "V3", "V4"):
        for X in DS.FILTER_TARGETS:
            ks = [k for k in fg if k.startswith(d) and k.endswith(f"|{X}")]
            bc, bg = [fg[k]["b_c"] for k in ks], [fg[k]["b_g"] for k in ks]
            fsum[f"{d}|{X}"] = {"units": len(ks), "b_c": [float(np.mean(bc)), float(np.min(bc)), float(np.max(bc))],
                                "b_g": [float(np.mean(bg)), float(np.min(bg)), float(np.max(bg))]}
    res["filter_gains"] = {"per_unit": fg, "summary": fsum}
    # ---- classifier
    res["usability"] = classifier(Q, us)
    lk = C.CACHE / "label_kinds.json"
    res["usability"]["label_kinds"] = json.loads(lk.read_text(encoding="utf-8")) if lk.exists() else None
    ex = {}
    for d, un in (("V1", "V1_r2_k0"), ("V4a", "V4a")):
        ex[d] = int(C.load_npz(CLF / f"{un}__H.npz")["excluded_incomplete"])
    res["usability"]["excluded_rows_without_fast_scalars_example_units"] = ex
    # ---- statements
    res["statements"] = statements(res)
    res["checks"] = json.loads((C.CACHE / "checks_repro.json").read_text(encoding="utf-8"))
    lkp = C.CACHE / "checks_leakage.json"
    res["checks_leakage"] = json.loads(lkp.read_text(encoding="utf-8")) if lkp.exists() else None
    res["runtime_readout_s"] = round(time.time() - t_start, 1)
    C.save_json(C.OUT_JSON, res)
    import render
    C.save_text(C.OUT_MD, render.render(res))
    C.log(f"written {C.OUT_JSON.name}, {C.OUT_MD.name}")


def lead_addendum(res, Ds, R, unit_d, unit_c, cell, compare, cells):
    import strata_inputs as SI
    S = C.load_npz(C.CACHE / "strata_inputs.npz")
    out = {"A1": {}, "A2": {}}
    v1_pairs = [("TX/T0", "TX[{X}]", "T0[{X}]"), ("KX/K0", "KX[{X}]", "K0[{X}]"), ("c2/t2 as built", "c2", "t2"),
                ("c/t as built", "c", "t"), ("c2/t2 gain-fitted", "c2L[{X}]", "t2"), ("c/t gain-fitted", "cL[{X}]", "t")]
    for scheme, labels, codes, designs in (("A1", SI.RB_LAB, S["rb_class"], ("V1", "V4a", "V4b")),
                                           ("A2", SI.AB_LAB, S["absent_class"], ("V1",))):
        for d in designs:
            D = Ds[d]
            pairs = v1_pairs if d == "V1" else v1_pairs[:1]
            rowsF = np.flatnonzero(D["mask"] & np.isfinite(R["tg_F"]))
            od = {}
            for k, lab in enumerate(labels):
                o = {"share_of_rows": float(np.mean(codes[rowsF] == k)), "rows_F": int(np.sum(codes[rowsF] == k)),
                     "rows_L5": int(np.sum(codes[D["mask"] & np.isfinite(R["tg_L5"])] == k))}
                for lab2, a_, b_ in pairs:
                    e = {}
                    for X in ("F", "L5"):
                        rows = np.flatnonzero(D["mask"] & np.isfinite(R[f"tg_{X}"]) & (codes == k))
                        if len(rows) == 0:
                            continue
                        e[f"against_{X}"], _ = compare(d, f"{X}|{scheme}={k}", rows, a_.format(X=X), b_.format(X=X), X)
                    shp, shd = cell(d, "F", rowsF, "discharge").share(a_.format(X="F"), b_.format(X="F"), "F",
                                                                       codes[rowsF] == k, f"{scheme}{k}")
                    _, shc = cell(d, "F", rowsF, "cluster").share(a_.format(X="F"), b_.format(X="F"), "F",
                                                                  codes[rowsF] == k, f"{scheme}{k}")
                    e["share_of_F_reduction"] = pack(shp, shd, shc)
                    o[lab2] = e
                od[lab] = o
                for key in [kk for kk in cells if kk[1] != "F"]:
                    del cells[key]
            cells.clear()
            out[scheme][d] = od
            C.log(f"{scheme} {d} done")
    a34 = json.loads((C.CACHE / "a3_a4.json").read_text(encoding="utf-8"))
    out["A3"], out["A4"] = a34["A3"], a34["A4"]
    out["A1_class_share_of_d1_rows"] = a34["rb_class_share_d1"]
    out["A2_class_share_of_d1_rows"] = a34["absent_class_share_d1"]
    # A4 sensitivity: T0 and TX refitted without the rows exposed without beam emission
    nb = C.CACHE / "trees_nobeam"
    if nb.exists():
        D = Ds["V1"]
        bo = S["beamoff"].astype(bool)
        for sub in D["subs"]:
            r = int(sub["name"])
            for L in ("T0", "TX"):
                for X in ("F", "L5"):
                    a = np.full(len(R["y"]), np.nan)
                    for k in range(C.K):
                        z = C.load_npz(nb / f"V1_r{r}_k{k}__{L}__{X}.npz")
                        a[z["idx"]] = z["est"]
                    sub["est"][f"{L}nb[{X}]"] = a
        sens = {}
        for X in ("F", "L5"):
            rows = np.flatnonzero(D["mask"] & np.isfinite(R[f"tg_{X}"]) & ~bo)
            sens[X] = {"rows": int(len(rows)), "rows_removed": int(np.sum(D["mask"] & np.isfinite(R[f"tg_{X}"]) & bo))}
            sens[X]["refitted without them"], _ = compare("V1", f"{X}|A4", rows, f"TXnb[{X}]", f"T0nb[{X}]", X)
            sens[X]["fits as in the main table"], _ = compare("V1", f"{X}|A4", rows, f"TX[{X}]", f"T0[{X}]", X)
            cells.clear()
        out["A4_sensitivity"] = sens
    res["lead_addendum"] = out


def cross_target(d, D, rows, cell):
    """On the L5 rows: each learner fitted to F scored against F beside fitted to L5 scored against L5 (Q1, Q9 and the
    retention of every learner), on the draws of the L5 cell."""
    pairs = [("trees TX/T0", "TX[F]", "T0[F]", "TX[L5]", "T0[L5]"), ("filters KX/K0", "KX[F]", "K0[F]", "KX[L5]", "K0[L5]")]
    if d == "V1":
        pairs += [("network c/t as built", "c", "t", "c", "t"), ("two-stage c2/t2 as built", "c2", "t2", "c2", "t2"),
                  ("network c/t gain-fitted", "cL[F]", "t", "cL[L5]", "t"),
                  ("two-stage c2/t2 gain-fitted", "c2L[F]", "t2", "c2L[L5]", "t2")]
    if d == "V1" or d.startswith("V4"):
        pairs += [("TE/T0", "TE[F]", "T0[F]", "TE[L5]", "T0[L5]"), ("TB/T0", "TB[F]", "T0[F]", "TB[L5]", "T0[L5]"),
                  ("TXlag/T0", "TXlag[F]", "T0[F]", "TXlag[L5]", "T0[L5]")]
    o = {}
    for lab, aF, bF, aL, bL in pairs:
        drF, drL = {}, {}
        for form in FORMS:
            c_ = cell(d, "L5", rows, form)
            pF, drF[form] = c_.skill(aF, bF, "F")
            pL, drL[form] = c_.skill(aL, bL, "L5")
        o[lab] = {"against_F": pack(pF, drF["discharge"], drF["cluster"]),
                  "against_L5": pack(pL, drL["discharge"], drL["cluster"]),
                  "ratio_L5_over_F": pack(pL / pF, drL["discharge"] / drF["discharge"], drL["cluster"] / drF["cluster"]),
                  "difference_F_minus_L5": pack(pF - pL, drF["discharge"] - drL["discharge"], drF["cluster"] - drL["cluster"])}
    return {"rows": int(len(rows)), "cells": o}


def statements(res):
    I, X = res["increments"], res["cross_target_L5_rows"]
    P = res["per_estimator"]
    S = {}

    def above(v):
        return v["cluster"][0] > 0

    def ivs(v):
        return f"{v['point']:+.3f}; c [{v['cluster'][0]:+.3f}, {v['cluster'][1]:+.3f}]"

    def put(k, text, v, holds, exp, note=""):
        S[k] = {"text": text, "value": v, "holds": bool(holds), "expected": exp, "note": note}

    v = X["V1"]["cells"]["two-stage c2/t2 as built"]["ratio_L5_over_F"]
    put("Q1", "V1, as built: increment of the two-stage estimator over its twin against L5 is less than half of that against F "
              "(interval of the ratio of the two MSE reductions below 0.5)", v, v["cluster"][1] < 0.5, "holds (0.1 to 0.4)")
    v = I["V1|L5"]["cells"]["TX/T0"]
    put("Q2", "V1, target L5, fitted to L5: TX over T0 above zero", v, above(v), "holds (+0.03 to +0.07)")
    v = I["V2|L5"]["cells"]["TX/T0"]
    put("Q3", "V2, target L5: TX over T0 above zero", v, above(v), "holds (+0.02 to +0.06)")
    v = I["V3|L5"]["cells"]["TX/T0"]
    put("Q4", "V3, target L5: TX over T0 above zero", v, above(v), "uncertain (+0.01 to +0.04)")
    v = I["V4a|L5"]["cells"]["TX/T0"]
    put("Q5", "V4 (fit to 31991, score from 32312), target L5: TX over T0 above zero", v, above(v), "uncertain (0.00 to +0.03)",
        f"V4b (fit to 32310): {ivs(I['V4b|L5']['cells']['TX/T0'])}")
    v = I["V1|L5"]["cells"]["TE/T0"]
    put("Q6", "V1, target L5: TE over T0 above zero (needs neither the beam nor the spectrometer)", v, above(v),
        "uncertain (0.00 to +0.03)")
    v = P["V1|L5"]["TX vs c2 gain-fitted"]
    put("Q7", "V1, target L5: TX[L5] not worse than the gain-fitted two-stage estimator (interval of its skill over it not "
              "entirely below zero)", v, v["cluster"][1] >= 0, "holds",
        f"interval above zero: {v['cluster'][0] > 0}")
    v = P["V4a|L5"]["T0[L5]"]["over_G"]
    put("Q8", "V4 (fit to 31991), target L5: T0[L5] over G above zero", v, above(v), "holds",
        f"V4b: {ivs(P['V4b|L5']['T0[L5]']['over_G'])}")
    v1 = X["V1"]["cells"]["trees TX/T0"]["difference_F_minus_L5"]
    v4 = X["V4a"]["cells"]["trees TX/T0"]["difference_F_minus_L5"]
    put("Q9-V1", "V1: TX over T0 against F (fitted to F) exceeds TX over T0 against L5 (fitted to L5), on the L5 rows", v1,
        above(v1), "holds")
    put("Q9-V4", "V4 (fit to 31991): the same", v4, above(v4), "uncertain",
        f"V4b: {ivs(X['V4b']['cells']['trees TX/T0']['difference_F_minus_L5'])}")
    u = res["usability"]["V4a"]["diff"]
    put("Q10", "V4 (fit to 31991): AUC with the fast scalars minus without above zero", u, u["cluster"][0] > 0,
        "holds (+0.01 to +0.05)", f"V4b: {ivs(res['usability']['V4b']['diff'])}")
    A1 = res["lead_addendum"]["A1"]["V1"]
    hi, lo = A1["above 1.03"]["TX/T0"], A1["below 0.9"]["TX/T0"]
    put("Q11", "(Addendum 1) V1, class r_B above 1.03: TX over T0 above zero against F and not above zero against L5",
        {"against_F": hi["against_F"], "against_L5": hi["against_L5"]},
        hi["against_F"]["cluster"][0] > 0 and not hi["against_L5"]["cluster"][0] > 0, "holds")
    put("Q12", "(Addendum 1) V1, class r_B below 0.9: TX over T0 against L5 above zero", lo["against_L5"],
        lo["against_L5"]["cluster"][0] > 0, "holds (+0.05 to +0.15)")
    sl = res["lead_addendum"]["A3"]["slopes"]["C18 form: r_B 0.6-1.6, log ratios clipped at 1.5"]
    put("Q13", "(Addendum 1) log-log slope of T_i on r_B above 0.5, and the interval of the slope of fresh V_rot within +-0.1 "
               "(discharges resampled)", {"T_i": sl["T_i"], "V_rot": sl["V_rot"]},
        sl["T_i"]["discharge"][0] > 0.5 and -0.1 <= sl["V_rot"]["discharge"][0] and sl["V_rot"]["discharge"][1] <= 0.1, "holds",
        f"rows {sl['rows']:,}, discharges {sl['discharges']}")
    return S


if __name__ == "__main__":
    main()
