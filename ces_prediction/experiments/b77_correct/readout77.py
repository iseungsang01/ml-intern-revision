"""B.77 readout (registration sections 5-7, Addendum 1 sections 4-10): every table and statement into the JSON in one
pass, then the markdown.  The log prints row counts and times only.

    py ces_prediction/experiments/b77_correct/readout77.py
Output: docs/paper/b77_correct.json / .md
"""
from __future__ import annotations

import importlib.util
import json
import time
import warnings

import common77 as M

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
K, C = M.K, M.C
N = M.N_BOOT
FORMS = ("discharge", "cluster")
DESIGNS = ("V1", "V2", "V3", "V4a", "V4b")
EST = ("O", "MED3", "G", "H", "T0", "TX", "C0", "CX", "CB", "CE")
FITTED = ("T0", "TX", "C0", "CX", "CB", "CE")
RB_LAB = ("below 0.9", "0.9-0.97", "0.97-1.03", "above 1.03", "r_B undefined")
AUC_DIFFS = (("PHX", "PH"), ("FHX", "FH"), ("FH", "PH"), ("FHX", "PHX"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------ helpers
def q(d):
    d = d[np.isfinite(d)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def pack(pt, dd, dc):
    return {"point": float(pt), "discharge": q(dd), "cluster": q(dc)}


def draws(name, nsub, nu):
    """B.75's scheme with seed 20261177: a sub drawn first, then its units resampled with replacement."""
    rng = M.rng_for(name)
    pick = rng.integers(0, nsub, N) if nsub > 1 else np.zeros(N, np.int64)
    out = []
    for si in range(nsub):
        sel = np.flatnonzero(pick == si)
        out.append((sel, rng.multinomial(nu, np.full(nu, 1.0 / nu), size=len(sel)).astype(np.float64)))
    return out


class Group:
    """One design and row set with strata.  Per stratum and form: per-unit sums of every per-row quantity in every sub,
    then one set of draws shared by every quantity of that stratum and form (stream "design|stratum|form")."""

    def __init__(self, name, rows, strata, unit_d, unit_c, nsub):
        self.name, self.rows, self.nsub, self.strata = name, rows, nsub, strata
        self.idx = {}
        for s, m in strata.items():
            for form in FORMS:
                uu = (unit_d if form == "discharge" else unit_c)[rows[m]]
                u, inv = np.unique(uu, return_inverse=True)
                self.idx[(s, form)] = (inv, len(u))
        self.sums = {k: {} for k in self.idx}
        self.tot = {}

    def add(self, e, si, v, only=None):
        if not np.all(np.isfinite(v)):
            raise SystemExit(f"FATAL: {self.name}: {e} has {int((~np.isfinite(v)).sum())} rows without a value")
        for (s, form), (inv, nu) in self.idx.items():
            if only is not None and s not in only:
                continue
            self.sums[(s, form)].setdefault(e, [None] * self.nsub)[si] = np.bincount(inv, v[self.strata[s]], nu)

    def add_all(self, e, v, only=None):
        for si in range(self.nsub):
            self.add(e, si, v, only)

    def finish(self, s):
        for form in FORMS:
            inv, nu = self.idx[(s, form)]
            W = draws(f"{self.name}|{s}|{form}", self.nsub, nu)
            t = {}
            for e, vecs in self.sums[(s, form)].items():
                if any(v is None for v in vecs):
                    raise SystemExit(f"FATAL: {self.name}: {e} missing in a sub")
                pts = np.array([v.sum() for v in vecs])
                dr = np.empty(N)
                for si, (sel, Wm) in enumerate(W):
                    dr[sel] = Wm @ vecs[si]
                t[e] = (pts, dr)
            self.tot[(s, form)] = t
            del W

    def n(self, s):
        return int(self.strata[s].sum())

    def counts(self, s):
        return {"rows": self.n(s), "discharges": self.idx[(s, "discharge")][1], "clusters": self.idx[(s, "cluster")][1]}

    def rmse(self, s, e):
        pts, _ = self.tot[(s, "discharge")][e]
        return float(np.mean(np.sqrt(pts / self.n(s))))

    def f(self, s, fn, *es):
        vals = []
        pt = None
        for form in FORMS:
            t = self.tot[(s, form)]
            vals.append(fn(*[t[e][1] for e in es]))
            if form == "discharge":
                pt = float(np.mean(fn(*[t[e][0] for e in es])))
        return pack(pt, vals[0], vals[1])

    def skill(self, s, a, b):
        return self.f(s, lambda A, B: 1 - A / B, a, b)

    def skill_diff(self, s, a1, b1, a2, b2):
        return self.f(s, lambda A1, B1, A2, B2: (1 - A1 / B1) - (1 - A2 / B2), a1, b1, a2, b2)


# ------------------------------------------------------------------------------------------ assembly
def subs_of(us, d):
    if d in ("V1", "V2"):
        subs = C.REPEATS if d == "V1" else C.V2_SEEDS
        return [[u for u in us if u["design"] == d and u["sub"] == s] for s in subs]
    if d == "V3":
        return [[u for u in us if u["design"] == "V3"]]
    return [[u for u in us if u["name"] == d]]


def load_full(units, L, n, key="est"):
    a = np.full(n, np.nan)
    seen = np.zeros(n, bool)
    for u in units:
        z = np.load(M.fit_path(u, L))
        if np.any(seen[z["idx"]]):
            raise SystemExit(f"FATAL: {L}: a row is scored twice in one sub")
        a[z["idx"]] = z[key]
        seen[z["idx"]] = True
    return a


def kept_flags(units, L, n, qv):
    """Addendum 1 section 6: per test unit, the floor(q n + 0.5) rows with the highest score dropped (stable order)."""
    kept = np.zeros(n, bool)
    for u in units:
        z = np.load(M.fit_path(u, L))
        idx, p = z["idx"], z["p"]
        k = int(np.floor(qv * len(idx) + 0.5))
        o = np.argsort(-p, kind="stable")
        keep = np.ones(len(idx), bool)
        keep[o[:k]] = False
        kept[idx] = keep
    return kept


def design_block(g, s, d):
    o = {"counts": g.counts(s), "rmse_eV": {e: g.rmse(s, e) for e in EST}}
    for e in EST[1:]:
        o[f"{e}/O"] = g.skill(s, e, "O")
    o["C0/T0"] = g.skill(s, "C0", "T0")
    o["TX/T0"] = g.skill(s, "TX", "T0")
    for e in ("CX", "CB", "CE"):
        o[f"{e}/C0"] = g.skill(s, e, "C0")
    o["CX/C0 minus TX/T0"] = g.skill_diff(s, "CX", "C0", "TX", "T0")
    if d == "V1" and s == "all":
        o["rmse_eV"]["CXp"] = g.rmse(s, "CXp")
        o["CXp/C0 (placebo)"] = g.skill(s, "CXp", "C0")
        o["CX/CXp"] = g.skill(s, "CX", "CXp")
    if s == "all":
        scr = {"no screening": {"mse_eV2": g.f(s, lambda A, B: A / B, "O", "N")}}
        for L in M.CLFS:
            for qv in M.QS:
                tag = f"{L}|q{int(round(100 * qv))}"
                scr[tag] = {"mse_eV2": g.f(s, lambda A, B: A / B, f"scrSE|{tag}", f"scrN|{tag}"),
                            "over no screening": g.f(s, lambda A, B, O_, N_: (A / B) / (O_ / N_), f"scrSE|{tag}", f"scrN|{tag}", "O", "N")}
        for qv in M.QS:
            qq = int(round(100 * qv))
            for a, b in (("FHX", "FH"), ("PHX", "PH"), ("FH", "PH"), ("FHX", "PHX")):
                ta, tb = f"{a}|q{qq}", f"{b}|q{qq}"
                scr[f"{a} over {b}|q{qq}"] = g.f(s, lambda A1, B1, A2, B2: (A1 / B1) / (A2 / B2),
                                                 f"scrSE|{ta}", f"scrN|{ta}", f"scrSE|{tb}", f"scrN|{tb}")
        o["screening"] = scr
    return o


def build_design(R, us, d, inset, base, unit_d, unit_c, rbc):
    n = len(R["y"])
    subs = subs_of(us, d)
    scored = np.isin(R["fidx"], np.concatenate([u["test"] for u in subs[0]]))
    rows = np.flatnonzero(scored & inset)
    for sub in subs[1:]:
        if not np.array_equal(np.flatnonzero(np.isin(R["fidx"], np.concatenate([u["test"] for u in sub])) & inset), rows):
            raise SystemExit(f"FATAL: {d}: the subs score different rows")
    strata = {"all": np.ones(len(rows), bool)}
    if d == "V1":
        for b in range(3):
            strata[f"block {C.BLOCKS[b][0]}"] = R["blk"][rows] == b
        for k in range(5):
            strata[f"rB {RB_LAB[k]}"] = rbc[rows] == k
    g = Group(d, rows, strata, unit_d, unit_c, len(subs))
    y = R["tg_L5"][rows]
    for e in ("O", "MED3", "G", "H"):
        g.add_all(e, (base[e][rows] - y) ** 2)
    g.add_all("N", np.ones(len(rows)), only=("all",))
    for si, units in enumerate(subs):
        for L in FITTED + (("CXp",) if d == "V1" else ()):
            a = load_full(units, L, n)
            g.add(L, si, (a[rows] - y) ** 2, only=None if L != "CXp" else ("all",))
        se_o = (base["O"][rows] - y) ** 2
        for L in M.CLFS:
            for qv in M.QS:
                tag = f"{L}|q{int(round(100 * qv))}"
                kept = kept_flags(units, L, n, qv)[rows]
                g.add(f"scrSE|{tag}", si, se_o * kept, only=("all",))
                g.add(f"scrN|{tag}", si, kept.astype(np.float64), only=("all",))
    out = {}
    for s in strata:
        g.finish(s)
        out[s] = design_block(g, s, d)
        g.tot.clear()
    M.log(f"{d}: rows {len(rows):,} done")
    return out, rows, subs


def lag_section(R, us, inset, unit_d, unit_c):
    n = len(R["y"])
    subs = subs_of(us, "V1")
    have = None
    for units in subs:
        a = load_full(units, "CXlag", n)
        h = np.isfinite(a)
        if have is None:
            have = h
        elif not np.array_equal(have, h):
            raise SystemExit("FATAL: CXlag rows differ between re-partitions")
    rows = np.flatnonzero(have & inset)
    g = Group("V1", rows, {"lag rows": np.ones(len(rows), bool)}, unit_d, unit_c, len(subs))
    y = R["tg_L5"][rows]
    for si, units in enumerate(subs):
        for L in ("C0", "CX", "CXlag"):
            g.add(L, si, (load_full(units, L, n)[rows] - y) ** 2)
    g.finish("lag rows")
    s = "lag rows"
    dropped = int(inset.sum() - len(rows))
    o = {"counts": g.counts(s), "rows_of_set_without_lag_inputs": dropped,
         "rmse_eV": {e: g.rmse(s, e) for e in ("C0", "CX", "CXlag")},
         "CXlag/C0": g.skill(s, "CXlag", "C0"), "CX/C0": g.skill(s, "CX", "C0"), "CXlag/CX": g.skill(s, "CXlag", "CX"),
         "CX/C0 minus CXlag/C0": g.skill_diff(s, "CX", "C0", "CXlag", "C0")}
    M.log(f"V1 lag rows: rows {len(rows):,} done")
    return o


# ------------------------------------------------------------------------------------------ AUC
def auc_design(R, d, rows, subs, lab, unit_d, unit_c, auc_matrix):
    n = len(R["y"])
    scores = []
    for units in subs:
        s = {L: load_full(units, L, n, key="p")[rows] for L in M.CLFS}
        for L in M.CLFS:
            if not np.all(np.isfinite(s[L])):
                raise SystemExit(f"FATAL: {d}: {L} lacks a score on a row")
        scores.append(s)
    lb = lab[rows]
    res = {"rows": int(len(rows)), "positives": int(lb.sum()), "positive_share": float(lb.mean())}
    from sklearn.metrics import roc_auc_score
    res["sklearn_check_first_sub"] = {L: float(roc_auc_score(lb, scores[0][L])) for L in M.CLFS}
    tmp = {}
    for form in FORMS:
        uu = (unit_d if form == "discharge" else unit_c)[rows]
        u, inv = np.unique(uu, return_inverse=True)
        nu = len(u)
        W = draws(f"{d}|flags|{form}", len(subs), nu)
        pts = {L: [] for L in M.CLFS}
        dr = {L: np.empty(N) for L in M.CLFS}
        for si, s in enumerate(scores):
            sel, Wm = W[si]
            for L in M.CLFS:
                A, P, Nn = auc_matrix(s[L], lb, inv, nu)
                pts[L].append(float(A.sum() / (P.sum() * Nn.sum())))
                dr[L][sel] = ((Wm @ A) * Wm).sum(1) / ((Wm @ P) * (Wm @ Nn))
        tmp[form] = (pts, dr)
        del W
    for L in M.CLFS:
        pts = tmp["discharge"][0][L]
        res[L] = {"point": float(np.mean(pts)), "per_sub": pts, "discharge": q(tmp["discharge"][1][L]),
                  "cluster": q(tmp["cluster"][1][L])}
    for a, b in AUC_DIFFS:
        pa, pb = np.array(tmp["discharge"][0][a]), np.array(tmp["discharge"][0][b])
        res[f"{a} minus {b}"] = {"point": float(np.mean(pa - pb)),
                                 "discharge": q(tmp["discharge"][1][a] - tmp["discharge"][1][b]),
                                 "cluster": q(tmp["cluster"][1][a] - tmp["cluster"][1][b])}
    M.log(f"{d}: AUC rows {len(rows):,} done")
    return res


# ------------------------------------------------------------------------------------------ statements
def statements(res):
    D = res["designs"]
    A = res["auc"]
    S = {}

    def put(k, text, v, holds, exp, note=""):
        S[k] = {"text": text, "value": v, "holds": bool(holds), "expected": exp, "note": note}

    def lo(v):
        return v["cluster"][0]

    def ivs(v):
        return f"{v['point']:+.3f}; c [{v['cluster'][0]:+.3f}, {v['cluster'][1]:+.3f}]"

    a = lambda d: D[d]["all"]   # noqa: E731
    v = a("V1")["C0/O"]
    put("A1", "V1: the MSE of C0 lies below that of O (skill above zero)", v, lo(v) > 0, "holds (+0.5 to +0.7)")
    v = a("V1")["C0/T0"]
    put("A2", "V1: the MSE of C0 lies below that of T0 (reading the frame helps)", v, lo(v) > 0, "holds (+0.05 to +0.30)")
    v = a("V1")["CX/C0"]
    put("A3", "V1: CX over C0 lies above zero", v, lo(v) > 0, "holds (+0.03 to +0.15)")
    v = a("V2")["CX/C0"]
    put("A4", "V2: CX over C0 lies above zero", v, lo(v) > 0, "holds")
    v = a("V3")["CX/C0"]
    put("A5", "V3: CX over C0 lies above zero", v, lo(v) > 0, "uncertain")
    v = a("V4a")["CX/C0"]
    put("A6", "V4 (fit to 31991): CX over C0 lies above zero", v, lo(v) > 0, "uncertain",
        f"V4b (fit to 32310): {ivs(a('V4b')['CX/C0'])}")
    v = a("V1")["CX/C0 minus TX/T0"]
    put("A7", "V1: CX over C0 exceeds TX over T0 (interval of the difference above zero)", v, lo(v) > 0,
        "uncertain, leaning to holds", f"CX/C0 {ivs(a('V1')['CX/C0'])}; TX/T0 {ivs(a('V1')['TX/T0'])}")
    v1, v4 = A["V1"]["FHX minus FH"], A["V4a"]["FHX minus FH"]
    put("A8", "V1 and V4: the area under the curve of FHX minus that of FH lies above zero under each",
        {"V1": v1, "V4a": v4}, lo(v1) > 0 and lo(v4) > 0, "holds", f"V4b: {ivs(A['V4b']['FHX minus FH'])}")
    v = a("V1")["screening"]["FHX over FH|q10"]
    put("A9", "V1, q = 10 %: the MSE of O on the rows kept by FHX lies below that on the rows kept by FH (interval of the "
              "ratio below 1)", v, v["cluster"][1] < 1, "uncertain",
        f"MSE of O (eV^2): kept by FHX {a('V1')['screening']['FHX|q10']['mse_eV2']['point']:.0f}, by FH "
        f"{a('V1')['screening']['FH|q10']['mse_eV2']['point']:.0f}, no screening {a('V1')['screening']['no screening']['mse_eV2']['point']:.0f}")
    v = a("V1")["CE/C0"]
    put("A10", "V1: CE over C0 lies above zero", v, lo(v) > 0, "uncertain")
    v = a("V1")["CXp/C0 (placebo)"]
    put("A11", "V1, placebo: the interval of CX over C0 contains zero or lies below it", v, lo(v) <= 0, "holds")
    return S


def expectation_met(st):
    out = {}
    for k, s in st.items():
        e = s["expected"]
        if e.startswith("holds"):
            met = s["holds"]
        elif e.startswith("does not hold"):
            met = not s["holds"]
        else:
            met = None
        rng = None
        if "(" in e and " to " in e:
            a, b = (float(x) for x in e[e.index("(") + 1:e.index(")")].split(" to "))
            v = s["value"]
            pts = [v["point"]] if "point" in v else [vv["point"] for vv in v.values()]
            rng = all(a <= p <= b for p in pts)
        out[k] = {"verdict_as_expected": met, "point_in_expected_range": rng}
    return out


# ------------------------------------------------------------------------------------------ main
def main():
    t_start = time.time()
    R = M.rows()
    S = K.load_npz(K.CACHE / "sets.npz")
    inset = M.row_set(R, S)
    I = K.load_inputs(M.RULE)
    Gd = K.grid_lean(("ti", "seg"))
    n = len(R["y"])
    base = {"O": R["y"].astype(np.float64), "G": I["G"], "H": I["H"], "MED3": np.full(n, np.nan)}
    rs = np.flatnonzero(inset)
    base["MED3"][rs] = M.med3(R["y"][rs], I["anc_g"][rs], Gd["ti"], Gd["seg"])
    lab = np.zeros(n, np.int64)
    lab[rs] = M.label(R["y"][rs], R["tg_L5"][rs])
    rbc = M.load_npz(C.CACHE / "strata_inputs.npz")["rb_class"]
    us, tab, fifth = M.units(R)
    _, unit_d = np.unique(R["fidx"], return_inverse=True)
    _, unit_c = np.unique(R["clus"], return_inverse=True)
    auc_matrix = load_module("readout73", C.HERE / "readout.py").auc_matrix
    res = {"registration": M.REG, "addendum": M.ADD, "n_boot": N, "seed": M.SEED,
           "row_set": {"d1_rows": int(n), "rows": int(inset.sum()), "discharges": int(len(np.unique(R["fidx"][inset]))),
                       "clusters": int(len(np.unique(R["clus"][inset]))),
                       "label_positives": int(lab[rs].sum()), "label_share": float(lab[rs].mean()),
                       "label_above_1.35": int(np.sum(R["y"][rs] / R["tg_L5"][rs] > M.HI)),
                       "label_below_0.74": int(np.sum(R["y"][rs] / R["tg_L5"][rs] < M.LO))},
           "designs": {}, "auc": {}, "counts": {}}
    for d in DESIGNS:
        out, rows, subs = build_design(R, us, d, inset, base, unit_d, unit_c, rbc)
        res["designs"][d] = out
        res["auc"][d] = auc_design(R, d, rows, subs, lab, unit_d, unit_c, auc_matrix)
        fr = {}
        for units in subs:
            for u in units:
                for L in FITTED + M.CLFS:
                    z = np.load(M.fit_path(u, L))
                    fr.setdefault(L, []).append(int(z["fit_rows"]))
        res["counts"][d] = {"scored": out["all"]["counts"], "units": [u["name"] for units in subs for u in units],
                            "fit_rows_per_unit": {L: {"min": min(v), "max": max(v)} for L, v in fr.items()}}
    res["lag_control"] = lag_section(R, us, inset, unit_d, unit_c)
    res["statements"] = statements(res)
    res["expectations"] = expectation_met(res["statements"])
    res["checks"] = {nm: json.loads((M.CHK / f"{nm}.json").read_text(encoding="utf-8"))
                     for nm in ("data", "b75", "compare", "frame", "leakage")}
    res["checks"]["compare"] = {L: res["checks"]["compare"][L]["all"] for L in ("T0", "TX")}
    res["runtime_readout_s"] = round(time.time() - t_start, 1)
    M.save_json(M.OUT_JSON, res)
    M.save_text(M.OUT_MD, load_module("render77", M.HERE / "render77.py").render(res))
    M.log(f"written {M.OUT_JSON.name}, {M.OUT_MD.name}")


if __name__ == "__main__":
    main()
