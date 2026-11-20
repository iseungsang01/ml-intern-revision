"""B.75 readout (registration sections 7-8, Addendum 1 sections 6-7): every table and statement in one pass, then the
markdown.

    py ces_prediction/experiments/b75_availability/readout.py
Output: docs/paper/b75_availability.json / .md
"""
from __future__ import annotations

import json
import platform
import time
import warnings

import common75 as K

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = K.C
N = C.N_BOOT
FORMS = ("discharge", "cluster")
DESIGNS = ("V1", "V2", "V3", "V4a", "V4b")
REF = {"common": "D10", "S": "D10S"}
RULES_OF = {"common": K.FIXED, "S": ("S", "D10S")}
BASE_RULE = {r: r for r in K.RULES}
BASE_RULE["D10S"] = "D10"


# ------------------------------------------------------------------------------------------ helpers
def q(d):
    d = d[np.isfinite(d)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def pack(pt, dd, dc):
    return {"point": float(pt), "discharge": q(dd), "cluster": q(dc)}


def draws(name, nsub, nu):
    rng = K.rng_for(name)
    pick = rng.integers(0, nsub, N) if nsub > 1 else np.zeros(N, np.int64)
    out = []
    for si in range(nsub):
        sel = np.flatnonzero(pick == si)
        out.append((sel, rng.multinomial(nu, np.full(nu, 1.0 / nu), size=len(sel)).astype(np.float64)))
    return out


class Group:
    """One design x target x row set (with strata).  Per stratum and form: per-unit SSE sums of every estimator in every
    sub, then one set of draws shared by every quantity of that stratum and form."""

    def __init__(self, name, rows, strata, unit_d, unit_c, nsub):
        self.name, self.rows, self.nsub = name, rows, nsub
        self.strata = strata                      # {stratum: bool mask over rows}
        self.idx = {}
        for s, m in strata.items():
            for form in FORMS:
                uu = (unit_d if form == "discharge" else unit_c)[rows[m]]
                u, inv = np.unique(uu, return_inverse=True)
                self.idx[(s, form)] = (inv, len(u))
        self.sums = {k: {} for k in self.idx}     # (s, form) -> e -> [vector per sub]
        self.tot = {}

    def add(self, e, si, se):
        if not np.all(np.isfinite(se)):
            raise SystemExit(f"FATAL: {self.name}: {e} has {int((~np.isfinite(se)).sum())} rows without a value")
        for (s, form), (inv, nu) in self.idx.items():
            self.sums[(s, form)].setdefault(e, [None] * self.nsub)[si] = np.bincount(inv, se[self.strata[s]], nu)

    def finish(self, s):
        """Draws for stratum s; tot[(s, form)][e] = (points per sub, draws)."""
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

    def has(self, s, e):
        return e in self.sums[(s, "discharge")]

    def n(self, s):
        return int(self.strata[s].sum())

    def counts(self, s):
        return {"rows": self.n(s), "discharges": self.idx[(s, "discharge")][1], "clusters": self.idx[(s, "cluster")][1]}

    def rmse(self, s, e):
        pts, _ = self.tot[(s, "discharge")][e]
        return float(np.mean(np.sqrt(pts / self.n(s))))

    def _f(self, s, fn, *es):
        vals = []
        for form in FORMS:
            t = self.tot[(s, form)]
            args = [t[e] for e in es]
            vals.append(fn(*[a[1] for a in args]))
            if form == "discharge":
                pt = float(np.mean(fn(*[a[0] for a in args])))
        return pack(pt, vals[0], vals[1])

    def skill(self, s, a, b):
        return self._f(s, lambda A, B: 1 - A / B, a, b)

    def ratio(self, s, a, b):
        return self._f(s, lambda A, B: A / B, a, b)

    def skill_diff(self, s, a1, b1, a2, b2):
        return self._f(s, lambda A1, B1, A2, B2: (1 - A1 / B1) - (1 - A2 / B2), a1, b1, a2, b2)

    def recovered(self, s, t0, tx, ref):
        return self._f(s, lambda T, X, F: (T - X) / (T - F), t0, tx, ref)


# ------------------------------------------------------------------------------------------ assembly
def subs_of(us, d):
    if d in ("V1", "V2"):
        subs = C.REPEATS if d == "V1" else C.V2_SEEDS
        return [[u for u in us if u["design"] == d and u["sub"] == s] for s in subs]
    if d == "V3":
        return [[u for u in us if u["design"] == "V3"]]
    return [[u for u in us if u["name"] == d]]


def load_est(units, L, r, X, n):
    a = np.full(n, np.nan)
    for u in units:
        p = K.FITS / f"{u['name']}__{L}__{r}__{X}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        a[z["idx"]] = z["est"]
    return a


def estimator_list(kind, X, design):
    """(sum name, learner file name or None, rule) of every estimator scored in a group."""
    v14 = design in ("V1", "V4a", "V4b")
    out = []
    for r in RULES_OF[kind]:
        br = BASE_RULE[r]
        out += [(f"H|{r}", None, br), (f"M5|{r}", None, br), (f"G|{r}", None, br)]
        out += [(f"T0|{r}", "T0", r), (f"TX|{r}", "TX", r)]
        if v14 and r in ("D10", "D650", "S"):
            out += [(f"{L}|{r}", L, r) for L in ("TX8", "TB", "TE")]
        if r not in ("D10", "D10S"):
            out += [(f"T0a|{r}", "T0a", r), (f"TXa|{r}", "TXa", r)]
    if kind == "common":
        out.append(("X0", "X0", "C"))
    if X == "L5":
        out.append(("O", None, None))
    return out


def build_group(name, rows, strata, subs, kind, X, design, R, base, unit_d, unit_c):
    g = Group(name, rows, strata, unit_d, unit_c, len(subs))
    y = R[f"tg_{X}"][rows]
    n = len(R["y"])
    for e, L, r in estimator_list(kind, X, design):
        if L is None:
            if e == "O":
                v = R["y"][rows]
            else:
                v = base[r][e.split("|")[0]][rows]
            se = (v - y) ** 2
            for si in range(len(subs)):
                g.add(e, si, se)
            continue
        for si, units in enumerate(subs):
            a = load_est(units, L, r, X, n)
            if a is None:
                raise SystemExit(f"FATAL: {name}: no file for {L} {r} {X}")
            g.add(e, si, (a[rows] - y) ** 2)
    return g


def rule_block(g, s, kind, X, design):
    v14 = design in ("V1", "V4a", "V4b")
    ref = REF[kind]
    out = {}
    for r in RULES_OF[kind]:
        o = {"rmse_eV": {}}
        for lab in ("H", "M5", "G", "T0", "TX8", "TX", "TB", "TE", "T0a", "TXa"):
            if g.has(s, f"{lab}|{r}"):
                o["rmse_eV"][lab] = g.rmse(s, f"{lab}|{r}")
        if kind == "common":
            o["rmse_eV"]["X0"] = g.rmse(s, "X0")
        if X == "L5":
            o["rmse_eV"]["O"] = g.rmse(s, "O")
        o["T0/G"] = g.skill(s, f"T0|{r}", f"G|{r}")
        o["T0/H"] = g.skill(s, f"T0|{r}", f"H|{r}")
        o["TX/T0"] = g.skill(s, f"TX|{r}", f"T0|{r}")
        if v14 and r in ("D10", "D650", "S"):
            for L in ("TX8", "TB", "TE"):
                o[f"{L}/T0"] = g.skill(s, f"{L}|{r}", f"T0|{r}")
        if kind == "common":
            o["TX/X0"] = g.skill(s, f"TX|{r}", "X0")
            o["T0/X0"] = g.skill(s, f"T0|{r}", "X0")
        if X == "L5":
            for lab in ("H", "G", "T0", "TX"):
                o[f"{lab}/O"] = g.skill(s, f"{lab}|{r}", "O")
        if r != ref:
            o["staleness T0"] = g.ratio(s, f"T0|{r}", f"T0|{ref}")
            o["staleness TX"] = g.ratio(s, f"TX|{r}", f"TX|{ref}")
            o["recovered share"] = g.recovered(s, f"T0|{r}", f"TX|{r}", f"T0|{ref}")
            o["applied T0: refit over applied"] = g.skill(s, f"T0|{r}", f"T0a|{r}")
            o["applied TX: refit over applied"] = g.skill(s, f"TX|{r}", f"TXa|{r}")
            o["applied TXa/T0a"] = g.skill(s, f"TXa|{r}", f"T0a|{r}")
            o["staleness T0a"] = g.ratio(s, f"T0a|{r}", f"T0|{ref}")
            o["staleness TXa"] = g.ratio(s, f"TXa|{r}", f"TX|{ref}")
        out[r] = o
    if kind == "common":
        out["D650 minus D10: TX/T0"] = g.skill_diff(s, "TX|D650", "T0|D650", "TX|D10", "T0|D10")
    return out


# ------------------------------------------------------------------------------------------ V3 / V3c
def v3c_section(R, us, sets, unit_d, unit_c):
    """V3 beside V3c per fifth (L5, D10 and D650, common set) and R13 (fifths 2 and 3 pooled, D10)."""
    n = len(R["y"])
    X = "L5"
    y_all = R["tg_L5"]
    res = {}
    for key, qs in (("fifth 2", (1,)), ("fifth 3", (2,)), ("fifth 4", (3,)), ("fifth 5", (4,)), ("fifths 2+3", (1, 2))):
        test = np.isin(R["fidx"], np.concatenate([next(u for u in us if u["name"] == f"V3_q{qq}")["test"] for qq in qs]))
        rows = np.flatnonzero(test & sets["common_L5"])
        has_c = all(qq in K.V3C_FIFTHS for qq in qs)
        nsub = len(K.V3C_SEEDS) if has_c else 1
        g = Group(f"V3c|{key}|{X}", rows, {"all": np.ones(len(rows), bool)}, unit_d, unit_c, nsub)
        y = y_all[rows]
        v3 = [u for u in us if u["design"] == "V3" and u["sub"] in qs]
        for L in ("T0", "TX"):
            for r in ("D10", "D650"):
                a = load_est(v3, L, r, X, n)
                se = (a[rows] - y) ** 2
                for si in range(nsub):
                    g.add(f"{L}|{r}", si, se)
                if has_c:
                    for si, s in enumerate(K.V3C_SEEDS):
                        uc = [u for u in us if u["design"] == "V3c" and u["sub"] == s and u["fifth"] in qs]
                        a = load_est(uc, L, r, X, n)
                        g.add(f"c{L}|{r}", si, (a[rows] - y) ** 2)
        g.finish("all")
        o = {**g.counts("all")}
        for r in ("D10", "D650"):
            o[f"V3 TX/T0 {r}"] = g.skill("all", f"TX|{r}", f"T0|{r}")
            o[f"V3 T0 RMSE {r}"] = g.rmse("all", f"T0|{r}")
            o[f"V3 TX RMSE {r}"] = g.rmse("all", f"TX|{r}")
            if has_c:
                o[f"V3c TX/T0 {r}"] = g.skill("all", f"cTX|{r}", f"cT0|{r}")
                o[f"V3 minus V3c TX/T0 {r}"] = g.skill_diff("all", f"TX|{r}", f"T0|{r}", f"cTX|{r}", f"cT0|{r}")
                o[f"V3 T0 over V3c T0 {r}"] = g.skill("all", f"T0|{r}", f"cT0|{r}")
                o[f"V3 TX over V3c TX {r}"] = g.skill("all", f"TX|{r}", f"cTX|{r}")
                o[f"V3c T0 RMSE {r}"] = g.rmse("all", f"cT0|{r}")
                o[f"V3c TX RMSE {r}"] = g.rmse("all", f"cTX|{r}")
        res[key] = o
        K.log(f"V3/V3c {key}: rows {len(rows):,}")
    return res


# ------------------------------------------------------------------------------------------ timing
def timing(R, sets, us):
    import stage_trees as ST
    from threadpoolctl import threadpool_limits
    G = K.grid_lean(("off", "time", "ti", "seg", "fast"))
    I = K.load_inputs("D650")
    u = next(x for x in us if x["name"] == "V1_r2_k0")
    FE = ST.Feats()
    wf = FE.window(u["fit"])
    bz, ez = K.zmeans(FE, u["fit"])
    ms_b, ms_e = K.run_mean10(bz, G["seg"]), K.run_mean10(ez, G["seg"])
    mr_b, mr_e = (K.run_mean10(x, G["seg"]) for x in K.raw_means(G["fast"]))
    H10 = K.history_matrix(R["tstep"], I)
    A = K.anchor_scalars(R["gi"], I["anc_g"], ms_b, ms_e, mr_b, mr_e)
    Xtx = np.column_stack([H10, wf[R["gi"]], A])
    sm = sets["common_L5"]
    fr = np.isin(R["fidx"], u["fit"]) & sm
    with threadpool_limits(1):
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(random_state=C.TREE_SEED).fit(Xtx[fr], (R["tg_L5"] - I["G"])[fr])
        te = np.flatnonzero(np.isin(R["fidx"], u["test"]) & sm)
        cached = np.load(K.FITS / f"{u['name']}__TX__D650__L5.npz")
        same = bool(np.array_equal(cached["idx"], te) and np.max(np.abs(cached["est"] - (I["G"][te] + m.predict(Xtx[te])))) == 0.0)
        cand = np.flatnonzero(sm)
        pick = K.rng_for("timing").choice(cand, 2020, replace=False)
        off, tm, ti, seg = G["off"], G["time"], G["ti"], G["seg"]
        obs_by_file = {}
        times = []
        for k, j in enumerate(pick):
            i = int(R["fidx"][j])
            a = off[i]
            if i not in obs_by_file:
                obs_by_file[i] = np.flatnonzero(np.isfinite(ti[a:off[i + 1]])) + a
            obs = obs_by_file[i]
            gi = int(R["gi"][j])
            # the row's visible values (the last up to 16 of its block at or before the D650 limit)
            tobs = tm[obs]
            p = int(np.searchsorted(tobs, I["lim"][j] + K.TOL, side="right"))
            b = int(np.searchsorted(obs, gi - int(np.sum(seg[a:gi] == seg[gi])), side="left"))
            g16 = obs[max(p - 16, b):p]
            tt, vv, xrow = tm[g16], ti[g16], Xtx[j:j + 1]
            t0 = time.perf_counter()
            gval = K.BI.predict_gp_causal(tt, vv, tm[gi])
            m.predict(xrow)
            dt = time.perf_counter() - t0
            if k >= 20:
                times.append(dt * 1000.0)
            if k < 50 and not np.isclose(gval, I["G"][j], rtol=0, atol=0):
                raise SystemExit("FATAL: timing GP differs from the cached G")
    t = np.array(times)
    cpu = platform.processor()
    try:
        import subprocess
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=60).stdout.strip() or cpu
    except Exception:  # noqa: BLE001
        pass
    import os
    return {"rows": int(len(t)), "rule": "D650", "model": f"{u['name']} TX D650 L5 (refit identical to cache: {same})",
            "median_ms": float(np.median(t)), "p99_ms": float(np.quantile(t, 0.99)), "mean_ms": float(t.mean()),
            "max_ms": float(t.max()), "threads": 1, "cpu": cpu, "logical_cpus": os.cpu_count(),
            "note": "shared machine: other jobs (a language-model server and other sessions) were running"}


# ------------------------------------------------------------------------------------------ statements
def statements(res):
    M = res["main"]
    S = {}

    def put(k, text, v, holds, exp, note=""):
        S[k] = {"text": text, "value": v, "holds": bool(holds), "expected": exp, "note": note}

    def lo(v):
        return v["cluster"][0]

    def ivs(v):
        return f"{v['point']:+.3f}; c [{v['cluster'][0]:+.3f}, {v['cluster'][1]:+.3f}]"

    c = lambda d: M[f"{d}|L5|common"]["all"]   # noqa: E731
    s_ = lambda d: M[f"{d}|L5|S"]["all"]       # noqa: E731
    v = c("V1")["D650"]["staleness T0"]
    put("R1", "V1, L5, common set: MSE(T0 under D650) / MSE(T0 under D10) lies above 1.5", v, lo(v) > 1.5, "holds (2 to 6)")
    v = c("V1")["D650"]["TX/T0"]
    put("R2", "V1, L5, D650: the increment lies above zero", v, lo(v) > 0, "holds (+0.05 to +0.25)")
    v = c("V1")["D650 minus D10: TX/T0"]
    put("R3", "V1, L5: the increment under D650 exceeds the increment under D10 (interval of the difference above zero)", v,
        lo(v) > 0, "uncertain, leaning to holds")
    v = c("V2")["D650"]["TX/T0"]
    put("R4", "V2, L5, D650: the increment lies above zero", v, lo(v) > 0, "holds")
    v = c("V3")["D650"]["TX/T0"]
    put("R5", "V3, L5, D650: the increment lies above zero", v, lo(v) > 0, "uncertain")
    v = c("V4a")["D650"]["TX/T0"]
    put("R6", "V4 (fit to 31991), L5, D650: the increment lies above zero", v, lo(v) > 0, "uncertain",
        f"V4b (fit to 32310): {ivs(c('V4b')['D650']['TX/T0'])}")
    v = s_("V1")["S"]["TX/T0"]
    put("R7", "V1, L5, rule S: the increment lies above zero", v, lo(v) > 0, "holds")
    v = s_("V4a")["S"]["TX/T0"]
    put("R8", "V4 (fit to 31991), L5, rule S: the increment lies above zero", v, lo(v) > 0, "uncertain",
        f"V4b: {ivs(s_('V4b')['S']['TX/T0'])}")
    v = c("V1")["D650"]["TE/T0"]
    put("R9", "V1, L5, D650: TE over T0 lies above zero", v, lo(v) > 0, "uncertain (+0.02 to +0.10)")
    v1, v4 = c("V1")["D650"]["T0/G"], c("V4a")["D650"]["T0/G"]
    put("R10", "V1 and V4, L5, D650: T0 over G lies above zero under each", {"V1": v1, "V4a": v4}, lo(v1) > 0 and lo(v4) > 0,
        "holds", f"V4b: {ivs(c('V4b')['D650']['T0/G'])}")
    v1, v4 = c("V1")["D10"]["TX/O"], c("V4a")["D10"]["TX/O"]
    put("R11", "V1 and V4, L5, D10, common set: the MSE of TX lies below that of the frame's own reading O under each "
               "(interval of 1 - MSE(TX)/MSE(O) above zero)", {"V1": v1, "V4a": v4}, lo(v1) > 0 and lo(v4) > 0,
        "holds (+0.3 to +0.6)", f"V4b: {ivs(c('V4b')['D10']['TX/O'])}")
    v = s_("V1")["S"]["TX/O"]
    put("R12", "V1, L5, rule S: the MSE of TX lies below that of O (interval of 1 - MSE(TX)/MSE(O) above zero)", v, lo(v) > 0,
        "does not hold", f"RMSE TX {s_('V1')['S']['rmse_eV']['TX']:.1f} eV, O {s_('V1')['S']['rmse_eV']['O']:.1f} eV")
    v = res["v3c"]["fifths 2+3"]["V3 minus V3c TX/T0 D10"]
    put("R13", "L5, D10, fifths 2 and 3 pooled: the increment of V3 minus the increment of V3c lies below zero", v,
        v["cluster"][1] < 0, "uncertain",
        f"V3 {ivs(res['v3c']['fifths 2+3']['V3 TX/T0 D10'])}; V3c {ivs(res['v3c']['fifths 2+3']['V3c TX/T0 D10'])}")
    return S


def expectation_met(st):
    """Whether the verdict agrees with the registered expectation ('uncertain' ones: not applicable)."""
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
    R = K.load_keys(C.CACHE / "rows.npz", K.ROW_KEYS + ("r2_k", "r3_k", "r4_k"))
    sets = K.load_npz(K.CACHE / "sets.npz")
    us, tab, fifth = K.units_all(R)
    _, unit_d = np.unique(R["fidx"], return_inverse=True)
    _, unit_c = np.unique(R["clus"], return_inverse=True)
    base = {}
    for r in K.RULES:
        I = K.load_inputs(r)
        base[r] = {"H": I["H"], "M5": I["M5"], "G": I["G"]}
    res = {"registration": K.REG, "addendum": K.ADD, "n_boot": N, "seed": K.SEED,
           "sets": json.loads((K.CACHE / "sets.json").read_text(encoding="utf-8")), "designs": describe(us, tab, fifth),
           "main": {}, "counts": {}}
    for d in DESIGNS:
        subs = subs_of(us, d)
        scored = np.isin(R["fidx"], np.concatenate([u["test"] for u in subs[0]]))
        for X in K.TARGETS:
            for kind in ("common", "S"):
                rows = np.flatnonzero(scored & sets[f"{kind}_{X}"])
                strata = {"all": np.ones(len(rows), bool)}
                if d == "V1":
                    for b in range(3):
                        strata[f"block {C.BLOCKS[b][0]}"] = R["blk"][rows] == b
                name = f"{d}|{X}|{kind}"
                g = build_group(name, rows, strata, subs, kind, X, d, R, base, unit_d, unit_c)
                out = {}
                for s in strata:
                    g.finish(s)
                    out[s] = {"counts": g.counts(s), **rule_block(g, s, kind, X, d)}
                    g.tot.clear()
                res["main"][name] = out
                K.log(f"{name}: rows {len(rows):,} done")
    res["v3c"] = v3c_section(R, us, sets, unit_d, unit_c)
    # ---- age under S on the S set
    I = K.load_inputs("S")
    ages = {}
    for X in K.TARGETS:
        m = sets[f"S_{X}"]
        a, la = I["age"][m] / 1000.0, I["lim_age"][m]
        qs = (0.1, 0.25, 0.5, 0.75, 0.9)
        ages[X] = {"rows": int(m.sum()),
                   "newest_value_age_s": {"quantiles": {str(p): float(np.quantile(a, p)) for p in qs},
                                          "bands": band_shares(a)},
                   "limit_age_s": {"quantiles": {str(p): float(np.quantile(la, p)) for p in qs}, "bands": band_shares(la)}}
    res["age_under_S"] = ages
    # ---- counts per design and fit sizes
    for d in DESIGNS:
        subs = subs_of(us, d)
        c_ = {}
        for X in K.TARGETS:
            for kind in ("common", "S"):
                c_[f"{kind}|{X}"] = res["main"][f"{d}|{X}|{kind}"]["all"]["counts"]
        fr = {}
        for units in subs:
            for u in units:
                for L, r, X in (("T0", "D10", "L5"), ("T0", "D650", "L5"), ("T0", "S", "L5"), ("T0", "D10S", "L5")):
                    z = np.load(K.FITS / f"{u['name']}__{L}__{r}__{X}.npz")
                    fr.setdefault(f"{L}|{r}|{X}", []).append(int(z["fit_rows"]))
        c_["fit_rows_per_unit_L5"] = {k: {"min": min(v), "max": max(v)} for k, v in fr.items()}
        res["counts"][d] = c_
    res["statements"] = statements(res)
    res["expectations"] = expectation_met(res["statements"])
    res["checks"] = {n: json.loads((K.CACHE / "checks" / f"{n}.json").read_text(encoding="utf-8"))
                     for n in ("gp", "repro", "visibility", "ages", "units")}
    res["timing"] = timing(R, sets, us)
    res["runtime_readout_s"] = round(time.time() - t_start, 1)
    K.save_json(K.OUT_JSON, res)
    K.save_text(K.OUT_MD, load_render().render(res))
    K.log(f"written {K.OUT_JSON.name}, {K.OUT_MD.name}")


def load_render():
    """This directory's render.py (b73_targets, first on sys.path, has a module of the same name)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("render75", K.HERE / "render.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def describe(us, tab, fifth):
    sh = tab["shot"]
    return {"d1_discharges": int(len(tab["fidx"])), "clusters_d1": int(len(np.unique(tab["clus"]))),
            "v3_fifths": [{"shots": [int(sh[fifth == q].min()), int(sh[fifth == q].max())], "discharges": int((fifth == q).sum())}
                          for q in range(5)],
            "units": [{"name": u["name"], "fit": int(len(u["fit"])), "test": int(len(u["test"]))} for u in us]}


def band_shares(a):
    h = np.histogram(a, (0.0, 0.1, 0.3, 1.0, np.inf))[0]
    return {nm: float(c / len(a)) for nm, c in zip(("0-0.1 s", "0.1-0.3 s", "0.3-1 s", "above 1 s"), h)}


if __name__ == "__main__":
    main()
