"""B.78 readout (registration sections 6-7, Addendum 1 sections 6-8): every table and statement in one pass into the
JSON, then the markdown.  The progress log carries row counts only.

    py ces_prediction/experiments/b78_sparse/readout78.py
Output: docs/paper/b78_sparse.json / .md
"""
from __future__ import annotations

import importlib.util
import json
import time
import warnings

import common78 as M

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = M.C
N = C.N_BOOT
FORMS = ("discharge", "cluster")
DESIGNS = ("V1", "V2", "V3", "V4a", "V4b")
STRATA_DESIGNS = ("V1", "V4a", "V4b")
POS = ("third 1", "third 2", "third 3", "beyond one spacing")
LNK = np.log(np.array(M.KS, np.float64))
LN_LO, LN_HI = 0.0, float(np.log(M.KS[-1]))
SENT = 1e3


# ------------------------------------------------------------------------------------------ helpers
def q(d):
    d = d[np.isfinite(d)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def pack(pt, dd, dc):
    return {"point": float(pt), "discharge": q(dd), "cluster": q(dc)}


def draws(name, nsub, nu):
    rng = M.rng_for(name)
    pick = rng.integers(0, nsub, N) if nsub > 1 else np.zeros(N, np.int64)
    out = []
    for si in range(nsub):
        sel = np.flatnonzero(pick == si)
        out.append((sel, rng.multinomial(nu, np.full(nu, 1.0 / nu), size=len(sel)).astype(np.float64)))
    return out


def eq_ln(T, y):
    """T: (m, 6) SSE of T0 under K1..K50 on the same rows; y: (m,) SSE of TX under one rule.  ln k' of the first point
    at which the piecewise-linear curve (ln k, SSE of T0) reaches y from below; -SENT when y <= SSE(T0 under K1)
    (k' below 1), +SENT when the curve stays below y up to K50 (k' above 50)."""
    m = len(y)
    out = np.full(m, SENT)
    below = y <= T[:, 0]
    out[below] = -SENT
    done = below.copy()
    for j in range(len(LNK) - 1):
        y0, y1 = T[:, j], T[:, j + 1]
        hit = ~done & (y1 >= y)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = LNK[j] + (y - y0) / (y1 - y0) * (LNK[j + 1] - LNK[j])
        out[hit] = x[hit]
        done |= hit
    return out


def kfmt(x):
    """ln k' -> k' (float) or a censored label."""
    if x < LN_LO - 1e-12:
        return "below 1"
    if x > LN_HI + 1e-12:
        return f"above {M.KS[-1]}"
    return float(np.exp(x))


class Group:
    """One design x target (common set) with strata.  Per stratum and form: per-unit SSE sums of every estimator in every
    sub, then one set of draws shared by every quantity of that stratum and form (B.75 readout, seed 20261178)."""

    def __init__(self, name, rows, strata, unit_d, unit_c, nsub):
        self.name, self.rows, self.nsub = name, rows, nsub
        self.strata = strata
        self.idx = {}
        for s, m in strata.items():
            for form in FORMS:
                uu = (unit_d if form == "discharge" else unit_c)[rows[m]]
                u, inv = np.unique(uu, return_inverse=True)
                self.idx[(s, form)] = (inv, len(u))
        self.sums = {k: {} for k in self.idx}
        self.tot = {}

    def add(self, e, si, se):
        if not np.all(np.isfinite(se)):
            raise SystemExit(f"FATAL: {self.name}: {e} has {int((~np.isfinite(se)).sum())} rows without a value")
        for (s, form), (inv, nu) in self.idx.items():
            self.sums[(s, form)].setdefault(e, [None] * self.nsub)[si] = np.bincount(inv, se[self.strata[s]], nu)

    def finish(self, s):
        for form in FORMS:
            inv, nu = self.idx[(s, form)]
            W = draws(f"{self.name}|common|{s}|{form}", self.nsub, nu)
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

    def recovered(self, s, t0, tx, ref):
        return self._f(s, lambda T, X, F: (T - X) / (T - F), t0, tx, ref)

    def eqrate(self, s, k):
        """Equivalent rate of TX under Kk (Addendum 1 section 6): point from the SSEs averaged over the splits (the rows
        are the same in every split), interval draw by draw; k' and the factor k / k'."""
        es = [f"T0|{r}" for r in M.RULES]
        ey = f"TX|K{k}"

        def factor(kp):
            if isinstance(kp, float):
                return k / kp
            return f"above {k}" if kp == "below 1" else f"below {k / M.KS[-1]:.2f}"

        out = {}
        for form in FORMS:
            t = self.tot[(s, form)]
            T = np.column_stack([t[e][1] for e in es])
            x = eq_ln(T, t[ey][1])
            lo, hi = (float(np.quantile(x, p)) for p in (0.025, 0.975))
            kp = [kfmt(lo), kfmt(hi)]
            out[form] = {"k_prime": kp, "factor_k_over_k_prime": [factor(kp[1]), factor(kp[0])],
                         "share_below_1": float(np.mean(x < LN_LO - 1e-12)),
                         "share_above_50": float(np.mean(x > LN_HI + 1e-12)),
                         "share_T0_curve_monotone": float(np.mean(np.all(np.diff(T, axis=1) >= 0, axis=1)))}
            if form == "discharge":
                Tp = np.array([[t[e][0].mean() for e in es]])
                kpp = kfmt(float(eq_ln(Tp, np.array([t[ey][0].mean()]))[0]))
                out["point"] = {"k_prime": kpp, "factor_k_over_k_prime": factor(kpp),
                                "T0_curve_monotone": bool(np.all(np.diff(Tp[0]) >= 0))}
        return out


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
        p = M.FITS / f"{u['name']}__{L}__{r}__{X}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        a[z["idx"]] = z["est"]
    return a


def estimator_list(X, design):
    out = []
    for r in M.RULES:
        out += [(f"H|{r}", None, r), (f"M5|{r}", None, r), (f"G|{r}", None, r), (f"T0|{r}", "T0", r), (f"TX|{r}", "TX", r)]
        if design in STRATA_DESIGNS and r in ("K1", "K10"):
            out += [(f"TB|{r}", "TB", r), (f"TE|{r}", "TE", r)]
    if design == "V1" and X == "L5":
        out += [("TXp|K10", "TXp", "K10"), ("TXo|K10", "TXo", "K10")]
    if X == "L5":
        out.append(("O", None, None))
    return out


def position_classes(I, fr, sel, gi_all, k):
    """Rows sel (indices or mask over the D1 rows): 0, 1, 2 = the thirds of the kept spacing (age of the newest visible
    value in frames, uf, against k: 3 uf < k, k <= 3 uf < 2k, 2k <= 3 uf < 3k); 3: uf >= k (the newest kept frame
    before t carries no visible value)."""
    anc = I["anc_g"][sel]
    gi = gi_all[sel]
    cls = np.full(len(gi), -1, np.int64)
    ok = anc >= 0
    uf = fr[gi[ok]] - fr[anc[ok]]
    cls[ok] = np.where(uf >= k, 3, np.where(3 * uf < k, 0, np.where(3 * uf < 2 * k, 1, 2)))
    return cls


def build_group(name, rows, strata, subs, X, design, R, base, unit_d, unit_c):
    g = Group(name, rows, strata, unit_d, unit_c, len(subs))
    y = R[f"tg_{X}"][rows]
    n = len(R["y"])
    for e, L, r in estimator_list(X, design):
        if L is None:
            v = R["y"][rows] if e == "O" else base[r][e.split("|")[0]][rows]
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


def rule_block(g, s, X):
    out = {}
    for j, r in enumerate(M.RULES):
        o = {"rmse_eV": {}}
        for lab in ("H", "M5", "G", "T0", "TX", "TB", "TE", "TXp", "TXo"):
            if g.has(s, f"{lab}|{r}"):
                o["rmse_eV"][lab] = g.rmse(s, f"{lab}|{r}")
        if X == "L5":
            o["rmse_eV"]["O"] = g.rmse(s, "O")
        o["T0/G"] = g.skill(s, f"T0|{r}", f"G|{r}")
        o["T0/H"] = g.skill(s, f"T0|{r}", f"H|{r}")
        o["TX/T0"] = g.skill(s, f"TX|{r}", f"T0|{r}")
        for lab in ("TB", "TE", "TXp", "TXo"):
            if g.has(s, f"{lab}|{r}"):
                o[f"{lab}/T0"] = g.skill(s, f"{lab}|{r}", f"T0|{r}")
        if g.has(s, f"TXo|{r}"):
            o["TX/TXo"] = g.skill(s, f"TX|{r}", f"TXo|{r}")
        if X == "L5":
            o["T0/O"] = g.skill(s, f"T0|{r}", "O")
            o["TX/O"] = g.skill(s, f"TX|{r}", "O")
        if r != "K1":
            o["cost T0"] = g.ratio(s, f"T0|{r}", "T0|K1")
            o["cost TX"] = g.ratio(s, f"TX|{r}", "TX|K1")
            o["recovered share"] = g.recovered(s, f"T0|{r}", f"TX|{r}", "T0|K1")
            prev = M.RULES[j - 1]
            o["TX over T0 of the next denser rule"] = {"denser_rule": prev, **g.skill(s, f"TX|{r}", f"T0|{prev}")}
        o["equivalent rate"] = g.eqrate(s, M.KOF[r])
        out[r] = o
    return out


def strata_block(g, s, r):
    o = {"counts": g.counts(s), "rmse_eV": {"T0": g.rmse(s, f"T0|{r}"), "TX": g.rmse(s, f"TX|{r}")},
         "TX/T0": g.skill(s, f"TX|{r}", f"T0|{r}")}
    return o


# ------------------------------------------------------------------------------------------ statements
def statements(res):
    S = {}

    def put(k, text, v, holds, exp, note=""):
        S[k] = {"text": text, "value": v, "holds": bool(holds), "expected": exp, "note": note}

    def ivs(v):
        return f"{v['point']:+.3f}; c [{v['cluster'][0]:+.3f}, {v['cluster'][1]:+.3f}]"

    c = lambda d: res["main"][f"{d}|L5"]["all"]   # noqa: E731
    v = c("V1")["K10"]["TX/T0"]
    put("K-1", "V1, K10: the increment lies above zero", v, v["cluster"][0] > 0, "holds (+0.10 to +0.30)")
    v = c("V2")["K10"]["TX/T0"]
    put("K-2", "V2, K10: the increment lies above zero", v, v["cluster"][0] > 0, "holds")
    v = c("V3")["K10"]["TX/T0"]
    put("K-3", "V3, K10: the increment lies above zero", v, v["cluster"][0] > 0, "uncertain, leaning to holds")
    v = c("V4a")["K10"]["TX/T0"]
    put("K-4", "V4 (fit to 31991), K10: the increment lies above zero", v, v["cluster"][0] > 0, "uncertain",
        f"V4b (fit to 32310): {ivs(c('V4b')['K10']['TX/T0'])}")
    v = c("V1")["K10"]["TX over T0 of the next denser rule"]
    put("K-5", "V1: the MSE of TX under K10 lies below that of T0 under K5 (interval of 1 - MSE(TX K10)/MSE(T0 K5) above zero)",
        {k_: v[k_] for k_ in ("point", "discharge", "cluster")}, v["cluster"][0] > 0, "uncertain",
        f"RMSE TX K10 {c('V1')['K10']['rmse_eV']['TX']:.1f} eV, T0 K5 {c('V1')['K5']['rmse_eV']['T0']:.1f} eV")
    v = c("V1")["K10"]["cost TX"]
    put("K-6", "V1: MSE(TX under K10) / MSE(TX under K1) lies below 2", v, v["cluster"][1] < 2, "holds (1.3 to 2.0)")
    v = c("V1")["K10"]["TE/T0"]
    put("K-7", "V1, K10: TE over T0 lies above zero", v, v["cluster"][0] > 0, "holds")
    v = c("V1")["K10"]["TXp/T0"]
    put("K-8", "V1, K10, placebo: the interval of TX over T0 contains zero or lies below it", v, v["cluster"][0] <= 0, "holds")
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
            rng = a <= s["value"]["point"] <= b
        out[k] = {"verdict_as_expected": met, "point_in_expected_range": rng}
    return out


# ------------------------------------------------------------------------------------------ main
def main():
    t_start = time.time()
    R = M.load_keys(C.CACHE / "rows.npz", K_ROWS)
    sets = M.load_npz(M.CACHE / "sets.npz")
    F = M.load_frames()
    fr = F["fr"]
    us, tab, fifth = M.units(R)
    _, unit_d = np.unique(R["fidx"], return_inverse=True)
    _, unit_c = np.unique(R["clus"], return_inverse=True)
    base, inp = {}, {}
    for r in M.RULES:
        I = M.load_inputs(r)
        base[r] = {"H": I["H"], "M5": I["M5"], "G": I["G"]}
        inp[r] = {"anc_g": I["anc_g"], "age": I["age"]}
    res = {"registration": M.REG, "addendum": M.ADD, "n_boot": N, "seed": M.SEED,
           "frames": json.loads((M.CACHE / "frames.json").read_text(encoding="utf-8")),
           "sets": json.loads((M.CACHE / "sets.json").read_text(encoding="utf-8")), "designs": describe(us, tab, fifth),
           "main": {}, "counts": {}}
    for d in DESIGNS:
        subs = subs_of(us, d)
        scored = np.isin(R["fidx"], np.concatenate([u["test"] for u in subs[0]]))
        for X in M.TARGETS:
            rows = np.flatnonzero(scored & sets[f"common_{X}"])
            strata = {"all": np.ones(len(rows), bool)}
            empty = []
            if d in STRATA_DESIGNS:
                for r in M.RULES[1:]:
                    cls = position_classes(inp[r], fr, rows, R["gi"], M.KOF[r])
                    if np.any(cls < 0):
                        raise SystemExit("FATAL: a common-set row without a visible value")
                    for ci, nm in enumerate(POS):
                        m = cls == ci
                        if m.any():
                            strata[f"{r}|{nm}"] = m
                        else:
                            empty.append(f"{r}|{nm}")
            name = f"{d}|{X}"
            g = build_group(name, rows, strata, subs, X, d, R, base, unit_d, unit_c)
            out = {"all": None, "positions": {}, "empty_positions": empty}
            for s in strata:
                g.finish(s)
                if s == "all":
                    out["all"] = {"counts": g.counts(s), **rule_block(g, s, X)}
                else:
                    r = s.split("|")[0]
                    out["positions"][s] = strata_block(g, s, r)
                g.tot.clear()
            res["main"][name] = out
            M.log(f"{name}: rows {len(rows):,}, strata {len(strata)} done")
    # ---- age of the newest visible value on the common set, by rule
    ages = {}
    for X in M.TARGETS:
        m = sets[f"common_{X}"]
        ages[X] = {}
        for r in M.RULES:
            a = inp[r]["age"][m]
            ages[X][r] = {"rows": int(m.sum()), "age_ms_quantiles": {str(p): float(np.quantile(a, p)) for p in (0.1, 0.5, 0.9)}}
            if r != "K1":
                cls = position_classes(inp[r], fr, m, R["gi"], M.KOF[r])
                ages[X][r]["share_by_position"] = {nm: float(np.mean(cls == ci)) for ci, nm in enumerate(POS)}
    res["age_on_common_set"] = ages
    # ---- counts per design and fit sizes
    for d in DESIGNS:
        subs = subs_of(us, d)
        c_ = {X: res["main"][f"{d}|{X}"]["all"]["counts"] for X in M.TARGETS}
        fr_ = {}
        for units in subs:
            for u in units:
                for X in M.TARGETS:
                    z = np.load(M.FITS / f"{u['name']}__T0__K1__{X}.npz")
                    fr_.setdefault(X, []).append(int(z["fit_rows"]))
        c_["fit_rows_per_unit"] = {k: {"min": min(v), "max": max(v)} for k, v in fr_.items()}
        res["counts"][d] = c_
    # ---- the anchor-row control: rows whose anchor-row fast scalars are not finite
    nf = {"fit": [], "test": []}
    for u in (x for x in us if x["design"] == "V1"):
        z = np.load(M.FITS / f"{u['name']}__TXo__K10__L5.npz")
        nf["fit"].append(int(z["nonfinite_fit_rows"]))
        nf["test"].append(int(z["nonfinite_test_rows"]))
    res["anchor_row_control_nonfinite_rows"] = {"fit_rows_per_unit_max": max(nf["fit"]), "test_rows_per_unit_max": max(nf["test"])}
    res["statements"] = statements(res)
    res["expectations"] = expectation_met(res["statements"])
    res["checks"] = {n: json.loads((M.CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("frames", "k1", "visibility", "units")}
    res["runtime_readout_s"] = round(time.time() - t_start, 1)
    M.save_json(M.OUT_JSON, res)
    M.save_text(M.OUT_MD, load_render().render(res))
    M.log(f"written {M.OUT_JSON.name}, {M.OUT_MD.name}")


K_ROWS = ("fidx", "grow", "gi", "y", "shot", "clus", "blk", "tstep", "tg_F", "tg_L5", "r2_k", "r3_k", "r4_k")


def load_render():
    spec = importlib.util.spec_from_file_location("render78", M.HERE / "render78.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def describe(us, tab, fifth):
    sh = tab["shot"]
    return {"d1_discharges": int(len(tab["fidx"])), "clusters_d1": int(len(np.unique(tab["clus"]))),
            "v3_fifths": [{"shots": [int(sh[fifth == q_].min()), int(sh[fifth == q_].max())], "discharges": int((fifth == q_).sum())}
                          for q_ in range(5)],
            "units": [{"name": u["name"], "fit": int(len(u["fit"])), "test": int(len(u["test"]))} for u in us]}


if __name__ == "__main__":
    main()
