"""B.76 readout (registration sections 8-9, Addendum 1 sections 7-8): every table and statement in one pass into the JSON,
then the markdown.

    py ces_prediction/experiments/b76_mast_availability/readout.py
Output: docs/paper/b76_mast_availability.json / .md.  The progress log carries row counts only.
"""
from __future__ import annotations

import importlib.util
import json
import time
import warnings

import common76 as K

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
N = K.N_BOOT
FORMS = ("shot", "session")
DESIGNS = ("M1", "M2", "M3", "M4")
CHUNK = 1000


# ------------------------------------------------------------------------------------------ helpers
def q(d):
    d = d[np.isfinite(d)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def pack(pt, ds, dss):
    return {"point": float(pt), "shot": q(ds), "session": q(dss)}


class Group:
    """One design x target x row set, with strata.  Per stratum and form: per-unit SSE sums of every estimator in every
    sub, then one set of draws (one stream per cell) shared by every quantity of that stratum and form."""

    def __init__(self, name, rows, strata, unit_s, unit_ss, nsub):
        self.name, self.rows, self.nsub = name, rows, nsub
        self.strata = strata
        self.idx = {}
        for s, m in strata.items():
            for form in FORMS:
                uu = (unit_s if form == "shot" else unit_ss)[rows[m]]
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
            es = list(self.sums[(s, form)])
            for e in es:
                if any(v is None for v in self.sums[(s, form)][e]):
                    raise SystemExit(f"FATAL: {self.name}: {e} missing in a sub")
            V = [np.column_stack([self.sums[(s, form)][e][si] for e in es]) for si in range(self.nsub)]
            rng = K.rng_for(f"{self.name}|{s}|{form}")
            pick = rng.integers(0, self.nsub, N) if self.nsub > 1 else np.zeros(N, np.int64)
            D = np.empty((N, len(es)))
            p = np.full(nu, 1.0 / nu)
            for si in range(self.nsub):
                sel = np.flatnonzero(pick == si)
                for a in range(0, len(sel), CHUNK):
                    ch = sel[a:a + CHUNK]
                    Wm = rng.multinomial(nu, p, size=len(ch)).astype(np.float64)
                    D[ch] = Wm @ V[si]
            self.tot[(s, form)] = {e: (np.array([V[si][:, k].sum() for si in range(self.nsub)]), D[:, k]) for k, e in enumerate(es)}

    def has(self, s, e):
        return e in self.sums[(s, "shot")]

    def n(self, s):
        return int(self.strata[s].sum())

    def counts(self, s):
        return {"rows": self.n(s), "shots": self.idx[(s, "shot")][1], "sessions": self.idx[(s, "session")][1]}

    def rmse(self, s, e):
        pts, _ = self.tot[(s, "shot")][e]
        return float(np.mean(np.sqrt(pts / self.n(s))))

    def _f(self, s, fn, *es):
        vals = []
        for form in FORMS:
            t = self.tot[(s, form)]
            args = [t[e] for e in es]
            vals.append(fn(*[a[1] for a in args]))
            if form == "shot":
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
    if d in ("M1", "M2"):
        return [[u for u in us if u["design"] == d and u["sub"] == s] for s in K.SEEDS]
    if d == "M3":
        return [[u for u in us if u["design"] == "M3"]]
    return [[u for u in us if u["name"] == "M4"]]


def load_est(units, L, r, X, n):
    a = np.full(n, np.nan)
    for u in units:
        z = np.load(K.FITS / f"{u['name']}__{L}__{r}__{X}.npz")
        a[z["idx"]] = z["est"]
    return a


def build_group(name, rows, strata, subs, setname, X, d, R, base, unit_s, unit_ss):
    g = Group(name, rows, strata, unit_s, unit_ss, len(subs))
    y = (R["y"] if X == "F" else R["L5"])[rows]
    n = len(R["y"])
    rules = K.RULES if setname == "common" else K.DA_RULES
    fixed = []          # (name, values over all rows)
    fitted = []         # (name, learner, rule)
    for r in rules:
        fixed += [(f"H|{r}", base[r]["H"]), (f"G|{r}", base[r]["G"])]
        if setname == "common":
            fixed += [(f"M5|{r}", base[r]["M5"]), (f"RT|{r}", base[r]["RT"])]
            fitted += [(f"T0|{r}", "T0", r), (f"TX|{r}", "TX", r), (f"RTa|{r}", "RTa", r)]
        else:
            fitted += [(f"{L}|{r}", L, r) for L in ("T0d", "TXd", "TD", "TA")]
    if setname == "common" and d == "M1" and X == "L5":
        fitted += [("TXp|D100", "TXp", "D100"), ("TXo|D100", "TXo", "D100")]
    if X == "L5":
        fixed.append(("O", R["y"]))
    for e, v in fixed:
        se = (v[rows] - y) ** 2
        for si in range(len(subs)):
            g.add(e, si, se)
    for e, L, r in fitted:
        for si, units in enumerate(subs):
            a = load_est(units, L, r, X, n)
            g.add(e, si, (a[rows] - y) ** 2)
    return g


def rule_block(g, s, X, setname):
    out = {}
    if setname == "common":
        for r in K.RULES:
            o = {"rmse_eV": {lab: g.rmse(s, f"{lab}|{r}") for lab in ("H", "M5", "G", "T0", "TX", "RT", "RTa")}}
            if X == "L5":
                o["rmse_eV"]["O"] = g.rmse(s, "O")
            o["T0/G"] = g.skill(s, f"T0|{r}", f"G|{r}")
            o["T0/H"] = g.skill(s, f"T0|{r}", f"H|{r}")
            o["TX/T0"] = g.skill(s, f"TX|{r}", f"T0|{r}")
            o["RT/H"] = g.skill(s, f"RT|{r}", f"H|{r}")
            o["RT/T0"] = g.skill(s, f"RT|{r}", f"T0|{r}")
            o["RTa/H"] = g.skill(s, f"RTa|{r}", f"H|{r}")
            o["RTa/T0"] = g.skill(s, f"RTa|{r}", f"T0|{r}")
            if X == "L5":
                for lab in ("H", "G", "T0", "TX"):
                    o[f"{lab}/O"] = g.skill(s, f"{lab}|{r}", "O")
            if r != "D5":
                o["staleness T0"] = g.ratio(s, f"T0|{r}", "T0|D5")
                o["staleness TX"] = g.ratio(s, f"TX|{r}", "TX|D5")
                o["recovered share"] = g.recovered(s, f"T0|{r}", f"TX|{r}", "T0|D5")
                o["TX/T0 minus under D5"] = g.skill_diff(s, f"TX|{r}", f"T0|{r}", "TX|D5", "T0|D5")
                o["TX/T0(D5)"] = g.skill(s, f"TX|{r}", "T0|D5")
            out[r] = o
        if g.has(s, "TXp|D100"):
            out["controls"] = {"rmse_eV": {"T0 D5": g.rmse(s, "T0|D5"), "T0 D100": g.rmse(s, "T0|D100"),
                                           "TX D100": g.rmse(s, "TX|D100"), "TXp D100": g.rmse(s, "TXp|D100"),
                                           "TXo D100": g.rmse(s, "TXo|D100")},
                               "TXp/T0": g.skill(s, "TXp|D100", "T0|D100"),
                               "TXo/T0": g.skill(s, "TXo|D100", "T0|D100"),
                               "TX/TXo": g.skill(s, "TX|D100", "TXo|D100"),
                               "TX(D100)/T0(D5)": g.skill(s, "TX|D100", "T0|D5"),
                               "T0(D100)/T0(D5)": g.skill(s, "T0|D100", "T0|D5")}
    else:
        for r in K.DA_RULES:
            o = {"rmse_eV": {lab: g.rmse(s, f"{lab}|{r}") for lab in ("H", "G", "T0d", "TXd", "TD", "TA")}}
            o["TD/T0"] = g.skill(s, f"TD|{r}", f"T0d|{r}")
            o["TA/T0"] = g.skill(s, f"TA|{r}", f"T0d|{r}")
            o["TX/T0"] = g.skill(s, f"TXd|{r}", f"T0d|{r}")
            o["TA/TX"] = g.skill(s, f"TA|{r}", f"TXd|{r}")
            out[r] = o
        out["D100 minus D5: TD/T0"] = g.skill_diff(s, "TD|D100", "T0d|D100", "TD|D5", "T0d|D5")
    return out


# ------------------------------------------------------------------------------------------ statements
EXPECT = {"S1": "holds (2 to 5)", "S2": "holds (+0.05 to +0.30)", "S3": "holds", "S4": "holds",
          "S5": "uncertain, leaning to holds", "S6": "uncertain", "S7": "uncertain", "S8": "uncertain", "S9": "holds",
          "S10": "uncertain"}
TEXT = {"S1": "M1: MSE(T0 under D100) / MSE(T0 under D5) lies above 1.5",
        "S2": "M1, D100: the increment lies above zero",
        "S3": "M1: the increment under D100 exceeds the increment under D5 (interval of the difference above zero)",
        "S4": "M2, D100: the increment lies above zero",
        "S5": "M3, D100: the increment lies above zero",
        "S6": "M4, D100: the increment lies above zero",
        "S7": "M1, D100: RTa has a lower MSE than H (interval of the skill above zero)",
        "S8": "M1, D100: TD over T0 lies above zero",
        "S9": "M1, D100, placebo: the interval of TX over T0 contains zero or lies below it",
        "S10": "M1, D5: the MSE of TX lies below that of the frame's own reading O against L5"}


def statements(res):
    M = res["main"]
    c = lambda d: M[f"{d}|L5|common"]["all"]   # noqa: E731
    S = {}

    def lo(v):
        return v["session"][0]

    def put(k, v, holds, note=""):
        S[k] = {"text": TEXT[k], "value": v, "holds": bool(holds), "expected": EXPECT[k], "note": note}

    v = c("M1")["D100"]["staleness T0"]
    put("S1", v, lo(v) > 1.5, "target L5 (Addendum 1 section 8)")
    v = c("M1")["D100"]["TX/T0"]
    put("S2", v, lo(v) > 0)
    v = c("M1")["D100"]["TX/T0 minus under D5"]
    put("S3", v, lo(v) > 0, f"increment under D5: {fmt(c('M1')['D5']['TX/T0'])}")
    v = c("M2")["D100"]["TX/T0"]
    put("S4", v, lo(v) > 0)
    v = c("M3")["D100"]["TX/T0"]
    put("S5", v, lo(v) > 0, "fifths 2-5 pooled")
    v = c("M4")["D100"]["TX/T0"]
    put("S6", v, lo(v) > 0)
    v = c("M1")["D100"]["RTa/H"]
    put("S7", v, lo(v) > 0, f"target L5; RT over H: {fmt(c('M1')['D100']['RT/H'])}")
    v = M["M1|L5|DA"]["all"]["D100"]["TD/T0"]
    put("S8", v, lo(v) > 0, "target L5; DA set, T0 refitted on it")
    v = c("M1")["controls"]["TXp/T0"]
    put("S9", v, lo(v) <= 0)
    v = c("M1")["D5"]["TX/O"]
    put("S10", v, lo(v) > 0, f"RMSE TX {c('M1')['D5']['rmse_eV']['TX']:.1f} eV, O {c('M1')['D5']['rmse_eV']['O']:.1f} eV")
    return S


def fmt(v):
    return f"{v['point']:+.3f}; shot [{v['shot'][0]:+.3f}, {v['shot'][1]:+.3f}]; session [{v['session'][0]:+.3f}, {v['session'][1]:+.3f}]"


def expectation_met(st):
    out = {}
    for k, s in st.items():
        e = s["expected"]
        met = s["holds"] if e.startswith("holds") else (not s["holds"]) if e.startswith("does not hold") else None
        rng = None
        if "(" in e and " to " in e:
            a, b = (float(x) for x in e[e.index("(") + 1:e.index(")")].split(" to "))
            rng = bool(a <= s["value"]["point"] <= b)
        out[k] = {"verdict_as_expected": met, "point_in_expected_range": rng}
    return out


# ------------------------------------------------------------------------------------------ main
def main():
    t_start = time.time()
    keys = ["shot", "y", "L5", "te_t"] + [f"{r}_{k}" for r in K.RULES for k in ("H", "M5", "G", "te_a")]
    R = K.load_rows(keys)
    S = K.load_npz(K.CACHE / "sets.npz")
    us, tab = K.units()
    sess_of = dict(zip(tab["shots"], tab["session"]))
    fifth_of = dict(zip(tab["shots"], tab["fifth"]))
    n = len(R["y"])
    shots_u, unit_s = np.unique(R["shot"], return_inverse=True)
    sess_row = np.array([sess_of.get(int(s), "none") for s in R["shot"]])
    _, unit_ss = np.unique(sess_row, return_inverse=True)
    fifth_row = np.array([fifth_of.get(int(s), -1) for s in R["shot"]])
    base = {r: {"H": R[f"{r}_H"], "G": R[f"{r}_G"], "M5": R[f"{r}_M5"],
                "RT": R[f"{r}_H"] * R["te_t"] / R[f"{r}_te_a"]} for r in K.RULES}
    res = {"registration": K.REG, "addendum": K.ADD, "n_boot": N, "seed": K.SEED,
           "sets": json.loads((K.CACHE / "sets.json").read_text(encoding="utf-8")),
           "shots": json.loads((K.CACHE / "shots.json").read_text(encoding="utf-8"))["info"], "main": {}}
    for d in DESIGNS:
        subs = subs_of(us, d)
        scored = np.isin(R["shot"], np.array(sorted(set().union(*[set(u["test"]) for u in subs[0]]))))
        for X in K.TARGETS:
            for setname in (("common", "DA") if d in ("M1", "M4") else ("common",)):
                rows = np.flatnonzero(scored & S[f"{setname}_{X}"])
                strata = {"all": np.ones(len(rows), bool)}
                if setname == "common" and d in ("M1", "M3"):
                    for qq in range(1 if d == "M3" else 0, K.NF):
                        strata[f"fifth {qq + 1}"] = fifth_row[rows] == qq
                name = f"{d}|{X}|{setname}"
                g = build_group(name, rows, strata, subs, setname, X, d, R, base, unit_s, unit_ss)
                out = {}
                for s in strata:
                    g.finish(s)
                    out[s] = {"counts": g.counts(s), **rule_block(g, s, X, setname)}
                    g.tot.clear()
                res["main"][name] = out
                K.log(f"{name}: rows {len(rows)} done")
    # alpha, beta of RTa
    ab = {}
    for d in DESIGNS:
        uu = [u for sub in subs_of(us, d) for u in sub]
        for X in K.TARGETS:
            for r in K.RULES:
                A = [np.load(K.FITS / f"{u['name']}__RTa__{r}__{X}.npz") for u in uu]
                al = np.array([float(z["alpha"]) for z in A])
                be = np.array([float(z["beta"]) for z in A])
                ab[f"{d}|{X}|{r}"] = {"alpha": {"mean": float(al.mean()), "min": float(al.min()), "max": float(al.max())},
                                      "beta": {"mean": float(be.mean()), "min": float(be.min()), "max": float(be.max())},
                                      "units": len(uu)}
    res["rta_coefficients"] = ab
    # fit sizes
    fs = {}
    for d in DESIGNS:
        uu = [u for sub in subs_of(us, d) for u in sub]
        for L, r, X in (("T0", "D5", "L5"), ("T0", "D100", "L5"), ("T0", "D100", "F")):
            v = [int(np.load(K.FITS / f"{u['name']}__{L}__{r}__{X}.npz")["fit_rows"]) for u in uu]
            fs[f"{d}|{L}|{r}|{X}"] = {"min": min(v), "max": max(v)}
        if d in ("M1", "M4"):
            v = [int(np.load(K.FITS / f"{u['name']}__TD__D100__L5.npz")["fit_rows"]) for u in uu]
            fs[f"{d}|TD|D100|L5"] = {"min": min(v), "max": max(v)}
    res["fit_rows"] = fs
    res["statements"] = statements(res)
    res["expectations"] = expectation_met(res["statements"])
    res["checks"] = {nm: json.loads((K.CHK / f"{nm}.json").read_text(encoding="utf-8"))
                     for nm in ("hold", "hold_diag", "visibility", "units", "thomson")}
    res["runtime_readout_s"] = round(time.time() - t_start, 1)
    K.save_json(K.OUT_JSON, res)
    spec = importlib.util.spec_from_file_location("render76", K.HERE / "render.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    K.save_text(K.OUT_MD, mod.render(res))
    K.log(f"written {K.OUT_JSON.name}, {K.OUT_MD.name}")


if __name__ == "__main__":
    main()
