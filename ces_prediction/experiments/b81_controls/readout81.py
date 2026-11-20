"""B.81 readout (registration sections 4-5, Addendum 1 section 4): the seventeen statements and the reported cells in one
pass into the JSON, then the markdown.  The progress log carries row counts and names only.

    py ces_prediction/experiments/b81_controls/readout81.py
Output: docs/paper/b81_controls.json / .md
"""
from __future__ import annotations

import importlib.util
import json
import time
import warnings

import common81 as B

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = B.C
N = C.N_BOOT
FORMS = ("cluster", "discharge")          # order of the draws in every stream (Addendum 1 section 4)
DESIGNS = ("V1", "V2", "V3", "V4a", "V4b")
COMPS = (("TX/T0", "TX", "T0"), ("TXp/T0", "TXp", "T0"), ("TXo/T0", "TXo", "T0"), ("TX/TXo", "TX", "TXo"),
         ("TE/T0", "TE", "T0"), ("TB/T0", "TB", "T0"))
STRATUM_COMPS = ("TXo/T0", "TXp/T0", "TX/TXo", "TX/T0")

# id, design, rule, comparison, statement text, judgement ("le0": lower end at most zero; "gt0": lower end above zero),
# registered expectation
STATEMENTS = (
    ("C1", "V1", "D650", "TXp/T0", "V1, D650: TXp over T0", "le0", "holds"),
    ("C2", "V1", "S", "TXp/T0", "V1, S: TXp over T0", "le0", "holds"),
    ("C3", "V1", "D650", "TX/TXo", "V1, D650: TX over TXo lies above zero (the present samples add to those of the anchor row)", "gt0", "holds"),
    ("C4", "V1", "S", "TX/TXo", "V1, S: TX over TXo lies above zero", "gt0", "holds"),
    ("C5", "V2", "D650", "TX/TXo", "V2, D650: TX over TXo lies above zero", "gt0", "holds"),
    ("C6", "V3", "D650", "TX/TXo", "V3, D650: TX over TXo lies above zero", "gt0", "uncertain, leaning to holds"),
    ("C7", "V4a", "D650", "TX/TXo", "V4a, D650: TX over TXo lies above zero", "gt0",
     "uncertain (the lead's one-unit point is +0.205 for present, -0.128 for the anchor row, no interval)"),
    ("C8", "V2", "S", "TX/TXo", "V2, S: TX over TXo lies above zero", "gt0", "holds"),
    ("C9", "V3", "S", "TX/TXo", "V3, S: TX over TXo lies above zero", "gt0", "uncertain, leaning to holds"),
    ("E1", "V2", "D650", "TE/T0", "V2, D650: TE over T0 lies above zero", "gt0", "holds"),
    ("E2", "V3", "D650", "TE/T0", "V3, D650: TE over T0 lies above zero", "gt0", "uncertain, leaning to holds"),
    ("E3", "V2", "D650", "TB/T0", "V2, D650: TB over T0 lies above zero", "gt0", "holds"),
    ("E4", "V3", "D650", "TB/T0", "V3, D650: TB over T0 lies above zero", "gt0",
     "uncertain (under V4a it is -0.014 [-0.257, +0.203])"),
    ("E5", "V2", "S", "TE/T0", "V2, S: TE over T0 lies above zero", "gt0", "holds"),
    ("E6", "V3", "S", "TE/T0", "V3, S: TE over T0 lies above zero", "gt0", "uncertain, leaning to holds"),
    ("E7", "V2", "S", "TB/T0", "V2, S: TB over T0 lies above zero", "gt0", "holds"),
    ("E8", "V3", "S", "TB/T0", "V3, S: TB over T0 lies above zero", "gt0", "uncertain"),
)
STREAM_OF = {(d, r, "all", c): sid for sid, d, r, c, *_ in STATEMENTS}


# ------------------------------------------------------------------------------------------ helpers
def q(d):
    d = d[np.isfinite(d)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def draws(rng, nsub, nu):
    """b75_availability/readout.draws with the generator passed in."""
    pick = rng.integers(0, nsub, N) if nsub > 1 else np.zeros(N, np.int64)
    out = []
    for si in range(nsub):
        sel = np.flatnonzero(pick == si)
        out.append((sel, rng.multinomial(nu, np.full(nu, 1.0 / nu), size=len(sel)).astype(np.float64)))
    return out


def subs_of(us, d):
    if d in ("V1", "V2"):
        subs = C.REPEATS if d == "V1" else C.V2_SEEDS
        return [[u for u in us if u["design"] == d and u["sub"] == s] for s in subs]
    if d == "V3":
        return [[u for u in us if u["design"] == "V3"]]
    return [[u for u in us if u["name"] == d]]


def load_est(units, L, r, n):
    a = np.full(n, np.nan)
    for u in units:
        z = np.load(B.est_path(u, L, r))
        a[z["idx"]] = z["est"]
    return a


class Cell:
    """One design x rule x stratum: per-unit SSE sums of every learner in every sub, per form."""

    def __init__(self, rows, unit_d, unit_c, se):
        self.n = len(rows)
        self.nsub = len(next(iter(se.values())))
        self.idx = {}
        for form in FORMS:
            uu = (unit_d if form == "discharge" else unit_c)[rows]
            u, inv = np.unique(uu, return_inverse=True)
            self.idx[form] = (inv, len(u))
        self.sums = {form: {L: [np.bincount(inv, s, nu) for s in se[L]] for L in se} for form, (inv, nu) in self.idx.items()}

    def counts(self):
        return {"rows": self.n, "discharges": self.idx["discharge"][1], "clusters": self.idx["cluster"][1]}

    def rmse(self, L):
        return float(np.mean([np.sqrt(v.sum() / self.n) for v in self.sums["discharge"][L]]))

    def skill(self, stream, A, R_):
        rng = B.rng_for(stream)
        out = {}
        for form in FORMS:
            inv, nu = self.idx[form]
            W = draws(rng, self.nsub, nu)
            da, dr = np.empty(N), np.empty(N)
            for si, (sel, Wm) in enumerate(W):
                da[sel] = Wm @ self.sums[form][A][si]
                dr[sel] = Wm @ self.sums[form][R_][si]
            out[form] = q(1 - da / dr)
            del W
        pts = [1 - a.sum() / b.sum() for a, b in zip(self.sums["discharge"][A], self.sums["discharge"][R_])]
        return {"point": float(np.mean(pts)), "discharge": out["discharge"], "cluster": out["cluster"], "stream": stream}


# ------------------------------------------------------------------------------------------ statements
def statements(main):
    S = {}
    for sid, d, r, comp, text, rule, exp in STATEMENTS:
        v = main[d][r][comp]
        lo = v["cluster"][0]
        holds = lo <= 0 if rule == "le0" else lo > 0
        note = ""
        if sid == "C7":
            w = main["V4b"]["D650"]["TX/TXo"]
            note = f"V4b (fit to 32310): {w['point']:+.3f}; c [{w['cluster'][0]:+.3f}, {w['cluster'][1]:+.3f}]"
        S[sid] = {"text": text, "design": d, "rule": r, "comparison": comp,
                  "judged": "lower end of the cluster interval at most zero" if rule == "le0" else
                  "lower end of the cluster interval above zero",
                  "value": {k: v[k] for k in ("point", "discharge", "cluster")}, "holds": bool(holds), "expected": exp,
                  "note": note}
    return S


def expectation_met(st):
    out = {}
    for k, s in st.items():
        e = s["expected"]
        met = s["holds"] if e.startswith("holds") else (not s["holds"] if e.startswith("does not hold") else None)
        out[k] = {"verdict_as_expected": met}
    return out


# ------------------------------------------------------------------------------------------ main
def main():
    t_start = time.time()
    chk = {nm: json.loads((B.CHK / f"{nm}.json").read_text(encoding="utf-8"))["passes"]
           for nm in ("reuse", "permutation", "anchor", "rows")}
    if not all(chk.values()):
        raise SystemExit(f"FATAL: a check has not passed: {chk}")
    R = B.K75.load_keys(C.CACHE / "rows.npz", ("fidx", "clus", "gi", "tg_L5"))
    sets = B.load_npz(B.K75.CACHE / "sets.npz")
    us, tab, fifth = B.units()
    _, unit_d = np.unique(R["fidx"], return_inverse=True)
    _, unit_c = np.unique(R["clus"], return_inverse=True)
    n = len(R["fidx"])
    y = R["tg_L5"]
    stE, _ = B.ecei_constant()
    anc = {r: B.K75.load_inputs(r)["anc_g"] for r in B.RULES}
    res = {"registration": B.REG, "addendum": B.ADD, "n_boot": N, "seed": B.SEED, "target": B.X,
           "rule_sets": {"D650": "common set (common_L5 of data/.b75/sets.npz)", "S": "S set (S_L5 of data/.b75/sets.npz)"},
           "sources": {"T0, TX": "data/.b75/fits (B.75)", "TE, TB of V1, V4a, V4b": "data/.b75/fits (B.75)",
                       "TE, TB of V2, V3": "data/.b81/fits (B.81)", "TXp, TXo": "data/.b81/fits (B.81)"},
           "main": {}, "ecei_moving": {}, "counts": {}}
    for d in DESIGNS:
        subs = subs_of(us, d)
        scored = np.isin(R["fidx"], np.concatenate([u["test"] for u in subs[0]]))
        res["main"][d] = {}
        for r in B.RULES:
            rows = np.flatnonzero(scored & sets[B.SETOF[r]])
            se = {}
            for L in B.LEARNERS:
                se[L] = []
                for units in subs:
                    a = load_est(units, L, r, n)[rows]
                    if not np.all(np.isfinite(a)):
                        raise SystemExit(f"FATAL: {d} {r} {L}: {int((~np.isfinite(a)).sum())} rows without an estimate")
                    se[L].append((a - y[rows]) ** 2)
            strata = {"all": np.ones(len(rows), bool)}
            if d == "V1":
                strata["ECEI moving"] = ~(stE[R["gi"][rows]] | stE[anc[r][rows]])
            for s, m in strata.items():
                cell = Cell(rows[m], unit_d, unit_c, {L: [v[m] for v in se[L]] for L in se})
                o = {"counts": cell.counts(), "rmse_eV": {L: cell.rmse(L) for L in B.LEARNERS}}
                comps = COMPS if s == "all" else [c for c in COMPS if c[0] in STRATUM_COMPS]
                for nm, A, Rf in comps:
                    stream = STREAM_OF.get((d, r, s, nm), f"{d}|{r}|{s}|{nm}")
                    o[nm] = cell.skill(stream, A, Rf)
                if s == "all":
                    res["main"][d][r] = o
                else:
                    o["rows_left_out"] = int((~m).sum())
                    o["rows_left_out_constant_at_row_t"] = int(stE[R["gi"][rows]].sum())
                    o["rows_left_out_constant_at_anchor_row"] = int(stE[anc[r][rows]].sum())
                    res["ecei_moving"][r] = o
                B.log(f"{d} {r} {s}: rows {int(m.sum()):,} done")
    # ---- counts
    for d in DESIGNS:
        subs = subs_of(us, d)
        c_ = {}
        for r in B.RULES:
            fr, nf_f, nf_t = [], [], []
            for units in subs:
                for u in units:
                    z = np.load(B.path81(u["name"], "TXo", r))
                    fr.append(int(z["fit_rows"]))
                    nf_f.append(int(z["nonfinite_fit_rows"]))
                    nf_t.append(int(z["nonfinite_test_rows"]))
            c_[r] = {**res["main"][d][r]["counts"], "fit_rows_per_unit": {"min": min(fr), "max": max(fr)},
                     "TXo_rows_with_nonfinite_anchor_row_scalars": {"fit_per_unit_max": max(nf_f), "test_per_unit_max": max(nf_t),
                                                                    "test_total": int(sum(nf_t))}}
        res["counts"][d] = c_
    res["designs"] = {"d1_discharges": int(len(tab["fidx"])), "clusters_d1": int(len(np.unique(tab["clus"]))),
                      "units": [{"name": u["name"], "fit": int(len(u["fit"])), "test": int(len(u["test"]))} for u in us]}
    res["statements"] = statements(res["main"])
    res["expectations"] = expectation_met(res["statements"])
    res["checks"] = chk
    res["runtime_readout_s"] = round(time.time() - t_start, 1)
    B.save_json(B.OUT_JSON, res)
    B.save_text(B.OUT_MD, load_render().render(res))
    B.log(f"written {B.OUT_JSON.name}, {B.OUT_MD.name}")


def load_render():
    spec = importlib.util.spec_from_file_location("render81", B.HERE / "render81.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    main()
