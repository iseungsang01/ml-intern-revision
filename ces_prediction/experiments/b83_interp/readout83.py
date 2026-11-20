"""B.83 readout (registration sections 3-4, Addendum 1 sections 2-6): per design the table of hold, PCHIP, causal GP,
offline GP, T0 and TX on the reference rows, the pairs T0 and TX over each of the four, the skills over hold and causal GP
on all scored rows, and the 14 statements.  Refuses to run unless the four checks have passed.  The progress log carries
names and row counts only.

    py ces_prediction/experiments/b83_interp/readout83.py
Output: docs/paper/b83_interp.json / .md
"""
from __future__ import annotations

import json
import time
import warnings

import common83 as M

import boot82 as BT  # noqa: E402  (b82_localize; draws and weighted quantiles)
import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = M.C
N = M.N_BOOT
FORMS = ("cluster", "discharge")
QS = (0.5, 0.75, 0.9)
MEASURES = ("RMSE", "MAE", "median absolute error", "p75 absolute error", "p90 absolute error",
            "median relative error (value above 100 eV)", "share within 10 %", "share within 20 %")
RATIOS = (("MAE ratio", "MAE"), ("median absolute error ratio", "median absolute error"),
          ("p75 absolute error ratio", "p75 absolute error"), ("p90 absolute error ratio", "p90 absolute error"),
          ("median relative error ratio", "median relative error (value above 100 eV)"))
DIFFS = (("share within 10 % difference", "share within 10 %"), ("share within 20 % difference", "share within 20 %"))
PAIR_MEASURES = ("MSE skill",) + tuple(r[0] for r in RATIOS) + tuple(x[0] for x in DIFFS)
SHORT = {"H": "hold", "P": "PCHIP", "G": "causal GP", "O": "offline GP", "T0": "T0", "TX": "TX"}
PAIRS = tuple((a, r) for a in ("T0", "TX") for r in M.REFS_OF_PAIR)

# id -> (design, A, R, measure, text, rule, expectation); rule: gt0 lower > 0; ge0hi upper >= 0; lt1 upper < 1
STATEMENTS = {
    "I1": ("V1", "TX", "P", "MSE skill", "V1: TX over PCHIP lies above zero", "gt0", "holds"),
    "I2": ("V1", "TX", "O", "MSE skill", "V1: TX over the offline GP lies above zero", "gt0", "holds"),
    "I3": ("V2", "TX", "P", "MSE skill", "V2: TX over PCHIP lies above zero", "gt0", "holds"),
    "I4": ("V2", "TX", "O", "MSE skill", "V2: TX over the offline GP lies above zero", "gt0", "uncertain, leaning to holds"),
    "I5": ("V3", "TX", "P", "MSE skill", "V3: TX over PCHIP lies above zero", "gt0", "holds"),
    "I6": ("V3", "TX", "O", "MSE skill", "V3: TX over the offline GP lies above zero", "gt0", "uncertain"),
    "I7": ("V4a", "TX", "P", "MSE skill", "V4a: TX over PCHIP lies above zero", "gt0", "holds"),
    "I8": ("V4a", "TX", "O", "MSE skill", "V4a: TX over the offline GP lies above zero", "gt0", "uncertain"),
    "I9": ("V1", "T0", "P", "MSE skill", "V1: T0 (history only) over PCHIP lies above zero", "gt0", "holds"),
    "I10": ("V1", "T0", "O", "MSE skill",
            "V1: T0 over the offline GP has an interval that contains zero or lies above it (the causal history-only "
            "estimate is not worse than the interpolation that waits)", "ge0hi", "holds"),
    "I11": ("V1", "TX", "O", "median absolute error ratio",
            "V1: the ratio of the median absolute errors, TX over the offline GP, lies below one", "lt1",
            "does not hold (in B.70 the offline GP has the lower median)"),
    "I12": ("V1", "TX", "P", "median absolute error ratio",
            "V1: the ratio of the median absolute errors, TX over PCHIP, lies below one", "lt1", "does not hold"),
    "I13": ("V1", "TX", "O", "MAE ratio", "V1: the ratio of the mean absolute errors, TX over the offline GP, lies below one",
            "lt1", "uncertain"),
    "I14": ("V1", "TX", "O", "p90 absolute error ratio",
            "V1: the ratio of the 90th percentiles of the absolute error, TX over the offline GP, lies below one", "lt1",
            "holds"),
}
JUDGED = {"gt0": "lower end of the cluster interval above zero",
          "ge0hi": "lower end of the cluster interval above zero, or the interval contains zero (upper end at least zero)",
          "lt1": "upper end of the cluster interval below one"}


def verdict(v, rule):
    lo, hi = v["cluster"]
    return {"gt0": lo > 0, "ge0hi": hi >= 0, "lt1": hi < 1}[rule]


class Cell:
    """Rows of one design (a row set), the units of both forms, per-sub absolute, squared and relative errors."""

    def __init__(self, rows, subs, R, refs, unit_d, unit_c, ests):
        self.rows = rows
        self.nsub = len(subs)
        y = R["tg_F"][rows]
        self.y = y
        self.U = BT.Units(rows, unit_d, unit_c)
        self.rel_ok = y > 100.0
        self.shared = {"H", "P", "G", "O"}
        self.e, self.se, self.rel = {}, {}, {}
        for L in ests:
            vs = []
            for sub in (subs if L not in self.shared else subs[:1]):
                a = M.estimates(sub, R, refs)[L][rows]
                if not np.all(np.isfinite(a)):
                    raise SystemExit(f"FATAL: {L}: rows without an estimate")
                vs.append(a)
            self.e[L] = [np.abs(a - y) for a in vs]
            self.se[L] = [(a - y) ** 2 for a in vs]
            self.rel[L] = [x / y for x in self.e[L]]
        self._wq = {}

    def k(self, L, si):
        return 0 if L in self.shared else si

    def wq(self, form, L, si, kind):
        sk = self.k(L, si)
        key = (form, L, sk, kind)
        if key not in self._wq:
            inv, nu = self.U.idx[form]
            if kind == "e":
                self._wq[key] = BT.WQ(self.e[L][sk], inv, nu)
            else:
                m = self.rel_ok
                self._wq[key] = BT.WQ(self.rel[L][sk][m], inv[m], nu)
        return self._wq[key]

    def stats(self, form, si, Wm, ests, pairs, robust):
        U = self.U
        inv, nu = U.idx[form]
        T = Wm @ np.bincount(inv, None, nu).astype(np.float64)
        out = {}
        for L in ests:
            sk = self.k(L, si)
            e, se, rel = self.e[L][sk], self.se[L][sk], self.rel[L][sk]
            sse = Wm @ U.sums(form, se)
            out[(L, "SSE")] = sse
            out[(L, "RMSE")] = np.sqrt(sse / T)
            if robust:
                out[(L, "MAE")] = (Wm @ U.sums(form, e)) / T
                out[(L, "share within 10 %")] = (Wm @ U.sums(form, (rel <= 0.10).astype(np.float64))) / T
                out[(L, "share within 20 %")] = (Wm @ U.sums(form, (rel <= 0.20).astype(np.float64))) / T
                qv = self.wq(form, L, si, "e").draws(Wm, QS)
                out[(L, "median absolute error")] = qv[:, 0]
                out[(L, "p75 absolute error")] = qv[:, 1]
                out[(L, "p90 absolute error")] = qv[:, 2]
                out[(L, "median relative error (value above 100 eV)")] = self.wq(form, L, si, "rel").draws(Wm, (0.5,))[:, 0]
        for A, Rf in pairs:
            nm = f"{A} over {Rf}"
            out[(nm, "MSE skill")] = 1 - out[(A, "SSE")] / out[(Rf, "SSE")]
            if robust:
                for rn, ms in RATIOS:
                    out[(nm, rn)] = out[(A, ms)] / out[(Rf, ms)]
                for dn, ms in DIFFS:
                    out[(nm, dn)] = out[(A, ms)] - out[(Rf, ms)]
        return out

    def run(self, stream, ests, pairs, robust):
        rng = M.rng_for(stream)
        vals = {}
        for form in FORMS:
            _, nu = self.U.idx[form]
            W = BT.draws(rng, self.nsub, nu)
            acc = {}
            for si, (sel, Wm) in enumerate(W):
                for k_, v in self.stats(form, si, Wm, ests, pairs, robust).items():
                    acc.setdefault(k_, np.empty(N))[sel] = v
            vals[form] = acc
            del W
        pts = {}
        for si in range(self.nsub):
            _, nu = self.U.idx["discharge"]
            for k_, v in self.stats("discharge", si, np.ones((1, nu)), ests, pairs, robust).items():
                pts.setdefault(k_, []).append(float(v[0]))
        out = {}
        for k_ in pts:
            if k_[1] == "SSE":
                continue
            o = {"point": float(np.mean(pts[k_])), "cluster": BT.q(vals["cluster"][k_]),
                 "discharge": BT.q(vals["discharge"][k_]), "stream": stream}
            if self.nsub > 1:
                o["per_split"] = pts[k_]
            out[k_] = o
        return out


def main():
    ok, failed = M.checks_passed()
    if not ok:
        raise SystemExit(f"FATAL: check {failed} has not passed; no readout")
    t_start = time.time()
    R = C.rows()
    refs = M.load_npz(M.REFS)
    us, _, _ = M.units()
    _, unit_d = np.unique(R["fidx"], return_inverse=True)
    _, unit_c = np.unique(R["clus"], return_inverse=True)
    res = {"registration": M.REG, "addendum": M.ADD, "n_boot": N, "seed": M.SEED, "target": "F (the reported value of row t)",
           "estimator_names": {k: M.EST_LABEL[k] for k in M.EST},
           "sources": {"H, G": "data/.b73/rows.npz (hold, gp; from data/.b64/base.npz)",
                       "T0, TX": "data/.b73/trees/{unit}__{T0,TX}__F.npz (B.73)",
                       "P, O": "data/.b83/refs.npz (baselines_interpolation.predict_pchip / predict_gp on B.70's dataset "
                               "arrays, row t left out; equal to B.70 bit for bit, check 'references')"},
           "measures": list(MEASURES), "pair_measures": list(PAIR_MEASURES),
           "counts": {}, "estimators": {}, "pairs": {}, "all_rows": {}, "statements": {}}
    for d in M.DESIGNS:
        subs, scored = M.assemble(d, R, us)
        ref = scored & refs["formed"]
        rows = np.flatnonzero(ref)
        cell = Cell(rows, subs, R, refs, unit_d, unit_c, M.EST)
        allv = cell.run(f"table|{d}", M.EST, PAIRS, True)
        res["counts"][d] = {"scored_rows": int(scored.sum()), "reference_rows": int(len(rows)),
                            "rows_left_out_missing_reference": int(scored.sum() - len(rows)),
                            "discharges": cell.U.count("discharge"), "clusters": cell.U.count("cluster"),
                            "rows_value_above_100_eV": int(cell.rel_ok.sum()), "splits": [s["name"] for s in subs]}
        res["estimators"][d] = {L: {ms: allv[(L, ms)] for ms in MEASURES} for L in M.EST}
        res["pairs"][d] = {f"{SHORT[A]} over {SHORT[Rf]}": {pm: allv[(f'{A} over {Rf}', pm)] for pm in PAIR_MEASURES}
                           for A, Rf in PAIRS}
        M.log(f"{d}: table and pairs on {len(rows):,} reference rows ({time.time() - t_start:.0f} s)")
        # statements of this design, each on its own stream
        for sid, (sd, A, Rf, meas, text, rule, exp) in STATEMENTS.items():
            if sd != d:
                continue
            robust = meas != "MSE skill"
            sv = cell.run(sid, (A, Rf), ((A, Rf),), robust)[(f"{A} over {Rf}", meas)]
            res["statements"][sid] = {"text": text, "design": d, "comparison": f"{SHORT[A]} over {SHORT[Rf]}",
                                      "measure": meas, "judged": JUDGED[rule],
                                      "value": {k_: sv[k_] for k_ in ("point", "discharge", "cluster")},
                                      "holds": bool(verdict(sv, rule)), "expected": exp}
            M.log(f"statement {sid} done")
        del cell
        # skills over hold and causal GP on all scored rows (Addendum 1 section 2)
        rows_all = np.flatnonzero(scored)
        ca = Cell(rows_all, subs, R, refs, unit_d, unit_c, ("H", "G", "T0", "TX"))
        pa = tuple((a, r) for a in ("T0", "TX") for r in ("H", "G"))
        av = ca.run(f"allrows|{d}", ("H", "G", "T0", "TX"), pa, False)
        res["all_rows"][d] = {"counts": {"rows": int(len(rows_all)), "discharges": ca.U.count("discharge"),
                                         "clusters": ca.U.count("cluster")},
                              "RMSE": {L: av[(L, "RMSE")] for L in ("H", "G", "T0", "TX")},
                              "pairs": {f"{SHORT[A]} over {SHORT[Rf]}": {"MSE skill": av[(f'{A} over {Rf}', 'MSE skill')]}
                                        for A, Rf in pa}}
        del ca
        M.log(f"{d}: all scored rows ({len(rows_all):,}) done")
    res["refs_info"] = json.loads((M.CACHE / "refs_info.json").read_text(encoding="utf-8"))
    res["checks"] = {nm: json.loads((M.CHK / f"{nm}.json").read_text(encoding="utf-8"))["passes"] for nm in M.CHECK_NAMES}
    res["runtime_readout_s"] = round(time.time() - t_start, 1)
    M.save_json(M.OUT_JSON, res)
    import render83
    M.save_text(M.OUT_MD, render83.render(res))
    M.log(f"written {M.OUT_JSON.name}, {M.OUT_MD.name}")


if __name__ == "__main__":
    main()
