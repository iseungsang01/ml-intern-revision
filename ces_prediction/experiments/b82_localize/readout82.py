"""B.82 readout (registration sections 3-7, Addendum 1 sections 6-8): parts A to E and the 23 statements in one pass into
the JSON, then the markdown.  Refuses to run unless the five checks have passed.  The progress log carries names and row
counts only.

    py ces_prediction/experiments/b82_localize/readout82.py
Output: docs/paper/b82_localize.json / .md
"""
from __future__ import annotations

import importlib.util
import json
import time
import warnings

import boot82 as BT
import common82 as M

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C, K75 = M.C, M.K75
N = BT.N
DESIGNS = ("V1", "V2", "V3", "V4a", "V4b")
CHECKS = ("reuse", "tau", "classes", "swap", "mirnov")
BLOCK_LAST = 2                                     # rows.npz blk: 0 up to 31991, 1 32002-32310, 2 from 32312

# ------------------------------------------------------------------------------------------ statements
# id -> (part, location, comparison or measure, text, judgement, expectation)
#   judgement: gt0 lower end > 0; le0 lower end <= 0; lt1 upper end < 1; gt05 lower end > 0.5
STATEMENTS = {
    "A1": ("A", ("V1", "D650", "all", "tau", "above 300 ms"), "TX/T0",
           "V1, D650, rows with tau above 300 ms: TX over T0 lies above zero", "gt0", "holds"),
    "A2": ("A", ("V1", "S", "all", "tau", "above 300 ms"), "TX/T0",
           "V1, S, rows with tau above 300 ms: TX over T0 lies above zero", "gt0", "holds"),
    "A3": ("A", ("V3", "D650", "all", "tau", "above 300 ms"), "TX/T0",
           "V3, D650, rows with tau above 300 ms: TX over T0 lies above zero", "gt0", "uncertain, leaning to holds"),
    "A4": ("A", ("V1", "D650", "all", "tau", "no notch before"), "TX/T0",
           "V1, D650, rows with no notch before: TX over T0 lies above zero", "gt0", "holds"),
    "A5": ("A", ("V1", "S", "all", "tau", "up to 100 ms minus above 300 ms"), "TX/T0",
           "V1, S: TX over T0 on rows with tau up to 100 ms minus that on rows with tau above 300 ms lies above zero "
           "(the gain is larger just after a notch)", "gt0", "uncertain, leaning to holds"),
    "A6": ("A", ("V1", "D650", "all", "c", "lowest third"), "TX/T0",
           "V1, D650, lowest third of c: TX over T0 lies above zero", "gt0", "uncertain"),
    "A7": ("A", ("V1", "D650", "all", "c", "highest third minus lowest third"), "TX/T0",
           "V1, D650: TX over T0 in the highest third of c minus that in the lowest third lies above zero", "gt0", "holds"),
    "A8": ("A", ("V1", "S", "all", "tau", "above 300 ms"), "TXN/TN",
           "V1, S, rows with tau above 300 ms: TXN over TN lies above zero (beam scalars given)", "gt0", "holds"),
    "B1": ("B", ("V1", "D650"), "MAE ratio", "V1, D650: MAE(TX) / MAE(T0) lies below one", "lt1", "holds"),
    "B2": ("B", ("V1", "D650"), "median absolute error ratio",
           "V1, D650: the ratio of the median absolute errors lies below one", "lt1", "uncertain"),
    "B3": ("B", ("V3", "D650"), "MAE ratio", "V3, D650: MAE ratio below one", "lt1", "uncertain, leaning to holds"),
    "B4": ("B", ("V1", "S"), "MAE ratio", "V1, S: MAE ratio below one", "lt1", "holds"),
    "B5": ("B", ("V1", "D650"), "share within 10 % difference",
           "V1, D650: share of rows within 10 % of the level, TX minus T0, lies above zero", "gt0",
           "uncertain, leaning to holds"),
    "B6": ("B", ("V1", "D650"), "share of discharges with positive skill",
           "V1, D650: the share of discharges whose own skill of TX over T0 is positive lies above one half", "gt05", "holds"),
    "B7": ("B", ("V4a", "D650"), "MAE ratio", "V4a, D650: MAE ratio below one", "lt1", "uncertain"),
    "B8": ("B", ("V1", "D650"), "p90 absolute error ratio",
           "V1, D650: the ratio of the 90th percentiles of the absolute error lies below one", "lt1", "holds"),
    "C1": ("C", ("V1", "D650"), "TM/T0",
           "V1, D650: TM over T0 has an interval that contains zero or lies below it (a diagnostic that is not at the "
           "radius adds nothing)", "le0", "uncertain"),
    "C2": ("C", ("V1", "D650"), "TX/TM", "V1, D650: TX over TM lies above zero", "gt0", "holds"),
    "C3": ("C", ("V3", "D650"), "TX/TM", "V3, D650: TX over TM lies above zero", "gt0", "uncertain, leaning to holds"),
    "C4": ("C", ("V1", "D650"), "TXs/T0", "V1, D650: TXs over T0 has an interval that contains zero or lies below it",
           "le0", "holds"),
    "C5": ("C", ("V1", "S"), "TXs/T0", "V1, S: TXs over T0 has an interval that contains zero or lies below it", "le0",
           "holds"),
    "C6": ("C", ("V1", "D650"), "TXM/TX",
           "V1, D650: TXM over TX has an interval that contains zero or lies below it (Mirnov adds nothing on top of the "
           "fast diagnostics)", "le0", "uncertain, leaning to holds"),
    "C7": ("C", ("V1", "S"), "TX/TM", "V1, S: TX over TM lies above zero", "gt0", "holds"),
}
JUDGED = {"gt0": "lower end of the cluster interval above zero", "le0": "lower end of the cluster interval at most zero",
          "lt1": "upper end of the cluster interval below one", "gt05": "lower end of the cluster interval above 0.5"}


def verdict(v, rule):
    lo, hi = v["cluster"]
    return {"gt0": lo > 0, "le0": lo <= 0, "lt1": hi < 1, "gt05": lo > 0.5}[rule]


def stream_of(part, loc, comp):
    for sid, (p, l, c, *_) in STATEMENTS.items():
        if p == part and tuple(l) == tuple(loc) and c == comp:
            return sid
    return f"{part}|" + "|".join(loc) + f"|{comp}"


# ------------------------------------------------------------------------------------------ data
class Data:
    def __init__(self):
        self.R = K75.load_keys(C.CACHE / "rows.npz", ("fidx", "clus", "gi", "tg_L5", "y", "blk", "shot"))
        self.sets75 = M.load_npz(K75.CACHE / "sets.npz")
        self.sets79 = M.load_npz(C.DATA / ".b79" / "sets.npz")
        self.us, self.tab, self.fifth = M.units()
        _, self.unit_d = np.unique(self.R["fidx"], return_inverse=True)
        _, self.unit_c = np.unique(self.R["clus"], return_inverse=True)
        self.n = len(self.R["fidx"])
        self.y = self.R["tg_L5"]
        t = M.load_npz(M.CACHE / "tau.npz")
        self.tau, self.taucls = t["tau"], t["cls"]
        self.cc = M.load_npz(M.CACHE / "cclass.npz")

    def subs(self, d):
        us = self.us
        if d in ("V1", "V2"):
            subs = C.REPEATS if d == "V1" else C.V2_SEEDS
            return [[u for u in us if u["design"] == d and u["sub"] == s] for s in subs]
        if d == "V3":
            return [[u for u in us if u["design"] == "V3"]]
        return [[u for u in us if u["name"] == d]]

    def scored(self, subs):
        return np.isin(self.R["fidx"], np.concatenate([u["test"] for u in subs[0]]))

    def est_path(self, u, L, r):
        if L in ("TN", "TXN"):
            return M.B79FITS / f"{u['name']}__{L}__{r}__L5.npz"
        if L in M.PART_C_LEARNERS or "_seed" in L:
            return M.path82(u["name"], L, r)
        if L in ("TE", "TB") and u["design"] in ("V2", "V3"):
            return M.B81FITS / f"{u['name']}__{L}__{r}__L5.npz"
        return M.B75FITS / f"{u['name']}__{L}__{r}__L5.npz"

    def est(self, units, L, r):
        a = np.full(self.n, np.nan)
        for u in units:
            z = np.load(self.est_path(u, L, r))
            a[z["idx"]] = z["est"]
        return a

    def se(self, subs, L, r, rows):
        out = []
        for units in subs:
            a = self.est(units, L, r)[rows]
            if not np.all(np.isfinite(a)):
                raise SystemExit(f"FATAL: {L} {r}: {int((~np.isfinite(a)).sum())} rows without an estimate")
            out.append((a - self.y[rows]) ** 2)
        return out

    def c_classes(self, subs, r, rows):
        """Class of c of each row, per sub (the unit of the sub that holds the row's discharge)."""
        out = []
        for units in subs:
            cls = np.full(len(rows), -1, np.int64)
            for u in units:
                m = np.isin(self.R["fidx"][rows], u["test"])
                cls[m] = self.cc[f"{u['name']}|{r}|cls"][rows[m]]
            if np.any(cls < 0):
                raise SystemExit("FATAL: a row without a class of c")
            out.append(cls)
        return out


# ------------------------------------------------------------------------------------------ part A
FAMILIES = (("B75", (("TX/T0", "TX", "T0"), ("TE/T0", "TE", "T0"), ("TB/T0", "TB", "T0")), ("T0", "TX", "TE", "TB")),
            ("B79", (("TXN/TN", "TXN", "TN"),), ("TN", "TXN")))
DIFFS = {"tau": ("up to 100 ms minus above 300 ms", 0, 2), "c": ("highest third minus lowest third", 2, 0)}
CLASSES = {"tau": M.TAU_CLASSES, "c": M.C_CLASSES}


def part_a_cell(D, d, r, fam, block, points_only=False):
    """One design x rule x family x block: per stratification and class, counts, RMSE, the comparisons (and the
    difference of two classes).  points_only: the class sums and pooled points for check 'classes', no draw."""
    _, comps, learners = next(f for f in FAMILIES if f[0] == fam)
    subs = D.subs(d)
    sets = D.sets75 if fam == "B75" else D.sets79
    rows = np.flatnonzero(D.scored(subs) & sets[M.SETOF[r]])
    if block == "block from 32312":
        rows = rows[D.R["blk"][rows] == BLOCK_LAST]
    se = {L: D.se(subs, L, r, rows) for L in learners}
    nsub = len(subs)
    strat = {"tau": [D.taucls[rows]] * nsub, "c": D.c_classes(subs, r, rows)}
    U = BT.Units(rows, D.unit_d, D.unit_c)
    out = {"counts": {"rows": len(rows), "discharges": U.count("discharge"), "clusters": U.count("cluster")}}
    for sname, cls in strat.items():
        o = {}
        for k, cname in enumerate(CLASSES[sname]):
            masks = [c == k for c in cls]
            nrow = [int(m.sum()) for m in masks]
            cnt = {"rows": nrow[0] if len(set(nrow)) == 1 else float(np.mean(nrow)),
                   "discharges": [U.count("discharge", m) for m in masks], "clusters": [U.count("cluster", m) for m in masks]}
            if nsub == 1:
                cnt["discharges"], cnt["clusters"] = cnt["discharges"][0], cnt["clusters"][0]
            else:
                cnt["rows_per_split"] = nrow
            co = {"counts": cnt}
            if min(nrow) == 0:
                co["empty"] = True
                o[cname] = co
                continue
            if points_only:
                # check 'classes': only the class sums, to be added up; nothing per class is formed or kept
                co["sse_per_split"] = {L: [float(se[L][si][masks[si]].sum()) for si in range(nsub)] for L in learners}
                o[cname] = co
                continue
            co["rmse_eV"] = {L: float(np.mean([np.sqrt(se[L][si][masks[si]].sum() / nrow[si]) for si in range(nsub)]))
                             for L in learners}
            for nm, A, Rf in comps:
                def vecs(form, si, A=A, Rf=Rf, masks=masks):
                    return {"A": U.sums(form, se[A][si] * masks[si]), "R": U.sums(form, se[Rf][si] * masks[si])}
                co[nm] = BT.boot(stream_of("A", (d, r, block, sname, cname), nm), U, nsub, vecs,
                                 lambda v: 1 - v["A"] / v["R"])
            o[cname] = co
        dn, k1, k2 = DIFFS[sname]
        if not points_only and all(not o[CLASSES[sname][k]].get("empty") for k in (k1, k2)):
            dd = {}
            for nm, A, Rf in comps:
                m1 = [c == k1 for c in cls]
                m2 = [c == k2 for c in cls]

                def vecs(form, si, A=A, Rf=Rf, m1=m1, m2=m2):
                    return {"A1": U.sums(form, se[A][si] * m1[si]), "R1": U.sums(form, se[Rf][si] * m1[si]),
                            "A2": U.sums(form, se[A][si] * m2[si]), "R2": U.sums(form, se[Rf][si] * m2[si])}
                dd[nm] = BT.boot(stream_of("A", (d, r, block, sname, dn), nm), U, nsub, vecs,
                                 lambda v: (1 - v["A1"] / v["R1"]) - (1 - v["A2"] / v["R2"]))
            o[dn] = dd
        out[sname] = o
    out["pooled_points"] = {nm: float(np.mean([1 - se[A][si].sum() / se[Rf][si].sum() for si in range(nsub)]))
                            for nm, A, Rf in comps}
    out["pooled_sse_per_split"] = {L: [float(x.sum()) for x in se[L]] for L in learners}
    return out


def part_a(D, points_only=False):
    res = {}
    for d in DESIGNS:
        res[d] = {}
        for r in M.RULES:
            res[d][r] = {}
            for block in (("all", "block from 32312") if d == "V1" else ("all",)):
                cell = {}
                for fam, *_ in FAMILIES:
                    cell[fam] = part_a_cell(D, d, r, fam, block, points_only)
                # merge the two families: the comparisons of both under each class
                merged = {"counts": cell["B75"]["counts"], "counts_TN_TXN_rows": cell["B79"]["counts"],
                          "pooled_points": {**cell["B75"]["pooled_points"], **cell["B79"]["pooled_points"]}}
                if points_only:
                    merged["pooled_sse_per_split"] = {**cell["B75"]["pooled_sse_per_split"], **cell["B79"]["pooled_sse_per_split"]}
                for sname in ("tau", "c"):
                    s = {}
                    for key in list(cell["B75"][sname].keys()):
                        a, b = cell["B75"][sname][key], cell["B79"][sname].get(key, {})
                        if key in CLASSES[sname]:
                            m = dict(a)
                            m["counts_TN_TXN_rows"] = b.get("counts")
                            if "rmse_eV" in b:
                                m["rmse_eV"] = {**m.get("rmse_eV", {}), **b["rmse_eV"]}
                            if "sse_per_split" in b:
                                m["sse_per_split"] = {**m.get("sse_per_split", {}), **b["sse_per_split"]}
                            for kk, vv in b.items():
                                if kk.endswith("/TN"):
                                    m[kk] = vv
                        else:
                            m = {**a, **b}
                        s[key] = m
                    merged[sname] = s
                res[d][r][block] = merged
                M.log(f"part A {d} {r} {block}: rows {merged['counts']['rows']:,}")
    return res


# ------------------------------------------------------------------------------------------ part B
EST_B = ("H", "G", "T0", "TX", "O")
QS = (0.5, 0.75, 0.9)
MEASURES = ("MAE", "median absolute error", "p75 absolute error", "p90 absolute error",
            "median relative error (L5 above 100 eV)", "share within 10 %", "share within 20 %")
RATIOS = (("MAE ratio", "MAE"), ("median absolute error ratio", "median absolute error"),
          ("p75 absolute error ratio", "p75 absolute error"), ("p90 absolute error ratio", "p90 absolute error"),
          ("median relative error ratio", "median relative error (L5 above 100 eV)"))
DIFF_SH = (("share within 10 % difference", "share within 10 %"), ("share within 20 % difference", "share within 20 %"))
DISC_MIN = 50


class BCell:
    """Rows, units and the per-estimator material of one design x rule of part B."""

    def __init__(self, D, d, r, base):
        self.subs = D.subs(d)
        self.nsub = len(self.subs)
        rows = np.flatnonzero(D.scored(self.subs) & D.sets75[M.SETOF[r]])
        self.rows = rows
        y = D.y[rows]
        self.U = BT.Units(rows, D.unit_d, D.unit_c)
        self.rel_ok = y > 100.0
        est = {"H": [base[r]["H"][rows]] * self.nsub, "G": [base[r]["G"][rows]] * self.nsub,
               "O": [D.R["y"][rows]] * self.nsub}
        for L in ("T0", "TX"):
            est[L] = [D.est(units, L, r)[rows] for units in self.subs]
        for L, v in est.items():
            for a in v:
                if not np.all(np.isfinite(a)):
                    raise SystemExit(f"FATAL: part B {d} {r} {L}: rows without an estimate")
        self.e = {L: [np.abs(a - y) for a in v] for L, v in est.items()}
        self.se = {L: [(a - y) ** 2 for a in v] for L, v in est.items()}
        self.rel = {L: [x / y for x in v] for L, v in self.e.items()}
        self.y = y
        fid = D.R["fidx"][rows]
        self.fid = fid
        uf, inv_f, cnt_f = np.unique(fid, return_inverse=True, return_counts=True)
        self.disc_ok = cnt_f >= DISC_MIN
        self.disc_files = uf
        self.inv_f = inv_f
        # unit (per form) of each eligible discharge
        self.disc_unit = {}
        for form in BT.FORMS:
            inv, nu = self.U.idx[form]
            first = np.zeros(len(uf), np.int64)
            first[inv_f[::-1]] = np.arange(len(rows))[::-1]
            self.disc_unit[form] = inv[first]
        self._wq = {}

    def wq(self, form, L, si, kind):
        """WQ of |error| (kind 'e') or of the relative error on rows with L5 above 100 eV (kind 'rel'); H, G, O are the
        same in every split."""
        sk = 0 if L in ("H", "G", "O") else si
        key = (form, L, sk, kind)
        if key not in self._wq:
            inv, nu = self.U.idx[form]
            if kind == "e":
                self._wq[key] = BT.WQ(self.e[L][sk], inv, nu)
            else:
                m = self.rel_ok
                self._wq[key] = BT.WQ(self.rel[L][sk][m], inv[m], nu)
        return self._wq[key]

    def disc_skill(self, si):
        a = np.bincount(self.inv_f, self.se["TX"][si], len(self.disc_files))
        b = np.bincount(self.inv_f, self.se["T0"][si], len(self.disc_files))
        return (1 - a / b)[self.disc_ok]

    def stats(self, form, si, Wm, need):
        """Every measure for the draws Wm (D x nu) of split si; Wm = ones gives the point."""
        U = self.U
        inv, nu = U.idx[form]
        T = Wm @ np.bincount(inv, None, nu).astype(np.float64)
        out = {}
        for L in need:
            if L == "disc":
                continue
            e, rel = self.e[L][si], self.rel[L][si]
            out[(L, "MAE")] = (Wm @ U.sums(form, e)) / T
            out[(L, "share within 10 %")] = (Wm @ U.sums(form, (rel <= 0.10).astype(np.float64))) / T
            out[(L, "share within 20 %")] = (Wm @ U.sums(form, (rel <= 0.20).astype(np.float64))) / T
            qv = self.wq(form, L, si, "e").draws(Wm, QS)
            out[(L, "median absolute error")] = qv[:, 0]
            out[(L, "p75 absolute error")] = qv[:, 1]
            out[(L, "p90 absolute error")] = qv[:, 2]
            out[(L, "median relative error (L5 above 100 eV)")] = self.wq(form, L, si, "rel").draws(Wm, (0.5,))[:, 0]
        if "T0" in need and "TX" in need:
            for nm, ms in RATIOS:
                out[("TX over T0", nm)] = out[("TX", ms)] / out[("T0", ms)]
            for nm, ms in DIFF_SH:
                out[("TX over T0", nm)] = out[("TX", ms)] - out[("T0", ms)]
        if "disc" in need:
            sk = self.disc_skill(si)
            du = self.disc_unit[form][self.disc_ok]
            w = Wm[:, du]
            out[("per discharge", "share of discharges with positive skill")] = (w @ (sk > 0).astype(np.float64)) / w.sum(1)
            wq = BT.WQ(sk, du, nu, K=64)
            qd = wq.draws(Wm, (0.25, 0.5, 0.75))
            for j, nm in enumerate(("lower quartile of the skill", "median of the skill", "upper quartile of the skill")):
                out[("per discharge", nm)] = qd[:, j]
        return out

    def run(self, stream, need):
        rng = M.rng_for(stream)
        vals = {}
        for form in BT.FORMS:
            _, nu = self.U.idx[form]
            W = BT.draws(rng, self.nsub, nu)
            acc = {}
            for si, (sel, Wm) in enumerate(W):
                st = self.stats(form, si, Wm, need)
                for k, v in st.items():
                    acc.setdefault(k, np.empty(N))[sel] = v
            vals[form] = acc
            del W
        pts = {}
        for si in range(self.nsub):
            _, nu = self.U.idx["discharge"]
            st = self.stats("discharge", si, np.ones((1, nu)), need)
            for k, v in st.items():
                pts.setdefault(k, []).append(float(v[0]))
        out = {}
        for k in pts:
            o = {"point": float(np.mean(pts[k])), "discharge": BT.q(vals["discharge"][k]), "cluster": BT.q(vals["cluster"][k]),
                 "stream": stream}
            if self.nsub > 1:
                o["per_split"] = pts[k]
            out[k] = o
        return out


def part_b(D):
    base = {}
    for r in ("D10", "D650", "S"):
        I = K75.load_inputs(r)
        base[r] = {"H": I["H"], "G": I["G"]}
    res, wqcheck = {}, None
    stmt_of = {(l, c): sid for sid, (p, l, c, *_) in STATEMENTS.items() if p == "B"}
    for d in DESIGNS:
        res[d] = {}
        for r in ("D10", "D650", "S"):
            cell = BCell(D, d, r, base)
            allv = cell.run(f"B|{d}|{r}", EST_B + ("disc",))
            for (loc, comp), sid in stmt_of.items():
                if loc == (d, r):
                    need = ("T0", "TX", "disc") if comp.startswith("share of discharges") else ("T0", "TX")
                    sv = cell.run(sid, need)
                    key = ("per discharge", comp) if comp.startswith("share of discharges") else ("TX over T0", comp)
                    allv[key] = sv[key]
            o = {"counts": {"rows": len(cell.rows), "discharges": cell.U.count("discharge"), "clusters": cell.U.count("cluster"),
                            "rows_with_L5_above_100_eV": int(cell.rel_ok.sum()),
                            "discharges_with_at_least_50_rows": int(cell.disc_ok.sum())},
                 "estimators": {L: {ms: allv[(L, ms)] for ms in MEASURES} for L in EST_B},
                 "rmse_eV": {L: float(np.mean([np.sqrt(x.mean()) for x in cell.se[L]])) for L in EST_B},
                 "TX over T0": {nm: allv[("TX over T0", nm)] for nm, _ in RATIOS + DIFF_SH},
                 "per discharge": {k[1]: v for k, v in allv.items() if k[0] == "per discharge"}}
            res[d][r] = o
            if d == "V1" and r == "D650":
                wqcheck = wq_direct(cell)
            M.log(f"part B {d} {r}: rows {len(cell.rows):,}")
    return res, wqcheck


def wq_direct(cell):
    """The weighted quantile against the quantile of the explicitly resampled rows, 20 draws (V1, D650, split 1, TX)."""
    rng = M.rng_for("wq-direct")
    out = {}
    for form in BT.FORMS:
        _, nu = cell.U.idx[form]
        Wm = rng.multinomial(nu, np.full(nu, 1.0 / nu), size=20).astype(np.float64)
        dev = 0.0
        for kind, qs in (("e", QS), ("rel", (0.5,))):
            wq = cell.wq(form, "TX", 0, kind)
            a = wq.draws(Wm, qs)
            b = np.array([wq.direct(Wm[i], qs) for i in range(20)])
            dev = max(dev, float(np.max(np.abs(a - b))))
        out[form] = {"draws": 20, "largest_abs_difference": dev, "identical": dev == 0.0}
    return out


# ------------------------------------------------------------------------------------------ part C
C_LEARNERS = ("T0", "TX") + M.PART_C_LEARNERS
C_COMPS = (("TM/T0", "TM", "T0"), ("TXM/T0", "TXM", "T0"), ("TXs/T0", "TXs", "T0"), ("TBi/T0", "TBi", "T0"),
           ("TBm/T0", "TBm", "T0"), ("TBo/T0", "TBo", "T0"), ("TX/T0", "TX", "T0"), ("TX/TM", "TX", "TM"),
           ("TXM/TX", "TXM", "TX"))


def part_c(D):
    res = {}
    for d in M.PART_C_UNITS:
        res[d] = {}
        subs = D.subs(d)
        for r in M.RULES:
            rows = np.flatnonzero(D.scored(subs) & D.sets75[M.SETOF[r]])
            se = {L: D.se(subs, L, r, rows) for L in C_LEARNERS}
            U = BT.Units(rows, D.unit_d, D.unit_c)
            o = {"counts": {"rows": len(rows), "discharges": U.count("discharge"), "clusters": U.count("cluster")},
                 "rmse_eV": {L: float(np.mean([np.sqrt(x.mean()) for x in se[L]])) for L in C_LEARNERS}}
            for nm, A, Rf in C_COMPS:
                def vecs(form, si, A=A, Rf=Rf):
                    return {"A": U.sums(form, se[A][si]), "R": U.sums(form, se[Rf][si])}
                o[nm] = BT.boot(stream_of("C", (d, r), nm), U, len(subs), vecs, lambda v: 1 - v["A"] / v["R"])
            # rows with missing inputs (no donor row in TXs; undefined Mirnov or BES-column scalars)
            miss = {}
            for L in M.PART_C_LEARNERS:
                tf, tt = 0, 0
                for units in subs:
                    for u in units:
                        z = np.load(M.path82(u["name"], L, r))
                        tt += int(z["nonfinite_test_rows"])
                        tf += int(z["nonfinite_fit_rows"])
                miss[L] = {"test_rows_with_a_missing_input_summed_over_units": tt, "fit_rows_with_a_missing_input_summed_over_units": tf}
            nod = 0
            for units in subs:
                for u in units:
                    z = np.load(M.path82(u["name"], "TXs", r))
                    nod += int((z["donor_grid_test"] < 0).sum())
            miss["TXs"]["test_rows_without_a_donor_row"] = nod
            o["missing_inputs"] = miss
            res[d][r] = o
            M.log(f"part C {d} {r}: rows {len(rows):,}")
    return res


# ------------------------------------------------------------------------------------------ part D
def part_d(D):
    res = {}
    for r in M.RULES:
        rr = {}
        r2 = []
        for name in M.SEED_UNITS:
            u = next(x for x in D.us if x["name"] == name)
            rows = np.flatnonzero(np.isin(D.R["fidx"], u["test"]) & D.sets75[M.SETOF[r]])
            y = D.y[rows]

            def sse(L):
                return float(((D.est([u], L, r)[rows] - y) ** 2).sum())
            reg = 1 - sse("TX") / sse("T0")
            per = [1 - sse(f"TX_seed{s}") / sse(f"T0_seed{s}") for s in M.SEEDS]
            rr[name] = summarize(reg, per)
            rr[name]["rows"] = len(rows)
            if name.startswith("V1_r2"):
                r2.append({"reg": (sse("TX"), sse("T0")), "seeds": [(sse(f"TX_seed{s}"), sse(f"T0_seed{s}")) for s in M.SEEDS]})
        reg = 1 - sum(x["reg"][0] for x in r2) / sum(x["reg"][1] for x in r2)
        per = [1 - sum(x["seeds"][j][0] for x in r2) / sum(x["seeds"][j][1] for x in r2) for j in range(len(M.SEEDS))]
        rr["V1 r2 pooled (five units)"] = summarize(reg, per)
        res[r] = rr
    return res


def summarize(reg, per):
    p = np.array(per)
    return {"registered_seed_20261116": float(reg), "per_seed": {str(s): float(v) for s, v in zip(M.SEEDS, p)},
            "min": float(p.min()), "median": float(np.median(p)), "max": float(p.max()), "sd_ddof1": float(p.std(ddof=1)),
            "sign_changes_among_the_ten_seeds": bool((p > 0).any() and (p <= 0).any()),
            "sign_differs_from_registered": bool(np.any(np.sign(p) != np.sign(reg)))}


# ------------------------------------------------------------------------------------------ assembly
def statements(res):
    S = {}
    for sid, (part, loc, comp, text, rule, exp) in STATEMENTS.items():
        if part == "A":
            d, r, block, strat, cls = loc
            v = res["part_A"][d][r][block][strat][cls][comp]
        elif part == "B":
            d, r = loc
            cell = res["part_B"][d][r]
            v = cell["per discharge"][comp] if comp.startswith("share of discharges") else cell["TX over T0"][comp]
        else:
            d, r = loc
            v = res["part_C"][d][r][comp]
        if v.get("stream") != sid:
            raise SystemExit(f"FATAL: {sid} was not drawn on its own stream")
        S[sid] = {"text": text, "part": part, "location": list(loc), "comparison": comp, "judged": JUDGED[rule],
                  "value": {k: v[k] for k in ("point", "discharge", "cluster")}, "holds": bool(verdict(v, rule)),
                  "expected": exp}
    return S


def expectation_met(st):
    out = {}
    for k, s in st.items():
        e = s["expected"]
        out[k] = {"verdict_as_expected": s["holds"] if e.startswith("holds") else
                  (not s["holds"] if e.startswith("does not hold") else None)}
    return out


def main():
    t_start = time.time()
    chk = {nm: json.loads((M.CHK / f"{nm}.json").read_text(encoding="utf-8"))["passes"] for nm in CHECKS}
    if not all(chk.values()):
        raise SystemExit(f"FATAL: a check has not passed: {chk}")
    D = Data()
    res = {"registration": M.REG, "addendum": M.ADD, "n_boot": N, "seed": M.SEED, "target": M.X,
           "rule_sets": {"D10": "common set (common_L5 of data/.b75/sets.npz)", "D650": "common set (common_L5)",
                         "S": "S set (S_L5)"},
           "sources": {"T0, TX (seed 20261116)": "data/.b75/fits (B.75)", "TE, TB of V1, V4a, V4b": "data/.b75/fits (B.75)",
                       "TE, TB of V2, V3": "data/.b81/fits (B.81)", "TN, TXN": "data/.b79/fits (B.79, beam columns as delivered)",
                       "TM, TXM, TXs, TBi, TBm, TBo, seed refits": "data/.b82/fits (B.82)"},
           "classes": {"tau": list(M.TAU_CLASSES), "c": list(M.C_CLASSES)}}
    res["part_A"] = part_a(D)
    res["part_B"], res["wq_direct_check"] = part_b(D)
    res["part_C"] = part_c(D)
    res["part_D"] = part_d(D)
    res["part_E"] = json.loads((M.CACHE / "facts.json").read_text(encoding="utf-8"))
    res["c_cut_points"] = {k.split("|")[0] + "|" + k.split("|")[1]: [float(x) for x in D.cc[k]]
                           for k in D.cc if k.endswith("|cuts")}
    res["designs"] = {"d1_discharges": int(len(D.tab["fidx"])), "clusters_d1": int(len(np.unique(D.tab["clus"]))),
                      "units": [{"name": u["name"], "fit": int(len(u["fit"])), "test": int(len(u["test"]))} for u in D.us]}
    res["statements"] = statements(res)
    res["expectations"] = expectation_met(res["statements"])
    res["checks"] = chk
    res["runtime_readout_s"] = round(time.time() - t_start, 1)
    M.save_json(M.OUT_JSON, res)
    M.save_text(M.OUT_MD, load_render().render(res))
    M.log(f"written {M.OUT_JSON.name}, {M.OUT_MD.name}")


def load_render():
    spec = importlib.util.spec_from_file_location("render82", M.HERE / "render82.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    main()
