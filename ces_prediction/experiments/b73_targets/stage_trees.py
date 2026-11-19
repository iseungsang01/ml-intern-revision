"""B.73 stage 3: the boosted regressions without a network (T0, TX, TB, TE, TXlag; registration section 4) and the
usability classifiers (section 6), fitted per unit of every design (designs.py), each cached on its own.

    py ces_prediction/experiments/b73_targets/stage_trees.py [--design V1,V2] [--learners T0,TX] [--targets F,L5]
                                                             [--units NAME,...] [--no-clf] [--clf-only]
Output: data/.b73/trees/{unit}__{learner}__{target}.npz (test-row estimates), data/.b73/clf/{unit}__{H|HX}.npz
"""
from __future__ import annotations

import argparse
import math
import time
import warnings

import common as C
import designs as DS

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
TREES = C.CACHE / "trees"
CLF = C.CACHE / "clf"


class Feats:
    """The eight fast scalars of B.64 (run_b33.window_feats of the BES and ECEI array means of the channels
    standardised with the statistics of the fit discharges, as run_b37.Ctx.fast_stats / feats), per unit."""

    def __init__(self, fast=None):
        G = C.grid()
        self.off, self.seg, self.fid = G["off"], G["seg"], G["fidx"]
        self.fast = G["fast"] if fast is None else fast
        del G

    def stats(self, fit_files):
        """Mean and sd (ddof 0) of each channel over all grid rows of the fit discharges, two-pass with exactly rounded
        sums (math.fsum), so that the result depends on those rows only, bit for bit."""
        rows = np.isin(self.fid, fit_files)
        X = self.fast[rows].astype(np.float64)
        n = X.shape[0]
        mu = np.array([math.fsum(X[:, c]) / n for c in range(X.shape[1])])
        var = np.array([math.fsum((X[:, c] - mu[c]) ** 2) / n for c in range(X.shape[1])])
        return mu, np.sqrt(var) + 1e-12

    def window(self, fit_files):
        mu, sd = self.stats(fit_files)
        z = (self.fast.astype(np.float64) - mu) / sd
        bes = z[:, 0].copy()
        for c in range(1, C.NB):                     # explicit sequential sums: row-wise, bit-reproducible
            bes += z[:, c]
        bes /= C.NB
        ecei = z[:, C.NB].copy()
        for c in range(C.NB + 1, C.NB + C.NE):
            ecei += z[:, c]
        ecei /= C.NE
        N = len(self.seg)
        out = np.empty((N, 8))
        for i in range(len(self.off) - 1):          # per file, as run_b37.feats: running sums never cross a file
            a, b = self.off[i], self.off[i + 1]
            out[a:b, :4] = C.window_feats(bes[a:b], self.seg[a:b])
            out[a:b, 4:] = C.window_feats(ecei[a:b], self.seg[a:b])
        return out   # N x 8

    def at(self, wf, gi):
        fast = wf[gi]
        lag_ok = (gi >= 1) & (self.seg[np.maximum(gi - 1, 0)] == self.seg[gi])
        fastl = np.full_like(fast, np.nan)
        fastl[lag_ok] = wf[gi[lag_ok] - 1]
        return fast, fastl


def history_matrix(R):
    gp, hold = R["gp"], R["hold"]
    return np.column_stack([R["tstep"], R["age"], hold, gp - hold, np.abs(gp - hold), R["thist"]])


def learner_inputs(H10, fast, fastl):
    return {"T0": H10, "TX": np.column_stack([H10, fast]), "TB": np.column_stack([H10, fast[:, :4]]),
            "TE": np.column_stack([H10, fast[:, 4:]]), "TXlag": np.column_stack([H10, fastl])}


def complete_mask(H10, fast, fastl, gp):
    return np.isfinite(H10).all(1) & np.isfinite(fast).all(1) & np.isfinite(fastl).all(1) & np.isfinite(gp)


def gbr():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=C.TREE_SEED)


def gbc():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=C.TREE_SEED)


def run_unit(u, R, H10, FE, learners, targets, do_clf, do_reg, Q, out_dir=TREES, exclude=None):
    todo = []
    spec = DS.TREE_SPEC[u["design"]]
    if do_reg:
        for L, tg in spec.items():
            if learners and L not in learners:
                continue
            for X in tg:
                if targets and X not in targets:
                    continue
                if not (out_dir / f"{u['name']}__{L}__{X}.npz").exists():
                    todo.append((L, X))
    clf_todo = []
    if do_clf and u["design"] in DS.CLF_DESIGNS:
        clf_todo = [m for m in ("H", "HX") if not (CLF / f"{u['name']}__{m}.npz").exists()]
    if not todo and not clf_todo:
        return
    t0 = time.time()
    wf = FE.window(u["fit"])
    fit_d = np.isin(R["fidx"], u["fit"])
    test_d = np.isin(R["fidx"], u["test"])
    test_idx = np.flatnonzero(test_d)
    if todo:
        fast, fastl = FE.at(wf, R["gi"])
        ok = complete_mask(H10, fast, fastl, R["gp"])
        XS = learner_inputs(H10, fast, fastl)
        for L, X in todo:
            tgt = R[f"tg_{X}"]
            fr = fit_d & ok & np.isfinite(tgt)
            if exclude is not None:
                fr &= ~exclude
            m = gbr().fit(XS[L][fr], (tgt - R["gp"])[fr])
            est = R["gp"][test_idx].copy()
            te_ok = ok[test_idx]
            est[te_ok] += m.predict(XS[L][test_idx[te_ok]])
            C.save_npz(out_dir / f"{u['name']}__{L}__{X}.npz", idx=test_idx, est=est, complete=te_ok,
                       fit_rows=np.int64(fr.sum()), n_iter=np.int64(m.n_iter_))
            C.log(f"  {u['name']} {L}[{X}]: fit rows {int(fr.sum()):,}, test rows {len(test_idx):,} "
                  f"({time.time() - t0:.0f}s)")
    if clf_todo:
        fastc, _ = FE.at(wf, Q["gidx"])
        hx = np.column_stack([Q["tstep"], Q["age"], Q["n10"], Q["prev_has"]]).astype(np.float64)
        okc = np.isfinite(hx).all(1) & np.isfinite(fastc).all(1)
        fq = np.isin(Q["fidx"], u["fit"]) & okc
        tq = np.flatnonzero(np.isin(Q["fidx"], u["test"]) & okc)
        XQ = {"H": hx, "HX": np.column_stack([hx, fastc])}
        for mname in clf_todo:
            m = gbc().fit(XQ[mname][fq], Q["label"][fq])
            p = m.predict_proba(XQ[mname][tq])[:, 1]
            C.save_npz(CLF / f"{u['name']}__{mname}.npz", idx=tq, p=p, fit_rows=np.int64(fq.sum()),
                       excluded_incomplete=np.int64((~okc & np.isin(Q["fidx"], np.concatenate([u["fit"], u["test"]]))).sum()))
            C.log(f"  {u['name']} classifier {mname}: fit rows {int(fq.sum()):,}, test rows {len(tq):,} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="")
    ap.add_argument("--learners", default="")
    ap.add_argument("--targets", default="")
    ap.add_argument("--units", default="")
    ap.add_argument("--no-clf", action="store_true")
    ap.add_argument("--clf-only", action="store_true")
    ap.add_argument("--exclude-beamoff", action="store_true",
                    help="A4: fit rows without the rows exposed without beam emission; output data/.b73/trees_nobeam/")
    a = ap.parse_args()
    designs = set(a.design.split(",")) if a.design else None
    learners = set(a.learners.split(",")) if a.learners else None
    targets = set(a.targets.split(",")) if a.targets else None
    names = set(a.units.split(",")) if a.units else None
    R = C.rows()
    Q = C.clf_rows() if not a.no_clf else None
    H10 = history_matrix(R)
    FE = Feats()
    us, _, _ = DS.units(R)
    for u in us:
        if designs and u["design"] not in designs:
            continue
        if names and u["name"] not in names:
            continue
        if a.exclude_beamoff:
            ex = C.load_npz(C.CACHE / "strata_inputs.npz")["beamoff"].astype(bool)
            run_unit(u, R, H10, FE, learners, targets, False, True, None, C.CACHE / "trees_nobeam", ex)
        else:
            run_unit(u, R, H10, FE, learners, targets, not a.no_clf, not a.clf_only, Q)
    C.log("trees: done")


if __name__ == "__main__":
    main()
