"""B.81 checks (registration section 6, Addendum 1 section 3).  Each run once, after the fits and before any readout.
No target, error or skill is formed here: only idx arrays, matrices and estimates are compared.

    py ces_prediction/experiments/b81_controls/checks81.py reuse        # T0, TX, TE, TB of V1_r2_k0, D650 = B.75, bit for bit
    py ces_prediction/experiments/b81_controls/checks81.py permutation  # TXp history columns = TX; permuted columns same multiset
    py ces_prediction/experiments/b81_controls/checks81.py anchor       # TXo fast scalars = fast scalars at the anchor row
    py ces_prediction/experiments/b81_controls/checks81.py rows         # every estimator of a comparison on the same rows
    py ces_prediction/experiments/b81_controls/checks81.py write        # docs/paper/b81_checks.md
Output: data/.b81/checks/{name}.json, docs/paper/b81_checks.md
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import common81 as B

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = B.C
CHK = B.CHK


def bitsame(a, b):
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def same_multiset(a, b):
    """Column-wise: sorted values identical byte for byte (NaN sorted to the end on both sides)."""
    return bitsame(np.sort(a, axis=0), np.sort(b, axis=0))


def same_rows_jointly(a, b):
    """The rows of a are a reordering of the rows of b (rows sorted lexicographically are identical)."""
    def srt(x):
        x = np.ascontiguousarray(x, np.float64)
        k = np.nan_to_num(x, nan=np.inf)
        o = np.lexsort(k.T[::-1])
        return x[o]
    return bitsame(srt(a), srt(b))


# ------------------------------------------------------------------------------------------ reuse
def check_reuse():
    ctx = B.Ctx()
    us, _, _ = B.units()
    u = next(x for x in us if x["name"] == "V1_r2_k0")
    rule = "D650"
    wf, ms_b, ms_e = ctx.unit_arrays(u)
    H10, f8, A, I = ctx.matrices(rule, wf, ms_b, ms_e)
    fr, te = ctx.rows_of(u, rule)
    tgt = ctx.R[f"tg_{B.X}"]
    out = {"unit": u["name"], "rule": rule, "target": B.X}
    ok = True
    for L in ("T0", "TX", "TE", "TB"):
        Xf, Xt = B.design_matrices(L, u, rule, H10, f8, A, I, wf, fr, te)
        est, m = B.fit_estimate(Xf, Xt, tgt, I["G"], fr, te)
        z = np.load(B.B75FITS / f"{u['name']}__{L}__{rule}__{B.X}.npz")
        same_idx = bool(np.array_equal(z["idx"], te))
        same_est = bitsame(z["est"], est)
        r = {"same_idx": same_idx, "est_bit_identical": same_est, "rows": int(len(te)),
             "fit_rows": int(len(fr)), "stored_fit_rows": int(z["fit_rows"]),
             "n_iter": int(m.n_iter_), "stored_n_iter": int(z["n_iter"])}
        if not same_est and same_idx:
            r["max_abs_dev_eV"] = float(np.max(np.abs(z["est"] - est)))
        out[L] = r
        ok &= same_idx and same_est
        B.log(f"reuse {L}: idx {same_idx}, est bit-identical {same_est}")
    out["passes"] = bool(ok)
    B.save_json(CHK / "reuse.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ permutation
def check_permutation():
    ctx = B.Ctx()
    us, _, _ = B.units()
    res, ok = {}, True
    for u in us:
        wf, ms_b, ms_e = ctx.unit_arrays(u)
        for rule in B.RULES:
            H10, f8, A, I = ctx.matrices(rule, wf, ms_b, ms_e)
            fr, te = ctx.rows_of(u, rule)
            Tf, Tt = B.design_matrices("TX", u, rule, H10, f8, A, I, wf, fr, te)
            Pf, Pt = B.design_matrices("TXp", u, rule, H10, f8, A, I, wf, fr, te)
            r = {}
            for part, T, P in (("fit", Tf, Pf), ("test", Tt, Pt)):
                r[part] = {"rows": int(len(T)), "history_bit_identical": bitsame(T[:, :10], P[:, :10]),
                           "permuted_columns_same_multiset": same_multiset(T[:, 10:], P[:, 10:]),
                           "permuted_rows_joint_reordering": same_rows_jointly(T[:, 10:], P[:, 10:]),
                           "columns": int(P.shape[1]),
                           "rows_moved": int((~(P[:, 10:] == T[:, 10:]).all(1) & np.isfinite(T[:, 10:]).all(1)).sum())}
                ok &= r[part]["history_bit_identical"] and r[part]["permuted_columns_same_multiset"] and \
                    r[part]["permuted_rows_joint_reordering"] and r[part]["columns"] == 24
            res[f"{u['name']}|{rule}"] = r
        B.log(f"permutation {u['name']}: done")
    out = {"cells": res, "units": len(us), "rules": list(B.RULES), "passes": bool(ok)}
    B.save_json(CHK / "permutation.json", out)
    print(json.dumps({"passes": out["passes"], "cells": len(res)}, indent=1))


# ------------------------------------------------------------------------------------------ anchor row
def check_anchor():
    ctx = B.Ctx()
    us, _, _ = B.units()
    u = next(x for x in us if x["name"] == "V1_r2_k0")
    G = B.K75.grid_lean(("off", "seg", "fast", "fidx"))
    off, seg, fast, gfid = G["off"], G["seg"], G["fast"], G["fidx"]
    wf, ms_b, ms_e = ctx.unit_arrays(u)
    mu, sd = ctx.FE.stats(u["fit"])
    out = {"unit": u["name"]}
    ok = True
    for rule in B.RULES:
        H10, f8, A, I = ctx.matrices(rule, wf, ms_b, ms_e)
        fr, te = ctx.rows_of(u, rule)
        _, Xt = B.design_matrices("TXo", u, rule, H10, f8, A, I, wf, fr, te)
        pick = np.sort(B.rng_for(f"anchor|{rule}").choice(len(te), 200, replace=False))
        n_same, n_nan, n_d1 = 0, 0, 0
        gi_to_row = {int(g): j for j, g in enumerate(ctx.R["gi"])}
        for p in pick:
            j = int(te[p])
            a = int(I["anc_g"][j])
            fi = int(gfid[a])
            lo, hi = int(off[fi]), int(off[fi + 1])
            z = (fast[lo:hi].astype(np.float64) - mu) / sd
            bes = z[:, 0].copy()
            for c in range(1, C.NB):
                bes += z[:, c]
            bes /= C.NB
            ecei = z[:, C.NB].copy()
            for c in range(C.NB + 1, C.NB + C.NE):
                ecei += z[:, c]
            ecei /= C.NE
            ref = np.concatenate([C.window_feats(bes, seg[lo:hi])[a - lo], C.window_feats(ecei, seg[lo:hi])[a - lo]])
            got = Xt[p, 10:18]
            same = bitsame(np.nan_to_num(got, nan=-7.25e300), np.nan_to_num(ref, nan=-7.25e300)) and \
                bool(np.array_equal(np.isnan(got), np.isnan(ref)))
            n_same += int(same)
            n_nan += int(np.isnan(ref).any())
            # beside it: where the anchor row is itself a D1 row, TX's own fast scalars of that row
            if a in gi_to_row:
                n_d1 += int(bitsame(np.nan_to_num(f8[gi_to_row[a]], nan=-7.25e300), np.nan_to_num(got, nan=-7.25e300)))
        out[rule] = {"rows": int(len(pick)), "test_rows": int(len(te)), "rows_identical": n_same,
                     "rows_with_undefined_anchor_scalars": n_nan,
                     "rows_whose_anchor_is_a_D1_row_and_equal_to_TX_fast_scalars_of_that_row": n_d1}
        ok &= n_same == len(pick)
        B.log(f"anchor {rule}: {n_same}/{len(pick)} identical")
    out["passes"] = bool(ok)
    B.save_json(CHK / "anchor.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ rows
def check_rows():
    R = B.K75.rows_lean()
    S = B.load_npz(B.K75.CACHE / "sets.npz")
    us, _, _ = B.units()
    res, ok = {}, True
    for u in us:
        for rule in B.RULES:
            te = np.flatnonzero(np.isin(R["fidx"], u["test"]) & S[B.SETOF[rule]])
            r = {"rows": int(len(te))}
            for L in B.LEARNERS:
                z = np.load(B.est_path(u, L, rule))
                same = bool(np.array_equal(z["idx"], te)) and len(z["est"]) == len(te)
                r[L] = same
                ok &= same
            res[f"{u['name']}|{rule}"] = r
            print(f"rows {u['name']:10s} {rule:5s} {len(te):7d} " + " ".join(f"{L}:{int(r[L])}" for L in B.LEARNERS))
    out = {"cells": res, "passes": bool(ok)}
    B.save_json(CHK / "rows.json", out)
    print(json.dumps({"passes": out["passes"], "cells": len(res)}, indent=1))


# ------------------------------------------------------------------------------------------ md
def write():
    j = {n: json.loads((CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("reuse", "permutation", "anchor", "rows")}
    ru, pe, an, ro = j["reuse"], j["permutation"], j["anchor"], j["rows"]
    L = ["# B.81 checks", "",
         f"Registration: `{B.REG}` section 6; implementation: `{B.ADD}` section 3. Each check was run once, after the fits and "
         "before any readout. Only idx arrays, matrices and estimates are compared; no target, error or skill was formed. "
         "No tolerance was changed after a run.", "",
         "## 1. Reuse", "",
         f"Unit {ru['unit']}, rule {ru['rule']}, target {ru['target']}: refitted with this batch's code and compared with "
         "`data/.b75/fits/`.", "",
         "| learner | test rows | same rows | estimates bit-identical | fit rows (this batch / B.75) | iterations (this batch / B.75) |",
         "|---|---|---|---|---|---|"]
    for Lr in ("T0", "TX", "TE", "TB"):
        x = ru[Lr]
        L.append(f"| {Lr} | {x['rows']:,} | {x['same_idx']} | {x['est_bit_identical']} | {x['fit_rows']:,} / {x['stored_fit_rows']:,} | "
                 f"{x['n_iter']} / {x['stored_n_iter']} |")
    L += ["", f"Check passes: **{ru['passes']}**", "",
          "## 2. Permutation", "",
          f"Every unit ({pe['units']}) and both rules: the TXp matrices rebuilt by the fit code against TX's on the same rows. "
          "History columns 0-9 bit-identical; each of columns 10-23 the same multiset of values (sorted, byte comparison); "
          "the fourteen columns of a row moved together (rows sorted lexicographically identical); within the fit rows and "
          "within the test rows separately.", "",
          "| unit, rule | fit rows | test rows | history identical (fit / test) | same multisets (fit / test) | joint reordering (fit / test) | rows moved (fit / test) |",
          "|---|---|---|---|---|---|---|"]
    for k, x in pe["cells"].items():
        f_, t_ = x["fit"], x["test"]
        L.append(f"| {k} | {f_['rows']:,} | {t_['rows']:,} | {f_['history_bit_identical']} / {t_['history_bit_identical']} | "
                 f"{f_['permuted_columns_same_multiset']} / {t_['permuted_columns_same_multiset']} | "
                 f"{f_['permuted_rows_joint_reordering']} / {t_['permuted_rows_joint_reordering']} | "
                 f"{f_['rows_moved']:,} / {t_['rows_moved']:,} |")
    L += ["", f"Check passes: **{pe['passes']}**", "",
          "## 3. Anchor row", "",
          f"Unit {an['unit']}: 200 test rows per rule. Columns 10-17 of TXo against the eight fast scalars recomputed at the "
          "anchor's grid index from the anchor's file alone (channels standardised with the unit's statistics, sequential "
          "array means, B.73 `window_feats`).", "",
          "| rule | test rows | rows drawn | identical | rows with undefined anchor-row fast scalars | anchor a D1 row and equal to TX's fast scalars of that row |",
          "|---|---|---|---|---|---|"]
    for rule in B.RULES:
        x = an[rule]
        L.append(f"| {rule} | {x['test_rows']:,} | {x['rows']} | {x['rows_identical']} | {x['rows_with_undefined_anchor_scalars']} | "
                 f"{x['rows_whose_anchor_is_a_D1_row_and_equal_to_TX_fast_scalars_of_that_row']} |")
    L += ["", f"Check passes: **{an['passes']}**", "",
          "## 4. Rows", "",
          "For every unit and both rules, the test rows of T0, TX, TXp, TXo, TE and TB (stored files of B.75 or B.81) against "
          "the rows of the unit's test discharges in the rule's set (D650: common set, L5; S: S set, L5).", "",
          "| unit, rule | rows | T0 | TX | TXp | TXo | TE | TB |", "|---|---|---|---|---|---|---|---|"]
    for k, x in ro["cells"].items():
        L.append(f"| {k} | {x['rows']:,} | " + " | ".join(str(x[Lr]) for Lr in B.LEARNERS) + " |")
    L += ["", f"Check passes: **{ro['passes']}**", ""]
    B.save_text(B.CHECKS_MD, "\n".join(L))
    print("written", B.CHECKS_MD)


if __name__ == "__main__":
    t0 = time.time()
    {"reuse": check_reuse, "permutation": check_permutation, "anchor": check_anchor, "rows": check_rows,
     "write": write}[sys.argv[1]]()
    B.log(f"done ({time.time() - t0:.0f}s)")
