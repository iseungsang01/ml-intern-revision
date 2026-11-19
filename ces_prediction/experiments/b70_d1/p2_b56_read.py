"""B.70 phase 2: B.56's pooled readout on the D1 rows with the refitted filters (PREREGISTRATION_B70.md, Addendum 2).

run_b56's Pooled class (two-level bootstrap, its seed and draws) on data/.b70p2/b56/ (p2_b56_fit.py), with
unit_sources pointed at the D1 runs and every pooled set = r2-r4.  The blocks of run_b56.analyze that are on the pooled
folds are copied (all rows, rows not after an absent frame, rows not flagged by the value rules, thirds by median T_i,
r2 alone); the four-split and time-split units are not re-read (C1).  C4: "rows not after an absent frame" = not rows 1-3
of the B.54 definition (blip13 of the D1 row tables); B.56's own ten-row mask is reported beside.

    py ces_prediction/experiments/b70_d1/p2_b56_read.py
Output: data/.b70p2/paper/b56_kfx.{json,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b56_kfx"))
import p2_common as PC  # noqa: E402
import run_b56 as B56  # noqa: E402

B54C = PC.P2 / "b54"


def setup():
    PC.use_d1()
    B56.CACHE = PC.P2 / "b56"
    B56.REPEATS = B56.CONFIRM = PC.REPEATS
    B56.unit_sources = lambda unit: ([PC.d1_run_dir("v2", int(unit[1:].split("_f")[0]), int(unit.split("_f")[1]))],
                                     [PC.d1_run_dir("v2h", int(unit[1:].split("_f")[0]), int(unit.split("_f")[1]))])


def rows54(unit):
    z = np.load(B54C / f"rows_{unit}.npz")
    return {k: z[k] for k in ("shot", "row", "plain", "blip13", "fidx", "grow", "y")}


def checks():
    """run_b56 check (b) on the D1 units; (a) compared recorded old numbers (C2): replaced by the r2-r4 point values
    against those computed from the D1 npz directly; (c) does not depend on the rows (run unchanged)."""
    g = B56.ns()
    b = {"units": {}, "passes": True}
    for r in PC.REPEATS:
        for k in range(5):
            u = f"r{r}_f{k}"
            z = np.load(B56.CACHE / f"{u}.npz")
            ok = z["gi"] >= 0
            vals = np.full(len(ok), np.nan)
            vals[ok] = g["Y"][z["shot"][ok], z["gi"][ok]]
            dmax = float(np.nanmax(np.abs(vals - z["y"]))) if ok.any() else np.inf
            fin = all(np.isfinite(z[f"est_{q}"]).all() for q in ("kf0", "kc", "kg", "kfx"))
            q = rows54(u)
            same = np.array_equal(q["shot"], z["shot"]) and np.array_equal(q["row"], z["row"])
            off = sorted(set(np.unique(q["grow"] - z["gi"]).tolist())) if same else None
            rec = {"rows_stored": int(len(ok)), "rows_located": int(ok.sum()), "max_abs_value_diff_eV": dmax,
                   "all_filter_outputs_finite": bool(fin), "same_rows_as_d1_b54": bool(same), "b54_grid_row_minus_mine": off}
            rec["passes"] = bool(rec["rows_located"] == rec["rows_stored"] and dmax <= 0.01 and fin and same and off == [1])
            b["passes"] &= rec["passes"]
            b["units"][u] = rec
    return {"b": b, "c": B56.check_c(), "all_pass": bool(b["passes"])}


def main():
    setup()
    chk = checks()
    chk["all_pass"] = bool(chk["all_pass"] and chk["c"]["passes"])
    if not chk["all_pass"]:
        (PC.paper() / "b56_kfx.json").write_text(json.dumps({"checks": chk, "stopped": True}, indent=1), encoding="utf-8")
        raise SystemExit("FATAL: a check fails")
    rng = np.random.default_rng(B56.SEED_BOOT)
    res = {"note": "B.70 phase 2: run_b56's pooled blocks on the D1 rows, filters refitted per D1 item 2.9 "
                   "(training discharges, held repeats removed); r2-r4 replaces both the confirmation set r3-r9 and "
                   "the development set r2 (r2 alone reported as dev_r2).", "checks": chk, "clip": B56.CLIP_MAIN}

    def block(reps, mask_fn=None, full=True):                                # copied from run_b56.analyze
        P = B56.Pooled(reps, mask_fn)
        o = {}
        pairs = [("kfx", "kf0"), ("kg", "kf0"), ("A", "kfx"), ("Tw", "kf0"), ("kfx", "Tw"), ("kc", "kf0"),
                 ("kfx", "kc"), ("kfx", "kg"), ("kfx", "G"), ("kfx", "O"), ("A", "G"), ("A", "Tw"), ("kf0", "G")]
        if full:
            pairs += [("kfx_clip0.1", "kf0"), ("kfx_clip0.4", "kf0")]
        for a, b in pairs:
            o[f"{a}_over_{b}"], o[f"{a}_over_{b}_per_repeat"] = P.skill(a, b, rng)
        if full:
            o["kf0_over_G_90"], _ = P.skill("kf0", "G", rng, lo=5, hi=95)
            o["median_ratio_kfx_over_hold"], o["median_ratio_per_repeat"] = P.med_ratio("kfx", "H", rng)
        o["median_abs_eV"] = {k: P.median(k) for k in ("A", "Tw", "G", "H", "O", "kf0", "kc", "kg", "kfx")}
        o["rows"] = {str(r): int(len(P.rows[r]["A"])) for r in reps}
        o["discharges"] = {str(r): int(P.rows[r]["_n"]) for r in reps}
        return o

    R = PC.REPEATS
    B56.log("pooled r2-r4 ...")
    res["confirm"] = block(R)
    B56.log("rows not after an absent frame (D1 definition) ...")
    res["confirm_quiet"] = block(R, lambda d, r: ~rows54(d["unit"])["blip13"].astype(bool), full=False)
    B56.log("rows not within ten rows after an absent frame (B.56's own mask, beside) ...")
    res["confirm_quiet_tenrows"] = block(R, lambda d, r: B56.quiet_mask(d), full=False)
    B56.log("plain rows ...")
    res["confirm_plain"] = block(R, lambda d, r: rows54(d["unit"])["plain"].astype(bool), full=False)
    F = np.load(B54C / "files.npz")
    lab = {}
    for r in R:
        u = np.unique(np.concatenate([np.load(B56.CACHE / f"r{r}_f{k}.npz")["shot"] for k in range(5)]))
        v = F["median_ti"][u]
        have = np.flatnonzero(np.isfinite(v))
        o = np.lexsort((F["shot"][u][have], v[have]))
        parts = [have[p] for p in np.array_split(o, 3)]
        L = np.full(641, -1)
        for j, p in enumerate(parts):
            L[u[p]] = j
        lab[r] = L
    for j, tn in enumerate(("coldest", "middle", "hottest")):
        B56.log(f"third {tn} ...")
        res[f"confirm_third_{tn}"] = block(R, lambda d, r, j=j: lab[r][d["shot"]] == j, full=False)
    B56.log("r2 alone ...")
    res["dev_r2"] = block((2,))
    c = res["confirm"]
    res["statements_pooled"] = {
        "KX1": {"text": "KFX over KF0", "value": c["kfx_over_kf0"]},
        "KX2": {"text": "Kg over KF0", "value": c["kg_over_kf0"]},
        "KX3": {"text": "network over KFX", "value": c["A_over_kfx"]},
        "KX4": {"text": "KF0 over the causal GP, 90 % interval (registered side: above -0.01)", "value": c["kf0_over_G_90"]},
        "KX5": {"text": "median |e| KFX / hold, interval below 1", "value": c["median_ratio_kfx_over_hold"]},
        "KX6": {"text": "KFX over KF0 on rows not after an absent frame", "value": res["confirm_quiet"]["kfx_over_kf0"]}}
    res["params"] = {f"r{r}_f{k}": json.loads((B56.CACHE / f"r{r}_f{k}.json").read_text(encoding="utf-8"))
                     for r in R for k in range(5)}
    (PC.paper() / "b56_kfx.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    B56.log("written data/.b70p2/paper/b56_kfx.json")


if __name__ == "__main__":
    main()
