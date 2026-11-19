"""B.70 phase 2: B.57 (floor, gap closure, persistence of the correction) on the D1 rows (PREREGISTRATION_B70.md,
Addendum 2).

run_b57's stages twins, checks, analyze with: CACHE = data/.b70p2/b57/, B55C = data/.b70p2/b55/ (p2_b55.py),
B56C = data/.b70p2/b56/ (refitted filters), REPEATS = r2-r4, DATA = data/.b70p2/view/ (junctions named like the stored
runs to the D1 re-scorings, for load_fold), outputs to data/.b70p2/paper/.  Check (ii) compared recorded old values (C2):
it is reported but not required.  C4: the cell "rows not after an absent frame" uses the B.54 definition (rows 1-3,
blip13 of the D1 row tables); B.57's own ten-row mask (after_absent10) is reported beside as quiet_tenrows.

    py ces_prediction/experiments/b70_d1/p2_b57.py [twins|checks|analyze|all]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b57_yardsticks"))
import p2_common as PC  # noqa: E402
import run_b57 as B57  # noqa: E402


def setup():
    PC.use_d1()
    B57.CACHE = PC.P2 / "b57"
    B57.B55C = PC.P2 / "b55"
    B57.B56C = PC.P2 / "b56"
    B57.REPEATS = PC.REPEATS
    B57.DATA = PC.view()
    B57.OUT_JSON, B57.OUT_MD = PC.paper() / "b57_yardsticks.json", PC.paper() / "b57_yardsticks.md"
    import run_b55 as B55
    B55.REPEATS = PC.REPEATS


_F = {}


def blip13(d):
    """rows 1-3 after an absent frame (B.54 definition) of the rows of a fold: files.npz flags at (shot, grid row)."""
    if "F" not in _F:
        _F["F"] = np.load(PC.P2 / "b54" / "files.npz")
    F = _F["F"]
    out = np.zeros(len(d["grow"]), bool)
    for fi in np.unique(d["shot"]):
        m = d["shot"] == fi
        out[m] = F[f"blip13_{int(fi)}"][d["grow"][m]]
    return out


def checks():
    res = B57.checks()
    res["ii_note"] = {"passes_required": False,
                      "note": "(ii) compares recorded old values (B.70 Addendum 2, C2): reported, not required on the D1 rows"}
    res["all_pass"] = bool(res["i"]["passes"] and res["iii"]["passes"])
    (B57.CACHE / "checks.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def analyze():
    """run_b57.analyze, copied, with the quiet cell on the D1 definition and B.57's own ten-row cell beside."""
    chk = json.loads((B57.CACHE / "checks.json").read_text(encoding="utf-8"))
    if not chk["all_pass"]:
        raise SystemExit("FATAL: a check failed; no statement is read")
    rng = np.random.default_rng(B57.SEED)
    reps = [B57.Rep(r) for r in B57.REPEATS]
    P = B57.pooled(reps, rng)
    repq = [B57.Rep(r, mask_fn=lambda d: ~blip13(d)) for r in B57.REPEATS]
    Pq = B57.pooled(repq, rng)
    rept = [B57.Rep(r, mask_fn=lambda d: ~B57.after_absent10(d)) for r in B57.REPEATS]
    Pt = B57.pooled(rept, rng)
    res = {"checks": chk, "all": P, "quiet": Pq, "quiet_tenrows": Pt,
           "rows": {str(r): int(R.S["all_n"].sum()) for r, R in zip(B57.REPEATS, reps)},
           "triple_rows": {str(r): int(R.S["tri_n"].sum()) for r, R in zip(B57.REPEATS, reps)},
           "chain_rows_h1": {str(r): int(R.S["n_h1"].sum()) for r, R in zip(B57.REPEATS, reps)},
           "chain_rows_h3": {str(r): int(R.S["n_h3"].sum()) for r, R in zip(B57.REPEATS, reps)}}
    n1 = {e: P[f"mse_over_s2_{e}"] for e in B57.EST}
    st = {"N1": {"value": n1, "rule": "every interval of MSE over the floor above 1", "holds": all(v[1] > 1 for v in n1.values())},
          "N2": {"value": P["gap_closure_A"], "rule": "interval above 0.5", "holds": P["gap_closure_A"][1] > 0.5},
          "N3": {"value": P["net_twin_rho1"], "rule": "interval above 0.5", "holds": P["net_twin_rho1"][1] > 0.5},
          "N4": {"value": P["kfx_kf0_rho1"], "rule": "interval above 0.5", "holds": P["kfx_kf0_rho1"][1] > 0.5},
          "N5": {"value": P["net_twin_rho3_minus_rho1"], "rule": "interval inside [-0.15, +0.15]",
                 "holds": P["net_twin_rho3_minus_rho1"][1] > -0.15 and P["net_twin_rho3_minus_rho1"][2] < 0.15}}
    for k in st:
        st[k]["holds"] = bool(st[k]["holds"])
    res["statements"] = st
    B57.OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    B57.OUT_MD.write_text(B57.render(res), encoding="utf-8", newline="\n")
    B57.log(f"written {B57.OUT_JSON}; " + ", ".join(f"{k} {'holds' if v['holds'] else 'does not hold'}" for k, v in st.items()))


def main():
    setup()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("twins", "all"):
        B57.stage_twins()
    if stage in ("checks", "all"):
        checks()
    if stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
