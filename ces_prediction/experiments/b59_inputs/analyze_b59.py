"""B.59 read-outs (PREREGISTRATION_B59.md sections 4 and 6).  CPU; reads comparison_errors_test.npz of the runs.

Statements (re-partitions r2, r3, r4 pooled; two-level interval: a re-partition is drawn among the three, then its
discharges are resampled, 20,000 draws -- the interval of analyze_b39.arm_p, copied below because it is a closure
there):
  L1  the network trained with the fast channels one row old (arm L) over the twin: interval above zero
  L2  the estimator of B.39 over that network: interval above zero
  N1  the network without Mirnov (v2n) over the twin: interval above zero
  N2  the estimator of B.39 over the network without Mirnov: interval inside [-0.01, +0.01]
  E1  the ECEI network (v2e) over the twin: interval above zero
  B1  the BES network (v2b) over the twin: interval above zero
Reported: every network over the causal GP, hold and the offline GP; the estimator over E and over B; B over E; in
the rows 1 to 3 after an absent frame (run_b54 row location, data/.b54/rows_r{r}_f{k}.npz, flag blip13) and in the
other rows, every network over the twin; median absolute errors; epochs run; the three re-partitions one by one;
the expectations of section 6 beside the findings.

    py ces_prediction/experiments/b59_inputs/analyze_b59.py [--partial]
Output: docs/paper/b59_inputs.json / .md.  Without --partial it stops, listing what is missing, unless all 60 runs and
the 30 B.39 references are present; with --partial an arm is read on the re-partitions whose five folds it has.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP / "b39_primary"))
import run_b59 as R  # noqa: E402
import analyze_b39 as A39  # noqa: E402  (load, same_rows, sums, ci, med)

REPO = HERE.parents[2]
OUT_JSON = REPO / "docs" / "paper" / "b59_inputs.json"
OUT_MD = REPO / "docs" / "paper" / "b59_inputs.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B59.md"
B_POOLED, SEED = 20_000, 20261159
K = 5
NETS = ("est", "twin", "L", "N", "E", "B")
LAB = {"est": "estimator (B.39)", "twin": "twin (B.39)", "L": "L: fast channels one row old", "N": "N: no Mirnov",
       "E": "E: ECEI + history", "B": "B: BES + history"}
REFS = {"gp_causal": "causal GP", "persistence": "hold", "gp": "offline GP"}
EQUIV = 0.01
STATEMENTS = {   # name -> (numerator, denominator, kind)
    "L1": ("L", "twin", "above"), "L2": ("est", "L", "above"), "N1": ("N", "twin", "above"),
    "N2": ("est", "N", "equiv"), "E1": ("E", "twin", "above"), "B1": ("B", "twin", "above")}
EXPECT = {"L1": "-0.01 to +0.02, does not hold", "L2": "+0.03 to +0.06, holds", "N1": "+0.035 to +0.06, holds",
          "N2": "-0.008 to +0.012, interval about 0.012 wide on each side; may fail by width",
          "E1": "+0.01 to +0.03, holds or fails by a small margin", "B1": "+0.035 to +0.06, holds"}
EXPECT_RANGE = {"L1": (-0.01, 0.02), "L2": (0.03, 0.06), "N1": (0.035, 0.06), "N2": (-0.008, 0.012),
                "E1": (0.01, 0.03), "B1": (0.035, 0.06)}


def run_path(net, r, k):
    if net == "est":
        return R.estimator_dir(r, k)
    if net == "twin":
        return R.twin_dir(r, k)
    return R.run_dir(net, r, k)


def have(net, r, k):
    return (run_path(net, r, k) / "comparison_errors_test.npz").exists()


def blip_flags(r, k, d):
    """Rows 1 to 3 after an absent frame for the stored rows of the estimator of fold (r, k): run_b54's row
    location (stage rows), read from its cache and joined on (shot index, row)."""
    p = EXP.parent.parent / "data" / ".b54" / f"rows_r{r}_f{k}.npz"
    if not p.exists():
        raise SystemExit(f"FATAL: {p} missing; run py ces_prediction/experiments/b54_pooled_rows/run_b54.py files, then rows")
    sys.path.insert(0, str(EXP / "b54_pooled_rows"))
    import run_b54
    z = run_b54.load_rows(r, k)
    if not (np.array_equal(z["shot"], d["shot"]) and np.array_equal(z["row"], d["row"])):
        raise SystemExit(f"FATAL: r{r} f{k}: the B.54 row table is not on the rows of the estimator")
    return z["blip13"].astype(bool)


# ------------------------------------------------------------------------------ data
def repeat_data(r, nets):
    """Per repeat: per-discharge sums of squared errors (all rows, blip rows, other rows) per network and reference,
    absolute errors per network, epochs."""
    cat = {q: [] for q in (*nets, *REFS, "shot", "blip")}
    epochs = {}
    for k in range(K):
        z = {n: A39.load(run_path(n, r, k)) for n in nets}
        for n in nets:
            A39.same_rows(z["est"], z[n], f"r{r} f{k} {n} vs estimator")
        cat["shot"].append(z["est"]["shot"])
        cat["blip"].append(blip_flags(r, k, z["est"]))
        for n in nets:
            cat[n].append(z[n]["model"])
        for q in REFS:
            cat[q].append(z["est"][q])
        for n in nets:
            m = json.loads((run_path(n, r, k) / "metrics.json").read_text(encoding="utf-8"))
            epochs[f"{n} f{k}"] = [m["epochs_run"], m["best_epoch"]]
    shot = np.concatenate(cat["shot"])
    blip = np.concatenate(cat["blip"])
    se = {q: np.concatenate(cat[q]) for q in (*nets, *REFS)}
    out = {"all": {}, "blip": {}, "other": {}}
    for q, v in se.items():
        out["all"][q] = A39.sums(shot, v)
        out["blip"][q] = A39.sums(shot, np.where(blip, v, 0.0))
        out["other"][q] = A39.sums(shot, np.where(blip, 0.0, v))
    return {"sums": out, "abs": {n: np.sqrt(se[n]) for n in nets}, "abs_ref": {q: np.sqrt(se[q]) for q in REFS},
            "rows": int(len(shot)), "discharges": int(len(np.unique(shot))), "blip_rows": int(blip.sum()), "epochs": epochs}


def hier(per, a, b, repeats, rng, group="all"):
    """Copied from analyze_b39.arm_p.hier (a closure there), with the repeats and the row group as arguments:
    a repeat is drawn, then its discharges are resampled with replacement.  Point = mean over repeats."""
    draws = rng.choice(np.array(repeats), size=B_POOLED)
    o = np.empty(B_POOLED)
    for r in repeats:
        sel = np.flatnonzero(draws == r)
        m, bb = per[r]["sums"][group][a], per[r]["sums"][group][b]
        idx = rng.integers(0, len(m), size=(len(sel), len(m)))
        o[sel] = 1.0 - m[idx].sum(1) / bb[idx].sum(1)
    point = float(np.mean([1 - per[r]["sums"][group][a].sum() / per[r]["sums"][group][b].sum() for r in repeats]))
    return [point, *A39.ci(o)], {str(r): float(1 - per[r]["sums"][group][a].sum() / per[r]["sums"][group][b].sum()) for r in repeats}


# ------------------------------------------------------------------------------ readout
def readout(partial):
    avail = {n: [r for r in R.REPEATS if all(have(n, r, k) for k in range(K))] for n in NETS}
    missing = [run_path(n, r, k).name for n in NETS for r in R.REPEATS for k in range(K) if not have(n, r, k)]
    if missing:
        print(f"[b59] missing comparison_errors_test.npz ({len(missing)}): {', '.join(missing)}")
        if not partial:
            raise SystemExit("[b59] not every run is present; nothing written (use --partial to read what is complete)")
    base = [r for r in R.REPEATS if r in avail["est"] and r in avail["twin"]]
    if not base:
        raise SystemExit("[b59] the B.39 estimator and twin are not complete on any re-partition")
    per = {}
    for r in base:
        nets = [n for n in NETS if r in avail[n]]
        per[r] = repeat_data(r, nets)
        per[r]["nets"] = nets
    rng = np.random.default_rng(SEED)

    def reps(*nets):
        return [r for r in base if all(n in per[r]["nets"] for n in nets)]

    def H(a, b, group="all"):
        rr = reps(a, b) if a not in REFS and b not in REFS else reps(*(x for x in (a, b) if x not in REFS))
        if not rr:
            return None
        v, p = hier(per, a, b, rr, rng, group)
        return {"skill": v, "per_repeat": p, "repeats": rr}

    res = {"registration": REG, "seed": SEED, "draws": B_POOLED, "partial": bool(missing), "missing_runs": missing,
           "repeats_read": {n: reps(n) for n in NETS},
           "rows": {str(r): {"rows": per[r]["rows"], "discharges": per[r]["discharges"], "blip_rows": per[r]["blip_rows"]} for r in base}}
    st = {}
    for name, (a, b, kind) in STATEMENTS.items():
        h = H(a, b)
        if h is None:
            st[name] = {"read": False}
            continue
        lo, hi = h["skill"][1], h["skill"][2]
        holds = bool(lo > 0) if kind == "above" else bool(lo >= -EQUIV and hi <= EQUIV)
        e = EXPECT_RANGE[name]
        st[name] = {"read": True, "numerator": a, "denominator": b, **h, "holds": holds,
                    "complete": len(h["repeats"]) == len(R.REPEATS), "expectation": EXPECT[name],
                    "point_inside_expected_range": bool(e[0] <= h["skill"][0] <= e[1])}
    res["statements"] = st
    rep = {"over_refs": {}, "pairs": {}, "blip_rows_over_twin": {}, "other_rows_over_twin": {}}
    for n in NETS:
        if not reps(n):
            continue
        rep["over_refs"][n] = {q: H(n, q) for q in REFS}
        if n != "twin":
            rep["blip_rows_over_twin"][n] = H(n, "twin", "blip")
            rep["other_rows_over_twin"][n] = H(n, "twin", "other")
    for a, b in (("est", "E"), ("est", "B"), ("B", "E"), ("est", "twin")):
        rep["pairs"][f"{a}_over_{b}"] = H(a, b)
    rep["median_abs_error_eV"] = {LAB.get(n, n): float(np.mean([np.median(per[r]["abs"][n]) for r in reps(n)]))
                                  for n in NETS if reps(n)}
    rep["median_abs_error_eV"].update({REFS[q]: float(np.mean([np.median(per[r]["abs_ref"][q]) for r in base])) for q in REFS})
    rep["epochs_run_best"] = {str(r): per[r]["epochs"] for r in base}
    res["reported"] = rep
    return res


# ------------------------------------------------------------------------------ render
def s3(x):
    return "not read" if x is None else f"{x['skill'][0]:+.3f} [{x['skill'][1]:+.3f}, {x['skill'][2]:+.3f}]"


def pr(x):
    return "-" if x is None else ", ".join(f"r{r} {v:+.3f}" for r, v in x["per_repeat"].items())


def render(res):
    S, P = res["statements"], res["reported"]
    L = ["# B.59 — what the estimator must read, with pooled power", "",
         f"Registered: {REG}. GPU runs (one training per fold, initialisation seed of the fold); read-outs on a CPU. T_i, cut "
         "population, the fifteen folds of re-partitions r2, r3, r4 of the pooled cross-validation of B.39. Skill of A over R = "
         "1 - sum of squared errors of A / that of R on the same rows; 95 % two-level interval (a re-partition drawn among those "
         "read, then its discharges resampled; 20,000 draws); point = mean over re-partitions.", ""]
    if res["partial"]:
        L += [f"**Partial read-out**: {len(res['missing_runs'])} runs missing ({', '.join(res['missing_runs'])}). Re-partitions read per "
              "network: " + "; ".join(f"{LAB[n]} {v}" for n, v in res["repeats_read"].items()) + ".", ""]
    L += ["## Statements", "", "| statement | skill (interval) | per re-partition | holds | expectation (section 6) | point inside the expected range |",
          "|---|---|---|---|---|---|"]
    txt = {"L1": "L over the twin, interval above zero", "L2": "estimator over L, interval above zero",
           "N1": "N over the twin, interval above zero", "N2": "estimator over N, interval inside [-0.01, +0.01]",
           "E1": "E over the twin, interval above zero", "B1": "B over the twin, interval above zero"}
    for name, x in S.items():
        if not x["read"]:
            L.append(f"| **{name}** {txt[name]} | not read | - | - | {EXPECT[name]} | - |")
            continue
        L.append(f"| **{name}** {txt[name]} | {s3(x)} | {pr(x)} | {'holds' if x['holds'] else 'does not hold'}"
                 f"{'' if x['complete'] else ' (partial)'} | {x['expectation']} | {x['point_inside_expected_range']} |")
    L += ["", "## Every network over the references", "", "| network | over the causal GP | over hold | over the offline GP |", "|---|---|---|---|"]
    for n, d in P["over_refs"].items():
        L.append(f"| {LAB[n]} | {s3(d['gp_causal'])} | {s3(d['persistence'])} | {s3(d['gp'])} |")
    L += ["", "| comparison | skill (interval) | per re-partition |", "|---|---|---|"]
    names = {"est_over_E": "estimator over E", "est_over_B": "estimator over B", "B_over_E": "B over E", "est_over_twin": "estimator over the twin"}
    for key, x in P["pairs"].items():
        L.append(f"| {names[key]} | {s3(x)} | {pr(x)} |")
    L += ["", "## Over the twin, by rows", "",
          "Rows 1 to 3 after an absent frame: run_b54 row location (data/.b54/rows_r{r}_f{k}.npz, flag blip13) on the rows of the estimator. "
          "Rows per re-partition: " + "; ".join(f"r{r}: {v['rows']:,} rows, {v['blip_rows']:,} after an absent frame, {v['discharges']} discharges"
                                               for r, v in res["rows"].items()) + ".", "",
          "| network | rows 1 to 3 after an absent frame | other rows |", "|---|---|---|"]
    for n in P["blip_rows_over_twin"]:
        L.append(f"| {LAB[n]} | {s3(P['blip_rows_over_twin'][n])} | {s3(P['other_rows_over_twin'][n])} |")
    L += ["", "## Median absolute errors (eV, mean over re-partitions)", "", "| network or reference | median absolute error |", "|---|---|"]
    for n, v in P["median_abs_error_eV"].items():
        L.append(f"| {n} | {v:.1f} |")
    L += ["", "## Epochs run (best epoch)", "", "| re-partition | runs |", "|---|---|"]
    for r, d in P["epochs_run_best"].items():
        L.append(f"| r{r} | " + "; ".join(f"{k} {v[0]} ({v[1]})" for k, v in d.items()) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="read the re-partitions each network has complete")
    args = ap.parse_args()
    res = readout(args.partial)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print("written", OUT_MD, "| partial:", res["partial"])


if __name__ == "__main__":
    main()
