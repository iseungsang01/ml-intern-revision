"""B.68 read-outs (PREREGISTRATION_B68.md sections 3, 4 and 6).  CPU; reads comparison_errors_test.npz of the runs.

Statements (re-partitions r2, r3, r4 pooled, cut population; two-level interval: a re-partition is drawn among the
three, then its discharges are resampled, 20,000 draws -- the interval of analyze_b39.arm_p, as in analyze_b59 /
analyze_b67):
  X1  X (seq_v2 trained with the auxiliary target) over the network of B.39 (v2): interval above zero
  X2  X over the twin of B.39 (v2h): interval above zero
  X3  median absolute error of X / that of the causal GP: interval below 1 (the same draws for both medians: a
      re-partition drawn, its discharges resampled; each median is the weighted median of the absolute errors of
      the rows with their discharge's resampling count as weight; point = mean over re-partitions of the ratio of
      the plain medians)
Reported: X, the network and the twin over the causal GP, hold and the offline GP; the network over the twin; in rows
1 to 3 after an absent frame (B.54 row table, data/.b54/rows_r{r}_f{k}.npz, flag blip13) and in the other rows, X over
the network, X over the twin and the network over the twin; median absolute errors and their ratios to the causal GP
(X, the network, the twin) and X / the network, with intervals; epochs; the auxiliary-target record of each run
(b68_aux.json); the expectations of section 6 beside the findings.

    py ces_prediction/experiments/b68_auxtarget/analyze_b68.py [--partial] [--out-dir DIR] [--selftest]
Output: docs/paper/b68_auxtarget.json / .md.  Without --partial it stops, listing what is missing, unless all 15 runs
and the 30 B.39 references are present; with --partial X is read on the re-partitions whose five folds it has (zero
B.68 runs: every statement is 'not read', exit 0).  --selftest (check_b68 only) reads the B.39 network in place of X;
it must be written outside docs/paper.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP / "b39_primary"))
sys.path.insert(0, str(EXP / "b54_pooled_rows"))
import run_b68 as R  # noqa: E402
import analyze_b39 as A39  # noqa: E402  (load, same_rows, sums, ci)

REPO = HERE.parents[2]
PAPER = REPO / "docs" / "paper"
B54C = R.DATA / ".b54"
REG = "ces_prediction/experiments/PREREGISTRATION_B68.md"
NB, SEED = 20_000, 20261168
K = 5
CHUNK = 100
NETS = ("est", "twin", "X")
LAB = {"est": "network v2 (B.39)", "twin": "twin v2h (B.39)", "X": "X: v2 trained with the auxiliary target"}
REFS = {"gp_causal": "causal GP", "persistence": "hold", "gp": "offline GP"}
STATEMENTS = {"X1": ("X", "est"), "X2": ("X", "twin")}
STEXT = {"X1": "X over the network (v2), interval above zero", "X2": "X over the twin (v2h), interval above zero",
         "X3": "median absolute error of X / that of the causal GP, interval below 1"}
EXPECT = {"X1": ("-0.01 to +0.03, uncertain", -0.01, 0.03, None), "X2": ("+0.04 to +0.07, met", 0.04, 0.07, True),
          "X3": ("uncertain; the network's median error is about that of the causal GP", None, None, None)}
SELFTEST = {"on": False}


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def run_path(net, r, k):
    if net == "est" or (SELFTEST["on"] and net == "X"):
        return R.estimator_dir(r, k)
    if net == "twin":
        return R.twin_dir(r, k)
    return R.run_dir(r, k)


def have(net, r, k):
    return (run_path(net, r, k) / "comparison_errors_test.npz").exists()


# ------------------------------------------------------------------------------ data
def repeat_data(r, nets):
    import run_b54
    cat = {q: [] for q in (*nets, *REFS, "shot", "blip")}
    epochs, aux = {}, {}
    for k in range(K):
        z = {n: A39.load(run_path(n, r, k)) for n in nets}
        for n in nets:
            A39.same_rows(z["est"], z[n], f"r{r} f{k} {n} vs the network")
        t = run_b54.load_rows(r, k)
        if not (np.array_equal(t["shot"], z["est"]["shot"]) and np.array_equal(t["row"], z["est"]["row"])):
            raise SystemExit(f"FATAL: r{r} f{k}: the B.54 row table is not on the rows of the network")
        cat["shot"].append(z["est"]["shot"])
        cat["blip"].append(t["blip13"].astype(bool))
        for n in nets:
            cat[n].append(z[n]["model"])
        for q in REFS:
            cat[q].append(z["est"][q])
        for n in nets:
            m = json.loads((run_path(n, r, k) / "metrics.json").read_text(encoding="utf-8"))
            epochs[f"{n} f{k}"] = [m["epochs_run"], m["best_epoch"]]
            if n == "X" and not SELFTEST["on"]:
                if m["seq_model"] != "v2":
                    raise SystemExit(f"FATAL: {R.run_dir(r, k).name} was not trained as v2")
                s = json.loads((R.run_dir(r, k) / R.SIDECAR).read_text(encoding="utf-8"))
                val = [p for p in s["passes"] if p["phase"] == "val"]
                b = m["best_epoch"] - 1
                aux[f"f{k}"] = {"weight": s["aux_weight"], "train_share_rows_with_target": s["train"]["share_rows_with_aux_target"],
                                "train_share_ti_rows_with_target": s["train"]["share_ti_measured_rows_with_aux_target"],
                                "val_main_mse_best": val[b]["main_masked_mse"] if b < len(val) else None,
                                "val_aux_mse_best": val[b]["aux_masked_mse"] if b < len(val) else None,
                                "best_val_masked_mse": m["best_val_masked_mse"]}
    c = {q: np.concatenate(v) for q, v in cat.items()}
    shot, blip = c["shot"], c["blip"]
    se = {q: c[q] for q in (*nets, *REFS)}
    sums = {"all": {}, "blip": {}, "other": {}}
    for q, v in se.items():
        sums["all"][q] = A39.sums(shot, v)
        sums["blip"][q] = A39.sums(shot, np.where(blip, v, 0.0))
        sums["other"][q] = A39.sums(shot, np.where(blip, 0.0, v))
    _, f = np.unique(shot, return_inverse=True)
    return {"sums": sums, "abs": {q: np.sqrt(se[q]) for q in (*nets, *REFS)}, "fidx": f, "n_dis": int(f.max()) + 1,
            "rows": int(len(shot)), "discharges": int(len(np.unique(shot))), "blip_rows": int(blip.sum()),
            "epochs": epochs, "aux": aux, "nets": list(nets)}


# ------------------------------------------------------------------------------ intervals
def hier(per, a, b, repeats, group="all"):
    """analyze_b39.arm_p.hier (as analyze_b67.hier): a repeat is drawn, then its discharges are resampled with
    replacement.  Point = mean over repeats."""
    rng = rng_for(f"skill|{a}|{b}|{group}|{','.join(map(str, repeats))}")
    draws = rng.choice(np.array(repeats), size=NB)
    o = np.empty(NB)
    for r in repeats:
        sel = np.flatnonzero(draws == r)
        m, bb = per[r]["sums"][group][a], per[r]["sums"][group][b]
        idx = rng.integers(0, len(m), size=(len(sel), len(m)))
        o[sel] = 1.0 - m[idx].sum(1) / bb[idx].sum(1)
    pr = {str(r): float(1 - per[r]["sums"][group][a].sum() / per[r]["sums"][group][b].sum()) for r in repeats}
    return [float(np.mean(list(pr.values()))), *A39.ci(o)], pr


def wmedian_direct(v, f, W):
    """Weighted (lower) median of v (n,) for every row of the discharge weights W (draws, n_dis); f = discharge of
    each row; the weight of a row is its discharge's resampling count: the first value in sorted order at which the
    cumulative weight reaches half the total.  Direct cumulative sums (slow; check_b68 compares wmedian with it)."""
    o = np.argsort(v, kind="stable")
    vs, fo = v[o], f[o]
    out = np.empty(W.shape[0])
    for a in range(0, W.shape[0], CHUNK):
        c = np.cumsum(W[a:a + CHUNK][:, fo], axis=1, dtype=np.int64)
        out[a:a + CHUNK] = vs[(2 * c >= c[:, -1:]).argmax(1)]
    return out


def wmedian(v, f, W, buckets=512):
    """The same value as wmedian_direct, found in two steps: the sorted rows are cut into buckets, the weight of every
    bucket under every draw is one product W @ (discharge x bucket counts) (integers, exact in float64), and the
    crossing is then located row by row inside the one bucket where the cumulative weight reaches half."""
    o = np.argsort(v, kind="stable")
    vs, fo = v[o], f[o]
    n, D = len(v), W.shape[0]
    size = -(-n // buckets)
    nb = -(-n // size)
    cnt = np.zeros((W.shape[1], nb))
    np.add.at(cnt, (fo, np.arange(n) // size), 1.0)
    cb = np.cumsum(W.astype(np.float64) @ cnt, axis=1)
    tot = cb[:, -1]
    k = (2 * cb >= tot[:, None]).argmax(1)
    before = np.where(k > 0, cb[np.arange(D), np.maximum(k - 1, 0)], 0.0)
    rows = k[:, None] * size + np.arange(size)[None, :]
    real = rows < n
    rr = np.minimum(rows, n - 1)
    w = np.where(real, W[np.arange(D)[:, None], fo[rr]], 0).astype(np.float64)
    c = before[:, None] + np.cumsum(w, axis=1)
    j = (2 * c >= tot[:, None]).argmax(1)
    return vs[rr[np.arange(D), j]]


_MED = {}


def med_draws(per, q, repeats):
    """Weighted medians of the absolute errors of q under one fixed set of two-level draws per set of re-partitions (a
    re-partition drawn, its discharges resampled: multinomial counts), shared by every quantity, so that ratios are
    formed draw by draw."""
    key = tuple(repeats)
    if key not in _MED:
        rng = rng_for(f"median|{','.join(map(str, repeats))}")
        draws = rng.choice(np.array(repeats), size=NB)
        W = {r: rng.multinomial(per[r]["n_dis"], np.full(per[r]["n_dis"], 1.0 / per[r]["n_dis"]),
                                size=int(np.sum(draws == r))).astype(np.int32) for r in repeats}
        _MED[key] = {"draws": draws, "W": W, "med": {}}
    c = _MED[key]
    if q not in c["med"]:
        o = np.empty(NB)
        for r in repeats:
            o[np.flatnonzero(c["draws"] == r)] = wmedian(per[r]["abs"][q], per[r]["fidx"], c["W"][r])
        c["med"][q] = o
    return c["med"][q]


def med_ratio(per, a, b, repeats):
    """Median absolute error of a / that of b; point = mean over re-partitions of the ratio of the plain medians."""
    o = med_draws(per, a, repeats) / med_draws(per, b, repeats)
    pr = {str(r): float(np.median(per[r]["abs"][a]) / np.median(per[r]["abs"][b])) for r in repeats}
    return [float(np.mean(list(pr.values()))), *A39.ci(o)], pr


# ------------------------------------------------------------------------------ readout
def readout(partial):
    avail = {n: [r for r in R.REPEATS if all(have(n, r, k) for k in range(K))] for n in NETS}
    missing = [run_path(n, r, k).name for n in NETS for r in R.REPEATS for k in range(K) if not have(n, r, k)]
    if missing:
        print(f"[b68] missing ({len(missing)}): {', '.join(missing)}")
        if not partial:
            raise SystemExit("[b68] not every run is present; nothing written (use --partial to read what is complete)")
    base = [r for r in R.REPEATS if r in avail["est"] and r in avail["twin"]]
    if not base:
        raise SystemExit("[b68] the B.39 network and twin are not complete on any re-partition")
    for r in base:
        for k in range(K):
            if not (B54C / f"rows_r{r}_f{k}.npz").exists():
                raise SystemExit(f"FATAL: {B54C / f'rows_r{r}_f{k}.npz'} missing (B.54 row tables)")
    per = {}
    for r in base:
        nets = [n for n in NETS if r in avail[n]]
        per[r] = repeat_data(r, nets)
        print(f"[b68] r{r}: {per[r]['rows']:,} rows, {per[r]['discharges']} discharges, networks {nets}", flush=True)

    def reps(*nets):
        return [r for r in base if all(n in per[r]["nets"] for n in nets if n not in REFS)]

    def H(a, b, group="all"):
        rr = reps(a, b)
        if not rr:
            return None
        v, p = hier(per, a, b, rr, group)
        return {"skill": v, "per_repeat": p, "repeats": rr}

    def M(a, b):
        rr = reps(a, b)
        if not rr:
            return None
        v, p = med_ratio(per, a, b, rr)
        return {"ratio": v, "per_repeat": p, "repeats": rr}

    res = {"registration": REG, "seed": SEED, "draws": NB, "selftest": SELFTEST["on"], "partial": bool(missing),
           "missing": missing, "repeats_read": {n: reps(n) for n in NETS},
           "rows": {str(r): {"rows": per[r]["rows"], "discharges": per[r]["discharges"], "blip_rows": per[r]["blip_rows"]}
                    for r in base}}
    st = {}
    for name, (a, b) in STATEMENTS.items():
        h = H(a, b)
        if h is None:
            st[name] = {"read": False}
            continue
        txt, lo, hi, exp_holds = EXPECT[name]
        st[name] = {"read": True, "numerator": a, "denominator": b, **h, "holds": bool(h["skill"][1] > 0),
                    "complete": len(h["repeats"]) == len(R.REPEATS), "expectation": txt,
                    "point_inside_expected_range": bool(lo <= h["skill"][0] <= hi),
                    "verdict_as_expected": None if exp_holds is None else bool((h["skill"][1] > 0) == exp_holds)}
    x3 = M("X", "gp_causal")
    st["X3"] = {"read": False} if x3 is None else {
        "read": True, **x3, "holds": bool(x3["ratio"][2] < 1.0), "complete": len(x3["repeats"]) == len(R.REPEATS),
        "expectation": EXPECT["X3"][0], "point_inside_expected_range": None, "verdict_as_expected": None}
    res["statements"] = st
    rep = {"over_refs": {}, "pairs": {"est_over_twin": H("est", "twin")}, "blip_rows": {}, "other_rows": {},
           "median_ratio": {}}
    for n in NETS:
        if reps(n):
            rep["over_refs"][n] = {q: H(n, q) for q in REFS}
    for a, b in (("X", "est"), ("X", "twin"), ("est", "twin")):
        rep["blip_rows"][f"{a}_over_{b}"] = H(a, b, "blip")
        rep["other_rows"][f"{a}_over_{b}"] = H(a, b, "other")
    for a, b in (("X", "gp_causal"), ("est", "gp_causal"), ("twin", "gp_causal"), ("X", "est")):
        rep["median_ratio"][f"{a}_over_{b}"] = x3 if (a, b) == ("X", "gp_causal") else M(a, b)
    rep["median_abs_error_eV"] = {LAB[n]: float(np.mean([np.median(per[r]["abs"][n]) for r in reps(n)]))
                                  for n in NETS if reps(n)}
    rep["median_abs_error_eV"].update({REFS[q]: float(np.mean([np.median(per[r]["abs"][q]) for r in base])) for q in REFS})
    rep["epochs_run_best"] = {str(r): per[r]["epochs"] for r in base}
    rep["aux_training"] = {str(r): per[r]["aux"] for r in base if per[r]["aux"]}
    res["reported"] = rep
    return res


# ------------------------------------------------------------------------------ render
def s3(x):
    return "not read" if x is None else f"{x['skill'][0]:+.3f} [{x['skill'][1]:+.3f}, {x['skill'][2]:+.3f}]"


def r3(x):
    return "not read" if x is None else f"{x['ratio'][0]:.3f} [{x['ratio'][1]:.3f}, {x['ratio'][2]:.3f}]"


def pr(x, fmt="+.3f"):
    return "-" if x is None else ", ".join(f"r{r} {v:{fmt}}" for r, v in x["per_repeat"].items())


def render(res):
    S, P = res["statements"], res["reported"]
    L = ["# B.68 — an auxiliary training target denoised by the neighbouring values", ""]
    if res["selftest"]:
        L += ["**SELFTEST** (check_b68): the B.39 network stands in for X. Not a result.", ""]
    L += [f"Registered: {REG}. GPU runs (one training per fold, initialisation seed of the fold); read-outs on a CPU. T_i, "
          "cut population, the fifteen folds of re-partitions r2, r3, r4 of the pooled cross-validation of B.39. X is seq_v2 "
          "trained with one more T_i head on the offline-GP estimate of T_i (the row's own value withheld; weight 0.5); "
          "it is scored with its main head only. Skill of A over B = 1 - sum of squared errors of A / that of B on the "
          "same rows; 95 % two-level interval (a re-partition drawn among those read, then its discharges resampled; "
          "20,000 draws); point = mean over re-partitions.", ""]
    if res["partial"]:
        L += [f"**Partial read-out**: {len(res['missing'])} runs missing. Re-partitions read per network: "
              + "; ".join(f"{LAB[n]} {v}" for n, v in res["repeats_read"].items()) + ".", ""]
    L += ["## Statements", "", "| statement | value (interval) | per re-partition | holds | expectation (section 6) | "
          "point inside the expected range |", "|---|---|---|---|---|---|"]
    for name in ("X1", "X2", "X3"):
        x = S[name]
        if not x["read"]:
            L.append(f"| **{name}** {STEXT[name]} | not read | - | - | {EXPECT[name][0]} | - |")
            continue
        val = r3(x) if name == "X3" else s3(x)
        per_r = pr(x, ".3f" if name == "X3" else "+.3f")
        inside = "-" if x["point_inside_expected_range"] is None else x["point_inside_expected_range"]
        L.append(f"| **{name}** {STEXT[name]} | {val} | {per_r} | {'holds' if x['holds'] else 'does not hold'}"
                 f"{'' if x['complete'] else ' (partial)'} | {x['expectation']} | {inside} |")
    L += ["", "## Every network over the references", "", "| network | over the causal GP | over hold | over the offline GP |",
          "|---|---|---|---|"]
    for n, d in P["over_refs"].items():
        L.append(f"| {LAB[n]} | {s3(d['gp_causal'])} | {s3(d['persistence'])} | {s3(d['gp'])} |")
    L += ["", f"Network over the twin: {s3(P['pairs']['est_over_twin'])} ({pr(P['pairs']['est_over_twin'])}).", "",
          "## By rows", "", "Rows 1 to 3 after an absent frame: B.54 row table (flag blip13) on the rows of the network. "
          "Rows per re-partition: " + "; ".join(f"r{r}: {v['rows']:,} rows, {v['blip_rows']:,} after an absent frame, "
                                               f"{v['discharges']} discharges" for r, v in res["rows"].items()) + ".", "",
          "| comparison | rows 1 to 3 after an absent frame | other rows |", "|---|---|---|"]
    lab2 = {"X": "X", "est": "network", "twin": "twin"}
    for key in P["blip_rows"]:
        a, b = key.split("_over_")
        L.append(f"| {lab2[a]} over {lab2[b]} | {s3(P['blip_rows'][key])} | {s3(P['other_rows'][key])} |")
    L += ["", "## Median absolute errors", "", "Point: mean over re-partitions of the plain median (eV) or of the ratio of "
          "the plain medians; interval: the two-level draws, each median the weighted median of the rows with their "
          "discharge's resampling count as weight, the same draws for numerator and denominator.", "",
          "| network or reference | median absolute error (eV) |", "|---|---|"]
    for n, v in P["median_abs_error_eV"].items():
        L.append(f"| {n} | {v:.1f} |")
    names = {"X_over_gp_causal": "X / causal GP (X3)", "est_over_gp_causal": "network / causal GP",
             "twin_over_gp_causal": "twin / causal GP", "X_over_est": "X / network"}
    L += ["", "| ratio of median absolute errors | value (interval) | per re-partition |", "|---|---|---|"]
    for key, x in P["median_ratio"].items():
        L.append(f"| {names[key]} | {r3(x)} | {pr(x, '.3f')} |")
    if P["aux_training"]:
        L += ["", "## The auxiliary target in training (b68_aux.json of each run)", "",
              "| re-partition | fold | weight | training rows with a target | measured T_i rows with a target | validation MSE "
              "of the main heads / of the auxiliary head at the best epoch (standardized) |", "|---|---|---|---|---|---|"]
        for r, d in P["aux_training"].items():
            for f, a in d.items():
                vm = "-" if a["val_main_mse_best"] is None else f"{a['val_main_mse_best']:.4f} / {a['val_aux_mse_best']:.4f}"
                L.append(f"| r{r} | {f} | {a['weight']} | {a['train_share_rows_with_target']:.4f} | "
                         f"{a['train_share_ti_rows_with_target']:.4f} | {vm} |")
    L += ["", "## Epochs run (best epoch)", "", "| re-partition | runs |", "|---|---|"]
    for r, d in P["epochs_run_best"].items():
        L.append(f"| r{r} | " + "; ".join(f"{k} {v[0]} ({v[1]})" for k, v in d.items()) + " |")
    L += ["", "## Choices fixed in the implementation", "",
          "- Networks of B.39 on the same folds: `.b39p_v2_cut_r{r}_f{k}` (the network) and `.b39p_v2h_cut_r{r}_f{k}` (the twin).",
          "- The auxiliary target is the offline-GP estimator of the B.39 harness evaluated on the harness's file array "
          "(rows with at least one measured target); a grid row with no measured target is inserted into that array at its "
          "time. It is NaN (masked) where the estimator gives no value (no earlier measured T_i in the block).",
          "- X3 interval: the weighted lower median under the discharge resampling counts; the point is the plain median.",
          ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="read the re-partitions each network has complete")
    ap.add_argument("--out-dir", default=str(PAPER))
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.selftest:
        if out_dir.resolve() == PAPER.resolve():
            raise SystemExit("FATAL: --selftest output must not go to docs/paper")
        SELFTEST["on"] = True
    res = readout(args.partial)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "b68_auxtarget.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out_dir / "b68_auxtarget.md").write_text(render(res), encoding="utf-8", newline="\n")
    print("written", out_dir / "b68_auxtarget.md", "| partial:", res["partial"], "| selftest:", res["selftest"])


if __name__ == "__main__":
    main()
