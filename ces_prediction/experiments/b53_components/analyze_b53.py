"""B.53 read-outs (PREREGISTRATION_B53.md sections 4-6).  CPU; reads comparison_errors_test.npz of the runs.

Statements (four splits; a statement holds if it holds on at least three):
  C1  D (differences) over the twin, interval above zero      C2  C (common mode) over the twin
  F1  fixed-budget estimator over the fixed-budget twin       F2  fixed-budget estimator over the causal GP
Reported: CD over the twin, over C and over D; D, C, CD against the BES network of B.39 and against the
causal GP; the fixed-budget twin against the early-stopped twin, the fixed-budget estimator against the
early-stopped one, and the increment under the fixed budget minus the increment of B.39 at the same
initialisation; epochs and validation losses; median absolute errors.

Skill of A over R = 1 - (sum of squared T_i errors of A) / (that of R) on the same test rows; interval 95 %,
2,000 resamples of the test discharges (analyze_b39.two_level with one initialisation); the difference of
two increments is resampled jointly (the same discharges for all four networks).

    py ces_prediction/experiments/b53_components/analyze_b53.py [--partial]
Output: docs/paper/b53_components.json / .md.  Without --partial it stops, listing what is missing, unless
every run of the batch and every B.39 reference is present.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b39_primary"))
import run_b53 as R  # noqa: E402
import analyze_b39 as A39  # noqa: E402  (load, same_rows, sums, ci, two_level, med)

REPO = HERE.parents[2]
OUT_JSON = REPO / "docs" / "paper" / "b53_components.json"
OUT_MD = REPO / "docs" / "paper" / "b53_components.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B53.md"
SEED = 20261118
N_BOOT = A39.N_BOOT          # 2,000 (registered)


def dirs(s):
    B39 = R.B39
    return {"C": R.c_dir("C", s), "D": R.c_dir("D", s), "CD": R.c_dir("CD", s),
            "Fv2": R.f_dir("v2", s), "Fv2h": R.f_dir("v2h", s),
            "twin": B39.twin_dir(s, s), "est": B39.split_run_dir("v2", s, s), "bes": B39.split_run_dir("v2b", s, s)}


def missing(s):
    return [d.name for d in dirs(s).values() if not (d / "comparison_errors_test.npz").exists()]


def one(rng, a, b, shot):
    return A39.two_level(A39.sums(shot, a)[None, :], A39.sums(shot, b)[None, :], rng)


def incr_diff(rng, shot, a1, b1, a2, b2):
    """(1 - a1/b1) - (1 - a2/b2) with the same discharges drawn for all four."""
    S = [A39.sums(shot, v) for v in (a1, b1, a2, b2)]
    n = len(S[0])
    idx = rng.integers(0, n, size=(N_BOOT, n))
    t = [np.take_along_axis(np.broadcast_to(x, (N_BOOT, n)), idx, 1).sum(1) for x in S]
    pt = float((1 - S[0].sum() / S[1].sum()) - (1 - S[2].sum() / S[3].sum()))
    return [pt, *A39.ci((1 - t[0] / t[1]) - (1 - t[2] / t[3]))]


def last_val(d):
    """Validation masked MSE of the last epoch trained (run.log), the one whose weights a fixed-budget run keeps."""
    p = d / "run.log"
    if not p.exists():
        return None
    v = [ln for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.startswith("[seq] epoch ")]
    if not v:
        return None
    return float(v[-1].split("val_mse=")[1].split()[0])


def run_meta(d):
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    return {"epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"], "best_val_masked_mse": m["best_val_masked_mse"],
            "last_val_masked_mse": last_val(d),
            "fixed_epochs": m["fixed_epochs"], "seq_model": m["seq_model"], "extra_ti_channels": m["extra_ti_channels"],
            "n_params": m["n_params"]}


def read_split(s, rng):
    D = dirs(s)
    z = {k: A39.load(d) for k, d in D.items()}
    for k in z:
        A39.same_rows(z["twin"], z[k], f"s{s} {k} vs twin")
    meta = {k: run_meta(d) for k, d in D.items()}
    for k, kk in (("C", 1), ("D", 4), ("CD", 5)):
        if meta[k]["seq_model"] != "v2hx" or meta[k]["extra_ti_channels"] != kk:
            raise SystemExit(f"FATAL: s{s} {k}: metrics.json says {meta[k]['seq_model']} with {meta[k]['extra_ti_channels']} channels")
    for k, mdl in (("Fv2", "v2"), ("Fv2h", "v2h")):
        if meta[k]["seq_model"] != mdl or not meta[k]["fixed_epochs"] or meta[k]["epochs_run"] != int(R.F_EPOCHS):
            raise SystemExit(f"FATAL: s{s} {k}: not a 40-epoch fixed-budget run of {mdl}: {meta[k]}")
    shot = z["twin"]["shot"]
    se = {k: v["model"] for k, v in z.items()}
    gp = z["twin"]["gp_causal"]
    o = {"rows": int(len(shot)), "discharges": int(len(np.unique(shot)))}
    o["C1_D_over_twin"] = one(rng, se["D"], se["twin"], shot)
    o["C2_C_over_twin"] = one(rng, se["C"], se["twin"], shot)
    o["CD_over_twin"] = one(rng, se["CD"], se["twin"], shot)
    o["CD_over_C"] = one(rng, se["CD"], se["C"], shot)
    o["CD_over_D"] = one(rng, se["CD"], se["D"], shot)
    o["over_bes_network"] = {k: one(rng, se[k], se["bes"], shot) for k in ("C", "D", "CD")}
    o["over_causal_gp"] = {k: one(rng, se[k], gp, shot) for k in ("C", "D", "CD", "Fv2", "Fv2h", "twin", "est", "bes")}
    o["bes_network_over_twin"] = one(rng, se["bes"], se["twin"], shot)
    o["F1_est_over_twin_fixed"] = one(rng, se["Fv2"], se["Fv2h"], shot)
    o["F2_est_over_gp_fixed"] = o["over_causal_gp"]["Fv2"]
    o["twin_fixed_over_twin_b39"] = one(rng, se["Fv2h"], se["twin"], shot)
    o["est_fixed_over_est_b39"] = one(rng, se["Fv2"], se["est"], shot)
    o["increment_b39"] = one(rng, se["est"], se["twin"], shot)
    o["increment_fixed_minus_b39"] = incr_diff(rng, shot, se["Fv2"], se["Fv2h"], se["est"], se["twin"])
    o["median_abs_error_eV"] = {"hold": A39.med(z["twin"]["persistence"]), "causal GP": A39.med(gp),
                                **{k: A39.med(v) for k, v in se.items()}}
    o["runs"] = meta
    for k in ("C1_D_over_twin", "C2_C_over_twin", "F1_est_over_twin_fixed", "F2_est_over_gp_fixed"):
        o[k.split("_")[0] + "_holds"] = bool(o[k][1] > 0)
    return o


def summary(res):
    done = [s for s in R.SEEDS if str(s) in res["splits"]]
    out = {"splits_read": done}
    for st in ("C1", "C2", "F1", "F2"):
        n = sum(res["splits"][str(s)][f"{st}_holds"] for s in done)
        out[st] = {"splits_holding": n, "splits_read": len(done), "holds": bool(n >= 3)}
    c1, c2 = out["C1"]["holds"], out["C2"]["holds"]
    out["paper_branch_C"] = ("C1 holds: the differences alone keep the increment" if c1 else
                             "C1 does not hold, C2 holds: a trained network needs the common mode" if c2 else
                             "neither holds: the extra-input route is a weaker reader than the BES network")
    out["paper_branch_F"] = ("F1 and F2 hold: the increment does not rest on early stopping"
                             if out["F1"]["holds"] and out["F2"]["holds"] else "F1 or F2 fails: stated as a limit")
    return out


def render(res):
    s3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                   # noqa: E731
    S, P = res["summary"], res["splits"]
    done = S["splits_read"]
    L = ["# B.53 — networks that read one component of the BES array, and training under a fixed budget", "",
         f"Registered: {REG}. GPU runs (one initialisation = split seed); read-outs on a CPU. T_i, cut population, test rows "
         "of every earlier read-out. Skill of A over R = 1 - sum of squared errors of A / that of R on the same rows; 95 % "
         "interval from 2,000 resamples of the test discharges (one initialisation, so the spread between initialisations is "
         "not in it).", "", f"Splits read: {len(done)} of 4." + (f" Missing: {res['missing']}" if res.get("missing") else ""), ""]
    for st, txt in (("C1", "D (differences) over the twin"), ("C2", "C (common mode) over the twin"),
                    ("F1", "fixed-budget estimator over the fixed-budget twin"), ("F2", "fixed-budget estimator over the causal GP")):
        L.append(f"- **{st}** ({txt}): interval above zero on {S[st]['splits_holding']} of {S[st]['splits_read']} -> "
                 f"{'holds' if S[st]['holds'] else 'does not hold'}.")
    L += ["", f"Paper (section 5): {S['paper_branch_C']}; {S['paper_branch_F']}.", "", "## Arm C: components", "",
          "| split | rows / discharges | C1: D over twin | C2: C over twin | CD over twin | BES network (B.39) over twin | CD over C | CD over D |",
          "|---|---|---|---|---|---|---|---|"]
    for s in done:
        o = P[str(s)]
        L.append(f"| {s} | {o['rows']:,} / {o['discharges']} | {s3(o['C1_D_over_twin'])} | {s3(o['C2_C_over_twin'])} | "
                 f"{s3(o['CD_over_twin'])} | {s3(o['bes_network_over_twin'])} | {s3(o['CD_over_C'])} | {s3(o['CD_over_D'])} |")
    L += ["", "| split | C over BES network | D over BES network | CD over BES network | C over causal GP | D over causal GP | CD over causal GP |",
          "|---|---|---|---|---|---|---|"]
    for s in done:
        o = P[str(s)]
        L.append(f"| {s} | {s3(o['over_bes_network']['C'])} | {s3(o['over_bes_network']['D'])} | {s3(o['over_bes_network']['CD'])} | "
                 f"{s3(o['over_causal_gp']['C'])} | {s3(o['over_causal_gp']['D'])} | {s3(o['over_causal_gp']['CD'])} |")
    L += ["", "## Arm F: fixed budget of 40 epochs", "",
          "| split | F1: estimator over twin (fixed) | F2: estimator over causal GP (fixed) | increment of B.39 (same init) | "
          "fixed minus B.39 increment | twin fixed over twin B.39 | estimator fixed over estimator B.39 |", "|---|---|---|---|---|---|---|"]
    for s in done:
        o = P[str(s)]
        L.append(f"| {s} | {s3(o['F1_est_over_twin_fixed'])} | {s3(o['F2_est_over_gp_fixed'])} | {s3(o['increment_b39'])} | "
                 f"{s3(o['increment_fixed_minus_b39'])} | {s3(o['twin_fixed_over_twin_b39'])} | {s3(o['est_fixed_over_est_b39'])} |")
    L += ["", "## Epochs, validation losses, median absolute errors", "",
          "| split | network | epochs run (best) | best validation masked MSE | last-epoch validation masked MSE | median absolute error (eV) |",
          "|---|---|---|---|---|---|"]
    lab = {"C": "C", "D": "D", "CD": "CD", "Fv2": "estimator, 40 fixed", "Fv2h": "twin, 40 fixed", "twin": "twin (B.39)",
           "est": "estimator (B.39)", "bes": "BES network (B.39)"}
    for s in done:
        o = P[str(s)]
        for k, name in lab.items():
            r = o["runs"][k]
            lv = "-" if r["last_val_masked_mse"] is None else f"{r['last_val_masked_mse']:.4f}"
            L.append(f"| {s} | {name} | {r['epochs_run']} ({r['best_epoch']}) | {r['best_val_masked_mse']:.4f} | {lv} | "
                     f"{o['median_abs_error_eV'][k]:.1f} |")
        L.append(f"| {s} | hold / causal GP | - | - | - | {o['median_abs_error_eV']['hold']:.1f} / {o['median_abs_error_eV']['causal GP']:.1f} |")
    L += ["", "Note: under the fixed budget the scored weights are those of epoch 40 (CES_SEQ_FIXED_EPOCHS=1), whose validation "
          "loss is the last-epoch column; 'best' is the best epoch seen, kept only by the early-stopped runs."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="read the splits that are complete")
    args = ap.parse_args()
    miss = {s: missing(s) for s in R.SEEDS}
    if any(miss.values()):
        for s, m in miss.items():
            if m:
                print(f"[b53] split {s}: missing comparison_errors_test.npz in {', '.join(m)}")
        if not args.partial or all(miss.values()):
            raise SystemExit("[b53] not every run is present; nothing written" + ("" if args.partial else " (use --partial to read complete splits)"))
    rng = np.random.default_rng(SEED)
    res = {"registration": REG, "seed": SEED, "n_boot": N_BOOT,
           "missing": {str(s): m for s, m in miss.items() if m},
           "splits": {str(s): read_split(s, rng) for s in R.SEEDS if not miss[s]}}
    res["summary"] = summary(res)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print("written", OUT_MD, "| splits:", res["summary"]["splits_read"])


if __name__ == "__main__":
    main()
