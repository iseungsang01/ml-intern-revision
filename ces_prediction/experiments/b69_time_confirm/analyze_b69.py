"""B.69 readouts (PREREGISTRATION_B69.md, sections 3 and 4).  CPU; reads the outer runs' comparison_errors_test.npz and
the second stage of stage2_b69.py.

Rows: the T_i rows of the test block scored by eval_seq (as in B.39), cut population; the same rows for the network,
the twin and the four seeds.  Estimators: the two-stage network (c2), the two-stage twin (t2), the network (c), the twin
(t), hold, the causal GP, the offline GP, PCHIP.  Intervals: 95 %, B.64's two-level bootstrap with the four seeds in
place of the re-partitions (`stage2_b64.cell` / `stage2_b64.medians`, imported): each draw takes one seed uniformly,
then resamples the test discharges; 20,000 draws for sums, 2,000 for medians.  Point = mean over the seeds.

Statements: C1 two-stage network over the causal GP, interval above zero; C2 over the network, above zero; C3 over the
two-stage twin, above zero; C4 median absolute error of the two-stage network over that of hold, interval below 1.

    py ces_prediction/experiments/b69_time_confirm/analyze_b69.py [--partial] [--out-dir DIR]
Output: docs/paper/b69_time_confirm.json / .md.  Without --partial it stops and writes nothing unless the eight outer
runs and the four second stages are complete.  With --partial it reads the seeds whose two outer runs are scored (the
first stage only, unless every such seed also has its second stage) and marks the output PARTIAL; with no scored seed
it exits cleanly and writes nothing.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage2_b69 as S2  # noqa: E402

S64, B54, P69 = S2.S64, S2.B54, S2.P69
B61 = S64.B61
REG = P69.REG
SEEDS = P69.SEEDS
TAG = S2.TAG
PAPER = P69.DATA.parent / "docs" / "paper"
N_BOOT, N_BOOT_Q = S64.N_BOOT, S64.N_BOOT_Q       # 20,000 / 2,000
LAB = {"c2": "two-stage network", "t2": "two-stage twin", "c": "network", "t": "twin", "hold": "hold",
       "gp": "causal GP", "off": "offline GP", "pchip": "PCHIP"}
EXPECT = {"C1": (0.12, 0.20), "C2": (0.05, 0.10), "C3": (0.04, 0.10), "C4": (0.88, 0.95)}
EST_FULL = ("c2", "t2", "c", "t", "hold", "gp", "off", "pchip")
PAIRS_FULL = [("c2", "c"), ("c2", "gp"), ("c2", "t2"), ("c2", "hold"), ("c2", "pchip"), ("c2", "off"),
              ("c", "gp"), ("c", "t"), ("c", "hold"), ("c", "pchip"), ("c", "off"),
              ("t2", "t"), ("t2", "gp"), ("t2", "hold"), ("t", "gp")]
MED_FULL = [("c2", "hold"), ("c2", "c"), ("c2", "gp"), ("c2", "t2"), ("c", "hold"), ("c", "gp"), ("t2", "hold")]
EST_FIRST = ("c", "t", "hold", "gp", "off", "pchip")
PAIRS_FIRST = [("c", "gp"), ("c", "t"), ("c", "hold"), ("c", "pchip"), ("c", "off"), ("t", "gp"), ("t", "hold")]
MED_FIRST = [("c", "hold"), ("c", "gp"), ("c", "t")]
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


# ------------------------------------------------------------------------------------------ data
def outer_scored(layout, s):
    return all((layout.outer_dir(a, s, TAG) / "comparison_errors_test.npz").exists() for a in P69.MODELS)


def seed_data(layout, s, second):
    c = B54.load_run(layout.outer_dir("v2", s, TAG))
    t = B54.load_run(layout.outer_dir("v2h", s, TAG))
    if not (len(c["shot"]) == len(t["shot"]) and np.array_equal(c["shot"], t["shot"]) and np.array_equal(c["row"], t["row"])):
        raise SystemExit(f"FATAL: s{s}: the network and the twin were not scored on the same rows")
    for q in ("y", "hold", "gp", "off", "pchip"):
        if not np.allclose(c[q], t[q], rtol=1e-6, atol=1e-6, equal_nan=True):
            raise SystemExit(f"FATAL: s{s}: the reference '{q}' differs between the network's and the twin's scoring")
    y = c["y"]
    u, f = np.unique(c["shot"], return_inverse=True)
    se = {"c": c["se"], "t": t["se"], "hold": c["hold"], "gp": c["gp"], "off": c["off"], "pchip": c["pchip"]}
    d = {"y": y, "f": f, "n": len(u), "se": se, "shot": c["shot"], "row": c["row"]}
    if second:
        z = np.load(S2.stage2_path(layout, s))
        if len(z["y"]) != len(y) or np.max(np.abs(z["y"] - y)) > 1e-6:
            raise SystemExit(f"FATAL: s{s}: the second-stage rows are not the scored rows of the outer runs")
        se["c2"] = (z["est_c2"] - y) ** 2
        se["t2"] = (z["est_t2"] - y) ** 2
        d["first_stage_dev"] = {e: float(np.max(np.abs((z[f"est_{e}"] - y) ** 2 - se[e]))) for e in ("c", "t")}
        d["fallback"] = {e: np.asarray(z[f"fallback_{e}"], bool) for e in ("c2", "t2")}
        d["info"] = json.loads(str(z["info"]))
    if not all(np.isfinite(v).all() for v in se.values()):
        raise SystemExit(f"FATAL: s{s}: a squared error is not finite on a scored row")
    d["keep_trim"] = se["hold"] < np.quantile(se["hold"], 1 - B61.TRIM)        # run_b61.load_data / stage2_b64
    return d


def same_block(D):
    """The four seeds are scored on the same rows with the same references (bit for bit)."""
    s0 = next(iter(D))
    o = {}
    for s in D:
        o[str(s)] = bool(np.array_equal(D[s]["shot"], D[s0]["shot"]) and np.array_equal(D[s]["row"], D[s0]["row"])
                         and np.array_equal(D[s]["y"], D[s0]["y"])
                         and all(np.array_equal(D[s]["se"][q], D[s0]["se"][q]) for q in ("hold", "gp", "off", "pchip")))
    return o


def run_records(layout, seeds):
    recs = []
    for s in seeds:
        for a in P69.MODELS:
            for j in (None, *range(layout.n_inner)):
                d = layout.outer_dir(a, s, TAG) if j is None else layout.inner_dir(a, s, TAG, j)
                if (d / "metrics.json").exists():
                    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
                    recs.append({"run": d.name, "epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"],
                                 "cap_bound": bool(m["epochs_run"] >= int(P69.B39.EPOCH_CAP))})
    return recs


# ------------------------------------------------------------------------------------------ readout
def readout(layout, seeds, second, partial, plumbing=False):
    D = {s: seed_data(layout, s, second) for s in seeds}
    reps = tuple(seeds)
    ests, pairs, med = (EST_FULL, PAIRS_FULL, MED_FULL) if second else (EST_FIRST, PAIRS_FIRST, MED_FIRST)
    all_rows = lambda r: np.ones(len(D[r]["y"]), bool)                                   # noqa: E731
    log(f"bootstrap: seeds {list(reps)}, second stage {second}, {N_BOOT:,} draws (sums), {N_BOOT_Q:,} (medians)")
    A, Aper = S64.cell(D, all_rows, ests, pairs, B54.rng_for("b69|all"), nboot=N_BOOT, reps=reps)
    Q = S64.medians(D, ests, med, B54.rng_for("b69|medians"), nboot=N_BOOT_Q, reps=reps)
    s0 = seeds[0]
    o = {"registration": REG, "partial": bool(partial), "plumbing_only": bool(plumbing), "seeds": list(reps),
         "second_stage": bool(second), "n_boot": N_BOOT, "n_boot_quantiles": N_BOOT_Q,
         "rows": int(len(D[s0]["y"])), "discharges_with_rows": int(D[s0]["n"]),
         "test_block_discharges": len(layout.manifest(s0, TAG)["test_files"]),
         "all_rows": A, "all_rows_per_seed": Aper, "medians": Q,
         "checks": {"same_rows_and_references_across_seeds": same_block(D), "runs": run_records(layout, seeds)}}
    if second:
        o["checks"]["first_stage_equals_stored_max_abs_dev_se"] = {str(s): D[s]["first_stage_dev"] for s in seeds}
        o["fallback_rows_share"] = {e: float(np.mean(np.concatenate([D[s]["fallback"][e] for s in seeds])))
                                    for e in ("c2", "t2")}
        o["fit_info"] = {str(s): {a: {k: v for k, v in D[s]["info"][a].items() if k != "inner"} | {
            "inner": [{q: x[q] for q in ("run", "discharges", "rows", "epochs_run", "best_epoch", "skill_over_gp")}
                      for x in D[s]["info"][a]["inner"]]} for a in D[s]["info"]} for s in seeds}
        st = {"C1": {"value": A["skill_mse|c2|gp"], "holds": bool(A["skill_mse|c2|gp"][1] > 0),
                     "text": "two-stage estimator over the causal GP: interval above zero"},
              "C2": {"value": A["skill_mse|c2|c"], "holds": bool(A["skill_mse|c2|c"][1] > 0),
                     "text": "two-stage estimator over the network: interval above zero"},
              "C3": {"value": A["skill_mse|c2|t2"], "holds": bool(A["skill_mse|c2|t2"][1] > 0),
                     "text": "two-stage estimator over the two-stage twin: interval above zero"},
              "C4": {"value": Q["median_ratio|c2|hold"], "holds": bool(Q["median_ratio|c2|hold"][2] < 1),
                     "text": "median absolute error of the two-stage estimator over that of hold: interval below 1"}}
        for k, v in st.items():
            lo, hi = EXPECT[k]
            v["expected_range"] = [lo, hi]
            v["point_inside_expected_range"] = bool(lo <= v["value"][0] <= hi)
        o["statements"] = st
        o["statements_read"] = not partial
    o["runtime_s"] = round(_time.time() - T0, 1)
    return o


# ------------------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def f3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def e1(x):
    return f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"


def render(o):
    A, Q, P = o["all_rows"], o["medians"], o["all_rows_per_seed"]
    tag = " (PLUMBING ONLY, NOT A RESULT)" if o["plumbing_only"] else (" (PARTIAL)" if o["partial"] else "")
    L = [f"# B.69 -- the two-stage estimator on the later discharges: a confirmation out of time{tag}", "",
         f"Registered: {o['registration']}. T_i, cut population, the temporal split of B.5: test block = the last "
         f"{o['test_block_discharges']} discharges by shot number ({o['discharges_with_rows']} carry scored rows; "
         f"{o['rows']:,} rows), seeds {', '.join(map(str, o['seeds']))}. Rows: the rows scored by eval_seq, as in B.39. "
         f"Intervals: 95 %, a seed drawn uniformly, then the test discharges resampled; {o['n_boot']:,} draws for sums, "
         f"{o['n_boot_quantiles']:,} for medians. Point = mean over the seeds.", ""]
    if o["partial"]:
        L += [f"**Partial:** seeds read {o['seeds']}; second stage {'included' if o['second_stage'] else 'not yet run'}. "
              "The statements are not read on a partial batch.", ""]
    est = EST_FULL if o["second_stage"] else EST_FIRST
    if o["second_stage"]:
        for k, v in o["statements"].items():
            val = f3(v["value"]) if k == "C4" else s3(v["value"])
            verdict = ("holds" if v["holds"] else "does not hold") if o["statements_read"] else "not read (partial)"
            lo, hi = v["expected_range"]
            rng = f"{lo:.2f} to {hi:.2f}" if k == "C4" else f"{lo:+.2f} to {hi:+.2f}"
            L.append(f"- **{k}** ({v['text']}): {val} -> **{verdict}**. Expected (registration sec. 6) {rng}; "
                     f"point {'inside' if v['point_inside_expected_range'] else 'outside'}.")
        L.append("")
    pairs = PAIRS_FULL if o["second_stage"] else PAIRS_FIRST
    L += ["## Skills (mean-squared error) and other sums", "",
          "| pair | MSE skill | Huber skill (200 eV) | 1 %-trimmed skill | RMSE difference (eV) | share within 20 %, difference |",
          "|---|---|---|---|---|---|"]
    for a, b in pairs:
        L.append(f"| {LAB[a]} over {LAB[b]} | {s3(A[f'skill_mse|{a}|{b}'])} | {s3(A[f'skill_huber|{a}|{b}'])} | "
                 f"{s3(A[f'skill_trim|{a}|{b}'])} | {s3(A[f'rmse_diff_eV|{a}|{b}'])} | {s3(A[f'share20_diff|{a}|{b}'])} |")
    L += ["", "| estimator | RMSE (eV) | mean absolute error (eV) | median absolute error (eV) | share within 20 % | within 10 % |",
          "|---|---|---|---|---|---|"]
    for e in est:
        L.append(f"| {LAB[e]} | {e1(A[f'rmse|{e}'])} | {e1(A[f'mae|{e}'])} | {e1(Q[f'median_eV|{e}'])} | "
                 f"{f3(A[f'share20|{e}'])} | {f3(A[f'share10|{e}'])} |")
    L += ["", "Median absolute error ratios: " + "; ".join(f"{LAB[k.split('|')[1]]} over {LAB[k.split('|')[2]]} {f3(v)}"
                                                         for k, v in Q.items() if k.startswith("median_ratio")) + ".", "",
          "## Per seed (MSE skill, point)", "", "| pair | " + " | ".join(f"s{s}" for s in o["seeds"]) + " |",
          "|---|" + "---|" * len(o["seeds"])]
    for a, b in pairs:
        L.append(f"| {LAB[a]} over {LAB[b]} | " + " | ".join(f"{P[f'skill_mse|{a}|{b}'][str(s)]:+.3f}" for s in o["seeds"]) + " |")
    ch = o["checks"]
    L += ["", "## Checks", "",
          "- Same scored rows, values and reference errors (hold, causal GP, offline GP, PCHIP) for every seed, bit for bit: "
          + ", ".join(f"s{k} {v}" for k, v in ch["same_rows_and_references_across_seeds"].items()) + "."]
    if o["second_stage"]:
        L.append("- First stage of the second-stage file equals the stored eval_seq error (max |difference of squared "
                 "errors|): " + "; ".join(f"s{s}: network {v['c']:.3g}, twin {v['t']:.3g}"
                                          for s, v in ch["first_stage_equals_stored_max_abs_dev_se"].items()) + ".")
        L.append(f"- Test rows where the network's second-stage inputs are missing and both arms keep the first stage "
                 f"(B.64 addendum 1): network {o['fallback_rows_share']['c2']:.4f}, twin {o['fallback_rows_share']['t2']:.4f} of rows.")
        L.append("- Second-stage fit rows (out-of-fold + validation): " + "; ".join(
            f"s{s}: network {v['v2']['fit_rows_out_of_fold']:,} + {v['v2']['fit_rows_validation']:,}, twin "
            f"{v['v2h']['fit_rows_out_of_fold']:,} + {v['v2h']['fit_rows_validation']:,}" for s, v in o["fit_info"].items()) + ".")
    capped = [x["run"] for x in ch["runs"] if x["cap_bound"]]
    L.append(f"- Runs read: {len(ch['runs'])}; reached the epoch cap of {P69.B39.EPOCH_CAP}: {capped or 'none'}.")
    return "\n".join(L) + "\n"


def write(o, out_dir, stem="b69_time_confirm"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(o, indent=1, default=float), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(render(o), encoding="utf-8", newline="\n")
    return out_dir / f"{stem}.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="read what is complete (first stage if the second is missing)")
    ap.add_argument("--out-dir", default=str(PAPER))
    args = ap.parse_args()
    layout = S2.b69_layout()
    scored = [s for s in SEEDS if outer_scored(layout, s)]
    staged = [s for s in scored if S2.stage2_path(layout, s).exists()]
    complete = len(scored) == len(SEEDS) and len(staged) == len(SEEDS)
    if not complete and not args.partial:
        miss = S2.missing_runs(layout, SEEDS) + [S2.stage2_path(layout, s).name for s in SEEDS
                                                 if not S2.stage2_path(layout, s).exists()]
        raise SystemExit(f"[b69] not complete ({len(miss)} missing, first: {miss[:4]}); nothing written "
                         "(use --partial to read what is complete)")
    if not scored:
        print("[b69] --partial: no seed has both outer runs scored yet; nothing to read, nothing written", flush=True)
        return
    second = bool(scored) and staged == scored
    o = readout(layout, scored, second, partial=not complete)
    path = write(o, args.out_dir)
    msg = ", ".join(f"{k} {'holds' if v['holds'] else 'does not hold'}" for k, v in o.get("statements", {}).items())
    print(f"[b69] written {path} | partial {o['partial']} | seeds {scored} | second stage {second}"
          + (f" | {msg}" if msg and complete else ""), flush=True)


if __name__ == "__main__":
    main()
