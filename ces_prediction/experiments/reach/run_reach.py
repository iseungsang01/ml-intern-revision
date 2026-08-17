"""Effective recurrent reach of the seq_v2 backbone (THESIS_RESULTS.md sec. 8ac).

Question (승상님, 2026-08-17): the window family fixes the past at W steps by
construction, while seq_v2's LSTM carries a gate-controlled state over the whole
contiguous block (median 298 rows = 3.0 s, p95 796, max 1482). How many of those steps
does the trained model actually USE? The answer prices the backbone choice: if the
curve is flat past a handful of steps, a short fixed-window or a small-receptive-field
TCN is justified; if it keeps climbing, the unbounded reach is doing real work.

**No retraining.** The four frozen B.1 backbone checkpoints (`.b1_seqv2_s{seed}_i{seed}`,
sec. 8x) are scored again on the SAME TEST population with the recurrent state reset
`ctx` steps before every scored row (`eval_seq._forward_truncated`). Only the recurrent
reach over the dense diagnostics is cut: the per-target carry-forward / staleness INPUT
channels are computed over the full block by `seq_data.build_blocks` and are left alone,
because the W = 2 window control receives that same carried history -- so `ctx = 2` is
the honest "same information as the window family" point, not a crippled model.

Protocol is B.1's verbatim (`b1_gate.GATE_ENV`): W = 2, held-free, 3 keV fit-failure
cut, per-file cap 500, per-shot standardization ON (part of seq_v2's definition).

Frozen artifacts are never touched: weights + metrics are copied into `data/.reach_s*`
and scoring writes only there. The `ctx = full` column is required to reproduce the
frozen `se_model` bit-identically, which is what proves the population is unchanged.

The four splits are scored concurrently by default (`--jobs`): they share nothing but the
read-only shot CSVs and the statistic does not depend on completion order, so this buys
wall time without touching a single number.

Usage (repo root):
  py ces_prediction/experiments/reach/run_reach.py --smoke   # seed 42, short ladder
  py ces_prediction/experiments/reach/run_reach.py           # full batch
  py ces_prediction/experiments/reach/run_reach.py --resume  # skip scored seeds
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
DATA = REPO_ROOT / "data"

sys.path.insert(0, str(CES_DIR))
sys.path.insert(1, str(CES_DIR / "experiments"))
from bootstrap_compare import BOOTSTRAP_SEED, TARGET_NAMES, _bootstrap  # noqa: E402
from b1_gate.run_b1_gate import GATE_ENV, AMBIENT_VARS  # noqa: E402

SEEDS = (42, 1, 7, 123)
# Log-spaced from "no recurrent state at all" to ~the median block length.
CONTEXTS = (1, 2, 3, 5, 10, 20, 50, 100, 300)
SMOKE_CONTEXTS = (1, 2, 10)


def backbone_dir(seed):
    """The B.1 backbone run: init seed = split seed (sec. 8x stage B diagonal)."""
    return DATA / f".b1_seqv2_s{seed}_i{seed}"


def run_step(cmd, env, log_path, label, artifacts=()):
    """Mirrors b1_gate.run_step: Windows CUDA teardown can fastfail after the
    real work is saved, so fresh artifacts override a non-zero rc."""
    start = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {label}: {' '.join(map(str, cmd))} =====\n")
        log.flush()
        proc = subprocess.run(
            [sys.executable, *map(str, cmd)],
            cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    mins = (time.time() - start) / 60.0
    ok = proc.returncode == 0
    if not ok and artifacts and all(
        Path(a).exists() and Path(a).stat().st_mtime >= start - 1 for a in artifacts
    ):
        print(f"[reach]   {label}: rc={proc.returncode} but artifacts fresh -> OK ({mins:.1f} min)", flush=True)
        ok = True
    else:
        print(f"[reach]   {label}: rc={proc.returncode} ({mins:.1f} min)", flush=True)
    return ok


def check_identity(new_npz, frozen_npz, seed):
    """ctx=full must reproduce the frozen B.1 se_model EXACTLY.

    This is the whole warrant for reusing the frozen population: if the full-block
    column is bit-identical to sec. 8x's, then every truncated column differs from it
    for exactly one reason -- the truncation.
    """
    new, old = np.load(new_npz), np.load(frozen_npz)
    for t in TARGET_NAMES:
        for key in (f"{t}_shot", f"{t}_dt_ms", f"{t}_se_pchip", f"{t}_se_model"):
            a, b = new[key], old[key]
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                print(f"[reach]   FATAL: s{seed} {key} does not reproduce the frozen artifact",
                      flush=True)
                return {"status": f"identity_failed:{key}"}
    return {"identity_vs_frozen": "bit-identical"}


def score_one(seed, contexts, smoke, resume):
    src = backbone_dir(seed)
    out_dir = DATA / (f".reach_s{seed}" + ("_smoke" if smoke else ""))
    npz = out_dir / "comparison_errors_test.npz"

    for need in (src / "weights" / "seq_lstm.pth", src / "metrics.json",
                 src / "comparison_errors_test.npz"):
        if not need.exists():
            raise SystemExit(f"FATAL: frozen B.1 backbone artifact missing: {need}")

    if resume and not smoke and npz.exists():
        # Re-verify rather than assume: a resumed seed must still carry the identity
        # evidence, otherwise the summary records `null` for it (as it first did).
        rec = {"seed": seed, "out_dir": str(out_dir), "status": "ok", "resumed": True}
        rec.update(check_identity(npz, src / "comparison_errors_test.npz", seed))
        print(f"[reach] === s{seed}: scored already, skipping (--resume); "
              f"identity {rec.get('identity_vs_frozen', rec['status'])}", flush=True)
        return rec

    # Copy ONLY what eval_seq reads; the frozen run stays untouched.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "weights").mkdir(parents=True)
    shutil.copy2(src / "weights" / "seq_lstm.pth", out_dir / "weights" / "seq_lstm.pth")
    shutil.copy2(src / "metrics.json", out_dir / "metrics.json")

    env = os.environ.copy()
    for var in AMBIENT_VARS:
        env.pop(var, None)
    env.update(GATE_ENV)
    env.update({
        "CES_SEED": str(seed),
        "CES_SPLIT_DIR": str(DATA / f".b1_w2cut_split_s{seed}"),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
        "CES_CONTROL_METRICS": str(DATA / f".b1_w2cut_s{seed}" / "metrics.json"),
        "CES_SEQ_MODEL": "v2",
        "CES_PER_SHOT_NORM": "1",
        "CES_SEQ_CONTEXTS": ",".join(str(c) for c in contexts),
    })

    print(f"[reach] === s{seed} -> {out_dir.name}  ctx={list(contexts)}", flush=True)
    start = time.time()
    ok = run_step([SEQ_DIR / "eval_seq.py"], env, out_dir / "run.log", "eval_seq(reach)",
                  artifacts=(npz, out_dir / "comparison_metrics.json"))
    record = {"seed": seed, "out_dir": str(out_dir),
              "status": "ok" if ok else "eval_failed",
              "minutes": round((time.time() - start) / 60.0, 1)}
    if not ok:
        return record

    record.update(check_identity(npz, src / "comparison_errors_test.npz", seed))
    return record


# Headline baseline (승상님, 2026-08-17): the past-only GP, not PCHIP. A reach probe
# asks how much PAST the model needs, so the reference must itself be causal -- PCHIP
# interpolates through future observations and cannot be run online at all. sec. 8x
# established gp_causal as the strongest deployable causal baseline; PCHIP is kept as a
# secondary column only so the numbers stay comparable to the sec. 8 record.
HEADLINE = "gp_causal"
# `persistence` is what the claim actually displaces: on the 10 ms grid, when CES is
# missing, holding the last value IS current practice (승상님, 2026-08-17). `pchip`
# keeps the numbers comparable to the sec. 8 record but cannot be run online.
BASELINES = ("gp_causal", "persistence", "pchip")
# "% of full" is taken against PERSISTENCE, not against the headline: the ratio needs a
# denominator that is both stable and meaningful, and the margin over gp_causal is small
# (T_i +0.08), so dividing by it amplifies noise into four-digit percentages. Persistence
# is what the model actually displaces, which makes "fraction of the recoverable margin"
# the quantity the ratio should express. The headline skill column is unaffected.
FRACTION_REF = "persistence"
# Below this |skill| even that ratio would be noise amplification rather than information.
FRACTION_FLOOR = 0.02
# Practical-equivalence floor on the paired deficit, in skill units. 0.002 is ~1/50 of
# the T_i margin over gp_causal and ~2 orders below the ctx=2 deficit, so it separates
# "indistinguishable from full inference" from any effect this study could act on.
PRACTICAL_EPS = 0.002


def analyze(seed, contexts, smoke):
    """Paired, shot-clustered: each ctx against the full-block inference path."""
    out_dir = DATA / (f".reach_s{seed}" + ("_smoke" if smoke else ""))
    z = np.load(out_dir / "comparison_errors_test.npz")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_target = {}
    for t in TARGET_NAMES:
        shot = z[f"{t}_shot"]
        se_full = z[f"{t}_se_model"]
        mse = {b: float(np.mean(z[f"{t}_se_{b}"])) for b in BASELINES}

        def skills(se):
            m = float(np.mean(se))
            return {f"skill_vs_{b}": (1.0 - m / mse[b]) if mse[b] > 0 else float("nan")
                    for b in BASELINES}

        full_sk = skills(se_full)
        skill_full = full_sk[f"skill_vs_{FRACTION_REF}"]
        usable = abs(skill_full) >= FRACTION_FLOOR
        rows = {"full": {**full_sk, "paired_vs_full": 0.0, "ci95": [0.0, 0.0],
                         "significant_deficit": False,
                         "recovered_fraction": 1.0 if usable else float("nan")}}
        for c in contexts:
            se_c = z[f"{t}_se_model_ctx{c}"]
            sk = skills(se_c)
            # skill_A_vs_B > 0 would mean the truncated arm beats full inference.
            res = _bootstrap(shot, se_c, se_full, rng)
            rows[str(c)] = {
                **sk,
                "paired_vs_full": res["skill_point"],
                "ci95": res["skill_ci95"],
                "significant_deficit": bool(res["skill_ci95"][1] < 0.0),
                "recovered_fraction": (sk[f"skill_vs_{FRACTION_REF}"] / skill_full
                                       if usable else float("nan")),
            }
        per_target[t] = {"n_rows": int(len(shot)), "n_shots": int(len(np.unique(shot))),
                         "headline_baseline": HEADLINE,
                         "full_skills": full_sk, "ladder": rows}
    return per_target


def summarize(records, contexts, smoke):
    summary = {
        "question": "how many contiguous past steps does the trained seq_v2 actually use?",
        "protocol": {"env": GATE_ENV, "seeds": [r["seed"] for r in records],
                     "contexts": list(contexts), "retrained": False,
                     "backbone_runs": {r["seed"]: str(backbone_dir(r["seed"])) for r in records},
                     "identity_check": {r["seed"]: r.get("identity_vs_frozen") for r in records}},
        "statistic": ("skill_vs_pchip per ctx; paired_vs_full = 1 - sum(SE_ctx)/sum(SE_full), "
                      "shot-clustered bootstrap B=10000 seed 12345 (bootstrap_compare._bootstrap)"),
        "per_seed": {}, "verdict": {},
    }
    for r in records:
        summary["per_seed"][r["seed"]] = analyze(r["seed"], contexts, smoke)

    for t in TARGET_NAMES:
        ladder = {}
        for c in ("full", *[str(c) for c in contexts]):
            cells = [summary["per_seed"][r["seed"]][t]["ladder"][c] for r in records]
            ladder[c] = {
                **{f"mean_skill_vs_{b}": float(np.mean([x[f"skill_vs_{b}"] for x in cells]))
                   for b in BASELINES},
                "mean_recovered_fraction": float(np.mean([x["recovered_fraction"] for x in cells])),
                "significant_deficit_seeds": sum(1 for x in cells if x["significant_deficit"]),
                "max_abs_paired_deficit": float(max(abs(x["paired_vs_full"]) for x in cells)),
            }
        # Two saturation readings, because significance alone misreads the tail here: the
        # arms are paired row-for-row, so a deficit of 1e-6 skill can still exclude zero
        # (CES_VT s7 does exactly that at ctx >= 20). The strict rule is reported for
        # completeness; the practical rule adds an effect-size floor of PRACTICAL_EPS.
        strict = next((int(c) for c in contexts
                       if ladder[str(c)]["significant_deficit_seeds"] == 0), None)
        practical = next((int(c) for c in contexts
                          if ladder[str(c)]["max_abs_paired_deficit"] < PRACTICAL_EPS), None)
        summary["verdict"][t] = {
            "ladder": ladder,
            "saturation_ctx_no_significant_deficit_any_seed": strict,
            "saturation_ctx_practical": practical,
            "practical_eps": PRACTICAL_EPS,
            "reading": (
                f"reach saturates at {practical} contiguous steps ({practical * 10} ms): "
                f"beyond that the worst paired deficit across {len(records)} splits is "
                f"< {PRACTICAL_EPS} skill"
                if practical is not None else
                "no tested ctx comes within the practical-equivalence floor of full inference"),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="seed 42, short ladder")
    ap.add_argument("--resume", action="store_true", help="skip seeds already scored")
    ap.add_argument("--jobs", type=int, default=4,
                    help="splits scored concurrently (default 4 = one per split)")
    args = ap.parse_args()

    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    contexts = SMOKE_CONTEXTS if args.smoke else CONTEXTS
    seeds = (42,) if args.smoke else SEEDS

    # Splits are scored independently and the statistic is order-free, so running them
    # concurrently changes wall time only -- never a number. (Latency is NOT measured
    # here; bench_causal_arms.py must still run alone.)
    start = time.time()
    jobs = max(1, min(args.jobs, len(seeds)))
    if jobs > 1:
        print(f"[reach] scoring {len(seeds)} splits with {jobs} concurrent workers", flush=True)
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(score_one, s, contexts, args.smoke, args.resume) for s in seeds]
            records = [f.result() for f in futures]
    else:
        records = [score_one(s, contexts, args.smoke, args.resume) for s in seeds]
    print(f"[reach] wall time: {(time.time() - start) / 60.0:.1f} min", flush=True)

    bad = [r for r in records if r["status"] != "ok"]
    for r in bad:
        print(f"[reach] FAILED: {r['out_dir']} ({r['status']})", flush=True)
    if bad:
        sys.exit(1)

    summary = summarize(records, contexts, args.smoke)
    out_path = DATA / (".reach_summary_smoke.json" if args.smoke else ".reach_summary.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for t in TARGET_NAMES:
        v = summary["verdict"][t]
        print(f"\n=== {t} ===   (headline = {HEADLINE}; ctx x 10 ms = seconds of context)")
        print(f"{'ctx':>6} {'vs gp_causal':>13} {'vs persist':>11} {'vs pchip':>9} "
              f"{'% of full*':>10} {'sig.def':>8} {'worst gap':>11}   (* margin over persistence)")
        for c in ("full", *[str(c) for c in contexts]):
            row = v["ladder"][c]
            print(f"{c:>6} {row['mean_skill_vs_gp_causal']:>+13.4f} "
                  f"{row['mean_skill_vs_persistence']:>+11.4f} "
                  f"{row['mean_skill_vs_pchip']:>+9.4f} "
                  f"{100 * row['mean_recovered_fraction']:>9.1f}% "
                  f"{row['significant_deficit_seeds']:>6}/{len(records)} "
                  f"{row['max_abs_paired_deficit']:>11.5f}")
        print(f"  -> {v['reading']}")
    print(f"\n[reach] summary saved to {out_path}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
