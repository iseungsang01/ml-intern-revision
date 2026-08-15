"""Regenerate a frozen seq run's TEST `comparison_metrics.json` without touching its npz.

Two B.1 backbone dirs (`.b1_seqv2_s42_i42`, `.b1_seqv2_s7_i7`) had their unsuffixed
report overwritten by a later VAL re-score (eval_seq.py used one filename for both
splits until 2026-08-15). The frozen `comparison_errors_test.npz` was never touched,
so every paired verdict stands; only the descriptive report was wrong.

Procedure per dir: copy weights + metrics.json into a scratch dir, run eval_seq.py
(TEST) there, verify against the frozen npz -- population keys bit-identical,
`se_model` RMSE drift < 0.01 (PROJECT_KNOWLEDGE "Bit-Identical Re-scoring Has a
Limit") -- then copy ONLY the regenerated report back as `comparison_metrics.json`.
The frozen npz and weights are never rewritten.

Usage (repo root):
  py ces_prediction/experiments/b1_gate/regen_test_report.py --dirs .b1_seqv2_s42_i42 .b1_seqv2_s7_i7
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_b1_gate import GATE_ENV, AMBIENT_VARS, run_step  # noqa: E402

POP_KEYS = ("shot", "dt_ms", "is_peak", "se_pchip", "se_linear", "se_persistence",
            "se_gp", "se_gp_causal", "y_true")
TARGETS = ("CES_TI", "CES_VT")
TOL = 0.01


def regen(dir_name, device, merge_keys=False):
    src = DATA / dir_name
    seed = int(json.loads((src / "metrics.json").read_text(encoding="utf-8"))["seed"])
    tmp = DATA / f".regen_{dir_name.lstrip('.')}"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "weights").mkdir(parents=True)
    shutil.copy2(src / "weights" / "seq_lstm.pth", tmp / "weights" / "seq_lstm.pth")
    shutil.copy2(src / "metrics.json", tmp / "metrics.json")
    metrics = json.loads((src / "metrics.json").read_text(encoding="utf-8"))

    env = os.environ.copy()
    for var in AMBIENT_VARS + ("CES_MODEL_FILE",):
        env.pop(var, None)
    env.update(GATE_ENV)
    env.update({
        "CES_SEED": str(seed), "CES_INIT_SEED": str(metrics["init_seed"]),
        "CES_SPLIT_DIR": str(DATA / f".b1_w2cut_split_s{seed}"),
        "CES_OUTPUT_DIR": str(tmp), "CES_SPLIT_TAG": "test",
        "CES_CONTROL_METRICS": str(DATA / f".b1_w2cut_s{seed}" / "metrics.json"),
        "CES_SEQ_MODEL": metrics["seq_model"],
        "CES_PER_SHOT_NORM": "1" if metrics["per_shot_input_norm"] else "0",
        "CES_SEQ_DEVICE": device,
    })
    ok = run_step([CES_DIR / "experiments" / "seq" / "eval_seq.py"], env, tmp / "regen.log",
                  f"regen-eval(test) {dir_name}",
                  artifacts=(tmp / "comparison_errors_test.npz", tmp / "comparison_metrics.json"))
    if not ok:
        raise SystemExit(f"FATAL: regen eval failed for {dir_name}")

    with np.load(src / "comparison_errors_test.npz") as z:
        ref = {k: z[k] for k in z.files}
    with np.load(tmp / "comparison_errors_test.npz") as z:
        new = {k: z[k] for k in z.files}
    ref_files = list(ref.keys())
    for t in TARGETS:
        for k in POP_KEYS:
            key = f"{t}_{k}"
            if key not in ref_files:
                continue
            if not np.array_equal(ref[key], new[key]):
                raise SystemExit(f"FATAL: {dir_name} {key} not bit-identical -> report NOT restored")
        drift = abs(float(np.sqrt(ref[f"{t}_se_model"].mean())) - float(np.sqrt(new[f"{t}_se_model"].mean())))
        if drift > TOL:
            raise SystemExit(f"FATAL: {dir_name} {t} se_model RMSE drift {drift:.4f} > {TOL}")
        print(f"[regen] {dir_name} {t}: population bit-identical, se_model RMSE drift {drift:.4f}")
    if merge_keys:
        # sec. 5.8-C5 additive re-score: every reference key verbatim (se_model included)
        # + the keys the older harness did not write. Original kept alongside.
        added = sorted(set(new) - set(ref))
        backup = src / "comparison_errors_test__pre_merge.npz"
        if not backup.exists():
            shutil.copy2(src / "comparison_errors_test.npz", backup)
        merged = dict(ref)
        for k in added:
            merged[k] = new[k]
        np.savez(src / "comparison_errors_test.npz", **merged)
        print(f"[regen] {dir_name}: merged +{len(added)} keys {added} (reference keys verbatim; "
              f"original at {backup.name})", flush=True)
    report = json.loads((tmp / "comparison_metrics.json").read_text(encoding="utf-8"))
    report["regenerated"] = ("2026-08-15: TEST report regenerated from the frozen weights in a "
                             "scratch dir after a val re-score overwrote it; frozen npz untouched, "
                             "population verified bit-identical, se_model drift bounded")
    (src / "comparison_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.rmtree(tmp)
    print(f"[regen] {dir_name}: comparison_metrics.json restored (TEST)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--merge-keys", action="store_true",
                    help="also merge keys the older harness lacked into the frozen npz (backup kept)")
    args = ap.parse_args()
    for d in args.dirs:
        regen(d, args.device, args.merge_keys)


if __name__ == "__main__":
    main()
