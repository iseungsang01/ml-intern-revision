"""Campaign-shift generalization: train on earlier discharges, test on later ones.

Every result in this paper so far rests on a *seeded random* file-level split. That
controls leakage between adjacent rows, which is the right thing for the scientific
claim, but it is not the question a deployment faces. A deployed nowcaster is
trained on the shots that already exist and then run on shots that do not exist yet.
If the instrument, the wall conditioning, or the operating scenario drifts between
campaigns, a random split will not show it -- it hands the model future shots to
learn from.

So: order the 641 discharges by shot number (30801-32751, contiguous) and cut
strictly in time -- earliest 65% train, next 20% val, latest 15% test. No shot in
test is earlier than any shot in train. Everything else is pinned to the headline
protocol (iter009, W=4, held-free per §8c, same caps), so the only changed variable
is the split rule, and the drop against the random-split family is the cost of
extrapolating forward in time.

Four runs vary the INIT seed only; the split is a single fixed partition, so the
replication here is over initialisation, not over splits. The shot-clustered CI
within each run still reflects shot-to-shot generalisation. Both facts are stated
rather than blurred into "4 seeds".

Usage (repo root):
  py ces_prediction/experiments/campaign/run_campaign_shift.py --smoke
  py ces_prediction/experiments/campaign/run_campaign_shift.py [--resume]

Summary: data/.campaign_summary.json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
sys.path.insert(0, str(CES_DIR / "experiments" / "window_sweep"))

DATA_DIR = Path(os.getenv("CES_DATA_DIR", REPO_ROOT / "data"))
MANIFEST = REPO_ROOT / "data" / ".campaign_split_manifest.json"
INIT_SEEDS = (42, 1, 7, 123)
TRAIN_FRAC, VAL_FRAC = 0.65, 0.20   # test = the remaining latest 15%

FULL_ENV = {
    "CES_TEST_FRACTION": "0.15",
    "CES_WINDOW_SIZE": "4",
    "CES_EPOCHS": "10",
    "CES_BATCH_SIZE": "512",
    "CES_LR": "1e-3",
    "CES_VAL_FRACTION": "0.2",
    "CES_MAX_TRAIN_SAMPLES": "200000",
    "CES_MAX_VAL_SAMPLES": "40000",
    "CES_MAX_TEST_SAMPLES": "40000",
    "CES_TEMPORAL_SUBSETS": "1",
    "CES_DROP_STUCK_TARGETS": "1",   # explicit, never inherited (§8c)
}
SMOKE_ENV = {
    **FULL_ENV,
    "CES_EPOCHS": "1",
    "CES_MAX_TRAIN_SAMPLES": "4000",
    "CES_MAX_VAL_SAMPLES": "1200",
    "CES_MAX_TEST_SAMPLES": "1200",
}


def shot_number(path):
    m = re.search(r"s(\d+)", Path(path).name)
    if not m:
        raise ValueError(f"cannot parse a shot number from {path}")
    return int(m.group(1))


def build_manifest():
    """Strictly temporal partition by shot number, written in the manifest format
    `CES_FILE_SPLIT_FROM` consumes (bare file names)."""
    files = sorted(DATA_DIR.glob("s*.csv"), key=shot_number)
    if not files:
        raise SystemExit(f"FATAL: no shot CSVs under {DATA_DIR} -- real data required.")
    n = len(files)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train, val, test = files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]
    manifest = {
        "split_rule": "temporal by shot number (no test shot precedes any train shot)",
        "train_files": [p.name for p in train],
        "val_files": [p.name for p in val],
        "test_files": [p.name for p in test],
        "shot_ranges": {
            "train": [shot_number(train[0]), shot_number(train[-1])],
            "val": [shot_number(val[0]), shot_number(val[-1])],
            "test": [shot_number(test[0]), shot_number(test[-1])],
        },
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
    }
    # the whole point of the experiment: verify the cut really is a cut
    assert manifest["shot_ranges"]["train"][1] < manifest["shot_ranges"]["val"][0]
    assert manifest["shot_ranges"]["val"][1] < manifest["shot_ranges"]["test"][0]
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[campaign] manifest -> {MANIFEST}")
    print(f"[campaign] train {manifest['counts']['train']} shots "
          f"{manifest['shot_ranges']['train']} | val {manifest['counts']['val']} "
          f"{manifest['shot_ranges']['val']} | test {manifest['counts']['test']} "
          f"{manifest['shot_ranges']['test']}")
    return manifest


def run_step(cmd, env, log_path, label):
    start = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {label}: {' '.join(map(str, cmd))} =====\n")
        log.flush()
        proc = subprocess.run([sys.executable, *map(str, cmd)], cwd=REPO_ROOT,
                              env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[campaign]   {label}: rc={proc.returncode} "
          f"({(time.time() - start) / 60.0:.1f} min)")
    return proc.returncode == 0


# Arms. `base` is §8n as published. `per_shot` is §8n's own named repair: standardize the
# fast diagnostics within each discharge, so the 1.22-sigma between-campaign input drift
# cannot slide the input distribution out from under the model (§8s). Everything else --
# the temporal manifest, the caps, the treatment, the architecture -- is identical, so the
# arm is the single controlled variable.
ARMS = {
    "base":     {"suffix": "",     "env": {"CES_PER_SHOT_NORM": "0"}},
    "per_shot": {"suffix": "_ps",  "env": {"CES_PER_SHOT_NORM": "1"}},
}


def one_run(seed, smoke, resume=False, arm="base"):
    sfx = ARMS[arm]["suffix"]
    tag = f"campaign{sfx}_s{seed}" + ("_smoke" if smoke else "")
    out_dir = REPO_ROOT / "data" / (f".campaign{sfx}_smoke" if smoke
                                    else f".campaign{sfx}_s{seed}")
    split_dir = REPO_ROOT / "data" / (f".campaign{sfx}_smoke_split" if smoke
                                      else f".campaign{sfx}_split_s{seed}")
    rec = {"seed": seed, "smoke": smoke, "arm": arm, "out_dir": str(out_dir), "status": "ok"}
    if resume and not smoke and (out_dir / "bootstrap_summary.json").exists():
        print(f"[campaign] === {tag}: complete, skipping (--resume)")
        rec["resumed"] = True
        return rec
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    env = os.environ.copy()
    for var in ("CES_ABLATE", "CES_MIN_SUBSET_SIZE"):
        env.pop(var, None)
    env.update(SMOKE_ENV if smoke else FULL_ENV)
    env.update({
        "CES_SEED": str(seed),
        "CES_INIT_SEED": str(seed),
        "CES_SPLIT_DIR": str(split_dir),
        "CES_OUTPUT_DIR": str(out_dir),
        "CES_SPLIT_TAG": "test",
        "CES_FILE_SPLIT_FROM": str(MANIFEST),
    })
    env.update(ARMS[arm]["env"])   # pinned explicitly, never inherited (§8f lesson)

    print(f"[campaign] === {tag} -> {out_dir.name}")
    start = time.time()
    try:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        for step, script in (("train", "train.py"),
                             ("compare(test)", "compare_baselines.py"),
                             ("bootstrap", "bootstrap_compare.py")):
            if not run_step([CES_DIR / script], env, log_path, step):
                rec["status"] = f"{step}_failed"
                return rec
    finally:
        rec["minutes"] = round((time.time() - start) / 60.0, 1)
    return rec


def summarize(records, manifest, out_path):
    rows = []
    for rec in sorted([r for r in records if not r["smoke"]], key=lambda r: r["seed"]):
        row = {"seed": rec["seed"], "status": rec["status"], "minutes": rec.get("minutes")}
        d = Path(rec["out_dir"])
        try:
            cm = json.loads((d / "comparison_metrics.json").read_text(encoding="utf-8"))
            bs = json.loads((d / "bootstrap_summary.json").read_text(encoding="utf-8"))
            for t in ("CES_TI", "CES_VT"):
                pt, b = cm["per_target"][t], bs["splits"]["test"][t]["pchip"]
                row[t] = {
                    "n": pt["n"], "rmse_model": pt["rmse_model"],
                    "rmse_pchip": pt["baselines"]["pchip"]["rmse"],
                    "rmse_persistence": pt["baselines"]["persistence"]["rmse"],
                    "skill_vs_pchip": pt["skill_vs_pchip"],
                    "ci95": b["skill_ci95"], "pass": b["pass"], "n_shots": b["n_shots"],
                }
        except (FileNotFoundError, KeyError) as exc:
            row["missing"] = repr(exc)
        rows.append(row)

    summary = {"split": manifest["shot_ranges"], "counts": manifest["counts"],
               "protocol": FULL_ENV, "runs": rows,
               "note": ("one fixed temporal split; the 4 runs vary INIT seed only, so "
                        "replication is over initialisation, not over splits")}
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[campaign] saved {out_path}")
    print(f"{'seed':>5} {'target':>7} {'n':>7} {'rmse':>9} {'pchip':>9} "
          f"{'skill':>8} {'95% CI':>20} {'gate':>5}")
    for row in rows:
        for t in ("CES_TI", "CES_VT"):
            e = row.get(t)
            if not e:
                continue
            ci = e["ci95"]
            print(f"{row['seed']:>5} {t:>7} {e['n']:>7} {e['rmse_model']:>9.2f} "
                  f"{e['rmse_pchip']:>9.2f} {e['skill_vs_pchip']:>+8.4f} "
                  f"[{ci[0]:+.3f}, {ci[1]:+.3f}]".rjust(20)
                  + f" {'PASS' if e['pass'] else 'n.s.':>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(INIT_SEEDS))
    ap.add_argument("--arm", choices=sorted(ARMS), default="base")
    args = ap.parse_args()

    manifest = build_manifest()

    records = []
    if args.smoke:
        records.append(one_run(args.seeds[0], smoke=True, arm=args.arm))
    else:
        for seed in args.seeds:
            records.append(one_run(seed, smoke=False, resume=args.resume, arm=args.arm))

    if args.smoke:
        bad = [r for r in records if r["status"] != "ok"]
        for r in records:
            print(f"[smoke] {r['status']} ({r.get('minutes')} min)")
        sys.exit(1 if bad else 0)

    out_name = f".campaign{ARMS[args.arm]['suffix']}_summary.json"
    summarize(records, manifest, REPO_ROOT / "data" / out_name)
    sys.exit(1 if [r for r in records if r["status"] != "ok"] else 0)


if __name__ == "__main__":
    main()
