"""B.12 batch runner: every arm x split seed x population, then the card.

Usage (from repo root, GPU):  py ces_prediction/experiments/b12/run_b12.py [--arms ...] [--cuts 3000 0]
A run whose metrics.json already exists is skipped, so an interrupted batch resumes.
Baselines must exist first (baselines_b12.py, CPU).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "data"
SEEDS = (42, 1, 7, 123)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["bilstm", "np", "tm_pre", "tm_scratch"])
    ap.add_argument("--cuts", nargs="*", type=int, default=[3000, 0])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    # Explicit on purpose: on 2026-10-09 a child process picked cpu while the parent saw CUDA.
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    log = DATA / ".b12_batch.log"
    for cut in args.cuts:
        for seed in args.seeds:
            if not (DATA / f".b12_base_cut{cut}_s{seed}" / "test_points.npz").exists():
                raise SystemExit(f"FATAL: run baselines_b12.py --seed {seed} --cut {cut} first")
            for arm in args.arms:
                out = DATA / f".b12_{arm}_cut{cut}_s{seed}"
                if (out / "metrics.json").exists():
                    print(f"[run_b12] skip {out.name} (done)")
                    continue
                t0 = time.time()
                with open(log, "a", encoding="utf-8") as fh:
                    rc = subprocess.run([sys.executable, str(HERE / "train_b12.py"), "--arm", arm,
                                         "--seed", str(seed), "--cut", str(cut), "--out", str(out),
                                         "--device", args.device],
                                        cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
                done = (out / "metrics.json").exists() and (out / "metrics.json").stat().st_mtime >= t0 - 1
                print(f"[run_b12] {out.name}: rc={rc} done={done} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                if rc != 0 and not done:  # Windows CUDA teardown can crash after the artifacts are written
                    raise SystemExit(f"FATAL: {out.name} failed -- see {log}")
    subprocess.run([sys.executable, str(HERE / "card_b12.py"), "--arms", *args.arms], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
