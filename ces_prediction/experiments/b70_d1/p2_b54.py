"""B.70 phase 2: B.54's row tables and readout on the D1 rows (PREREGISTRATION_B70.md, Addendum 2, C1-C4).

run_b54's stages run unchanged with its constants pointed at data/.b70p2/b54/ (cache), data/.b70p2/paper/ (outputs and
the D1 b39_primary.json its check (a) reads) and REPEATS = r2-r4; run_b39.pooled_run_dir points at the D1 runs.
Row-independent inputs are reused: files.npz (classes, blip rows, median T_i) and the B.50 propensity fits p_s*.npz.

    py ces_prediction/experiments/b70_d1/p2_b54.py [rows|prop|checks|readout|all]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b54_pooled_rows"))
import p2_common as PC  # noqa: E402
import run_b54 as B54  # noqa: E402

CACHE = PC.P2 / "b54"
OLD = PC.DATA / ".b54"


def setup():
    PC.use_d1()
    PC.copy_old("b48_reread.json", "b45_ablation_rows.json", "b50_missing_like.json", "b52_grading_readouts.json")
    PC.write_b39_d1()
    (CACHE / "prop").mkdir(parents=True, exist_ok=True)
    for n in ("files.npz", "files_checks.json"):
        if not (CACHE / n).exists():
            shutil.copy2(OLD / n, CACHE / n)
    for key in ["42"] + [f"r{r}f{k}" for r in PC.REPEATS for k in range(5)]:
        if not (CACHE / "prop" / f"p_s{key}.npz").exists():
            shutil.copy2(OLD / "prop" / f"p_s{key}.npz", CACHE / "prop" / f"p_s{key}.npz")
    if not (CACHE / "prop" / "join_42.npz").exists():
        shutil.copy2(OLD / "prop" / "join_42.npz", CACHE / "prop" / "join_42.npz")
    B54.CACHE, B54.PROP = CACHE, CACHE / "prop"
    B54.PAPER = PC.paper()
    B54.OUT_JSON, B54.OUT_MD = B54.PAPER / "b54_pooled_rows.json", B54.PAPER / "b54_pooled_rows.md"
    B54.REPEATS = PC.REPEATS


def main():
    setup()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("rows", "all"):
        B54.stage_rows()
    if stage in ("prop", "all"):
        B54.stage_prop()
    if stage in ("checks", "all"):
        try:
            B54.stage_checks()
        except SystemExit as e:
            chk = json.loads((CACHE / "checks.json").read_text(encoding="utf-8"))
            print("checks:", {k: chk[k].get("passes") for k in "abcde"}, flush=True)
            raise e
    if stage in ("readout", "all"):
        B54.stage_readout()


if __name__ == "__main__":
    main()
