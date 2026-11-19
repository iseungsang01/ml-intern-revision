"""B.70 phase 2: shared redirections (PREREGISTRATION_B70.md, Addendum 2, C1-C2).

`use_d1()` points run_b39.pooled_run_dir at the D1 re-scorings (data/.b70_rescore/b39p_*, which hold copies of each
stored run's metrics.json and weights next to its D1 comparison_errors_test.npz) and sets run_b39.REPEATS to r2-r4.
Every module that calls run_b39.pooled_run_dir at run time (run_b54, run_b55, run_b57, analyze_b39, ...) then reads the
D1 rows.  `paper()` is data/.b70p2/paper/: the D1 counterpart of docs/paper/ for the readouts that read each other.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP / "b39_primary"))
import runs_b70 as RB  # noqa: E402
import run_b39 as B39  # noqa: E402

DATA = RB.DATA
P2 = DATA / ".b70p2"
PAPER_D1 = P2 / "paper"
PAPER_OLD = REPO / "docs" / "paper"
REPEATS = (2, 3, 4)
_ORIG = {"pooled_run_dir": B39.pooled_run_dir, "REPEATS": B39.REPEATS}


def d1_run_dir(model, r, k):
    if model not in ("v2", "v2h") or r not in REPEATS:
        raise SystemExit(f"FATAL: no D1 re-scoring of {model} r{r} f{k}")
    return RB.OUT_ROOT / _ORIG["pooled_run_dir"](model, r, k).name.lstrip(".")


def limit_threads(n=2):
    """Another agent shares the CPU: cap torch's threads at n whatever a runner asks for."""
    import torch
    orig = torch.set_num_threads
    if not getattr(orig, "_b70_capped", False):
        def capped(k):
            orig(min(int(k), n))
        capped._b70_capped = True
        torch.set_num_threads = capped
    torch.set_num_threads(n)


def use_d1():
    limit_threads()
    B39.pooled_run_dir = d1_run_dir
    B39.REPEATS = REPEATS
    return B39


def view():
    """data/.b70p2/view/: junctions named like the stored runs (.b39p_{v2,v2h}_cut_r{r}_f{k}) to the D1 re-scorings,
    for code that builds the stored path from DATA directly (run_b57.load_fold)."""
    import _winapi
    v = P2 / "view"
    v.mkdir(parents=True, exist_ok=True)
    for m in ("v2", "v2h"):
        for r in REPEATS:
            for k in range(5):
                link = v / _ORIG["pooled_run_dir"](m, r, k).name
                if not link.exists():
                    _winapi.CreateJunction(str(d1_run_dir(m, r, k)), str(link))
    return v


def b54_d1(B54):
    """run_b54 pointed at the D1 row tables (data/.b70p2/b54) and the D1 paper directory."""
    c = P2 / "b54"
    B54.CACHE, B54.PROP = c, c / "prop"
    B54.PAPER = paper()
    B54.OUT_JSON, B54.OUT_MD = B54.PAPER / "b54_pooled_rows.json", B54.PAPER / "b54_pooled_rows.md"
    B54.REPEATS = REPEATS
    return B54


def paper():
    PAPER_D1.mkdir(parents=True, exist_ok=True)
    return PAPER_D1


def copy_old(*names):
    """Old docs/paper files needed only by the row-independent checks (B.54 (e))."""
    paper()
    for n in names:
        if not (PAPER_D1 / n).exists():
            shutil.copy2(PAPER_OLD / n, PAPER_D1 / n)


def write_b39_d1():
    """P1/P2 of analyze_b39.arm_p on the D1 runs of r2-r4 (its own code and seed) -> data/.b70p2/paper/b39_primary.json."""
    out = paper() / "b39_primary.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    use_d1()
    import analyze_b39 as A39
    A39.B.REPEATS = REPEATS
    P = A39.arm_p()
    if not P.get("complete"):
        raise SystemExit("FATAL: arm_p on the D1 runs is not complete")
    res = {"note": "analyze_b39.arm_p on the D1 re-scorings of r2-r4 (B.70 phase 2, Addendum 2 C2); not B.39's file",
           "P": P}
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res
