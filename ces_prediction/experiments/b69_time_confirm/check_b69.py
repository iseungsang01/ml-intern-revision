"""B.69 checks before any GPU run (PREREGISTRATION_B69.md section 3, 'Checks before any run').  CPU only; no network is
trained; no GPU is touched (CUDA_VISIBLE_DEVICES is empty in this process and in every subprocess).

  (a) the dry run lists the 40 runs in the registered order (outer v2 / v2h per seed, then inner seed x fold x model).
  (b) the test block is the same for the four seeds (97 discharges) and disjoint from every training, validation and
      inner set.
  (c) the 40 envs differ from run_b39.pooled_env only in the split directory (CES_SPLIT_DIR and the split's harness
      statistics CES_CONTROL_METRICS), the seed (CES_SEED, CES_INIT_SEED) and the output directory; the outer envs equal
      B.29's temporal env (run_b29.run_env, family 'camp') at cap 80; nothing is inherited from the shell.
  (d) the causal-GP and hold references on the test block reproduce those stored for B.29's temporal runs
      (`.b29_norm0_camp_cut_s{s}`): stored references identical across the four seeds; eval_seq run on the CPU under the
      B.69 outer env (v2: B.29's s42 network; v2h: B.29 addendum 1's s42 twin as a stand-in checkpoint) reproduces them
      bit for bit; the second stage's own hold and causal GP (base rows) against them (B.64's 0.5 eV rule).
  (e) the B.64 second stage runs end to end through stage2_b69's binding on the stored B.64 artifacts of fold r2 f0 and
      reproduces data/.b64/stage2_r2_f0.npz; B.69's base rows equal B.64's.
  (f) no tracked file is changed by B.69 (git status --porcelain): nothing tracked under ces_prediction/ or in B.69's
      write set (data/, docs/paper/b69_*); other tracked changes in the working tree are listed by directory.
  (g) analyze_b69.py --partial with no scored run exits cleanly and writes nothing; without --partial it stops.
  (h) plumbing, not a result: the full B.69 path (rows, stage2_b64.fit_fold, analyze_b69 readout and statements) on
      seed 42 with the CPU-scored stand-ins of (d) as outer AND inner networks (the out-of-fold rows are in-sample).

    py ces_prediction/experiments/b69_time_confirm/check_b69.py [--fresh]
Output: docs/paper/b69_checks.md (also printed).  Exit 1 if a check fails.  Scratch: data/.b69_check/, data/.b69_plumb/
(reused when present; --fresh redoes the CPU scorings and fits).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage2_b69 as S2  # noqa: E402
import analyze_b69 as AN  # noqa: E402

S64, B54, P69 = S2.S64, S2.B54, S2.P69
B39, B29 = P69.B39, P69.B29
import run_b64 as R64  # noqa: E402  (on sys.path through stage2_b64)

DATA, REPO = P69.DATA, P69.DATA.parent
OUT_MD = REPO / "docs" / "paper" / "b69_checks.md"
CHECK = DATA / ".b69_check"
PLUMB = DATA / ".b69_plumb"
T = "CES_TI"
REF_COLS = ("shot", "row", "y_true", "se_persistence", "se_gp_causal", "se_gp", "se_pchip")
STANDIN = {"v2": lambda s: DATA / f".b29_norm0_camp_cut_s{s}", "v2h": lambda s: DATA / f".b29a1_v2h_cut_s{s}"}
B29_TEMPORAL = lambda s: DATA / f".b29_norm0_camp_cut_s{s}"          # noqa: E731
LINES, FAILS = [], []
T0 = time.time()


def put(s=""):
    LINES.append(s)
    print(s.encode("ascii", "replace").decode("ascii"), flush=True)


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def verdict(name, ok, detail):
    put(f"- **{name}**: {'PASS' if ok else 'FAIL'} -- {detail}")
    if not ok:
        FAILS.append(name)


def same(a, b):
    """Bit-for-bit equality; NaN equal to NaN for floating arrays."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        return False
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))


def run_py(args, env=None, timeout=5400):
    return subprocess.run([sys.executable, "-B", *map(str, args)], cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=timeout)


# ------------------------------------------------------------------------------ (a)
def check_a():
    put("## (a) Dry run")
    put()
    p = run_py([HERE / "run_b69.py", "--dry-run"], env={**os.environ})
    got = re.findall(r"^\s*\d+ (outer|inner) (\S+)\s+\|", p.stdout, flags=re.M)
    want = [(k, P69.run_dir(m, s, j).name) for k, m, s, j in P69.plan()]
    order_ok = got == want
    reg = ([("outer", m, s, None) for s in (42, 1, 7, 123) for m in ("v2", "v2h")]
           + [("inner", m, s, j) for s in (42, 1, 7, 123) for j in range(4) for m in ("v2", "v2h")])
    plan_ok = P69.plan() == reg
    summary = re.search(r"^(\d+) runs: outer (\d+), inner (\d+).*$", p.stdout, flags=re.M)
    tail = [ln for ln in p.stdout.splitlines() if "runs differ only in" in ln]
    put(f"`run_b69.py --dry-run`: rc {p.returncode}; {len(got)} run lines; summary line: "
        f"`{summary.group(0) if summary else 'missing'}`; env line: `{tail[0].strip() if tail else 'missing'}`.")
    put()
    put("| # | kind | run | # | kind | run |")
    put("|---|---|---|---|---|---|")
    for i in range(0, len(got), 2):
        a = got[i]
        b = got[i + 1] if i + 1 < len(got) else ("", "")
        put(f"| {i + 1} | {a[0]} | {a[1]} | {i + 2} | {b[0]} | {b[1]} |")
    put()
    verdict("(a) dry run", p.returncode == 0 and len(got) == 40 and order_ok and plan_ok and bool(tail),
            f"40 runs in the registered order: outer runs first (seed 42, 1, 7, 123; v2 then v2h), then the inner runs "
            f"(seed, inner fold 0-3, v2 then v2h): {order_ok and plan_ok}")
    put()


# ------------------------------------------------------------------------------ (b)
def shots(names):
    return sorted(int(re.search(r"(\d+)", n).group(1)) for n in names)


def check_b():
    put("## (b) The test block")
    put()
    M = {s: P69.outer_manifest(s) for s in P69.SEEDS}
    te = {s: set(M[s]["test_files"]) for s in M}
    tr = {s: set(M[s]["train_files"]) for s in M}
    va = {s: set(M[s]["val_files"]) for s in M}
    same_test = all(te[s] == te[42] for s in M)
    n_test = len(te[42])
    sets = []
    for s in M:
        sets += [(f"s{s} training", tr[s]), (f"s{s} validation", va[s])]
        for j, f in enumerate(P69.inner_folds(s)):
            man = P69.inner_manifest(s, j)
            sets += [(f"s{s} i{j} inner training", set(man["train_files"])), (f"s{s} i{j} inner held-out", set(f)),
                     (f"s{s} i{j} inner validation", set(man["val_files"]))]
    inter = {name: len(x & te[42]) for name, x in sets}
    disjoint = all(v == 0 for v in inter.values())
    same_tr = all(tr[s] == tr[42] for s in M) and all(va[s] == va[42] for s in M)
    z = np.load(B29_TEMPORAL(42) / "comparison_errors_test.npz")
    n_scored = len(np.unique(z[f"{T}_shot"]))
    put("| quantity | value |")
    put("|---|---|")
    put(f"| test files per seed | {', '.join(f's{s} {len(te[s])}' for s in M)} |")
    put(f"| test block identical for the four seeds | {same_test} |")
    put(f"| test block shot numbers | {shots(te[42])[0]} to {shots(te[42])[-1]} |")
    put(f"| training / validation shot numbers | {shots(tr[42])[0]} to {shots(tr[42])[-1]} / "
        f"{shots(va[42])[0]} to {shots(va[42])[-1]} |")
    put(f"| training / validation sizes | {len(tr[42])} / {len(va[42])} |")
    put(f"| training and validation sets identical for the four seeds | {same_tr} |")
    put(f"| sets checked against the test block / largest intersection | {len(inter)} / {max(inter.values())} |")
    put(f"| test discharges with scored T_i rows (B.29 s42 scoring) / rows | {n_scored} / {len(z[f'{T}_shot']):,} |")
    put()
    if same_tr:
        put("Note: the B.5 temporal split's training (shots up to 31991) and validation (32002 to 32310) discharges are the "
            "same for the four seeds; the seeds differ in the initialisation and, in B.69, in the inner-fold partition "
            "(rng seed (s, 0)), not in the training / validation partition.")
        put()
    verdict("(b) test block", same_test and n_test == P69.TEST_BLOCK and disjoint,
            f"{n_test} discharges, identical for the four seeds, disjoint from all {len(inter)} training / validation / "
            f"inner sets")
    put()
    if n_test != P69.TEST_BLOCK:
        finish()
        raise SystemExit(f"FATAL: the test block has {n_test} discharges, not {P69.TEST_BLOCK}; stopping")


# ------------------------------------------------------------------------------ (c)
def check_c():
    put("## (c) Environments")
    put()
    rows, bad = [], []
    for kind, m, s, j in P69.plan():
        mine = P69.ces_vars(P69.run_env(m, s, j, write=False))
        ref = P69.ces_vars(P69.ref_env(m, B39.pooled_run_dir(m, *P69.REF_FOLD)))
        keys = sorted(x[0] for x in P69.env_diff(mine, ref))
        rows.append((P69.run_dir(m, s, j).name, keys))
        if keys != sorted(P69.SWAPPED):
            bad.append(P69.run_dir(m, s, j).name)
    put(f"Reference: `run_b39.pooled_env(model, r{P69.REF_FOLD[0]}, f{P69.REF_FOLD[1]})` "
        f"(the env of `{B39.pooled_run_dir('v2', *P69.REF_FOLD).name}` / `{B39.pooled_run_dir('v2h', *P69.REF_FOLD).name}`).")
    put()
    put("| run | CES variables that differ from the reference |")
    put("|---|---|")
    for name, keys in rows:
        put(f"| {name} | {', '.join(keys)} |")
    put()
    # the outer envs against B.29's temporal env (cap 80, the same arm) and B.29's registered temporal runs (cap 30)
    eq29, d30 = [], []
    for s in P69.SEEDS:
        for m in P69.MODELS:
            out = P69.run_dir(m, s)
            mine = P69.ces_vars(P69.run_env(m, s, None, write=False))
            arm = "v2h" if m == "v2h" else "norm0"
            b29_80 = P69.ces_vars(B29.run_env(arm, "camp", "cut", s, out, False, epochs=B39.EPOCH_CAP, model=m))
            eq29.append(mine == b29_80)
            if m == "v2":
                d30.append(sorted(x[0] for x in P69.env_diff(mine, P69.ces_vars(
                    B29.run_env("norm0", "camp", "cut", s, B29_TEMPORAL(s), False)))))
    mm = []
    for s in P69.SEEDS:
        m = json.loads((B29_TEMPORAL(s) / "metrics.json").read_text(encoding="utf-8"))
        mm.append(m["seed"] == s and m["init_seed"] == s and not m["per_shot_input_norm"] and m["drop_stuck_targets"]
                  and m["seq_model"] == "v2")
    # nothing inherited from the shell
    os.environ["CES_B69_LEAK_PROBE"] = "1"
    try:
        leak = "CES_B69_LEAK_PROBE" in P69.run_env("v2", 42, None, write=False)
    finally:
        del os.environ["CES_B69_LEAK_PROBE"]
    put("| quantity | value |")
    put("|---|---|")
    put(f"| runs whose diff is exactly {', '.join(P69.SWAPPED)} | {len(rows) - len(bad)} of {len(rows)} |")
    put(f"| outer envs equal to `run_b29.run_env(arm, 'camp', 'cut', s, out, epochs='80', model)` | {sum(eq29)} of {len(eq29)} |")
    put(f"| v2 outer env vs B.29's registered temporal run of the same seed (cap 30): differing variables | "
        f"{'; '.join(', '.join(d) for d in d30)} |")
    put(f"| B.29 temporal runs' metrics.json: seed = init = s, train-file standardization, held-free, v2 | {sum(mm)} of 4 |")
    put(f"| a CES_* variable set in the shell reaches a run env | {leak} |")
    put()
    put("CES_CONTROL_METRICS is the split's harness statistics (read by eval_seq only, for the reference round trip); "
        "it is counted with the split directory, as B.29 paired it with `.b5_camp_cut_split_s{s}`.")
    put()
    verdict("(c) environments", not bad and all(eq29) and all(d == ["CES_OUTPUT_DIR", "CES_SEQ_EPOCHS"] for d in d30)
            and all(mm) and not leak,
            "every env differs from the pooled estimator's only in the split directory (and its harness statistics), "
            "the seed and the output directory")
    put()


# ------------------------------------------------------------------------------ (d)
def cpu_eval(model, s, fresh):
    """eval_seq on the CPU under the B.69 outer env of (model, s), on a copy of a stored checkpoint."""
    out = CHECK / f"eval_{model}_s{s}"
    npz = out / "comparison_errors_test.npz"
    src = STANDIN[model](s)
    if fresh and out.exists():
        shutil.rmtree(out)
    if not npz.exists():
        (out / "weights").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "metrics.json", out / "metrics.json")
        shutil.copyfile(src / "weights" / "seq_lstm.pth", out / "weights" / "seq_lstm.pth")
        env = P69.run_env(model, s, None, write=False)
        env.update({"CES_OUTPUT_DIR": str(out), "CES_SEQ_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
        log(f"CPU eval_seq: {model} s{s} on {src.name}")
        t0 = time.time()
        p = run_py([P69.SEQ_DIR / "eval_seq.py"], env=env)
        (out / "check_eval.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
        (out / "check_minutes.txt").write_text(f"{(time.time() - t0) / 60:.1f}")
        log(f"eval_seq rc {p.returncode} in {(time.time() - t0) / 60:.1f} min")
        if not npz.exists():
            return None, out
    return np.load(npz), out


def check_d(fresh):
    put("## (d) References on the test block")
    put()
    Z = {s: np.load(B29_TEMPORAL(s) / "comparison_errors_test.npz") for s in P69.SEEDS}
    across = {s: all(np.array_equal(Z[s][f"{T}_{c}"], Z[42][f"{T}_{c}"], equal_nan=True) for c in REF_COLS) for s in Z}
    put("| quantity | value |")
    put("|---|---|")
    put(f"| B.29 temporal runs `.b29_norm0_camp_cut_s{{s}}`: rows and references ({', '.join(REF_COLS)}) identical to s42, "
        f"bit for bit | {', '.join(f's{s} {v}' for s, v in across.items())} |")
    ok = all(across.values())
    for model in P69.MODELS:
        z, out = cpu_eval(model, 42, fresh)
        if z is None:
            put(f"| CPU eval_seq under the B.69 env of {model} s42 | FAILED, see {out / 'check_eval.log'} |")
            ok = False
            continue
        mins = (out / "check_minutes.txt").read_text() if (out / "check_minutes.txt").exists() else "?"
        eq = {c: bool(np.array_equal(z[f"{T}_{c}"], Z[42][f"{T}_{c}"], equal_nan=True)) for c in REF_COLS}
        line = (f"| CPU eval_seq under the B.69 outer env of `{P69.run_dir(model, 42).name}` (device cpu) on "
                f"`{STANDIN[model](42).name}`'s checkpoint ({mins} min): rows {len(z[f'{T}_shot']):,}; equal bit for bit to "
                f"the B.29 stored ones: " + ", ".join(f"{c} {v}" for c, v in eq.items()))
        if model == "v2":
            a = np.sqrt(z[f"{T}_se_model"]) - np.sqrt(Z[42][f"{T}_se_model"])
            line += (f"; the network's own error vs B.29's GPU scoring: max |difference| {np.max(np.abs(a)):.3g} eV, "
                     f"rows above 0.5 eV {int((np.abs(a) > 0.5).sum())}")
        put(line + " |")
        ok = ok and all(eq.values())
    put()
    env = P69.ces_vars(P69.run_env("v2", 42, None, write=False))
    ce = dict(env, CES_OUTPUT_DIR=str(CHECK / "eval_v2_s42"), CES_SEQ_DEVICE="cpu")
    put(f"The CPU scoring env differs from the B.69 outer env only in: "
        f"{', '.join(k for k, _, _ in P69.env_diff(ce, env))} (plus CUDA_VISIBLE_DEVICES and thread counts). "
        "The seeds 1, 7, 123 differ from 42 in CES_SEED / CES_INIT_SEED only, and B.29's references, scored under "
        "CES_SEED = s, are identical across the four seeds (first row).")
    put()
    verdict("(d) references", ok, "hold, causal GP, offline GP and PCHIP references on the test block reproduce B.29's "
                                  "stored temporal references bit for bit under the B.69 env (both models)")
    put()
    return ok


# ------------------------------------------------------------------------------ (e)
def check_e(fresh):
    put("## (e) The B.64 second stage through stage2_b69's binding")
    put()
    b69 = np.load(S2.CACHE / "base.npz")
    b64 = np.load(DATA / ".b64" / "base.npz")
    base_eq = sorted(b69.files) == sorted(b64.files) and all(same(b69[k], b64[k]) for k in b64.files)
    cache = CHECK / "b64_r2_f0"
    if fresh and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(S2.CACHE / "base.npz", cache / "base.npz")
    # B.64's own manifest function reads run_b64 through stage2_b64's global P, which the binding replaces; the layout
    # therefore names run_b64.outer_manifest, the function that stage2_b64.manifest returns.
    lay = S2.Layout(cache, outer_dir=B39.pooled_run_dir, inner_dir=R64.run_dir, manifest=R64.outer_manifest,
                    inner_folds=R64.inner_folds, rows=B54.load_rows)
    before = {n: getattr(S64, n) for n in S2.BOUND_NAMES}
    t0 = time.time()
    msg = S2.fit_one(lay, 2, 0)
    mins = (time.time() - t0) / 60
    restored = all(getattr(S64, n) is before[n] for n in S2.BOUND_NAMES)
    mine = np.load(cache / "stage2_r2_f0.npz")
    ref = np.load(DATA / ".b64" / "stage2_r2_f0.npz")
    put("| array | equal bit for bit | max abs difference |")
    put("|---|---|---|")
    ok = sorted(mine.files) == sorted(ref.files)
    for k in sorted(ref.files):
        if k == "info":
            continue
        a, b = np.asarray(mine[k]), np.asarray(ref[k])
        eq = same(a, b)
        d = float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))) if a.shape == b.shape and a.size else float("nan")
        put(f"| {k} | {eq} | {d:.3g} |")
        ok = ok and eq
    im, ir = json.loads(str(mine["info"])), json.loads(str(ref["info"]))
    drop = ("seconds_fit", "seconds_per_test_row_second_stage")
    info_eq = all({k: v for k, v in im[a].items() if k not in drop} == {k: v for k, v in ir[a].items() if k not in drop}
                  for a in ir)
    put()
    put("| quantity | value |")
    put("|---|---|")
    put(f"| B.69 base rows (`data/.b69/base.npz`) equal to B.64's (`data/.b64/base.npz`), every array | {base_eq} |")
    put(f"| `stage2_b64.fit_fold` run through `stage2_b69.bound` on r2 f0 (B.39 outer networks, B.64 inner networks, "
        f"B.54 row table) | {msg} ({mins:.1f} min) |")
    put(f"| fold information equal to the stored one (apart from run times) | {info_eq} |")
    put(f"| stage2_b64's module names restored after the call | {restored} |")
    put()
    verdict("(e) B.64 second stage imported", ok and info_eq and base_eq and restored,
            "fit_fold of stage2_b64, unchanged, reproduces the stored test-row estimates of B.64 fold r2 f0 "
            "(both arms, two-stage, fallback and missing-input variants) bit for bit")
    put()


# ------------------------------------------------------------------------------ (h)
def check_h(fresh):
    put("## (h) Plumbing of the B.69 path on stand-ins (NOT A RESULT)")
    put()
    if fresh and PLUMB.exists():
        shutil.rmtree(PLUMB)
    PLUMB.mkdir(parents=True, exist_ok=True)
    if not (PLUMB / "base.npz").exists():
        shutil.copyfile(S2.CACHE / "base.npz", PLUMB / "base.npz")
    stand = lambda arm, r, k: CHECK / f"eval_{arm}_s{r}"                                 # noqa: E731
    lay = S2.Layout(PLUMB, outer_dir=stand, inner_dir=lambda arm, r, k, j: stand(arm, r, k),
                    manifest=lambda r, k: P69.outer_manifest(r), inner_folds=lambda r, k: P69.inner_folds(r),
                    rows=lambda r, k: S2.load_rows(PLUMB, r))
    S2.stage_rows(lay, [42])
    t0 = time.time()
    msg = S2.fit_one(lay, 42, S2.TAG)
    o = AN.readout(lay, [42], second=True, partial=False, plumbing=True)
    path = AN.write(o, PLUMB, "b69_time_confirm_PLUMBING")
    info = o["fit_info"]["42"]
    st = o["statements"]
    put("Stand-ins for seed 42: the outer AND inner network = B.29's temporal network `.b29_norm0_camp_cut_s42` (cap 30); "
        "the twin = B.29 addendum 1's `.b29a1_v2h_cut_s42` (trained on a file-level split that shares discharges with the "
        "test block); both scored on the CPU in (d). The out-of-fold rows are in-sample, so nothing here is a result.")
    put()
    put("| quantity | value |")
    put("|---|---|")
    put(f"| rows / discharges with rows / test-block discharges | {o['rows']:,} / {o['discharges_with_rows']} / {o['test_block_discharges']} |")
    put(f"| stage2_b64.fit_fold on the temporal block | {msg} ({(time.time() - t0) / 60:.1f} min incl. readout) |")
    for a in ("v2", "v2h"):
        v = info[a]
        put(f"| {a}: recomputed first stage / hold / causal GP vs the eval_seq errors, rows above 0.5 eV (of "
            f"{v['test_rows']:,}); max deviation (eV) | {v['rows_dev_above_0.5_eV']['first_stage']} / "
            f"{v['rows_dev_above_0.5_eV']['hold']} / {v['rows_dev_above_0.5_eV']['gp']}; {v['max_dev_first_stage_eV']:.3g} / "
            f"{v['max_dev_hold_eV']:.3g} / {v['max_dev_gp_eV']:.3g} |")
    put(f"| test rows that keep the first stage (B.64 addendum 1): network / twin | "
        f"{o['fallback_rows_share']['c2']:.4f} / {o['fallback_rows_share']['t2']:.4f} |")
    put(f"| statements computed (values meaningless): C1 / C2 / C3 / C4 | {AN.s3(st['C1']['value'])} / {AN.s3(st['C2']['value'])} / "
        f"{AN.s3(st['C3']['value'])} / {AN.f3(st['C4']['value'])} |")
    put(f"| readout written | `{path.relative_to(REPO).as_posix()}` |")
    put()
    ok = all(k in o["statements"] for k in ("C1", "C2", "C3", "C4")) and o["rows"] > 0
    verdict("(h) plumbing", ok, "rows, the imported second stage and the readout run end to end on the temporal block")
    put()


# ------------------------------------------------------------------------------ (g)
def check_g():
    put("## (g) analyze_b69.py with no run")
    put()
    lay = S2.b69_layout()
    scored = [s for s in P69.SEEDS if AN.outer_scored(lay, s)]
    out = CHECK / "analyze_g"
    if out.exists():
        shutil.rmtree(out)
    p1 = run_py([HERE / "analyze_b69.py", "--partial", "--out-dir", out], env={**os.environ})
    w1 = sorted(x.name for x in out.glob("*")) if out.exists() else []
    p2 = run_py([HERE / "analyze_b69.py", "--out-dir", out], env={**os.environ})
    w2 = sorted(x.name for x in out.glob("*")) if out.exists() else []
    put("| call | rc | last line | files written |")
    put("|---|---|---|---|")
    last = lambda p: ((p.stdout + p.stderr).strip().splitlines() or [""])[-1]        # noqa: E731
    put(f"| `analyze_b69.py --partial` | {p1.returncode} | {last(p1)} | {w1 or 'none'} |")
    put(f"| `analyze_b69.py` | {p2.returncode} | {last(p2)} | {w2 or 'none'} |")
    put()
    put(f"Seeds with both outer runs scored when this check ran: {len(scored)}.")
    put()
    verdict("(g) analyze with no run", (not scored) and p1.returncode == 0 and not w1 and p2.returncode != 0 and not w2,
            "--partial exits cleanly and writes nothing; without --partial it stops and writes nothing")
    put()


# ------------------------------------------------------------------------------ (f)
def check_f():
    put("## (f) Tracked files")
    put()
    p = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    tracked = [ln[3:] for ln in lines if not ln.startswith("??")]
    new = [ln[3:] for ln in lines if ln.startswith("??")]
    # B.69 writes data/ (gitignored), docs/paper/b69_*.{md,json} and its own new directory, nothing else
    mine = [x for x in tracked if x.startswith("ces_prediction/") or x.startswith("docs/paper/b69_") or x.startswith("data/")]
    others = sorted({x.rsplit("/", 1)[0] for x in tracked if x not in mine})
    put(f"`git status --porcelain`: tracked files changed under `ces_prediction/` or in B.69's write set "
        f"(`docs/paper/b69_*`, `data/`): {len(mine)}{' -- ' + ', '.join(mine) if mine else ''}. New files of B.69: "
        + ", ".join(f"`{x}`" for x in new if "b69" in x) + ".")
    put()
    if tracked:
        put(f"Other tracked files changed in the working tree while this check ran: {len(tracked) - len(mine)}, in "
            f"{', '.join(f'`{d}/`' for d in others)}. They are outside every path the B.69 scripts write (data/.b69*, "
            "docs/paper/b69_*); they come from concurrent work on the paper, not from this batch.")
        put()
    verdict("(f) no tracked file changed by B.69", p.returncode == 0 and not mine,
            "nothing tracked under ces_prediction/ or in B.69's write set changed; B.69 adds only new files")
    put()


# ------------------------------------------------------------------------------ GPU time estimate (reported)
def gpu_estimate():
    put("## GPU time (reported, from earlier logs)")
    put()
    r39 = json.loads((DATA / ".b39_runs.json").read_text(encoding="utf-8"))
    r64 = json.loads((DATA / ".b64_runs.json").read_text(encoding="utf-8"))
    o = {m: statistics.median([x["minutes"] for x in r39 if x["run"].startswith(f".b39p_{m}_cut") and x["status"] == "ok"])
         for m in P69.MODELS}
    i = {m: statistics.median([x["minutes"] for x in r64 if x.get("model") == m and x["status"] == "ok"]) for m in P69.MODELS}
    tot = 4 * (o["v2"] + o["v2h"]) + 16 * (i["v2"] + i["v2h"])
    put(f"Median minutes per run: outer (train + eval, B.39 pooled cap 80) v2 {o['v2']:.1f}, v2h {o['v2h']:.1f}; inner "
        f"(train only, B.64) v2 {i['v2']:.1f}, v2h {i['v2h']:.1f}. Estimate for the 40 runs: {tot / 60:.1f} h "
        f"(8 outer {4 * (o['v2'] + o['v2h']):.0f} min, 32 inner {16 * (i['v2'] + i['v2h']):.0f} min); the second stage and "
        "the readout run on the CPU afterwards (minutes).")
    put()


def finish():
    head = [f"# B.69 checks before any GPU run", "",
            f"Registration: {P69.REG}. CPU only (CUDA_VISIBLE_DEVICES empty), no network trained. Run "
            f"{time.strftime('%Y-%m-%d %H:%M')} by check_b69.py; {(time.time() - T0) / 60:.1f} min. "
            f"Result: {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}.", ""]
    OUT_MD.write_text("\n".join(head + LINES) + "\n", encoding="utf-8", newline="\n")
    print(f"written {OUT_MD} | {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="redo the CPU scorings and the fits")
    args = ap.parse_args()
    if not (S2.CACHE / "base.npz").exists():
        log("building data/.b69/base.npz (stage2_b64.stage_base)")
        S2.stage_base(S2.b69_layout())
    CHECK.mkdir(parents=True, exist_ok=True)
    check_a()
    check_b()
    check_c()
    d_ok = check_d(args.fresh)
    check_e(args.fresh)
    if d_ok:
        check_h(args.fresh)
    else:
        verdict("(h) plumbing", False, "skipped: the CPU scorings of (d) are missing")
    check_g()
    gpu_estimate()
    check_f()
    finish()
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
