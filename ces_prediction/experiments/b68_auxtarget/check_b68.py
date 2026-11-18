"""B.68 checks before any GPU run (PREREGISTRATION_B68.md section 3, 'Checks before any run').  CPU only; one CPU
optimisation step on a tiny batch is the only training.

  (a) dry run: run_b68.py --dry-run lists the 15 runs and starts nothing.
  (b) the auxiliary target reproduces the offline-GP reference of B.39 on the test rows of r2 f0: the harness's file
      array of every discharge (KSTAR_CES_Dataset under the fold's env) is exactly the grid rows with at least one
      measured target; on every stored test row the cached target equals the estimator re-evaluated on the harness's
      array (bit for bit) and (target - measured value)^2 equals the stored se_gp of the B.39 network's npz.
  (c) withheld-row rule: at sampled rows with a measured T_i, replacing that row's T_i (by 1.25 x, + 1000 eV, NaN)
      leaves the target at the row unchanged (bit for bit); replacing the nearest earlier measured T_i changes it.
  (d) loss: on a real CPU batch of training discharges of r2 f0 (statistics fit as train_seq fits them), batched_aux
      yields train_seq.batched's tensors; the auxiliary arrays are the cached targets standardized with the run's
      statistics, row by row; with weight 0 the wrapper's loss equals seq_v2's (mse + 0.1 penalty, bit for bit) and
      one optimisation step through the wrapper gives the step of train_seq.masked_pass; with weight 0.5 the loss
      equals the main loss + 0.5 x the auxiliary MSE computed independently in float64 from the cache; the
      validation pass returns the main heads' masked MSE only; under the same seed the seq_v2 part of the new model
      starts from the weights of plain seq_v2; its state dict is plain seq_v2.
  (e) a CPU run of train_b68.py (fold r2 f0, 2 training files, 1 epoch = one optimisation step) writes a checkpoint that
      loads into plain seq_v2 with strict loading, and the unchanged seq/eval_seq.py (CES_SEQ_MODEL=v2) scores it on the
      whole test split of r2 f0, on the rows of the B.39 network with identical reference errors.
  (f) the 15 envs equal run_b59.arm_env("N", r, k, out) apart from CES_SEQ_MODEL and CES_OUTPUT_DIR (and equal
      run_b39.pooled_env("v2") of the fold apart from CES_OUTPUT_DIR); a B68_* variable in the shell is removed.
  (h) analyze_b68.py: without --partial it stops on zero runs and writes nothing; with --partial it exits cleanly; the
      selftest (the B.39 network in place of X) reads every statement; the fast weighted median equals the direct one.
  (g) git status: no tracked file is changed (run last).

    py ces_prediction/experiments/b68_auxtarget/check_b68.py [--fresh]
Output: docs/paper/b68_checks.md (also printed).  Exit 1 if a check fails.  The CPU run of (e) is written to
data/.b68_check_train_r2_f0/ and reused when present (--fresh redoes it).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_b68 as R  # noqa: E402
import aux_b68 as A  # noqa: E402  (seq/ and ces_prediction/ on sys.path)
import target_b68 as T  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
B39, DATA = R.B39, R.DATA
REPO = HERE.parents[2]
OUT_MD = REPO / "docs" / "paper" / "b68_checks.md"
TI = "CES_TI"
FOLD = (2, 0)
CHECK_TRAIN = DATA / ".b68_check_train_r2_f0"
CHECK_ANALYZE = DATA / ".b68_check_analyze"
CPU = torch.device("cpu")
MSE_CFG = {"kind": "mse", "sigma_phys": None, "delta_norm": None, "nu": None}
T0 = time.time()
LINES, FAILS = [], []


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def put(m=""):
    LINES.append(m)
    print(m, flush=True)


def verdict(name, ok, detail):
    put(f"- **{name}**: {'PASS' if ok else 'FAIL'} -- {detail}")
    if not ok:
        FAILS.append(name)


def run_py(args, env=None, timeout=3600):
    return subprocess.run([sys.executable, "-B", *map(str, args)], cwd=REPO, env=env, capture_output=True,
                          text=True, timeout=timeout)


def cpu_env(out, **extra):
    env = R.arm_env(*FOLD, out)
    env.update({"CES_SEQ_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4", **extra})
    return env


def pin_env(env):
    for key in [x for x in os.environ if x.startswith("CES_")]:
        del os.environ[key]
    os.environ.update({x: v for x, v in env.items() if x.startswith("CES_")})


def maxdiff(a, b):
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)


# ------------------------------------------------------------------------------ (a)
def check_dry():
    put("## (a) Dry run")
    put()
    p = run_py([HERE / "run_b68.py", "--dry-run"])
    listed = [ln for ln in p.stdout.splitlines() if ln[:2].strip().isdigit() and " arm X " in ln]
    last = [ln for ln in p.stdout.splitlines() if " runs: " in ln]
    dirs = [ln.split("|")[1].strip() for ln in listed]
    want = [R.run_dir(r, k).name for r in R.REPEATS for k in range(5)]
    ok = p.returncode == 0 and len(listed) == 15 and dirs == want
    put(f"`run_b68.py --dry-run`: return code {p.returncode}; {len(listed)} runs listed ({last[0].strip() if last else '-'}); "
        f"directories {dirs[0] if dirs else '-'} .. {dirs[-1] if dirs else '-'} in the registered order: {dirs == want}.")
    put()
    verdict("(a) dry run", ok, "15 runs (r2 f0 .. r4 f4, one initialisation per fold), nothing started")
    put()


# ------------------------------------------------------------------------------ (b)
def check_target(grid, cache):
    put("## (b) The auxiliary target reproduces the offline-GP reference of B.39 (test rows of r2 f0)")
    put()
    import baselines_interpolation as B
    from evaluate import _load_stats, build_clean_val_subset
    est = R.estimator_dir(*FOLD)
    env = B39.pooled_env("v2", *FOLD, est)
    pin_env(env)
    man = json.loads((Path(env["CES_SPLIT_DIR"]) / "split_manifest.json").read_text(encoding="utf-8"))
    stats = _load_stats(json.loads(Path(env["CES_CONTROL_METRICS"]).read_text(encoding="utf-8")))
    log("harness dataset of r2 f0")
    dataset, _, _ = build_clean_val_subset(Path(env["CES_DATA_DIR"]), int(env["CES_WINDOW_SIZE"]), stats,
                                           set(man["test_files"]), int(env["CES_MAX_VAL_SAMPLES"]), int(env["CES_SEED"]))
    tc = int(dataset._column_slices["time"])
    ti_c, vt_c = (int(c) for c in dataset._column_slices["target"])
    same_files, n_files = 0, 0
    for p, fa in zip(dataset.valid_files, dataset.file_arrays):
        g = grid[Path(p).name]
        n_files += 1
        same_files += bool(np.array_equal(fa[:, [tc, ti_c, vt_c]], g[T.harness_rows(g)][:, :3], equal_nan=True))
    z = np.load(est / "comparison_errors_test.npz")
    shot, row, y, se_gp = z[f"{TI}_shot"], z[f"{TI}_row"], z[f"{TI}_y_true"], z[f"{TI}_se_gp"]
    n = len(shot)
    aux, harness, kind, ti_same = np.empty(n), np.empty(n), np.empty(n, np.int8), 0
    log(f"re-evaluating the estimator on {n:,} stored test rows")
    for j in range(n):
        s, ri = int(shot[j]), int(row[j])
        fa = dataset.file_arrays[s]
        e = cache[Path(dataset.valid_files[s]).name]
        g = int(np.searchsorted(e["time"], fa[ri, tc]))
        if e["time"][g] != fa[ri, tc]:
            raise SystemExit("FATAL: a stored test row is not a grid row")
        aux[j], kind[j] = e["aux"][g], e["kind"][g]
        ti_same += bool(e["ti"][g] == fa[ri, ti_c])
        harness[j] = B.predict("gp", fa, tc, ti_c, ri)
    se_mine = (aux - y) ** 2
    pred_same = int(np.sum(aux == harness))
    se_same = int(np.sum(se_mine == se_gp))
    d = np.abs(np.sqrt(se_mine) - np.sqrt(se_gp))
    put("| quantity | value |")
    put("|---|---|")
    put(f"| discharges whose harness file array (time, T_i, V_rot) equals the grid rows with >= 1 measured target | "
        f"{same_files} of {n_files} |")
    put(f"| stored test rows (B.39 network r2 f0) / with a measured T_i in the cache equal to the harness's value | {n:,} / {ti_same:,} |")
    put(f"| rows where the cached target equals the estimator re-evaluated on the harness array (bit for bit) | {pred_same:,} |")
    put(f"| rows where (cached target - y_true)^2 equals the stored se_gp (bit for bit); max abs difference of the absolute "
        f"errors (eV) | {se_same:,}; {float(d.max()):.3g} |")
    put(f"| skill of the cached target over the stored offline GP on these rows (0 = the same errors) | "
        f"{1 - se_mine.sum() / se_gp.sum():+.2e} |")
    put(f"| row kinds of the stored rows | {dict(zip(*[x.tolist() for x in np.unique(kind, return_counts=True)]))} "
        f"(0 = T_i measured, own value withheld) |")
    put()
    ok = (same_files == n_files and ti_same == n and pred_same == n and se_same == n and bool(np.all(kind == T.KIND_TI)))
    verdict("(b) target = stored offline-GP reference", ok,
            f"bit for bit on all {n:,} test rows of r2 f0: the cache was evaluated on exactly the harness's file arrays "
            "(all discharges), and its value and squared error are those of the stored offline-GP reference")
    put()
    return dataset


# ------------------------------------------------------------------------------ (c)
def check_withheld(grid, cache, n_rows=300):
    put("## (c) Withheld-row rule")
    put()
    rng = np.random.default_rng(68)
    cand = [(nm, i) for nm in sorted(cache) for i in np.flatnonzero((cache[nm]["kind"] == T.KIND_TI)
                                                                     & cache[nm]["past"] & cache[nm]["future"])]
    pick = [cand[j] for j in rng.choice(len(cand), size=n_rows, replace=False)]
    repro = unchanged = changed = 0
    per = {"x1.25": 0, "+1000 eV": 0, "NaN": 0}
    for nm, i in pick:
        arr = grid[nm]
        ref = cache[nm]["aux"][i]
        repro += bool(T.aux_at(arr, i) == ref)
        ok_all = True
        for lab, f in (("x1.25", lambda v: v * 1.25), ("+1000 eV", lambda v: v + 1000.0), ("NaN", lambda v: np.nan)):
            a2 = arr.copy()
            a2[i, T.TI_COL] = np.float32(f(float(arr[i, T.TI_COL])))
            same = T.aux_at(a2, i) == ref
            per[lab] += bool(same)
            ok_all &= bool(same)
        unchanged += ok_all
        lab_rows = T.harness_rows(arr)
        prev = [r for r in lab_rows[lab_rows < i] if np.isfinite(arr[r, T.TI_COL])][-1]
        a3 = arr.copy()
        a3[prev, T.TI_COL] = np.float32(float(arr[prev, T.TI_COL]) * 1.25)
        changed += bool(T.aux_at(a3, i) != ref)
    # rows without any measured target (inserted into the harness array) reproduce too
    ins = [(nm, i) for nm in sorted(cache) for i in np.flatnonzero(cache[nm]["kind"] == T.KIND_INSERTED)]
    ins = [ins[j] for j in rng.choice(len(ins), size=min(100, len(ins)), replace=False)]
    ins_ok = sum(bool(np.array_equal([T.aux_at(grid[nm], i)], [cache[nm]["aux"][i]], equal_nan=True)) for nm, i in ins)
    put("| quantity | value |")
    put("|---|---|")
    put(f"| sampled rows with a measured T_i and measured values on both sides (seed 68), from {len(cand):,} | {len(pick)} |")
    put(f"| target re-evaluated on the unchanged discharge equals the cache | {repro} of {len(pick)} |")
    put(f"| target unchanged when the row's own T_i is replaced: x1.25 / +1000 eV / NaN | {per['x1.25']} / {per['+1000 eV']} / "
        f"{per['NaN']} of {len(pick)} |")
    put(f"| control: target changed when the nearest earlier measured T_i is multiplied by 1.25 | {changed} of {len(pick)} |")
    put(f"| rows without a measured target, target re-evaluated equals the cache (NaN = NaN) | {ins_ok} of {len(ins)} |")
    put()
    ok = repro == unchanged == len(pick) and changed == len(pick) and ins_ok == len(ins)
    verdict("(c) withheld-row rule", ok, "the target at a measured row does not depend on that row's value (bit for bit), "
            "and it does depend on the neighbouring measured values")
    put()


# ------------------------------------------------------------------------------ (d)
def check_loss(grid, dims, cache):
    put("## (d) The loss on a real CPU batch")
    put()
    import train_seq
    from seq_data import fit_stats, build_blocks
    from model_seq_v2 import SeqCESLSTMv2
    env = R.arm_env(*FOLD, R.run_dir(*FOLD))
    man = json.loads((Path(env["CES_SPLIT_DIR"]) / "split_manifest.json").read_text(encoding="utf-8"))
    train_names = [nm for nm in man["train_files"] if nm in grid]
    stats = fit_stats(grid, dims, train_names)
    zero_ti = float((0.0 - stats["target"]["mean"][0]) / stats["target"]["std"][0])
    init = int(env["CES_INIT_SEED"])
    names = train_names[:4]
    owned = []
    for nm in names:
        for b in A.attach_aux(build_blocks(grid[nm], dims, stats, per_shot_norm=False), grid[nm], nm, stats, cache[nm]):
            owned.append((nm, b))
    blocks = [b for _, b in owned]
    # tensors of the two batchers
    same_batches = True
    for shuffle in (False, True):
        b1 = list(train_seq.batched(blocks, 16, CPU, shuffle, random.Random(5)))
        b2 = list(A.batched_aux(blocks, 16, CPU, shuffle, random.Random(5)))
        same_batches &= len(b1) == len(b2) and all(all(torch.equal(p, q) for p, q in zip(u, v[:4])) for u, v in zip(b1, b2))
    x, y, m, lengths, a, am = next(A.batched_aux(blocks, 16, CPU, False, None))
    # auxiliary arrays against the cache, row by row (by time), standardized independently in float64
    mean, std = float(stats["target"]["mean"][0]), float(stats["target"]["std"][0])
    zref, mref = np.zeros(a.shape), np.zeros(a.shape)
    for j, (nm, b) in enumerate(owned):
        e = cache[nm]
        for i, t in enumerate(b["time"]):
            g = int(np.searchsorted(e["time"], t))
            v = (float(e["aux"][g]) - mean) / std
            if np.isfinite(v):
                zref[j, i], mref[j, i] = v, 1.0
    align = (np.array_equal(am.numpy(), mref) and float(np.abs(a.numpy() - zref).max()) < 1e-5)
    # models: the seq_v2 part of the new model starts from plain seq_v2's weights
    torch.manual_seed(init)
    plain = SeqCESLSTMv2()
    torch.manual_seed(init)
    auxm = A.SeqCESLSTMv2Aux()
    sd_p, sd_a = plain.state_dict(), auxm.state_dict()
    init_same = list(sd_p) == list(sd_a) and all(torch.equal(sd_p[k], sd_a[k]) for k in sd_p)
    full_keys = sorted(set(auxm.full_state_dict()) - set(sd_a))
    plain2 = SeqCESLSTMv2()
    plain2.load_state_dict(sd_a, strict=True)
    refused = False
    try:
        auxm.load_state_dict(auxm.full_state_dict())
    except RuntimeError:
        refused = True
    # forward: main heads equal plain seq_v2 (eval, and train mode under the same dropout seed)
    fwd = {}
    for mode in ("eval", "train"):
        plain.train(mode == "train")
        auxm.train(mode == "train")
        with torch.no_grad():
            torch.manual_seed(11)
            o_p = plain(x, lengths)
            torch.manual_seed(11)
            o_a, _ = auxm.forward_with_aux(x, lengths)
            torch.manual_seed(11)
            o_f = auxm(x, lengths)
        fwd[mode] = bool(torch.equal(o_p, o_a) and torch.equal(o_p, o_f))
    plain.eval()
    auxm.eval()
    with torch.no_grad():
        o_p = plain(x, lengths)
        se = ((o_p - y) ** 2) * m
        ref0 = se.sum() / m.sum().clamp(min=1.0) + 0.1 * (torch.relu(zero_ti - o_p[..., 0]) * m[..., 0]).sum() \
            / m[..., 0].sum().clamp(min=1.0)
        t0 = A.batch_terms(auxm, x, y, m, lengths, a, am, zero_ti, 0.0)
        t5 = A.batch_terms(auxm, x, y, m, lengths, a, am, zero_ti, A.AUX_WEIGHT)
    ax = t5["aux_out"].double().numpy()
    aux_np = float(np.sum(((ax - zref) ** 2) * mref) / mref.sum())
    ref5 = float(ref0) + A.AUX_WEIGHT * aux_np
    loss0_same = bool(torch.equal(t0["loss"], ref0))
    rel5 = abs(float(t5["loss"]) - ref5) / abs(ref5)
    relaux = abs(float(t5["aux_mse"]) - aux_np) / aux_np
    # one optimisation step: train_seq.masked_pass on plain seq_v2 vs the wrapper with weight 0 / 0.5
    steps = {}
    for w in ("orig", 0.0, A.AUX_WEIGHT):
        torch.manual_seed(init)
        mdl = SeqCESLSTMv2() if w == "orig" else A.SeqCESLSTMv2Aux()
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
        torch.manual_seed(13)
        if w == "orig":
            ret = train_seq.masked_pass(mdl, blocks, 16, CPU, zero_ti, optimizer=opt, rng=random.Random(42), loss_cfg=MSE_CFG)
            lg = None
        else:
            lg = []
            ret = A.make_masked_pass(w, lg)(mdl, blocks, 16, CPU, zero_ti, optimizer=opt, rng=random.Random(42),
                                            loss_cfg=MSE_CFG)
        steps[w] = (ret, mdl.state_dict(), lg)
    d0 = maxdiff(steps["orig"][1], steps[0.0][1])
    ti_keys = [k for k in steps["orig"][1] if "_ti" in k]
    vt_keys = [k for k in steps["orig"][1] if "_vt" in k]
    d5_ti = max(float((steps["orig"][1][k] - steps[A.AUX_WEIGHT][1][k]).abs().max()) for k in ti_keys)
    d5_vt = max(float((steps["orig"][1][k] - steps[A.AUX_WEIGHT][1][k]).abs().max()) for k in vt_keys)
    ret_same = steps["orig"][0] == steps[0.0][0]
    nb = steps[0.0][2][0]["batches"]
    # validation pass (no optimizer): the wrapper with weight 0.5 returns the main heads' masked MSE only
    torch.manual_seed(init)
    vp = SeqCESLSTMv2()
    torch.manual_seed(init)
    va = A.SeqCESLSTMv2Aux()
    v_orig = train_seq.masked_pass(vp, blocks, 16, CPU, zero_ti, loss_cfg=MSE_CFG)
    v_mine = A.make_masked_pass(A.AUX_WEIGHT, [])(va, blocks, 16, CPU, zero_ti, loss_cfg=MSE_CFG)
    put(f"Batch: the first {len(names)} training discharges of r2 f0 ({len(blocks)} blocks, {int(lengths.sum()):,} rows, "
        f"{int(m[..., 0].sum()):,} measured T_i, {int(am.sum()):,} rows with an auxiliary target); statistics fit on the "
        f"{len(train_names)} training discharges as train_seq does; initialisation seed {init}.")
    put()
    put("| quantity | value |")
    put("|---|---|")
    put(f"| batched_aux gives train_seq.batched's x, y, mask, lengths (in order and shuffled) | {same_batches} |")
    put(f"| aux_mask = rows with a cached target; aux = cached target standardized with the run's T_i statistics (max abs "
        f"diff to the float64 reference) | {align} ({float(np.abs(a.numpy() - zref).max()):.1e}) |")
    put(f"| same seed: the seq_v2 part of SeqCESLSTMv2Aux = plain seq_v2 (keys and values); state_dict keys = seq_v2's; "
        f"auxiliary keys left out | {init_same}; {list(sd_p) == list(sd_a)}; {full_keys} |")
    put(f"| its state dict loads into plain seq_v2 with strict=True; a dict carrying the auxiliary head is refused | True; {refused} |")
    put(f"| main heads = plain seq_v2 output, eval mode / train mode (same dropout seed) | {fwd['eval']} / {fwd['train']} |")
    put(f"| weight 0: loss = seq_v2's masked MSE + 0.1 penalty (bit for bit) | {loss0_same} ({float(t0['loss']):.6f}) |")
    put(f"| weight 0.5: loss vs main + 0.5 x auxiliary MSE computed in float64 from the cache, relative difference; auxiliary "
        f"MSE relative difference | {rel5:.1e} ({float(t5['loss']):.6f} vs {ref5:.6f}); {relaux:.1e} |")
    put(f"| one optimisation step ({nb} batch): train_seq.masked_pass on seq_v2 vs the wrapper with weight 0: returned values "
        f"equal; max abs difference of the seq_v2 weights after the step | {ret_same}; {d0:.1e} |")
    put(f"| same step with weight 0.5: max abs change against weight 0 in the T_i branch / in the V_rot branch | "
        f"{d5_ti:.1e} / {d5_vt:.1e} |")
    put(f"| validation pass: train_seq.masked_pass (seq_v2) vs the wrapper with weight 0.5 (same seq_v2 weights): returned "
        f"(main MSE, penalty, main MSE) equal | {v_orig == v_mine} ({v_mine[0]:.6f}) |")
    put()
    ok = (same_batches and align and init_same and refused and len(full_keys) == 4 and all(k.startswith(A.AUX_PREFIX) for k in full_keys)
          and fwd["eval"] and fwd["train"]
          and loss0_same and rel5 < 1e-6 and relaux < 1e-5 and ret_same and d0 < 1e-6 and d5_ti > 0 and nb == 1
          and v_orig == v_mine)
    verdict("(d) loss", ok, "weight 0 reproduces seq_v2's loss and optimisation step; weight 0.5 adds exactly 0.5 x the "
            "masked MSE of the auxiliary head on the cached target; early stopping sees the main heads only")
    put()


# ------------------------------------------------------------------------------ (e)
def check_checkpoint(fresh):
    put("## (e) A CPU run of train_b68.py: the checkpoint is plain seq_v2 and eval_seq scores it")
    put()
    out = CHECK_TRAIN
    npz = out / "comparison_errors_test.npz"
    if fresh or not (npz.exists() and (out / R.SIDECAR).exists()):
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        log("train_b68.py on the CPU (2 training files, 1 epoch)")
        t0 = time.time()
        p = run_py([R.TRAIN], env=cpu_env(out, CES_SEQ_EPOCHS="1", CES_SEQ_MAX_FILES="2"), timeout=3600)
        (out / "check_train.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
        log(f"train rc {p.returncode} in {(time.time() - t0) / 60:.1f} min; eval_seq.py on the CPU (whole test split)")
        t1 = time.time()
        q = run_py([R.EVAL], env=cpu_env(out), timeout=7200)
        (out / "check_eval.log").write_text(q.stdout + "\n" + q.stderr, encoding="utf-8")
        (out / "check_minutes.txt").write_text(f"{(t1 - t0) / 60:.1f} {(time.time() - t1) / 60:.1f}")
        log(f"eval rc {q.returncode} in {(time.time() - t1) / 60:.1f} min")
    else:
        log(f"reusing {out.name}")
    have = all(pth.exists() for pth in (out / "weights" / "seq_lstm.pth", out / "metrics.json", out / R.SIDECAR, npz))
    if not have:
        put(f"FAILED: outputs missing, see {out / 'check_train.log'} and {out / 'check_eval.log'}")
        verdict("(e) checkpoint", False, "the CPU run did not write its outputs")
        return
    import seq_models
    sd = torch.load(out / "weights" / "seq_lstm.pth", map_location="cpu")
    plain = seq_models.SEQ_MODELS["v2"]()
    res = plain.load_state_dict(sd, strict=True)
    strict_ok = not res.missing_keys and not res.unexpected_keys and not any(k.startswith(A.AUX_PREFIX) for k in sd)
    met = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    side = json.loads((out / R.SIDECAR).read_text(encoding="utf-8"))
    tr = [p for p in side["passes"] if p["phase"] == "train"]
    e = np.load(npz)
    z = np.load(R.estimator_dir(*FOLD) / "comparison_errors_test.npz")
    same_rows = np.array_equal(e[f"{TI}_shot"], z[f"{TI}_shot"]) and np.array_equal(e[f"{TI}_row"], z[f"{TI}_row"])
    refs = same_rows and all(np.array_equal(e[f"{TI}_se_{q}"], z[f"{TI}_se_{q}"]) for q in ("persistence", "pchip", "gp", "gp_causal")) \
        and np.array_equal(e[f"{TI}_y_true"], z[f"{TI}_y_true"])
    cm = json.loads((out / "comparison_metrics.json").read_text(encoding="utf-8"))
    rmse = cm["per_target"][TI]["rmse_model"]
    ev_log = (out / "check_eval.log").read_text(encoding="utf-8") if (out / "check_eval.log").exists() else ""
    tr_log = (out / "check_train.log").read_text(encoding="utf-8") if (out / "check_train.log").exists() else ""
    mins = (out / "check_minutes.txt").read_text().split() if (out / "check_minutes.txt").exists() else ["?", "?"]
    put("| quantity | value |")
    put("|---|---|")
    put(f"| train_b68.py (fold r2 f0 env, CES_SEQ_DEVICE=cpu, CES_SEQ_MAX_FILES=2, CES_SEQ_EPOCHS=1; {mins[0]} min): "
        f"seq_model / epochs / optimisation steps (training batches) | {met['seq_model']} / {met['epochs_run']} / "
        f"{sum(p['batches'] for p in tr)} |")
    put(f"| hooks active in the training process: load_grid_files calls / build_blocks calls / auxiliary weight / training "
        f"rows with a target | {side['grid_calls']} / {side['block_calls']} / {side['aux_weight']} / "
        f"{side['train']['rows_with_aux_target']:,} of {side['train']['rows']:,} |")
    put(f"| the saved checkpoint: keys {len(sd)}, none of the auxiliary head; loads into SEQ_MODELS['v2'] with strict=True "
        f"(here and inside train_b68.py) | {strict_ok}; {'[b68] checkpoint loads as plain seq_v2' in tr_log} |")
    put(f"| seq/eval_seq.py unchanged, CES_SEQ_MODEL=v2, whole test split of r2 f0 ({mins[1]} min): rows / same rows as the "
        f"B.39 network / reference errors and measured values identical / RMSE of the one-step network (eV) | "
        f"{len(e[f'{TI}_shot']):,} / {same_rows} / {refs} / {rmse:.1f} |")
    put()
    ok = (strict_ok and met["seq_model"] == "v2" and met["epochs_run"] == 1 and sum(p["batches"] for p in tr) == 1
          and side["grid_calls"] == 1 and side["aux_weight"] == A.AUX_WEIGHT and side["block_calls"] >= 4
          and same_rows and refs and np.isfinite(rmse) and "saved" in ev_log)
    verdict("(e) checkpoint", ok, "one CPU optimisation step through train_b68.py; the checkpoint is plain seq_v2 (strict "
            "loading), and the unchanged eval_seq.py scores it as v2 on the B.39 network's rows of r2 f0")
    put()


# ------------------------------------------------------------------------------ (f)
def check_envs():
    put("## (f) The 15 environments")
    put()
    sys.path.insert(0, str(HERE.parent / "b59_inputs"))
    import run_b59 as R59
    os.environ["B68_LEFTOVER"] = "1"
    try:
        keys, est_keys, n, b68_keys = set(), set(), 0, set()
        ok = True
        for r in R.REPEATS:
            for k in range(5):
                mine = R.arm_env(r, k, R.run_dir(r, k))
                ref = R59.arm_env("N", r, k, R59.run_dir("N", r, k))
                b68_keys |= {x for x in mine if x.startswith("B68_")}
                d = {x for x in set(mine) | set(ref) if mine.get(x) != ref.get(x) and not x.startswith("B68_")}
                keys |= d
                ok &= d == {"CES_SEQ_MODEL", "CES_OUTPUT_DIR"} and mine["CES_SEQ_MODEL"] == "v2" and ref["CES_SEQ_MODEL"] == "v2n"
                est = B39.pooled_env("v2", r, k, B39.pooled_run_dir("v2", r, k))
                est_keys |= {x for x in set(mine) | set(est) if mine.get(x) != est.get(x) and not x.startswith("B68_")}
                ok &= mine["CES_OUTPUT_DIR"] == str(R.run_dir(r, k))
                A.check_env(mine)
                n += 1
    finally:
        del os.environ["B68_LEFTOVER"]
    ok &= est_keys == {"CES_OUTPUT_DIR"} and not b68_keys
    put(f"For {n} folds (whole env, CES_* and the rest; B68_* keys aside, since B.59 does not strip them): variables that "
        f"differ from B.59 arm N of the fold: "
        f"{', '.join(sorted(keys))} (CES_SEQ_MODEL v2 for v2n); from run_b39.pooled_env('v2') of the fold: "
        f"{', '.join(sorted(est_keys))}; B68_* variables in the run env with B68_LEFTOVER=1 set in the shell: "
        f"{sorted(b68_keys) or 'none'}; aux_b68.check_env accepts every env.")
    put()
    verdict("(f) environments", ok, "each env is the B.39 network's env of its fold with only the output directory changed, "
            "i.e. B.59 arm N's apart from CES_SEQ_MODEL and CES_OUTPUT_DIR")
    put()


# ------------------------------------------------------------------------------ (h)
def check_analyze():
    put("## (h) analyze_b68.py")
    put()
    if CHECK_ANALYZE.exists():
        shutil.rmtree(CHECK_ANALYZE)
    res = {}
    for name, args in (("strict", []), ("partial", ["--partial"]), ("selftest", ["--selftest"])):
        d = CHECK_ANALYZE / name
        p = run_py([HERE / "analyze_b68.py", *args, "--out-dir", d], timeout=1800)
        j = d / "b68_auxtarget.json"
        res[name] = {"rc": p.returncode, "written": j.exists(), "json": json.loads(j.read_text(encoding="utf-8")) if j.exists() else None,
                     "tail": [ln.replace(str(REPO) + os.sep, "") for ln in (p.stdout + p.stderr).strip().splitlines()[-1:]]}
    put("| run | return code | output written | statements read | last line |")
    put("|---|---|---|---|---|")
    for name, v in res.items():
        stt = {k: x["read"] for k, x in v["json"]["statements"].items()} if v["json"] else "-"
        put(f"| {name} | {v['rc']} | {v['written']} | {stt} | {v['tail'][0][:160] if v['tail'] else ''} |")
    put()
    pj, sj = res["partial"]["json"], res["selftest"]["json"]
    ok = (res["strict"]["rc"] != 0 and not res["strict"]["written"]
          and res["partial"]["rc"] == 0 and pj is not None and not any(x["read"] for x in pj["statements"].values())
          and pj["reported"]["pairs"]["est_over_twin"] is not None
          and res["selftest"]["rc"] == 0 and sj is not None and all(x["read"] for x in sj["statements"].values()) and sj["selftest"])
    if sj:
        x1, x2, x3 = sj["statements"]["X1"]["skill"], sj["statements"]["X2"]["skill"], sj["statements"]["X3"]["ratio"]
        m = sj["reported"]["median_ratio"]
        put(f"Selftest (the B.39 network read as X): X1 {x1[0]:+.3f} [{x1[1]:+.3f}, {x1[2]:+.3f}] (the network over itself); X2 "
            f"{x2[0]:+.3f} [{x2[1]:+.3f}, {x2[2]:+.3f}] (= the B.39 network over the twin; registration: +0.053 [+0.027, +0.076]); "
            f"X3 {x3[0]:.3f} [{x3[1]:.3f}, {x3[2]:.3f}] (= the network / causal GP {m['est_over_gp_causal']['ratio'][0]:.3f}).")
        put()
        ok &= x1 == [0.0, 0.0, 0.0] and x3 == m["est_over_gp_causal"]["ratio"]
    # the fast weighted median against the direct cumulative sums
    import analyze_b68 as AN
    z = [np.load(R.estimator_dir(2, k) / "comparison_errors_test.npz") for k in range(5)]
    shot = np.concatenate([q[f"{TI}_shot"] for q in z])
    v = np.sqrt(np.concatenate([q[f"{TI}_se_gp_causal"] for q in z]))
    _, f = np.unique(shot, return_inverse=True)
    nd = int(f.max()) + 1
    W = np.random.default_rng(1).multinomial(nd, np.full(nd, 1.0 / nd), size=300).astype(np.int32)
    W[0] = 1
    fast, direct = AN.wmedian(v, f, W), AN.wmedian_direct(v, f, W)
    wm_ok = bool(np.array_equal(fast, direct)) and float(np.sort(v)[(len(v) - 1) // 2]) == float(fast[0])
    put(f"Weighted median (analyze_b68.wmedian, bucketed) against the direct cumulative sums on the causal-GP errors of r2 "
        f"({len(v):,} rows, 300 resampling draws, the first with unit weights): equal {bool(np.array_equal(fast, direct))}; "
        f"unit weights give the lower plain median: {wm_ok}.")
    put()
    ok &= wm_ok
    verdict("(h) analyze", ok, "without --partial it stops and writes nothing; with --partial on zero B.68 runs it exits 0 "
            "with every statement 'not read'; the selftest reads X1-X3 and the row and median readouts")
    put()


# ------------------------------------------------------------------------------ (g)
def check_git():
    put("## (g) No tracked file is changed")
    put()
    p = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    tracked = [ln for ln in lines if not ln.startswith("??")]
    mine = [ln for ln in lines if ln.startswith("??") and "b68" in ln]
    put(f"`git status --porcelain`: {len(lines)} lines; tracked files changed: {tracked or 'none'}; untracked B.68 paths: {mine}.")
    put()
    verdict("(g) no tracked file changed", p.returncode == 0 and not tracked,
            "only untracked paths; the auxiliary head and loss are installed at run time and no shared file is edited")
    put()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="redo the CPU run in data/.b68_check_train_r2_f0")
    args = ap.parse_args()
    put("# B.68 checks before any GPU run")
    put()
    put(f"Registered: ces_prediction/experiments/PREREGISTRATION_B68.md (section 3). Written by "
        f"ces_prediction/experiments/b68_auxtarget/check_b68.py on {time.strftime('%Y-%m-%d %H:%M')}; CPU only (one "
        "optimisation step on a tiny batch in (d) and (e)). Real data: data/s*.csv.")
    put()
    cache, meta = T.load_cache()
    cov = meta["coverage"]["block_rows"]
    put(f"Auxiliary-target cache: {T.CACHE_NPZ.relative_to(REPO)} (build_aux_b68.py, built {meta['built']}: "
        f"{meta['discharges']} discharges, {meta['grid_rows']:,} grid rows; estimator {meta['seconds_estimator']:.0f} s wall / "
        f"{meta['cpu_seconds_estimator']:.0f} s CPU on {meta['workers']} workers). Rows of grid blocks with a target: "
        f"{cov['all']['with_target']:,} of {cov['all']['rows']:,} ({cov['all']['share_with_target']:.4f}); with a measured T_i: "
        f"{cov[T.KIND_LABEL[T.KIND_TI]]['share_with_target']:.4f}; the GP with values on both sides "
        f"{cov['all']['gp_with_both_sides']:,}, the estimator's fallback to the last earlier value "
        f"{cov['all']['fallback_last_earlier_value']:,}, no value {cov['all']['no_value']:,}.")
    put()
    check_dry()
    from seq_data import load_grid_files
    log("grid")
    grid, dims = load_grid_files(DATA, T.DROP_STUCK, T.SPIKE_CUT_EV)
    for nm, e in cache.items():
        T.verify_against_grid(e, grid[nm], nm)
    check_target(grid, cache)
    check_withheld(grid, cache)
    check_loss(grid, dims, cache)
    check_checkpoint(args.fresh)
    check_envs()
    check_analyze()
    check_git()
    put("## Result")
    put()
    put("All checks pass." if not FAILS else "FAILED: " + ", ".join(FAILS))
    OUT_MD.write_text("\n".join(LINES) + "\n", encoding="utf-8", newline="\n")
    log(f"written {OUT_MD}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
