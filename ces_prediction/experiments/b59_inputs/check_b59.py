"""B.59 checks before any GPU run (PREREGISTRATION_B59.md section 3).  CPU only, no training.

  (a) a dry run of every arm (run_b59.py --dry-run --arms X, each arm and all four).
  (b) arm L changes the fast channels and nothing else: on every file of the grid (data treatment of the pooled
      estimator, statistics of the training discharges of fold r2 f0) the blocks of the wrapper with a shift of one
      row differ from those of seq_data.build_blocks only in x[:, :15], which equal the rows before; the
      training path and the scoring path call the same wrapper (the only references to build_blocks in train_seq and
      eval_seq go through the module global that fast_shift rebinds; called through those globals the blocks are
      shifted).
  (c) with a shift of zero rows the blocks are identical, array by array, to those of the estimator of B.39, and the
      env of arm L without B59_FAST_SHIFT is the env of the pooled estimator of B.39 (whole dict).
  (d) the first row of every block keeps its own values.
  (e) arms N, E, B give the env of the matching runs of B.39 arm A except for the split, the initialisation seed
      and the output directory (and differ from the pooled estimator of their fold only in CES_SEQ_MODEL).
  (f) every run is scored on the rows of the estimator of its fold: the row-defining env equals the estimator's;
      the frozen pooled estimator of r2 f0 scored on the CPU through eval_b59L.py with shift 0 and with shift 1
      gives the stored rows of that estimator (and, with shift 0, its stored errors); B.39 arm A scored v2n / v2e /
      v2b on the same rows as v2 on all four splits.

    py ces_prediction/experiments/b59_inputs/check_b59.py
Output: docs/paper/b59_checks.md (also printed).  Exit 1 if a check fails.  The two CPU scorings of (f) are written
to data/.b59_check_s{0,1}_r2_f0/ and reused when present.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import dis  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_b59 as R  # noqa: E402
import fast_shift as FS  # noqa: E402   (puts seq/ and ces_prediction/ on sys.path)

B39, DATA = R.B39, R.DATA
REPO = HERE.parents[2]
OUT_MD = REPO / "docs" / "paper" / "b59_checks.md"
T = "CES_TI"
ROW_FOLD = (2, 0)
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


def run_py(args, env=None, timeout=1800):
    return subprocess.run([sys.executable, "-B", *map(str, args)], cwd=REPO, env=env, capture_output=True,
                          text=True, timeout=timeout)


# ------------------------------------------------------------------------------ (a)
def check_dry():
    put("## (a) Dry run of every arm")
    put()
    put("| arms | return code | runs listed | runs per arm |")
    put("|---|---|---|---|")
    ok = True
    for arms in (["L"], ["N"], ["E"], ["B"], list(R.ARMS)):
        p = run_py([HERE / "run_b59.py", "--dry-run", "--arms", *arms])
        last = [ln for ln in p.stdout.splitlines() if " runs: " in ln]
        n = len([ln for ln in p.stdout.splitlines() if ln[:2].strip().isdigit() and " arm " in ln])
        good = p.returncode == 0 and n == 15 * len(arms) and bool(last)
        ok &= good
        put(f"| {' '.join(arms)} | {p.returncode} | {n} | {last[0].strip() if last else p.stderr.strip()[-200:]} |")
    put()
    verdict("(a) dry run", ok, "every arm alone and all four together list 15 runs per arm (60 in all) and start nothing")
    put()


# ------------------------------------------------------------------------------ (b) (c) (d)
def code_refs(mod, name):
    """(code object name, opname) of every instruction in a module (nested code included) that names `name`."""
    src = Path(mod.__file__).read_text(encoding="utf-8")
    hits = []

    def walk(co):
        for ins in dis.get_instructions(co):
            if ins.argval == name and ins.opname not in ("LOAD_CONST",):
                hits.append((co.co_name, ins.opname))
        for c in co.co_consts:
            if isinstance(c, types.CodeType):
                walk(c)

    walk(compile(src, mod.__file__, "exec"))
    return hits


def check_blocks():
    r, k = ROW_FOLD
    env = R.arm_env("L", r, k, R.run_dir("L", r, k))
    for key in [x for x in os.environ if x.startswith("CES_") or x.startswith("B59_")]:
        del os.environ[key]
    os.environ.update({x: v for x, v in env.items() if x.startswith("CES_")})
    os.environ["CES_SEQ_DEVICE"] = "cpu"
    import seq_data
    original = seq_data.build_blocks
    grid, dims = seq_data.load_grid_files(Path(env["CES_DATA_DIR"]), env["CES_DROP_STUCK_TARGETS"] == "1")
    man = json.loads((Path(env["CES_SPLIT_DIR"]) / "split_manifest.json").read_text(encoding="utf-8"))
    stats = seq_data.fit_stats(grid, dims, [n for n in man["train_files"] if n in grid])
    per_shot = env["CES_PER_SHOT_NORM"] == "1"
    w0, st0 = FS.wrap(original, 0)
    w1, st1 = FS.wrap(original, 1)
    n_fast = dims["n_fast"]
    agg = {"files": 0, "blocks": 0, "rows": 0, "s0_identical_blocks": 0, "s1_rest_identical_blocks": 0,
           "s1_fast_equal_prev": 0, "first_row_own": 0, "rows_changed": 0, "rows_unchanged_shift1": 0}
    groups = {"train": set(man["train_files"]), "val": set(man["val_files"]), "test": set(man["test_files"])}
    by_group = {g: 0 for g in groups}
    for nm in sorted(grid):
        b = original(grid[nm], dims, stats, per_shot_norm=per_shot)
        b0 = w0(grid[nm], dims, stats, per_shot_norm=per_shot)
        b1 = w1(grid[nm], dims, stats, per_shot_norm=per_shot)
        agg["files"] += 1
        for g, s in groups.items():
            by_group[g] += int(nm in s)
        if not (len(b) == len(b0) == len(b1)):
            raise SystemExit(f"FATAL: block count differs in {nm}")
        for x, y0, y1 in zip(b, b0, b1):
            agg["blocks"] += 1
            agg["rows"] += x["x"].shape[0]
            if all(np.array_equal(x[q], y0[q]) for q in ("x", "y", "mask", "time")) and x["x"].dtype == y0["x"].dtype:
                agg["s0_identical_blocks"] += 1
            if (all(np.array_equal(x[q], y1[q]) for q in ("y", "mask", "time"))
                    and np.array_equal(x["x"][:, n_fast:], y1["x"][:, n_fast:]) and x["x"].shape == y1["x"].shape):
                agg["s1_rest_identical_blocks"] += 1
            if np.array_equal(y1["x"][1:, :n_fast], x["x"][:-1, :n_fast]):
                agg["s1_fast_equal_prev"] += 1
            if np.array_equal(y1["x"][0, :n_fast], x["x"][0, :n_fast]):
                agg["first_row_own"] += 1
            diff = (y1["x"][:, :n_fast] != x["x"][:, :n_fast]).any(1)
            agg["rows_changed"] += int(diff.sum())
            agg["rows_unchanged_shift1"] += int((~diff[1:]).sum())
    # the paths that training and scoring take
    tr_refs = code_refs(__import__("train_seq"), "build_blocks")
    ev_refs = code_refs(__import__("eval_seq"), "build_blocks")
    train_seq, s_tr = FS.install_train(1)
    nm0 = sorted(man["train_files"])[0]
    via_train = train_seq.build_blocks(grid[nm0], dims, stats, per_shot_norm=per_shot)
    seq_data.build_blocks = original      # training and scoring run in separate processes; restore before the eval install
    eval_seq, s_ev = FS.install_eval(1)
    nm1 = sorted(man["test_files"])[0]
    via_eval = eval_seq.build_blocks(grid[nm1], dims, stats, per_shot_norm=per_shot)
    ref_tr = original(grid[nm0], dims, stats, per_shot_norm=per_shot)
    ref_ev = original(grid[nm1], dims, stats, per_shot_norm=per_shot)
    path_ok = (all(np.array_equal(a["x"][1:, :n_fast], c["x"][:-1, :n_fast]) and np.array_equal(a["x"][:, n_fast:], c["x"][:, n_fast:])
                   for a, c in zip(via_train, ref_tr))
               and all(np.array_equal(a["x"][1:, :n_fast], c["x"][:-1, :n_fast]) and np.array_equal(a["x"][:, n_fast:], c["x"][:, n_fast:])
                       for a, c in zip(via_eval, ref_ev))
               and s_tr["calls"] == 1 and s_ev["calls"] == 1)
    seq_data.build_blocks = original
    train_seq.build_blocks = original
    eval_seq.build_blocks = original
    allowed_ops = ("IMPORT_FROM", "STORE_NAME", "LOAD_GLOBAL")
    tr_ok = all(op in allowed_ops for _, op in tr_refs) and any(op == "LOAD_GLOBAL" for _, op in tr_refs)
    ev_ok = all(op in allowed_ops for _, op in ev_refs) and any(op == "LOAD_GLOBAL" for _, op in ev_refs)

    put("## (b) Arm L changes the fast channels and nothing else, in training and in scoring alike")
    put()
    put(f"Data treatment of the pooled estimator (fold r{r} f{k}: split {Path(env['CES_SPLIT_DIR']).name}, CES_PER_SHOT_NORM="
        f"{env['CES_PER_SHOT_NORM']}, CES_DROP_STUCK_TARGETS={env['CES_DROP_STUCK_TARGETS']}, CES_TI_SPIKE_CUT_EV="
        f"{env['CES_TI_SPIKE_CUT_EV']}); statistics of its {len(man['train_files'])} training discharges. Every file of the grid: "
        f"{agg['files']} files ({by_group['train']} train, {by_group['val']} validation, {by_group['test']} test), "
        f"{agg['blocks']:,} blocks, {agg['rows']:,} rows.")
    put()
    put("| quantity | value |")
    put("|---|---|")
    put(f"| blocks whose dt, CES-history channels, y, mask and time are identical under shift 1 | {agg['s1_rest_identical_blocks']:,} of {agg['blocks']:,} |")
    put(f"| blocks whose fast channels at rows 1.. equal those of the row before (shift 1) | {agg['s1_fast_equal_prev']:,} of {agg['blocks']:,} |")
    put(f"| rows whose fast channels change under shift 1 | {agg['rows_changed']:,} of {agg['rows']:,} |")
    put(f"| rows after the first of a block whose fast channels happen to equal the previous row's (no change) | {agg['rows_unchanged_shift1']:,} |")
    put(f"| references to build_blocks in train_seq.py | {sorted(set(tr_refs))} |")
    put(f"| references to build_blocks in eval_seq.py | {sorted(set(ev_refs))} |")
    put(f"| shifted blocks returned through train_seq.build_blocks ({nm0}) and eval_seq.build_blocks ({nm1}) after install | {path_ok} |")
    put()
    ok_b = (agg["s1_rest_identical_blocks"] == agg["blocks"] and agg["s1_fast_equal_prev"] == agg["blocks"]
            and path_ok and tr_ok and ev_ok)
    verdict("(b) only the fast channels", ok_b,
            "under shift 1 only x[:, :15] changes, and it equals the row before in every block of every file "
            "(train, validation and test files alike); train_seq (the `make` lambda of main) and eval_seq (build_pred_lookup) "
            "reach build_blocks only through the module global that fast_shift rebinds, and called through those "
            "globals the blocks come back shifted")
    put()
    put("## (c) With a shift of zero rows the inputs are those of the estimator of B.39")
    put()
    est_env = B39.pooled_env("v2", r, k, B39.pooled_run_dir("v2", r, k))
    envs_ok, n_env = True, 0
    for rr in R.REPEATS:
        for kk in range(5):
            out = R.run_dir("L", rr, kk)
            mine = R.arm_env("L", rr, kk, out, shift=0)
            mine.pop("B59_FAST_SHIFT")
            mine["CES_OUTPUT_DIR"] = str(B39.pooled_run_dir("v2", rr, kk))
            envs_ok &= mine == B39.pooled_env("v2", rr, kk, B39.pooled_run_dir("v2", rr, kk))
            n_env += 1
    m = json.loads((B39.pooled_run_dir("v2", r, k) / "metrics.json").read_text(encoding="utf-8"))
    rec_ok = (m["seq_model"] == "v2" and m["per_shot_input_norm"] is False and m["drop_stuck_targets"] is True
              and m["init_seed"] == int(est_env["CES_INIT_SEED"]) and m["seed"] == int(est_env["CES_SEED"])
              and m["extra_ti_channels"] == 0 and m["fixed_epochs"] is False)
    stats_ok = all(np.allclose(np.asarray(m["normalization"]["stats"][g][q], float), np.asarray(stats[g][q], float), rtol=0, atol=1e-6)
                   for g in ("bes", "ecei", "mc", "target") for q in ("mean", "std"))
    put("| quantity | value |")
    put("|---|---|")
    put(f"| blocks identical (x, y, mask, time; dtype) under shift 0 | {agg['s0_identical_blocks']:,} of {agg['blocks']:,} |")
    put(f"| arm L env without B59_FAST_SHIFT (output dir set to the estimator's) equal to run_b39.pooled_env('v2', r, k) | {envs_ok} ({n_env} folds) |")
    put(f"| the stored estimator .b39p_v2_cut_r{r}_f{k}: metrics.json agrees with that env | {rec_ok} |")
    put(f"| standardization statistics recomputed here equal those stored by the estimator (atol 1e-6) | {stats_ok} |")
    put()
    verdict("(c) shift 0 = estimator", agg["s0_identical_blocks"] == agg["blocks"] and envs_ok and rec_ok and stats_ok,
            "with a shift of zero rows every block equals seq_data.build_blocks array by array under the estimator's env "
            "and statistics (the scoring with shift 0 in (f) also reproduces the estimator's stored errors)")
    put()
    put("## (d) The first row of every block keeps its own values")
    put()
    verdict("(d) first row", agg["first_row_own"] == agg["blocks"],
            f"{agg['first_row_own']:,} of {agg['blocks']:,} blocks keep the fast channels of their first row under shift 1")
    put()
    return agg


# ------------------------------------------------------------------------------ (e)
def check_env_neb():
    put("## (e) Arms N, E, B: the env of B.39 arm A except for the split, the initialisation and the output directory")
    put()
    allowed = {"CES_SPLIT_DIR", "CES_CONTROL_METRICS", "CES_SEED", "CES_INIT_SEED", "CES_OUTPUT_DIR"}
    put("| arm | model | runs | variables that differ from B.39 arm A (any split seed) | non-CES variables equal | differ from the pooled estimator of the fold | "
        "B.39 arm A metrics.json agrees (model, standardization, held, extras, early stop) |")
    put("|---|---|---|---|---|---|---|")
    ok = True
    for arm in ("N", "E", "B"):
        model = R.MODEL[arm]
        seen, non_ok, est_keys = set(), True, set()
        for rr in R.REPEATS:
            for kk in range(5):
                out = R.run_dir(arm, rr, kk)
                mine = R.arm_env(arm, rr, kk, out)
                for s in B39.SEEDS:
                    ref = B39.split_env(model, s, s, B39.split_run_dir(model, s, s))
                    seen |= {x for x, _, _ in R.env_diff(R.ces_vars(mine), R.ces_vars(ref))}
                    non_ok &= {a: b for a, b in mine.items() if not a.startswith("CES_")} == \
                        {a: b for a, b in ref.items() if not a.startswith("CES_")}
                est = B39.pooled_env("v2", rr, kk, out)
                est_keys |= {x for x, _, _ in R.env_diff(R.ces_vars(mine), R.ces_vars(est))}
        rec_ok = True
        for s in B39.SEEDS:
            mm = json.loads((B39.split_run_dir(model, s, s) / "metrics.json").read_text(encoding="utf-8"))
            mine = R.arm_env(arm, 2, 0, R.run_dir(arm, 2, 0))
            rec_ok &= (mm["seq_model"] == mine["CES_SEQ_MODEL"] and mm["per_shot_input_norm"] == (mine["CES_PER_SHOT_NORM"] == "1")
                       and mm["drop_stuck_targets"] == (mine["CES_DROP_STUCK_TARGETS"] == "1") and mm["extra_ti_channels"] == 0
                       and mm["fixed_epochs"] is False and mine["CES_SEQ_FIXED_EPOCHS"] == "0"
                       and mm["epochs_run"] <= int(mine["CES_SEQ_EPOCHS"]))
        good = seen <= allowed and non_ok and est_keys == {"CES_SEQ_MODEL"} and rec_ok
        ok &= good
        put(f"| {arm} | {model} | 15 | {', '.join(sorted(seen))} | {non_ok} | {', '.join(sorted(est_keys))} | {rec_ok} |")
    put()
    verdict("(e) env of N, E, B", ok,
            "against B.39 arm A only CES_SPLIT_DIR, CES_CONTROL_METRICS (the harness statistics of the split), CES_SEED "
            "(the seed that names the split), CES_INIT_SEED and CES_OUTPUT_DIR differ; against the pooled estimator of the "
            "same fold only CES_SEQ_MODEL")
    put()


# ------------------------------------------------------------------------------ (f)
def cpu_score(S):
    r, k = ROW_FOLD
    out = DATA / f".b59_check_s{S}_r{r}_f{k}"
    npz = out / "comparison_errors_test.npz"
    if npz.exists():
        log(f"CPU scoring shift {S}: reused {out.name}")
        mp = out / "check_minutes.txt"
        return out, (float(mp.read_text()) if mp.exists() else None)
    src = B39.pooled_run_dir("v2", r, k)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "metrics.json", out / "metrics.json")
    shutil.copy2(src / "weights" / "seq_lstm.pth", out / "weights" / "seq_lstm.pth")
    env = R.arm_env("L", r, k, out, shift=S)
    env.update({"CES_SEQ_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": "", "B59_ALLOW_NO_TRAIN_RECORD": "1"})
    t0 = time.time()
    p = subprocess.run([sys.executable, "-B", str(HERE / "eval_b59L.py")], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=3000)
    (out / "check_eval.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
    log(f"CPU scoring shift {S}: rc {p.returncode} in {(time.time() - t0) / 60:.1f} min")
    (out / "check_minutes.txt").write_text(f"{(time.time() - t0) / 60:.1f}")
    if not npz.exists():
        raise SystemExit(f"FATAL: CPU scoring with shift {S} failed; see {out / 'check_eval.log'}")
    return out, round((time.time() - t0) / 60, 1)


def check_rows():
    put("## (f) Every run is scored on the rows of the estimator of its fold")
    put()
    r, k = ROW_FOLD
    row_vars_ok = True
    for arm in R.ARMS:
        for rr in R.REPEATS:
            for kk in range(5):
                mine = R.arm_env(arm, rr, kk, R.run_dir(arm, rr, kk))
                est = B39.pooled_env("v2", rr, kk, B39.pooled_run_dir("v2", rr, kk))
                keys = {x for x, _, _ in R.env_diff(R.ces_vars(mine), R.ces_vars(est))}
                row_vars_ok &= keys <= {"CES_SEQ_MODEL", "CES_OUTPUT_DIR", "B59_FAST_SHIFT"}
    est = np.load(B39.pooled_run_dir("v2", r, k) / "comparison_errors_test.npz")
    res = {}
    for S in (0, 1):
        out, mins = cpu_score(S)
        z = np.load(out / "comparison_errors_test.npz")
        same = (len(z[f"{T}_shot"]) == len(est[f"{T}_shot"]) and np.array_equal(z[f"{T}_shot"], est[f"{T}_shot"])
                and np.array_equal(z[f"{T}_row"], est[f"{T}_row"]))
        refs = all(np.array_equal(z[f"{T}_se_{q}"], est[f"{T}_se_{q}"]) for q in ("persistence", "pchip", "gp", "gp_causal"))
        e_m, e_s = np.sqrt(z[f"{T}_se_model"].astype(float)), np.sqrt(est[f"{T}_se_model"].astype(float))
        res[S] = {"rows": int(len(z[f"{T}_shot"])), "same_rows": bool(same), "refs_identical": bool(refs),
                  "max_abs_err_diff_eV": float(np.max(np.abs(e_m - e_s))) if same else None,
                  "p50_p99_abs_err_diff_eV": [float(np.percentile(np.abs(e_m - e_s), q)) for q in (50, 99)] if same else None,
                  "skill_vs_stored_estimator": float(1 - z[f"{T}_se_model"].sum() / est[f"{T}_se_model"].sum()) if same else None,
                  "skill_vs_gp_causal": float(1 - z[f"{T}_se_model"].sum() / z[f"{T}_se_gp_causal"].sum()),
                  "minutes": mins,
                  "shift_eval": json.loads((out / "b59_shift_eval.json").read_text(encoding="utf-8"))}
    # B.39 arm A: v2n / v2e / v2b scored on the rows of v2 (four splits)
    arm_a = {}
    for s in B39.SEEDS:
        ref = np.load(B39.split_run_dir("v2", s, s) / "comparison_errors_test.npz")
        for m in B39.ABLATIONS:
            z = np.load(B39.split_run_dir(m, s, s) / "comparison_errors_test.npz")
            arm_a[f"s{s} {m}"] = bool(np.array_equal(z[f"{T}_shot"], ref[f"{T}_shot"]) and np.array_equal(z[f"{T}_row"], ref[f"{T}_row"]))
    put("| quantity | value |")
    put("|---|---|")
    put(f"| every run of the four arms: CES variables that differ from the estimator of its fold within CES_SEQ_MODEL, CES_OUTPUT_DIR, B59_FAST_SHIFT | {row_vars_ok} (60 runs) |")
    for S in (0, 1):
        x = res[S]
        put(f"| frozen estimator r{r} f{k} scored on the CPU through eval_b59L.py, shift {S}: rows / same rows as stored / references identical | "
            f"{x['rows']:,} / {x['same_rows']} / {x['refs_identical']} |")
        put(f"| shift {S}: |difference| of the absolute error to the stored (GPU) scoring, median / 99th pct / max (eV); skill over the stored "
            f"estimator; skill over the causal GP; wrapper calls; minutes | {x['p50_p99_abs_err_diff_eV'][0]:.3g} / "
            f"{x['p50_p99_abs_err_diff_eV'][1]:.3g} / {x['max_abs_err_diff_eV']:.3g}; {x['skill_vs_stored_estimator']:+.2e}; "
            f"{x['skill_vs_gp_causal']:+.4f}; {x['shift_eval']['wrapper_calls']}; {x['minutes']} |")
    put(f"| B.39 arm A: v2n, v2e, v2b scored on the rows of v2 (4 splits x 3) | {sum(arm_a.values())} of {len(arm_a)} |")
    put()
    ok = (row_vars_ok and all(res[S]["same_rows"] and res[S]["refs_identical"] for S in (0, 1))
          and abs(res[0]["skill_vs_stored_estimator"]) < 1e-3 and res[0]["p50_p99_abs_err_diff_eV"][1] < 0.5
          and res[1]["p50_p99_abs_err_diff_eV"][1] > 5.0 and all(arm_a.values()))
    verdict("(f) scored rows", ok,
            "the rows are fixed by the split, the harness statistics, CES_SEED, the window, the caps and the data treatment, "
            "which every run shares with the estimator of its fold; scoring through the arm-L path gives the estimator's rows "
            "with either shift (shift 0 also reproduces its errors: skill over the stored GPU scoring within 1e-3, 99 % of rows within 0.5 eV; "
            "the residue is CPU against cuDNN arithmetic), and the network does not enter the rows "
            "(B.39 arm A)")
    put()
    put("Inferred, not run: that N, E, B runs of the pooled folds will be scored on the estimator's rows rests on the "
        "shared env (verified above) and on eval_seq choosing rows without the network (shown on the four splits of B.39 arm A); "
        "it is checked again at readout (analyze_b59.py stops if a run's rows differ).")
    put()


def main():
    put("# B.59 checks before any GPU run")
    put()
    put(f"Registered: ces_prediction/experiments/PREREGISTRATION_B59.md (section 3). Written by "
        f"ces_prediction/experiments/b59_inputs/check_b59.py on {time.strftime('%Y-%m-%d %H:%M')}; CPU only, no network "
        "trained. Real data: data/s*.csv (641 files).")
    put()
    check_dry()
    check_blocks()
    check_env_neb()
    check_rows()
    put("## Result")
    put()
    put("All checks pass." if not FAILS else "FAILED: " + ", ".join(FAILS))
    OUT_MD.write_text("\n".join(LINES) + "\n", encoding="utf-8", newline="\n")
    log(f"written {OUT_MD}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
