"""B.67 checks before any GPU run (PREREGISTRATION_B67.md section 3, 'Checks before any run').  CPU only; no network
is trained (randomly initialised weights stand in where a v2s / v2a checkpoint is needed).

  (a) dry run of each arm and of both (run_b67.py --dry-run).
  (b) forward shapes and parameter counts of v2s and v2a; the source groups against seq_data's layout on a real file;
      uniform gates give the plain concatenation of the embeddings (S) / their mean (A); a state dict reloads under the
      registered name.
  (c) causality: perturbing the inputs after row t leaves the outputs and gates up to t unchanged (both models, train
      mode with dropout and eval mode, whole block and packed); the gates of a row read that row only; the V_rot
      branch never sees the fast channels.
  (d) the wrapper registers the variants for training and for scoring: train_seq and eval_seq reach SEQ_MODELS (and
      eval_seq load_grid_files / build_clean_val_subset) only through module globals; train_b67.main() run up to the
      first training pass builds a SeqCESLSTMv2Gate of the right mode under the run's env (v2s and v2a; nothing is
      trained or written); eval_b67.main() with v2a runs eval_seq.compare through the prediction lookup (two files);
      eval_b67.py with v2s scores fold r2 f0 end to end on the CPU (see (f)).
  (e) the 30 planned envs equal those of B.59 arm N (run_b59.arm_env('N', r, k, out)) apart from CES_SEQ_MODEL and the
      output directory; B.59 arm N's stored metrics.json agree with them.
  (f) the gate readout is aligned to the scored rows: the B.39 network of r2 f0 rescored on the CPU through the gate
      readout's table reproduces its stored (GPU) errors; the same table with the rows shifted by one does not; the
      end-to-end CPU scoring of v2s of (d) is on the network's rows with identical reference errors, and its gate file
      is on those rows with the squared errors reproduced bit for bit in the scoring process.
  (g) analyze_b67.py: without --partial it stops on zero runs and writes nothing; with --partial it exits cleanly; the
      selftest (network for S, twin for A, synthetic gates) reads every statement and the gate readout.
  (h) no tracked file is changed (git status --porcelain).

    py ces_prediction/experiments/b67_gated/check_b67.py [--fresh]
Output: docs/paper/b67_checks.md (also printed).  Exit 1 if a check fails.  The CPU scorings are written to
data/.b67_check_*/ and reused when present (--fresh redoes them).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import copy  # noqa: E402
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
import run_b67 as R  # noqa: E402
import model_seq_v2gate as M  # noqa: E402  (seq/ and ces_prediction/ on sys.path)
import gates_b67 as G  # noqa: E402
import torch  # noqa: E402

B39, DATA = R.B39, R.DATA
REPO = HERE.parents[2]
OUT_MD = REPO / "docs" / "paper" / "b67_checks.md"
T = "CES_TI"
FOLD = (2, 0)
CHECK_EVAL = {"S": DATA / ".b67_check_eval_S_r2_f0", "A": DATA / ".b67_check_eval_A_r2_f0"}
CHECK_TRAIN = {"S": DATA / ".b67_check_train_S_r2_f0", "A": DATA / ".b67_check_train_A_r2_f0"}
CHECK_ANALYZE = DATA / ".b67_check_analyze"
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


def cpu_env(arm, out, **extra):
    env = R.arm_env(arm, *FOLD, out)
    env.update({"CES_SEQ_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4", **extra})
    return env


def pin_env(env):
    for key in [x for x in os.environ if x.startswith("CES_")]:
        del os.environ[key]
    os.environ.update({x: v for x, v in env.items() if x.startswith("CES_")})


def registry():
    M.register()
    import seq_models
    return seq_models.SEQ_MODELS


# ------------------------------------------------------------------------------ (a)
def check_dry():
    put("## (a) Dry run of each arm")
    put()
    put("| arms | return code | runs listed | runs per arm |")
    put("|---|---|---|---|")
    ok = True
    for arms in (["S"], ["A"], list(R.ARMS)):
        p = run_py([HERE / "run_b67.py", "--dry-run", "--arms", *arms])
        last = [ln for ln in p.stdout.splitlines() if " runs: " in ln]
        n = len([ln for ln in p.stdout.splitlines() if ln[:2].strip().isdigit() and " arm " in ln])
        good = p.returncode == 0 and n == 15 * len(arms) and bool(last)
        ok &= good
        put(f"| {' '.join(arms)} | {p.returncode} | {n} | {last[0].strip() if last else p.stderr.strip()[-200:]} |")
    put()
    verdict("(a) dry run", ok, "each arm alone and both together list 15 runs per arm (30 in all) and start nothing")
    put()


# ------------------------------------------------------------------------------ (b) (c)
def check_model(grid, dims, est_stats):
    reg = registry()
    put("## (b) Forward shapes, parameter counts, source groups")
    put()
    from model_seq_v2sub import GROUPS as SUB_GROUPS
    from seq_data import infer_columns, build_blocks, N_FEATURES
    bes, ecei, mc = infer_columns(sorted(DATA.glob("s*.csv"))[0])
    layout_names = (all(c.startswith("BES_") for c in bes) and all(c.startswith("ECEI_") for c in ecei)
                    and all(c.startswith("MC") for c in mc) and (len(bes), len(ecei), len(mc)) == (9, 4, 2))
    groups_ok = (M.GROUPS == (("bes", 0, 9), ("ecei", 9, 13), ("mc", 13, 15), ("ces", 15, 22))
                 and all(tuple(range(a, b)) == SUB_GROUPS[g] for g, a, b in M.GROUPS if g in SUB_GROUPS)
                 and dims["n_fast"] == 15 and N_FEATURES == 22 and dims["bes"] == 9 and dims["ecei"] == 4 and dims["mc"] == 2)
    # the 22 channels of real blocks: fast channels are the standardized BES / ECEI / MC columns, then dt and the history
    nm = sorted(grid)[0]
    arr = grid[nm]
    blk = build_blocks(arr, dims, est_stats, per_shot_norm=False)[0]
    n = blk["x"].shape[0]
    a0 = int(np.flatnonzero(arr[:, 0] == blk["time"][0])[0])
    raw = arr[a0:a0 + n].astype(np.float64)
    zs = {g: (raw[:, 3 + o:3 + o + w] - est_stats[g]["mean"].astype(np.float64)) / est_stats[g]["std"].astype(np.float64)
          for g, o, w in (("bes", 0, 9), ("ecei", 9, 4), ("mc", 13, 2))}
    t = raw[:, 0]
    dt = np.zeros(n)
    dt[1:] = np.diff(t)
    last, stale_ref = None, np.zeros(n)
    for i in range(n):
        if last is not None:
            stale_ref[i] = np.log1p(t[i] - last)
        if np.isfinite(raw[i, 1]):
            last = t[i]
    x = blk["x"].astype(np.float64)
    chan_ok = (np.allclose(x[:, 0:9], zs["bes"], atol=1e-4) and np.allclose(x[:, 9:13], zs["ecei"], atol=1e-4)
               and np.allclose(x[:, 13:15], zs["mc"], atol=1e-4) and np.allclose(x[:, 15], np.log1p(dt), atol=1e-6)
               and np.allclose(x[:, 17], stale_ref, atol=1e-5) and set(np.unique(x[:, 18])) <= {0.0, 1.0}
               and set(np.unique(x[:, 21])) <= {0.0, 1.0})
    put("| quantity | v2s | v2a | v2 (reference) |")
    put("|---|---|---|---|")
    shapes, params, gsum, uni = {}, {}, {}, {}
    torch.manual_seed(0)
    xr = torch.randn(3, 64, 22)
    for v in ("v2s", "v2a", "v2"):
        m = reg[v]()
        m.eval()
        with torch.no_grad():
            o = m(xr)
            shapes[v] = tuple(o.shape)
            params[v] = (m.n_params, sum(p.numel() for p in m.parameters()))
            if v != "v2":
                o2, g = m.forward_with_gates(xr)
                gsum[v] = (tuple(g.shape), float((g.sum(-1) - 1).abs().max()), bool(torch.equal(o, o2)),
                           tuple(m.lstm_ti.weight_ih_l0.shape))
                mu = copy.deepcopy(m)
                mu.gate_mlp[-1].weight.zero_()
                mu.gate_mlp[-1].bias.zero_()
                a, u = mu.gates_and_input(xr)
                e = mu.embeddings(xr)
                want = torch.cat(e, -1) if v == "v2s" else torch.stack(e, -2).mean(-2)
                uni[v] = (float((a - 0.25).abs().max()), float((u - want).abs().max()))
                sd = m.state_dict()
                m2 = reg[v]()
                m2.load_state_dict(sd, strict=True)
                m2.eval()
                gsum[v] += (bool(torch.equal(m2(xr), o)),)
    put(f"| output shape for x (3, 64, 22) | {shapes['v2s']} | {shapes['v2a']} | {shapes['v2']} |")
    put(f"| parameters (model.n_params / counted) | {params['v2s'][0]:,} / {params['v2s'][1]:,} | {params['v2a'][0]:,} / "
        f"{params['v2a'][1]:,} | {params['v2'][0]:,} / {params['v2'][1]:,} |")
    put(f"| gates shape; max abs(sum of gates - 1) | {gsum['v2s'][0]}; {gsum['v2s'][1]:.1e} | {gsum['v2a'][0]}; {gsum['v2a'][1]:.1e} | - |")
    put(f"| T_i LSTM input weight (4 x 160, input width) | {gsum['v2s'][3]} | {gsum['v2a'][3]} | (640, 22) |")
    put(f"| forward == forward_with_gates[0]; state dict reloads (strict) and gives the same output | {gsum['v2s'][2]}, {gsum['v2s'][4]} | "
        f"{gsum['v2a'][2]}, {gsum['v2a'][4]} | - |")
    put(f"| gate output layer zeroed: max abs(gate - 0.25); max abs(LSTM input - plain concatenation (S) / mean (A) of the embeddings) | "
        f"{uni['v2s'][0]:.1e}; {uni['v2s'][1]:.1e} | {uni['v2a'][0]:.1e}; {uni['v2a'][1]:.1e} | - |")
    put()
    put(f"Source groups: {', '.join(f'{lab} x[..., {a}:{b}]' for lab, (_, a, b) in zip(M.SOURCE_LABELS, M.GROUPS))}. "
        f"CSV columns of the first file: {len(bes)} BES_*, {len(ecei)} ECEI_*, {len(mc)} MC* in that order; grid dims {dims}; "
        f"model_seq_v2sub.GROUPS {dict(SUB_GROUPS)}. On the first block of {nm} ({n} rows) the 22 channels of seq_data.build_blocks "
        f"are: the BES, ECEI and MC columns standardized with the network's training statistics (x[:, 0:15]), log1p of the "
        f"step (x[:, 15]), and the T_i staleness log1p(t - last measured T_i) (x[:, 17]); the has-flags (x[:, 18], x[:, 21]) "
        f"are 0 or 1: {chan_ok}.")
    put()
    ok_b = (shapes["v2s"] == shapes["v2a"] == (3, 64, 2) and all(params[v][0] == params[v][1] < 1_000_000 for v in params)
            and all(gsum[v][0] == (3, 64, 4) and gsum[v][1] < 1e-5 and gsum[v][2] and gsum[v][4] for v in gsum)
            and gsum["v2s"][3] == (640, 128) and gsum["v2a"][3] == (640, 32)
            and all(uni[v][0] < 1e-7 and uni[v][1] < 1e-6 for v in uni) and groups_ok and layout_names and chan_ok)
    verdict("(b) shapes and parameters", ok_b,
            f"v2s {params['v2s'][0]:,} and v2a {params['v2a'][0]:,} parameters (< 1,000,000; seq_v2 {params['v2'][0]:,}); "
            "outputs (B, L, 2), gates (B, L, 4) summing to one; the groups are BES 0-8, ECEI 9-12, Mirnov 13-14, CES-derived "
            "15-21 of the real layout; uniform gates give the plain concatenation (S) / the mean (A)")
    put()

    put("## (c) Causality")
    put()
    put("| model | mode | lengths | outputs and gates at rows <= t unchanged (max abs diff) | later outputs changed | "
        "gates of other rows unchanged when only row t is perturbed | V_rot output unchanged when every fast channel is perturbed |")
    put("|---|---|---|---|---|---|---|")
    ok_c = True
    B, L, t = 3, 64, 29
    for v in ("v2s", "v2a"):
        torch.manual_seed(1)
        m = reg[v]()
        g0 = torch.Generator().manual_seed(2)
        x = torch.randn(B, L, 22, generator=g0)
        x_after = x.clone()
        x_after[:, t + 1:] = 3.0 * torch.randn(B, L - t - 1, 22, generator=g0)
        x_row = x.clone()
        x_row[:, t] = 3.0 * torch.randn(B, 22, generator=g0)
        x_fast = x.clone()
        x_fast[..., :15] = 3.0 * torch.randn(B, L, 15, generator=g0)
        for mode in ("eval", "train"):
            m.train(mode == "train")
            for lengths in (None, torch.tensor([L, L - 9, t + 6])):
                outs = {}
                for key, xx in (("x", x), ("after", x_after), ("row", x_row), ("fast", x_fast)):
                    torch.manual_seed(7)                      # the same dropout masks in train mode
                    with torch.no_grad():
                        outs[key] = m.forward_with_gates(xx, lengths)
                (o, g), (oa, ga), (orow, grow_), (of, _) = outs["x"], outs["after"], outs["row"], outs["fast"]
                d = max(float((o[:, :t + 1] - oa[:, :t + 1]).abs().max()), float((g[:, :t + 1] - ga[:, :t + 1]).abs().max()))
                later = bool((o[0, t + 1:] - oa[0, t + 1:]).abs().max() > 0)
                other = torch.ones(L, dtype=torch.bool)
                other[t] = False
                row_local = bool(torch.equal(g[:, other], grow_[:, other])) and not torch.equal(g[:, t], grow_[:, t])
                lim = L if lengths is None else int(lengths.min())
                vrot = bool(torch.equal(o[:, :lim, 1], of[:, :lim, 1]))
                good = d == 0.0 and later and row_local and vrot
                ok_c &= good
                put(f"| {v} | {mode} | {'whole block' if lengths is None else 'packed ' + str(lengths.tolist())} | "
                    f"{d == 0.0} ({d:.1e}) | {later} | {row_local} | {vrot} |")
    put()
    verdict("(c) causality", ok_c,
            f"with the inputs after row t = {t} replaced, every output and gate at rows 0..{t} is identical (bit for bit), in eval "
            "mode and in train mode (dropout masks drawn from the same seed), on whole blocks and packed sequences; a row's "
            "gates change only when that row changes; the V_rot output does not move when the 15 fast channels are replaced")
    put()


# ------------------------------------------------------------------------------ (d)
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
    return sorted(set(hits))


def make_check_run(arm, out, est_dir):
    """A run directory holding the network's metrics.json (statistics of fold r2 f0) with seq_model set to the arm's
    variant and randomly initialised weights of that variant (initialisation seed of the fold).  Not trained."""
    reg = registry()
    variant = R.MODEL[arm]
    (out / "weights").mkdir(parents=True, exist_ok=True)
    m = json.loads((est_dir / "metrics.json").read_text(encoding="utf-8"))
    torch.manual_seed(int(R.arm_env(arm, *FOLD, out)["CES_INIT_SEED"]))
    model = reg[variant]()
    m.update({"seq_model": variant, "n_params": model.n_params,
              "b67_check": "randomly initialised, NOT trained; statistics copied from " + est_dir.name})
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out / "weights" / "seq_lstm.pth")


def probe_train():
    """Subprocess body: train_b67.main() under the run's env, stopped at its first training pass."""
    import train_seq
    import train_b67

    class Reached(Exception):
        pass

    seen = {}

    def first_pass(model, blocks, *a, **k):
        seen["model"], seen["blocks"] = model, len(blocks)
        raise Reached()

    train_seq.masked_pass = first_pass
    try:
        train_b67.main()
    except Reached:
        pass
    import seq_models
    m = seen.get("model")
    out = Path(os.environ["CES_OUTPUT_DIR"])
    print("PROBE " + json.dumps({
        "reached": m is not None, "variant": os.environ["CES_SEQ_MODEL"], "class": type(m).__name__ if m else None,
        "mode": getattr(m, "mode", None), "n_params": getattr(m, "n_params", None), "train_blocks": seen.get("blocks"),
        "registry_is_train_seq_registry": train_seq.SEQ_MODELS is seq_models.SEQ_MODELS,
        "written": sorted(p.name for p in out.rglob("*")) if out.exists() else []}), flush=True)


def probe_eval():
    """Subprocess body: eval_b67.main() with v2a, through eval_seq.compare up to the prediction lookup (2 files)."""
    import eval_seq
    import eval_b67

    class Reached(Exception):
        pass

    seen = {}
    orig_lookup, orig_cap = eval_seq.build_pred_lookup, G.install_captures

    def lookup(data_dir, eval_names, *a, **k):
        names = list(eval_names)[:2]
        res = orig_lookup(data_dir, names, *a, **k)
        seen["files"] = names
        seen["entries"] = sum(len(v) for v in res["full"].values())
        seen["finite"] = bool(all(np.isfinite(np.asarray(list(v.values()))).all() for v in res["full"].values()))
        raise Reached()

    def captures(es):
        c = orig_cap(es)
        seen["cap"] = c
        return c

    eval_seq.build_pred_lookup = lookup
    G.install_captures = captures
    try:
        eval_b67.main()
    except Reached:
        pass
    import seq_models
    cap = seen.get("cap", {})
    print("PROBE " + json.dumps({
        "reached": "entries" in seen, "variant": os.environ["CES_SEQ_MODEL"], "files": seen.get("files"),
        "entries": seen.get("entries"), "finite": seen.get("finite"),
        "registry_is_eval_seq_registry": eval_seq.SEQ_MODELS is seq_models.SEQ_MODELS,
        "grid_calls": cap.get("grid_calls"), "dataset_calls": cap.get("dataset_calls")}), flush=True)


def parse_probe(p):
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("PROBE ")]
    if p.returncode != 0 or not lines:
        return {"error": f"rc {p.returncode}: {p.stderr.strip()[-400:]}"}
    return json.loads(lines[-1][6:])


def check_registry(fresh):
    put("## (d) The wrapper registers the variants for training and for scoring")
    put()
    import train_seq
    import eval_seq
    refs = {"train_seq SEQ_MODELS": code_refs(train_seq, "SEQ_MODELS"),
            "eval_seq SEQ_MODELS": code_refs(eval_seq, "SEQ_MODELS"),
            "eval_seq load_grid_files": code_refs(eval_seq, "load_grid_files"),
            "eval_seq build_clean_val_subset": code_refs(eval_seq, "build_clean_val_subset")}
    allowed = ("IMPORT_FROM", "STORE_NAME", "LOAD_GLOBAL")
    refs_ok = all(all(op in allowed for _, op in v) and any(op == "LOAD_GLOBAL" for _, op in v) for v in refs.values())
    put("| quantity | value |")
    put("|---|---|")
    for k, v in refs.items():
        put(f"| references to {k} | {v} |")
    probes = {}
    for arm in ("S", "A"):
        out = CHECK_TRAIN[arm]
        if out.exists():
            shutil.rmtree(out)
        log(f"train probe {arm}")
        p = run_py([HERE / "check_b67.py", "--probe-train"], env=cpu_env(arm, out, CES_SEQ_MAX_FILES="2"), timeout=1800)
        probes[arm] = parse_probe(p)
        put(f"| train_b67.main() under the env of {arm} r2 f0 (CPU, 2 training files), stopped at the first training pass | {probes[arm]} |")
    tr_ok = all(probes[a].get("reached") and probes[a].get("class") == "SeqCESLSTMv2Gate" and probes[a].get("mode") == a
                and probes[a].get("variant") == R.MODEL[a] and probes[a].get("registry_is_train_seq_registry")
                and probes[a].get("written") == [] and probes[a].get("n_params", 0) < 1_000_000 for a in ("S", "A"))
    est = B39.pooled_run_dir("v2", *FOLD)
    outA = CHECK_EVAL["A"]
    if outA.exists():
        shutil.rmtree(outA)
    make_check_run("A", outA, est)
    log("eval probe A")
    pe = parse_probe(run_py([HERE / "check_b67.py", "--probe-eval"], env=cpu_env("A", outA), timeout=1800))
    put(f"| eval_b67.main() with v2a (random weights) under the env of A r2 f0 (CPU), through eval_seq.compare up to the "
        f"prediction lookup of two test files | {pe} |")
    ev_ok = bool(pe.get("reached") and pe.get("entries", 0) > 0 and pe.get("finite") and pe.get("registry_is_eval_seq_registry")
                 and pe.get("grid_calls") == 1 and pe.get("dataset_calls") == 1)
    put()
    verdict("(d) registry seen by training and scoring", refs_ok and tr_ok and ev_ok,
            "train_seq (main) and eval_seq (build_pred_lookup) resolve CES_SEQ_MODEL through the module global SEQ_MODELS, "
            "the dict register() inserts into; under the run's env train_b67.main() builds a SeqCESLSTMv2Gate of mode S / A "
            "(stopped before any optimisation step, nothing written) and eval_b67.main() predicts with v2a through "
            "eval_seq's own lookup; eval_seq reaches load_grid_files and build_clean_val_subset only through the module "
            "globals that the gate readout's captures wrap (each called once); the end-to-end scoring with v2s is in (f)")
    put()


# ------------------------------------------------------------------------------ (e)
def check_envs():
    put("## (e) The 30 planned environments equal those of B.59 arm N apart from CES_SEQ_MODEL and the directories")
    put()
    sys.path.insert(0, str(HERE.parent / "b59_inputs"))
    import run_b59 as R59
    put("| arm | runs | variables that differ from B.59 arm N of the same fold (whole env) | differ from the B.39 network of the fold "
        "| B.59 arm N metrics.json agree (init and split seeds, standardization, held, fixed epochs, cap) |")
    put("|---|---|---|---|---|")
    ok = True
    for arm in R.ARMS:
        keys, est_keys, n, rec_ok = set(), set(), 0, True
        for r in R.REPEATS:
            for k in range(5):
                mine = R.arm_env(arm, r, k, R.run_dir(arm, r, k))
                ref = R59.arm_env("N", r, k, R59.run_dir("N", r, k))
                d = {x for x in set(mine) | set(ref) if mine.get(x) != ref.get(x)}
                keys |= d
                ok &= d == {"CES_SEQ_MODEL", "CES_OUTPUT_DIR"} and mine["CES_SEQ_MODEL"] == R.MODEL[arm] \
                    and ref["CES_SEQ_MODEL"] == "v2n"
                est = B39.pooled_env("v2", r, k, B39.pooled_run_dir("v2", r, k))
                est_keys |= {x for x in set(mine) | set(est) if mine.get(x) != est.get(x)}
                mp = R59.run_dir("N", r, k) / "metrics.json"
                if mp.exists():
                    m = json.loads(mp.read_text(encoding="utf-8"))
                    rec_ok &= (m["seq_model"] == "v2n" and m["init_seed"] == int(mine["CES_INIT_SEED"])
                               and m["seed"] == int(mine["CES_SEED"]) and m["per_shot_input_norm"] is False
                               and m["drop_stuck_targets"] is True and m["fixed_epochs"] is False
                               and m["epochs_run"] <= int(mine["CES_SEQ_EPOCHS"]) and m["extra_ti_channels"] == 0)
                else:
                    rec_ok = False
                n += 1
        ok &= est_keys == {"CES_SEQ_MODEL", "CES_OUTPUT_DIR"} and rec_ok
        put(f"| {arm} | {n} | {', '.join(sorted(keys))} | {', '.join(sorted(est_keys))} | {rec_ok} |")
    put()
    verdict("(e) environments", ok,
            "for every fold of r2 to r4, the whole env (CES_* and the rest) of S and of A equals that of B.59 arm N except "
            "CES_SEQ_MODEL (v2s / v2a for v2n) and CES_OUTPUT_DIR; against the pooled network of B.39 likewise; the "
            "stored B.59 arm N runs record the same seeds and data treatment")
    put()


# ------------------------------------------------------------------------------ (f)
def check_alignment(grid, dims, fresh):
    put("## (f) The gate readout is aligned to the scored rows")
    put()
    from evaluate import _load_stats, build_clean_val_subset
    est = B39.pooled_run_dir("v2", *FOLD)
    env = B39.pooled_env("v2", *FOLD, est)
    pin_env(env)
    os.environ["CES_SEQ_DEVICE"] = "cpu"
    S = G.env_settings()
    man = json.loads((S["split_dir"] / "split_manifest.json").read_text(encoding="utf-8"))
    eval_files = set(man["test_files"])
    stats = _load_stats(json.loads(S["control_metrics"].read_text(encoding="utf-8")))
    log("dataset")
    dataset, _, _ = build_clean_val_subset(S["data_dir"], S["window_size"], stats, eval_files, S["max_val_samples"], S["seed"])
    metrics = json.loads((est / "metrics.json").read_text(encoding="utf-8"))
    my_stats = _load_stats(metrics)
    model = G.load_model(est, "v2", torch.device("cpu"))
    log("network r2 f0 table (CPU)")
    tables = G.block_tables(model, grid, dims, sorted(eval_files), my_stats, torch.device("cpu"), S["per_shot"])
    z = np.load(est / "comparison_errors_test.npz")
    c0 = G.check(G.align(tables, dataset, z[f"{T}_shot"], z[f"{T}_row"], 0), z)
    c1 = G.check(G.align(tables, dataset, z[f"{T}_shot"], z[f"{T}_row"], 1), z)
    put("| quantity | value |")
    put("|---|---|")
    put(f"| B.39 network r2 f0 rescored on the CPU through the gate readout's table: rows / found / squared errors identical "
        f"to the stored (GPU) ones | {c0['rows']:,} / {c0['found']:,} / {c0['identical_rows']:,} |")
    put(f"| same: |difference of the absolute errors| median / 99th pct / max (eV); skill over the stored errors | "
        f"{c0['p50_abs_err_diff_eV']:.3g} / {c0['p99_abs_err_diff_eV']:.3g} / {c0['max_abs_err_diff_eV']:.3g}; "
        f"{c0['skill_vs_stored']:+.2e} |")
    put(f"| negative control, every row looked up one row later: found / median / 99th pct (eV); skill over the stored errors; "
        f"check passes | {c1['found']:,} / {c1['p50_abs_err_diff_eV']:.3g} / {c1['p99_abs_err_diff_eV']:.3g}; "
        f"{c1['skill_vs_stored']:+.2e}; {c1['passes']} |")
    align_ok = c0["passes"] and not c1["passes"]
    # the end-to-end CPU scoring of v2s (eval_b67.py as a script)
    out = CHECK_EVAL["S"]
    npz, gnpz = out / "comparison_errors_test.npz", out / R.GATES_NPZ
    if fresh or not (npz.exists() and gnpz.exists()):
        if out.exists():
            shutil.rmtree(out)
        make_check_run("S", out, est)
        log("end-to-end CPU scoring of v2s (random weights), eval_b67.py")
        t0 = time.time()
        p = run_py([HERE / "eval_b67.py"], env=cpu_env("S", out), timeout=5400)
        (out / "check_eval.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
        (out / "check_minutes.txt").write_text(f"{(time.time() - t0) / 60:.1f}")
        log(f"eval_b67.py rc {p.returncode} in {(time.time() - t0) / 60:.1f} min")
        if not (npz.exists() and gnpz.exists()):
            put(f"| end-to-end CPU scoring of v2s | FAILED, see {out / 'check_eval.log'} |")
            verdict("(f) gate alignment", False, "eval_b67.py did not write its outputs")
            return
    else:
        log(f"end-to-end CPU scoring of v2s: reused {out.name}")
    e = np.load(npz)
    gz = np.load(gnpz)
    gchk = json.loads((out / R.GATES_JSON).read_text(encoding="utf-8"))
    same = (len(e[f"{T}_shot"]) == len(z[f"{T}_shot"]) and np.array_equal(e[f"{T}_shot"], z[f"{T}_shot"])
            and np.array_equal(e[f"{T}_row"], z[f"{T}_row"]))
    refs = same and all(np.array_equal(e[f"{T}_se_{q}"], z[f"{T}_se_{q}"]) for q in ("persistence", "pchip", "gp", "gp_causal")) \
        and np.array_equal(e[f"{T}_y_true"], z[f"{T}_y_true"])
    grows = np.array_equal(gz["shot"], e[f"{T}_shot"]) and np.array_equal(gz["row"], e[f"{T}_row"])
    g = gz["gates"].astype(np.float64)
    mins = (out / "check_minutes.txt").read_text() if (out / "check_minutes.txt").exists() else "?"
    put(f"| eval_b67.py, v2s with random weights, fold r2 f0 on the CPU ({mins} min): rows / same rows as the B.39 network / "
        f"reference errors and measured values identical | {len(e[f'{T}_shot']):,} / {same} / {refs} |")
    put(f"| its gate file: on the scored rows / rows found / squared errors reproduced bit for bit in the scoring process / "
        f"gates (N, 4), max abs(sum - 1) / check passes | {grows} / {gchk['found']:,} / {gchk['identical_rows']:,} of {gchk['rows']:,} / "
        f"{tuple(g.shape)}, {float(np.abs(g.sum(1) - 1).max()):.1e} / {gchk['passes']} |")
    put(f"| its gate file: share of rows whose T_i staleness in the network's input is 10 ms (rows right after a measured frame, "
        f"before excluding rows 1 to 3 after an absent frame) | {float(np.mean(np.abs(gz['ti_stale_s'] - 0.010) < 0.0005)):.3f} |")
    put()
    e2e_ok = same and refs and grows and gchk["passes"] and gchk["identical_rows"] == gchk["rows"] == gchk["found"]
    verdict("(f) gate alignment", align_ok and e2e_ok,
            "the gate readout looks rows up under eval_seq's own key: on the stored B.39 network of r2 f0 it reproduces the "
            "stored GPU errors within the B.59 tolerance (CPU against cuDNN arithmetic), and shifted by one row it fails; "
            "eval_b67.py scores v2s on the network's rows with identical reference errors, and its gate file is on those "
            "rows with every squared error reproduced bit for bit")
    put()


# ------------------------------------------------------------------------------ (g)
def check_analyze():
    put("## (g) analyze_b67.py")
    put()
    if CHECK_ANALYZE.exists():
        shutil.rmtree(CHECK_ANALYZE)
    res = {}
    for name, args in (("strict", []), ("partial", ["--partial"]), ("selftest", ["--selftest"])):
        d = CHECK_ANALYZE / name
        p = run_py([HERE / "analyze_b67.py", *args, "--out-dir", d], timeout=1800)
        j = d / "b67_gated.json"
        res[name] = {"rc": p.returncode, "written": j.exists(), "json": json.loads(j.read_text(encoding="utf-8")) if j.exists() else None,
                     "tail": (p.stdout + p.stderr).strip().splitlines()[-1:] if (p.stdout + p.stderr).strip() else []}
    put("| run | return code | output written | statements read | last line |")
    put("|---|---|---|---|---|")
    for name, v in res.items():
        stt = {k: x["read"] for k, x in v["json"]["statements"].items()} if v["json"] else "-"
        put(f"| {name} | {v['rc']} | {v['written']} | {stt} | {v['tail'][0][:160] if v['tail'] else ''} |")
    put()
    pj, sj = res["partial"]["json"], res["selftest"]["json"]
    ok = (res["strict"]["rc"] != 0 and not res["strict"]["written"]
          and res["partial"]["rc"] == 0 and pj is not None and not any(x["read"] for x in pj["statements"].values())
          and not pj["gate_readout"]["read"] and pj["reported"]["pairs"]["est_over_twin"] is not None
          and res["selftest"]["rc"] == 0 and sj is not None and all(x["read"] for x in sj["statements"].values())
          and sj["gate_readout"]["read"] and sj["selftest"])
    if sj:
        # the selftest reads the B.39 network as 'S' and the twin as 'A': S over the twin must equal the B.39 increment
        g2 = sj["statements"]["G2"]["skill"]
        put(f"Selftest: S over the twin (= the B.39 network over the twin on r2 to r4) {g2[0]:+.3f} [{g2[1]:+.3f}, {g2[2]:+.3f}]; "
            f"G1 (the network over itself) {sj['statements']['G1']['skill'][0]:+.3f}; G4 on synthetic gates "
            f"{sj['statements']['G4']['value'][0]:+.4f}.")
        put()
        ok &= abs(sj["statements"]["G1"]["skill"][0]) < 1e-12
    verdict("(g) analyze", ok,
            "without --partial it stops and writes nothing; with --partial on zero B.67 runs it exits 0 with every B.67 "
            "statement 'not read'; the selftest runs every readout (statements, rows, gate classes and differences)")
    put()


# ------------------------------------------------------------------------------ (h)
def check_git():
    put("## (h) No tracked file is changed")
    put()
    p = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    tracked = [ln for ln in lines if not ln.startswith("??")]
    mine = [ln for ln in lines if ln.startswith("??") and ("b67_gated" in ln or "b67_" in ln)]
    put(f"`git status --porcelain`: {len(lines)} lines; tracked files changed: {tracked or 'none'}; "
        f"untracked B.67 paths: {mine}.")
    put()
    verdict("(h) no tracked file changed", p.returncode == 0 and not tracked,
            "only untracked paths; the variants are registered at run time and no shared file is edited")
    put()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="redo the CPU scorings in data/.b67_check_*")
    ap.add_argument("--probe-train", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--probe-eval", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.probe_train:
        return probe_train()
    if args.probe_eval:
        return probe_eval()
    put("# B.67 checks before any GPU run")
    put()
    put(f"Registered: ces_prediction/experiments/PREREGISTRATION_B67.md (section 3). Written by "
        f"ces_prediction/experiments/b67_gated/check_b67.py on {time.strftime('%Y-%m-%d %H:%M')}; CPU only, no network "
        "trained (randomly initialised v2s / v2a weights where a checkpoint is needed). Real data: data/s*.csv.")
    put()
    check_dry()
    from seq_data import load_grid_files
    from evaluate import _load_stats
    log("grid")
    grid, dims = load_grid_files(DATA, True, 3000.0)
    est_stats = _load_stats(json.loads((B39.pooled_run_dir("v2", *FOLD) / "metrics.json").read_text(encoding="utf-8")))
    check_model(grid, dims, est_stats)
    check_registry(args.fresh)
    check_envs()
    check_alignment(grid, dims, args.fresh)
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
