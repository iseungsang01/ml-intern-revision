"""B.5 full re-score under the confirmed protocol, both populations
(PREREGISTRATION_W2.md sec. 6, B.5 -- plan committed 2026-08-15 before the run).

Every W = 4-based analysis of the draft is recomputed at W = 2 · held-free, in the
CUT population (CES_TI_SPIKE_CUT_EV=3000; B.1/B.3 artifacts reused) and the INCLUSIVE
population (cut 0; new backbone / ladder runs, window control = the frozen no-cut W = 2
sweep family re-scored additively with the current harness). Stages, GPU first:

  incl_window   re-score .wsweep_hf_w2_s* -> .b5i_w2_s* (all keys; population verified
                bit-identical vs the frozen sweep npz, se_model drift bounded)
  incl_backbone seq_v2 x4 under cut 0 (.b5i_seqv2_s*), val + TEST, paired vs .b5i_w2_s*
  val_backbone  additive val scoring of the cut backbone (seeds 1, 123)
  ladder_incl   anchor+delta and b3k8 x4 under cut 0 (.b5i_anchor_s*, .b5i_b3k8_s*)
  campaign      temporal split (65/20/15): window OFF / window ON (cut only) / seq_v2,
                4 init seeds, both populations
  cut_sens      backbone at 2.5 / 4 keV cuts (.b5c2500_seqv2_s*, .b5c4000_seqv2_s*)
  ablate        window family scored with CES_ABLATE in {no_fast, no_history} (copies)
  analyze       CPU: headline, MNAR (W = 2 in-domain), conformal, peak strata, large-gap
                pooled, coverage / PR2, ladder, campaign, cut sensitivity, ablation
                -> data/.b5_summary.json

Usage (repo root):
  py ces_prediction/experiments/b5_rescore/run_b5.py --stage incl_window --smoke
  py ces_prediction/experiments/b5_rescore/run_b5.py --stage all --resume
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
PAIRED = CES_DIR / "experiments" / "paired_model_compare.py"
MODEL_ANCHOR = CES_DIR / "experiments" / "anchor" / "model_anchor.py"
DATA = REPO_ROOT / "data"
for sub in ("b1_gate", "campaign", "mnar", "uq"):
    sys.path.insert(0, str(CES_DIR / "experiments" / sub))
sys.path.insert(0, str(CES_DIR))
from run_b1_gate import GATE_ENV, AMBIENT_VARS, SMOKE_OVERRIDES, run_step  # noqa: E402

SEEDS = (42, 1, 7, 123)
TARGETS = ("CES_TI", "CES_VT")
POPS = {
    "cut": {"cut": "3000",
            "seq": ".b1_seqv2_s{s}_i{s}", "win": ".b1_w2cut_s{s}",
            "split": ".b1_w2cut_split_s{s}",
            "b3": ".b3c_b3k8_s{s}", "anchor": ".b3_anchor_s{s}"},
    "incl": {"cut": "0",
             "seq": ".b5i_seqv2_s{s}", "win": ".b5i_w2_s{s}", "win_ref": ".wsweep_hf_w2_s{s}",
             "split": ".b1_manifest_s{s}",
             "b3": ".b5i_b3k8_s{s}", "anchor": ".b5i_anchor_s{s}"},
}
# Analysis-stage population filter (dry runs on the finished population only).
ACTIVE_POPS = tuple(p for p in POPS if p in os.getenv("CES_B5_POPS", "cut,incl").split(","))
CAMPAIGN_MANIFEST = DATA / ".campaign_split_manifest.json"
CUT_SENS = ("2500", "4000")
ABLATIONS = ("no_fast", "no_history")
B3_EPOCH_CAP = "100"


def d(pop, key, s):
    return DATA / POPS[pop][key].format(s=s)


def base_env(pop, smoke=False, cut=None):
    env = os.environ.copy()
    for var in AMBIENT_VARS + ("CES_MODEL_FILE",):
        env.pop(var, None)
    env.update(GATE_ENV)
    if smoke:
        env.update(SMOKE_OVERRIDES)
    env["CES_TI_SPIKE_CUT_EV"] = POPS[pop]["cut"] if cut is None else str(cut)
    return env


def _bootstrap_step(env, out_dir, log):
    return run_step([CES_DIR / "bootstrap_compare.py"], env, log, "bootstrap",
                    artifacts=(out_dir / "bootstrap_summary.json",))


def _paired(env, a_npz, b_npz, out_json, log, label):
    return run_step([PAIRED, "--a", a_npz, "--b", b_npz, "--out", out_json], env, log, label,
                    artifacts=(out_json,))


def _copy_window_run(src, dst):
    (dst / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "weights" / "multimodal_ces.pth", dst / "weights" / "multimodal_ces.pth")
    shutil.copy2(src / "metrics.json", dst / "metrics.json")


def _verify_population(ref_npz, new_npz, tol=0.01):
    ref, new = np.load(ref_npz), np.load(new_npz)
    report = {"bit_identical_keys": [], "se_model_rmse_drift": {}}
    for k in ref.files:
        if k.endswith("_se_model"):
            continue
        if k not in new.files:
            return False, f"key {k} missing in re-score"
        if not np.array_equal(ref[k], new[k]):
            return False, f"key {k} not bit-identical"
        report["bit_identical_keys"].append(k)
    for t in TARGETS:
        drift = abs(float(np.sqrt(ref[f"{t}_se_model"].mean())) - float(np.sqrt(new[f"{t}_se_model"].mean())))
        report["se_model_rmse_drift"][t] = drift
        if drift > tol:
            return False, f"{t}_se_model RMSE drift {drift:.4f} > {tol}"
    return True, report


# ---------------------------------------------------------------- GPU stages
def stage_incl_window(smoke, resume):
    recs = []
    for s in ((42,) if smoke else SEEDS):
        out = d("incl", "win", s) if not smoke else DATA / ".b5i_w2_s42_smoke"
        ref = d("incl", "win_ref", s)
        done = out / "bootstrap_summary.json"
        if resume and not smoke and done.exists() and (out / "rescore_verify.json").exists():
            print(f"[b5] === {out.name}: complete, skipping", flush=True); continue
        _copy_window_run(ref, out)
        env = base_env("incl", smoke)
        env.update({"CES_SEED": str(s), "CES_INIT_SEED": str(s),
                    "CES_SPLIT_DIR": str(d("incl", "split", s)), "CES_OUTPUT_DIR": str(out),
                    "CES_SPLIT_TAG": "test", "CES_PER_SHOT_NORM": "0"})
        log = out / "run.log"
        print(f"[b5] === {out.name} (re-score of {ref.name})", flush=True)
        if not run_step([CES_DIR / "compare_baselines.py"], env, log, "compare(test)",
                        artifacts=(out / "comparison_errors_test.npz",)):
            recs.append({"seed": s, "status": "compare_failed"}); continue
        if smoke:
            recs.append({"seed": s, "status": "ok"}); continue
        ok, rep = _verify_population(ref / "comparison_errors_test.npz", out / "comparison_errors_test.npz")
        (out / "rescore_verify.json").write_text(json.dumps({"ok": ok, "detail": rep}, indent=2), encoding="utf-8")
        if not ok:
            print(f"[b5]   FATAL: population mismatch vs frozen sweep npz: {rep}", flush=True)
            recs.append({"seed": s, "status": "population_mismatch"}); continue
        print(f"[b5]   population bit-identical vs {ref.name}; se_model drift {rep['se_model_rmse_drift']}", flush=True)
        if not _bootstrap_step(env, out, log):
            recs.append({"seed": s, "status": "bootstrap_failed"}); continue
        recs.append({"seed": s, "status": "ok"})
    return recs


def _seq_run(out, split_dir, control_metrics, model, cut, seed, init, smoke, resume, log,
             epochs=None, val=True, pairs=()):
    """train_seq -> eval(val) -> eval(test) -> bootstrap -> paired list."""
    env = base_env("cut", smoke, cut=cut)
    env.update({"CES_SEED": str(seed), "CES_INIT_SEED": str(init),
                "CES_SPLIT_DIR": str(split_dir), "CES_OUTPUT_DIR": str(out),
                "CES_CONTROL_METRICS": str(control_metrics), "CES_SEQ_MODEL": model,
                "CES_PER_SHOT_NORM": "1"})
    if epochs:
        env["CES_SEQ_EPOCHS"] = epochs
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40", "CES_SEQ_DEVICE": "cpu"})
    out.mkdir(parents=True, exist_ok=True)
    trained = resume and (out / "weights" / "seq_lstm.pth").exists() and (out / "metrics.json").exists()
    if trained:
        print("[b5]   train: weights exist, skipping", flush=True)
    elif not run_step([SEQ_DIR / "train_seq.py"], env, log, "train",
                      artifacts=(out / "weights" / "seq_lstm.pth", out / "metrics.json")):
        return "train_failed"
    if val and not (resume and (out / "comparison_errors_val.npz").exists()):
        env_v = dict(env); env_v["CES_SPLIT_TAG"] = "val"
        if not run_step([SEQ_DIR / "eval_seq.py"], env_v, log, "eval(val)",
                        artifacts=(out / "comparison_errors_val.npz",)):
            return "val_failed"
    env["CES_SPLIT_TAG"] = "test"
    if not (resume and (out / "comparison_errors_test.npz").exists()):
        if not run_step([SEQ_DIR / "eval_seq.py"], env, log, "eval(test)",
                        artifacts=(out / "comparison_errors_test.npz", out / "comparison_metrics.json")):
            return "compare_failed"
    if smoke:
        return "ok"
    if not _bootstrap_step(env, out, log):
        return "bootstrap_failed"
    for name, other in pairs:
        if not _paired(env, out / "comparison_errors_test.npz", other, out / name, log, name[:-5]):
            return "paired_failed"
    return "ok"


def stage_incl_backbone(smoke, resume):
    recs = []
    for s in ((42,) if smoke else SEEDS):
        out = d("incl", "seq", s) if not smoke else DATA / ".b5i_seqv2_s42_smoke"
        win = d("incl", "win", s) if not smoke else DATA / ".b5i_w2_s42_smoke"
        ref = d("incl", "win_ref", s)
        print(f"[b5] === {out.name}", flush=True)
        st = _seq_run(out, d("incl", "split", s), ref / "metrics.json", "v2", "0", s, s, smoke, resume,
                      out / "run.log", pairs=[("paired_vs_w2.json", win / "comparison_errors_test.npz")])
        recs.append({"seed": s, "status": st})
    return recs


def stage_val_backbone(smoke, resume):
    """Additive val scoring + bootstrap of the frozen cut backbone (B.1 wrote paired
    verdicts only). Val is scored with the split-tagged report policy, so the TEST
    report is never touched."""
    recs = []
    for s in SEEDS:
        out = d("cut", "seq", s)
        env = base_env("cut")
        env.update({"CES_SEED": str(s), "CES_INIT_SEED": str(s), "CES_SPLIT_DIR": str(d("cut", "split", s)),
                    "CES_OUTPUT_DIR": str(out), "CES_CONTROL_METRICS": str(d("cut", "win", s) / "metrics.json"),
                    "CES_SEQ_MODEL": "v2", "CES_PER_SHOT_NORM": "1", "CES_SPLIT_TAG": "val"})
        ok = True
        if not (out / "comparison_errors_val.npz").exists():
            print(f"[b5] === val scoring {out.name}", flush=True)
            ok = run_step([SEQ_DIR / "eval_seq.py"], env, out / "run.log", "eval(val)",
                          artifacts=(out / "comparison_errors_val.npz",))
        if ok and not (out / "bootstrap_summary.json").exists():
            ok = _bootstrap_step(env, out, out / "run.log")
        recs.append({"seed": s, "status": "ok" if ok else "val_failed"})
    return recs


def stage_ladder_incl(smoke, resume):
    recs = []
    seeds = (42,) if smoke else SEEDS
    for s in seeds:
        # anchor+delta under cut 0 (window family rules: per-shot OFF)
        out = d("incl", "anchor", s) if not smoke else DATA / ".b5i_anchor_s42_smoke"
        split = DATA / (f".b5i_anchor_split_s{s}" + ("_smoke" if smoke else ""))
        log = out / "run.log"
        if not (resume and (out / "bootstrap_summary.json").exists()):
            out.mkdir(parents=True, exist_ok=True)
            env = base_env("incl", smoke)
            env.update({"CES_SEED": str(s), "CES_INIT_SEED": str(s), "CES_SPLIT_DIR": str(split),
                        "CES_OUTPUT_DIR": str(out), "CES_SPLIT_TAG": "test",
                        "CES_FILE_SPLIT_FROM": str(d("incl", "split", s) / "split_manifest.json"),
                        "CES_PER_SHOT_NORM": "0", "CES_MODEL_FILE": str(MODEL_ANCHOR)})
            print(f"[b5] === {out.name}", flush=True)
            if split.exists():
                shutil.rmtree(split)
            split.mkdir(parents=True)
            if not run_step([CES_DIR / "train.py"], env, log, "train",
                            artifacts=(out / "metrics.json", out / "weights" / "multimodal_ces.pth")):
                recs.append({"seed": s, "arm": "anchor", "status": "train_failed"}); continue
            if not smoke:
                got = json.loads((split / "split_manifest.json").read_text(encoding="utf-8"))
                want = json.loads((d("incl", "split", s) / "split_manifest.json").read_text(encoding="utf-8"))
                if sorted(got["test_files"]) != sorted(want["test_files"]):
                    recs.append({"seed": s, "arm": "anchor", "status": "test_isolation_failed"}); continue
                env["CES_SPLIT_DIR"] = str(d("incl", "split", s))
            if not run_step([CES_DIR / "compare_baselines.py"], env, log, "compare(test)",
                            artifacts=(out / "comparison_errors_test.npz",)):
                recs.append({"seed": s, "arm": "anchor", "status": "compare_failed"}); continue
            if not smoke and not _bootstrap_step(env, out, log):
                recs.append({"seed": s, "arm": "anchor", "status": "bootstrap_failed"}); continue
        recs.append({"seed": s, "arm": "anchor", "status": "ok"})
        if smoke:
            continue
        # b3k8 under cut 0
        out_b = d("incl", "b3", s)
        print(f"[b5] === {out_b.name}", flush=True)
        st = _seq_run(out_b, d("incl", "split", s), d("incl", "win_ref", s) / "metrics.json", "b3k8", "0",
                      s, s, False, resume, out_b / "run.log", epochs=B3_EPOCH_CAP, val=False,
                      pairs=[("paired_vs_anchor.json", out / "comparison_errors_test.npz"),
                             ("paired_vs_seqv2.json", d("incl", "seq", s) / "comparison_errors_test.npz"),
                             ("paired_vs_w2.json", d("incl", "win", s) / "comparison_errors_test.npz")])
        recs.append({"seed": s, "arm": "b3k8", "status": st})
    return recs


def stage_campaign(smoke, resume):
    from run_campaign_shift import build_manifest  # noqa: E402  (writes CAMPAIGN_MANIFEST)
    build_manifest()
    recs = []
    inits = (42,) if smoke else SEEDS
    for pop in ("cut", "incl"):
        for init in inits:
            sfx = "_smoke" if smoke else ""
            win = DATA / f".b5_camp_{pop}_win_s{init}{sfx}"
            split = DATA / f".b5_camp_{pop}_split_s{init}{sfx}"
            log = win / "run.log"
            # window OFF (the pairing control)
            if not (resume and (win / "bootstrap_summary.json").exists()):
                win.mkdir(parents=True, exist_ok=True)
                env = base_env(pop, smoke)
                env.update({"CES_SEED": str(init), "CES_INIT_SEED": str(init), "CES_SPLIT_DIR": str(split),
                            "CES_OUTPUT_DIR": str(win), "CES_SPLIT_TAG": "test",
                            "CES_FILE_SPLIT_FROM": str(CAMPAIGN_MANIFEST), "CES_PER_SHOT_NORM": "0"})
                print(f"[b5] === {win.name}", flush=True)
                if split.exists():
                    shutil.rmtree(split)
                split.mkdir(parents=True)
                if not run_step([CES_DIR / "train.py"], env, log, "train",
                                artifacts=(win / "metrics.json", win / "weights" / "multimodal_ces.pth")):
                    recs.append({"pop": pop, "init": init, "arm": "win", "status": "train_failed"}); continue
                if not run_step([CES_DIR / "compare_baselines.py"], env, log, "compare(test)",
                                artifacts=(win / "comparison_errors_test.npz",)):
                    recs.append({"pop": pop, "init": init, "arm": "win", "status": "compare_failed"}); continue
                if not smoke and not _bootstrap_step(env, win, log):
                    recs.append({"pop": pop, "init": init, "arm": "win", "status": "bootstrap_failed"}); continue
            recs.append({"pop": pop, "init": init, "arm": "win", "status": "ok"})
            if smoke:
                continue
            # window ON (cut population only -- the sec. 8s per-shot re-check at W = 2)
            if pop == "cut":
                ps = DATA / f".b5_camp_{pop}_winps_s{init}"
                if not (resume and (ps / "paired_vs_win.json").exists()):
                    ps.mkdir(parents=True, exist_ok=True)
                    split_ps = DATA / f".b5_camp_{pop}_split_ps_s{init}"
                    env = base_env(pop)
                    env.update({"CES_SEED": str(init), "CES_INIT_SEED": str(init), "CES_SPLIT_DIR": str(split_ps),
                                "CES_OUTPUT_DIR": str(ps), "CES_SPLIT_TAG": "test",
                                "CES_FILE_SPLIT_FROM": str(CAMPAIGN_MANIFEST), "CES_PER_SHOT_NORM": "1"})
                    print(f"[b5] === {ps.name}", flush=True)
                    if split_ps.exists():
                        shutil.rmtree(split_ps)
                    split_ps.mkdir(parents=True)
                    lg = ps / "run.log"
                    ok = (run_step([CES_DIR / "train.py"], env, lg, "train",
                                   artifacts=(ps / "metrics.json", ps / "weights" / "multimodal_ces.pth"))
                          and run_step([CES_DIR / "compare_baselines.py"], dict(env, CES_SPLIT_DIR=str(split)),
                                       lg, "compare(test)", artifacts=(ps / "comparison_errors_test.npz",))
                          and _bootstrap_step(env, ps, lg)
                          and _paired(env, ps / "comparison_errors_test.npz", win / "comparison_errors_test.npz",
                                      ps / "paired_vs_win.json", lg, "paired-vs-win"))
                    recs.append({"pop": pop, "init": init, "arm": "winps", "status": "ok" if ok else "failed"})
            # seq_v2 backbone on the temporal split
            seq = DATA / f".b5_camp_{pop}_seq_s{init}"
            print(f"[b5] === {seq.name}", flush=True)
            st = _seq_run(seq, split, win / "metrics.json", "v2", POPS[pop]["cut"], init, init, False, resume,
                          seq / "run.log", val=False,
                          pairs=[("paired_vs_win.json", win / "comparison_errors_test.npz")])
            recs.append({"pop": pop, "init": init, "arm": "seq", "status": st})
    return recs


def stage_cut_sens(smoke, resume):
    recs = []
    for cut in CUT_SENS:
        for s in ((42,) if smoke else SEEDS):
            out = DATA / (f".b5c{cut}_seqv2_s{s}" + ("_smoke" if smoke else ""))
            print(f"[b5] === {out.name}", flush=True)
            # self-population scoring: the run's own stats serve as the harness stats
            out.mkdir(parents=True, exist_ok=True)
            st = _seq_run(out, d("cut", "split", s), out / "metrics.json", "v2", cut, s, s, smoke, resume,
                          out / "run.log", val=False)
            recs.append({"cut": cut, "seed": s, "status": st})
    return recs


def stage_ablate(smoke, resume):
    recs = []
    for pop in ("cut", "incl"):
        for abl in ABLATIONS:
            for s in ((42,) if smoke else SEEDS):
                out = DATA / (f".b5_abl_{pop}_{abl}_s{s}" + ("_smoke" if smoke else ""))
                src = d(pop, "win", s)
                if resume and not smoke and (out / "paired_vs_base.json").exists():
                    continue
                if not (src / "weights" / "multimodal_ces.pth").exists():
                    recs.append({"pop": pop, "abl": abl, "seed": s, "status": "source_missing"}); continue
                _copy_window_run(src, out)
                env = base_env(pop, smoke)
                env.update({"CES_SEED": str(s), "CES_INIT_SEED": str(s), "CES_SPLIT_DIR": str(d(pop, "split", s)),
                            "CES_OUTPUT_DIR": str(out), "CES_SPLIT_TAG": "test", "CES_PER_SHOT_NORM": "0",
                            "CES_ABLATE": abl})
                log = out / "run.log"
                print(f"[b5] === {out.name}", flush=True)
                ok = run_step([CES_DIR / "compare_baselines.py"], env, log, f"compare(test,{abl})",
                              artifacts=(out / "comparison_errors_test.npz",))
                if ok and not smoke:
                    ok = (_bootstrap_step(env, out, log)
                          and _paired(env, out / "comparison_errors_test.npz", src / "comparison_errors_test.npz",
                                      out / "paired_vs_base.json", log, "paired-vs-base"))
                recs.append({"pop": pop, "abl": abl, "seed": s, "status": "ok" if ok else "failed"})
    return recs


# ---------------------------------------------------------------- CPU analysis
def _load_report(run_dir):
    p = run_dir / "comparison_metrics_test.json"
    if not p.exists():
        p = run_dir / "comparison_metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _gates(run_dir):
    return json.loads((run_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))["splits"]["test"]


def _paired_json(path):
    return json.loads(path.read_text(encoding="utf-8"))["targets"]


def _skill_from_npz(z, t, base="pchip"):
    return float(1.0 - z[f"{t}_se_model"].mean() / z[f"{t}_se_{base}"].mean())


def analyze_headline():
    out = {}
    for pop in ACTIVE_POPS:
        rows = {}
        for arm in ("seq", "win"):
            per = {}
            for s in SEEDS:
                rd = d(pop, arm, s)
                g = _gates(rd)
                per[s] = {t: {b: {"skill": g[t][b]["skill_point"], "ci": g[t][b]["skill_ci95"], "pass": g[t][b]["pass"]}
                              for b in ("pchip", "persistence", "gp", "gp_causal") if b in g[t]}
                          for t in TARGETS}
            rows[arm] = per
            rows[f"{arm}_pr4_pass"] = {t: {b: sum(int(per[s][t][b]["pass"]) for s in SEEDS if b in per[s][t])
                                           for b in ("pchip", "persistence", "gp_causal")} for t in TARGETS}
        pair_name = "paired_vs_w2cut.json" if pop == "cut" else "paired_vs_w2.json"
        rows["seq_vs_win_paired"] = {s: {t: {"skill": _paired_json(d(pop, "seq", s) / pair_name)[t]["skill_point"],
                                             "a_better": _paired_json(d(pop, "seq", s) / pair_name)[t]["a_better"],
                                             "b_better": _paired_json(d(pop, "seq", s) / pair_name)[t]["b_better"]}
                                         for t in TARGETS} for s in SEEDS}
        out[pop] = rows
    return out


def analyze_mnar():
    import analyze_mnar as M  # noqa: E402
    import pandas as pd
    from dataset import KSTAR_CES_Dataset, TIME_COLUMN, TARGET_COLUMNS, STUCK_GAP_SECONDS  # noqa: E402

    def load_grid_pop(path, cut_ev):
        df = pd.read_csv(path)
        cols = list(df.columns)
        bes = [c for c in cols if c.startswith("BES")]
        ecei = [c for c in cols if c.startswith("ECEI")]
        mc = [c for c in cols if c.startswith("MC")]
        df = df.dropna(subset=[TIME_COLUMN, *bes, *ecei, *mc]).reset_index(drop=True)
        if df.empty:
            return None
        if cut_ev > 0:
            KSTAR_CES_Dataset._nan_out_ti_spikes(df, cut_ev)
        time = df[TIME_COLUMN].to_numpy(dtype=np.float64)
        if len(time) >= 2:
            block = np.concatenate(([0], np.cumsum(np.diff(time) >= STUCK_GAP_SECONDS)))
            for col in TARGET_COLUMNS:
                v = df[col].to_numpy(dtype=np.float64, copy=True)
                v[KSTAR_CES_Dataset._stuck_repeat_mask(v, block)] = np.nan
                df[col] = v
        arr = df.loc[:, [TIME_COLUMN, *TARGET_COLUMNS, *bes, *ecei, *mc]].to_numpy(dtype=np.float64)
        slices = {"time": 0, "CES_TI": 1, "CES_VT": 2,
                  "bes": list(range(3, 3 + len(bes))),
                  "ecei": list(range(3 + len(bes), 3 + len(bes) + len(ecei)))}
        return arr, slices

    M.WINDOW = 2
    M.NPZ_SUFFIX = ""
    out = {"_design": {"window": 2, "in_domain": "nearest past observation within 2 rows (W = 2)",
                       "strata": "dt bin (15/25/45 ms) x input-only activity flag",
                       "min_scored_per_stratum": M.MIN_SCORED_PER_STRATUM,
                       "target_distribution": "genuinely-missing rows in-domain (mis_dom); coverage = mass of surviving strata"}}
    for pop in ACTIVE_POPS:
        cut = float(POPS[pop]["cut"])
        M.load_grid = lambda p, _c=cut: load_grid_pop(p, _c)
        M.SPLIT_SRC = {s: d(pop, "split", s) for s in SEEDS}
        weights = {}
        for s in SEEDS:
            for t in TARGETS:
                weights[(s, t)] = M.weights_from_grid(s, t)
        out[pop] = {}
        for arm in ("seq", "win"):
            M.RUN_DIR = {s: d(pop, arm, s) for s in SEEDS}
            M._NPZ_CACHE.clear()
            res = {}
            for t in TARGETS:
                per_seed = []
                for s in SEEDS:
                    w = weights[(s, t)]
                    scored = M.scored_strata(s, t)
                    row = {"seed": s, "n_missing_any": w["n_missing_any"], "n_missing_dom": w["n_missing_dom"],
                           "in_domain_fraction": (w["n_missing_dom"] / w["n_missing_any"]) if w["n_missing_any"] else None}
                    for base in ("pchip", "persistence"):
                        sk, cov = M.reweighted_skill(scored, w["mis_dom"], base)
                        ci = M.bootstrap_reweighted(s, t, w["mis_dom"], base)
                        row[base] = {"reweighted_skill": sk, "coverage": cov,
                                     "ci95": ci["ci95"] if ci else None, "pass": ci["pass"] if ci else None,
                                     "unweighted_skill": _skill_from_npz(M._npz(s), t, base)}
                    per_seed.append(row)
                res[t] = per_seed
            out[pop][arm] = res
    return out


def analyze_conformal():
    import analyze_conformal as C  # noqa: E402
    out = {"_design": {"alpha": C.ALPHA, "cal": "val split of the same run", "eval": "TEST",
                       "modes": ["global", "mondrian"], "arms": list(C.ARMS)}}
    for pop in ACTIVE_POPS:
        C.RUN_DIR = {s: d(pop, "seq", s) for s in SEEDS}
        out[pop] = {}
        for t in TARGETS:
            per = {}
            for s in SEEDS:
                cal = C.load(s, "comparison_errors_val.npz", t)
                test = C.load(s, "comparison_errors_test.npz", t)
                per[s] = {mode: {arm: C.evaluate(cal, test, arm, mode) for arm in C.ARMS} for mode in ("global", "mondrian")}
            wins = {mode: sum(1 for s in SEEDS if per[s][mode]["model"] and per[s][mode]["pchip"]
                              and per[s][mode]["model"]["winkler_mean"] < per[s][mode]["pchip"]["winkler_mean"])
                    for mode in ("global", "mondrian")}
            wins_p = {mode: sum(1 for s in SEEDS if per[s][mode]["model"] and per[s][mode]["persistence"]
                                and per[s][mode]["model"]["winkler_mean"] < per[s][mode]["persistence"]["winkler_mean"])
                      for mode in ("global", "mondrian")}
            out[pop][t] = {"per_seed": per, "winkler_wins_vs_pchip": wins, "winkler_wins_vs_persistence": wins_p}
    return out


def analyze_peak_and_gap():
    from bootstrap_compare import _bootstrap, BOOTSTRAP_SEED  # noqa: E402
    out = {"peak": {}, "largegap": {}}
    for pop in ACTIVE_POPS:
        out["peak"][pop] = {}
        out["largegap"][pop] = {}
        for arm in ("seq", "win"):
            peak = {t: {"peak": [], "non_peak": []} for t in TARGETS}
            pooled = {t: {k: [] for k in ("shot", "dt", "model", "pchip", "persistence")} for t in TARGETS}
            for s in SEEDS:
                z = np.load(d(pop, arm, s) / "comparison_errors_test.npz")
                for t in TARGETS:
                    ip = z[f"{t}_is_peak"].astype(bool)
                    for label, m in (("peak", ip), ("non_peak", ~ip)):
                        rng = np.random.default_rng(BOOTSTRAP_SEED)
                        entry = {"seed": s, "n": int(m.sum())}
                        if m.sum() > 50:
                            for base in ("pchip", "persistence"):
                                r = _bootstrap(z[f"{t}_shot"][m], z[f"{t}_se_model"][m], z[f"{t}_se_{base}"][m], rng)
                                entry[base] = {"skill": r["skill_point"], "ci": r["skill_ci95"], "pass": r["pass"]}
                        peak[t][label].append(entry)
                    pooled[t]["shot"].append(z[f"{t}_shot"]); pooled[t]["dt"].append(z[f"{t}_dt_ms"])
                    pooled[t]["model"].append(z[f"{t}_se_model"]); pooled[t]["pchip"].append(z[f"{t}_se_pchip"])
                    pooled[t]["persistence"].append(z[f"{t}_se_persistence"])
            out["peak"][pop][arm] = peak
            gap = {}
            for t in TARGETS:
                P = {k: np.concatenate(v) for k, v in pooled[t].items()}
                gap[t] = {}
                for label, (lo, hi) in {"dt <= 15 ms": (0.0, 15.0), "dt > 15 ms": (15.0, np.inf),
                                        "dt > 45 ms": (45.0, np.inf)}.items():
                    m = (P["dt"] > lo) & (P["dt"] <= hi)
                    e = {"n": int(m.sum()), "n_shots": int(len(np.unique(P["shot"][m])))}
                    if m.sum() > 50:
                        rng = np.random.default_rng(BOOTSTRAP_SEED)
                        for base in ("pchip", "persistence"):
                            r = _bootstrap(P["shot"][m], P["model"][m], P[base][m], rng)
                            e[base] = {"skill": r["skill_point"], "ci": r["skill_ci95"], "pass": r["pass"]}
                    gap[t][label] = e
            out["largegap"][pop][arm] = gap
    return out


def analyze_coverage():
    out = {}
    for pop in ACTIVE_POPS:
        out[pop] = {}
        for arm in ("seq", "win"):
            per = {}
            for s in SEEDS:
                z = np.load(d(pop, arm, s) / "comparison_errors_test.npz")
                rep = _load_report(d(pop, arm, s))
                per[s] = {t: {"n_scored": int(len(z[f"{t}_shot"])),
                              "n_dt_gt_15ms": int((z[f"{t}_dt_ms"] > 15.0).sum()),
                              "n_dt_gt_45ms": int((z[f"{t}_dt_ms"] > 45.0).sum()),
                              "future_neighbor_fraction": rep["per_target"][t]["future_neighbor_fraction"],
                              "pr2_persistence_fallback_rate": 1.0 - rep["per_target"][t]["future_neighbor_fraction"]}
                          for t in TARGETS}
            out[pop][arm] = per
    return out


def analyze_ladder():
    out = {}
    for pop in ACTIVE_POPS:
        rows = {}
        for s in SEEDS:
            r = {}
            for arm in ("anchor", "b3", "win", "seq"):
                z = np.load(d(pop, arm, s) / "comparison_errors_test.npz")
                r[arm] = {t: _skill_from_npz(z, t) for t in TARGETS}
                if arm == "anchor":
                    r["persistence"] = {t: float(1.0 - z[f"{t}_se_persistence"].mean() / z[f"{t}_se_pchip"].mean()) for t in TARGETS}
            b3 = d(pop, "b3", s)
            pa = _paired_json(b3 / "paired_vs_anchor.json"); pb = _paired_json(b3 / "paired_vs_seqv2.json")
            r["b3_vs_anchor"] = {t: {"skill": pa[t]["skill_point"], "a_better": pa[t]["a_better"], "b_better": pa[t]["b_better"]} for t in TARGETS}
            r["b3_vs_seq"] = {t: {"skill": pb[t]["skill_point"], "ci": pb[t]["skill_ci95"]} for t in TARGETS}
            rows[s] = r
        win4 = sum(int(rows[s]["b3_vs_anchor"]["CES_TI"]["a_better"]) for s in SEEDS)
        vtdef = sum(int(rows[s]["b3_vs_anchor"]["CES_VT"]["b_better"]) for s in SEEDS)
        meanb = float(np.mean([rows[s]["b3_vs_seq"]["CES_TI"]["skill"] for s in SEEDS]))
        out[pop] = {"per_seed": rows,
                    "verdict": {"b3_vs_anchor_TI_sig_win": f"{win4}/4", "b3_vs_anchor_VT_sig_deficit": f"{vtdef}/4",
                                "mean_b3_vs_seq_TI": meanb, "interpretable_rung": bool(win4 == 4 and vtdef == 0),
                                "within_backbone_tolerance": bool(meanb >= -0.05)}}
    return out


def analyze_campaign():
    out = {}
    for pop in ACTIVE_POPS:
        rows = {}
        for init in SEEDS:
            win = DATA / f".b5_camp_{pop}_win_s{init}"
            seq = DATA / f".b5_camp_{pop}_seq_s{init}"
            r = {}
            for arm, rd in (("win", win), ("seq", seq)):
                g = _gates(rd)
                r[arm] = {t: {b: {"skill": g[t][b]["skill_point"], "pass": g[t][b]["pass"]}
                              for b in ("pchip", "persistence", "gp_causal") if b in g[t]} for t in TARGETS}
            pj = _paired_json(seq / "paired_vs_win.json")
            r["seq_vs_win"] = {t: {"skill": pj[t]["skill_point"], "a_better": pj[t]["a_better"], "b_better": pj[t]["b_better"]} for t in TARGETS}
            if pop == "cut":
                ps = DATA / f".b5_camp_{pop}_winps_s{init}"
                g = _gates(ps); pp = _paired_json(ps / "paired_vs_win.json")
                r["winps"] = {t: {b: {"skill": g[t][b]["skill_point"], "pass": g[t][b]["pass"]}
                                  for b in ("pchip", "persistence") if b in g[t]} for t in TARGETS}
                r["winps_vs_win"] = {t: {"skill": pp[t]["skill_point"], "a_better": pp[t]["a_better"], "b_better": pp[t]["b_better"]} for t in TARGETS}
            rows[init] = r
        summ = {}
        for arm in ("win", "seq") + (("winps",) if pop == "cut" else ()):
            summ[arm] = {t: {b: sum(int(rows[i][arm][t][b]["pass"]) for i in SEEDS if b in rows[i][arm][t])
                             for b in ("pchip", "persistence")} for t in TARGETS}
        summ["seq_vs_win_TI_positive"] = sum(int(rows[i]["seq_vs_win"]["CES_TI"]["skill"] > 0) for i in SEEDS)
        summ["seq_vs_win_TI_sig"] = sum(int(rows[i]["seq_vs_win"]["CES_TI"]["a_better"]) for i in SEEDS)
        out[pop] = {"per_init": rows, "pr4_pass_counts": summ,
                    "manifest": json.loads(CAMPAIGN_MANIFEST.read_text(encoding="utf-8"))["shot_ranges"]}
    return out


def analyze_cut_sens():
    out = {}
    for cut in ("3000",) + CUT_SENS:
        per = {}
        for s in SEEDS:
            rd = d("cut", "seq", s) if cut == "3000" else DATA / f".b5c{cut}_seqv2_s{s}"
            g = _gates(rd)
            per[s] = {t: {b: {"skill": g[t][b]["skill_point"], "pass": g[t][b]["pass"]}
                          for b in ("pchip", "persistence", "gp_causal") if b in g[t]} for t in TARGETS}
        out[cut] = {"per_seed": per,
                    "pass_counts": {t: {b: sum(int(per[s][t][b]["pass"]) for s in SEEDS if b in per[s][t])
                                        for b in ("pchip", "persistence", "gp_causal")} for t in TARGETS}}
    return out


def analyze_ablation():
    out = {}
    for pop in ACTIVE_POPS:
        out[pop] = {}
        for abl in ABLATIONS:
            per = {}
            for s in SEEDS:
                rd = DATA / f".b5_abl_{pop}_{abl}_s{s}"
                z = np.load(rd / "comparison_errors_test.npz")
                pj = _paired_json(rd / "paired_vs_base.json")
                per[s] = {t: {"skill_vs_pchip": _skill_from_npz(z, t),
                              "skill_vs_persistence": _skill_from_npz(z, t, "persistence"),
                              "paired_vs_base": pj[t]["skill_point"], "sig_worse": pj[t]["b_better"]}
                          for t in TARGETS}
            out[pop][abl] = per
    return out


def stage_analyze():
    summary = {"protocol": {"W": 2, "held": "free", "populations": {p: {"CES_TI_SPIKE_CUT_EV": POPS[p]["cut"]} for p in ACTIVE_POPS}}}
    for name, fn in (("headline", analyze_headline), ("coverage_pr2", analyze_coverage), ("ladder", analyze_ladder), ("conformal", analyze_conformal),
                     ("peak_and_largegap", analyze_peak_and_gap), ("campaign", analyze_campaign),
                     ("cut_sensitivity", analyze_cut_sens), ("ablation", analyze_ablation), ("mnar", analyze_mnar)):
        try:
            summary[name] = fn()
            print(f"[b5] analysis {name}: ok", flush=True)
        except Exception as exc:  # noqa: BLE001 -- report partial results, never hide which block failed
            summary[name] = {"error": repr(exc)}
            print(f"[b5] analysis {name}: FAILED {exc!r}", flush=True)
    out_path = DATA / ".b5_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"[b5] summary saved {out_path}", flush=True)
    return summary


STAGES = {"incl_window": stage_incl_window, "incl_backbone": stage_incl_backbone,
          "val_backbone": stage_val_backbone, "ladder_incl": stage_ladder_incl,
          "campaign": stage_campaign, "cut_sens": stage_cut_sens, "ablate": stage_ablate}
GPU_ORDER = ("incl_window", "incl_backbone", "val_backbone", "ladder_incl", "campaign", "cut_sens", "ablate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["analyze", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    prereg = (CES_DIR / "experiments" / "PREREGISTRATION_W2.md").read_text(encoding="utf-8")
    if "B.5 실행 규칙" not in prereg:
        raise SystemExit("FATAL: B.5 plan not committed to PREREGISTRATION_W2.md.")

    start = time.time()
    stages = GPU_ORDER if args.stage == "all" else ((args.stage,) if args.stage != "analyze" else ())
    bad = []
    for st in stages:
        print(f"[b5] ##### stage {st}", flush=True)
        recs = STAGES[st](args.smoke, args.resume)
        bad += [dict(r, stage=st) for r in recs if r["status"] != "ok"]
        print(f"[b5] stage {st} done ({(time.time() - start) / 60:.0f} min elapsed)", flush=True)
    for r in bad:
        print(f"[b5] FAILED: {r}", flush=True)
    if args.smoke:
        sys.exit(1 if bad else 0)
    if bad:
        sys.exit(1)
    if args.stage in ("analyze", "all"):
        stage_analyze()
    sys.exit(0)


if __name__ == "__main__":
    main()
