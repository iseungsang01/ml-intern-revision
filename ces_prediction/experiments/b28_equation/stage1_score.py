"""B.28 stage 1 (PREREGISTRATION_B28.md sec. 3): score the two-term equation on the FROZEN rows.

No training, CPU only. For each frozen backbone run (cut: `.b1_seqv2_s{s}_i{s}`, the headline
runs; the registration's `_i42` was a naming slip, see PREREGISTRATION_B28.md sec. 6; inclusive:
`.b5i_seqv2_s{s}`) the exact TEST row population of its `comparison_errors_test.npz` is
re-derived through the same harness call (`build_clean_val_subset` + the keep / baseline-NaN
mask of eval_seq.py) and verified against the npz (shot ids, y_true, dt_ms must match). On
those rows three arms are scored against the npz's own reference errors (PCHIP, persistence,
causal GP, GP) and against the backbone:

  (a) EWMA of the past observed T_i, fixed alpha in {0.25, 0.35, 0.45}, NO fitted parameter
      -- alpha* chosen on the split's VAL files (never on TEST);
  (b) EWMA(alpha*) + a linear term in the 15 fast-channel 10 ms increments, ridge-fitted on
      the VAL files' observed rows (16 coefficients);
  (c) the backbone (se_model from the npz).

Skill = 1 - sum SE_arm / sum SE_ref, shot-cluster bootstrap CI (B = 10000, seed 12345, the
bootstrap_compare.py resampling). Written to data/.b28_stage1.json. Console output is ASCII.

Run from the repo root:  py ces_prediction/experiments/b28_equation/stage1_score.py [--smoke]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR / "experiments" / "b1_gate"))
sys.path.insert(0, str(SEQ_DIR))
sys.path.insert(0, str(CES_DIR))
from run_b1_gate import GATE_ENV, AMBIENT_VARS  # noqa: E402

SEEDS = (42, 1, 7, 123)
POPS = {
    "cut": {"cut": "3000", "run": ".b1_seqv2_s{s}_i{s}", "control_metrics": ".b1_w2cut_s{s}",
            "split": ".b1_w2cut_split_s{s}"},
    "incl": {"cut": "0", "run": ".b5i_seqv2_s{s}", "control_metrics": ".wsweep_hf_w2_s{s}",
             "split": ".b1_manifest_s{s}"},
}
ALPHAS = (0.25, 0.35, 0.45)
REFS = ("pchip", "persistence", "gp_causal", "gp")
B_BOOT = 10000
SEED_BOOT = 12345
RIDGE = 1e-2
OUT = DATA / ".b28_stage1.json"
N_BES, N_ECEI, N_FAST = 9, 4, 15


def pin_env(cut):
    for var in AMBIENT_VARS:
        os.environ.pop(var, None)
    os.environ.update(GATE_ENV)
    os.environ["CES_TI_SPIKE_CUT_EV"] = cut
    os.environ["CES_PER_SHOT_NORM"] = "1"
    os.environ["CES_DROP_STUCK_TARGETS"] = "1"


# ---------------------------------------------------------------- frozen rows
def frozen_rows(control_metrics, manifest, seed, npz):
    """Re-derive the npz's CES_TI row population through the harness; verify; return rows."""
    from evaluate import _load_stats, _persistence_from_history, build_clean_val_subset  # noqa: E402
    import baselines_interpolation as B  # noqa: E402
    stats = _load_stats(json.loads(control_metrics.read_text(encoding="utf-8")))
    test_files = set(json.loads(manifest.read_text(encoding="utf-8"))["test_files"])
    max_rows = int(os.environ["CES_MAX_VAL_SAMPLES"])          # eval_seq reads this one
    dataset, idxs, _ = build_clean_val_subset(DATA, 2, stats, test_files, max_rows, seed)
    time_col = int(dataset._column_slices["time"])
    tc = int(dataset._column_slices["target"][0])
    t_mean = torch.as_tensor(stats["target"]["mean"], dtype=torch.float32)
    t_std = torch.as_tensor(stats["target"]["std"], dtype=torch.float32)
    shot, fname, row, tkey, y, dt = [], [], [], [], [], []
    for i in idxs:
        s = dataset[i]
        if float(s["target_mask"][0]) <= 0.5:
            continue
        _, has = _persistence_from_history(s["ces_history"][None])
        if not bool(has[0, 0]):
            continue
        fi = int(dataset.sample_file_indices[i])
        fa = dataset.file_arrays[fi]
        r = int(s["row_index"])
        times, values, tt = B.build_neighbor_set(fa, time_col, tc, r)
        preds = [B.PREDICTORS[m](times, values, tt) for m in ("persistence", "linear", "pchip", "ar_local")]
        if any(np.isnan(p) for p in preds):
            continue
        past = times < tt
        shot.append(fi)
        fname.append(Path(dataset.valid_files[fi]).name)
        row.append(r)
        tkey.append(float(fa[r, time_col]))
        y.append(float((s["target"] * t_std + t_mean)[0]))
        dt.append((tt - float(times[past].max())) * 1000.0)
    shot = np.asarray(shot); y = np.asarray(y, dtype=np.float32); dt = np.asarray(dt)
    n_npz = len(npz["CES_TI_shot"])
    if len(shot) != n_npz:
        raise SystemExit(f"FATAL: re-derived {len(shot)} rows, npz has {n_npz}")
    if not np.array_equal(shot, npz["CES_TI_shot"]):
        raise SystemExit("FATAL: shot ids differ from the npz")
    # The cut s1/s7/s123 npz files (eval_seq of 2026-08-14) predate the `_y_true` key; for them
    # the row identity is checked on shot ids and dt_ms alone (both exact) and y_true is unverified.
    has_y = "CES_TI_y_true" in npz.files
    dy = float(np.max(np.abs(y - npz["CES_TI_y_true"]))) if has_y else float("nan")
    ddt = float(np.max(np.abs(dt - npz["CES_TI_dt_ms"])))
    if (has_y and dy > 1e-2) or ddt > 1e-6:
        raise SystemExit(f"FATAL: y_true / dt_ms differ from the npz (max {dy:.3g} eV, {ddt:.3g} ms)")
    return {"shot": shot, "file": fname, "row": np.asarray(row), "time": np.asarray(tkey),
            "y": y, "dt_ms": dt, "check": {"n": int(n_npz), "max_dy_eV": dy, "max_ddt_ms": ddt,
                                            "y_bit_identical": bool(has_y and dy == 0.0),
                                            "y_true_in_npz": bool(has_y)}}


# ---------------------------------------------------------------- features
def ewma_table(arr, dims, my_stats, cut):
    """One shot -> {time -> feature row}: EWMA_alpha (eV) of the past observed T_i per alpha,
    carried (eV), the 15 per-shot-z increments, |dECEI| mean, stale [s], has, y (eV), obs."""
    from seq_data import build_blocks  # noqa: E402
    mean, std = float(my_stats["target"]["mean"][0]), float(my_stats["target"]["std"][0])
    table = {}
    for b in build_blocks(arr, dims, my_stats, per_shot_norm=True):
        x, mask = b["x"], b["mask"][:, 0]
        L = x.shape[0]
        fast = x[:, :N_FAST]
        d = np.zeros_like(fast); d[1:] = fast[1:] - fast[:-1]
        e = [None] * len(ALPHAS)
        for i in range(L):
            cur = [0.0 if v is None else v for v in e]
            yphys = b["y"][i, 0] * std + mean if mask[i] > 0 else np.nan
            table[float(b["time"][i])] = {
                "ewma": cur, "carried": x[i, 16] * std + mean, "has": x[i, 18] > 0.5,
                "d": d[i], "dE": float(np.abs(d[i, N_BES:N_BES + N_ECEI]).mean()),
                "stale_s": float(np.expm1(x[i, 17])), "y": yphys, "obs": mask[i] > 0,
            }
            if mask[i] > 0:
                v = b["y"][i, 0] * std + mean
                for j, a in enumerate(ALPHAS):
                    e[j] = v if e[j] is None else a * v + (1 - a) * e[j]
    return table


def stack(tables, keys):
    """keys: list of (file, time) -> arrays."""
    E = np.zeros((len(keys), len(ALPHAS))); P = np.zeros(len(keys)); D = np.zeros((len(keys), N_FAST))
    dE = np.zeros(len(keys)); Y = np.zeros(len(keys)); S = np.zeros(len(keys))
    for n, (f, t) in enumerate(keys):
        r = tables[f][t]
        E[n] = r["ewma"]; P[n] = r["carried"]; D[n] = r["d"]; dE[n] = r["dE"]; Y[n] = r["y"]; S[n] = r["stale_s"]
    return E, P, D, dE, Y, S


# ---------------------------------------------------------------- statistics
def shot_sums(shot, v):
    shots, inv = np.unique(shot, return_inverse=True)
    s = np.zeros(len(shots)); np.add.at(s, inv, v)
    return s


def boot_skill(shot, se_a, se_ref, rng, B):
    sa, sr = shot_sums(shot, se_a), shot_sums(shot, se_ref)
    n = len(sa)
    point = 1.0 - sa.sum() / sr.sum()
    idx = rng.integers(0, n, size=(B, n))
    sk = 1.0 - sa[idx].sum(1) / sr[idx].sum(1)
    sk = sk[np.isfinite(sk)]
    lo, hi = np.percentile(sk, [2.5, 97.5])
    return {"skill_point": float(point), "skill_ci95": [float(lo), float(hi)], "n_shots": int(n)}


def boot_ratio(shot, se_a, se_c, se_ref, rng, B):
    """ratio = skill_a / skill_c against the same reference, with its shot-cluster CI."""
    sa, sc, sr = shot_sums(shot, se_a), shot_sums(shot, se_c), shot_sums(shot, se_ref)
    n = len(sa)
    point = (1 - sa.sum() / sr.sum()) / (1 - sc.sum() / sr.sum())
    idx = rng.integers(0, n, size=(B, n))
    ra = 1 - sa[idx].sum(1) / sr[idx].sum(1); rc = 1 - sc[idx].sum(1) / sr[idx].sum(1)
    q = ra / rc
    q = q[np.isfinite(q)]
    lo, hi = np.percentile(q, [2.5, 97.5])
    return {"ratio_point": float(point), "ratio_ci95": [float(lo), float(hi)]}


# ---------------------------------------------------------------- one split
def score_split(pop, seed, smoke):
    p = POPS[pop]
    run = DATA / p["run"].format(s=seed)
    control_metrics = DATA / p["control_metrics"].format(s=seed) / "metrics.json"
    manifest = DATA / p["split"].format(s=seed) / "split_manifest.json"
    for needed in (run / "comparison_errors_test.npz", run / "metrics.json", control_metrics, manifest):
        if not needed.exists():
            raise SystemExit(f"FATAL: missing {needed}")
    pin_env(p["cut"])
    t0 = time.time()
    npz = np.load(run / "comparison_errors_test.npz")
    rows = frozen_rows(control_metrics, manifest, seed, npz)
    print(f"[b28-1] {pop} s{seed}: frozen rows re-derived n={rows['check']['n']} "
          f"(y bit-identical={rows['check']['y_bit_identical']}, {time.time()-t0:.0f}s)", flush=True)

    from seq_data import load_grid_files  # noqa: E402
    from evaluate import _load_stats  # noqa: E402
    my_stats = _load_stats(json.loads((run / "metrics.json").read_text(encoding="utf-8")))
    grid, dims = load_grid_files(DATA, drop_stuck_targets=True, ti_spike_cut_ev=float(p["cut"]))
    man = json.loads(manifest.read_text(encoding="utf-8"))
    val_files = [f for f in man["val_files"] if f in grid]
    test_files = sorted(set(rows["file"]))
    if smoke:
        val_files = val_files[:30]
    tables = {f: ewma_table(grid[f], dims, my_stats, p["cut"]) for f in val_files + test_files}

    # VAL rows: every observed T_i row with a past observation (the fitting population)
    vkeys = [(f, t) for f in val_files for t, r in tables[f].items() if r["obs"] and r["has"]]
    Ev, Pv, Dv, _, Yv, _ = stack(tables, vkeys)
    mse_pv = ((Yv - Pv) ** 2).mean()
    val_skill = {str(a): float(1 - ((Yv - Ev[:, j]) ** 2).mean() / mse_pv) for j, a in enumerate(ALPHAS)}
    j_star = int(np.argmax([val_skill[str(a)] for a in ALPHAS]))
    alpha_star = ALPHAS[j_star]
    A = np.concatenate([Dv, np.ones((len(Yv), 1))], 1)
    resid = Yv - Ev[:, j_star]
    lam = RIDGE * len(Yv)
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ resid)

    # TEST: the frozen rows
    tkeys = list(zip(rows["file"], rows["time"].tolist()))
    missing = [k for k in tkeys if k[1] not in tables[k[0]]]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} frozen rows have no grid feature row, e.g. {missing[0]}")
    Et, Pt, Dt, dEt, Yt, St = stack(tables, tkeys)
    if float(np.max(np.abs(Yt - rows["y"]))) > 1e-2:
        raise SystemExit("FATAL: grid y does not match the frozen y_true")
    y = rows["y"].astype(np.float64)
    shot = rows["shot"]
    se = {"model": npz["CES_TI_se_model"].astype(np.float64)}
    for ref in REFS:
        se[ref] = npz[f"CES_TI_se_{ref}"].astype(np.float64)
    for j, a in enumerate(ALPHAS):
        se[f"ewma_{a}"] = (Et[:, j] - y) ** 2
    pred_b = Et[:, j_star] + np.concatenate([Dt, np.ones((len(y), 1))], 1) @ w
    se["ewma_star"] = se[f"ewma_{alpha_star}"]
    se["ewma_inc"] = (pred_b - y) ** 2

    rng = np.random.default_rng(SEED_BOOT)
    B = 300 if smoke else B_BOOT
    arms = ["ewma_star", "ewma_inc", "model"] + [f"ewma_{a}" for a in ALPHAS]
    skills = {arm: {ref: boot_skill(shot, se[arm], se[ref], rng, B) for ref in REFS} for arm in arms}
    ratios = {arm: {ref: boot_ratio(shot, se[arm], se["model"], se[ref], rng, B) for ref in ("persistence", "pchip")}
              for arm in ("ewma_star", "ewma_inc")}
    # strata (reported): top-10 % |dECEI| rows vs rest; staleness bins
    top = dEt >= np.percentile(dEt, 90)
    strata = {}
    for name, m in (("top10_dECEI", top), ("rest", ~top),
                    ("stale_le15ms", rows["dt_ms"] <= 15), ("stale_15_55ms", (rows["dt_ms"] > 15) & (rows["dt_ms"] <= 55)),
                    ("stale_gt55ms", rows["dt_ms"] > 55)):
        strata[name] = {"n": int(m.sum())}
        if m.sum() >= 20:
            for arm in ("ewma_star", "ewma_inc", "model"):
                strata[name][arm] = {ref: float(1 - se[arm][m].sum() / se[ref][m].sum()) for ref in ("persistence", "pchip")}
    std_ti = float(my_stats["target"]["std"][0])
    rec = {"pop": pop, "seed": seed, "run": run.name, "check": rows["check"], "n_rows": int(len(y)),
           "n_val_rows": int(len(Yv)), "alpha_star": alpha_star, "val_skill_vs_persistence": val_skill,
           "coef_inc_eV_per_sigma": w[:N_FAST].tolist(), "coef_bias_eV": float(w[N_FAST]),
           "coef_sum_BES_eV": float(w[:N_BES].sum()), "coef_sum_ECEI_eV": float(w[N_BES:N_BES + N_ECEI].sum()),
           "coef_sum_MC_eV": float(w[N_BES + N_ECEI:N_FAST].sum()),
           "skills": skills, "ratios": ratios, "strata": strata,
           "share_persistence_se_in_top10": float(se["persistence"][top].sum() / se["persistence"].sum()),
           "minutes": round((time.time() - t0) / 60, 1)}
    r_p = ratios["ewma_inc"]["persistence"]["ratio_point"]
    print(f"[b28-1] {pop} s{seed}: alpha*={alpha_star} model/pers={skills['model']['persistence']['skill_point']:+.3f} "
          f"ewma/pers={skills['ewma_star']['persistence']['skill_point']:+.3f} "
          f"ewma+inc/pers={skills['ewma_inc']['persistence']['skill_point']:+.3f} ratio={r_p:.3f} "
          f"| vs pchip: model {skills['model']['pchip']['skill_point']:+.3f} ewma {skills['ewma_star']['pchip']['skill_point']:+.3f} "
          f"ewma+inc {skills['ewma_inc']['pchip']['skill_point']:+.3f} ({rec['minutes']} min)", flush=True)
    return rec


def verdict(records):
    """PREREGISTRATION_B28.md sec. 3 rules, applied to whatever splits are present."""
    out = {}
    for pop in POPS:
        rs = [r for r in records if r["pop"] == pop]
        if not rs:
            continue
        ratio_b = [r["ratios"]["ewma_inc"]["persistence"]["ratio_point"] for r in rs]
        ratio_a = [r["ratios"]["ewma_star"]["persistence"]["ratio_point"] for r in rs]
        ewma_vs_pchip = [r["skills"]["ewma_star"]["pchip"]["skill_point"] for r in rs]
        inc_vs_pchip = [r["skills"]["ewma_inc"]["pchip"]["skill_point"] for r in rs]
        out[pop] = {
            "n_splits": len(rs),
            "H28_1_ratio_ewma_inc_vs_persistence": ratio_b,
            "H28_1_pass_count_ge_0.75": int(sum(v >= 0.75 for v in ratio_b)),
            "H28_1_supported": bool(len(rs) == 4 and all(v >= 0.75 for v in ratio_b)),
            "ratio_ewma_only": ratio_a,
            "H28_2_ewma_vs_pchip_skill": ewma_vs_pchip,
            "H28_2_inside_pm0.05_count": int(sum(abs(v) <= 0.05 for v in ewma_vs_pchip)),
            "H28_2_supported": bool(len(rs) == 4 and all(abs(v) <= 0.05 for v in ewma_vs_pchip)),
            "ewma_inc_vs_pchip_skill": inc_vs_pchip,
            "model_vs_pchip_skill": [r["skills"]["model"]["pchip"]["skill_point"] for r in rs],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one split (cut s42), B=300, 30 val files")
    ap.add_argument("--pops", nargs="*", default=list(POPS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = ap.parse_args()
    out_path = DATA / (".b28_stage1_smoke.json" if args.smoke else ".b28_stage1.json")
    cells = [("cut", 42)] if args.smoke else [(p, s) for p in args.pops for s in args.seeds]
    records = []
    if out_path.exists() and not args.smoke:
        records = json.loads(out_path.read_text(encoding="utf-8")).get("splits", [])
        done = {(r["pop"], r["seed"]) for r in records}
        cells = [c for c in cells if c not in done]
        print(f"[b28-1] resuming: {len(done)} splits done, {len(cells)} to go", flush=True)
    for pop, seed in cells:
        records.append(score_split(pop, seed, args.smoke))
        payload = {"protocol": {"alphas": ALPHAS, "ridge": RIDGE, "B": B_BOOT, "seed_boot": SEED_BOOT,
                                "refs": REFS, "env": GATE_ENV, "per_shot_norm": 1},
                   "splits": records, "verdict": verdict(records)}
        out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[b28-1] wrote {out_path}", flush=True)
    print(json.dumps(verdict(records), indent=1))


if __name__ == "__main__":
    main()
