"""B.28 stage 3 (PREREGISTRATION_B28.md sec. 2 "Stage 3", sec. 6 addendum of 2026-11-09): the
three registered measurements that would overturn the stage-2 verdict ("the event-row nonlinear
term is needed", sec. 8bz-2). Measurements, not claims: no headline changes here.

  jac     Jacobian d y_TI(t) / d x(t-k, c) on EVERY frozen TEST row of the 8 backbone runs
          (cut `.b1_seqv2_s{s}_i{s}`, inclusive `.b5i_seqv2_s{s}`; rows re-derived and verified
          exactly as stage 1 did), 100-step window, CPU, batched autograd. From it: the per-group
          lag kernels, the saturation form of the lag-0 gain (sech^2 vs Cauchy vs flat, all
          rows x channels), the state dependence of the lag-0 gain, the amplitude decay of the
          carried-T_i gradient, and a cross-fitted (shot-parity) regression of the residual
          r_t = backbone - trained equation (`.b28_b28_{pop}_s{s}`) on candidate term families.
  trunc   Context-truncation deficit at 1/2/3/5/10 (and 20/50) steps on the same frozen
          checkpoints. Cut: the sec. 8ac artifacts `.reach_s{s}` (identity re-verified against the
          frozen npz). Inclusive: eval_seq with CES_SEQ_CONTEXTS on a copy of `.b5i_seqv2_s{s}`
          (CPU), identity verified the same way. Per-step decay of the skill deficit and of the
          reference-free MSE excess, fitted log-linearly; joined with the amplitude decay of jac.
  mcnorm  One seq_v2 re-train (cut split 42, init 42, the B.1 env, CES_SEQ_LOG_WIH=1, GPU) whose
          first-layer W_ih column norms are logged per epoch; the "random walk on a white input"
          reading of the MC columns (readout sec. 2.2) is checked on the trajectory. Stage 2's own
          runs did not log it (disclosed in PREREGISTRATION_B28.md sec. 6).

Usage (repo root):
  py ces_prediction/experiments/b28_equation/stage3_measure.py --part jac|trunc|mcnorm|all
                                                               [--smoke] [--resume] [--seeds 42,1]
Writes data/.b28_stage3_<part>*.json per piece and merges everything into data/.b28_stage3.json.
Console output is ASCII (cp949 console).
"""

import argparse
import ctypes
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
HERE = Path(__file__).resolve().parent
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CES_DIR / "experiments" / "b1_gate"))
sys.path.insert(0, str(SEQ_DIR))
sys.path.insert(0, str(CES_DIR))
import stage1_score as S1  # noqa: E402  (pin_env, frozen_rows, POPS, SEEDS)
from run_b1_gate import GATE_ENV, AMBIENT_VARS, run_step  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

SEEDS = S1.SEEDS
POPS = S1.POPS
N_BES, N_ECEI, N_FAST = S1.N_BES, S1.N_ECEI, S1.N_FAST
GROUPS = {"BES": list(range(0, N_BES)), "ECEI": list(range(N_BES, N_BES + N_ECEI)),
          "MC": list(range(N_BES + N_ECEI, N_FAST))}
COL_DT, COL_CAR, COL_STALE, COL_HAS = 15, 16, 17, 18
CTX, LAGS, BATCH = 100, 12, 512
RUNGS = (1, 2, 3, 5, 10)
RUNGS_INCL = (1, 2, 3, 5, 10, 20, 50)
REFS = ("gp_causal", "persistence", "pchip")
RIDGE = 1e-2
OUT = DATA / ".b28_stage3.json"


def keep_awake(on):
    if os.name != "nt":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def skill(se_a, se_ref):
    return float(1.0 - se_a.sum() / se_ref.sum())


def loglin_rho(ks, vals):
    """vals ~ A * rho^k on the k with vals > 0 -> rho, A, R^2 (None if < 2 usable points)."""
    ks = np.asarray(ks, float); vals = np.asarray(vals, float)
    m = vals > 0
    if m.sum() < 2:
        return {"rho": None, "A": None, "r2": None, "n_pts": int(m.sum())}
    k, lv = ks[m], np.log(vals[m])
    slope, icpt = np.polyfit(k, lv, 1)
    fit = slope * k + icpt
    ss = ((lv - lv.mean()) ** 2).sum()
    r2 = 1.0 - ((lv - fit) ** 2).sum() / ss if ss > 0 else None
    return {"rho": float(np.exp(slope)), "A": float(np.exp(icpt)), "r2": None if r2 is None else float(r2),
            "n_pts": int(m.sum()), "memory_steps": float(-1.0 / slope) if slope < 0 else None}


# ====================================================================== part (i): Jacobian
def jac_split(pop, seed, smoke, resume):
    out_json = DATA / f".b28_stage3_jac_{pop}_s{seed}{'_smoke' if smoke else ''}.json"
    if resume and out_json.exists():
        print(f"[b28-3 jac] {pop} s{seed}: done already, skipping (--resume)", flush=True)
        return json.loads(out_json.read_text(encoding="utf-8"))
    t0 = time.time()
    p = POPS[pop]
    run = DATA / p["run"].format(s=seed)
    eq_run = DATA / f".b28_b28_{pop}_s{seed}"
    control_metrics = DATA / p["control_metrics"].format(s=seed) / "metrics.json"
    manifest = DATA / p["split"].format(s=seed) / "split_manifest.json"
    for needed in (run / "comparison_errors_test.npz", run / "weights" / "seq_lstm.pth",
                   eq_run / "weights" / "seq_lstm.pth", eq_run / "comparison_errors_test.npz",
                   control_metrics, manifest):
        if not needed.exists():
            raise SystemExit(f"FATAL: missing {needed}")
    S1.pin_env(p["cut"])
    npz = np.load(run / "comparison_errors_test.npz")
    rows = S1.frozen_rows(control_metrics, manifest, seed, npz)
    print(f"[b28-3 jac] {pop} s{seed}: frozen rows n={rows['check']['n']} "
          f"(y bit-identical={rows['check']['y_bit_identical']}, {time.time()-t0:.0f}s)", flush=True)

    from seq_data import load_grid_files, build_blocks  # noqa: E402
    from evaluate import _load_stats  # noqa: E402
    from seq_models import SEQ_MODELS  # noqa: E402
    my_stats = _load_stats(json.loads((run / "metrics.json").read_text(encoding="utf-8")))
    eq_stats = _load_stats(json.loads((eq_run / "metrics.json").read_text(encoding="utf-8")))
    mean, std = float(my_stats["target"]["mean"][0]), float(my_stats["target"]["std"][0])
    eq_mean, eq_std = float(eq_stats["target"]["mean"][0]), float(eq_stats["target"]["std"][0])
    stats_match = abs(mean - eq_mean) < 1e-6 * max(1.0, abs(mean)) and abs(std - eq_std) < 1e-6 * std
    grid, dims = load_grid_files(DATA, drop_stuck_targets=True, ti_spike_cut_ev=float(p["cut"]))
    model = SEQ_MODELS["v2"]()
    model.load_state_dict(torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
    model.eval()
    eq = SEQ_MODELS["b28"]()
    eq.load_state_dict(torch.load(eq_run / "weights" / "seq_lstm.pth", map_location="cpu"))
    eq.eval()
    eq_phys = eq.physics(target_std=eq_stats["target"]["std"])

    # blocks of the frozen rows' files; exact-float (file, time) lookup as stage 1 used
    blocks, lookup = [], {}
    for f in sorted(set(rows["file"])):
        for b in build_blocks(grid[f], dims, my_stats, per_shot_norm=True):
            bi = len(blocks)
            blocks.append(b)
            for i, t in enumerate(b["time"].tolist()):
                lookup[(f, float(t))] = (bi, i)
    idx = []
    for f, t in zip(rows["file"], rows["time"].tolist()):
        if (f, t) not in lookup:
            raise SystemExit(f"FATAL: frozen row {(f, t)} has no block row")
        idx.append(lookup[(f, t)])
    if smoke:
        idx = idx[:2000]
    N = len(idx)
    y = rows["y"].astype(np.float64)[:N]
    shot = rows["shot"][:N]
    dt_ms = rows["dt_ms"][:N]

    # full-block predictions: backbone and the trained equation (+ its EWMA state m)
    with torch.no_grad():
        for b in blocks:
            x = torch.from_numpy(b["x"])[None]
            b["pred_bb"] = model(x)[0, :, 0].numpy() * std + mean
            m, term = eq.ti_terms(x)
            b["m_eq"] = m[0].numpy() * eq_std + eq_mean
            b["pred_eq"] = (m + term + eq.b0)[0].numpy() * eq_std + eq_mean
    pb = np.array([blocks[bi]["pred_bb"][i] for bi, i in idx], np.float64)
    pe = np.array([blocks[bi]["pred_eq"][i] for bi, i in idx], np.float64)
    me = np.array([blocks[bi]["m_eq"][i] for bi, i in idx], np.float64)
    se_bb_npz = npz["CES_TI_se_model"].astype(np.float64)[:N]
    eq_npz = np.load(eq_run / "comparison_errors_test.npz")
    pair_ok = (np.array_equal(eq_npz["CES_TI_shot"][:N], shot)
               and np.array_equal(eq_npz["CES_TI_dt_ms"][:N], dt_ms))
    se_eq_npz = eq_npz["CES_TI_se_model"].astype(np.float64)[:N]
    check = {**rows["check"], "stats_match_eq_run": bool(stats_match), "eq_rows_paired": bool(pair_ok),
             "max_abs_err_diff_backbone_eV": float(np.max(np.abs(np.abs(pb - y) - np.sqrt(se_bb_npz)))),
             "max_abs_err_diff_equation_eV": float(np.max(np.abs(np.abs(pe - y) - np.sqrt(se_eq_npz)))),
             "n_rows_used": int(N)}
    print(f"[b28-3 jac] {pop} s{seed}: full-block recompute vs npz: backbone max|d err| "
          f"{check['max_abs_err_diff_backbone_eV']:.2f} eV, equation {check['max_abs_err_diff_equation_eV']:.2f} eV, "
          f"paired={pair_ok}", flush=True)

    # per-row inputs
    def at(off):
        return np.stack([blocks[bi]["x"][max(i - off, 0)] for bi, i in idx]).astype(np.float64)
    X0, X1, X2, X3 = at(0), at(1), at(2), at(3)
    d0 = X0[:, :N_FAST] - X1[:, :N_FAST]
    d1 = X1[:, :N_FAST] - X2[:, :N_FAST]
    d2 = X2[:, :N_FAST] - X3[:, :N_FAST]
    stale_s = np.expm1(X0[:, COL_STALE])
    carried = X0[:, COL_CAR] * std + mean
    dev_z = (carried - me) / std                  # carried observation minus the equation's EWMA state
    lens = np.array([min(i + 1, CTX) for bi, i in idx])

    # batched Jacobian on the 100-step window ending at the row
    J = np.zeros((N, LAGS, X0.shape[1]), np.float32)
    ywin = np.zeros(N)
    t1 = time.time()
    for L in np.unique(lens):
        sel = np.where(lens == L)[0]
        for s0 in range(0, len(sel), BATCH):
            ss = sel[s0:s0 + BATCH]
            xw = torch.from_numpy(np.stack([blocks[idx[r][0]]["x"][idx[r][1] - L + 1: idx[r][1] + 1] for r in ss]))
            xw.requires_grad_(True)
            out = model(xw)[:, -1, 0]
            out.sum().backward()
            g = xw.grad.numpy()[:, ::-1]           # lag 0 first
            k = min(LAGS, int(L))
            J[ss, :k] = g[:, :k]
            ywin[ss] = out.detach().numpy() * std + mean
    check["max_abs_window_vs_full_eV"] = float(np.max(np.abs(ywin - pb)))
    check["n_rows_window_shorter_than_ctx"] = int((lens < CTX).sum())
    print(f"[b28-3 jac] {pop} s{seed}: Jacobian on {N} rows ({time.time()-t1:.0f}s), "
          f"window vs full max {check['max_abs_window_vs_full_eV']:.2f} eV", flush=True)

    res = {"pop": pop, "seed": seed, "run": run.name, "eq_run": eq_run.name, "check": check,
           "n_rows": int(N), "eq_physics": eq_phys}

    # ---- 1. lag kernels per group (sum over the group's channels, mean over rows), z/z units
    kern = {}
    for g, cols in GROUPS.items():
        prof = J[:, :, cols].sum(2).mean(0)
        kern[g] = {"lag_profile": prof.tolist(), "impulse": float(prof[0]),
                   "increment_index_minus_l1_over_l0": float(-prof[1] / prof[0]) if prof[0] != 0 else None,
                   "step_over_impulse": float(prof.sum() / prof[0]) if prof[0] != 0 else None,
                   "impulse_eV_per_sigma_sum": float(prof[0] * std)}
    for name, col in (("carried", COL_CAR), ("stale", COL_STALE), ("dt", COL_DT), ("has", COL_HAS)):
        kern[name] = {"lag_profile": J[:, :, col].mean(0).tolist()}
    res["kernels"] = kern

    # ---- 2. saturation form of the lag-0 gain, all rows x channels per group
    def fit_forms(delta, grad):
        a_d = np.abs(delta)
        kap = np.logspace(np.log10(0.1), np.log10(20.0), 80)
        best = {}
        for form, fn in (("sech2", lambda k: 1.0 / np.cosh(delta / k) ** 2),
                         ("cauchy", lambda k: 1.0 / (1.0 + (delta / k) ** 2))):
            sse_best, k_best, a_best = np.inf, None, None
            for k in kap:
                phi = fn(k)
                a = float((phi * grad).sum() / (phi * phi).sum())
                sse = float(((grad - a * phi) ** 2).sum())
                if sse < sse_best:
                    sse_best, k_best, a_best = sse, float(k), a
            best[form] = {"kappa_sigma": k_best, "a_z": a_best, "a_eV_per_sigma": a_best * std, "sse": sse_best}
        a0 = float(grad.mean())
        best["flat"] = {"a_z": a0, "a_eV_per_sigma": a0 * std, "sse": float(((grad - a0) ** 2).sum())}
        edges = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (a_d >= lo) & (a_d < hi)
            if m.sum() >= 30:
                bins.append({"lo": lo, "hi": None if hi == np.inf else hi, "n": int(m.sum()),
                             "mean_abs_delta": float(a_d[m].mean()), "mean_grad_z": float(grad[m].mean()),
                             "mean_grad_eV_per_sigma": float(grad[m].mean() * std)})
        ref = bins[0]["mean_grad_z"] if bins else None
        for b_ in bins:
            b_["grad_rel_to_quiet_bin"] = float(b_["mean_grad_z"] / ref) if ref else None
        best["bins"] = bins
        return best
    sat = {}
    for g in ("BES", "ECEI"):
        cols = GROUPS[g]
        sat[g] = fit_forms(d0[:, cols].ravel(), J[:, 0, cols].astype(np.float64).ravel())
        sat[g]["trained_eq_kappa_sigma"] = eq_phys["kappa_sigma"]
    res["saturation"] = sat

    # ---- 3. state dependence of the lag-0 gain (the equation has none)
    def ols_partial(Y, feats):
        names = list(feats)
        F = np.column_stack([feats[n] for n in names])
        F = (F - F.mean(0)) / (F.std(0) + 1e-12)
        A = np.column_stack([F, np.ones(len(Y))])
        w = np.linalg.lstsq(A, Y, rcond=None)[0]
        sst = ((Y - Y.mean()) ** 2).sum()
        r2 = 1 - ((Y - A @ w) ** 2).sum() / sst
        out = {"r2": float(r2), "coef_std": {n: float(w[j]) for j, n in enumerate(names)}, "partial_r2": {}}
        for j, n in enumerate(names):
            keep = [k for k in range(len(names)) if k != j] + [len(names)]
            wj = np.linalg.lstsq(A[:, keep], Y, rcond=None)[0]
            out["partial_r2"][n] = float(r2 - (1 - ((Y - A[:, keep] @ wj) ** 2).sum() / sst))
        return out
    state = {}
    for g in ("BES", "ECEI"):
        cols = GROUPS[g]
        Gm = J[:, 0, cols].astype(np.float64).mean(1)
        feats = {"abs_inc_now": np.abs(d0[:, cols]).mean(1), "abs_inc_prev": np.abs(d1[:, cols]).mean(1),
                 "abs_inc_prev2": np.abs(d2[:, cols]).mean(1), "stale_s": stale_s, "abs_dev_carried_minus_ewma": np.abs(dev_z),
                 "level": X0[:, cols].mean(1)}
        state[g] = ols_partial(Gm, feats)
        # direction: gain by terciles of the previous step's |increment|
        q = np.quantile(feats["abs_inc_prev"], [1 / 3, 2 / 3])
        terc = np.digitize(feats["abs_inc_prev"], q)
        state[g]["gain_by_prev_inc_tercile_eV_per_sigma"] = [float(Gm[terc == k].mean() * std) for k in range(3)]
        qs = np.quantile(stale_s, [1 / 3, 2 / 3])
        ts = np.digitize(stale_s, qs)
        state[g]["gain_by_stale_tercile_eV_per_sigma"] = [float(Gm[ts == k].mean() * std) for k in range(3)]
    res["state_dependence"] = state

    # ---- 4. residual r = backbone - equation, cross-fitted by shot parity
    r = pb - pe
    se_p = npz["CES_TI_se_pchip"].astype(np.float64)[:N]
    se_per = npz["CES_TI_se_persistence"].astype(np.float64)[:N]
    fams = {
        "d0_linear": d0,
        "lag1_inc": d1,
        "lag2_inc": d2,
        "momentum_d0xd1": d0 * d1,
        "gain_x_stale": d0 * stale_s[:, None],
        "gain_x_dev": d0 * dev_z[:, None],
        "level": X0[:, :N_FAST],
        "abs_inc": np.abs(d0),
        "ewma_state": np.column_stack([dev_z, stale_s, (me - mean) / std]),
    }
    fams["all"] = np.column_stack(list(fams.values()))
    fams["lag1+momentum+gain_state"] = np.column_stack([fams["lag1_inc"], fams["momentum_d0xd1"],
                                                        fams["gain_x_stale"], fams["gain_x_dev"]])
    shots_u = np.unique(shot)
    fold_of_shot = {s: k % 2 for k, s in enumerate(shots_u)}
    fold = np.array([fold_of_shot[s] for s in shot])
    dE = np.abs(d0[:, GROUPS["ECEI"]]).mean(1)
    top = dE >= np.percentile(dE, 90)
    sk_bb, sk_eq = skill((pb - y) ** 2, se_p), skill((pe - y) ** 2, se_p)
    resid = {"skill_vs_pchip_backbone": sk_bb, "skill_vs_pchip_equation": sk_eq,
             "skill_vs_persistence_backbone": skill((pb - y) ** 2, se_per),
             "skill_vs_persistence_equation": skill((pe - y) ** 2, se_per),
             "resid_rms_eV": float(np.sqrt((r ** 2).mean())), "resid_rms_top10_eV": float(np.sqrt((r[top] ** 2).mean())),
             "resid_var_share_top10": float((r[top] ** 2).sum() / (r ** 2).sum()), "families": {}}
    for name, F in fams.items():
        rhat = np.zeros(N)
        for k in (0, 1):
            tr, te = fold != k, fold == k
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-12
            A = np.column_stack([(F[tr] - mu) / sd, np.ones(tr.sum())])
            lam = RIDGE * tr.sum()
            w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ r[tr])
            rhat[te] = np.column_stack([(F[te] - mu) / sd, np.ones(te.sum())]) @ w
        sk_fix = skill((pe + rhat - y) ** 2, se_p)
        sst = ((r - r.mean()) ** 2).sum()
        resid["families"][name] = {
            "n_features": int(F.shape[1]),
            "r2_cv": float(1 - ((r - rhat) ** 2).sum() / sst),
            "r2_cv_top10_dECEI": float(1 - ((r - rhat)[top] ** 2).sum() / ((r[top] - r[top].mean()) ** 2).sum()),
            "r2_cv_rest": float(1 - ((r - rhat)[~top] ** 2).sum() / ((r[~top] - r[~top].mean()) ** 2).sum()),
            "skill_vs_pchip_eq_plus_fit": sk_fix,
            "gap_closed": float((sk_fix - sk_eq) / (sk_bb - sk_eq)) if sk_bb != sk_eq else None,
        }
    res["residual"] = resid

    # ---- 5. amplitude decay of the carried-T_i gradient (joins with the truncation deficit)
    prof = J[:, :, COL_CAR].astype(np.float64).mean(0)
    ratios = [float(prof[k + 1] / prof[k]) if prof[k] != 0 else None for k in range(6)]
    amp = {"lag_profile": prof.tolist(), "ratios": ratios, "sum_lag0_11": float(prof.sum()),
           "fit_lag0_3": loglin_rho(range(4), prof[:4]), "fit_lag1_5": loglin_rho(range(1, 6), prof[1:6]),
           "by_stale": {}}
    for name, m in (("le15ms", dt_ms <= 15), ("15_55ms", (dt_ms > 15) & (dt_ms <= 55)), ("gt55ms", dt_ms > 55)):
        if m.sum() >= 50:
            pm = J[m][:, :, COL_CAR].astype(np.float64).mean(0)
            amp["by_stale"][name] = {"n": int(m.sum()), "lag_profile": pm[:6].tolist(),
                                     "fit_lag0_3": loglin_rho(range(4), pm[:4])}
    res["amplitude"] = amp
    res["minutes"] = round((time.time() - t0) / 60, 1)
    out_json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    f_best = max((k for k in resid["families"] if k != "all"), key=lambda k: resid["families"][k]["gap_closed"] or -9)
    print(f"[b28-3 jac] {pop} s{seed}: rho_amp(l0-3)={amp['fit_lag0_3']['rho']:.3f} "
          f"kappa sech2 BES={sat['BES']['sech2']['kappa_sigma']:.2f} ECEI={sat['ECEI']['sech2']['kappa_sigma']:.2f} "
          f"(eq {eq_phys['kappa_sigma']:.2f}) | resid R2 all={resid['families']['all']['r2_cv']:.3f} "
          f"gap closed all={resid['families']['all']['gap_closed']:.2f} best single={f_best} "
          f"{resid['families'][f_best]['gap_closed']:.2f} ({res['minutes']} min)", flush=True)
    return res


# ====================================================================== part (i-b): residual, kernel-motivated families
# Added 2026-11-09 AFTER the cut-s42 Jacobian was read (disclosed in sec. 8bz-3): its fast-channel
# lag kernel is "lag 0 positive, lags 1-11 together cancelling it" -- the backbone responds to
# x_t minus a ~10-step running mean of the channel, not to the 10 ms increment -- and its lag-0
# gain is flat to ~2 sigma and cut off beyond 3 sigma, not tanh(kappa ~ 1). These families test
# those two readings as residual terms, cross-fitted by shot parity like part (i).
def resid2_split(pop, seed, smoke, resume):
    out_json = DATA / f".b28_stage3_resid2_{pop}_s{seed}{'_smoke' if smoke else ''}.json"
    if resume and out_json.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))
    t0 = time.time()
    p = POPS[pop]
    run = DATA / p["run"].format(s=seed)
    eq_run = DATA / f".b28_b28_{pop}_s{seed}"
    control_metrics = DATA / p["control_metrics"].format(s=seed) / "metrics.json"
    manifest = DATA / p["split"].format(s=seed) / "split_manifest.json"
    S1.pin_env(p["cut"])
    npz = np.load(run / "comparison_errors_test.npz")
    rows = S1.frozen_rows(control_metrics, manifest, seed, npz)
    from seq_data import load_grid_files, build_blocks  # noqa: E402
    from evaluate import _load_stats  # noqa: E402
    from seq_models import SEQ_MODELS  # noqa: E402
    my_stats = _load_stats(json.loads((run / "metrics.json").read_text(encoding="utf-8")))
    eq_stats = _load_stats(json.loads((eq_run / "metrics.json").read_text(encoding="utf-8")))
    mean, std = float(my_stats["target"]["mean"][0]), float(my_stats["target"]["std"][0])
    eq_mean, eq_std = float(eq_stats["target"]["mean"][0]), float(eq_stats["target"]["std"][0])
    grid, dims = load_grid_files(DATA, drop_stuck_targets=True, ti_spike_cut_ev=float(p["cut"]))
    model = SEQ_MODELS["v2"]()
    model.load_state_dict(torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
    model.eval()
    eq = SEQ_MODELS["b28"]()
    eq.load_state_dict(torch.load(eq_run / "weights" / "seq_lstm.pth", map_location="cpu"))
    eq.eval()
    blocks, lookup = [], {}
    for f in sorted(set(rows["file"])):
        for b in build_blocks(grid[f], dims, my_stats, per_shot_norm=True):
            bi = len(blocks)
            blocks.append(b)
            for i, t in enumerate(b["time"].tolist()):
                lookup[(f, float(t))] = (bi, i)
    idx = [lookup[(f, t)] for f, t in zip(rows["file"], rows["time"].tolist())]
    if smoke:
        idx = idx[:2000]
    N = len(idx)
    y = rows["y"].astype(np.float64)[:N]
    shot = rows["shot"][:N]
    with torch.no_grad():
        for b in blocks:
            x = torch.from_numpy(b["x"])[None]
            b["pred_bb"] = model(x)[0, :, 0].numpy() * std + mean
            m, term = eq.ti_terms(x)
            b["m_eq"] = m[0].numpy() * eq_std + eq_mean
            b["pred_eq"] = (m + term + eq.b0)[0].numpy() * eq_std + eq_mean
    pb = np.array([blocks[bi]["pred_bb"][i] for bi, i in idx], np.float64)
    pe = np.array([blocks[bi]["pred_eq"][i] for bi, i in idx], np.float64)
    me = np.array([blocks[bi]["m_eq"][i] for bi, i in idx], np.float64)
    K = 12
    lagged = np.zeros((N, K, N_FAST))
    for k in range(K):
        lagged[:, k] = np.stack([blocks[bi]["x"][max(i - k, 0), :N_FAST] for bi, i in idx])
    X0 = np.stack([blocks[bi]["x"][i] for bi, i in idx]).astype(np.float64)
    d0 = lagged[:, 0] - lagged[:, 1]
    stale_s = np.expm1(X0[:, COL_STALE])
    dev_z = (X0[:, COL_CAR] * std + mean - me) / std
    hp10 = lagged[:, 0] - lagged[:, 1:11].mean(1)
    hp5 = lagged[:, 0] - lagged[:, 1:6].mean(1)
    # causal EWMA of the past levels (beta = 0.3, oldest lag first)
    ew = lagged[:, K - 1].copy()
    for k in range(K - 2, 0, -1):
        ew = 0.3 * lagged[:, k] + 0.7 * ew
    hp_ewma = lagged[:, 0] - ew
    clip3 = np.clip(d0, -3.0, 3.0)
    gate3 = d0 * (np.abs(d0) <= 3.0)
    fams = {
        "d0_linear": d0,
        "d0_clip3sigma": clip3,
        "d0_gate3sigma": gate3,
        "hp_mean5": hp5,
        "hp_mean10": hp10,
        "hp_ewma_b0.3": hp_ewma,
        "hp_mean10_gate3": hp10 * (np.abs(hp10) <= 3.0),
        "fir_lags0_11": lagged.reshape(N, -1),
        "ewma_state": np.column_stack([dev_z, stale_s, (me - mean) / std]),
    }
    fams["hp_mean10+ewma_state"] = np.column_stack([fams["hp_mean10"], fams["ewma_state"]])
    fams["fir+ewma_state"] = np.column_stack([fams["fir_lags0_11"], fams["ewma_state"]])
    r = pb - pe
    se_p = npz["CES_TI_se_pchip"].astype(np.float64)[:N]
    shots_u = np.unique(shot)
    fold_of = {s_: k % 2 for k, s_ in enumerate(shots_u)}
    fold = np.array([fold_of[s_] for s_ in shot])
    dE = np.abs(d0[:, GROUPS["ECEI"]]).mean(1)
    top = dE >= np.percentile(dE, 90)
    sk_bb, sk_eq = skill((pb - y) ** 2, se_p), skill((pe - y) ** 2, se_p)
    res = {"pop": pop, "seed": seed, "n_rows": int(N), "skill_vs_pchip_backbone": sk_bb,
           "skill_vs_pchip_equation": sk_eq, "families": {}, "fir_group_kernel_from_fit": {}}
    for name, F in fams.items():
        rhat = np.zeros(N)
        ws = []
        for k in (0, 1):
            tr, te = fold != k, fold == k
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-12
            A = np.column_stack([(F[tr] - mu) / sd, np.ones(tr.sum())])
            lam = RIDGE * tr.sum()
            w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ r[tr])
            rhat[te] = np.column_stack([(F[te] - mu) / sd, np.ones(te.sum())]) @ w
            ws.append(w[:-1] / sd)
        sk_fix = skill((pe + rhat - y) ** 2, se_p)
        sst = ((r - r.mean()) ** 2).sum()
        res["families"][name] = {
            "n_features": int(F.shape[1]), "r2_cv": float(1 - ((r - rhat) ** 2).sum() / sst),
            "r2_cv_top10_dECEI": float(1 - ((r - rhat)[top] ** 2).sum() / ((r[top] - r[top].mean()) ** 2).sum()),
            "r2_cv_rest": float(1 - ((r - rhat)[~top] ** 2).sum() / ((r[~top] - r[~top].mean()) ** 2).sum()),
            "skill_vs_pchip_eq_plus_fit": sk_fix,
            "gap_closed": float((sk_fix - sk_eq) / (sk_bb - sk_eq)) if sk_bb != sk_eq else None}
        if name == "fir_lags0_11":
            wk = np.mean(ws, 0).reshape(K, N_FAST)       # eV per z, lag x channel
            res["fir_group_kernel_from_fit"] = {g: (wk[:, cols].sum(1)).tolist() for g, cols in GROUPS.items()}
        if name == "hp_mean10":
            wk = np.mean(ws, 0)
            res["hp_mean10_coef_eV_per_sigma"] = {g: float(wk[cols].sum()) for g, cols in GROUPS.items()}
    res["minutes"] = round((time.time() - t0) / 60, 1)
    out_json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    fb = max(res["families"], key=lambda k: res["families"][k]["gap_closed"] or -9)
    print(f"[b28-3 resid2] {pop} s{seed}: gap closed hp_mean10={res['families']['hp_mean10']['gap_closed']:+.2f} "
          f"fir={res['families']['fir_lags0_11']['gap_closed']:+.2f} fir+state={res['families']['fir+ewma_state']['gap_closed']:+.2f} "
          f"best={fb} ({res['minutes']} min)", flush=True)
    return res


# ====================================================================== part (ii): truncation
def ladder_from_npz(z, frozen, rungs):
    for key in ("CES_TI_shot", "CES_TI_dt_ms", "CES_TI_se_pchip"):
        if not np.array_equal(z[key], frozen[key], equal_nan=True):
            raise SystemExit(f"FATAL: truncation artifact does not reproduce the frozen {key}")
    se_full = z["CES_TI_se_model"].astype(np.float64)
    # The cut ladder (.reach_s*, GPU) reproduces the frozen se_model bit-identically; the inclusive
    # ladder is a CPU re-evaluation and differs in float32 round-off only. Rows are identical
    # (checked above); the full-block error difference is recorded, and must stay below 5 eV.
    d_err = float(np.max(np.abs(np.sqrt(se_full) - np.sqrt(frozen["CES_TI_se_model"].astype(np.float64)))))
    if d_err > 5.0:
        raise SystemExit(f"FATAL: truncation artifact's full-block errors differ from the frozen ones by {d_err:.2f} eV")
    refs = {r: z[f"CES_TI_se_{r}"].astype(np.float64) for r in REFS}
    avail = [k for k in rungs if f"CES_TI_se_model_ctx{k}" in z.files]
    out = {"identity_vs_frozen": "bit-identical" if d_err == 0.0 else f"rows identical, full-block max |d err| {d_err:.3f} eV",
           "rungs": avail, "n_rows": int(len(se_full)),
           "skill_full": {r: skill(se_full, refs[r]) for r in REFS}, "ladder": {}}
    for k in avail:
        se_k = z[f"CES_TI_se_model_ctx{k}"].astype(np.float64)
        out["ladder"][str(k)] = {"skill": {r: skill(se_k, refs[r]) for r in REFS},
                                 "deficit": {r: out["skill_full"][r] - skill(se_k, refs[r]) for r in REFS},
                                 "mse_excess_eV2": float((se_k - se_full).mean())}
    ks = [k for k in avail if k <= 10]
    out["fit"] = {}
    for r in REFS:
        out["fit"][f"deficit_vs_{r}_k_le10"] = loglin_rho(ks, [out["ladder"][str(k)]["deficit"][r] for k in ks])
        k235 = [k for k in avail if k in (2, 3, 5)]
        out["fit"][f"deficit_vs_{r}_k235"] = loglin_rho(k235, [out["ladder"][str(k)]["deficit"][r] for k in k235])
    out["fit"]["mse_excess_k_le10"] = loglin_rho(ks, [out["ladder"][str(k)]["mse_excess_eV2"] for k in ks])
    out["fit"]["mse_excess_k235"] = loglin_rho([k for k in avail if k in (2, 3, 5)],
                                               [out["ladder"][str(k)]["mse_excess_eV2"] for k in avail if k in (2, 3, 5)])
    return out


def trunc_cut(seed):
    src = DATA / f".b1_seqv2_s{seed}_i{seed}" / "comparison_errors_test.npz"
    reach = DATA / f".reach_s{seed}" / "comparison_errors_test.npz"
    if not reach.exists():
        raise SystemExit(f"FATAL: sec. 8ac artifact missing: {reach}")
    out = ladder_from_npz(np.load(reach), np.load(src), (1, 2, 3, 5, 10, 20, 50, 100, 300))
    out.update({"pop": "cut", "seed": seed, "source": reach.parent.name})
    return out


def trunc_incl(seed, smoke, resume):
    src = DATA / f".b5i_seqv2_s{seed}"
    out_dir = DATA / (f".b28trunc_incl_s{seed}" + ("_smoke" if smoke else ""))
    npz = out_dir / "comparison_errors_test.npz"
    if not (resume and npz.exists()):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "weights").mkdir(parents=True)
        shutil.copy2(src / "weights" / "seq_lstm.pth", out_dir / "weights" / "seq_lstm.pth")
        shutil.copy2(src / "metrics.json", out_dir / "metrics.json")
        env = os.environ.copy()
        for var in AMBIENT_VARS:
            env.pop(var, None)
        env.update(GATE_ENV)
        env.update({"CES_TI_SPIKE_CUT_EV": "0", "CES_SEED": str(seed),
                    "CES_SPLIT_DIR": str(DATA / f".b1_manifest_s{seed}"), "CES_OUTPUT_DIR": str(out_dir),
                    "CES_SPLIT_TAG": "test", "CES_CONTROL_METRICS": str(DATA / f".wsweep_hf_w2_s{seed}" / "metrics.json"),
                    "CES_SEQ_MODEL": "v2", "CES_PER_SHOT_NORM": "1", "CES_SEQ_DEVICE": "cpu",
                    "CES_SEQ_CONTEXTS": ",".join(str(k) for k in RUNGS_INCL)})
        if smoke:
            env["CES_SEQ_MAX_FILES"] = "40"
        t0 = time.time()
        ok = run_step([SEQ_DIR / "eval_seq.py"], env, out_dir / "run.log", "eval_seq(trunc incl)",
                      artifacts=(npz, out_dir / "comparison_metrics.json"))
        print(f"[b28-3 trunc] incl s{seed}: eval_seq {'ok' if ok else 'FAILED'} ({(time.time()-t0)/60:.1f} min)", flush=True)
        if not ok:
            return {"pop": "incl", "seed": seed, "status": "eval_failed"}
    out = ladder_from_npz(np.load(npz), np.load(src / "comparison_errors_test.npz"), RUNGS_INCL)
    out.update({"pop": "incl", "seed": seed, "source": out_dir.name})
    return out


# ====================================================================== part (iii): MC norms
MCNORM_DIR = DATA / ".b28_mcnorm_s42"


def mcnorm_train(smoke, resume):
    out_dir = MCNORM_DIR if not smoke else DATA / ".b28_mcnorm_s42_smoke"
    log_json = out_dir / "wih_norms.json"
    if resume and log_json.exists() and (out_dir / "weights" / "seq_lstm.pth").exists():
        print("[b28-3 mcnorm] training done already, skipping (--resume)", flush=True)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for var in AMBIENT_VARS:
        env.pop(var, None)
    env.update(GATE_ENV)
    env.update({"CES_SEED": "42", "CES_INIT_SEED": "42",
                "CES_SPLIT_DIR": str(DATA / ".b1_w2cut_split_s42"), "CES_OUTPUT_DIR": str(out_dir),
                "CES_SPLIT_TAG": "test", "CES_CONTROL_METRICS": str(DATA / ".b1_w2cut_s42" / "metrics.json"),
                "CES_SEQ_MODEL": "v2", "CES_PER_SHOT_NORM": "1", "CES_SEQ_DEVICE": "cuda",
                "CES_SEQ_LOG_WIH": "1"})
    if smoke:
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40", "CES_SEQ_DEVICE": "cpu"})
    t0 = time.time()
    ok = run_step([SEQ_DIR / "train_seq.py"], env, out_dir / "run.log", "train_seq(v2, W_ih log)",
                  artifacts=(out_dir / "weights" / "seq_lstm.pth", log_json))
    print(f"[b28-3 mcnorm] train {'ok' if ok else 'FAILED'} ({(time.time()-t0)/60:.1f} min)", flush=True)
    if not ok:
        raise SystemExit("FATAL: mcnorm training failed")
    return out_dir


def mcnorm_analyze(out_dir):
    log = json.loads((out_dir / "wih_norms.json").read_text(encoding="utf-8"))
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    steps = np.array([e["optimizer_steps"] for e in log], float)
    epochs = [e["epoch"] for e in log]
    cols = {"MC": GROUPS["MC"], "BES": GROUPS["BES"], "ECEI": GROUPS["ECEI"], "dt": [COL_DT],
            "carried": [COL_CAR], "stale": [COL_STALE], "has": [COL_HAS]}
    res = {"run": out_dir.name, "epochs": epochs, "optimizer_steps": steps.tolist(),
           "best_epoch": metrics.get("best_epoch"), "n_epochs_run": int(max(epochs)), "trajectory": {}, "gate_f": {}}
    for name, cc in cols.items():
        traj = np.array([np.mean([e["col_norm_total"][c] for c in cc]) for e in log])
        fgate = np.array([np.mean([e["col_norm_gate"]["f"][c] for c in cc]) for e in log])
        res["trajectory"][name] = traj.tolist()
        res["gate_f"][name] = fgate.tolist()
    # random-walk check: norm^2 - norm_0^2 linear in optimizer steps?
    rw = {}
    for name in ("MC", "BES", "ECEI"):
        tr = np.array(res["trajectory"][name])
        g2 = tr ** 2 - tr[0] ** 2
        if len(steps) >= 3:
            slope, icpt = np.polyfit(steps, g2, 1)
            fit = slope * steps + icpt
            sst = ((g2 - g2.mean()) ** 2).sum()
            r2 = float(1 - ((g2 - fit) ** 2).sum() / sst) if sst > 0 else None
            # saturation: ratio of growth in the second half to the first half (1 = linear, < 1 saturating)
            h = len(g2) // 2
            sat = float((g2[-1] - g2[h]) / (g2[h] - g2[0])) if (g2[h] - g2[0]) != 0 else None
            rw[name] = {"norm_init": float(tr[0]), "norm_final": float(tr[-1]), "norm_at_best": float(tr[res["best_epoch"]]) if res["best_epoch"] is not None and res["best_epoch"] < len(tr) else None,
                        "growth_sq_slope_per_step": float(slope), "r2_linear_in_steps": r2, "second_half_over_first_half_growth": sat,
                        "monotone_increasing": bool(np.all(np.diff(tr) >= -1e-6))}
    res["random_walk_check"] = rw
    if rw.get("MC") and rw.get("BES") and rw["BES"]["growth_sq_slope_per_step"] != 0:
        res["mc_over_bes_growth_slope"] = float(rw["MC"]["growth_sq_slope_per_step"] / rw["BES"]["growth_sq_slope_per_step"])
    # the frozen B.1 checkpoint of the same split/init, for reference
    frozen = DATA / ".b1_seqv2_s42_i42" / "weights" / "seq_lstm.pth"
    if frozen.exists():
        sd = torch.load(frozen, map_location="cpu")
        w = sd["lstm_ti.weight_ih_l0"]
        n = w.norm(dim=0).numpy()
        res["frozen_b1_s42_i42_col_norm"] = {name: float(np.mean(n[cc])) for name, cc in cols.items()}
    return res


# ====================================================================== driver
def merge(part, payload):
    cur = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    cur[part] = payload
    cur["updated"] = time.strftime("%Y-%m-%d %H:%M")
    OUT.write_text(json.dumps(cur, indent=1), encoding="utf-8")


def summarize_jac(recs):
    s = {}
    for pop in ("cut", "incl"):
        rs = [r for r in recs if r["pop"] == pop]
        if not rs:
            continue
        fam_names = list(rs[0]["residual"]["families"])
        s[pop] = {
            "n_splits": len(rs),
            "rho_amp_lag0_3": [r["amplitude"]["fit_lag0_3"]["rho"] for r in rs],
            "kappa_sech2_BES": [r["saturation"]["BES"]["sech2"]["kappa_sigma"] for r in rs],
            "kappa_sech2_ECEI": [r["saturation"]["ECEI"]["sech2"]["kappa_sigma"] for r in rs],
            "kappa_trained_eq": [r["eq_physics"]["kappa_sigma"] for r in rs],
            "sse_sech2_over_flat_BES": [r["saturation"]["BES"]["sech2"]["sse"] / r["saturation"]["BES"]["flat"]["sse"] for r in rs],
            "sse_sech2_over_flat_ECEI": [r["saturation"]["ECEI"]["sech2"]["sse"] / r["saturation"]["ECEI"]["flat"]["sse"] for r in rs],
            "state_r2_BES": [r["state_dependence"]["BES"]["r2"] for r in rs],
            "state_r2_ECEI": [r["state_dependence"]["ECEI"]["r2"] for r in rs],
            "gap_closed": {f: [r["residual"]["families"][f]["gap_closed"] for r in rs] for f in fam_names},
            "r2_cv": {f: [r["residual"]["families"][f]["r2_cv"] for r in rs] for f in fam_names},
            "increment_index": {g: [r["kernels"][g]["increment_index_minus_l1_over_l0"] for r in rs] for g in GROUPS},
        }
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["jac", "resid2", "trunc", "mcnorm", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--pops", default="cut,incl")
    a = ap.parse_args()
    seeds = tuple(int(s) for s in a.seeds.split(","))
    pops = tuple(a.pops.split(","))
    if a.smoke:
        seeds, pops = (42,), ("cut",)
    keep_awake(True)
    try:
        if a.part in ("jac", "all"):
            recs = [jac_split(pop, s, a.smoke, a.resume) for pop in pops for s in seeds]
            merge("jac" + ("_smoke" if a.smoke else ""), {"splits": recs, "summary": summarize_jac(recs)})
        if a.part in ("resid2", "all"):
            recs = [resid2_split(pop, s, a.smoke, a.resume) for pop in pops for s in seeds]
            fam_names = list(recs[0]["families"])
            summ = {pop: {f: [r["families"][f]["gap_closed"] for r in recs if r["pop"] == pop] for f in fam_names}
                    for pop in pops if any(r["pop"] == pop for r in recs)}
            merge("resid2" + ("_smoke" if a.smoke else ""), {"splits": recs, "summary": summ})
        if a.part in ("trunc", "all"):
            recs = []
            if "cut" in pops:
                recs += [trunc_cut(s) for s in seeds]
            if "incl" in pops:
                recs += [trunc_incl(s, a.smoke, a.resume) for s in seeds]
            summ = {}
            for pop in pops:
                rs = [r for r in recs if r["pop"] == pop and "fit" in r]
                if rs:
                    summ[pop] = {ref: [r["fit"][f"deficit_vs_{ref}_k_le10"]["rho"] for r in rs] for ref in REFS}
                    summ[pop]["mse_excess"] = [r["fit"]["mse_excess_k_le10"]["rho"] for r in rs]
                    summ[pop]["mse_excess_k235"] = [r["fit"]["mse_excess_k235"]["rho"] for r in rs]
            merge("trunc" + ("_smoke" if a.smoke else ""), {"splits": recs, "summary": summ})
            for r in recs:
                if "fit" in r:
                    print(f"[b28-3 trunc] {r['pop']} s{r['seed']}: rho(deficit vs gp_causal, k<=10)="
                          f"{r['fit']['deficit_vs_gp_causal_k_le10']['rho']} rho(mse excess)={r['fit']['mse_excess_k_le10']['rho']}", flush=True)
        if a.part in ("mcnorm", "all"):
            out_dir = mcnorm_train(a.smoke, a.resume)
            res = mcnorm_analyze(out_dir)
            merge("mcnorm" + ("_smoke" if a.smoke else ""), res)
            rw = res["random_walk_check"]
            print(f"[b28-3 mcnorm] MC norm {rw['MC']['norm_init']:.3f} -> {rw['MC']['norm_final']:.3f} "
                  f"(BES {rw['BES']['norm_init']:.3f} -> {rw['BES']['norm_final']:.3f}); "
                  f"MC/BES growth slope {res.get('mc_over_bes_growth_slope')}; linear R2 MC {rw['MC']['r2_linear_in_steps']}", flush=True)
    finally:
        keep_awake(False)
    print("[b28-3] done", flush=True)


if __name__ == "__main__":
    main()
