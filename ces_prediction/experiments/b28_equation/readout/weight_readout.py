"""Weight / response readout of the frozen seq_v2 T_i branch (CPU only, no training).

Usage (from the repo root):
    py <this file> [--quick]

Reads every frozen seq_v2 checkpoint (B.1 cut 4x4, B.5 included 4, B.27 cut folds),
computes weight-only quantities (W_ih column norms per gate, W_hh SVD, forget-gate
bias memory, head readout), then on REAL held-out test blocks: gate statistics,
input Jacobians by lag, impulse / step responses of the T_i output, and a linear
surrogate fitted on val rows and scored on test rows (persistence-relative skill).

Writes weight_readout.json next to this file. Console output is ASCII only.
Nothing under the repository is written.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # B.27c owns the GPU
os.environ.setdefault("CES_DROP_STUCK_TARGETS", "1")
import sys, json, math, time
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(4)
from scipy.stats import spearmanr

REPO = Path(r"C:\Users\lss\Documents\GitHub\ml-intern-revision")
DATA = REPO / "data"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "ces_prediction" / "experiments" / "seq"))
sys.path.insert(0, str(REPO / "ces_prediction"))
import seq_data                                   # noqa: E402
from model_seq_v2 import SeqCESLSTMv2             # noqa: E402

QUICK = "--quick" in sys.argv
SEEDS = (42, 1, 7, 123)
INITS = (42, 1, 7, 123)
H = 160
CH_NAMES = ["BES_0105", "BES_0106", "BES_0107", "BES_0205", "BES_0206", "BES_0207",
            "BES_0305", "BES_0306", "BES_0307",
            "ECEI_GT1103", "ECEI_GT1203", "ECEI_GT1303", "ECEI_GT1403",
            "MC1T03", "MC1T16",
            "dt", "TI_carried", "TI_stale", "TI_has", "VT_carried", "VT_stale", "VT_has"]
GROUPS = {"BES": list(range(0, 9)), "ECEI": list(range(9, 13)), "MC": [13, 14],
          "dt": [15], "TI_carried": [16], "TI_stale": [17], "TI_has": [18],
          "VT_carried": [19], "VT_stale": [20], "VT_has": [21]}
GATES = ["i", "f", "g", "o"]


def checkpoints():
    out = []
    for s in SEEDS:
        for i in INITS:
            d = DATA / f".b1_seqv2_s{s}_i{i}"
            if (d / "weights" / "seq_lstm.pth").exists():
                out.append(dict(tag=f"b1_s{s}_i{i}", pop="cut", seed=s, init=i, dir=d))
    for s in SEEDS:
        d = DATA / f".b5i_seqv2_s{s}"
        if (d / "weights" / "seq_lstm.pth").exists():
            out.append(dict(tag=f"b5i_s{s}", pop="incl", seed=s, init=None, dir=d))
    for r in (0, 1):
        for f in range(5):
            d = DATA / f".b27_seqv2_cut_r{r}_f{f}"
            if (d / "weights" / "seq_lstm.pth").exists():
                out.append(dict(tag=f"b27_cut_r{r}_f{f}", pop="cut", seed=None, init=None, dir=d))
    return out


# ------------------------------------------------------------------ weight-only readouts
def weight_readout(sd):
    r = {}
    W = sd["lstm_ti.weight_ih_l0"].numpy()                      # (640, 22)
    r["col_norm_all"] = np.linalg.norm(W, axis=0).tolist()
    r["col_norm_gate"] = {g: np.linalg.norm(W[k * H:(k + 1) * H], axis=0).tolist()
                          for k, g in enumerate(GATES)}
    r["group_share"] = {}
    tot = float((W ** 2).sum())
    for g, idx in GROUPS.items():
        r["group_share"][g] = float((W[:, idx] ** 2).sum() / tot)
    # W_hh spectra
    for layer in (0, 1):
        Whh = sd[f"lstm_ti.weight_hh_l{layer}"].numpy()
        sv = np.linalg.svd(Whh, compute_uv=False)
        r[f"whh_l{layer}_sv_top5"] = sv[:5].tolist()
        r[f"whh_l{layer}_stable_rank"] = float((sv ** 2).sum() / sv[0] ** 2)
        r[f"whh_l{layer}_gate_specnorm"] = {g: float(np.linalg.svd(Whh[k * H:(k + 1) * H], compute_uv=False)[0])
                                            for k, g in enumerate(GATES)}
        bf = (sd[f"lstm_ti.bias_ih_l{layer}"] + sd[f"lstm_ti.bias_hh_l{layer}"]).numpy()[H:2 * H]
        f0 = 1.0 / (1.0 + np.exp(-bf))
        tau = -1.0 / np.log(np.clip(f0, 1e-6, 1 - 1e-6))     # steps (10 ms each)
        r[f"forget_bias_l{layer}"] = dict(f0_median=float(np.median(f0)), f0_p90=float(np.percentile(f0, 90)),
                                          tau_steps_median=float(np.median(tau)),
                                          tau_steps_p90=float(np.percentile(tau, 90)),
                                          n_units_tau_gt5=int((tau > 5).sum()),
                                          n_units_tau_gt20=int((tau > 20).sum()))
        Wih1 = sd[f"lstm_ti.weight_ih_l{layer}"].numpy()
        r[f"wih_l{layer}_fro"] = float(np.linalg.norm(Wih1))
        r[f"whh_l{layer}_fro"] = float(np.linalg.norm(Whh))
    # head
    h0 = sd["head_ti.0.weight"].numpy()                        # (64,160)
    h2 = sd["head_ti.2.weight"].numpy()[0]                     # (64,)
    r["head_hidden_colnorm"] = np.linalg.norm(h0, axis=0).tolist()
    r["head_hidden_top10"] = np.argsort(-np.linalg.norm(h0, axis=0))[:10].tolist()
    r["head_bias_out"] = float(sd["head_ti.2.bias"].numpy()[0])
    r["head_lin_gain_at_zero"] = (h2 * 0.5) @ h0               # GELU'(0)=0.5 -> linearized readout vector (160,)
    r["head_lin_gain_at_zero"] = r["head_lin_gain_at_zero"].tolist()
    # V_rot branch for contrast
    Wv = sd["lstm_vt.weight_ih_l0"].numpy()
    r["vt_col_norm"] = np.linalg.norm(Wv, axis=0).tolist()      # 7 slow channels
    return r


# ------------------------------------------------------------------ manual LSTM (gate access)
def lstm_layer(x, sd, prefix, layer):
    """x (L, n_in) -> h (L,H), gates dict of (L,H). Same math as nn.LSTM (order i,f,g,o)."""
    Wih = sd[f"{prefix}.weight_ih_l{layer}"]
    Whh = sd[f"{prefix}.weight_hh_l{layer}"]
    b = sd[f"{prefix}.bias_ih_l{layer}"] + sd[f"{prefix}.bias_hh_l{layer}"]
    L = x.shape[0]
    h = torch.zeros(H if prefix == "lstm_ti" else 64)
    c = torch.zeros_like(h)
    hs, fs, is_, os_ = [], [], [], []
    xw = x @ Wih.T + b
    for t in range(L):
        g = xw[t] + h @ Whh.T
        i, f, gg, o = g.chunk(4)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(gg)
        h = o * torch.tanh(c)
        hs.append(h); fs.append(f); is_.append(i); os_.append(o)
    return torch.stack(hs), dict(f=torch.stack(fs), i=torch.stack(is_), o=torch.stack(os_))


def ti_forward_manual(x, sd):
    h0, g0 = lstm_layer(x, sd, "lstm_ti", 0)
    h1, g1 = lstm_layer(h0, sd, "lstm_ti", 1)
    hn = torch.nn.functional.layer_norm(h1, (H,), sd["norm_ti.weight"], sd["norm_ti.bias"])
    z = hn @ sd["head_ti.0.weight"].T + sd["head_ti.0.bias"]
    y = torch.nn.functional.gelu(z) @ sd["head_ti.2.weight"].T + sd["head_ti.2.bias"]
    return y[:, 0], (g0, g1), hn


# ------------------------------------------------------------------ data
_GRID = {}


def grid_for(pop):
    if pop not in _GRID:
        cut = 3000.0 if pop == "cut" else 0.0
        t0 = time.time()
        g, dims = seq_data.load_grid_files(DATA, drop_stuck_targets=True, ti_spike_cut_ev=cut)
        print(f"[data] pop={pop} files={len(g)} ({time.time()-t0:.0f}s)")
        _GRID[pop] = (g, dims)
    return _GRID[pop]


def split_files(ck):
    if ck["seed"] is not None:
        m = json.loads((DATA / f".wsweep_hf_w2_s{ck['seed']}" / "metrics.json").read_text(encoding="utf-8"))["split"]
        return m["val_files"], m["test_files"]
    m = json.loads((ck["dir"] / "metrics.json").read_text(encoding="utf-8"))
    sp = m.get("split") or {}
    return sp.get("val_files", []), sp.get("test_files", [])


def blocks_for(ck, names, stats, n_files, min_len=60):
    grid, dims = grid_for(ck["pop"])
    rng = np.random.RandomState(0)
    names = [n for n in names if n in grid]
    names = list(rng.choice(names, size=min(n_files, len(names)), replace=False))
    out = []
    for n in sorted(names):
        for b in seq_data.build_blocks(grid[n], dims, stats, per_shot_norm=True):
            if b["x"].shape[0] >= min_len:
                b["file"] = n
                out.append(b)
    return out


# ------------------------------------------------------------------ response readouts
def response_readout(ck, sd, model, stats):
    val_names, test_names = split_files(ck)
    nf = 6 if QUICK else 24
    test_blocks = blocks_for(ck, test_names, stats, nf)
    val_blocks = blocks_for(ck, val_names, stats, nf)
    if not test_blocks:
        return {"note": "no test blocks"}
    r = {"n_test_blocks": len(test_blocks), "n_val_blocks": len(val_blocks)}
    ti_std = float(stats["target"]["std"][0])

    # 0. manual LSTM == nn.LSTM check on one block
    xb = torch.from_numpy(test_blocks[0]["x"])
    with torch.no_grad():
        y_ref = model(xb[None])[0, :, 0]
        y_man, _, _ = ti_forward_manual(xb, sd)
    r["manual_vs_module_maxabs"] = float((y_ref - y_man).abs().max())

    # 1. input channel std on real rows (scale for the column norms)
    X = np.concatenate([b["x"] for b in test_blocks])
    r["input_std"] = X.std(axis=0).tolist()
    r["input_mean"] = X.mean(axis=0).tolist()

    # 2. gate statistics (geometric-mean forget gate -> effective memory)
    f_log = {0: [], 1: []}
    with torch.no_grad():
        for b in test_blocks[:12]:
            _, (g0, g1), _ = ti_forward_manual(torch.from_numpy(b["x"]), sd)
            f_log[0].append(torch.log(g0["f"].clamp_min(1e-6)).mean(0))
            f_log[1].append(torch.log(g1["f"].clamp_min(1e-6)).mean(0))
    for layer in (0, 1):
        lf = torch.stack(f_log[layer]).mean(0).numpy()      # per-unit mean log f
        tau = -1.0 / lf
        r[f"forget_data_l{layer}"] = dict(f_geo_median=float(np.exp(np.median(lf))),
                                          tau_steps_median=float(np.median(tau)),
                                          tau_steps_p90=float(np.percentile(tau, 90)),
                                          tau_steps_max=float(tau.max()),
                                          n_units_tau_gt5=int((tau > 5).sum()),
                                          n_units_tau_gt20=int((tau > 20).sum()))

    # 3. Jacobian of y_TI(t) wrt x(t-k, c), k = 0..LAGS-1, on observed-T_i rows
    LAGS, CTX = 12, 100
    rows = []
    rng = np.random.RandomState(1)
    per_block = 4 if QUICK else 10
    for bi, b in enumerate(test_blocks):
        obs = np.where((b["mask"][:, 0] > 0) & (np.arange(b["x"].shape[0]) >= CTX))[0]
        if len(obs) == 0:
            continue
        for t in rng.choice(obs, size=min(per_block, len(obs)), replace=False):
            rows.append((bi, int(t)))
    J = np.zeros((len(rows), LAGS, 22))
    stale_s = np.zeros(len(rows)); ev_e = np.zeros(len(rows)); ev_b = np.zeros(len(rows))
    K_anchor = np.zeros(len(rows))
    for n, (bi, t) in enumerate(rows):
        xb_ = test_blocks[bi]["x"]
        xw = torch.from_numpy(xb_[t - CTX:t + 1]).clone().requires_grad_(True)
        y = model(xw[None])[0, -1, 0]
        y.backward()
        gfull = xw.grad.numpy()[::-1]                      # lag 0 first, CTX+1 lags
        J[n] = gfull[:LAGS]
        stale_s[n] = math.expm1(float(xb_[t, 17]))         # seconds since last T_i obs
        ev_e[n] = np.abs(xb_[t, 9:13] - xb_[t - 1, 9:13]).mean()
        ev_b[n] = np.abs(xb_[t, 0:9] - xb_[t - 1, 0:9]).mean()
        # gain on the LAST observation = sum of d y / d carried over the run where carried is constant
        run = int(round(stale_s[n] / 0.01))
        run = max(1, min(run, CTX))
        K_anchor[n] = gfull[:run, 16].sum()
    r["n_jacobian_rows"] = len(rows)
    # effective |grad| share: |grad| x channel std (constant channels like dt/has vanish)
    xstd = np.asarray(r["input_std"])
    Jeff = np.abs(J) * xstd[None, None, :]
    Je = Jeff.mean(0); totE = Je.sum()
    r["jac_eff_group_share_alllag"] = {g: float(Je[:, idx].sum() / totE) for g, idx in GROUPS.items()}
    r["jac_eff_group_share_lag0"] = {g: float(Je[0, idx].sum() / Je[0].sum()) for g, idx in GROUPS.items()}
    # anchor gain K(s) by staleness bin (seconds)
    bins = [(0.0, 0.015), (0.015, 0.055), (0.055, 0.205), (0.205, 10.0)]
    r["K_anchor_by_stale"] = []
    for lo, hi in bins:
        m_ = (stale_s >= lo) & (stale_s < hi)
        if m_.sum() >= 3:
            r["K_anchor_by_stale"].append(dict(lo=lo, hi=hi, n=int(m_.sum()), K_mean=float(K_anchor[m_].mean()),
                                               K_sd=float(K_anchor[m_].std()),
                                               ecei_lag0=float(J[m_][:, 0, 9:13].sum(1).mean()),
                                               bes_lag0=float(J[m_][:, 0, 0:9].sum(1).mean())))
    # event-conditional sensitivity: top decile of |dECEI| / |dBES| vs rest
    r["event_conditional"] = {}
    for nm, ev, idx in (("ECEI", ev_e, list(range(9, 13))), ("BES", ev_b, list(range(0, 9)))):
        thr = np.percentile(ev, 90)
        top = ev >= thr
        g_top = J[top][:, 0, idx].sum(1); g_rest = J[~top][:, 0, idx].sum(1)
        r["event_conditional"][nm] = dict(thr=float(thr), n_top=int(top.sum()),
                                          grad_lag0_top=float(g_top.mean()), grad_lag0_rest=float(g_rest.mean()),
                                          absgrad_lag0_top=float(np.abs(g_top).mean()), absgrad_lag0_rest=float(np.abs(g_rest).mean()))
    Jm = J.mean(0)                                        # signed mean (LAGS,22)
    Ja = np.abs(J).mean(0)
    r["jac_signed_mean"] = Jm.tolist()
    r["jac_abs_mean"] = Ja.tolist()
    # group shares of |grad| at lag 0 and summed over lags
    tot0 = Ja[0].sum(); totL = Ja.sum()
    r["jac_group_share_lag0"] = {g: float(Ja[0, idx].sum() / tot0) for g, idx in GROUPS.items()}
    r["jac_group_share_alllag"] = {g: float(Ja[:, idx].sum() / totL) for g, idx in GROUPS.items()}
    # increment index per fast group: -(sum_{k>=1} g_k)/g_0 using signed means (1 => pure increment)
    r["increment_index"] = {}
    for g in ("BES", "ECEI", "MC"):
        idx = GROUPS[g]
        g0 = Jm[0, idx].sum(); gl = Jm[1:, idx].sum()
        r["increment_index"][g] = float(-gl / g0) if abs(g0) > 1e-9 else None
        r[f"jac_signed_{g}_by_lag"] = Jm[:, idx].sum(1).tolist()
    r["jac_signed_TI_carried_by_lag"] = Jm[:, 16].tolist()
    r["jac_signed_TI_stale_by_lag"] = Jm[:, 17].tolist()
    # per-channel ranking at lag 0 (abs) -- for cross-checkpoint Spearman
    r["jac_abs_lag0_per_channel"] = Ja[0].tolist()
    r["jac_abs_sumlag_per_channel"] = Ja.sum(0).tolist()

    # 4. impulse and step responses (+1 z-unit on a channel group at t0, real context)
    HOR = 30
    resp = {}
    with torch.no_grad():
        for g in ("BES", "ECEI", "MC", "TI_carried"):
            idx = GROUPS[g]
            imp, step = [], []
            for b in test_blocks[:16]:
                x = torch.from_numpy(b["x"])
                L = x.shape[0]
                if L < CTX + HOR + 2:
                    continue
                t0 = CTX
                base = model(x[None])[0, :, 0]
                xi = x.clone(); xi[t0, idx] += 1.0
                xs = x.clone(); xs[t0:, idx] += 1.0
                yi = model(xi[None])[0, :, 0]; ys = model(xs[None])[0, :, 0]
                imp.append((yi - base)[t0:t0 + HOR].numpy())
                step.append((ys - base)[t0:t0 + HOR].numpy())
            if not imp:
                continue
            imp = np.array(imp); step = np.array(step)
            im = imp.mean(0); sm = step.mean(0)
            # recovery time: first lag where |imp| falls below 1/e of |imp[0]|
            a0 = im[0]
            tau_e = None
            for k in range(1, HOR):
                if abs(im[k]) < abs(a0) / math.e:
                    tau_e = k; break
            asym = sm[-5:].mean(); tau63 = None
            for k in range(HOR):
                if abs(sm[k]) >= 0.63 * abs(asym):
                    tau63 = k; break
            resp[g] = dict(step_tau63_steps=tau63, impulse_sum_norm=float(im.sum()),
                           impulse_mean=im.tolist(), impulse_sd=imp.std(0).tolist(),
                           step_mean=sm.tolist(), A0_norm=float(a0), A0_eV=float(a0 * ti_std),
                           tau_e_steps=tau_e, step_asymptote_norm=float(sm[-5:].mean()),
                           step_asymptote_eV=float(sm[-5:].mean() * ti_std),
                           n=int(len(imp)))
    r["response"] = resp

    # 5. linear surrogate: fit on val rows, score on test rows (persistence-relative)
    def design(blocks):
        Xs, Ym, Yt, P = [], [], [], []
        for b in blocks:
            x = b["x"]; L = x.shape[0]
            with torch.no_grad():
                ym = model(torch.from_numpy(x)[None])[0, :, 0].numpy()
            obs = np.where((b["mask"][:, 0] > 0) & (x[:, 18] > 0) & (np.arange(L) >= 2))[0]
            if len(obs) == 0:
                continue
            fast = x[:, :15]
            d1 = np.zeros_like(fast); d1[1:] = fast[1:] - fast[:-1]
            feats = np.concatenate([x[obs, 16:17],                # TI carried
                                    x[obs, 17:18],                # TI stale
                                    fast[obs],                    # level, 15
                                    d1[obs],                      # 10 ms increment, 15
                                    np.ones((len(obs), 1))], 1)
            Xs.append(feats); Ym.append(ym[obs]); Yt.append(b["y"][obs, 0]); P.append(x[obs, 16])
        return np.concatenate(Xs), np.concatenate(Ym), np.concatenate(Yt), np.concatenate(P)
    Xv, Ymv, Ytv, Pv = design(val_blocks)
    Xt, Ymt, Ytt, Pt = design(test_blocks)
    sur = {}
    # residual form: pred = carried + A beta ; persistence is beta = 0 exactly. Ridge lam=1e-2*n.
    def resid_fit(cols, Av, yv):
        A = Av[:, cols]; lam = 1e-2 * len(yv)
        return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yv)
    stale_v = Xv[:, 1]; stale_t = Xt[:, 1]
    inc_e_v = np.abs(Xv[:, 17 + 9:17 + 13]).mean(1); inc_e_t = np.abs(Xt[:, 17 + 9:17 + 13]).mean(1)
    def gated(X, stale, inc):
        fresh = (stale < np.log1p(0.015)).astype(float)[:, None]
        lvl = X[:, 2:17]; inc_ = X[:, 17:32]
        return np.concatenate([X[:, 1:2], 1 - fresh, inc_, inc_ * (1 - fresh), lvl * (1 - fresh),
                               np.abs(inc_), X[:, 32:33]], 1)
    Gv = gated(Xv, stale_v, inc_e_v); Gt = gated(Xt, stale_t, inc_e_t)
    rspecs = {"res_inc_only": list(range(2, 17)), "res_stale+inc": [0, 1] + list(range(2, 17)),
              "res_stale+inc+gatedinc": [0, 1] + list(range(2, 32)),
              "res_full_gated": list(range(Gv.shape[1]))}
    mse_p_t = float(((Ytt - Pt) ** 2).mean())
    top_t = inc_e_t >= np.percentile(inc_e_t, 90)
    for name, cols in rspecs.items():
        for target_name, yv in (("model", Ymv - Pv), ("label", Ytv - Pv)):
            w = resid_fit(cols, Gv, yv)
            pred = Pt + Gt[:, cols] @ w
            e2 = (Ytt - pred) ** 2; e2p = (Ytt - Pt) ** 2
            sur[f"{name}|fit_{target_name}"] = dict(
                r2_vs_model_test=float(1 - ((Ymt - pred) ** 2).mean() / Ymt.var()),
                skill_vs_persistence=float(1 - e2.mean() / mse_p_t),
                skill_top10_inc=float(1 - e2[top_t].mean() / e2p[top_t].mean()),
                skill_rest=float(1 - e2[~top_t].mean() / e2p[~top_t].mean()),
                coef=w.tolist(), cols=cols)
    sur["model_skill_top10_inc"] = float(1 - ((Ytt - Ymt) ** 2)[top_t].mean() / ((Ytt - Pt) ** 2)[top_t].mean())
    sur["model_skill_rest"] = float(1 - ((Ytt - Ymt) ** 2)[~top_t].mean() / ((Ytt - Pt) ** 2)[~top_t].mean())
    sur["share_persistence_mse_in_top10"] = float(((Ytt - Pt) ** 2)[top_t].sum() / ((Ytt - Pt) ** 2).sum())
    specs = {"persist_only": [0, 32], "anchor+stale": [0, 1, 32],
             "anchor+ECEI_level": [0, 1] + [2 + i for i in range(9, 13)] + [32],
             "anchor+ECEI_level+inc": [0, 1] + [2 + i for i in range(9, 13)] + [17 + i for i in range(9, 13)] + [32],
             "anchor+all_level": [0, 1] + list(range(2, 17)) + [32],
             "anchor+all_level+inc": list(range(33))}
    for name, cols in specs.items():
        A = Xv[:, cols]
        for target_name, yv in (("model", Ymv), ("label", Ytv)):
            w, *_ = np.linalg.lstsq(A, yv, rcond=None)
            pred = Xt[:, cols] @ w
            mse_p = float(((Ytt - Pt) ** 2).mean())
            sur[f"{name}|fit_{target_name}"] = dict(
                r2_vs_model_test=float(1 - ((Ymt - pred) ** 2).mean() / Ymt.var()),
                skill_vs_persistence=float(1 - ((Ytt - pred) ** 2).mean() / mse_p),
                coef=w.tolist(), cols=cols)
    sur["model_skill_vs_persistence_test"] = float(1 - ((Ytt - Ymt) ** 2).mean() / ((Ytt - Pt) ** 2).mean())
    sur["n_val_rows"] = int(len(Ymv)); sur["n_test_rows"] = int(len(Ymt))
    r["surrogate"] = sur
    return r


# ------------------------------------------------------------------ main
def main():
    cks = checkpoints()
    print(f"[ck] {len(cks)} checkpoints: " + ", ".join(c["tag"] for c in cks))
    results = {}
    for ck in cks:
        sd = torch.load(ck["dir"] / "weights" / "seq_lstm.pth", map_location="cpu")
        res = {"pop": ck["pop"], "seed": ck["seed"], "init": ck["init"], "weights": weight_readout(sd)}
        do_resp = ck["tag"].startswith("b1_") or ck["tag"].startswith("b5i_")
        if QUICK:
            do_resp = ck["tag"] in ("b1_s42_i42", "b5i_s42")
        if do_resp:
            m = json.loads((ck["dir"] / "metrics.json").read_text(encoding="utf-8"))
            stats = {g: {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}
                     for g, d in m["normalization"]["stats"].items()}
            model = SeqCESLSTMv2(n_extra_vt=0, n_extra_ti=0)
            model.load_state_dict(sd); model.eval()
            t0 = time.time()
            res["response"] = response_readout(ck, sd, model, stats)
            print(f"[resp] {ck['tag']} done ({time.time()-t0:.0f}s) "
                  f"jac_rows={res['response'].get('n_jacobian_rows')} "
                  f"surrogate_model_skill={res['response']['surrogate']['model_skill_vs_persistence_test']:.3f}")
        results[ck["tag"]] = res
        w = res["weights"]
        print(f"[w] {ck['tag']}: share BES={w['group_share']['BES']:.3f} ECEI={w['group_share']['ECEI']:.3f} "
              f"MC={w['group_share']['MC']:.3f} TIcar={w['group_share']['TI_carried']:.3f} "
              f"whh0_sv1={w['whh_l0_sv_top5'][0]:.2f} tau_bias_med={w['forget_bias_l0']['tau_steps_median']:.1f}")

    # ---- cross-checkpoint agreement
    agg = {}
    tags = list(results)
    cut_tags = [t for t in tags if results[t]["pop"] == "cut"]
    inc_tags = [t for t in tags if results[t]["pop"] == "incl"]

    def spearman_matrix(key_fn, tags, idx):
        vals = [np.asarray(key_fn(results[t]))[idx] for t in tags]
        n = len(vals); rho = np.full((n, n), np.nan)
        for a in range(n):
            for b in range(n):
                rho[a, b] = spearmanr(vals[a], vals[b]).correlation if len(idx) > 2 else np.nan
        off = rho[~np.eye(n, dtype=bool)]
        return dict(mean=float(np.nanmean(off)), min=float(np.nanmin(off)), n=n)
    fast_idx = list(range(15)); bes_idx = list(range(9))
    agg["spearman_colnorm_fast_cut"] = spearman_matrix(lambda r: r["weights"]["col_norm_all"], cut_tags, fast_idx)
    agg["spearman_colnorm_bes_cut"] = spearman_matrix(lambda r: r["weights"]["col_norm_all"], cut_tags, bes_idx)
    agg["spearman_colnorm_all22_cut"] = spearman_matrix(lambda r: r["weights"]["col_norm_all"], cut_tags, list(range(22)))
    resp_tags = [t for t in tags if "response" in results[t] and "jac_abs_lag0_per_channel" in results[t]["response"]]
    if len(resp_tags) > 1:
        agg["spearman_jac_lag0_fast"] = spearman_matrix(lambda r: r["response"]["jac_abs_lag0_per_channel"], resp_tags, fast_idx)
        agg["spearman_jac_sumlag_fast"] = spearman_matrix(lambda r: r["response"]["jac_abs_sumlag_per_channel"], resp_tags, fast_idx)
        agg["spearman_jac_sumlag_bes"] = spearman_matrix(lambda r: r["response"]["jac_abs_sumlag_per_channel"], resp_tags, bes_idx)
    # mean channel ranks
    def mean_rank(key_fn, tags, idx):
        ranks = []
        for t in tags:
            v = np.asarray(key_fn(results[t]))[idx]
            ranks.append(np.argsort(np.argsort(-v)) + 1)
        return np.mean(ranks, 0).tolist()
    agg["mean_rank_colnorm_fast_cut"] = dict(zip(CH_NAMES[:15], mean_rank(lambda r: r["weights"]["col_norm_all"], cut_tags, fast_idx)))
    if resp_tags:
        agg["mean_rank_jac_sumlag_fast"] = dict(zip(CH_NAMES[:15], mean_rank(lambda r: r["response"]["jac_abs_sumlag_per_channel"], resp_tags, fast_idx)))
    # group shares summary
    for pop, tg in (("cut", cut_tags), ("incl", inc_tags)):
        if not tg:
            continue
        gs = {g: [results[t]["weights"]["group_share"][g] for t in tg] for g in GROUPS}
        agg[f"group_share_{pop}"] = {g: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for g, v in gs.items()}
        rt = [t for t in tg if t in resp_tags]
        if rt:
            js = {g: [results[t]["response"]["jac_group_share_alllag"][g] for t in rt] for g in GROUPS}
            agg[f"jac_group_share_alllag_{pop}"] = {g: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for g, v in js.items()}
            ii = {g: [results[t]["response"]["increment_index"][g] for t in rt] for g in ("BES", "ECEI", "MC")}
            agg[f"increment_index_{pop}"] = {g: dict(mean=float(np.mean([x for x in v if x is not None])), vals=v) for g, v in ii.items()}
            agg[f"tau_e_{pop}"] = {g: [results[t]["response"]["response"].get(g, {}).get("tau_e_steps") for t in rt] for g in ("BES", "ECEI", "MC", "TI_carried")}
            agg[f"A0_eV_{pop}"] = {g: [results[t]["response"]["response"].get(g, {}).get("A0_eV") for t in rt] for g in ("BES", "ECEI", "MC", "TI_carried")}
            agg[f"step_asym_eV_{pop}"] = {g: [results[t]["response"]["response"].get(g, {}).get("step_asymptote_eV") for t in rt] for g in ("BES", "ECEI", "MC", "TI_carried")}
            agg[f"jac_eff_group_share_alllag_{pop}"] = {g: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for g, v in
                                                       {g: [results[t]["response"]["jac_eff_group_share_alllag"][g] for t in rt] for g in GROUPS}.items()}
            agg[f"K_anchor_by_stale_{pop}"] = [results[t]["response"]["K_anchor_by_stale"] for t in rt]
            agg[f"event_conditional_{pop}"] = [results[t]["response"]["event_conditional"] for t in rt]
            agg[f"step_tau63_{pop}"] = {g: [results[t]["response"]["response"].get(g, {}).get("step_tau63_steps") for t in rt] for g in ("BES", "ECEI", "MC", "TI_carried")}
            agg[f"impulse_sum_{pop}"] = {g: [results[t]["response"]["response"].get(g, {}).get("impulse_sum_norm") for t in rt] for g in ("BES", "ECEI", "MC", "TI_carried")}
            agg[f"forget_data_l0_{pop}"] = [results[t]["response"]["forget_data_l0"] for t in rt]
            agg[f"forget_data_l1_{pop}"] = [results[t]["response"]["forget_data_l1"] for t in rt]
            sk = {}
            for name in results[rt[0]]["response"]["surrogate"]:
                if "|" in name:
                    sk[name] = dict(skill=[results[t]["response"]["surrogate"][name]["skill_vs_persistence"] for t in rt],
                                    r2=[results[t]["response"]["surrogate"][name]["r2_vs_model_test"] for t in rt])
            sk["model"] = [results[t]["response"]["surrogate"]["model_skill_vs_persistence_test"] for t in rt]
            sk["model_top10"] = [results[t]["response"]["surrogate"]["model_skill_top10_inc"] for t in rt]
            sk["model_rest"] = [results[t]["response"]["surrogate"]["model_skill_rest"] for t in rt]
            sk["share_pers_mse_top10"] = [results[t]["response"]["surrogate"]["share_persistence_mse_in_top10"] for t in rt]
            for name in list(sk):
                if "|" in name and "res_" in name:
                    sk[name]["top10"] = [results[t]["response"]["surrogate"][name]["skill_top10_inc"] for t in rt]
                    sk[name]["rest"] = [results[t]["response"]["surrogate"][name]["skill_rest"] for t in rt]
            agg[f"surrogate_{pop}"] = sk
    results["_aggregate"] = agg
    results["_channels"] = CH_NAMES
    out = HERE / ("weight_readout_quick.json" if QUICK else "weight_readout.json")
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"[done] wrote {out}")
    print(json.dumps({k: v for k, v in agg.items() if k.startswith("spearman") or k.startswith("increment") or k.startswith("tau_e")}, indent=1))


if __name__ == "__main__":
    main()
