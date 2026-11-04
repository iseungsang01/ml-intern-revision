"""Second pass (CPU, no training): EWMA-of-observations surrogate, amplitude linearity of
the fast-channel response, forget-gate behaviour on event vs quiet steps.
Writes weight_readout2.json. Run from the repo root: py <this file>"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, math, time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import weight_readout as W                      # noqa: E402
from model_seq_v2 import SeqCESLSTMv2           # noqa: E402

ALPHAS = (0.25, 0.35, 0.45, 0.6, 0.8)


def ewma_features(b):
    """Causal EWMA over OBSERVED T_i values (z units), one column per alpha, plus n_obs so far."""
    x, mask = b["x"], b["mask"][:, 0]
    L = x.shape[0]
    out = np.zeros((L, len(ALPHAS)))
    nobs = np.zeros(L)
    e = [None] * len(ALPHAS); n = 0
    for i in range(L):
        for j in range(len(ALPHAS)):
            out[i, j] = 0.0 if e[j] is None else e[j]
        nobs[i] = n
        if mask[i] > 0:
            v = b["y"][i, 0]
            for j, a in enumerate(ALPHAS):
                e[j] = v if e[j] is None else a * v + (1 - a) * e[j]
            n += 1
    return out, nobs


def design(blocks, model):
    F, Ym, Yt, P = [], [], [], []
    for b in blocks:
        x = b["x"]; L = x.shape[0]
        with torch.no_grad():
            ym = model(torch.from_numpy(x)[None])[0, :, 0].numpy()
        ew, nobs = ewma_features(b)
        obs = np.where((b["mask"][:, 0] > 0) & (nobs >= 3) & (np.arange(L) >= 3))[0]
        if len(obs) == 0:
            continue
        fast = x[:, :15]
        d1 = np.zeros_like(fast); d1[1:] = fast[1:] - fast[:-1]
        feats = np.concatenate([ew[obs], x[obs, 17:18], fast[obs], d1[obs], np.ones((len(obs), 1))], 1)
        F.append(feats); Ym.append(ym[obs]); Yt.append(b["y"][obs, 0]); P.append(x[obs, 16])
    return np.concatenate(F), np.concatenate(Ym), np.concatenate(Yt), np.concatenate(P)


def surrogates(val_blocks, test_blocks, model):
    Fv, Ymv, Ytv, Pv = design(val_blocks, model)
    Ft, Ymt, Ytt, Pt = design(test_blocks, model)
    nA = len(ALPHAS)
    col = dict(stale=nA, lvl=list(range(nA + 1, nA + 16)), inc=list(range(nA + 16, nA + 31)), one=nA + 31)
    inc_e_t = np.abs(Ft[:, [nA + 16 + i for i in range(9, 13)]]).mean(1)
    top = inc_e_t >= np.percentile(inc_e_t, 90)
    mse_p = ((Ytt - Pt) ** 2).mean()
    res = {"n_val": int(len(Ymv)), "n_test": int(len(Ymt)),
           "model_skill": float(1 - ((Ytt - Ymt) ** 2).mean() / mse_p),
           "model_skill_top10": float(1 - ((Ytt - Ymt) ** 2)[top].mean() / ((Ytt - Pt) ** 2)[top].mean()),
           "model_skill_rest": float(1 - ((Ytt - Ymt) ** 2)[~top].mean() / ((Ytt - Pt) ** 2)[~top].mean()),
           "model_r2_vs_label": float(1 - ((Ytt - Ymt) ** 2).mean() / Ytt.var())}
    # (a) fixed-form EWMA, no fitting at all
    for j, a in enumerate(ALPHAS):
        pred = Ft[:, j]
        res[f"ewma_fixed_a{a}"] = dict(skill=float(1 - ((Ytt - pred) ** 2).mean() / mse_p),
                                       r2_vs_model=float(1 - ((Ymt - pred) ** 2).mean() / Ymt.var()),
                                       skill_top10=float(1 - ((Ytt - pred) ** 2)[top].mean() / ((Ytt - Pt) ** 2)[top].mean()),
                                       skill_rest=float(1 - ((Ytt - pred) ** 2)[~top].mean() / ((Ytt - Pt) ** 2)[~top].mean()))
    # (b) fitted: residual over the best fixed EWMA; ridge on val, scored on test
    best_a = max(ALPHAS, key=lambda a: res[f"ewma_fixed_a{a}"]["skill"])
    jb = ALPHAS.index(best_a); res["best_alpha"] = best_a
    base_v, base_t = Fv[:, jb], Ft[:, jb]

    def fit(cols, target_v):
        A = Fv[:, cols]; lam = 1e-2 * len(target_v)
        return np.linalg.solve(A.T @ A + lam * np.eye(len(cols)), A.T @ target_v)
    specs = {"ewma+inc": col["inc"] + [col["one"]],
             "ewma+inc_ECEI": [col["inc"][i] for i in range(9, 13)] + [col["one"]],
             "ewma+inc_BES": [col["inc"][i] for i in range(0, 9)] + [col["one"]],
             "ewma+inc+lvl": col["inc"] + col["lvl"] + [col["one"]],
             "ewma+inc+lvl+stale": col["inc"] + col["lvl"] + [col["stale"], col["one"]]}
    for name, cols in specs.items():
        for tn, tv in (("label", Ytv - base_v), ("model", Ymv - base_v)):
            w = fit(cols, tv); pred = base_t + Ft[:, cols] @ w
            res[f"{name}|fit_{tn}"] = dict(
                skill=float(1 - ((Ytt - pred) ** 2).mean() / mse_p),
                r2_vs_model=float(1 - ((Ymt - pred) ** 2).mean() / Ymt.var()),
                skill_top10=float(1 - ((Ytt - pred) ** 2)[top].mean() / ((Ytt - Pt) ** 2)[top].mean()),
                skill_rest=float(1 - ((Ytt - pred) ** 2)[~top].mean() / ((Ytt - Pt) ** 2)[~top].mean()),
                coef=w.tolist())
    # (c) free linear combination of the EWMA columns (= learn the smoother kernel) + increments
    cols = list(range(nA)) + col["inc"] + [col["one"]]
    for tn, tv in (("label", Ytv), ("model", Ymv)):
        w = fit(cols, tv); pred = Ft[:, cols] @ w
        res[f"kernel_free+inc|fit_{tn}"] = dict(skill=float(1 - ((Ytt - pred) ** 2).mean() / mse_p),
                                                r2_vs_model=float(1 - ((Ymt - pred) ** 2).mean() / Ymt.var()),
                                                coef_ewma=w[:nA].tolist())
    return res


def amplitude_linearity(test_blocks, model, ti_std):
    out = {}
    CTX = 100
    with torch.no_grad():
        for g in ("BES", "ECEI"):
            idx = W.GROUPS[g]
            curve = {}
            for delta in (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0):
                vals = []
                for b in test_blocks[:16]:
                    x = torch.from_numpy(b["x"])
                    if x.shape[0] < CTX + 5:
                        continue
                    base = model(x[None])[0, CTX, 0]
                    xi = x.clone(); xi[CTX, idx] += delta
                    vals.append(float(model(xi[None])[0, CTX, 0] - base))
                curve[str(delta)] = dict(dy_eV=float(np.mean(vals) * ti_std), dy_per_sigma_eV=float(np.mean(vals) * ti_std / delta))
            out[g] = curve
    return out


def forget_on_events(test_blocks, sd):
    """Forget-gate statistics on event steps (top decile |dECEI|+|dBES|) vs quiet steps."""
    fe, fq, fmax_e, fmax_q = [], [], [], []
    with torch.no_grad():
        for b in test_blocks[:12]:
            x = torch.from_numpy(b["x"])
            _, (g0, g1), _ = W.ti_forward_manual(x, sd)
            d = np.abs(np.diff(b["x"][:, :13], axis=0)).mean(1)
            d = np.concatenate([[0], d])
            thr = np.percentile(d, 90); ev = d >= thr
            f = g0["f"].numpy()
            fe.append(f[ev].mean()); fq.append(f[~ev].mean())
            fmax_e.append(f[ev].max(1).mean()); fmax_q.append(f[~ev].max(1).mean())
    return dict(f_mean_event=float(np.mean(fe)), f_mean_quiet=float(np.mean(fq)),
                f_unitmax_event=float(np.mean(fmax_e)), f_unitmax_quiet=float(np.mean(fmax_q)))


def main():
    cks = [c for c in W.checkpoints() if c["tag"].startswith(("b1_", "b5i_"))]
    results = {}
    for ck in cks:
        t0 = time.time()
        sd = torch.load(ck["dir"] / "weights" / "seq_lstm.pth", map_location="cpu")
        m = json.loads((ck["dir"] / "metrics.json").read_text(encoding="utf-8"))
        stats = {g: {k: np.asarray(v, dtype=np.float64) for k, v in d.items()} for g, d in m["normalization"]["stats"].items()}
        model = SeqCESLSTMv2(n_extra_vt=0, n_extra_ti=0); model.load_state_dict(sd); model.eval()
        val_names, test_names = W.split_files(ck)
        tb = W.blocks_for(ck, test_names, stats, 24); vb = W.blocks_for(ck, val_names, stats, 24)
        r = {"pop": ck["pop"], "surrogate": surrogates(vb, tb, model),
             "amplitude": amplitude_linearity(tb, model, float(stats["target"]["std"][0])),
             "forget_events": forget_on_events(tb, sd)}
        results[ck["tag"]] = r
        s = r["surrogate"]
        print(f"[2] {ck['tag']} ({time.time()-t0:.0f}s) model={s['model_skill']:.3f} "
              f"ewma_best(a={s['best_alpha']})={s[f'ewma_fixed_a{s['best_alpha']}']['skill']:.3f} "
              f"ewma+inc={s['ewma+inc|fit_label']['skill']:.3f} ewma+inc+lvl={s['ewma+inc+lvl|fit_label']['skill']:.3f} "
              f"kernel_free+inc={s['kernel_free+inc|fit_label']['skill']:.3f} "
              f"f_ev={r['forget_events']['f_mean_event']:.3f} f_q={r['forget_events']['f_mean_quiet']:.3f}")
    (HERE / "weight_readout2.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("[done]")


if __name__ == "__main__":
    main()
