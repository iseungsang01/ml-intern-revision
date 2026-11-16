"""B.33 (PREREGISTRATION_B33.md): do clusters of the fast-channel window carry structure that the
trained estimator leaves unused?

CPU only; nothing is trained.  Reads the frozen B.30 row tables and the shot files.
Run from the repo root:
    py ces_prediction/experiments/b33_clusters/run_b33.py
Output: docs/paper/b33_clusters.json / .md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b30_thinned"))
import run_b30 as R  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b33_clusters.json"
OUT_MD = R.REPO / "docs" / "paper" / "b33_clusters.md"
SEED, N_BOOT = 20261116, 2000
K_PRIMARY, K_ALL = 6, (3, 4, 6, 8, 12)
MAX_FIT_ROWS, MIN_SHARE, MIN_FIT = 200_000, 0.02, 20
FEATS = ("BES level", "BES departure", "BES sd of 100 ms", "BES increment",
         "ECEI level", "ECEI departure", "ECEI sd of 100 ms", "ECEI increment", "time step / 10 ms")


def window_feats(x, seg):
    """level, departure from the mean of the preceding ten rows, their sd, one-row increment."""
    n = len(x)
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    out = np.full((n, 4), np.nan)
    i = np.arange(10, n)
    ok = seg[i] == seg[i - 10]
    i = i[ok]
    m = (c1[i] - c1[i - 10]) / 10.0
    v = np.maximum((c2[i] - c2[i - 10]) / 10.0 - m * m, 0.0)
    out[i, 0], out[i, 1], out[i, 2], out[i, 3] = x[i], x[i] - m, np.sqrt(v), x[i] - x[i - 1]
    return out


def ci(b):
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def skill(a, b, f, W):
    """1 - SSE_a / SSE_b with bootstrap over the discharges indexed by f (0..n-1)."""
    n = W.shape[1]
    sa, sb = np.bincount(f, a, n), np.bincount(f, b, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(1 - sa.sum() / sb.sum()), 1 - (W @ sa) / (W @ sb)


def main():
    R.pin_env()
    _t, _B, _ls, load_grid_files, _bb, _gap, _m = R.imports()
    grid, dims = load_grid_files(R.DATA, True, R.POPS["cut"])
    names = sorted(grid)
    nb, ne = dims["bes"], dims["ecei"]
    shot = np.array([int(re.search(r"(\d+)", nm).group(1)) for nm in names])
    seg, tstep, first, after10 = {}, {}, {}, {}
    for i, nm in enumerate(names):
        t = grid[nm][:, 0].astype(np.float64)
        dt = np.diff(t)
        seg[i] = np.concatenate(([0], np.cumsum(dt >= 0.5)))
        tstep[i] = np.concatenate(([np.nan], dt / 0.010))
        fl, a10 = np.zeros(len(t), bool), np.zeros(len(t), bool)
        for j in np.flatnonzero((dt > 0.015) & (dt < 0.045)):
            fl[j + 1] = True
            a10 |= (t >= t[j + 1]) & (t < t[j + 1] + 0.100 - 1e-6)
        first[i], after10[i] = fl, a10
    rng = np.random.default_rng(SEED)
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B33.md", "features": list(FEATS), "splits": {}}
    for s in R.SEEDS:
        z = np.load(R.ROWS_DIR / f"rows_cut_s{s}.npz")
        tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
        test = np.unique(tab["file"])
        fit_files = [i for i in range(len(names)) if i not in set(test.tolist())]
        X = np.concatenate([grid[names[i]][:, 3:3 + nb + ne + dims["mc"]] for i in fit_files]).astype(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-12

        def feats(i):
            a = (grid[names[i]][:, 3:3 + nb + ne].astype(np.float64) - mu[:nb + ne]) / sd[:nb + ne]
            return np.column_stack([window_feats(a[:, :nb].mean(1), seg[i]),
                                    window_feats(a[:, nb:].mean(1), seg[i]), tstep[i]])

        F = {i: feats(i) for i in range(len(names))}
        fit = np.concatenate([F[i] for i in fit_files])
        fit = fit[np.isfinite(fit).all(1)]
        fm, fs = fit.mean(0), fit.std(0) + 1e-12
        if len(fit) > MAX_FIT_ROWS:
            fit = fit[np.random.default_rng(SEED).choice(len(fit), MAX_FIT_ROWS, replace=False)]
        scale = lambda a: np.clip((a - fm) / fs, -10, 10)                     # noqa: E731
        fr = np.array([F[int(f)][int(r)] for f, r in zip(tab["file"], tab["row"])])
        base = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in ("hold", "causal", "hist80", "y")], axis=0)
        ok = base & np.isfinite(fr).all(1)
        y, c, h, hold = (tab[e][ok].astype(np.float64) for e in ("y", "causal", "hist80", "hold"))
        fidx = tab["file"][ok].astype(int)
        uf, f = np.unique(fidx, return_inverse=True)
        even = (shot[uf] % 2 == 0)[f]
        fx = fr[ok]
        dB, dE = fx[:, 1], fx[:, 5]
        is_first = np.array([first[int(a)][int(b)] for a, b in zip(tab["file"][ok], tab["row"][ok])])
        is_a10 = np.array([after10[int(a)][int(b)] for a, b in zip(tab["file"][ok], tab["row"][ok])])
        W = rng.multinomial(len(uf), np.full(len(uf), 1.0 / len(uf)), size=N_BOOT).astype(float)
        se_c, se_h, se_hold = (c - y) ** 2, (h - y) ** 2, (hold - y) ** 2
        A = np.column_stack([np.ones(len(y)), dB, dE])
        r = y - c

        def cross(labels, nk):
            pred = np.zeros(len(y))
            for tr, te in ((even, ~even), (~even, even)):
                g = np.linalg.lstsq(A[tr], r[tr], rcond=None)[0]
                for k in range(nk):
                    m_tr, m_te = tr & (labels == k), te & (labels == k)
                    beta = np.linalg.lstsq(A[m_tr], r[m_tr], rcond=None)[0] if m_tr.sum() >= MIN_FIT else g
                    pred[m_te] = A[m_te] @ beta
            return pred

        glob = cross(np.zeros(len(y), int), 1)
        se_g = (c + glob - y) ** 2
        out = {"scored_rows": int(ok.sum()), "left_out_share": float(1 - ok.sum() / base.sum()),
               "test_discharges": int(len(uf)), "fold_E_discharges": int((shot[uf] % 2 == 0).sum()),
               "global_vs_uncorrected": [skill(se_g, se_c, f, W)[0], *ci(skill(se_g, se_c, f, W)[1])], "k": {}}
        for nk in K_ALL:
            km = KMeans(n_clusters=nk, n_init=10, random_state=SEED).fit(scale(fit))
            lab = km.predict(scale(fx))
            share = np.bincount(lab, minlength=nk) / len(lab)
            big = np.flatnonzero(share >= MIN_SHARE)
            se_k = (c + cross(lab, nk) - y) ** 2
            p, b = skill(se_k, se_g, f, W)
            p2, b2 = skill(se_k, se_c, f, W)
            cell = {"K2_cluster_vs_global": [p, *ci(b)], "K2_holds": bool(np.quantile(b[np.isfinite(b)], 0.025) > 0),
                    "cluster_vs_uncorrected": [p2, *ci(b2)], "K1": []}
            k1_ok = True
            for pick, ev in ((even, ~even), (~even, even)):
                inc = {k: 1 - se_c[pick & (lab == k)].sum() / se_h[pick & (lab == k)].sum()
                       for k in big if (pick & (lab == k)).sum() >= MIN_FIT}
                hi, lo = max(inc, key=inc.get), min(inc, key=inc.get)
                mh, ml = ev & (lab == hi), ev & (lab == lo)
                n = W.shape[1]

                def sk(m, WW):
                    sa, sb = np.bincount(f[m], se_c[m], n), np.bincount(f[m], se_h[m], n)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        return 1 - sa.sum() / sb.sum(), 1 - (WW @ sa) / (WW @ sb)

                (ph, bh), (pl, bl) = sk(mh, W), sk(ml, W)
                d = ci(bh - bl)
                cell["K1"].append({"picked_on": "E" if pick is even else "O", "high": int(hi), "low": int(lo),
                                   "difference": float(ph - pl), "ci95": d, "holds": bool(d[0] > 0)})
                k1_ok &= d[0] > 0
            cell["K1_holds"] = bool(k1_ok)
            if nk == K_PRIMARY:
                cl = []
                for k in range(nk):
                    m = lab == k
                    if not m.any():
                        continue
                    cl.append({"cluster": k, "share": float(share[k]), "centre": [float(v) for v in km.cluster_centers_[k]],
                               "hold_rmse_eV": float(np.sqrt(se_hold[m].mean())),
                               "causal_rmse_eV": float(np.sqrt(se_c[m].mean())),
                               "increment": float(1 - se_c[m].sum() / se_h[m].sum()),
                               "share_of_first_rows_after_a_blip": float((m & is_first).sum() / max(1, is_first.sum())),
                               "share_within_ten_rows_after_a_blip": float((m & is_a10).sum() / max(1, m.sum()))})
                cell["clusters"] = cl
                cell["first_rows_after_a_blip"] = int(is_first.sum())
            out["k"][str(nk)] = cell
        res["splits"][str(s)] = out
        print(s, "done", flush=True)
    kp = str(K_PRIMARY)
    res["verdict"] = {"K1_splits": sum(res["splits"][str(s)]["k"][kp]["K1_holds"] for s in R.SEEDS),
                      "K2_splits": sum(res["splits"][str(s)]["k"][kp]["K2_holds"] for s in R.SEEDS)}
    res["verdict"]["K1_holds"] = res["verdict"]["K1_splits"] >= 3
    res["verdict"]["K2_holds"] = res["verdict"]["K2_splits"] >= 3
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                    # noqa: E731
    v = res["verdict"]
    L = ["# B.33 — clusters of the fast-channel window (registered: PREREGISTRATION_B33.md)", "",
         "CPU, nothing trained. Frozen B.30 row tables, T_i, cut population, natural cadence. k-means on nine causal "
         "features of the 100 ms fast-channel window, fitted on the non-test files of each split. Intervals: 95 %, "
         "2,000 resamples of discharges.", "",
         f"- **K1** (the fast-channel increment differs between clusters; picked on one parity fold, tested on the other): "
         f"holds on {v['K1_splits']} of 4 splits -> {'holds' if v['K1_holds'] else 'does not hold'}.",
         f"- **K2** (a per-cluster affine correction of the causal variant's residual against a global one, cross-fitted): "
         f"holds on {v['K2_splits']} of 4 splits -> {'holds' if v['K2_holds'] else 'does not hold'}.", "",
         f"## Primary, k = {K_PRIMARY}", "",
         "| split | scored rows | left out (no window) | K1 picked on E: high minus low on O | K1 picked on O: on E | "
         "K2 per-cluster over global | per-cluster over uncorrected | global over uncorrected |", "|---|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        o = res["splits"][str(s)]
        c = o["k"][kp]
        k1 = [f"{x['difference']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" for x in c["K1"]]
        L.append(f"| {s} | {o['scored_rows']:,} | {o['left_out_share']:.3f} | {k1[0]} | {k1[1]} | {f3(c['K2_cluster_vs_global'])} | "
                 f"{f3(c['cluster_vs_uncorrected'])} | {f3(o['global_vs_uncorrected'])} |")
    L += ["", "## Sensitivity to k (decides nothing)", "", "| k | K2 per-cluster over global, four splits | splits with K2 | splits with K1 |",
          "|---|---|---|---|"]
    for nk in K_ALL:
        xs = [res["splits"][str(s)]["k"][str(nk)] for s in R.SEEDS]
        L.append(f"| {nk} | " + ", ".join(f"{x['K2_cluster_vs_global'][0]:+.3f}" for x in xs) +
                 f" | {sum(x['K2_holds'] for x in xs)} | {sum(x['K1_holds'] for x in xs)} |")
    for s in R.SEEDS:
        c = res["splits"][str(s)]["k"][kp]
        L += ["", f"## K3: the clusters of split {s} ({c['first_rows_after_a_blip']} first rows after a blip among the scored rows)", "",
              "| cluster | share of rows | BES level | BES departure | BES sd | ECEI level | ECEI departure | time step | hold RMSE (eV) | "
              "causal variant RMSE (eV) | increment | share of the first rows after a blip | rows within ten after a blip |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in sorted(c["clusters"], key=lambda q: -q["share"]):
            ce = x["centre"]
            L.append(f"| {x['cluster']} | {x['share']:.3f} | {ce[0]:+.2f} | {ce[1]:+.2f} | {ce[2]:+.2f} | {ce[4]:+.2f} | {ce[5]:+.2f} | "
                     f"{ce[8]:+.2f} | {x['hold_rmse_eV']:.1f} | {x['causal_rmse_eV']:.1f} | {x['increment']:+.3f} | "
                     f"{x['share_of_first_rows_after_a_blip']:.3f} | {x['share_within_ten_rows_after_a_blip']:.3f} |")
    L += ["", "Cluster centres are in standardized feature units (zero = mean of the fit rows)."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT_MD, v)


if __name__ == "__main__":
    main()
