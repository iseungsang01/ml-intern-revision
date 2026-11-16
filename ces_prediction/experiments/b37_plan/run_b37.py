"""B.37 (PREREGISTRATION_B37.md): four readouts on frozen estimators.

V  error model, interval, switch and layer fitted on validation discharges, applied to test rows
F  prediction of an unusable CES value
Q  skill with the fit noise taken out
S  the frozen network fed means of the raw stream instead of snapshots

CPU only; no network is trained.  Run from the repo root:
    py ces_prediction/experiments/b37_plan/run_b37.py [V F Q S]
Output: docs/paper/b37_plan.json / .md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b30_thinned"))
sys.path.insert(0, str(HERE.parent / "b33_clusters"))
sys.path.insert(0, str(HERE.parent / "b34_error_model"))
import run_b30 as R  # noqa: E402
from run_b33 import window_feats  # noqa: E402
from run_b34 import ci, wq  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b37_plan.json"
OUT_MD = R.REPO / "docs" / "paper" / "b37_plan.md"
RAW = R.DATA / "raw_mus" / ".cache"
RAW_SHOTS = (31097, 31114, 31357, 31359, 31747, 31873, 31902, 31921, 31923, 32027, 32092, 32097)
SEED, N_BOOT = 20261116, 1000


def gbr():
    return HistGradientBoostingRegressor(random_state=SEED)


class Ctx:
    def __init__(self):
        R.pin_env()
        (self.torch, self.B, self.load_stats, load_grid_files, self.build_blocks, self.GAP, self.MODELS) = R.imports()
        self.grid, self.dims = load_grid_files(R.DATA, True, R.POPS["cut"])
        self.names = sorted(self.grid)
        self.nb, self.ne = self.dims["bes"], self.dims["ecei"]
        self.shot = np.array([int(re.search(r"(\d+)", nm).group(1)) for nm in self.names])
        self.seg, self.tstep = {}, {}
        for i, nm in enumerate(self.names):
            dt = np.diff(self.grid[nm][:, 0].astype(np.float64))
            self.seg[i] = np.concatenate(([0], np.cumsum(dt >= 0.5)))
            self.tstep[i] = np.concatenate(([np.nan], dt / 0.010))

    def manifest(self, s):
        return json.loads((R.DATA / f".b1_w2cut_split_s{s}" / "split_manifest.json").read_text(encoding="utf-8"))

    def net(self, s):
        run = R.DATA / f".b29_norm0_cut_s{s}"
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if m["seq_model"] != "v2" or bool(m["per_shot_input_norm"]):
            raise SystemExit(f"FATAL: {run.name} is not the causal variant")
        stats = self.load_stats(m)
        model = self.MODELS["v2"]()
        model.load_state_dict(self.torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
        model.eval()
        return model, stats

    def predict(self, arr, net):
        """T_i estimate of the causal variant at every row of one file (natural cadence)."""
        model, stats = net
        tm = arr[:, 0].astype(np.float64)
        spans = R.block_spans(tm, self.GAP)
        blks = self.build_blocks(arr, self.dims, stats, per_shot_norm=False, expanding=False)
        if len(blks) != len(spans):
            raise SystemExit("FATAL: block spans do not match build_blocks")
        out = np.full(len(tm), np.nan)
        mean, std = float(stats["target"]["mean"][0]), float(stats["target"]["std"][0])
        with self.torch.no_grad():
            for blk, (a, b) in zip(blks, spans):
                out[a:b] = model(self.torch.from_numpy(blk["x"]).unsqueeze(0))[0][:, 0].numpy() * std + mean
        return out, spans

    def fast_stats(self, test_idx):
        X = np.concatenate([self.grid[self.names[i]][:, 3:3 + self.nb + self.ne]
                            for i in range(len(self.names)) if i not in test_idx]).astype(np.float64)
        return X.mean(0), X.std(0) + 1e-12

    def feats(self, i, mu, sd):
        a = (self.grid[self.names[i]][:, 3:3 + self.nb + self.ne].astype(np.float64) - mu) / sd
        return np.column_stack([window_feats(a[:, :self.nb].mean(1), self.seg[i]),
                                window_feats(a[:, self.nb:].mean(1), self.seg[i]), self.tstep[i]])

    def rows(self, i, net):
        """Scored T_i rows of file i at the natural cadence: row, y, causal, hold, gp, age (ms)."""
        arr = self.grid[self.names[i]]
        pred, spans = self.predict(arr, net)
        tm, y = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
        out = []
        for a, b in spans:
            obs = np.flatnonzero(~np.isnan(y[a:b])) + a
            for pos in range(1, len(obs)):
                r = obs[pos]
                lo = max(0, pos - R.GP_SIDE)
                gp = self.B.predict_gp_causal(tm[obs[lo:pos]], y[obs[lo:pos]], tm[r])
                out.append((r, y[r], pred[r], y[obs[pos - 1]], gp, (tm[r] - tm[obs[pos - 1]]) * 1000.0))
        return np.array(out) if out else np.zeros((0, 6))


def inputs(fr, age, h, c, g):
    return np.column_stack([fr, age, h, c - h, c - g, np.abs(c - h), np.abs(c - g)])


# ------------------------------------------------------------------------------------------ V
def part_v(C, rng):
    res = {}
    for s in R.SEEDS:
        man, net = C.manifest(s), C.net(s)
        z = np.load(R.ROWS_DIR / f"rows_cut_s{s}.npz")
        tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
        test = set(np.unique(tab["file"]).tolist())
        mu, sd = C.fast_stats(test)
        # check on five test files
        k1 = tab["k"] == 1
        dev, cnt = 0, 0
        for i in sorted(test)[:5]:
            rw = C.rows(i, net)
            m = k1 & (tab["file"] == i)
            lut = {int(r): j for j, r in enumerate(rw[:, 0])}
            for r, yv, cv, hv, gv in zip(tab["row"][m], tab["y"][m], tab["causal"][m], tab["hold"][m], tab["gp"][m]):
                j = lut.get(int(r))
                cnt += 1
                if j is None or max(abs(rw[j, 2] - cv), abs(rw[j, 3] - hv), abs(rw[j, 4] - gv)) > 0.5:
                    dev += 1
        if cnt == 0 or dev / cnt > 0.01:
            raise SystemExit(f"FATAL: check failed on split {s}: {dev} of {cnt} rows deviate")
        val = [C.names.index(n) for n in sorted(man["val_files"]) if n in C.grid]
        Xv, yv, cv, hv = [], [], [], []
        for i in val:
            rw = C.rows(i, net)
            if not len(rw):
                continue
            fr = C.feats(i, mu, sd)[rw[:, 0].astype(int)]
            ok = np.isfinite(fr).all(1) & np.isfinite(rw).all(1)
            Xv.append(inputs(fr[ok], rw[ok, 5], rw[ok, 3], rw[ok, 2], rw[ok, 4]))
            yv.append(rw[ok, 1]); cv.append(rw[ok, 2]); hv.append(rw[ok, 3])
        Xv, yv, cv, hv = np.concatenate(Xv), np.concatenate(yv), np.concatenate(cv), np.concatenate(hv)
        F = {i: C.feats(i, mu, sd) for i in test}
        fr = np.array([F[int(f)][int(r)] for f, r in zip(tab["file"], tab["row"])])
        ok = k1 & np.all([np.isfinite(tab[e]) for e in ("hold", "gp", "causal", "y")], axis=0) & np.isfinite(fr).all(1)
        y, c, h, g = (tab[e][ok].astype(np.float64) for e in ("y", "causal", "hold", "gp"))
        Xt = inputs(fr[ok], tab["age_ms"][ok], h, c, g)
        uf, f = np.unique(tab["file"][ok].astype(int), return_inverse=True)
        n = len(uf)
        ec, eh, ecv, ehv = np.abs(c - y), np.abs(h - y), np.abs(cv - yv), np.abs(hv - yv)
        m_size = gbr().fit(Xv, np.log(ecv + 1.0))
        size = np.exp(m_size.predict(Xt))
        q_in = np.quantile(ecv / np.exp(m_size.predict(Xv)), 0.90)
        half, const = q_in * size, np.quantile(ecv, 0.90)
        # reported: scale from out-of-fold predictions inside the validation rows (two halves)
        halfv = rng.random(len(yv)) < 0.5
        oof = np.zeros(len(yv))
        for tr, te in ((halfv, ~halfv), (~halfv, halfv)):
            oof[te] = np.exp(gbr().fit(Xv[tr], np.log(ecv[tr] + 1.0)).predict(Xv[te]))
        half_oof = np.quantile(ecv / oof, 0.90) * size
        use = gbr().fit(Xv, ehv - ecv).predict(Xt) > 0
        es = np.where(use, ec, eh)
        pa = gbr().fit(Xv, yv - cv).predict(Xt)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
        Wr = W[:, f]
        sums = lambda v: W @ np.bincount(f, v, n)                              # noqa: E731
        top = size >= np.quantile(size, 0.90)
        one = np.ones(len(y))

        def cov(hw):
            cc = (ec <= hw).astype(float)
            return [float(cc.mean()), *ci(sums(cc) / sums(one))]

        o = {"validation_discharges": len(val), "validation_rows": int(len(yv)), "test_rows": int(len(y)),
             "check_rows": cnt, "check_deviating": dev,
             "V1_share": [float((ec[top] ** 2).sum() / (ec ** 2).sum()), *ci(sums(ec ** 2 * top) / sums(ec ** 2))],
             "V2_coverage": cov(half), "V2_width_ratio": [float(np.median(half) / const), *ci(wq(half, Wr) / const)],
             "coverage_scale_out_of_fold": cov(half_oof), "width_ratio_scale_out_of_fold": float(np.median(half_oof) / const),
             "coverage_constant": cov(np.full(len(y), const)),
             "median_half_width_eV": float(np.median(half)), "constant_half_width_eV": float(const),
             "V3_median_switch_over_causal": [float(np.median(es) / np.median(ec)), *ci(wq(es, Wr) / wq(ec, Wr))],
             "V3_median_switch_over_hold": [float(np.median(es) / np.median(eh)), *ci(wq(es, Wr) / wq(eh, Wr))],
             "V3_skill_switch_over_causal": [float(1 - (es ** 2).sum() / (ec ** 2).sum()), *ci(1 - sums(es ** 2) / sums(ec ** 2))],
             "share_rows_using_causal": float(use.mean()),
             "V4_layer_over_causal": [float(1 - ((y - c - pa) ** 2).sum() / (ec ** 2).sum()),
                                      *ci(1 - sums((y - c - pa) ** 2) / sums(ec ** 2))],
             "median_layer_over_causal": [float(np.median(np.abs(y - c - pa)) / np.median(ec)),
                                          *ci(wq(np.abs(y - c - pa), Wr) / wq(ec, Wr))]}
        o["V1_holds"] = bool(o["V1_share"][1] > 0.30)
        o["V2_holds"] = bool(0.87 <= o["V2_coverage"][0] <= 0.93 and o["V2_width_ratio"][2] < 1)
        o["V3_holds"] = bool(o["V3_median_switch_over_causal"][2] < 1 and o["V3_median_switch_over_hold"][2] < 1
                             and o["V3_skill_switch_over_causal"][1] > -0.02)
        o["V4_holds"] = bool(o["V4_layer_over_causal"][1] > 0)
        res[str(s)] = o
        print("V", s, {k: o[k] for k in ("V1_holds", "V2_holds", "V3_holds", "V4_holds")}, flush=True)
    return res


# ------------------------------------------------------------------------------------------ F
def part_f(C, rng):
    res = {}
    for s in R.SEEDS:
        test = set(C.names.index(n) for n in C.manifest(s)["test_files"] if n in C.grid)
        mu, sd = C.fast_stats(test)
        parts = {True: [], False: []}
        for i in range(len(C.names)):
            y = C.grid[C.names[i]][:, 1].astype(np.float64)
            u = np.isnan(y).astype(float)
            n = len(u)
            fr = C.feats(i, mu, sd)
            since = np.zeros(n)
            for j in range(1, n):
                since[j] = 0 if u[j - 1] == 0 else min(50, since[j - 1] + 1)
            ok = np.flatnonzero(np.isfinite(fr).all(1))
            hist = np.column_stack([*(u[ok - k] for k in range(1, 6)), fr[ok, 8], since[ok]])
            parts[i in test].append((np.column_stack([hist, fr[ok, :8]]), u[ok], np.full(len(ok), i), u[ok - 1]))
        Xf, yf = np.concatenate([p[0] for p in parts[False]]), np.concatenate([p[1] for p in parts[False]])
        Xt, yt = np.concatenate([p[0] for p in parts[True]]), np.concatenate([p[1] for p in parts[True]])
        ft, prev = np.concatenate([p[2] for p in parts[True]]), np.concatenate([p[3] for p in parts[True]])
        ph = HistGradientBoostingClassifier(random_state=SEED).fit(Xf[:, :7], yf).predict_proba(Xt[:, :7])[:, 1]
        pf = HistGradientBoostingClassifier(random_state=SEED).fit(Xf, yf).predict_proba(Xt)[:, 1]
        uf = np.unique(ft)
        idx = {i: np.flatnonzero(ft == i) for i in uf}
        onset = prev == 0

        def boot(mask):
            d = []
            for _ in range(300):
                q = np.concatenate([idx[i] for i in rng.choice(uf, len(uf))])
                q = q[mask[q]]
                if 0 < yt[q].sum() < len(q):
                    d.append(roc_auc_score(yt[q], pf[q]) - roc_auc_score(yt[q], ph[q]))
            return ci(np.array(d))

        allr = np.ones(len(yt), bool)
        o = {"fit_rows": int(len(yf)), "test_rows": int(len(yt)), "base_rate": float(yt.mean()),
             "auc_history": float(roc_auc_score(yt, ph)), "auc_full": float(roc_auc_score(yt, pf)),
             "F1_difference": [float(roc_auc_score(yt, pf) - roc_auc_score(yt, ph)), *boot(allr)],
             "ap_history": float(average_precision_score(yt, ph)), "ap_full": float(average_precision_score(yt, pf)),
             "onset_rows": int(onset.sum()), "onset_base_rate": float(yt[onset].mean()),
             "onset_auc_history": float(roc_auc_score(yt[onset], ph[onset])),
             "onset_auc_full": float(roc_auc_score(yt[onset], pf[onset])),
             "onset_difference": [float(roc_auc_score(yt[onset], pf[onset]) - roc_auc_score(yt[onset], ph[onset])), *boot(onset)],
             "onset_ap_history": float(average_precision_score(yt[onset], ph[onset])),
             "onset_ap_full": float(average_precision_score(yt[onset], pf[onset]))}
        o["F1_holds"] = bool(o["F1_difference"][1] > 0)
        res[str(s)] = o
        print("F", s, o["F1_difference"], o["onset_difference"], flush=True)
    return res


# ------------------------------------------------------------------------------------------ Q
def part_q(C, rng):
    res = {}
    for s in R.SEEDS:
        z = np.load(R.ROWS_DIR / f"rows_cut_s{s}.npz")
        tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
        ok = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in ("hold", "gp", "causal", "hist80", "y")], axis=0)
        uf, f = np.unique(tab["file"][ok].astype(int), return_inverse=True)
        n = len(uf)
        d1, d2, fd = [], [], []
        for j, i in enumerate(uf):
            arr = C.grid[C.names[i]]
            t, y = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
            m = (np.isfinite(y[:-2]) & np.isfinite(y[1:-1]) & np.isfinite(y[2:])
                 & (np.abs(np.diff(t)[:-1] - 0.010) < 1e-4) & (np.abs(np.diff(t)[1:] - 0.010) < 1e-4))
            d1.append((y[1:-1] - y[:-2])[m]); d2.append((y[2:] - y[1:-1])[m]); fd.append(np.full(int(m.sum()), j))
        d1, d2, fd = np.concatenate(d1), np.concatenate(d2), np.concatenate(fd)
        lo, hi = np.quantile(np.concatenate([d1, d2]), [0.01, 0.99])
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)

        def sig2(a, b):
            cnt = np.bincount(fd, minlength=n).astype(float)
            sab, sa, sb = np.bincount(fd, a * b, n), np.bincount(fd, a, n), np.bincount(fd, b, n)
            p = -(sab.sum() / cnt.sum() - sa.sum() / cnt.sum() * sb.sum() / cnt.sum())
            N = W @ cnt
            return p, -((W @ sab) / N - (W @ sa) / N * (W @ sb) / N)

        y = tab["y"][ok].astype(np.float64)
        se = {e: (tab[e][ok].astype(np.float64) - y) ** 2 for e in ("hold", "gp", "hist80", "causal")}
        cntr = np.bincount(f, minlength=n).astype(float)
        mse = {e: (np.bincount(f, se[e], n).sum() / cntr.sum(), (W @ np.bincount(f, se[e], n)) / (W @ cntr)) for e in se}
        o = {"triples": int(len(d1)), "rmse_eV": {e: float(np.sqrt(mse[e][0])) for e in se}}
        for lab, (a, b) in (("plain", (d1, d2)), ("clipped", (np.clip(d1, lo, hi), np.clip(d2, lo, hi)))):
            p, bs = sig2(a, b)
            c = {"sigma_eV": [float(np.sqrt(max(p, 0))), *ci(np.sqrt(np.maximum(bs, 0)))]}
            for ref in ("hold", "gp", "hist80"):
                raw = 1 - mse["causal"][0] / mse[ref][0]
                with np.errstate(invalid="ignore", divide="ignore"):
                    cb = 1 - (mse["causal"][1] - bs) / (mse[ref][1] - bs)
                c[f"causal_vs_{ref}"] = {"raw": float(raw), "corrected": [float(1 - (mse["causal"][0] - p) / (mse[ref][0] - p)), *ci(cb)]}
            o[lab] = c
        res[str(s)] = o
        print("Q", s, o["plain"]["sigma_eV"], o["clipped"]["sigma_eV"], flush=True)
    return res


# ------------------------------------------------------------------------------------------ S
def part_s(C, rng):
    assign = {}
    for s in R.SEEDS:
        tf = C.manifest(s)["test_files"]
        for sh in RAW_SHOTS:
            if sh not in assign and f"s{sh}.csv" in tf:
                assign[sh] = s
    nets, per = {}, []
    for sh, s in sorted(assign.items()):
        if s not in nets:
            nets[s] = C.net(s)
        name = f"s{sh}.csv"
        arr = C.grid[name].copy()
        t = arr[:, 0].astype(np.float64)
        hdr = [c for c in pd.read_csv(R.DATA / name, nrows=0).columns]
        cols = {"BES": [c for c in hdr if c.startswith("BES_")], "ECEI": [c for c in hdr if c.startswith("ECEI_")]}
        start = {"BES": 3, "ECEI": 3 + C.nb}
        variants = {k: arr.copy() for k in ("sample", "mean10", "mean1")}
        inside = np.ones(len(t), bool)
        devs = []
        for diag in ("BES", "ECEI"):
            with np.load(RAW / f"s{sh}_{diag}.npz", allow_pickle=True) as z:
                meta = json.loads(str(z["meta"]))
                x = z["x"].astype(np.float64)
            order = [meta["channels"].index(c) for c in cols[diag]]
            x = x[:, order]
            fs, t0 = float(meta["fs_hz"]), float(meta["t0"])
            idx = np.rint((t - t0) * fs).astype(int)
            w10, w1 = int(round(fs * 0.010)), int(round(fs * 0.001))
            ins = (idx - w10 + 1 >= 0) & (idx < len(x))
            inside &= ins
            cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, 0)])
            ii = idx[ins]
            a, b = start[diag], start[diag] + x.shape[1]
            delivered = arr[ins, a:b].astype(np.float64)
            devs.append(float(np.median(np.abs(x[ii] - delivered) / (np.abs(delivered) + 1e-9))))
            variants["sample"][ins, a:b] = x[ii]
            variants["mean10"][ins, a:b] = (cs[ii + 1] - cs[ii + 1 - w10]) / w10
            variants["mean1"][ins, a:b] = (cs[ii + 1] - cs[ii + 1 - w1]) / w1
        y = arr[:, 1].astype(np.float64)
        m = inside & np.isfinite(y)
        pred = {"delivered": C.predict(arr, nets[s])[0]}
        for k, v in variants.items():
            pred[k] = C.predict(v, nets[s])[0]
        sdr = float(np.mean(variants["mean10"][inside, 3:3 + C.nb].std(0) / arr[inside, 3:3 + C.nb].std(0)))
        o = {"shot": sh, "split": s, "rows": int(m.sum()), "median_relative_deviation_of_the_raw_sample": devs,
             "bes_sd_ratio_mean10_over_delivered": sdr}
        for k in pred:
            o[f"sse_{k}"] = float(((pred[k][m] - y[m]) ** 2).sum())
        o["skill_mean10"] = 1 - o["sse_mean10"] / o["sse_delivered"]
        per.append(o)
        print("S", sh, s, o["rows"], round(o["skill_mean10"], 3), devs, flush=True)
    A = np.array([[p["sse_delivered"], p["sse_sample"], p["sse_mean10"], p["sse_mean1"]] for p in per])
    W = rng.multinomial(len(per), np.full(len(per), 1.0 / len(per)), size=2000).astype(float)
    T, Bt = A.sum(0), W @ A
    res = {"per_discharge": per, "discharges": len(per), "rows": int(sum(p["rows"] for p in per))}
    for j, k in ((1, "sample"), (2, "mean10"), (3, "mean1")):
        res[f"skill_{k}_over_delivered"] = [float(1 - T[j] / T[0]), *ci(1 - Bt[:, j] / Bt[:, 0])]
    res["S1_holds"] = bool(res["skill_mean10_over_delivered"][1] > 0)
    return res


def render(res):
    s3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                   # noqa: E731
    f3 = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"                      # noqa: E731
    L = ["# B.37 — four readouts on frozen estimators (registered: PREREGISTRATION_B37.md)", "",
         "CPU, no network trained. T_i, cut population, natural cadence. Intervals: 95 %, resamples of discharges.", ""]
    if "V" in res:
        V = res["V"]
        n = lambda k: sum(V[str(s)][f"{k}_holds"] for s in R.SEEDS)             # noqa: E731
        L += ["## V. Models fitted on validation discharges, applied once to the test rows", "",
              f"- **V1** (the tenth of rows with the largest predicted size holds more than 0.30 of the squared error): {n('V1')} of 4 -> {'holds' if n('V1') >= 3 else 'does not hold'}.",
              f"- **V2** (the adaptive 90 % interval covers 0.87 to 0.93 and is shorter than the constant one): {n('V2')} of 4 -> {'holds' if n('V2') >= 3 else 'does not hold'}.",
              f"- **V3** (the switch lowers the median error against both and keeps the mean): {n('V3')} of 4 -> {'holds' if n('V3') >= 3 else 'does not hold'}.",
              f"- **V4** (a boosted layer on the signed residual lowers the error): {n('V4')} of 4 -> {'holds' if n('V4') >= 3 else 'does not hold'}.", "",
              "| split | validation discharges / rows | test rows | V1 share | V2 coverage | V2 median half-width over constant | "
              "half-widths (eV): adaptive median / constant | coverage of the constant interval | reported: coverage with the scale from "
              "out-of-fold predictions (width ratio) |", "|---|---|---|---|---|---|---|---|---|"]
        for s in R.SEEDS:
            o = V[str(s)]
            L.append(f"| {s} | {o['validation_discharges']} / {o['validation_rows']:,} | {o['test_rows']:,} | {f3(o['V1_share'])} | "
                     f"{f3(o['V2_coverage'])} | {f3(o['V2_width_ratio'])} | {o['median_half_width_eV']:.1f} / {o['constant_half_width_eV']:.1f} | "
                     f"{f3(o['coverage_constant'])} | {f3(o['coverage_scale_out_of_fold'])} ({o['width_ratio_scale_out_of_fold']:.3f}) |")
        L += ["", "| split | V3 median error, switch over causal | switch over hold | skill of the switch over causal | rows using the causal variant | "
              "V4 layer over causal (skill) | median error, layer over causal |", "|---|---|---|---|---|---|---|"]
        for s in R.SEEDS:
            o = V[str(s)]
            L.append(f"| {s} | {f3(o['V3_median_switch_over_causal'])} | {f3(o['V3_median_switch_over_hold'])} | "
                     f"{s3(o['V3_skill_switch_over_causal'])} | {o['share_rows_using_causal']:.3f} | {s3(o['V4_layer_over_causal'])} | "
                     f"{f3(o['median_layer_over_causal'])} |")
        L += ["", "Check before the readout (five test files per split, recomputed against the row table): " +
              ", ".join(f"{V[str(s)]['check_deviating']} of {V[str(s)]['check_rows']:,} rows deviate" for s in R.SEEDS) + ".", ""]
    if "F" in res:
        F = res["F"]
        n = sum(F[str(s)]["F1_holds"] for s in R.SEEDS)
        L += ["## F. Predicting an unusable CES value", "",
              f"- **F1** (the fast-channel window adds to the history of unusable rows, area under the ROC curve): {n} of 4 -> "
              f"{'holds' if n >= 3 else 'does not hold'}.", "",
              "| split | test rows | base rate | AUC history / full | F1 difference | average precision history / full | onsets: rows | "
              "base rate | AUC history / full | difference | average precision history / full |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in R.SEEDS:
            o = F[str(s)]
            L.append(f"| {s} | {o['test_rows']:,} | {o['base_rate']:.3f} | {o['auc_history']:.3f} / {o['auc_full']:.3f} | "
                     f"{s3(o['F1_difference'])} | {o['ap_history']:.3f} / {o['ap_full']:.3f} | {o['onset_rows']:,} | "
                     f"{o['onset_base_rate']:.3f} | {o['onset_auc_history']:.3f} / {o['onset_auc_full']:.3f} | "
                     f"{s3(o['onset_difference'])} | {o['onset_ap_history']:.3f} / {o['onset_ap_full']:.3f} |")
        L.append("")
    if "Q" in res:
        Q = res["Q"]
        L += ["## Q. Skill of the causal variant with the fit noise taken out (reported)", "",
              "sigma^2 = minus the lag-1 autocovariance of the 10 ms increments on the test discharges; subtracted from both mean squared errors.", "",
              "| split | sigma, plain (eV) | sigma, clipped (eV) | RMSE hold / GP / twin / causal (eV) | over hold: raw, clipped-corrected | "
              "over the causal GP: raw, clipped-corrected | over the twin: raw, clipped-corrected | over the causal GP, plain-corrected |",
              "|---|---|---|---|---|---|---|---|"]
        for s in R.SEEDS:
            o = Q[str(s)]
            r, c, p = o["rmse_eV"], o["clipped"], o["plain"]
            cell = lambda ref: f"{c[f'causal_vs_{ref}']['raw']:+.3f}, {s3(c[f'causal_vs_{ref}']['corrected'])}"   # noqa: E731
            L.append(f"| {s} | {p['sigma_eV'][0]:.1f} [{p['sigma_eV'][1]:.1f}, {p['sigma_eV'][2]:.1f}] | "
                     f"{c['sigma_eV'][0]:.1f} [{c['sigma_eV'][1]:.1f}, {c['sigma_eV'][2]:.1f}] | "
                     f"{r['hold']:.1f} / {r['gp']:.1f} / {r['hist80']:.1f} / {r['causal']:.1f} | {cell('hold')} | {cell('gp')} | "
                     f"{cell('hist80')} | {s3(p['causal_vs_gp']['corrected'])} |")
        L.append("")
    if "S" in res:
        S = res["S"]
        L += ["## S. Means of the raw stream instead of snapshots", "",
              f"{S['discharges']} discharges, {S['rows']:,} scored rows inside the raw windows.", "",
              f"- **S1** (10 ms means over the network as delivered): {s3(S['skill_mean10_over_delivered'])} -> "
              f"{'holds' if S['S1_holds'] else 'does not hold'}.",
              f"- Reported: 1 ms means {s3(S['skill_mean1_over_delivered'])}; the raw sample at the stamp (check) "
              f"{s3(S['skill_sample_over_delivered'])}.", "",
              "| discharge | split | rows | skill, 10 ms means over delivered | BES sd, 10 ms means over delivered | "
              "median relative deviation of the raw sample from the delivered value (BES, ECEI) |", "|---|---|---|---|---|---|"]
        for p in S["per_discharge"]:
            d = p["median_relative_deviation_of_the_raw_sample"]
            L.append(f"| {p['shot']} | {p['split']} | {p['rows']} | {p['skill_mean10']:+.3f} | {p['bes_sd_ratio_mean10_over_delivered']:.3f} | "
                     f"{d[0]:.2e}, {d[1]:.2e} |")
    return "\n".join(L) + "\n"


def main():
    parts = [a for a in sys.argv[1:] if a in "VFQS"] or list("VFQS")
    res = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    C = Ctx()
    fn = {"V": part_v, "F": part_f, "Q": part_q, "S": part_s}
    for i, p in enumerate("VFQS"):
        if p in parts:
            res[p] = fn[p](C, np.random.default_rng(SEED + i))
            OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
            OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
            print(p, "written", flush=True)


if __name__ == "__main__":
    main()
