# -*- coding: utf-8 -*-
"""Pick the shots worth re-acquiring at microsecond resolution.

The CSVs this repo trains on are a 100 Hz (10 ms) grid. That grid is where the Mirnov
problem lives: MC is a dB/dt snapshot sampled without an anti-aliasing filter, so its
lag-1 autocorrelation is ~0.00 while BES/ECEI sit at ~+0.59 (`analyze_data_evidence.py`
claim B). Phase is gone -- but for a uniformly random sampling phase E[x^2] = A^2/2 still
holds, so a rolling RMS of MC recovers the *mode-amplitude envelope* even from the aliased
grid. That envelope is the instrument used here to ask which discharges would actually pay
back a raw microsecond fetch.

Three axes, all measured on the real shot CSVs (no synthetic data, full 641-file scan):

  A. label      -- clean, independent CES supervision under the confirmed protocol's data
                   treatment (CES_TI fit-failure cut at 3 keV, held/forward-fill removal),
                   plus the length of the discharge.
  B. diagnostic -- do BES / ECEI actually move: dynamic range, sustained level steps
                   (L->H-like), repeated fast crashes (ELM-like), and the share of
                   sample-to-sample variance that is already aliased high-frequency content
                   (that share is exactly what a microsecond fetch would resolve).
  C. Mirnov     -- amplitude, how much of it survives dropping the 5 largest samples
                   (an RMS carried by a handful of samples is an electrical spike, not a
                   mode), what fraction of the discharge sits in sustained (>= 30 ms) hot
                   stretches, two-coil envelope coherence (a global mode is seen by both
                   toroidally separated coils), and coupling to the BES fluctuation level.

Split membership comes from the confirmed W = 2 protocol's frozen manifests
(`data/.b1_w2cut_split_s{42,1,7,123}`), so a shot proposed as a paper test case is a test
shot under the protocol the paper actually reports.

Usage (repo root):  py ces_prediction/experiments/hires_shots/select_hires_shots.py
Writes: shot_metrics.csv, shot_scored.csv, FINAL_10.csv, FINAL_10.png (next to this file).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("CES_DATA_DIR", REPO_ROOT / "data"))
HERE = Path(__file__).resolve().parent
SEEDS = (42, 1, 7, 123)
INITS = (42, 1, 7, 123)

GAP_S = 0.5            # contiguous-block boundary, matches dataset.py STUCK_GAP_SECONDS
TI_SPIKE_EV = 3000.0   # confirmed protocol CES_TI_SPIKE_CUT_EV
ENV_WIN = 5            # 50 ms rolling window for the MC amplitude envelope

# The final list. Half of it is chosen by the literature cross-check
# (`literature_crosscheck.py`) rather than by this script's score: a shot whose physics is
# already published is a shot whose raw microsecond data can be checked against a known
# answer. The rest is chosen by score, to cover the axes the published shots miss
# (CES_VT sampling, very long discharges, the strongest Mirnov activity).
#
# Split membership is NOT reassigned: every shot keeps the role the confirmed W = 2
# protocol's seed-42 manifest already gives it (3 test / 4 val / 3 train).
#
# No two shots in the list come from the same session. Adjacent shot numbers are repeat
# discharges that share plasma setup and diagnostic gain, and `session_similarity.py`
# measures how much that matters: over all 641 shots, pairs with a gap <= 2 sit at a
# median |dTi| of 92 eV against 265 eV for random pairs, and a median calibration distance
# of 0.82 against 4.28 (one-sided p < 1e-4). The earlier version of this list contained
# two such pairs (#31921/#31923 at the 0.0th percentile of all 176k pair distances, and
# #31357/#31359); one member of each was dropped and replaced from the same split side.
FINAL = [
    # --- literature-confirmed -------------------------------------------------------
    (31921, "test",  "PUBLISHED (FIRE mode, 2 papers): FIRE 5.40 s vs H-mode 8.05 s on CES "
                     "edge profiles, plus BES bispectral WCM analysis; also the best data "
                     "shot we have (rank 2/121) and the highest MC-turbulence coupling of "
                     "all 641"),
    (31873, "test",  "PUBLISHED (Nat. Commun. 2024): fully automated ELM suppression with "
                     "ML-integrated RMP; our window covers the whole suppressed phase"),
    (31359, "val",   "PUBLISHED (Nat. Commun. 2024): no n=1 ERMP -> density ETB at 5.5 s, ELMs, "
                     "core Ti drops; 246 CES_VT and the liveliest BES/ECEI of the published set"),
    (32027, "val",   "PUBLISHED (PanoMHD): clear L/H transition with 100-300 kHz cross-power / "
                     "cross-phase spectrograms -- the frequency band a microsecond fetch buys"),
    # --- chosen by the data score ----------------------------------------------------
    (31114, "test",  "largest clean MC amplitude among test shots; model gains on both targets"),
    (32097, "val",   "strongest Mirnov shot overall: RMS 17.3, two-coil coherence 0.93 (rank 1)"),
    (31745, "val",   "highest two-coil envelope coherence of any candidate (0.96) at RMS 16.6 "
                     "-- replaces the mode-coherence role the dropped #31923/#31357 played"),
    (31604, "train", "largest spike-free MC (RMS 21.3, kurtosis ~0): steady-state mode"),
    (31074, "train", "balanced: two-coil coherence 0.74, 7.5 s, 446 CES_VT"),
    (31937, "train", "longest discharge (15.2 s), most labels (1479 TI / 722 VT), quiet MC"),
]
# Dropped to remove same-session pairs; each replacement came from the same split side.
DROPPED_FOR_SESSION = {
    31923: "same session as #31921 (gap 2): summary distance at the 0.0th percentile of all "
           "176k pairs, |dTi| 21 eV. Kept #31921 -- 2 papers, 296 CES_VT, data rank 2/121. "
           "Replaced by #32027 (val, published)",
    31357: "same session as #31359 (gap 2): calibration distance at the 2.2th percentile. "
           "Kept #31359 -- 246 vs 44 CES_VT, MC 8.3 vs 5.4, MC-BES coupling +0.32 vs -0.08. "
           "Replaced by #31745 (val, two-coil coherence 0.96)",
}
# Every dataset shot the literature cross-check confirmed (see literature_crosscheck.py).
# Every dataset shot the literature cross-check confirmed (see literature_crosscheck.py).
CONFIRMED_SHOTS = {31921, 31923, 31873, 31276, 31357, 31359, 32027, 31888}
# Published shots that did NOT make the list, and why (see SELECTION.md).
REJECTED_PUBLISHED = {
    31276: "MC RMS 12.7 collapses to 32 % when the 5 largest samples are dropped "
           "(kurtosis 363) -- an electrical spike, not mode activity",
    31888: "disruption example shot; MC RMS carried by spikes (trim ratio 0.36, kurtosis 105)",
    31923: "dropped for same-session overlap with #31921, not for quality -- see "
           "DROPPED_FOR_SESSION",
    31357: "dropped for same-session overlap with #31359, not for quality -- see "
           "DROPPED_FOR_SESSION",
}


# --------------------------------------------------------------------------- helpers
def blocks_of(t):
    if len(t) == 0:
        return np.zeros(0, dtype=int)
    return np.concatenate(([0], np.cumsum(np.diff(t) >= GAP_S)))


def main_block(df):
    """Every CSV carries two sentinel rows (t = 0 s and t = 30 s) outside the plasma
    phase; the discharge itself is one contiguous block. That block is also the window a
    microsecond fetch would request, so everything is measured on it.

    The sentinels are dropped by what they ARE -- edge rows with neither CES target -- and
    not by the GAP_S gap alone. Gap is not sufficient: in #31937 the discharge begins at
    0.402 s, so the t = 0 sentinel sits 0.402 s away, inside GAP_S, and was absorbed into
    the main block. That put `t_start = 0.000` on the fetch request: 0.4 s of time that
    exists in no diagnostic, asked for because a padding row looked contiguous."""
    tgt = [c for c in ("CES_TI", "CES_VT") if c in df.columns]
    if tgt:
        observed = df[tgt].notna().any(axis=1).to_numpy()
        if observed.any():
            first, last = observed.argmax(), len(observed) - 1 - observed[::-1].argmax()
            df = df.iloc[first:last + 1].reset_index(drop=True)
    t = df["time"].to_numpy(float)
    blk = blocks_of(t)
    if len(blk) == 0:
        return df
    return df.loc[blk == int(np.argmax(np.bincount(blk)))].reset_index(drop=True)


def held_mask(values, block):
    out = np.zeros(len(values), dtype=bool)
    prev_val, prev_blk = np.nan, -1
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        if prev_blk == block[i] and not np.isnan(prev_val) and v == prev_val:
            out[i] = True
        else:
            prev_val, prev_blk = v, block[i]
    return out


def robust_scale(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) >= 4 else np.nan


def rolling_rms(x, win=ENV_WIN):
    x2 = np.where(np.isfinite(x), np.asarray(x, float) ** 2, np.nan)
    s = pd.Series(x2).rolling(win, center=True, min_periods=max(3, win // 2)).mean()
    return np.sqrt(s.to_numpy() * 2.0)   # E[x^2] = A^2/2 for a uniform random phase


def target_stats(raw, block, prefix, spike_cut):
    n = len(raw)
    vals = np.asarray(raw, float).copy()
    obs = np.isfinite(vals)
    d = {f"{prefix}_obs_frac": float(obs.mean()) if n else 0.0}
    if spike_cut:
        spikes = obs & (vals > spike_cut)
        d[f"{prefix}_spike_frac"] = float(spikes.mean()) if n else 0.0
        vals[spikes] = np.nan
    else:
        d[f"{prefix}_spike_frac"] = 0.0
    hm = held_mask(vals, block)
    d[f"{prefix}_held_frac"] = float(hm.mean()) if n else 0.0
    vals[hm] = np.nan
    ok = np.isfinite(vals)
    d[f"{prefix}_clean_n"] = int(ok.sum())
    d[f"{prefix}_clean_frac"] = float(ok.mean()) if n else 0.0
    if ok.sum() >= 5:
        v = vals[ok]
        d[f"{prefix}_std"] = float(np.std(v))
        d[f"{prefix}_range"] = float(np.percentile(v, 95) - np.percentile(v, 5))
        idx = np.flatnonzero(ok)
        same = block[idx[1:]] == block[idx[:-1]]
        dv, dt = np.diff(v)[same], np.diff(idx)[same]
        adj = dt <= 2
        if adj.sum() >= 5:
            step = np.abs(dv[adj])
            sc = robust_scale(v)
            d[f"{prefix}_transient"] = float(np.percentile(step, 95) / sc) if sc and sc > 0 else np.nan
        else:
            d[f"{prefix}_transient"] = np.nan
    else:
        d[f"{prefix}_std"] = d[f"{prefix}_range"] = d[f"{prefix}_transient"] = np.nan
    return d, vals


def group_stats(df, cols, block, prefix):
    d = {f"{prefix}_n_ch": len(cols)}
    if not cols:
        return d
    X = df[cols].to_numpy(float)
    stds, ac1s, hf = [], [], []
    for j in range(X.shape[1]):
        x = X[:, j]
        ok = np.isfinite(x)
        if ok.sum() < 10:
            continue
        xv, b = x[ok], block[ok]
        s = float(np.std(xv))
        stds.append(s)
        same = b[1:] == b[:-1]
        a, c = xv[:-1], xv[1:]
        if same.sum() > 10 and np.std(a[same]) > 0 and np.std(c[same]) > 0:
            ac1s.append(float(np.corrcoef(a[same], c[same])[0, 1]))
        if s > 0:
            hf.append(float(np.std(np.diff(xv)) / (s * np.sqrt(2))))
    d[f"{prefix}_dead_ch"] = int(sum(1 for s in stds if s <= 1e-9))
    d[f"{prefix}_ac1"] = float(np.mean(ac1s)) if ac1s else np.nan
    d[f"{prefix}_hf_ratio"] = float(np.median(hf)) if hf else np.nan
    g = np.nanmean(X, axis=1)
    ok = np.isfinite(g)
    if ok.sum() > 20:
        gg = g[ok]
        sc = robust_scale(gg)
        d[f"{prefix}_dyn_range"] = float((np.percentile(gg, 95) - np.percentile(gg, 5)) / sc) \
            if sc and sc > 0 else np.nan
    else:
        d[f"{prefix}_dyn_range"] = np.nan
    return d


def crash_events(sig, t):
    """Repeated fast negative excursions (ELM-crash proxy) and their spacing."""
    s = pd.Series(sig).rolling(9, center=True, min_periods=3).median().to_numpy()
    resid = sig - s
    sc = robust_scale(resid)
    if not np.isfinite(sc) or sc <= 0:
        return 0, np.nan
    idx = np.flatnonzero(resid < -4 * sc)
    if len(idx) < 3:
        return int(len(idx)), np.nan
    gaps = np.diff(t[idx])
    gaps = gaps[gaps > 0.005]
    return int(len(idx)), float(np.median(gaps)) if len(gaps) else np.nan


def level_step(sig, t):
    """Largest sustained step in the running level (L->H-like transition)."""
    s = pd.Series(sig).rolling(15, center=True, min_periods=5).median().to_numpy()
    ok = np.isfinite(s)
    if ok.sum() < 40:
        return np.nan
    s, tt = s[ok], t[ok]
    n = len(s)
    best = 0.0
    for i in range(int(0.1 * n), int(0.9 * n)):
        v = abs(np.median(s[i:i + 25]) - np.median(s[max(0, i - 25):i]))
        best = max(best, v)
    denom = np.percentile(s, 95) - np.percentile(s, 5)
    return float(best / denom) if denom > 0 else np.nan


# ----------------------------------------------------------------------------- scan
def scan_file(path):
    df = main_block(pd.read_csv(path))
    name = os.path.basename(path)
    t = df["time"].to_numpy(float)
    blk = blocks_of(t)
    rec = {"shot": int(name[1:].split(".")[0]), "file": name, "n_rows": len(df),
           "t_start": float(t.min()) if len(t) else np.nan,
           "t_end": float(t.max()) if len(t) else np.nan,
           "t_span": float(t.max() - t.min()) if len(t) else np.nan}
    ti_d, ti_clean = target_stats(df["CES_TI"].to_numpy(), blk, "ti", TI_SPIKE_EV)
    vt_d, _ = target_stats(df["CES_VT"].to_numpy(), blk, "vt", None)
    rec.update(ti_d)
    rec.update(vt_d)

    bes_c = [c for c in df.columns if c.startswith("BES_")]
    ece_c = [c for c in df.columns if c.startswith("ECEI_")]
    mc_c = [c for c in df.columns if c.startswith("MC")]
    rec.update(group_stats(df, bes_c, blk, "bes"))
    rec.update(group_stats(df, ece_c, blk, "ecei"))

    bes = np.nanmean(df[bes_c].to_numpy(float), axis=1) if bes_c else np.zeros(len(df))
    n_crash, period = crash_events(bes, t)
    rec["bes_crash_n"], rec["bes_crash_period_s"] = n_crash, period
    rec["bes_step_rel"] = level_step(bes, t)

    mc = df[mc_c].to_numpy(float)
    rms_full, rms_trim, ac1s, kurt = [], [], [], []
    for j in range(mc.shape[1]):
        x = mc[:, j][np.isfinite(mc[:, j])]
        if len(x) < 30:
            continue
        rms_full.append(np.sqrt(np.mean(x ** 2)))
        rms_trim.append(np.sqrt(np.mean(x[np.argsort(np.abs(x))[:-5]] ** 2)))
        s = np.std(x)
        if s > 0:
            kurt.append(float(np.mean(((x - x.mean()) / s) ** 4) - 3.0))
            if np.std(x[:-1]) > 0 and np.std(x[1:]) > 0:
                ac1s.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
    rec["mc_rms"] = float(np.median(rms_full)) if rms_full else np.nan
    rec["mc_rms_trim_ratio"] = float(np.median(rms_trim) / np.median(rms_full)) if rms_full else np.nan
    rec["mc_kurt"] = float(np.median(kurt)) if kurt else np.nan
    rec["mc_ac1"] = float(np.mean(ac1s)) if ac1s else np.nan

    E = np.vstack([rolling_rms(mc[:, j]) for j in range(mc.shape[1])]) if mc.size else None
    if E is not None:
        env = np.nanmean(E, axis=0)
        med = np.nanmedian(env)
        hot = np.isfinite(env) & (env > 2 * med) if np.isfinite(med) and med > 0 \
            else np.zeros(len(env), bool)
        # count only stretches of >= 3 consecutive samples (30 ms)
        sustained, run = 0, 0
        for h in list(hot) + [False]:
            if h:
                run += 1
            else:
                if run >= 3:
                    sustained += run
                run = 0
        n_env = max(1, int(np.isfinite(env).sum()))
        rec["mc_sustained_frac"] = float(sustained / n_env)
        rec["mc_hot_frac"] = float(hot.mean())
        pair = []
        for i in range(E.shape[0]):
            for j in range(i + 1, E.shape[0]):
                m = np.isfinite(E[i]) & np.isfinite(E[j])
                if m.sum() > 20 and np.std(E[i][m]) > 0 and np.std(E[j][m]) > 0:
                    pair.append(float(np.corrcoef(E[i][m], E[j][m])[0, 1]))
        rec["mc_env_coh"] = float(np.mean(pair)) if pair else np.nan
        bes_env = pd.Series(np.abs(bes - pd.Series(bes).rolling(15, center=True, min_periods=5)
                                   .median().to_numpy())).rolling(5, center=True, min_periods=3) \
            .mean().to_numpy()
        m = np.isfinite(env) & np.isfinite(bes_env)
        rec["mc_bes_coupling"] = float(np.corrcoef(env[m], bes_env[m])[0, 1]) \
            if m.sum() > 30 and np.std(env[m]) > 0 and np.std(bes_env[m]) > 0 else np.nan
    return rec


def split_membership(d):
    files = sorted(glob.glob(str(DATA / "s*.csv")))
    f2s = {os.path.basename(f): int(os.path.basename(f)[1:].split(".")[0]) for f in files}
    for s in SEEDS:
        man_path = DATA / f".b1_w2cut_split_s{s}" / "split_manifest.json"
        if not man_path.exists():
            print(f"  (no frozen manifest for seed {s}: {man_path})")
            continue
        man = json.loads(man_path.read_text())
        role = {f2s[f]: k for k in ("train", "val", "test") for f in man[f"{k}_files"]}
        d[f"split_s{s}"] = d.shot.map(role)
    have = [f"split_s{s}" for s in SEEDS if f"split_s{s}" in d]
    d["n_test_seeds"] = sum((d[c] == "test").astype(int) for c in have) if have else 0
    return d


def test_skill(d):
    """Per-shot test skill of the paper's main model (seq_v2, 16 runs) and the W=2 control."""
    files = sorted(glob.glob(str(DATA / "s*.csv")))
    idx2shot = {i: int(os.path.basename(f)[1:].split(".")[0]) for i, f in enumerate(files)}
    recs = {}
    runs = [(f".b1_seqv2_s{s}_i{i}", "seq") for s in SEEDS for i in INITS]
    runs += [(f".b1_w2cut_s{s}", "win") for s in SEEDS]
    for run, label in runs:
        p = DATA / run / "comparison_errors_test.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        have = set(z.keys())
        for tgt in ("CES_TI", "CES_VT"):
            if f"{tgt}_shot" not in have:
                continue
            sh = z[f"{tgt}_shot"]
            for base in ("pchip", "persistence"):
                if f"{tgt}_se_{base}" not in have:
                    continue
                em, eb = z[f"{tgt}_se_model"], z[f"{tgt}_se_{base}"]
                for u in np.unique(sh):
                    m = sh == u
                    if m.sum() < 15 or float(eb[m].mean()) <= 0:
                        continue
                    key = (idx2shot[int(u)], f"{label}_{tgt}_vs_{base}")
                    recs.setdefault(key, []).append(1.0 - float(em[m].mean()) / float(eb[m].mean()))
    out = {}
    for (shot, col), v in recs.items():
        out.setdefault(shot, {})[col] = float(np.mean(v))
    return d.merge(pd.DataFrame(out).T, left_on="shot", right_index=True, how="left")


# ---------------------------------------------------------------------------- score
def score(d):
    def pct(col):
        return d[col].astype(float).rank(pct=True)

    def sweet(col, lo, hi):
        v = d[col].astype(float)
        s = np.ones(len(v))
        s = np.where(v < lo, np.clip(v / lo, 0, 1), s)
        s = np.where(v > hi, np.clip(1 - (v - hi) / (1.0 - hi + 1e-9), 0, 1), s)
        return pd.Series(s, index=d.index)

    d["pass_gate"] = (
        (d.n_rows >= 250) & (d.ti_clean_n >= 200) & (d.vt_clean_n >= 60)
        & (d.ti_clean_frac >= 0.85) & (d.ti_spike_frac <= 0.02)
        & (d.bes_dead_ch == 0) & (d.ecei_dead_ch == 0) & d.mc_rms.notna()
        & (d.bes_ac1 > 0.2) & (d.ecei_ac1 > 0.2)
    )
    # An MC RMS that collapses when the 5 largest samples are dropped is an electrical
    # spike, not a mode -- it must not buy a slot in the fetch list.
    d["artifact_free"] = (d.mc_rms_trim_ratio >= 0.60) & (d.mc_kurt <= 80)
    d["mc_value"] = (0.30 * pct("mc_rms") + 0.20 * pct("mc_sustained_frac")
                     + 0.20 * pct("mc_env_coh") + 0.15 * pct("mc_bes_coupling")
                     + 0.15 * pct("mc_rms_trim_ratio"))
    d["label_value"] = 0.35 * pct("ti_clean_n") + 0.40 * pct("vt_clean_n") + 0.25 * pct("t_span")
    d["crash_ok"] = ((d.bes_crash_period_s.between(0.03, 0.40)) & (d.bes_crash_n >= 8)).astype(float)
    d["diag_value"] = (0.22 * pct("bes_dyn_range") + 0.22 * pct("ecei_dyn_range")
                       + 0.18 * pct("bes_step_rel") + 0.12 * sweet("bes_hf_ratio", 0.25, 0.75)
                       + 0.12 * sweet("ecei_hf_ratio", 0.25, 0.75) + 0.14 * d.crash_ok)
    d["score"] = 0.35 * d.mc_value + 0.35 * d.label_value + 0.30 * d.diag_value
    return d


# ----------------------------------------------------------------------------- plot
def plot_final(d):
    fig, axes = plt.subplots(len(FINAL), 1, figsize=(13, 2.6 * len(FINAL)))
    for ax, (shot, role, why) in zip(axes, FINAL):
        df = main_block(pd.read_csv(DATA / f"s{shot}.csv"))
        t = df.time.to_numpy(float)
        mc = df[[c for c in df.columns if c.startswith("MC")]].to_numpy(float)
        env = np.nanmean(np.vstack([rolling_rms(mc[:, j]) for j in range(mc.shape[1])]), axis=0)
        bes = np.nanmean(df[[c for c in df.columns if c.startswith("BES_")]].to_numpy(float), axis=1)
        ece = np.nanmean(df[[c for c in df.columns if c.startswith("ECEI_")]].to_numpy(float), axis=1)
        ti = df.CES_TI.to_numpy(float)
        ti = np.where(ti > TI_SPIKE_EV, np.nan, ti)
        vt = df.CES_VT.to_numpy(float)

        def n01(x):
            lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
            return (x - lo) / max(1e-9, hi - lo)

        top = np.nanmax(env)
        ax.plot(t, env, color="tab:red", lw=1.0, label="MC envelope")
        ax.plot(t, n01(bes) * top, color="tab:blue", lw=0.7, alpha=0.6, label="BES (scaled)")
        ax.plot(t, n01(ece) * top, color="tab:green", lw=0.7, alpha=0.6, label="ECEI (scaled)")
        ax2 = ax.twinx()
        ax2.plot(t, ti / 1000, color="k", lw=1.4, label="CES_TI [keV]")
        ax3 = ax.twinx()
        ax3.spines.right.set_position(("axes", 1.06))
        m = np.isfinite(vt)
        ax3.plot(t[m], vt[m], color="tab:orange", lw=1.1, alpha=0.9, label="CES_VT [km/s]")
        ax.set_title(f"#{shot}  [{role}]  {why}", fontsize=9, loc="left")
        for a in (ax, ax2, ax3):
            a.tick_params(labelsize=7)
        ax.set_ylabel("MC env / fast", fontsize=7)
        ax2.set_ylabel("T_i [keV]", fontsize=7)
        ax3.set_ylabel("V_rot [km/s]", fontsize=7)
        if shot == FINAL[0][0]:
            ax.legend(fontsize=6, loc="upper left")
    fig.suptitle("Shots selected for a microsecond-resolution raw fetch", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(HERE / "FINAL_10.png", dpi=100)
    plt.close(fig)


def main():
    files = sorted(glob.glob(str(DATA / "s*.csv")))
    if not files:
        raise SystemExit(f"no shot CSVs under {DATA}. Point CES_DATA_DIR at the real folder.")
    print(f"scanning {len(files)} shot files from {DATA}")
    rows = []
    for i, f in enumerate(files, 1):
        rows.append(scan_file(f))
        if i % 100 == 0:
            print(f"  {i}/{len(files)}")
    d = pd.DataFrame(rows).sort_values("shot").reset_index(drop=True)
    d.to_csv(HERE / "shot_metrics.csv", index=False, encoding="utf-8")

    d = split_membership(d)
    d = test_skill(d)
    d = score(d)
    d.sort_values("score", ascending=False).to_csv(HERE / "shot_scored.csv", index=False,
                                                   encoding="utf-8")

    ok = d[d.pass_gate & d.artifact_free].sort_values("score", ascending=False).reset_index(drop=True)
    print(f"\ngate-passing: {int(d.pass_gate.sum())}/{len(d)}   "
          f"artifact-free among them: {len(ok)}")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    show = ["shot", "split_s42", "n_test_seeds", "t_span", "ti_clean_n", "vt_clean_n", "mc_rms",
            "mc_rms_trim_ratio", "mc_kurt", "mc_sustained_frac", "mc_env_coh", "mc_bes_coupling",
            "bes_step_rel", "bes_crash_period_s", "mc_value", "label_value", "diag_value", "score"]
    for role in ("test", "val", "train"):
        print(f"\n=== artifact-free ranking, split_s42 = {role} (top 10) ===")
        print(ok[ok.split_s42 == role][show].head(10).round(3).to_string(index=False))

    sel = d[d.shot.isin([s for s, _, _ in FINAL])].copy()
    sel["role"] = sel.shot.map({s: r for s, r, _ in FINAL})
    sel["why"] = sel.shot.map({s: w for s, _, w in FINAL})
    sel = sel.set_index("shot").loc[[s for s, _, _ in FINAL]].reset_index()
    sel.to_csv(HERE / "FINAL_10.csv", index=False, encoding="utf-8")
    print("\n=== the selected 10 ===")
    print(sel[show + ["t_start", "t_end", "role"]].round(3).to_string(index=False))
    print("\nrank inside the artifact-free pool (published shots often fall outside the\n"
          "quality gate because CES_VT was never measured -- that is expected, not a defect):")
    for s in sel.shot:
        idx = ok.index[ok.shot == s]
        pos = f"{int(idx[0]) + 1}/{len(ok)}" if len(idx) else "outside the gated pool"
        note = " [published]" if int(s) in CONFIRMED_SHOTS else ""
        print(f"  #{s}: {pos}{note}")
    print("\npublished shots deliberately left out:")
    for s, why in REJECTED_PUBLISHED.items():
        print(f"  #{s}: {why}")
    print("\ndropped to remove same-session pairs:")
    for s, why in DROPPED_FOR_SESSION.items():
        print(f"  #{s}: {why}")
    picked = sorted(int(x) for x in sel.shot)
    gaps = [(picked[i], picked[i + 1], picked[i + 1] - picked[i]) for i in range(len(picked) - 1)]
    print(f"\nsmallest shot-number gap in the final list: {min(g for _, _, g in gaps)} "
          f"({[f'{a}->{b}:{g}' for a, b, g in gaps if g == min(x for _, _, x in gaps)]})")
    roles = sel.split_s42.value_counts().to_dict()
    assert all(sel.role == sel.split_s42), "role must equal the frozen s42 split membership"
    print(f"split (unchanged from the frozen s42 manifest): {roles}")
    plot_final(d)
    print(f"\nwrote {HERE / 'FINAL_10.csv'} and FINAL_10.png")


if __name__ == "__main__":
    main()
