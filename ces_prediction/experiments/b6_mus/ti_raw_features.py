"""B.6-T feature extraction: raw BES / ECEI burst energy in the causal 10 ms window -> T_i rows.

PREREGISTRATION_B6T.md sec. 2. The 100 Hz shot CSVs carry ONE sample of each fast channel
per 10 ms row; an ELM crash or any sub-10 ms burst inside the window is seen only if the
snapshot happens to land on it. The delivered 500 kHz streams see every burst. For each
of the 13 fast channels (9 BES + 4 ECEI) and each grid row t:

    1 ms energy      e_k = std of the raw samples in the k-th 1 ms slice of the stream
    robust z         z_k = (e_k - median_shot) / (1.4826 * MAD_shot)      (per shot, per channel)
    window           the ten 1 ms slices in (t - 10 ms, t]  -- strictly causal
    features         log1p(max(z, 0)) over the window, log1p(mean(z, 0)) over the window

plus one timing channel: log1p(milliseconds since the last burst onset on BES_0206), a
burst onset being the first 1 ms slice with z >= 8 after a slice below 8, onsets closer
than 5 ms merged (the detector PROJECT_KNOWLEDGE / THESIS sec. 8bd used on the four test
discharges). K = 2 * 13 + 1 = 27. Rows outside the delivered raw span are zeros. No channel
is z-score exempt: the loader's per-shot z-scoring applies to all 27.

Layout written to `data/.b6_features_ti/mus_features_s{shot}.npz` (the loader's file name
convention) with `feature_meta.json` beside it.
"""

from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[0] / "hires_shots"))

FAST = ("BES_0105", "BES_0106", "BES_0107", "BES_0205", "BES_0206", "BES_0207",
        "BES_0305", "BES_0306", "BES_0307",
        "ECEI_GT1103", "ECEI_GT1203", "ECEI_GT1303", "ECEI_GT1403")
BURST_CHANNEL = "BES_0206"
BURST_Z = 8.0
MERGE_S = 0.005
WINDOW_MS = 10


def ms_energy(t, x, fs):
    n1 = int(round(fs * 1e-3))
    m = len(x) // n1
    e = np.std(x[:m * n1].reshape(m, n1), axis=1)
    tt = t[:m * n1].reshape(m, n1)[:, 0]          # slice start times
    med = np.median(e)
    mad = np.median(np.abs(e - med)) + 1e-12
    return tt, (e - med) / (1.4826 * mad)


def burst_onsets(tt, z, thr=BURST_Z, merge_s=MERGE_S):
    on = np.flatnonzero((z[1:] >= thr) & (z[:-1] < thr)) + 1
    keep = []
    for i in on:
        if not keep or tt[i] - tt[keep[-1]] > merge_s:
            keep.append(i)
    return tt[np.array(keep, dtype=int)] if keep else np.array([])


def window_features(tt, z, grid, cover):
    """(log1p max z, log1p mean z) over the ten 1 ms slices ending at each grid row."""
    # slice k covers [tt[k], tt[k] + 1 ms); the window (t - 10 ms, t] holds the slices
    # whose END is <= t and START > t - 10 ms, i.e. indices [i - 10, i) with i = first
    # slice starting after t - 1 ms ... use searchsorted on slice starts with t - 1 ms.
    idx = np.searchsorted(tt, grid - 1e-3, side="right")
    mx = np.zeros(len(grid))
    mn = np.zeros(len(grid))
    for j, i in enumerate(idx):
        if cover[j] and i > 0:
            w = np.maximum(z[max(i - WINDOW_MS, 0):i], 0.0)
            mx[j] = np.log1p(w.max())
            mn[j] = np.log1p(w.mean())
    return mx, mn


def extract_shot(shot, grid, load_raw):
    feats = []
    cover = None
    for ch in FAST:
        (t, x), fs = load_raw(shot, ch)
        tt, z = ms_energy(t, x, fs)
        if cover is None:
            cover = (grid >= tt[0] + WINDOW_MS * 1e-3) & (grid <= tt[-1] + 1e-3)
        mx, mn = window_features(tt, z, grid, cover)
        feats += [mx, mn]
        if ch == BURST_CHANNEL:
            onsets = burst_onsets(tt, z)
            t_stream0 = tt[0]
    tsb = np.zeros(len(grid))
    for j, g in enumerate(grid):
        if cover[j]:
            prev = onsets[onsets <= g]
            tsb[j] = np.log1p(1e3 * (g - (prev[-1] if len(prev) else t_stream0)))
    feats.append(tsb)
    feat = np.vstack(feats).T.astype(np.float32)
    meta = {"k": feat.shape[1], "z_exempt_channels": [], "channels": list(FAST),
            "per_channel": ["log1p_max_z", "log1p_mean_z"], "timing": "log1p_ms_since_burst",
            "burst_channel": BURST_CHANNEL, "burst_z": BURST_Z, "merge_s": MERGE_S,
            "window_ms": WINDOW_MS, "slice_ms": 1}
    return feat, meta, cover


def main():
    from raw_loader import load_raw, RAW_READY
    from folds import TEST, POOL, COMPANIONS
    from mus_features import write_shot
    import pandas as pd
    if not RAW_READY:
        raise SystemExit("raw_loader is a stub.")
    root = HERE.parents[2]
    out_dir = root / "data" / ".b6_features_ti"
    for shot in (*TEST, *POOL, *COMPANIONS):
        grid = pd.read_csv(root / "data" / f"s{shot}.csv", usecols=["time"])["time"].to_numpy(float)
        feat, meta, cover = extract_shot(shot, grid, load_raw)
        write_shot(out_dir, shot, grid, feat, meta)
        print(f"[b6t] s{shot}: {feat.shape[0]} rows x {feat.shape[1]} ch, covered {int(cover.sum())}")
    print(f"[b6t] wrote {out_dir}")


if __name__ == "__main__":
    main()
