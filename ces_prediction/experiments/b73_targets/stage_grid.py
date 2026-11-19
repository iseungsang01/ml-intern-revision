"""B.73 stage 1: the cut (3000 eV), held-free grid of seq_data.load_grid_files, concatenated and cached.

    py ces_prediction/experiments/b73_targets/stage_grid.py
Output: data/.b73/grid.npz  (names, shot, off[n+1], time, ti, fast[BES 9, ECEI 4] float32, seg, fidx, grow)
"""
from __future__ import annotations

import os
import sys

import common as C

import numpy as np  # noqa: E402


def pin_env():
    for k in [k for k in os.environ if k.startswith("CES_")]:
        del os.environ[k]
    os.environ.update({"CES_DATA_DIR": str(C.DATA), "CES_DROP_STUCK_TARGETS": "1", "CES_SEQ_NORM_EXPANDING": "0",
                       "CES_SEQ_ZERO_MC": "0", "CES_PER_SHOT_NORM": "0", "CES_TI_SPIKE_CUT_EV": "3000"})


def main():
    out = C.CACHE / "grid.npz"
    if out.exists():
        C.log("grid: cached")
        return
    pin_env()
    sys.path.insert(0, str(C.EXP / "seq"))
    sys.path.insert(1, str(C.REPO / "ces_prediction"))
    from seq_data import load_grid_files
    C.log("loading the grid (seq_data.load_grid_files, held removed, cut 3000 eV)")
    g, dims = load_grid_files(C.DATA, True, C.CUT_EV)
    if dims["bes"] != C.NB or dims["ecei"] != C.NE:
        raise SystemExit(f"FATAL: dims {dims}")
    names = sorted(g)
    base = np.load(C.DATA / ".b64" / "base.npz", allow_pickle=True)
    if [str(x) for x in base["names"]] != names:
        raise SystemExit("FATAL: base.npz was built on another file list")
    lens = np.array([len(g[n]) for n in names])
    off = np.concatenate(([0], np.cumsum(lens)))
    tm = np.concatenate([g[n][:, 0].astype(np.float64) for n in names])
    ti = np.concatenate([g[n][:, 1].astype(np.float64) for n in names])
    fast = np.concatenate([g[n][:, 3:3 + C.NB + C.NE] for n in names]).astype(np.float32)
    fidx = np.repeat(np.arange(len(names)), lens)
    grow = np.concatenate([np.arange(L) for L in lens])
    seg = np.zeros(len(tm), np.int64)
    s0 = 0
    for i in range(len(names)):
        a, b = off[i], off[i + 1]
        dt = np.diff(tm[a:b])
        if np.any(dt <= 0):
            raise SystemExit(f"FATAL: {names[i]}: time not increasing")
        seg[a:b] = s0 + np.concatenate(([0], np.cumsum(dt >= C.SEG_GAP)))
        s0 = seg[b - 1] + 1
    C.save_npz(out, names=np.array(names), shot=np.array([C.shot_of(n) for n in names]), off=off, time=tm, ti=ti,
               fast=fast, seg=seg, fidx=fidx, grow=grow)
    C.log(f"grid: {len(names)} files, {len(tm):,} rows written")


if __name__ == "__main__":
    main()
