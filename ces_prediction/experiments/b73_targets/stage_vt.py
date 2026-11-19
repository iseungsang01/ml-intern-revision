"""B.73 (Addendum 1 of the session lead, A3): the rotation values as delivered (no held removal), aligned to the cached
grid, for the 'fresh V_rot' of A3.

    py ces_prediction/experiments/b73_targets/stage_vt.py
Output: data/.b73/vt_raw.npz (vt: float64 per grid row)
"""
from __future__ import annotations

import sys

import common as C
import stage_grid as SG

import numpy as np  # noqa: E402


def main():
    out = C.CACHE / "vt_raw.npz"
    if out.exists():
        C.log("vt: cached")
        return
    SG.pin_env()
    sys.path.insert(0, str(C.EXP / "seq"))
    sys.path.insert(1, str(C.REPO / "ces_prediction"))
    from seq_data import load_grid_files
    g, _ = load_grid_files(C.DATA, False, 0.0)
    G = C.grid()
    names = [str(x) for x in G["names"]]
    if sorted(g) != names:
        raise SystemExit("FATAL: file lists differ")
    vt = np.concatenate([g[n][:, 2].astype(np.float64) for n in names])
    tm = np.concatenate([g[n][:, 0].astype(np.float64) for n in names])
    if len(tm) != len(G["time"]) or np.max(np.abs(tm - G["time"])) > 1e-6:
        raise SystemExit("FATAL: the rows differ from the cached grid")
    C.save_npz(out, vt=vt)
    C.log(f"vt: {len(vt):,} rows written")


if __name__ == "__main__":
    main()
