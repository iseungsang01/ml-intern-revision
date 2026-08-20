"""Fetch windows for the literature-first list, on the same proxy the request doc uses.

There is no NBI channel in the CSVs, so "from when ohmic heating ends and the beams come
on" cannot be read directly. The one proxy the data offers is the same one the 2026-08-18
request used: CES rotation is only fittable while a beam is present, so the first CLEAN
CES_VT sample marks the beam phase. Where a shot has no usable rotation at all, that
proxy does not exist and the window falls back to the labelled dataset block.
"""
import os
import sys
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(os.environ.get("CES_DATA_DIR", HERE.parents[2] / "data"))
sys.path.insert(0, str(HERE))
from select_hires_shots import main_block, blocks_of, held_mask, TI_SPIKE_EV  # noqa: E402

SEL = [(31921, "test", "LIT"), (31873, "test", "LIT"), (31114, "test", "data"),
       (31686, "test", "data"),
       (31097, "pool", "LIT"), (31359, "pool", "LIT"), (31747, "pool", "LIT"),
       (32027, "pool", "LIT"), (31914, "pool", "data"), (31368, "pool", "data"),
       (31357, "comp", "LIT"), (31923, "comp", "LIT"),
       (32097, "alt", "data"), (31902, "alt", "data"), (31937, "alt", "data")]
MIN_VT = 30          # fewer clean rotation samples than this is not a beam phase marker


def A(x):
    return str(x).encode("ascii", "replace").decode("ascii")


def clean(df, col, blk, spike):
    v = df[col].to_numpy(float) if col in df.columns else np.full(len(df), np.nan)
    ok = np.isfinite(v)
    if spike:
        ok &= ~(ok & (v > TI_SPIKE_EV))
    ok &= ~held_mask(v, blk)
    return ok


rows = []
for shot, role, src in SEL:
    df = pd.read_csv(DATA / f"s{shot}.csv")
    df = main_block(df)
    t = df["time"].to_numpy(float)
    blk = blocks_of(t)
    ti_ok = clean(df, "CES_TI", blk, spike=True)
    vt_ok = clean(df, "CES_VT", blk, spike=False)
    lab = ti_ok | vt_ok
    d_lo, d_hi = float(t[lab].min()), float(t[lab].max())          # dataset window
    v_lo_try = float(t[vt_ok].min()) if vt_ok.any() else float("nan")
    v_hi_try = float(t[vt_ok].max()) if vt_ok.any() else float("nan")
    # A short-lived rotation fit does NOT mark a beam turn-on. #31357 fits V_rot for only
    # 0.64 s of a 3.98 s labelled block; reading that as "the beam came on at 6.33 s"
    # would cut the shot to 0.65 s and throw away 330 T_i labels -- and it is the paired
    # ERMP-ON control of #31359, so the comparison the paper designed needs the block.
    covers = ((v_hi_try - v_lo_try) / (d_hi - d_lo)) if vt_ok.any() and d_hi > d_lo else 0.0
    if vt_ok.sum() >= MIN_VT and covers >= 0.50:
        v_lo, v_hi = v_lo_try, v_hi_try
        # Start at the beam phase, as asked: the window opens where rotation first
        # fits. The END stays at the last label -- beams do not switch off when the
        # rotation fit stops, and T_i supervision past that point is the main target.
        lo, hi, basis = v_lo, d_hi, "V_rot onset"
        ti_outside = int((t[ti_ok] < v_lo).sum())
    else:
        v_lo = v_hi = float("nan")
        lo, hi, basis = d_lo, d_hi, "dataset block (no usable V_rot)"
        ti_outside = 0
    rows.append(dict(shot=shot, role=role, src=src, lo=lo, hi=hi, span=hi - lo,
                     basis=basis, v_lo=v_lo, v_hi=v_hi, d_lo=d_lo, d_hi=d_hi,
                     n_ti=int(ti_ok.sum()), n_vt=int(vt_ok.sum()), ti_outside=ti_outside))

r = pd.DataFrame(rows)
print(A(f"{'shot':>6} {'role':>5} {'src':>4} {'window [s]':>18} {'span':>6} "
        f"{'Ti':>5} {'Vrot':>5}  basis"))
for _, x in r.iterrows():
    print(A(f"{x.shot:>6} {x.role:>5} {x.src:>4} "
            f"{x.lo:>8.3f} - {x.hi:<7.3f} {x.span:>6.2f} "
            f"{x.n_ti:>5} {x.n_vt:>5}  {x.basis}"))
core = r[r.role.isin(("test", "pool"))]
comp = r[r.role == "comp"]
print(A(f"\ncore 10 (the request): {core.span.sum():.2f} s over {len(core)} shots"))
print(A(f"  + 2 companions:      {comp.span.sum():.2f} s"
        f"  -> {core.span.sum() + comp.span.sum():.2f} s total"))
print(A("alternates priced separately: "
        + ", ".join(f"#{int(x.shot)} {x.span:.2f}s"
                    for _, x in r[r.role == "alt"].iterrows())))
print(A("\nV_rot observed interval vs the dataset block (where they differ):"))
for _, x in r.iterrows():
    if np.isfinite(x.v_lo) and (abs(x.v_lo - x.d_lo) > 0.05 or abs(x.v_hi - x.d_hi) > 0.05):
        print(A(f"  #{int(x.shot)}: V_rot {x.v_lo:.3f}-{x.v_hi:.3f} vs dataset "
                f"{x.d_lo:.3f}-{x.d_hi:.3f}; T_i labels outside V_rot: {x.ti_outside}"))
r.to_csv(HERE / "fetch_windows.csv", index=False, encoding="utf-8")
print(A(f"\nwrote {HERE / 'fetch_windows.csv'}"))
