"""B.76 (PREREGISTRATION_B76.md with Addendum 1): shared paths, constants, the per-chord history inputs under an
availability rule, and the per-shot row build.

Imports the MAST loaders of experiments/mast/ (edits nothing) and baselines_interpolation.py.  Caches under data/.b76/,
written whole and then renamed.  CPU only, two threads per process.  Run every stage from the repo root with `py`.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(EXP / "mast"))
import run_gate as rg  # noqa: E402
import run_coupling as rc  # noqa: E402
import explore_fast as ef  # noqa: E402
import explore_ped as ep  # noqa: E402

sys.path.insert(1, str(REPO / "ces_prediction"))
import baselines_interpolation as BI  # noqa: E402

DATA = REPO / "data"
CACHE = DATA / ".b76"
SHOTS = CACHE / "shots"
FITS = CACHE / "fits"
CHK = CACHE / "checks"
PAPER = REPO / "docs" / "paper"
OUT_JSON, OUT_MD, CHECKS_MD = PAPER / "b76_mast_availability.json", PAPER / "b76_mast_availability.md", PAPER / "b76_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B76.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B76_ADDENDUM1.md"

SEED = 20261176
TREE_SEED = 20261116
N_BOOT = 20_000
STEP = rg.STEP_MS                 # 5 ms
TOL_MS = 0.5
REL, FRAC = rg.POPS["W"], 0.5
DELAYS = {"D5": 5.0, "D15": 15.0, "D25": 25.0, "D50": 50.0, "D100": 100.0, "D150": 150.0}
RULES = tuple(DELAYS)
DA_RULES = ("D5", "D100")
TARGETS = ("F", "L5")
BAND = (0.015, 0.05)
SEEDS = (1, 2, 3)
NF = 5
HIST_NAMES = ("age", "H", "G-H", "|G-H|", "mean10", "sd10", "H-mean10", "inc", "n10", "rho")
TS_NAMES = ("te_t", "ne_t", "te_a", "ne_a", "dte", "dne", "lnrte", "lnrne", "dedge_t")
DA_NAMES = ("da_t", "dmx_t", "da_a", "dmx_a", "dda", "ddmx")
T0_ = _time.time()


def log(m):
    print(f"[{_time.time() - T0_:7.0f}s] {m}", flush=True)


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, path)


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------ shots, geometry, sessions
def listed_shots():
    return [int(x) for x in rc.SHOTS_TXT.read_text().split()]


def meta_tables():
    import pandas as pd
    cat = pd.read_parquet(rg.META, columns=["shot_id", "radii_magnetic__radius_efit", "radii_router_radius_efit",
                                            "radii_router_radius_d_alpha", "shot_experiment_date", "timestamp"])
    return cat.set_index("shot_id")


def geo_of(cat, s):
    """explore_ped.build's geometry lookup (copied)."""
    if s not in cat.index:
        return None
    ax = float(cat.loc[s, "radii_magnetic__radius_efit"])
    for col in ("radii_router_radius_efit", "radii_router_radius_d_alpha"):
        ro = float(cat.loc[s, col])
        if np.isfinite(ax) and np.isfinite(ro) and ro > ax:
            return ax, ro
    return None


def session_of(cat, s):
    """Calendar day (Addendum 1 section 1): shot_experiment_date, else the date of timestamp."""
    import pandas as pd
    d = cat.loc[s, "shot_experiment_date"]
    if pd.isna(d):
        d = cat.loc[s, "timestamp"]
    return str(pd.Timestamp(d).date())


# ------------------------------------------------------------------------------------------ history inputs
def chord_history(tf_ms, y, g, rows_j, delay_ms):
    """Inputs of rows (block frames rows_j) of one chord under the rule with this delay.  tf_ms: frame times of the
    block frames (ms, from the block start), y: values, g: good mask (the chord's arrays from the block start onward).
    Reads y and g only at frames whose time is at most the row's limit (t_j - delay + tolerance).
    Returns dict of arrays over rows: H, age, M5, G, th (n x 5), anc (frame or -1), nvis."""
    obs = np.flatnonzero(g)
    tobs = tf_ms[obs]
    n = len(rows_j)
    H, age, M5, Gp = (np.full(n, np.nan) for _ in range(4))
    th = np.full((n, 5), np.nan)
    anc = np.full(n, -1, np.int64)
    nvis = np.zeros(n, np.int64)
    lim = tf_ms[rows_j] - delay_ms
    p_all = np.searchsorted(tobs, lim + TOL_MS, side="right")
    gp_cache = {}
    for q in range(n):
        j = int(rows_j[q])
        p = int(p_all[q])
        nvis[q] = p
        if p == 0:
            continue
        last = int(obs[p - 1])
        if last >= j or tf_ms[last] > lim[q] + TOL_MS:
            raise SystemExit(f"FATAL: a visible value at or after frame {j}")
        H[q] = y[last]
        age[q] = tf_ms[j] - tf_ms[last]
        anc[q] = last
        v = y[obs[max(p - 10, 0):p]]
        M5[q] = np.median(v[-5:])
        if len(v) >= 3:
            m = v.mean()
            th[q] = (m, v.std(), v[-1] - m, v[-1] - v[-2], len(v))
        key = (p, j)
        if key not in gp_cache:
            g16 = obs[max(p - 16, 0):p]
            gp_cache[key] = BI.predict_gp_causal(tf_ms[g16] / 1000.0, y[g16], tf_ms[j] / 1000.0)
        Gp[q] = gp_cache[key]
    return {"H": H, "age": age, "M5": M5, "G": Gp, "th": th, "anc": anc, "nvis": nvis}


def history_matrix(I, rho):
    """The ten history scalars (Addendum 1 section 4)."""
    return np.column_stack([I["age"], I["H"], I["G"] - I["H"], np.abs(I["G"] - I["H"]), I["th"][:, 0], I["th"][:, 1],
                            I["th"][:, 2], I["th"][:, 3], I["th"][:, 4], rho])


# ------------------------------------------------------------------------------------------ per-shot pieces
def thomson_grids(ex, Rc, tf_abs, r_axis):
    """Frame-window Thomson at the chords: TEw, NEw (L x C), the input edge distance (L x C), the band distance (L x C,
    explore_ped.add_extra), the number of laser samples finite at the chord in the window (L x C) and of laser times."""
    t, R, te, ne, g = ex["ts"]
    TE = np.stack([ep.to_chords(Rc, R[j][np.isfinite(te[j])], te[j][np.isfinite(te[j])]) for j in range(len(t))])
    NE = np.stack([ep.to_chords(Rc, R[j][np.isfinite(ne[j])], ne[j][np.isfinite(ne[j])]) for j in range(len(t))])
    TEw = ep.window_mean(t, TE, tf_abs, 0)
    NEw = ep.window_mean(t, NE, tf_abs, 0) / 1e19
    edge = np.array([ep.edge_radius(R[j], ne[j], r_axis) for j in range(len(t))])
    # band: median of the profile edges within three frames (explore_ped.add_extra)
    lo = np.searchsorted(t, tf_abs - 7.5, side="left")
    hi = np.searchsorted(t, tf_abs + 7.5, side="right")
    eb = np.full(len(tf_abs), np.nan)
    for i in range(len(tf_abs)):
        if hi[i] > lo[i] and np.isfinite(edge[lo[i]:hi[i]]).any():
            eb[i] = np.nanmedian(edge[lo[i]:hi[i]])
    ew = ep.window_mean(t, edge[:, None], tf_abs, 0)[:, 0]
    dband = eb[:, None] - Rc[None, :]
    dedge = ew[:, None] - Rc[None, :]
    lo = np.searchsorted(t, tf_abs - 2.5, side="left")
    hi = np.searchsorted(t, tf_abs + 2.5, side="left")
    cf = np.vstack([np.zeros((1, len(Rc)), np.int64), np.cumsum(np.isfinite(TE), axis=0)])
    nts = cf[hi] - cf[lo]
    return TEw, NEw, dedge, dband, nts, hi - lo, g


def dalpha_grid(shot, t_a, L):
    drv = rc.load_drivers(shot)
    m = np.full(L, np.nan)
    mx = np.full(L, np.nan)
    if "dalpha" in drv:
        tdv, dv = drv["dalpha"]
        m0 = rc.frame_means(tdv, dv, t_a, 0, L - 1)
        mx0 = ef.frame_max(tdv, dv, t_a, 0, L - 1)
        sc = np.nanmedian(np.abs(m0)) if np.isfinite(m0).any() else np.nan
        if np.isfinite(sc) and sc > 0:
            m, mx = m0 / sc, mx0 / sc
    return m, mx


def prepare_shot(shot, cat):
    """Shot filters (Addendum 1 section 1).  Returns (status, d, p, a, chords, ex)."""
    geo = geo_of(cat, shot)
    d = rg.load_shot(shot)
    if d is None:
        return "no_act_file", None
    if geo is None:
        return "no_geometry", None
    tt = d["time_ms"]
    dd = np.diff(tt[np.isfinite(tt)])
    if not (len(dd) and abs(float(np.median(dd)) - STEP) <= 0.1):
        return "median_step", None
    if d["R"].shape[0] != d["T"].shape[1]:
        return "radius_shape", None
    p = ef.prepare(d, "T_i", REL, FRAC)
    if p["status"] != "ok":
        return "prepare_" + p["status"], None
    if p["n_bad_dt"] > 0 or not p["t_finite"]:
        return "block_steps", None
    if not (ep.EXTRA / f"{shot}.npz").exists():
        return "no_extra_file", None
    ex = ep.load_extra(shot)
    if "ts" not in ex:
        return "no_thomson", None
    a, b = rc.block_of(d, "T_i", REL)
    good = rg.good_masks(d, REL)["T_i"]
    chords = np.nonzero(good[a:b + 1].mean(0) >= FRAC)[0]
    if not (b - a + 1 == p["L"] and np.array_equal(good[a:b + 1][:, chords], p["G"])):
        raise SystemExit(f"FATAL: shot {shot}: block or chords differ from explore_fast.prepare")
    return "ok", {"d": d, "p": p, "a": a, "b": b, "chords": chords, "ex": ex, "geo": geo, "good": good}


def build_shot(shot, cat):
    """Candidate rows of one shot with their inputs under every rule (Addendum 1 sections 2-5)."""
    st, P = prepare_shot(shot, cat)
    if st != "ok":
        return {"status": np.array(st)}
    d, p, a, chords, ex, (ax, ro) = P["d"], P["p"], P["a"], P["chords"], P["ex"], P["geo"]
    L, G = p["L"], p["G"]
    y = d["T"][a:a + L][:, chords].astype(np.float64)
    sig = d["Te"][a:a + L][:, chords].astype(np.float64)
    Rc = d["R"][chords].astype(np.float64)
    rho = (Rc - ax) / (ro - ax)
    t_a = float(d["time_ms"][a])
    tf = STEP * np.arange(L)
    tf_abs = t_a + tf
    TEw, NEw, dedge, dband, nts, nlas, sysname = thomson_grids(ex, Rc, tf_abs, ax)
    da, dmx = dalpha_grid(shot, t_a, L)
    band = (dband >= BAND[0]) & (dband < BAND[1])
    ts_t = np.isfinite(TEw) & np.isfinite(NEw)
    cand = G & band & ts_t & np.isfinite(dedge)
    counts = np.array([int(G.sum()), int((G & band).sum()), int((G & band & ts_t).sum()), int(cand.sum())], np.int64)
    jj, cc = np.nonzero(cand)
    o = np.lexsort((jj, cc))           # by chord, then frame
    jj, cc = jj[o], cc[o]
    n = len(jj)
    out = {"status": np.array("ok"), "counts": counts, "L": np.int64(L), "t_a": np.float64(t_a),
           "ts_sys": np.array(sysname), "n_chords": np.int64(len(chords)),
           "shot": np.full(n, shot, np.int32), "c": cc.astype(np.int16), "craw": chords[cc].astype(np.int16),
           "R": Rc[cc], "rho": rho[cc], "j": jj.astype(np.int32), "t": tf_abs[jj], "y": y[jj, cc], "sig": sig[jj, cc],
           "prevgood": np.where(jj >= 1, G[np.maximum(jj - 1, 0), cc], False),
           "te_t": TEw[jj, cc], "ne_t": NEw[jj, cc], "dedge_t": dedge[jj, cc], "dband": dband[jj, cc],
           "nts": nts[jj, cc].astype(np.int16), "nlas": nlas[jj].astype(np.int16), "da_t": da[jj], "dmx_t": dmx[jj]}
    # L5
    later = np.full((n, 5), np.nan)
    for k in range(1, 6):
        jk = jj + k
        ok = jk < L
        jkc = np.minimum(jk, L - 1)
        gk = ok & G[jkc, cc]
        later[gk, k - 1] = y[jkc[gk], cc[gk]]
    nl = np.isfinite(later).sum(1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l5 = np.nanmedian(later, 1)
    out["L5"] = np.where(nl >= 3, l5, np.nan)
    out["nL5"] = nl.astype(np.int8)
    # rules
    for r, dl in DELAYS.items():
        H, age, M5, Gp, anc, nvis = (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan),
                                     np.full(n, -1, np.int64), np.zeros(n, np.int64))
        th = np.full((n, 5), np.nan)
        for c in np.unique(cc):
            q = np.flatnonzero(cc == c)
            I = chord_history(tf, y[:, c], G[:, c], jj[q], dl)
            H[q], age[q], M5[q], Gp[q], th[q], anc[q], nvis[q] = I["H"], I["age"], I["M5"], I["G"], I["th"], I["anc"], I["nvis"]
        ok = anc >= 0
        ac = np.maximum(anc, 0)
        out[f"{r}_H"], out[f"{r}_age"], out[f"{r}_M5"], out[f"{r}_G"], out[f"{r}_th"] = H, age, M5, Gp, th
        out[f"{r}_anc"], out[f"{r}_nvis"] = anc.astype(np.int32), nvis.astype(np.int32)
        out[f"{r}_te_a"] = np.where(ok, TEw[ac, cc], np.nan)
        out[f"{r}_ne_a"] = np.where(ok, NEw[ac, cc], np.nan)
        out[f"{r}_da_a"] = np.where(ok, da[ac], np.nan)
        out[f"{r}_dmx_a"] = np.where(ok, dmx[ac], np.nan)
        out[f"{r}_nts_a"] = np.where(ok, nts[ac, cc], 0).astype(np.int16)
    return out


# ------------------------------------------------------------------------------------------ learner inputs from the row table
def ts_matrix(R, r):
    te_t, ne_t, te_a, ne_a = R["te_t"], R["ne_t"], R[f"{r}_te_a"], R[f"{r}_ne_a"]
    with np.errstate(all="ignore"):
        rte, rne = te_t / te_a, ne_t / ne_a
        lr = np.where(rte > 0, np.log(np.where(rte > 0, rte, 1.0)), np.nan)
        ln_ = np.where(rne > 0, np.log(np.where(rne > 0, rne, 1.0)), np.nan)
    return np.column_stack([te_t, ne_t, te_a, ne_a, te_t - te_a, ne_t - ne_a, lr, ln_, R["dedge_t"]])


def da_matrix(R, r):
    return np.column_stack([R["da_t"], R["dmx_t"], R[f"{r}_da_a"], R[f"{r}_dmx_a"], R["da_t"] - R[f"{r}_da_a"],
                            R["dmx_t"] - R[f"{r}_dmx_a"]])


def hist_matrix(R, r):
    return np.column_stack([R[f"{r}_age"], R[f"{r}_H"], R[f"{r}_G"] - R[f"{r}_H"], np.abs(R[f"{r}_G"] - R[f"{r}_H"]),
                            R[f"{r}_th"], R["rho"]])


def load_rows(keys=None):
    z = np.load(CACHE / "rows.npz", allow_pickle=False)
    return {k: z[k] for k in (keys or z.files)}


def units():
    """(list of units, design table {shots, session, fifth})."""
    u = json.loads((CACHE / "units.json").read_text(encoding="utf-8"))
    return u["units"], u["table"]
