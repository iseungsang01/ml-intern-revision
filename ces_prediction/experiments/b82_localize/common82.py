"""B.82 (PREREGISTRATION_B82.md with Addendum 1): shared paths, constants, tau, the classes of c, the Mirnov and BES-column
scalars, the anchor rows of every grid row and the donor assignment of the swap control.

Reuses b81_controls/common81.py (which imports b75_availability/common75.py and b73_targets/common.py: threads 2, CUDA
hidden) and b78_sparse (frame index, learner matrices).  Caches under data/.b82/.  Run every stage from the repo root.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b81_controls"))
import common81 as B81  # noqa: E402  (imports common75, fits78; sets OMP/MKL/OPENBLAS threads to 2)

import numpy as np  # noqa: E402

K75 = B81.K75
F78 = B81.F78
C = K75.C
import common78 as M78  # noqa: E402  (on sys.path through common81)

CACHE = C.DATA / ".b82"
FITS = CACHE / "fits"
CHK = CACHE / "checks"
LOGS = CACHE / "logs"
PAPER = C.PAPER
OUT_JSON, OUT_MD, CHECKS_MD = PAPER / "b82_localize.json", PAPER / "b82_localize.md", PAPER / "b82_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B82.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B82_ADDENDUM1.md"
B75FITS, B79FITS, B81FITS = K75.FITS, C.DATA / ".b79" / "fits", B81.FITS

SEED = 20261182
X = "L5"
RULES = ("D650", "S")
SETOF = {"D10": "common_L5", "D650": "common_L5", "S": "S_L5"}
TOL_MS = 3.0                                   # tolerance of a stamp comparison (B.75), in ms
TAU_CLASSES = ("up to 100 ms", "above 100 and up to 300 ms", "above 300 ms", "no notch before")
C_CLASSES = ("lowest third", "middle third", "highest third", "c not finite")
MC_COLS = ("MC1T03", "MC1T16")
# BES channels of the grid in file order: BES_0105, 0106, 0107, 0205, 0206, 0207, 0305, 0306, 0307 (checked in
# stage_inputs82).  The last two digits name the radial column.
BES_NAMES = ("BES_0105", "BES_0106", "BES_0107", "BES_0205", "BES_0206", "BES_0207", "BES_0305", "BES_0306", "BES_0307")
BES_COLUMNS = {"TBi": (0, 3, 6), "TBm": (1, 4, 7), "TBo": (2, 5, 8)}   # columns 05, 06, 07
PART_C_UNITS = ("V1", "V3", "V4a")
PART_C_LEARNERS = ("TM", "TXM", "TXs", "TBi", "TBm", "TBo")
SEED_UNITS = ("V4a", "V4b", "V1_r2_k0", "V1_r2_k1", "V1_r2_k2", "V1_r2_k3", "V1_r2_k4")
SEEDS = tuple(range(1, 11))
log = C.log
save_npz, save_json, save_text, load_npz, load_keys = C.save_npz, C.save_json, C.save_text, C.load_npz, K75.load_keys


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def units():
    return B81.units()


def gbr(seed=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=C.TREE_SEED if seed is None else seed)


# ------------------------------------------------------------------------------------------ tau
def tau_ms(tm, gi, jS_g):
    """tau (ms) of D1 rows: stamp of row t minus the stamp of the first row after the most recent absent step that ends
    at or before t (B.75's S row j, so that row is j + 1); NaN when the block has no absent step before t."""
    out = np.full(len(gi), np.nan)
    has = jS_g >= 0
    out[has] = (tm[gi[has]] - tm[jS_g[has] + 1]) * 1000.0
    if np.any(out[has] < 0):
        raise SystemExit("FATAL: an absent step ends after its row")
    return out


def tau_class(tau):
    """0 up to 100 ms, 1 above 100 and up to 300 ms, 2 above 300 ms, 3 no notch before; class edges with the 3 ms
    tolerance of every stamp comparison (Addendum 1 section 2)."""
    cls = np.full(len(tau), 3, np.int64)
    f = np.isfinite(tau)
    t = tau[f]
    cls[f] = np.where(t <= 100.0 + TOL_MS, 0, np.where(t <= 300.0 + TOL_MS, 1, 2))
    return cls


# ------------------------------------------------------------------------------------------ c
def c_of(A):
    """c = |a2(BES)| + |a2(ECEI)| from the six anchor scalars (columns 1 and 4)."""
    return np.abs(A[:, 1]) + np.abs(A[:, 4])


def c_class(c, cuts):
    cls = np.full(len(c), 3, np.int64)
    f = np.isfinite(c)
    cls[f] = np.where(c[f] <= cuts[0], 0, np.where(c[f] <= cuts[1], 1, 2))
    return cls


# ------------------------------------------------------------------------------------------ generic channel scalars
def zmean_cols(fast64, mu, sd, cols):
    """Mean of the standardised channels `cols`, by explicit sequential sums (as Feats.window)."""
    z0 = (fast64[:, cols[0]] - mu[cols[0]]) / sd[cols[0]]
    acc = z0.copy()
    for c in cols[1:]:
        acc += (fast64[:, c] - mu[c]) / sd[c]
    acc /= len(cols)
    return acc


def raw_mean_cols(fast64, cols):
    acc = fast64[:, cols[0]].copy()
    for c in cols[1:]:
        acc += fast64[:, c]
    acc /= len(cols)
    return acc


def window_per_file(x, off, seg):
    out = np.empty((len(x), 4))
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]
        out[a:b] = C.window_feats(x[a:b], seg[a:b])
    return out


def seven_scalars(zm, rawm, off, seg, gi, anc_g, mr_raw=None):
    """The seven scalars of one array mean: the four of the present (window_feats at row t) and a1, a2, a3 (anchor)."""
    w4 = window_per_file(zm, off, seg)
    ms = K75.run_mean10(zm, seg)
    mr = K75.run_mean10(rawm, seg) if mr_raw is None else mr_raw
    A = K75.anchor_scalars(gi, anc_g, ms, ms, mr, mr)[:, :3]
    return np.column_stack([w4[gi], A])


# ------------------------------------------------------------------------------------------ anchors of every grid row
def anchors_all(G, rule):
    """The anchor row (global index; -1 when nothing is visible) of every grid row under `rule`, by the visibility rule
    of B.75 (common75.file_inputs: the last measured value of the block of t with stamp at most the limit + 3 ms)."""
    tm, ti, seg, off = G["time"], G["ti"], G["seg"], G["off"]
    out = np.full(len(tm), -1, np.int64)
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]
        t_, ti_, sg = tm[a:b], ti[a:b], seg[a:b]
        rows = np.arange(b - a)
        lim, _ = K75.limit_stamps(t_, sg, rows, rule)
        obs = np.flatnonzero(np.isfinite(ti_))
        newblk = np.concatenate(([True], sg[1:] != sg[:-1]))
        first = np.maximum.accumulate(np.where(newblk, rows, 0))
        p = np.searchsorted(t_[obs], lim + K75.TOL, side="right")
        bb = np.searchsorted(obs, first, side="left")
        ok = p > bb
        loc = np.full(b - a, -1, np.int64)
        loc[ok] = obs[p[ok] - 1]
        if np.any(loc[ok] >= rows[ok]):
            raise SystemExit("FATAL: a visible value at or after its row")
        out[a:b] = np.where(ok, loc + a, -1)
    return out


# ------------------------------------------------------------------------------------------ segments and frames
def segment_ordinal(G):
    """Ordinal of the segment within its file, counting only the segments of grid rows (a file's first and last rows,
    which are not grid rows, get -1)."""
    seg, fid, grow, off = G["seg"], G["fidx"], G["grow"], G["off"]
    N = len(seg)
    last = np.concatenate((fid[1:] != fid[:-1], [True]))
    real = (grow > 0) & ~last
    out = np.full(N, -1, np.int64)
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]
        s = seg[a:b]
        m = real[a:b]
        u = np.unique(s[m])
        loc = np.full(b - a, -1, np.int64)
        loc[m] = np.searchsorted(u, s[m])
        out[a:b] = loc
    return out


def frames_of_segment0(fr, sord, off, files):
    """Number of frames of the first segment of each file (largest frame index + 1)."""
    out = {}
    for i in files:
        a, b = off[i], off[i + 1]
        m = sord[a:b] == 0
        out[int(i)] = int(fr[a:b][m].max()) + 1 if m.any() else 0
    return out


def donors(u, rule, nfr0):
    """One donor per discharge of each side of unit u (Addendum 1 section 4).  Returns {recipient file: donor file}."""
    rng = rng_for(f"swap|{u['name']}|{rule}")
    out = {}
    for side in ("fit", "test"):
        D = np.sort(np.asarray(u[side]))
        for i in D:
            other = D[D != i]
            elig = other[np.array([nfr0[int(j)] >= nfr0[int(i)] for j in other], bool)]
            pool = elig if len(elig) else other
            out[int(i)] = int(pool[rng.integers(0, len(pool))])
    return out


class DonorIndex:
    """(file, segment ordinal, frame index) -> global grid row, for grid rows with a frame index."""

    def __init__(self, G, fr, sord):
        ok = (fr >= 0) & (sord >= 0)
        g = np.flatnonzero(ok)
        self.key = (G["fidx"][g].astype(np.int64) * 64 + sord[g]) * 4096 + fr[g]
        if np.any(fr[g] >= 4096) or np.any(sord[g] >= 64):
            raise SystemExit("FATAL: frame or ordinal out of range of the key")
        o = np.argsort(self.key, kind="stable")
        self.key, self.g = self.key[o], g[o]
        if np.any(np.diff(self.key) == 0):
            raise SystemExit("FATAL: duplicate (file, ordinal, frame)")
        self.fr, self.sord = fr, sord

    def lookup(self, donor_file, gi):
        k = (donor_file.astype(np.int64) * 64 + self.sord[gi]) * 4096 + self.fr[gi]
        p = np.searchsorted(self.key, k)
        p = np.minimum(p, len(self.key) - 1)
        hit = self.key[p] == k
        return np.where(hit, self.g[p], -1)


# ------------------------------------------------------------------------------------------ context of the fits
class Ctx(B81.Ctx):
    """B.81's context (rows, B.75 sets, fast-channel statistics) plus the grid, Mirnov columns, frames, anchors."""

    def __init__(self):
        super().__init__()
        G = K75.grid_lean(("off", "time", "ti", "seg", "fidx", "grow", "fast"))
        self.G = G
        self.off = G["off"]
        self.fast64 = G["fast"].astype(np.float64)
        mc = load_npz(CACHE / "mc.npz")
        self.mc32 = mc["mc"]
        self.mc64 = self.mc32.astype(np.float64)
        import stage_trees as ST
        self.FEmc = ST.Feats(fast=self.mc32)
        self.mcraw_run = K75.run_mean10(raw_mean_cols(self.mc64, (0, 1)), self.seg)
        self.bcraw_run = {L: K75.run_mean10(raw_mean_cols(self.fast64, cols), self.seg) for L, cols in BES_COLUMNS.items()}
        fz = load_npz(M78.CACHE / "frames.npz")
        self.fr = fz["fr"]
        self.sord = segment_ordinal(G)
        self.didx = DonorIndex(G, self.fr, self.sord)
        self.nfr0 = frames_of_segment0(self.fr, self.sord, self.off, np.unique(self.R["fidx"]))
        self._anc_all = {}

    def anc_all(self, rule):
        if rule not in self._anc_all:
            self._anc_all[rule] = anchors_all(self.G, rule)
        return self._anc_all[rule]

    def mirnov7(self, u, anc_g):
        mu, sd = self.FEmc.stats(u["fit"])
        zm = zmean_cols(self.mc64, mu, sd, (0, 1))
        return seven_scalars(zm, None, self.off, self.seg, self.R["gi"], anc_g, mr_raw=self.mcraw_run)

    def bescol7(self, u, L, anc_g):
        mu, sd = self.FE.stats(u["fit"])
        zm = zmean_cols(self.fast64, mu, sd, BES_COLUMNS[L])
        return seven_scalars(zm, None, self.off, self.seg, self.R["gi"], anc_g, mr_raw=self.bcraw_run[L])

    def swap14(self, u, rule, wf, ms_b, ms_e, rows):
        """The fourteen fast and anchor columns of the donor rows of `rows` (NaN where there is no donor row), the donor
        grid rows and the donor files."""
        R = self.R
        dmap = donors(u, rule, self.nfr0)
        fid = R["fidx"][rows]
        dfile = np.array([dmap[int(f)] for f in fid], np.int64)
        dg = self.didx.lookup(dfile, R["gi"][rows])
        out = np.full((len(rows), 14), np.nan)
        ok = dg >= 0
        g = dg[ok]
        out[ok, :8] = wf[g]
        anc = self.anc_all(rule)[g]
        out[ok, 8:] = K75.anchor_scalars(g, anc, ms_b, ms_e, self.mr_b, self.mr_e)
        return out, dg, dfile


def matrix(ctx, L, u, rule, H10, f8, A, I, wf, ms_b, ms_e, rows, cache):
    """Learner matrix of the new learners (Addendum 1 section 3); T0 and TX by fits78.inputs_of."""
    if L in ("T0", "TX"):
        return F78.inputs_of(L, H10, f8, A, rows)
    if L in ("TM", "TXM"):
        if "mc" not in cache:
            cache["mc"] = ctx.mirnov7(u, I["anc_g"])
        m7 = cache["mc"][rows]
        base = F78.inputs_of("T0" if L == "TM" else "TX", H10, f8, A, rows)
        return np.column_stack([base, m7])
    if L == "TXs":
        base = F78.inputs_of("TX", H10, f8, A, rows)
        s14, _, _ = ctx.swap14(u, rule, wf, ms_b, ms_e, rows)
        base[:, 10:24] = s14
        return base
    if L in BES_COLUMNS:
        key = "bc" + L
        if key not in cache:
            cache[key] = ctx.bescol7(u, L, I["anc_g"])
        return np.column_stack([H10[rows], cache[key][rows]])
    raise KeyError(L)


def fit_units(us):
    """Units of part C: the 15 of V1, the 4 of V3 and V4a."""
    return [u for u in us if u["design"] in ("V1", "V3") or u["name"] == "V4a"]


def seed_units(us):
    return [u for u in us if u["name"] in SEED_UNITS]


def path82(name, L, r):
    return FITS / f"{name}__{L}__{r}__{X}.npz"
