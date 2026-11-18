"""B.66 (PREREGISTRATION_B66.md): the increment by regime -- discharges and phases with and without edge bursts.

CPU only; stored errors only (no network is run, nothing is trained).  The burst detector is the 10 ms BES
departure rule that B.54 uses for its quiet rows (`run_b45.Classes.of_file`, which calls
`posthoc_b32.channel_departures`), run again on every file; rows after an absent frame (class 1) and low beam
emission (class 2) are excluded as in B.54.  Rows and pooled errors: the B.54 row tables of the B.39 pooled runs
(data/.b54/rows_r{r}_f{k}.npz); KFX and KF0: data/.b56/r{r}_f{k}.npz (r3..r9).

Stages (repo root, in this order; cached under data/.b66/):
    py ces_prediction/experiments/b66_regimes/run_b66.py detect    # detector on every file, onsets, rates
    py ces_prediction/experiments/b66_regimes/run_b66.py counts    # checks + class counts (no error is read)
    py ces_prediction/experiments/b66_regimes/run_b66.py readout   # R1..R4 and the reported quantities
Output: docs/paper/b66_regimes.{json,md}
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the GPU belongs to the running training batch
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "b54_pooled_rows"))
import run_b54 as B54  # noqa: E402  (adds b30/b37/b41/b44/b45/b50/b39/b32 to sys.path)
from run_b54 import B27, B39, B44, B45  # noqa: E402
from posthoc_b32 import channel_departures  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_B66.md"
DATA = B54.DATA
CACHE = DATA / ".b66"
B54C = DATA / ".b54"
B56C = DATA / ".b56"
PAPER = B54.PAPER
OUT_JSON = PAPER / "b66_regimes.json"
OUT_MD = PAPER / "b66_regimes.md"
REPEATS = tuple(B39.REPEATS)                 # 2..9
KFX_REPEATS = tuple(range(3, 10))            # registration: KFX and KF0 on r3 to r9
K = B27.K
NBOOT = 20_000
SEED = 20261166
ROW_S = 0.010                                # seconds per grid row
BURST_S = 0.030                              # burst phase: within 30 ms after an onset
INTER_S = 0.050                              # inter-burst phase: at least 50 ms from any onset
TOL_T = 1e-6
MIN_CELL = 30                                # two-way table: increment when a cell has >= 30 discharges
QUIET_RECORDED = 0.889                       # B.54 check (e): quiet share on the stored rows of split 42
DET_CLASSES = (3, 4, 5, 6)                   # rows on which a burst can be detected (no blip, departure defined)
CLASSES = ("A", "B", "C")
CLAB = {"A": "A no burst", "B": "B rate up to the median of the non-zero rates",
        "C": "C rate above that median"}
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def q95(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


# ------------------------------------------------------------------------------------------ stage detect
def segments(t):
    """Segment id per grid row: a new segment at a step of 0.5 s or more (run_b45.SEG_BREAK_S)."""
    return np.concatenate(([0], np.cumsum(np.diff(t) >= B45.SEG_BREAK_S)))


def stage_detect():
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "detect.npz"
    if out.exists():
        log("detect: cached")
        return
    C = B54.ctx41()
    Cl = B45.Classes(C)
    F54 = B54.load_files()
    arrs, same_as_b54 = {}, []
    for i, nm in enumerate(C.names):
        cls, _ = Cl.of_file(nm)                               # the detector, run again on this file
        same_as_b54.append(bool(np.array_equal(cls, F54[f"cls_{i}"])))
        _tc, d, _after = channel_departures(DATA / nm)        # index = grid row - 1 (as in Classes.of_file)
        defined = np.isfinite(d).all(1)
        up_in = defined & (np.nan_to_num(d, nan=0.0) > B45.DEP_SD).any(1)
        n = len(cls)
        up = np.zeros(n, bool)
        up[1:-1] = up_in
        t = C.t[i]
        seg = segments(t)
        prev_up = np.zeros(n, bool)
        prev_up[1:] = up[:-1] & (seg[1:] == seg[:-1])
        onset = (cls == 3) & ~prev_up                         # first row of an upward excursion, not a blip row
        det_rows = np.isin(cls, DET_CLASSES)
        arrs[f"onset_{i}"] = np.flatnonzero(onset)
        arrs[f"t_{i}"] = t
        arrs[f"det_rows_{i}"] = np.array(int(det_rows.sum()))
        arrs[f"cls_{i}"] = cls
        arrs[f"cont_{i}"] = np.array(int(((cls == 3) & prev_up).sum()))
        if (i + 1) % 100 == 0:
            log(f"detector {i + 1}/{len(C.names)}")
    arrs["names"] = np.array(C.names)
    arrs["same_as_b54"] = np.array(same_as_b54)
    np.savez_compressed(out, **arrs)
    log(f"detect: written; classes identical to the B.54 cache in {sum(same_as_b54)}/{len(same_as_b54)} files")


def load_detect():
    z = np.load(CACHE / "detect.npz")
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------ universe and classes
def universe(names):
    """The 639 discharges of the pooled re-partitions: the test files of the five folds of r2 (file indices)."""
    pos = {nm: j for j, nm in enumerate(names)}
    tests = [json.loads((B27.split_dir(2, k) / "split_manifest.json").read_text(encoding="utf-8"))["test_files"]
             for k in range(K)]
    return np.array(sorted(pos[nm] for m in tests for nm in m), np.int64)


def regime_classes(Dt):
    names = list(Dt["names"])
    U = universe(names)
    nfile = len(names)
    onsets = np.array([len(Dt[f"onset_{i}"]) for i in range(nfile)], float)
    secs = np.array([float(Dt[f"det_rows_{i}"]) * ROW_S for i in range(nfile)])
    rate = np.full(nfile, np.nan)
    ok = secs > 0
    rate[ok] = onsets[ok] / secs[ok]
    ru = rate[U]
    nz = ru[np.isfinite(ru) & (ru > 0)]
    med_nz = float(np.median(nz))
    lab = np.full(nfile, "", dtype="<U1")
    for j in U:
        if not np.isfinite(rate[j]):
            continue
        lab[j] = "A" if rate[j] == 0 else ("B" if rate[j] <= med_nz else "C")
    return {"U": U, "rate": rate, "onsets": onsets, "secs": secs, "median_nonzero_rate": med_nz, "lab": lab}


def phase_masks(Dt, fidx, grow, cls_row):
    """Burst phase: 0 <= t - onset <= 30 ms for some onset; inter-burst: |t - onset| >= 50 ms for every onset.
    Only rows of classes 3 to 6 (blip rows and rows without a defined departure excluded)."""
    burst = np.zeros(len(fidx), bool)
    inter = np.zeros(len(fidx), bool)
    for f in np.unique(fidx):
        sel = np.flatnonzero(fidx == f)
        t = Dt[f"t_{f}"]
        ton = t[Dt[f"onset_{f}"]]
        tr = t[grow[sel]]
        if len(ton) == 0:
            inter[sel] = True
            continue
        j = np.searchsorted(ton, tr + TOL_T, side="right") - 1          # last onset at or before the row
        since = np.where(j >= 0, tr - ton[np.clip(j, 0, None)], np.inf)
        burst[sel] = since <= BURST_S + TOL_T
        jn = np.clip(np.searchsorted(ton, tr, side="left"), 0, len(ton) - 1)
        jp = np.clip(jn - 1, 0, len(ton) - 1)
        dist = np.minimum(np.abs(tr - ton[jn]), np.abs(tr - ton[jp]))
        inter[sel] = dist >= INTER_S - TOL_T
    det = np.isin(cls_row, DET_CLASSES)
    return burst & det, inter & det


# ------------------------------------------------------------------------------------------ stage counts (no errors)
def stage_counts():
    Dt = load_detect()
    names = list(Dt["names"])
    G = regime_classes(Dt)
    F = B54.load_files()
    chk = {}
    # (a) the detector run again equals the classes cached by B.54 in every file
    chk["a_detector_equals_b54_cache"] = {"files": int(len(Dt["same_as_b54"])),
                                          "identical": int(Dt["same_as_b54"].sum()),
                                          "passes": bool(Dt["same_as_b54"].all())}
    # (b) quiet-row share of B.54 check (e) on the stored rows of split 42 (only the row indices are used)
    C = B54.ctx41()
    valid = B54.valid_names(C)
    d = B54.load_run(DATA / ".b39_v2_cut_s42_i42")
    d = {"shot": d["shot"], "row": d["row"]}
    pos = {nm: j for j, nm in enumerate(C.names)}
    cls42 = []
    for u in np.unique(d["shot"]):
        sel = np.flatnonzero(d["shot"] == u)
        nm = valid[int(u)]
        gi = B45.compact_index(C.grid[nm])
        cls42.append(Dt[f"cls_{pos[nm]}"][gi[d["row"][sel]]])
    cls42 = np.concatenate(cls42)
    qs = float((cls42 == 6).mean())
    chk["b_quiet_share_split42"] = {"mine": qs, "recorded": QUIET_RECORDED,
                                    "recorded_full": json.loads((B54C / "checks.json").read_text())["e"]["registered"]["quiet_share"]["recorded"],
                                    "passes": bool(round(qs, 3) == QUIET_RECORDED)}
    # (c) B.54 checks pass; per fold the stored classes of B.54 equal the detector's; B.56 rows aligned
    b54chk = json.loads((B54C / "checks.json").read_text(encoding="utf-8"))
    cc = {"b54_checks_pass": bool(b54chk["passes"]), "folds": {}}
    allok = bool(b54chk["passes"])
    for r in REPEATS:
        for k in range(K):
            z = np.load(B54C / f"rows_r{r}_f{k}.npz")
            fidx, grow = z["fidx"], z["grow"]
            mine = np.array([Dt[f"cls_{a}"][b] for a, b in zip(fidx, grow)], np.int8)
            rec = {"rows": int(len(fidx)), "classes_equal_b54": bool(np.array_equal(mine, z["cls"]))}
            if r in KFX_REPEATS:
                q = np.load(B56C / f"r{r}_f{k}.npz")
                rec["b56_rows_aligned"] = bool(np.array_equal(q["shot"], z["shot"]) and np.array_equal(q["row"], z["row"])
                                               and np.allclose(q["y"], z["y"], atol=0.01))
                allok &= rec["b56_rows_aligned"]
            allok &= rec["classes_equal_b54"]
            cc["folds"][f"r{r}f{k}"] = rec
    cc["passes"] = bool(allok)
    chk["c_rows"] = cc
    # (d) every discharge of the universe has rows on which a burst can be detected; 639 discharges
    nd = int(np.sum(~np.isfinite(G["rate"][G["U"]])))
    # Addendum 1: the discharges without detectable rows form the group "rate not defined" (no class, no statement)
    chk["d_universe"] = {"discharges": int(len(G["U"])), "without_detectable_rows": nd,
                         "rule": "addendum 1: group 'rate not defined', in no class and no statement",
                         "passes": bool(len(G["U"]) == 639 and nd == 28)}
    chk["passes"] = bool(all(v["passes"] for v in chk.values() if isinstance(v, dict)))

    # class counts: discharges, scored rows (per repeat), median T_i, rate thresholds
    lab = G["lab"]
    counts = {"median_nonzero_rate_per_s": G["median_nonzero_rate"], "classes": {}}
    rows_by = {c: {} for c in CLASSES}
    ymed_by = {c: [] for c in CLASSES}
    phase_rows = {"burst_BC": {}, "inter_BC": {}, "burst_C": {}, "inter_C": {}, "neither_BC": {}}
    for r in REPEATS:
        fid, yv, ph = [], [], {"b": [], "i": [], "c": []}
        for k in range(K):
            z = np.load(B54C / f"rows_r{r}_f{k}.npz")
            fid.append(z["fidx"])
            yv.append(z["y"])
            b, i_ = phase_masks(Dt, z["fidx"], z["grow"], z["cls"])
            ph["b"].append(b)
            ph["i"].append(i_)
        fid, yv = np.concatenate(fid), np.concatenate(yv)
        pb, pi = np.concatenate(ph["b"]), np.concatenate(ph["i"])
        L = lab[fid]
        for c in CLASSES:
            rows_by[c][str(r)] = int((L == c).sum())
            if r == 2:
                ymed_by[c] = float(np.median(yv[L == c]))
        bc = np.isin(L, ("B", "C"))
        phase_rows["burst_BC"][str(r)] = int((pb & bc).sum())
        phase_rows["inter_BC"][str(r)] = int((pi & bc).sum())
        phase_rows["burst_C"][str(r)] = int((pb & (L == "C")).sum())
        phase_rows["inter_C"][str(r)] = int((pi & (L == "C")).sum())
        phase_rows["neither_BC"][str(r)] = int((bc & ~pb & ~pi).sum())
        phase_rows.setdefault("rows_BC", {})[str(r)] = int(bc.sum())
    U = G["U"]
    und = U[~np.isfinite(G["rate"][U])]
    und_rows = {}
    for r in REPEATS:
        und_rows[str(r)] = int(sum(np.isin(np.load(B54C / f"rows_r{r}_f{k}.npz")["fidx"], und).sum() for k in range(K)))
    counts["rate_not_defined"] = {"discharges": int(len(und)), "files": [names[j] for j in und],
                                  "scored_rows_per_repeat": und_rows,
                                  "grid_rows_range": [int(min(len(Dt[f"t_{j}"]) for j in und)), int(max(len(Dt[f"t_{j}"]) for j in und))]}
    for c in CLASSES:
        m = U[lab[U] == c]
        rt = G["rate"][m]
        counts["classes"][c] = {
            "discharges": int(len(m)), "share_of_639": float(len(m) / len(U)),
            "scored_rows_per_repeat": rows_by[c], "scored_rows_r2": rows_by[c]["2"],
            "median_of_discharge_median_ti_eV": float(np.nanmedian(F["median_ti"][m])),
            "median_scored_ti_r2_eV": ymed_by[c],
            "rate_min_per_s": float(rt.min()), "rate_median_per_s": float(np.median(rt)), "rate_max_per_s": float(rt.max()),
            "onsets_total": int(G["onsets"][m].sum()), "detectable_seconds_total": float(G["secs"][m].sum())}
    counts["phase_rows"] = phase_rows
    # cross-count with the temperature thirds of all 639 discharges (no error read)
    v = F["median_ti"][U]
    have = np.flatnonzero(np.isfinite(v))
    parts = [have[p] for p in B44.split_thirds(v[have], F["shot"][U][have])]
    xt = {c: {tn: int(np.sum(lab[U[p]] == c)) for tn, p in zip(B44.THIRDS, parts)} for c in CLASSES}
    counts["class_by_third_discharges_639"] = xt
    counts["continuation_rows_class3_total"] = int(sum(int(Dt[f"cont_{i}"]) for i in U))
    res = {"registration": REG, "checks": chk, "counts": counts, "rate_per_discharge": {
        names[j]: {"rate_per_s": float(G["rate"][j]), "class": str(lab[j])} for j in U}}
    (CACHE / "counts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(CACHE / "classes.npz", lab=lab, rate=G["rate"], U=U)
    OUT_MD.write_text(render_counts(res) + "\n_Readout not yet computed._\n", encoding="utf-8", newline="\n")
    log("checks: " + ", ".join(f"{k} {v['passes']}" for k, v in chk.items() if isinstance(v, dict)))
    log(f"quiet share split 42: {qs:.4f} (recorded {QUIET_RECORDED})")
    log(f"median of non-zero rates: {G['median_nonzero_rate']:.3f} /s")
    for c in CLASSES:
        cc_ = counts["classes"][c]
        log(f"class {c}: {cc_['discharges']} discharges, r2 rows {cc_['scored_rows_r2']}, median T_i "
            f"{cc_['median_of_discharge_median_ti_eV']:.0f} eV, rate {cc_['rate_min_per_s']:.3f}..{cc_['rate_max_per_s']:.3f} /s")
    log(f"class x third: {xt}")
    log(f"phase rows r2: " + ", ".join(f"{k} {v['2']}" for k, v in phase_rows.items()))
    if not chk["passes"]:
        raise SystemExit("FATAL: a check fails; no statement is read (data/.b66/counts.json)")
    log("all checks pass; counts written to " + str(OUT_MD))


# ------------------------------------------------------------------------------------------ readout
def hier_multi(pairs, reps, nboot, rng):
    """analyze_b39.arm_p.hier generalised: a repeat drawn uniformly, then its discharges resampled with
    replacement; the same draws for every (num, den) pair so that differences can be formed.
    pairs: list of dict repeat -> (num per discharge, den per discharge).  Returns points (mean over repeats of
    the per-repeat skill), per-repeat values, and the draw arrays."""
    draws = rng.choice(np.array(reps), size=nboot)
    o = [np.empty(nboot) for _ in pairs]
    for r in reps:
        sel = np.flatnonzero(draws == r)
        n = len(pairs[0][r][0])
        for c0 in range(0, len(sel), 2500):
            ss = sel[c0:c0 + 2500]
            idx = rng.integers(0, n, size=(len(ss), n))
            for j, p in enumerate(pairs):
                num, den = p[r]
                with np.errstate(invalid="ignore", divide="ignore"):
                    o[j][ss] = 1.0 - num[idx].sum(1) / den[idx].sum(1)
    per = [{str(r): float(1.0 - p[r][0].sum() / p[r][1].sum()) for r in reps} for p in pairs]
    pts = [float(np.mean(list(pp.values()))) for pp in per]
    return pts, per, o


def stage_readout():
    cj = json.loads((CACHE / "counts.json").read_text(encoding="utf-8"))
    if not cj["checks"]["passes"]:
        raise SystemExit("FATAL: checks do not pass; no readout")
    Dt = load_detect()
    Z = np.load(CACHE / "classes.npz")
    lab = Z["lab"]
    F = B54.load_files()
    D = {}
    for r in REPEATS:
        parts = [np.load(B54C / f"rows_r{r}_f{k}.npz") for k in range(K)]
        d = {kk: np.concatenate([p[kk] for p in parts]) for kk in ("fidx", "grow", "cls", "y", "se_c", "se_t", "se_gp", "se_hold")}
        b, i_ = phase_masks(Dt, d["fidx"], d["grow"], d["cls"])
        d["burst"], d["inter"] = b, i_
        if r in KFX_REPEATS:
            se_kfx, se_kf0 = [], []
            for k in range(K):
                q = np.load(B56C / f"r{r}_f{k}.npz")
                se_kfx.append((q["est_kfx"] - q["y"]) ** 2)
                se_kf0.append((q["est_kf0"] - q["y"]) ** 2)
            d["se_kfx"], d["se_kf0"] = np.concatenate(se_kfx), np.concatenate(se_kf0)
        u, f = np.unique(d["fidx"], return_inverse=True)
        d["u"], d["f"], d["n"] = u, f, len(u)
        d["L"] = lab[d["fidx"]]
        D[r] = d
    log("repeats loaded: " + ", ".join(f"r{r} {len(D[r]['y']):,} rows / {D[r]['n']} discharges" for r in REPEATS))

    def ds(r, x, m):
        return np.bincount(D[r]["f"][m], x[m], D[r]["n"])

    def pair(a, b, mask_fn, reps):
        return {r: (ds(r, D[r][a], mask_fn(r)), ds(r, D[r][b], mask_fn(r))) for r in reps}

    REFS = (("inc", "se_c", "se_t", REPEATS), ("gp", "se_c", "se_gp", REPEATS),
            ("hold", "se_c", "se_hold", REPEATS), ("kfx", "se_kfx", "se_kf0", KFX_REPEATS))

    def cell(name, mask_fn):
        o = {}
        for key, a, b, reps in REFS:
            pts, per, dr = hier_multi([pair(a, b, mask_fn, reps)], reps, NBOOT, rng_for(f"{name}|{key}"))
            o[key] = [pts[0], *q95(dr[0])]
            o[key + "_per_repeat"] = per[0]
        o["rows_per_repeat"] = {str(r): int(mask_fn(r).sum()) for r in REPEATS}
        o["discharges_with_rows_mean"] = float(np.mean([len(np.unique(D[r]["f"][mask_fn(r)])) for r in REPEATS]))
        o["median_scored_ti_eV_mean"] = float(np.mean([np.median(D[r]["y"][mask_fn(r)]) for r in REPEATS]))
        return o

    rep = {}
    rep["all"] = cell("all", lambda r: np.ones(len(D[r]["y"]), bool))
    log(f"all rows: inc {rep['all']['inc']} gp {rep['all']['gp']} kfx {rep['all']['kfx']}")
    rep["class"] = {c: cell(f"class{c}", lambda r, c=c: D[r]["L"] == c) for c in CLASSES}
    for c in CLASSES:
        log(f"class {c}: inc {rep['class'][c]['inc']} gp {rep['class'][c]['gp']} kfx {rep['class'][c]['kfx']}")
    BC = lambda r: np.isin(D[r]["L"], ("B", "C"))                                       # noqa: E731
    rep["class_BC"] = cell("classBC", BC)
    rep["phase"] = {
        "burst_BC": cell("burst_BC", lambda r: BC(r) & D[r]["burst"]),
        "inter_BC": cell("inter_BC", lambda r: BC(r) & D[r]["inter"]),
        "burst_C": cell("burst_C", lambda r: (D[r]["L"] == "C") & D[r]["burst"]),
        "inter_C": cell("inter_C", lambda r: (D[r]["L"] == "C") & D[r]["inter"]),
        "burst_B": cell("burst_B", lambda r: (D[r]["L"] == "B") & D[r]["burst"]),
        "inter_B": cell("inter_B", lambda r: (D[r]["L"] == "B") & D[r]["inter"]),
    }
    for k_, v in rep["phase"].items():
        log(f"phase {k_}: inc {v['inc']} gp {v['gp']}")

    # R4: increment in A minus increment in C, the same draws
    pA = pair("se_c", "se_t", lambda r: D[r]["L"] == "A", REPEATS)
    pC = pair("se_c", "se_t", lambda r: D[r]["L"] == "C", REPEATS)
    pts, per, dr = hier_multi([pA, pC], REPEATS, NBOOT, rng_for("R4|A-C"))
    diff = dr[0] - dr[1]
    per_d = {k_: per[0][k_] - per[1][k_] for k_ in per[0]}
    rep["A_minus_C_inc"] = {"value": [float(np.mean(list(per_d.values()))), *q95(diff)], "per_repeat": per_d}
    # the same difference over the causal GP (reported)
    pA = pair("se_c", "se_gp", lambda r: D[r]["L"] == "A", REPEATS)
    pC = pair("se_c", "se_gp", lambda r: D[r]["L"] == "C", REPEATS)
    pts, per, dr = hier_multi([pA, pC], REPEATS, NBOOT, rng_for("A-C|gp"))
    per_d = {k_: per[0][k_] - per[1][k_] for k_ in per[0]}
    rep["A_minus_C_gp"] = {"value": [float(np.mean(list(per_d.values()))), *q95(dr[0] - dr[1])], "per_repeat": per_d}
    log(f"A - C inc {rep['A_minus_C_inc']['value']}, gp {rep['A_minus_C_gp']['value']}")

    # two-way table: class by temperature third (thirds within each repeat as B.54)
    thirds = {}
    for r in REPEATS:
        u = D[r]["u"]
        v = F["median_ti"][u]
        have = np.flatnonzero(np.isfinite(v))
        parts = [have[p] for p in B44.split_thirds(v[have], F["shot"][u][have])]
        lt = np.full(D[r]["n"], -1)
        for j, idx in enumerate(parts):
            lt[idx] = j
        thirds[r] = lt
    tw = {}
    for c in CLASSES:
        for j, tn in enumerate(B44.THIRDS):
            m_fn = lambda r, c=c, j=j: (D[r]["L"] == c) & (thirds[r][D[r]["f"]] == j)       # noqa: E731
            nd = {str(r): int(len(np.unique(D[r]["f"][m_fn(r)]))) for r in REPEATS}
            e = {"discharges_per_repeat": nd, "discharges_min": min(nd.values()),
                 "rows_mean": float(np.mean([m_fn(r).sum() for r in REPEATS]))}
            if e["discharges_min"] >= MIN_CELL:
                for key, a, b, reps in REFS[:2]:
                    pts, per, dr = hier_multi([pair(a, b, m_fn, reps)], reps, NBOOT, rng_for(f"tw|{c}|{tn}|{key}"))
                    e[key] = [pts[0], *q95(dr[0])]
                    e[key + "_per_repeat"] = per[0]
            tw[f"{c}|{tn}"] = e
    # thirds alone (for reference beside the table)
    for j, tn in enumerate(B44.THIRDS):
        m_fn = lambda r, j=j: thirds[r][D[r]["f"]] == j                                  # noqa: E731
        pts, per, dr = hier_multi([pair("se_c", "se_t", m_fn, REPEATS)], REPEATS, NBOOT, rng_for(f"third|{tn}|inc"))
        tw[f"all|{tn}"] = {"inc": [pts[0], *q95(dr[0])], "inc_per_repeat": per[0],
                           "discharges_min": int(min(len(np.unique(D[r]["f"][m_fn(r)])) for r in REPEATS))}
    rep["two_way"] = tw
    log("two-way: " + "; ".join(f"{k_} {v.get('inc', ['n<30'])[0] if 'inc' in v else 'n<30'}" for k_, v in tw.items()))

    st = {
        "R1": {"text": "class A (no bursts): increment over the twin above zero", "value": rep["class"]["A"]["inc"]},
        "R2": {"text": "class C (frequent bursts): increment over the twin above zero", "value": rep["class"]["C"]["inc"]},
        "R3": {"text": "inter-burst phase of classes B and C: increment over the twin above zero",
               "value": rep["phase"]["inter_BC"]["inc"]},
        "R4": {"text": "increment in class A minus class C: interval excludes zero", "value": rep["A_minus_C_inc"]["value"]},
    }
    for k_ in ("R1", "R2", "R3"):
        st[k_]["holds"] = bool(st[k_]["value"][1] > 0)
    st["R4"]["holds"] = bool(st["R4"]["value"][1] > 0 or st["R4"]["value"][2] < 0)
    exp = expectations(st, cj)
    out = {**cj, "seed": SEED, "n_boot": NBOOT, "statements": st, "reported": rep, "expectations": exp,
           "choices": CHOICES, "runtime_readout_s": round(_time.time() - T0, 1)}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render_counts(cj) + render_readout(out), encoding="utf-8", newline="\n")
    for k_, v in st.items():
        log(f"{k_}: {s3(v['value'])} {'holds' if v['holds'] else 'does not hold'}")
    log("written " + str(OUT_MD))


EXPECT = [("R1", "+0.00 to +0.04, uncertain", 0.00, 0.04, None),
          ("R2", "+0.05 to +0.09, met", 0.05, 0.09, True),
          ("R3", "met", None, None, True),
          ("R4", "uncertain", None, None, None)]


def expectations(st, cj):
    E = []
    A = cj["counts"]["classes"]["A"]
    med = {c: cj["counts"]["classes"][c]["median_of_discharge_median_ti_eV"] for c in CLASSES}
    E.append({"key": "class A", "expectation": "about a third of the discharges and colder",
              "observed": f"{A['share_of_639']:.3f} of 639; median T_i A {med['A']:.0f}, B {med['B']:.0f}, C {med['C']:.0f} eV",
              "range_met": bool(0.25 <= A["share_of_639"] <= 0.42 and med["A"] < min(med["B"], med["C"]))})
    for key, text, lo, hi, verdict in EXPECT:
        v = st[key]["value"]
        E.append({"key": key, "expectation": text, "observed": f"{s3(v)}; {'holds' if st[key]['holds'] else 'does not hold'}",
                  "range_met": None if lo is None else bool(lo <= v[0] <= hi),
                  "verdict_met": None if verdict is None else bool(st[key]["holds"] == verdict)})
    return E


CHOICES = [
    "Detector: `run_b45.Classes.of_file` (the 10 ms BES departure rule that B.54 uses for its quiet rows; "
    "`posthoc_b32.channel_departures`: a BES channel more than 3 standard deviations from its mean over the ten "
    "rows before), run again on all 641 files and compared with B.54's cached classes. A burst row is an upward "
    "departure (class 3); rows after an absent frame (class 1) and low beam emission (class 2) take priority over "
    "class 3 in that code, which is the exclusion of beam blips of B.54. Downward departures (class 4) are not "
    "bursts (B.32: upward departures go with the raw-stream bursts, downward ones with dips of the beam).",
    "Burst onset: a class-3 grid row whose previous grid row in the same segment (steps < 0.5 s) is not an upward "
    "departure (an upward row in class 1 or 2 counts as upward, so an excursion that starts inside a blip window "
    "has no onset). The registration's 'the detector of B.31' is read as this 10 ms rule, since the raw-stream "
    "detector of B.31 exists for twelve discharges only and the registration says the detector is the one used by "
    "B.54 and B.58 for quiet rows.",
    "Burst rate: onsets per second of rows on which a burst can be detected (classes 3 to 6, 10 ms per row): "
    "rows without a defined departure (group 7) and blip rows (classes 1, 2) are not counted in the seconds.",
    "Universe: the 639 discharges of the fold manifests of the pooled re-partitions. Classes A (rate 0), B "
    "(0 < rate <= median of the non-zero rates of the 639), C (above), fixed before any error is read.",
    "Phases (classes B and C only, rows of classes 3 to 6): burst phase = 0 <= t - onset <= 30 ms for some onset "
    "(the onset row included); inter-burst = at least 50 ms from every onset, before or after. Rows 30 to 50 ms "
    "after, or less than 50 ms before, an onset are in neither phase.",
    "Class readouts: every scored row of the discharges of the class (no row excluded). Increment = 1 - SSE "
    "network / SSE twin; skill over the causal GP and over hold likewise; KFX over KF0 on r3 to r9 from data/.b56.",
    "Intervals: the two-level bootstrap of analyze_b39.arm_p (a re-partition drawn uniformly, then its discharges "
    "resampled with replacement; rows outside the cell contribute zero), 20,000 draws, one generator per quantity "
    "default_rng([20261166, crc32(name)]); point = mean over re-partitions. R4: the difference of the two skills "
    "under the same draws.",
    "Two-way table: thirds by the median measured T_i of a discharge within each re-partition (run_b44.split_thirds, "
    "as B.54); a cell's increment is computed when the cell has at least 30 discharges in every re-partition.",
    "Addendum 1: the 28 discharges whose files are too short for a defined departure (no row with ten rows of "
    "segment before) form the group 'rate not defined', in no class and no statement; thresholds from the other 611.",
    "Checks: (a) detector run again equals the B.54 cache in all files; (b) quiet-row share on the stored rows of "
    "split 42 equals 0.889 to three decimals; (c) B.54's checks pass, the classes of every pooled fold equal B.54's, "
    "B.56 rows of r3 to r9 identical to B.54's; (d) 639 discharges, each with detectable rows.",
]


# ------------------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def render_counts(res):
    L = ["# B.66 -- the increment by regime (edge-burst rate)", "",
         f"Registration: `{REG}`. CPU only, stored errors. Script: "
         "`ces_prediction/experiments/b66_regimes/run_b66.py`.", "",
         "## Checks (before any error was read)", "",
         "| check | result | passes |", "|---|---|---|"]
    c = res["checks"]
    L.append(f"| (a) detector run again = B.54 cached classes | {c['a_detector_equals_b54_cache']['identical']}/"
             f"{c['a_detector_equals_b54_cache']['files']} files identical | {c['a_detector_equals_b54_cache']['passes']} |")
    L.append(f"| (b) quiet-row share, stored rows of split 42 | {c['b_quiet_share_split42']['mine']:.4f} "
             f"(recorded {c['b_quiet_share_split42']['recorded_full']:.4f}) | {c['b_quiet_share_split42']['passes']} |")
    L.append(f"| (c) B.54 checks pass; fold classes = B.54; B.56 rows aligned (r3-r9) | "
             f"{len(c['c_rows']['folds'])} folds | {c['c_rows']['passes']} |")
    L.append(f"| (d) universe | {c['d_universe']['discharges']} discharges, "
             f"{c['d_universe']['without_detectable_rows']} without detectable rows | {c['d_universe']['passes']} |")
    k = res["counts"]
    L += ["", "## Class counts (fixed before any error was read)", "",
          f"Median of the non-zero burst rates over the 639 discharges: {k['median_nonzero_rate_per_s']:.3f} onsets per "
          "second of detectable rows.", "",
          "| class | discharges (share) | scored rows r2 | median T_i of discharges (eV) | median scored T_i r2 (eV) | rate range (/s) | rate median (/s) |",
          "|---|---|---|---|---|---|---|"]
    for cl in CLASSES:
        v = k["classes"][cl]
        L.append(f"| {CLAB[cl]} | {v['discharges']} ({v['share_of_639']:.3f}) | {v['scored_rows_r2']:,} | "
                 f"{v['median_of_discharge_median_ti_eV']:.0f} | {v['median_scored_ti_r2_eV']:.0f} | "
                 f"{v['rate_min_per_s']:.3f} to {v['rate_max_per_s']:.3f} | {v['rate_median_per_s']:.3f} |")
    u_ = k["rate_not_defined"]
    L.append(f"| rate not defined (addendum 1; no class, no statement) | {u_['discharges']} | "
             f"{u_['scored_rows_per_repeat']['2']:,} | | | {u_['grid_rows_range'][0]} to {u_['grid_rows_range'][1]} grid rows | |")
    L += ["", "Discharges by class and temperature third (thirds of the 639 by median T_i):", "",
          "| class | " + " | ".join(B44.THIRDS) + " |", "|---|" + "---|" * len(B44.THIRDS)]
    for cl in CLASSES:
        L.append(f"| {cl} | " + " | ".join(str(k["class_by_third_discharges_639"][cl][t]) for t in B44.THIRDS) + " |")
    pr = k["phase_rows"]
    L += ["", "Scored rows by phase, re-partition r2 (classes B and C): "
          f"burst {pr['burst_BC']['2']:,}, inter-burst {pr['inter_BC']['2']:,}, neither {pr['neither_BC']['2']:,} "
          f"of {pr['rows_BC']['2']:,}; class C alone: burst {pr['burst_C']['2']:,}, inter-burst {pr['inter_C']['2']:,}.", ""]
    return "\n".join(L) + "\n"


def render_readout(out):
    st, rp = out["statements"], out["reported"]
    L = ["## Statements (r2 to r9 pooled, 20,000 draws)", "",
         "| statement | value [95 %] | holds | expectation | per re-partition |", "|---|---|---|---|---|"]
    ex = {e["key"]: e for e in out["expectations"]}
    perk = {"R1": rp["class"]["A"]["inc_per_repeat"], "R2": rp["class"]["C"]["inc_per_repeat"],
            "R3": rp["phase"]["inter_BC"]["inc_per_repeat"], "R4": rp["A_minus_C_inc"]["per_repeat"]}
    for k_, v in st.items():
        L.append(f"| **{k_}**: {v['text']} | {s3(v['value'])} | **{'holds' if v['holds'] else 'does not hold'}** | "
                 f"{ex[k_]['expectation']} | {', '.join(f'{x:+.3f}' for x in perk[k_].values())} |")
    L += ["", "## By class and phase", "",
          "| rows | discharges (mean) | rows (r2) | median scored T_i (eV) | over twin | over causal GP | over hold | KFX over KF0 (r3-r9) |",
          "|---|---|---|---|---|---|---|---|"]

    def row(name, v):
        L.append(f"| {name} | {v['discharges_with_rows_mean']:.0f} | {v['rows_per_repeat']['2']:,} | "
                 f"{v['median_scored_ti_eV_mean']:.0f} | {s3(v['inc'])} | {s3(v['gp'])} | {s3(v['hold'])} | {s3(v['kfx'])} |")
    row("all scored rows", rp["all"])
    for c in CLASSES:
        row(f"class {CLAB[c]}", rp["class"][c])
    row("classes B and C", rp["class_BC"])
    for k_, nm in (("burst_BC", "burst phase, B and C"), ("inter_BC", "inter-burst phase, B and C"),
                   ("burst_B", "burst phase, B"), ("inter_B", "inter-burst phase, B"),
                   ("burst_C", "burst phase, C"), ("inter_C", "inter-burst phase, C")):
        row(nm, rp["phase"][k_])
    L += ["", f"Class A minus class C: over the twin {s3(rp['A_minus_C_inc']['value'])}, over the causal GP "
          f"{s3(rp['A_minus_C_gp']['value'])}.", "",
          "## Class by temperature third (increment over the twin; cells with at least 30 discharges)", "",
          "| class | " + " | ".join(B44.THIRDS) + " |", "|---|" + "---|" * len(B44.THIRDS)]
    tw = rp["two_way"]
    for c in list(CLASSES) + ["all"]:
        cells = []
        for t in B44.THIRDS:
            e = tw[f"{c}|{t}"]
            if "inc" in e:
                cells.append(f"{s3(e['inc'])} (n {e['discharges_min']})")
            else:
                cells.append(f"n {e['discharges_min']} (< {MIN_CELL})")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "Over the causal GP in the same cells:", "", "| class | " + " | ".join(B44.THIRDS) + " |",
          "|---|" + "---|" * len(B44.THIRDS)]
    for c in CLASSES:
        L.append(f"| {c} | " + " | ".join(s3(tw[f'{c}|{t}']['gp']) if "gp" in tw[f"{c}|{t}"] else "-" for t in B44.THIRDS) + " |")
    L += ["", "## Expectations (registration section 6)", "", "| key | expectation | observed | range met | verdict met |",
          "|---|---|---|---|---|"]
    for e in out["expectations"]:
        L.append(f"| {e['key']} | {e['expectation']} | {e['observed']} | {e.get('range_met')} | {e.get('verdict_met', '-')} |")
    L += ["", "## Choices fixed in the implementation", ""] + [f"- {c}" for c in out["choices"]] + [""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    {"detect": stage_detect, "counts": stage_counts, "readout": stage_readout}[stage]()
