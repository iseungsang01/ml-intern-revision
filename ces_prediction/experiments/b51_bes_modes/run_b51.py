"""B.51 (PREREGISTRATION_B51.md): the beam-emission array split into its common mode and its differences
between channels.

Part A (model-free): control at the isolated absent frames of B.35 (|change of g| / |change of c|), and the
association of 200 ms block changes of T_i with block changes of the components (B.43 S machinery: s_blocks,
spearmanr per discharge, circular-shift null, summarize_s), statements A1 to A3.
Part B: a second stage on the residual of the history-only twin, fitted on the validation discharges of each
split and applied once to its test rows (B.37 V machinery: run_b37.Ctx.rows with the twin as network, gbr()
of V4), models 0 / C / D / CD, statements B1 and B2.

CPU only; no network is trained (gradient-boosted regressions are fitted on the CPU, as registered).
Nothing in the earlier runners is edited; their functions are imported.  Run from the repo root:
    py ces_prediction/experiments/b51_bes_modes/run_b51.py a            # part A, writes the outputs
    py ces_prediction/experiments/b51_bes_modes/run_b51.py b [SEED ...]  # part B rows (cached), then readout
Cache: data/.b51/ (gitignored).  Output: docs/paper/b51_bes_modes.json / .md
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import sys  # noqa: E402

sys.dont_write_bytecode = True          # importing run_b43 imports docs/paper/ppcf/fig_trace; write nothing there

import json  # noqa: E402
import time as _time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for sub in ("b30_thinned", "b37_plan", "b34_error_model", "b33_clusters", "b41_events", "b32_two_sources",
            "b45_ablation_rows", "b48_reread", "b43_target_checks", "b44_strata", "b42_delay", "b31_bursts"):
    sys.path.insert(0, str(EXP / sub))
import run_b30 as R  # noqa: E402
import run_b48 as B48  # noqa: E402  (patches run_b30.imports to 4 torch threads)
import run_b37 as R37  # noqa: E402
import run_b41 as B41  # noqa: E402
import run_b45 as B45  # noqa: E402
import run_b43 as B43  # noqa: E402
import run_b32 as B32  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_B51.md"
DATA, REPO = R.DATA, R.REPO
CACHE = DATA / ".b51"
OUT_JSON = REPO / "docs" / "paper" / "b51_bes_modes.json"
OUT_MD = REPO / "docs" / "paper" / "b51_bes_modes.md"
SEEDS = (42, 1, 7, 123)
SEED, N_BOOT = 20261118, 2000
CUT_EV = 3000.0
B_ROWS, B_MIN_MEAS, B_MIN_PAIRS = 20, 10, 10          # 200 ms blocks, half measured, at least ten pairs
AFTER_ROWS = 10
MAX_BAD_SHARE = 0.02
CHECK_EV, CHECK_SHARE = 0.5, 0.99
COMPS = ("c", "g", "g1", "g2", "h", "h1", "h2")
FEAT_COMPS = ("c", "g1", "g2", "h1", "h2")
MODELS = {"0": [], "C": ["c"], "D": ["g1", "g2", "h1", "h2"], "CD": ["c", "g1", "g2", "h1", "h2"]}
BASE = ("age_ms", "hold", "twin", "gp", "hold_minus_twin", "time_step")
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))] if len(b) else [float("nan")] * 2


def boot_w(rng, n):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)


# ------------------------------------------------------------------------------------------ components
def bes_columns():
    hdr = pd.read_csv(sorted(DATA.glob("s*.csv"))[0], nrows=0).columns.tolist()
    bes = [c for c in hdr if c.startswith("BES_")]
    if len(bes) != 9:
        raise SystemExit(f"FATAL: expected 9 BES channels, got {bes}")
    col = {k: [j for j, c in enumerate(bes) if c[6:8] == k] for k in ("05", "06", "07")}
    row = {k: [j for j, c in enumerate(bes) if c[4:6] == k] for k in ("01", "02", "03")}
    if any(len(v) != 3 for v in (*col.values(), *row.values())):
        raise SystemExit(f"FATAL: BES channels do not form a 3 x 3 array: {bes}")
    return bes, col, row


BES, COL, ROW = bes_columns()


def components(arr):
    """Rows of a grid array -> dict of the components (NaN where a BES channel is not positive), and that mask."""
    X = arr[:, 3:3 + 9].astype(np.float64)
    bad = ~(X > 0).all(1)
    L = np.log(np.where(X > 0, X, np.nan))
    cm = {k: L[:, v].mean(1) for k, v in COL.items()}
    rm = {k: L[:, v].mean(1) for k, v in ROW.items()}
    o = {"c": L.mean(1), "g1": cm["06"] - cm["05"], "g2": cm["07"] - cm["06"], "g": cm["07"] - cm["05"],
         "h1": rm["02"] - rm["01"], "h2": rm["03"] - rm["02"]}
    o["h"] = o["h1"] + o["h2"]
    return o, bad


def count_bad(grid):
    n = bad = 0
    files = 0
    for nm in sorted(grid):
        a = grid[nm][1:-1]
        _, b = components(a)
        n += len(a)
        bad += int(b.sum())
        files += int(b.any())
    out = {"grid_rows": n, "rows_with_nonpositive_bes": bad, "share": bad / n, "files_with_such_rows": files}
    if bad / n > MAX_BAD_SHARE:
        raise SystemExit(f"FATAL: {bad} of {n} grid rows have a non-positive BES channel (> 2 %); the run stops")
    return out


# ------------------------------------------------------------------------------------------ part A
def control(grid_raw):
    """B.35 events (rule of run_b35.events on the grid without held removal, T_i above 3 keV removed)."""
    names = sorted(grid_raw)
    ev = []                       # file, absent frames, delta c, delta g, delta h
    for fi, nm in enumerate(names):
        a = grid_raw[nm][1:-1]
        if len(a) < 12:
            continue
        t = a[:, 0].astype(np.float64)
        ti = a[:, 1].astype(np.float64)
        ti = np.where(ti <= 3000, ti, np.nan)
        cp, _ = components(a)
        dt = np.diff(t)
        for j in np.flatnonzero((dt > 0.015) & (dt < 0.045)):
            if j < 5 or not np.all(np.abs(dt[j - 5:j] - 0.010) < 1e-4):
                continue
            pre = ti[j - 5:j + 1]
            if np.isfinite(pre).sum() < 3:
                continue
            d = [cp[k][j + 1] - cp[k][j - 5:j + 1].mean() for k in ("c", "g", "h")]
            ev.append((fi, int(round(dt[j] / 0.010)) - 1, *d))
    e = np.array(ev, float)
    ref = json.loads((REPO / "docs" / "paper" / "b35_blip_response.json").read_text(encoding="utf-8"))
    uf, f = np.unique(e[:, 0].astype(int), return_inverse=True)
    chk = {"events": int(len(e)), "discharges": int(len(uf)), "b35_events": ref["events"], "b35_discharges": ref["discharges"],
           "events_one_absent_frame": int((e[:, 1] == 1).sum()), "b35_one": ref["events_one_absent_frame"]}
    chk["equal"] = bool(chk["events"] == ref["events"] and chk["discharges"] == ref["discharges"]
                        and chk["events_one_absent_frame"] == ref["events_one_absent_frame"])
    if not chk["equal"]:
        raise SystemExit(f"FATAL: the B.35 events are not reproduced: {chk}")
    fin = np.isfinite(e[:, 2:]).all(1)
    rng = np.random.default_rng(SEED + 2)
    W = boot_w(rng, len(uf))
    out = {"check_b35_events": chk, "events_with_finite_components": int(fin.sum())}
    for lab, m in (("all", fin), ("one_absent_frame", fin & (e[:, 1] == 1))):
        idx = np.flatnonzero(m)
        ff = f[idx]
        ac, ag, ah = np.abs(e[idx, 2]), np.abs(e[idx, 3]), np.abs(e[idx, 4])
        pc, bc = B32.wmedian(ac, ff, W)
        pg, bg = B32.wmedian(ag, ff, W)
        ph, bh = B32.wmedian(ah, ff, W)
        o = {"events": int(len(idx)), "median_abs_change_c": [pc, *ci(bc)], "median_abs_change_g": [pg, *ci(bg)],
             "median_abs_change_h": [ph, *ci(bh)],
             "ratio_g_over_c": [pg / pc, *ci(bg / bc)], "ratio_h_over_c": [ph / pc, *ci(bh / bc)]}
        for k, col in (("c", 2), ("g", 3), ("h", 4)):
            p, b = B32.wmedian(e[idx, col], ff, W)
            o[f"median_signed_change_{k}"] = [p, *ci(b)]
        with np.errstate(divide="ignore", invalid="ignore"):
            o["median_of_event_ratio_g_over_c"] = float(np.median(ag / ac))
            o["median_of_event_ratio_h_over_c"] = float(np.median(ah / ac))
        out[lab] = o
    out["ratio"] = out["all"]["ratio_g_over_c"]
    log(f"control: {len(e)} events; |dg|/|dc| {out['ratio'][0]:.3f} [{out['ratio'][1]:.3f}, {out['ratio'][2]:.3f}]")
    return out


def rank_resid(a, z):
    ra, rz = rankdata(a), rankdata(z)
    X = np.column_stack([np.ones(len(a)), rz])
    beta = np.linalg.lstsq(X, ra, rcond=None)[0]
    return ra - X @ beta


def pcorr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def part_a_assoc(grid):
    names = sorted(grid)
    med_ti = {}
    per = {k: [] for k in (*COMPS, "g_given_c", "c_given_g")}
    nan_drop = {k: 0 for k in per}
    pairs_total = pairs_noexcl = 0
    used = 0
    rec_ecei = []
    for nm in names:
        a = grid[nm][1:-1]
        t, ti = a[:, 0].astype(np.float64), a[:, 1].astype(np.float64)
        med_ti[nm] = float(np.nanmedian(ti)) if np.isfinite(ti).any() else np.nan
        # check: the B.43 after-the-fact read-out (ECEI level, no exclusion, at least ten pairs)
        ecei = a[:, 12:16].astype(np.float64).mean(1)
        dT, dX = B43.s_blocks(t, ti, ecei, B_ROWS, B_MIN_MEAS)
        if len(dT) >= 10:
            r, r0 = spearmanr(dT, dX)[0], spearmanr(dT, np.roll(dX, len(dT) // 2))[0]
            if np.isfinite(r) and np.isfinite(r0):
                rec_ecei.append((nm, float(r), float(r0)))
        cp, bad = components(a)
        after = B43.after_rows(t)
        excl = ((after >= 1) & (after <= AFTER_ROWS)) | bad
        d = {}
        for k in COMPS:
            dT_k, dX_k = B43.s_blocks(t, ti, np.where(excl, np.nan, cp[k]), B_ROWS, B_MIN_MEAS)
            d[k] = dX_k
            if k == "c":
                dT_ref = dT_k
                pairs_noexcl += len(B43.s_blocks(t, ti, cp["c"], B_ROWS, B_MIN_MEAS)[0])
            elif not np.array_equal(dT_k, dT_ref):
                raise SystemExit("FATAL: block pairs differ between components")
        keep = np.isfinite(d["c"])
        for k in COMPS:
            if not np.array_equal(np.isfinite(d[k]), keep):
                raise SystemExit("FATAL: excluded pairs differ between components")
        dT = dT_ref[keep]
        d = {k: v[keep] for k, v in d.items()}
        n = len(dT)
        if n < B_MIN_PAIRS:
            continue
        used += 1
        pairs_total += n
        for k in COMPS:
            r, r0 = spearmanr(dT, d[k])[0], spearmanr(dT, np.roll(d[k], n // 2))[0]
            if np.isfinite(r) and np.isfinite(r0):
                per[k].append((nm, float(r), float(r0)))
            else:
                nan_drop[k] += 1
        for lab, x, z in (("g_given_c", d["g"], d["c"]), ("c_given_g", d["c"], d["g"])):
            rT, rx = rank_resid(dT, z), rank_resid(x, z)
            r, r0 = pcorr(rT, rx), pcorr(rT, np.roll(rx, n // 2))
            if np.isfinite(r) and np.isfinite(r0):
                per[lab].append((nm, r, r0))
            else:
                nan_drop[lab] += 1
    ref = json.loads((REPO / "docs" / "paper" / "b43_target_checks.json").read_text(encoding="utf-8"))
    ref10 = ref["after_fact_S"]["by_min_pairs"]["10"]
    chk = {"discharges": len(rec_ecei), "median_rho": float(np.median([r[1] for r in rec_ecei])),
           "median_null": float(np.median([r[2] for r in rec_ecei])),
           "b43_discharges": ref10["discharges"], "b43_median_rho": ref10["median_rho"][0],
           "b43_median_null": ref10["median_null"][0]}
    chk["equal"] = bool(chk["discharges"] == chk["b43_discharges"] and abs(chk["median_rho"] - chk["b43_median_rho"]) < 1e-9
                        and abs(chk["median_null"] - chk["b43_median_null"]) < 1e-9)
    if not chk["equal"]:
        raise SystemExit(f"FATAL: the B.43 read-out (ECEI, ten pairs) is not reproduced: {chk}")
    out = {"check_b43_ecei_min10": chk, "discharges_with_ten_pairs": used, "block_pairs": pairs_total,
           "block_pairs_before_exclusion_all_discharges": pairs_noexcl, "nan_rho_dropped": nan_drop, "by_component": {}}
    for k, rr in per.items():
        # the same generator state for every component: identical discharge resamples when the discharges agree
        s = B43.summarize_s(rr, med_ti, np.random.default_rng(SEED + 1), nan_drop[k])
        s["pairs"] = int(pairs_total)
        out["by_component"][k] = s
        log(f"A {k}: {s['discharges']} discharges, median rho {s['median_rho'][0]:+.3f}, null {s['median_null'][0]:+.3f}, "
            f"diff {s['diff'][0]:+.3f} [{s['diff'][1]:+.3f}, {s['diff'][2]:+.3f}]")
    bc = out["by_component"]
    out["A1"] = {"diff": bc["g"]["diff"], "holds": bool(bc["g"]["diff"][1] > 0), "discharges": bc["g"]["discharges"]}
    out["A2"] = {"diff": bc["c"]["diff"], "holds": bool(bc["c"]["diff"][1] > 0), "discharges": bc["c"]["discharges"]}
    out["A3"] = {"diff": bc["g_given_c"]["diff"], "holds": bool(bc["g_given_c"]["diff"][1] > 0),
                 "discharges": bc["g_given_c"]["discharges"]}
    return out


def stage_a():
    R.pin_env()
    load_grid_files = R.imports()[3]
    grid, dims = load_grid_files(DATA, True, CUT_EV)
    if len(grid) != 641 or dims["bes"] != 9 or dims["ecei"] != 4:
        raise SystemExit("FATAL: expected 641 files, 9 BES and 4 ECEI channels")
    log("cut grid loaded")
    res = {"bad_rows": count_bad(grid)}
    log(f"non-positive BES rows: {res['bad_rows']}")
    res.update(part_a_assoc(grid))
    del grid
    grid_raw, _ = load_grid_files(DATA, False, 0.0)
    log("raw grid loaded")
    res["control"] = control(grid_raw)
    res["runtime_s"] = round(_time.time() - T0, 1)
    return res


# ------------------------------------------------------------------------------------------ part B
class TwinCtx(R37.Ctx):
    """run_b37.Ctx with the network pointed at the history-only twin (model v2h, per-discharge standardization,
    as run_b30.NETS['cut']['hist80']); `rows` of run_b37.Ctx is used unchanged."""

    def net(self, s):
        run = DATA / B48.TWIN.format(s=s)
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if (m["seq_model"] != "v2h" or not bool(m["per_shot_input_norm"]) or m.get("per_shot_norm_expanding")
                or m.get("zero_mc") or m.get("extra_vt_channels") or m.get("extra_ti_channels") or m.get("train_ctx")):
            raise SystemExit(f"FATAL: {run.name} is not the history-only twin")
        stats = self.load_stats(m)
        model = self.MODELS["v2h"]()
        model.load_state_dict(self.torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
        model.eval()
        return model, stats

    def predict(self, arr, net):
        model, stats = net
        tm = arr[:, 0].astype(np.float64)
        spans = R.block_spans(tm, self.GAP)
        blks = self.build_blocks(arr, self.dims, stats, per_shot_norm=True, expanding=False)
        if len(blks) != len(spans):
            raise SystemExit("FATAL: block spans do not match build_blocks")
        out = np.full(len(tm), np.nan)
        mean, std = float(stats["target"]["mean"][0]), float(stats["target"]["std"][0])
        with self.torch.no_grad():
            for blk, (a, b) in zip(blks, spans):
                out[a:b] = model(self.torch.from_numpy(blk["x"]).unsqueeze(0))[0][:, 0].numpy() * std + mean
        return out, spans


def rows_cache(C, s):
    """Validation rows (twin, hold, causal GP by the B.37 machinery) and the check on every test row."""
    p = CACHE / f"rows_s{s}.npz"
    if p.exists():
        return
    t0 = _time.time()
    man, net = C.manifest(s), C.net(s)
    tab = B48.load_tab(B48.CACHE / f"rows_cut_s{s}.npz")
    k1 = tab["k"] == 1
    test = sorted(set(np.unique(tab["file"][k1]).astype(int).tolist()))
    if {C.names[i] for i in test} - set(man["test_files"]):
        raise SystemExit(f"FATAL: split {s}: row-table files outside the test set")
    # check 1: row = grid row, file = index into the sorted grid (y of the table against the grid T_i)
    yg = np.array([C.grid[C.names[int(f)]][int(r), 1] for f, r in zip(tab["file"][k1], tab["row"][k1])], np.float64)
    yt = tab["y"][k1].astype(np.float64)
    ymis = int((~(np.abs(yg - yt) <= 1e-3)).sum())
    # check 2: the machinery on every test discharge against the row table
    lut = {(int(f), int(r)): j for j, (f, r) in enumerate(zip(tab["file"][k1], tab["row"][k1]))}
    sub = {e: tab[e][k1].astype(np.float64) for e in ("hist80", "hold", "gp", "y", "age_ms")}
    dev = {e: 0 for e in ("twin", "hold", "gp", "y", "age")}
    maxd = {e: 0.0 for e in dev}
    seen = 0
    for i in test:
        rw = C.rows(i, net)
        for r, yv, tw, hv, gv, ag in rw:
            j = lut.get((i, int(r)))
            if j is None:
                continue
            seen += 1
            for e, mine, theirs in (("twin", tw, sub["hist80"][j]), ("hold", hv, sub["hold"][j]), ("gp", gv, sub["gp"][j]),
                                    ("y", yv, sub["y"][j]), ("age", ag, sub["age_ms"][j])):
                d = abs(mine - theirs)
                if np.isnan(mine) and np.isnan(theirs):
                    continue
                if not d <= CHECK_EV:
                    dev[e] += 1
                if np.isfinite(d):
                    maxd[e] = max(maxd[e], float(d))
    n = int(k1.sum())
    chk = {"table_rows_k1": n, "y_vs_grid_mismatch": ymis, "rows_recomputed_and_matched": seen,
           "missing": n - seen, "deviating": dev, "max_abs_dev": maxd,
           "share_twin_within_0.5_eV": 1 - (dev["twin"] + n - seen) / n,
           "share_hold_within_0.5_eV": 1 - (dev["hold"] + n - seen) / n,
           "share_gp_within_0.5_eV": 1 - (dev["gp"] + n - seen) / n}
    chk["passes"] = bool(ymis == 0 and chk["share_twin_within_0.5_eV"] >= CHECK_SHARE
                         and chk["share_hold_within_0.5_eV"] >= CHECK_SHARE and chk["share_gp_within_0.5_eV"] >= CHECK_SHARE)
    log(f"s{s}: check {json.dumps(chk)}")
    CACHE.mkdir(exist_ok=True)
    (CACHE / f"check_s{s}.json").write_text(json.dumps(chk, indent=1), encoding="utf-8")
    if not chk["passes"]:
        raise SystemExit(f"FATAL: split {s}: the machinery does not reproduce the row table; the run stops")
    val = [C.names.index(nm) for nm in sorted(man["val_files"]) if nm in C.grid]
    acc = []
    for i in val:
        rw = C.rows(i, net)
        if len(rw):
            acc.append(np.column_stack([np.full(len(rw), i), rw]))
    V = np.concatenate(acc)
    np.savez_compressed(p, fidx=V[:, 0].astype(np.int32), row=V[:, 1].astype(np.int32), y=V[:, 2], twin=V[:, 3],
                        hold=V[:, 4], gp=V[:, 5], age_ms=V[:, 6], n_val_files=len(val))
    log(f"s{s}: {len(val)} validation discharges, {len(V)} rows cached ({_time.time() - t0:.0f} s)")


def dep10(x, seg):
    """x minus the mean of the ten rows before in the same segment (window_feats column 1, NaN-safe)."""
    n = len(x)
    ok = np.isfinite(x)
    c1 = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(~ok)))
    out = np.full(n, np.nan)
    i = np.arange(10, n)
    i = i[(seg[i] == seg[i - 10]) & (cn[i] - cn[i - 10] == 0) & ok[i]]
    out[i] = x[i] - (c1[i] - c1[i - 10]) / 10.0
    return out


class Feats:
    def __init__(self, C, train_names):
        self.C = C
        vals = {k: [] for k in FEAT_COMPS}
        for nm in train_names:
            cp, _ = components(C.grid[nm][1:-1])
            for k in FEAT_COMPS:
                vals[k].append(cp[k])
        self.mu = {k: float(np.nanmean(np.concatenate(v))) for k, v in vals.items()}
        self.cache = {}

    def of_file(self, i):
        if i not in self.cache:
            C = self.C
            cp, _ = components(C.grid[C.names[i]])
            seg = C.seg[i]
            cols = []
            for k in FEAT_COMPS:
                x = cp[k]
                d = dep10(x, seg)
                dprev = np.full(len(x), np.nan)
                same = np.concatenate(([False], seg[1:] == seg[:-1]))
                dprev[1:] = np.where(same[1:], d[:-1], np.nan)
                cols += [x - self.mu[k], d, dprev]
            self.cache[i] = np.column_stack(cols)
        return self.cache[i]


def design(F, fidx, rows, age, hold, twin, gp, tstep):
    base = np.column_stack([age, hold, twin, gp, hold - twin, tstep])
    fidx, rows = np.asarray(fidx).astype(int), np.asarray(rows).astype(int)
    fr = np.full((len(rows), 3 * len(FEAT_COMPS)), np.nan)
    for i in np.unique(fidx):
        m = fidx == i
        fr[m] = F.of_file(int(i))[rows[m]]
    return base, fr


def model_cols(name):
    idx = []
    for k in MODELS[name]:
        j = FEAT_COMPS.index(k)
        idx += [3 * j, 3 * j + 1, 3 * j + 2]
    return idx


def readout_split(C, Cl, s, valid, b39):
    man = C.manifest(s)
    train = [nm for nm in sorted(man["train_files"]) if nm in C.grid]
    F = Feats(C, train)
    z = np.load(CACHE / f"rows_s{s}.npz")
    V = {k: z[k] for k in z.files}
    tab = B48.load_tab(B48.CACHE / f"rows_cut_s{s}.npz")
    k1 = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in ("y", "hold", "gp", "hist80", "age_ms")], axis=0)
    # ---------- the scoring code against B.48 (population of run_b44: k = 1, hold, GP, twin, causal, y finite)
    pop = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in ("hold", "gp", "hist80", "causal", "y")], axis=0)
    y_ = tab["y"][pop].astype(np.float64)
    se_ = {e: (tab[e][pop].astype(np.float64) - y_) ** 2 for e in ("hold", "gp", "hist80", "causal")}
    b48 = json.loads((REPO / "docs" / "paper" / "b48_reread.json").read_text(encoding="utf-8"))
    ov = b48["b44"]["splits"][str(s)]["overall"]
    sc = {"causal_over_twin_mine": float(1 - se_["causal"].sum() / se_["hist80"].sum()), "causal_over_twin_b48": ov["hist80"][0],
          "causal_over_gp_mine": float(1 - se_["causal"].sum() / se_["gp"].sum()), "causal_over_gp_b48": ov["gp"][0],
          "twin_over_gp_mine": float(1 - se_["hist80"].sum() / se_["gp"].sum()),
          "twin_over_gp_b42_L0_in_b48": b48["b42"]["splits"][str(s)]["0"]["skill"]["twin_vs_gp"][0]}
    sc["equal_to_3_decimals"] = bool(round(sc["causal_over_twin_mine"], 3) == round(sc["causal_over_twin_b48"], 3)
                                     and round(sc["causal_over_gp_mine"], 3) == round(sc["causal_over_gp_b48"], 3))
    if not sc["equal_to_3_decimals"]:
        raise SystemExit(f"FATAL: split {s}: the scoring code does not reproduce B.48: {sc}")
    # ---------- design matrices
    tstep_t = np.array([C.tstep[int(f)][int(r)] for f, r in zip(tab["file"][k1], tab["row"][k1])])
    bt, ft = design(F, tab["file"][k1], tab["row"][k1], tab["age_ms"][k1].astype(np.float64), tab["hold"][k1].astype(np.float64),
                    tab["hist80"][k1].astype(np.float64), tab["gp"][k1].astype(np.float64), tstep_t)
    okt = np.isfinite(bt).all(1) & np.isfinite(ft).all(1)
    vfin = np.all([np.isfinite(V[e]) for e in ("y", "twin", "hold", "gp", "age_ms")], axis=0)
    tstep_v = np.array([C.tstep[int(f)][int(r)] for f, r in zip(V["fidx"], V["row"])])
    bv, fv = design(F, V["fidx"], V["row"], V["age_ms"], V["hold"], V["twin"], V["gp"], tstep_v)
    okv = vfin & np.isfinite(bv).all(1) & np.isfinite(fv).all(1)
    y = tab["y"][k1][okt].astype(np.float64)
    tw = tab["hist80"][k1][okt].astype(np.float64)
    gp = tab["gp"][k1][okt].astype(np.float64)
    tfile, trow = tab["file"][k1][okt].astype(int), tab["row"][k1][okt].astype(int)
    yv, twv = V["y"][okv], V["twin"][okv]
    pred = {}
    for name in MODELS:
        cols = model_cols(name)
        Xv = np.column_stack([bv[okv], fv[okv][:, cols]])
        Xt = np.column_stack([bt[okt], ft[okt][:, cols]])
        pred[name] = R37.gbr().fit(Xv, yv - twv).predict(Xt)
    se = {name: (y - tw - pred[name]) ** 2 for name in MODELS}
    se["twin"], se["gp"] = (y - tw) ** 2, (y - gp) ** 2
    # quiet rows (B.45 classes, class 6)
    quiet = np.array([Cl.of_file(C.names[i])[0][r] == 6 for i, r in zip(tfile, trow)])
    uf, f = np.unique(tfile, return_inverse=True)
    n = len(uf)
    W = boot_w(np.random.default_rng(SEED + s), n)

    def sk(a, b, m=None):
        m = np.ones(len(y), bool) if m is None else m
        sa, sb = np.bincount(f, se[a] * m, n), np.bincount(f, se[b] * m, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return [float(1 - sa.sum() / sb.sum()), *ci(1 - (W @ sa) / (W @ sb))]

    pairs = [("D", "0"), ("C", "0"), ("CD", "0"), ("CD", "C"), ("CD", "D")]
    o = {"train_discharges": len(train), "validation_discharges": int(z["n_val_files"]),
         "validation_rows_all": int(len(V["y"])), "validation_rows": int(okv.sum()),
         "test_rows_k1": int(k1.sum()), "test_rows": int(okt.sum()), "test_rows_dropped_by_features": int((~okt).sum()),
         "test_discharges": n, "quiet_rows": int(quiet.sum()), "train_means": F.mu, "scoring_check": sc,
         "skill": {f"{a}_over_{b}": sk(a, b) for a, b in pairs}}
    for name in MODELS:
        o["skill"][f"{name}_over_twin"] = sk(name, "twin")
        o["skill"][f"{name}_over_gp"] = sk(name, "gp")
    o["skill"]["twin_over_gp"] = sk("twin", "gp")
    o["quiet"] = {f"{a}_over_{b}": sk(a, b, quiet) for a, b in pairs}
    for name in MODELS:
        o["quiet"][f"{name}_over_twin"] = sk(name, "twin", quiet)
    s0, sD, sCD = (np.bincount(f, se[k], n) for k in ("0", "D", "CD"))
    with np.errstate(invalid="ignore", divide="ignore"):
        o["ratio_reduction_D_over_CD"] = [float((s0.sum() - sD.sum()) / (s0.sum() - sCD.sum())),
                                          *ci((W @ (s0 - sD)) / (W @ (s0 - sCD)))]
    o["rmse_eV"] = {k: float(np.sqrt(v.mean())) for k, v in se.items()}
    o["B1_holds"] = bool(o["skill"]["D_over_0"][1] > 0)
    o["B2_holds"] = bool(o["skill"]["C_over_0"][1] > 0)
    # ---------- the BES network of B.39 against the twin, on the rows both score
    bes = B45.load_run(DATA / f".b39_v2b_cut_s{s}_i{s}")
    twr = B45.load_run(DATA / B48.TWIN.format(s=s))
    if not (np.array_equal(bes["shot"], twr["shot"]) and np.array_equal(bes["row"], twr["row"])):
        raise SystemExit(f"FATAL: split {s}: the BES network and the twin are not scored on the same rows")
    key = {}
    for j, (sh, r) in enumerate(zip(bes["shot"], bes["row"])):
        nm = valid[int(sh)]
        gi = B45.compact_index(C.grid[nm])
        key[(C.names.index(nm), int(gi[int(r)]))] = j
    both = np.array([key.get((int(i), int(r)), -1) for i, r in zip(tfile, trow)])
    m = both >= 0
    jj = both[m]
    yagree = float(np.mean(np.abs(bes["y"][jj] - y[m]) <= CHECK_EV))
    twagree = float(np.mean(np.abs(np.sqrt(twr["model"][jj]) - np.abs(tw[m] - y[m])) <= CHECK_EV))
    se["bes_net"] = np.zeros(len(y))
    se["twin_stored"] = np.zeros(len(y))
    se["bes_net"][m] = bes["model"][jj]
    se["twin_stored"][m] = twr["model"][jj]
    o["bes_network"] = {"rows_both": int(m.sum()), "harness_rows": int(len(bes["shot"])),
                        "y_agree_within_0.5_eV": yagree, "stored_twin_error_agrees_within_0.5_eV": twagree,
                        "bes_net_over_twin_stored": sk("bes_net", "twin_stored", m),
                        "b39_D2_all_harness_rows": b39["A"][str(s)]["D2_bes_over_twin"]}
    for name in MODELS:
        o["bes_network"][f"{name}_over_twin_same_rows"] = sk(name, "twin", m)
    log(f"s{s}: B1 D/0 {o['skill']['D_over_0']} B2 C/0 {o['skill']['C_over_0']} CD/0 {o['skill']['CD_over_0']} "
        f"ratio {o['ratio_reduction_D_over_CD']}")
    return o


def stage_b(seeds):
    C = TwinCtx()
    log("grid loaded (run_b37.Ctx)")
    for s in seeds:
        rows_cache(C, s)
    if not all((CACHE / f"rows_s{s}.npz").exists() for s in SEEDS):
        log("not all splits cached yet; readout skipped")
        return None
    C41 = B41.Ctx()
    if C41.names != C.names:
        raise SystemExit("FATAL: the two grids differ")
    Cl = B45.Classes(C41)
    valid = [nm for nm in C.names if np.isfinite(C.grid[nm][:, 1:3]).any()]
    b39 = json.loads((REPO / "docs" / "paper" / "b39_primary.json").read_text(encoding="utf-8"))
    res = {"checks": {str(s): json.loads((CACHE / f"check_s{s}.json").read_text(encoding="utf-8")) for s in SEEDS},
           "splits": {}}
    for s in SEEDS:
        res["splits"][str(s)] = readout_split(C, Cl, s, valid, b39)
    res["csv_check_classes"] = Cl.csv_check
    for k in ("B1", "B2"):
        nh = sum(res["splits"][str(s)][f"{k}_holds"] for s in SEEDS)
        res[k] = {"splits": int(nh), "holds": bool(nh >= 3)}
    res["runtime_readout_s"] = round(_time.time() - T0, 1)
    return res


# ------------------------------------------------------------------------------------------ verdicts, render
def expectations(res):
    E = []
    A, B = res.get("A"), res.get("B")
    if A:
        r = A["control"]["ratio"][0]
        E.append(("control ratio 0.05 to 0.3", f"{r:.3f}", 0.05 <= r <= 0.3))
        d = A["A2"]["diff"][0]
        E.append(("A2 holds, difference to the null +0.10 to +0.35",
                  f"{d:+.3f}, {'holds' if A['A2']['holds'] else 'does not hold'}", A["A2"]["holds"] and 0.10 <= d <= 0.35))
        d = A["A1"]["diff"][0]
        E.append(("A1 uncertain: difference +0.00 to +0.15", f"{d:+.3f}, {'holds' if A['A1']['holds'] else 'does not hold'}",
                  0.0 <= d <= 0.15))
        E.append(("A3 uncertain (no range)", f"{A['A3']['diff'][0]:+.3f}, {'holds' if A['A3']['holds'] else 'does not hold'}", None))
    if B:
        S = [str(s) for s in SEEDS]
        sp = B["splits"]
        v = [sp[s]["skill"]["C_over_0"][0] for s in S]
        E.append(("model C over model 0: +0.01 to +0.06, B2 holds on three or four splits",
                  f"points {', '.join(f'{x:+.3f}' for x in v)}; B2 on {B['B2']['splits']} of 4",
                  all(0.01 <= x <= 0.06 for x in v) and B["B2"]["splits"] in (3, 4)))
        v = [sp[s]["skill"]["D_over_0"][0] for s in S]
        E.append(("model D over model 0: 0.00 to +0.04, B1 holds on one to three splits",
                  f"points {', '.join(f'{x:+.3f}' for x in v)}; B1 on {B['B1']['splits']} of 4",
                  all(0.0 <= x <= 0.04 for x in v) and B["B1"]["splits"] in (1, 2, 3)))
        v = [sp[s]["skill"]["CD_over_0"][0] for s in S]
        E.append(("model CD over model 0: +0.02 to +0.08", ", ".join(f"{x:+.3f}" for x in v), all(0.02 <= x <= 0.08 for x in v)))
        v = [sp[s]["ratio_reduction_D_over_CD"][0] for s in S]
        E.append(("reduction of D over that of CD: 0.2 to 0.7", ", ".join(f"{x:.3f}" for x in v), all(0.2 <= x <= 0.7 for x in v)))
    return [{"expectation": a, "observed": b, "met": (None if c is None else bool(c))} for a, b, c in E]


def paper_branch(res):
    A, B = res.get("A"), res.get("B")
    out = []
    if B:
        b1, b2 = B["B1"]["holds"], B["B2"]["holds"]
        if b1:
            out.append("B1 holds: the caveat on the beam is restated with the differences between channels"
                       + (" (A1 or A3 holds and is quoted with it)." if A and (A["A1"]["holds"] or A["A3"]["holds"]) else "."))
        elif b2:
            out.append("B1 does not hold and B2 holds: the information is in the common mode, where the beam and the density are "
                       "not separated; the caveat stands in full and the deciding measurement stays the beam power.")
        else:
            out.append("Neither B1 nor B2 holds: a second stage on the validation discharges does not recover the increment "
                       "from these features; nothing is concluded about the network; the deciding measurement is a network "
                       "retrained on the differences alone (GPU).")
    if A:
        r = A["control"]["ratio"][0]
        out.append(f"Control ratio {r:.3f} " + ("is above 0.5: no sentence of the paper calls the differences beam-free."
                                               if r > 0.5 else "is not above 0.5."))
    return out


CHOICES = [
    "Components: natural logarithm of the nine raw BES values of a grid row; c = mean over the nine; the column of a channel "
    "is the last two digits of its name (05, 06, 07), the poloidal row the two digits before (01, 02, 03); g1 = mean(col 06) - "
    "mean(col 05), g2 = mean(col 07) - mean(col 06), g = g1 + g2, h1 = mean(row 02) - mean(row 01), h2 = mean(row 03) - "
    "mean(row 02), h = h1 + h2. The non-positive count is over the grid rows without the first and last row of each file.",
    "Control: the events of run_b35.events (step of more than 15 and less than 45 ms after five 10 ms steps, T_i observed in "
    "at least three of the six rows before, values above 3 keV removed, held values kept), rebuilt on the grid of "
    "seq_data.load_grid_files without held removal and checked equal in number to b35_blip_response.json. Change = value in "
    "the first row after the step minus the mean of the six rows before (rows j - 5 to j). The ratio is the median over events "
    "of |change of g| over the median of |change of c|; the median over events of the per-event ratio is reported beside it. "
    "Interval: 2,000 resamples of the discharges (numpy default_rng(20261120)).",
    "Part A uses run_b43.s_blocks unchanged (20-row blocks inside segments split at steps of 0.5 s or more, trailing partial "
    "block dropped, a block used when at least 10 of its rows have a T_i value, block means of T_i over the measured rows and "
    "of the component over all rows, pairs of consecutive used blocks) on the cut grid with held values removed, and "
    "run_b43.summarize_s for the median, the null and the interval. Exclusion: the component is set to NaN on every row that "
    "is one of the ten rows after an absent frame (run_b43.after_rows: rows 1 to 10 after a 20 to 40 ms step inside the "
    "segment) or has a non-positive BES channel, and pairs whose component change is not finite are dropped. A block that "
    "contains the rows after a gap is thereby left out; when the gap falls exactly between two blocks, the block after it is "
    "left out (the block before has none of its rows after the gap). The pairs and the discharges are the same for every "
    "component. Discharges with at least ten pairs after the exclusion.",
    "A3: within a discharge, the ranks of the change of T_i and of the change of g are each regressed linearly on the ranks of "
    "the change of c; the partial correlation is the Pearson correlation of the two residuals; the null shifts the residual of "
    "g circularly by floor(n/2). The same for c after g (reported).",
    "Intervals of part A: every component is summarised with a fresh numpy default_rng(20261119), so the discharge resamples "
    "are the same for every component that keeps the same discharges.",
    "Part B: twin, hold and causal GP on the validation discharges with run_b37.Ctx.rows (unchanged) and the network pointed at "
    "the twin `data/.b29a1_v2h_cut_s{seed}` (model v2h, per-discharge standardization, as run_b30.NETS). The same machinery was "
    "run on every test discharge and matched by (file, row) to every k = 1 row of the B.48 table. Row table rows: file = index "
    "into the sorted file names of the cut grid, row = grid row (checked: the T_i of the grid at that row equals y on every "
    "row).",
    "Part B population: k = 1 rows of the B.48 table with y, hold, the causal GP, the twin and the age finite, and every base "
    "and component feature finite (the same rows for the four models); validation rows by the same rule. Time step = the step "
    "to the row before over 10 ms (run_b37.Ctx.tstep). Component features: s minus its mean over the grid rows of the training "
    "discharges of the split; s minus the mean of s over the ten rows before in the same segment (0.5 s rule of run_b37.Ctx); "
    "the latter at the row before (NaN when that row is in another segment).",
    "Regression: run_b37.gbr() (HistGradientBoostingRegressor, random_state 20261116, defaults), fitted on the validation rows "
    "with target y minus the twin, applied once to the test rows. Skill of model M over N = 1 - SSE(y - twin - correction_M) / "
    "SSE(y - twin - correction_N). Intervals: 2,000 resamples of the test discharges, numpy default_rng(20261118 + split seed), "
    "the same resamples for every quantity of a split.",
    "Quiet rows: class 6 of run_b45.Classes (verbatim). 'The same three skills restricted to quiet rows' is read as CD over 0, "
    "over C and over D; D over 0, C over 0 and every model over the twin are also given.",
    "BES network of B.39: the stored squared errors of `data/.b39_v2b_cut_s{seed}_i{seed}` and of the twin run, mapped from "
    "the compacted row index of the harness to the grid row (run_b45.compact_index) and intersected with the rows of part B.",
]


def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def f3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def hs(h):
    return "holds" if h else "does not hold"


def render(res):
    L = [f"# B.51 -- the beam-emission array split into its common mode and its differences between channels "
         f"(registered: {REG})", "",
         "CPU; no network trained (part B fits gradient-boosted regressions). T_i, cut population. c = mean of the "
         "logarithms of the nine BES channels (common mode); g1, g2 = radial differences (column 06 - 05, 07 - 06), g = "
         "g1 + g2; h1, h2 = poloidal differences (row 02 - 01, 03 - 02), h = h1 + h2. Intervals: 95 %, 2,000 resamples of "
         "discharges. A statement of part B holds if its interval is above zero on at least three of the four splits.", ""]
    A, B = res.get("A"), res.get("B")
    L += ["## Registered statements", ""]
    if A:
        for k, txt in (("A1", "change of T_i against change of g, minus the shifted null"),
                       ("A2", "the same for the common mode c"),
                       ("A3", "the same for g with the change of c removed from both (partial rank correlation)")):
            a = A[k]
            L.append(f"- **{k}** ({txt}): **{s3(a['diff'])}** ({a['discharges']} discharges) -> **{hs(a['holds'])}**.")
    else:
        L.append("- Part A not run.")
    if B:
        S = [str(s) for s in SEEDS]
        for k, key, txt in (("B1", "D_over_0", "model D (differences) over model 0"), ("B2", "C_over_0", "model C (common mode) over model 0")):
            L.append(f"- **{k}** ({txt}, interval above zero): " + "; ".join(f"split {s} {s3(B['splits'][s]['skill'][key])}" for s in S)
                     + f"; {B[k]['splits']} of 4 -> **{hs(B[k]['holds'])}**.")
    else:
        L.append("- Part B not run yet.")
    L += ["", "What goes into the paper (section 6): " + " ".join(paper_branch(res)), ""]
    if A:
        c = A["control"]
        L += ["## Part A", "", "### Control at the isolated absent frames (reported)", "",
              f"Events of B.35: {c['check_b35_events']['events']:,} in {c['check_b35_events']['discharges']} discharges "
              f"(b35_blip_response.json: {c['check_b35_events']['b35_events']:,}, {c['check_b35_events']['b35_discharges']}; "
              f"equal: {c['check_b35_events']['equal']}); {c['events_with_finite_components']:,} with every component finite.", "",
              "| events | n | median abs change c | g | h | ratio g / c | ratio h / c | median of the per-event ratio g / c, h / c | "
              "median signed change c | g | h |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for lab, txt in (("all", "all"), ("one_absent_frame", "one absent frame")):
            o = c[lab]
            L.append(f"| {txt} | {o['events']:,} | {f3(o['median_abs_change_c'])} | {f3(o['median_abs_change_g'])} | "
                     f"{f3(o['median_abs_change_h'])} | **{f3(o['ratio_g_over_c'])}** | {f3(o['ratio_h_over_c'])} | "
                     f"{o['median_of_event_ratio_g_over_c']:.3f}, {o['median_of_event_ratio_h_over_c']:.3f} | "
                     f"{s3(o['median_signed_change_c'])} | {s3(o['median_signed_change_g'])} | {s3(o['median_signed_change_h'])} |")
        bc = A["by_component"]
        lab = {"c": "c (common mode)", "g": "g (radial, 07 - 05)", "g1": "g1 (06 - 05)", "g2": "g2 (07 - 06)",
               "h": "h1 + h2 (poloidal, 03 - 01)", "h1": "h1 (02 - 01)", "h2": "h2 (03 - 02)",
               "g_given_c": "g after c removed (partial)", "c_given_g": "c after g removed (partial)"}
        L += ["", "### Association of 200 ms block changes with the change of T_i", "",
              f"{A['discharges_with_ten_pairs']} discharges with at least ten pairs of blocks after the exclusion, "
              f"{A['block_pairs']:,} pairs (before the exclusion, all discharges: {A['block_pairs_before_exclusion_all_discharges']:,}). "
              "Median over discharges of the Spearman correlation; null = one series shifted circularly by half its length.", "",
              "| component | discharges | median rho | median null | difference | share rho > 0 |", "|---|---|---|---|---|---|"]
        for k in (*COMPS, "g_given_c", "c_given_g"):
            o = bc[k]
            L.append(f"| {lab[k]} | {o['discharges']} | {s3(o['median_rho'])} | {s3(o['median_null'])} | **{s3(o['diff'])}** | "
                     f"{o['share_rho_positive']:.3f} |")
        L += ["", "By third of the discharge median T_i (difference to the null):", "",
              "| component | low third | middle | high |", "|---|---|---|---|"]
        for k in ("c", "g", "g_given_c", "h"):
            th = bc[k]["thirds"]
            L.append(f"| {lab[k]} | " + " | ".join(f"{s3(x['diff'])} ({x['discharges']})" if "diff" in x else f"({x['discharges']})"
                                                     for x in th) + " |")
        ce = A["check_b43_ecei_min10"]
        L += ["", f"Check: the B.43 read-out (ECEI level, no exclusion, at least ten pairs) recomputed with the same functions: "
              f"{ce['discharges']} discharges, median rho {ce['median_rho']:+.4f}, null {ce['median_null']:+.4f} (B.43: "
              f"{ce['b43_discharges']}, {ce['b43_median_rho']:+.4f}, {ce['b43_median_null']:+.4f}); equal: {ce['equal']}.", "",
              f"Rows with a non-positive BES channel: {A['bad_rows']['rows_with_nonpositive_bes']} of {A['bad_rows']['grid_rows']:,} "
              f"grid rows ({A['bad_rows']['files_with_such_rows']} files); dropped.", ""]
    if B:
        S = [str(s) for s in SEEDS]
        sp = B["splits"]
        L += ["## Part B", "", "### Checks", "",
              "| split | k = 1 rows | y vs grid T_i mismatches | rows recomputed | twin within 0.5 eV | hold | causal GP | max dev twin / hold / GP (eV) | "
              "scoring code: causal over twin, over GP (mine / B.48) | twin over GP (mine / B.42 L = 0 in B.48) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in S:
            c = B["checks"][s]
            sc = sp[s]["scoring_check"]
            L.append(f"| {s} | {c['table_rows_k1']:,} | {c['y_vs_grid_mismatch']} | {c['rows_recomputed_and_matched']:,} | "
                     f"{c['share_twin_within_0.5_eV']:.5f} | {c['share_hold_within_0.5_eV']:.5f} | {c['share_gp_within_0.5_eV']:.5f} | "
                     f"{c['max_abs_dev']['twin']:.2e} / {c['max_abs_dev']['hold']:.2e} / {c['max_abs_dev']['gp']:.2e} | "
                     f"{sc['causal_over_twin_mine']:+.4f} / {sc['causal_over_twin_b48']:+.4f}, {sc['causal_over_gp_mine']:+.4f} / "
                     f"{sc['causal_over_gp_b48']:+.4f} | {sc['twin_over_gp_mine']:+.4f} / {sc['twin_over_gp_b42_L0_in_b48']:+.4f} |")
        L += ["", "### Rows", "", "| split | training discharges | validation discharges / rows (all, used) | test rows (k = 1, used, dropped) | "
              "test discharges | quiet rows |", "|---|---|---|---|---|---|"]
        for s in S:
            o = sp[s]
            L.append(f"| {s} | {o['train_discharges']} | {o['validation_discharges']} / {o['validation_rows_all']:,}, {o['validation_rows']:,} | "
                     f"{o['test_rows_k1']:,}, {o['test_rows']:,}, {o['test_rows_dropped_by_features']:,} | {o['test_discharges']} | "
                     f"{o['quiet_rows']:,} |")
        L += ["", "### Skills (all rows)", "", "| split | D over 0 (B1) | C over 0 (B2) | CD over 0 | CD over C | CD over D | "
              "reduction of D / reduction of CD |", "|---|---|---|---|---|---|---|"]
        for s in S:
            k = sp[s]["skill"]
            L.append(f"| {s} | {s3(k['D_over_0'])} | {s3(k['C_over_0'])} | {s3(k['CD_over_0'])} | {s3(k['CD_over_C'])} | "
                     f"{s3(k['CD_over_D'])} | {f3(sp[s]['ratio_reduction_D_over_CD'])} |")
        L += ["", "| split | 0 over twin | C over twin | D over twin | CD over twin | 0 over causal GP | C over GP | D over GP | "
              "CD over GP | twin over GP |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in S:
            k = sp[s]["skill"]
            L.append(f"| {s} | " + " | ".join(s3(k[f"{m}_over_twin"]) for m in MODELS) + " | "
                     + " | ".join(s3(k[f"{m}_over_gp"]) for m in MODELS) + f" | {s3(k['twin_over_gp'])} |")
        L += ["", "### Quiet rows (class 6 of B.45)", "", "| split | CD over 0 | CD over C | CD over D | D over 0 | C over 0 | "
              "CD over twin | 0 over twin |", "|---|---|---|---|---|---|---|---|"]
        for s in S:
            q = sp[s]["quiet"]
            L.append(f"| {s} | {s3(q['CD_over_0'])} | {s3(q['CD_over_C'])} | {s3(q['CD_over_D'])} | {s3(q['D_over_0'])} | "
                     f"{s3(q['C_over_0'])} | {s3(q['CD_over_twin'])} | {s3(q['0_over_twin'])} |")
        L += ["", "### For comparison: the BES network of B.39 over the twin on the rows both score", "",
              "| split | rows both / harness rows | y agrees, stored twin error agrees (share within 0.5 eV) | BES network over twin | "
              "B.39 D2 (all harness rows) | 0 over twin | C over twin | D over twin | CD over twin |", "|---|---|---|---|---|---|---|---|---|"]
        for s in S:
            b = sp[s]["bes_network"]
            L.append(f"| {s} | {b['rows_both']:,} / {b['harness_rows']:,} | {b['y_agree_within_0.5_eV']:.4f}, "
                     f"{b['stored_twin_error_agrees_within_0.5_eV']:.4f} | {s3(b['bes_net_over_twin_stored'])} | "
                     f"{s3(b['b39_D2_all_harness_rows'])} | " + " | ".join(s3(b[f"{m}_over_twin_same_rows"]) for m in MODELS) + " |")
        L += ["", "RMSE (eV): " + "; ".join(f"split {s}: " + ", ".join(f"{k} {v:.1f}" for k, v in sp[s]["rmse_eV"].items()) for s in S) + ".", ""]
    E = res.get("expectations", [])
    if E:
        L += ["## Expectations of the registration (section 7)", "", "| expectation | observed | met |", "|---|---|---|"]
        for e in E:
            L.append(f"| {e['expectation']} | {e['observed']} | {'n/a' if e['met'] is None else ('yes' if e['met'] else 'no')} |")
        L += ["", f"{sum(1 for e in E if e['met'])} of {sum(1 for e in E if e['met'] is not None)} expectations with a range met.", ""]
    L += ["## Choices made where the registration left room", ""] + [f"- {c}" for c in CHOICES]
    L += ["", "## Deviations", "", res.get("deviations", "None from the registered definitions, rows, intervals or rules."), "",
          "## Limits", "", "The common mode is not a pure beam factor (attenuation depends on the density along the beam). The "
          "fast channels are single raw samples. Part B shows what information the components carry beyond the twin on about "
          "127 validation discharges per split, not what the network of B.39 uses. One initialisation of the twin. The test "
          "rows are those of every earlier readout.", "",
          f"Run time: part A {res['A']['runtime_s'] if A else 'n/a'} s; part B readout {res['B']['runtime_readout_s'] if B else 'n/a'} s "
          f"(cache build per split: {res.get('cache_runtime_s', {})})."]
    return "\n".join(L) + "\n"


def write(res):
    res["registration"] = REG
    res["expectations"] = expectations(res)
    res["paper_branch"] = paper_branch(res)
    res["choices"] = CHOICES
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"written {OUT_MD.name}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    res = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    if stage == "a":
        res["A"] = stage_a()
        write(res)
    elif stage == "b":
        seeds = [int(x) for x in sys.argv[2:]] or list(SEEDS)
        t = _time.time()
        b = stage_b(seeds)
        rt = res.get("cache_runtime_s", {})
        rt["+".join(map(str, seeds))] = round(_time.time() - t, 1)
        res["cache_runtime_s"] = rt
        if b is not None:
            res["B"] = b
        write(res)
    elif stage == "render":
        write(res)
    else:
        raise SystemExit("usage: run_b51.py a | b [SEED ...] | render")


if __name__ == "__main__":
    main()
