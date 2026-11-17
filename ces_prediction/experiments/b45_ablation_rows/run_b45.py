"""B.45 (PREREGISTRATION_B45.md): in which rows each fast-channel group carries the increment.

Row-level squared errors of the B.39 ablation networks (full causal variant, ECEI-only, BES-only,
no-Mirnov; initialisation = split seed) and the history-only twin, read inside six exclusive classes of
the scored test rows (after an absent frame, low beam emission, upward / downward BES departure, after a
departure, quiet) plus a seventh group (departure not defined).

CPU only; nothing is trained and no network is run.  Run from the repo root:
    py ces_prediction/experiments/b45_ablation_rows/run_b45.py
Output: docs/paper/b45_ablation_rows.json / .md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b41_events"))
sys.path.insert(0, str(HERE.parent / "b32_two_sources"))
import run_b41 as B41  # noqa: E402  (adds b30_thinned and b31_bursts to sys.path)
from posthoc_b32 import channel_departures  # noqa: E402

R = B41.R
REPO = R.REPO
DATA = R.DATA
OUT_JSON = REPO / "docs" / "paper" / "b45_ablation_rows.json"
OUT_MD = REPO / "docs" / "paper" / "b45_ablation_rows.md"
B39_JSON = REPO / "docs" / "paper" / "b39_primary.json"
REG = "ces_prediction/experiments/PREREGISTRATION_B45.md"
SEEDS = (42, 1, 7, 123)
SEED, N_BOOT = 20261118, 2000
T = "CES_TI"
SEG_BREAK_S = 0.5             # segments end at a step of 0.5 s or more
STEP_BREAK_DATASET = 0.5      # dataset.py: contiguous blocks are steps < 0.5 s
DEP_SD = 3.0
AFTER_DEP_ROWS = 3
CHECK_ABS_EV = 0.5            # |sqrt(se_mine) - sqrt(se_stored)| tolerance for check (c) (B.30 amendment 1)
CHECK_SHARE = 0.99
N_CHECK_DISCHARGES = 5
NETS = {"full": ".b39_v2_cut_s{s}_i{s}", "ecei": ".b39_v2e_cut_s{s}_i{s}", "bes": ".b39_v2b_cut_s{s}_i{s}",
        "nomirnov": ".b39_v2n_cut_s{s}_i{s}", "twin": ".b29a1_v2h_cut_s{s}"}
NLAB = {"full": "full", "ecei": "ECEI-only", "bes": "BES-only", "nomirnov": "no-Mirnov", "twin": "twin",
        "gp_causal": "causal GP", "persistence": "hold"}
B39_KEYS = {"ecei": "D1_ecei_over_twin", "bes": "D2_bes_over_twin", "nomirnov": "nomirnov_over_twin",
            "full": "full_over_twin_same_initialisation"}
GROUPS = ("1", "2", "3", "4", "5", "6", "7", "out12", "in12", "all")
GLAB = {"1": "1 after an absent frame", "2": "2 low beam emission", "3": "3 upward departure",
        "4": "4 downward departure", "5": "5 after a departure", "6": "6 quiet",
        "7": "7 departure not defined", "out12": "classes 3 to 6 (outside 1 and 2)",
        "in12": "classes 1 and 2", "all": "all scored rows"}
SKILLS = [("full", "twin"), ("bes", "twin"), ("ecei", "twin"), ("nomirnov", "twin"),
          ("full", "bes"), ("full", "ecei"), ("bes", "gp_causal")]
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def boot_w(rng, n):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)


# ------------------------------------------------------------------------------ loading and checks
def load_run(path):
    z = np.load(path / "comparison_errors_test.npz")
    d = {"shot": z[f"{T}_shot"].astype(np.int64), "row": z[f"{T}_row"].astype(np.int64),
         "model": z[f"{T}_se_model"].astype(np.float64), "y": z[f"{T}_y_true"].astype(np.float64)}
    for r in ("gp_causal", "persistence"):
        d[r] = z[f"{T}_se_{r}"].astype(np.float64)
    return d


def check_a(runs, s):
    ref = runs["full"]
    out = {}
    for k, d in runs.items():
        same = len(d["shot"]) == len(ref["shot"]) and np.array_equal(d["shot"], ref["shot"]) \
            and np.array_equal(d["row"], ref["row"])
        refs = same and all(np.allclose(d[r], ref[r], rtol=1e-6, atol=1e-6) for r in ("gp_causal", "persistence")) \
            and np.allclose(d["y"], ref["y"], rtol=1e-6, atol=1e-6)
        out[k] = {"same_rows": bool(same), "reference_errors_agree": bool(refs)}
        if not (same and refs):
            raise SystemExit(f"FATAL: check (a) split {s}: run {k} not scored on the same rows / references differ")
    return out


def check_b(runs, s, b39):
    out = {}
    tw = runs["twin"]["model"].sum()
    for k, key in B39_KEYS.items():
        mine = float(1 - runs[k]["model"].sum() / tw)
        theirs = float(b39["A"][str(s)][key][0])
        ok = round(mine, 3) == round(theirs, 3)
        out[k] = {"mine": mine, "b39": theirs, "equal_to_3_decimals": bool(ok)}
        if not ok:
            raise SystemExit(f"FATAL: check (b) split {s} {k}: {mine:.5f} vs B.39 {theirs:.5f}")
    return out


def compact_index(arr):
    """Rows of the grid that dataset.py keeps (at least one of the two targets finite)."""
    return np.flatnonzero(np.isfinite(arr[:, 1:3]).any(1))


def hold_compacted(arr, gi, rows):
    """W = 2 hold of dataset.py: the T_i of the previous kept row, if in the same block and finite."""
    t, ti = arr[gi, 0].astype(np.float64), arr[gi, 1].astype(np.float64)
    h = np.full(len(rows), np.nan)
    for j, r in enumerate(rows):
        if 1 <= r < len(gi) and (t[r] - t[r - 1]) < STEP_BREAK_DATASET:
            h[j] = ti[r - 1]
    return h, ti[np.clip(rows, 0, len(gi) - 1)]


def hold_grid(arr, rows):
    """Alternative reading: row = grid row, hold = last finite T_i before it in the same segment."""
    t, ti = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
    seg = np.concatenate(([0], np.cumsum(np.diff(t) >= SEG_BREAK_S)))
    h = np.full(len(rows), np.nan)
    y = np.full(len(rows), np.nan)
    for j, r in enumerate(rows):
        if r >= len(t):
            continue
        y[j] = ti[r]
        k = r - 1
        while k >= 0 and seg[k] == seg[r]:
            if np.isfinite(ti[k]):
                h[j] = ti[k]
                break
            k -= 1
    return h, y


def match(h, y, se_ref, y_ref):
    ok = np.isfinite(h) & np.isfinite(y)
    se = np.where(ok, (h - y) ** 2, np.nan)
    good = ok & (np.abs(np.sqrt(np.where(ok, se, 0.0)) - np.sqrt(se_ref)) <= CHECK_ABS_EV) \
        & (np.abs(np.where(ok, y, 0.0) - y_ref) <= CHECK_ABS_EV)
    return good


def check_c(C, valid, ref, s, rng):
    shots = np.unique(ref["shot"])
    pick = np.sort(rng.choice(shots, size=min(N_CHECK_DISCHARGES, len(shots)), replace=False))
    res = {"discharges": [valid[int(i)] for i in pick], "compacted": {}, "grid": {}}
    totc = totg = n = 0
    for i in pick:
        sel = ref["shot"] == i
        arr = C.grid[valid[int(i)]]
        rows = ref["row"][sel]
        gi = compact_index(arr)
        h, y = hold_compacted(arr, gi, rows)
        gc = match(h, y, ref["persistence"][sel], ref["y"][sel])
        h2, y2 = hold_grid(arr, rows)
        gg = match(h2, y2, ref["persistence"][sel], ref["y"][sel])
        res["compacted"][valid[int(i)]] = [int(gc.sum()), int(sel.sum())]
        res["grid"][valid[int(i)]] = [int(gg.sum()), int(sel.sum())]
        totc += int(gc.sum())
        totg += int(gg.sum())
        n += int(sel.sum())
    res["rows"] = n
    res["share_compacted"] = totc / n
    res["share_grid"] = totg / n
    res["passed"] = bool(res["share_compacted"] >= CHECK_SHARE)
    # after the fact: every test discharge of the split under the convention found
    tc = 0
    for i in shots:
        sel = ref["shot"] == i
        arr = C.grid[valid[int(i)]]
        gi = compact_index(arr)
        h, y = hold_compacted(arr, gi, ref["row"][sel])
        tc += int(match(h, y, ref["persistence"][sel], ref["y"][sel]).sum())
    res["after_fact_all_discharges_share_compacted"] = tc / len(ref["shot"])
    if not res["passed"]:
        raise SystemExit(f"FATAL: check (c) split {s}: compacted {res['share_compacted']:.4f}, "
                         f"grid {res['share_grid']:.4f}; no convention reproduces 99 % of the rows")
    return res


# ------------------------------------------------------------------------------ classes
class Classes:
    def __init__(self, C):
        self.C = C
        self.cache = {}
        self.csv_check = {"files": 0, "row_mismatch": 0, "time_maxdiff_s": 0.0}

    def of_file(self, name):
        """Class (1..7) of every grid row of the file (first and last row: 0, never scored)."""
        if name in self.cache:
            return self.cache[name]
        C = self.C
        i = C.names.index(name)
        arr = C.grid[name]
        n = len(arr)
        tc, d, after = channel_departures(DATA / name)          # index = grid row - 1
        self.csv_check["files"] += 1
        if len(tc) != n - 2:
            self.csv_check["row_mismatch"] += 1
            raise SystemExit(f"FATAL: {name}: CSV rows ({len(tc)}) do not match the grid rows minus two ({n - 2})")
        self.csv_check["time_maxdiff_s"] = max(self.csv_check["time_maxdiff_s"],
                                              float(np.max(np.abs(tc - arr[1:-1, 0].astype(np.float64)))))
        seg = np.concatenate(([0], np.cumsum(np.diff(tc) >= SEG_BREAK_S)))
        defined = np.isfinite(d).all(1)
        up = defined & (np.nan_to_num(d, nan=0.0) > DEP_SD).any(1)
        dn = defined & (np.nan_to_num(d, nan=0.0) < -DEP_SD).any(1)
        dep = up | dn
        m = len(tc)
        after_dep = np.zeros(m, bool)
        for k in range(1, AFTER_DEP_ROWS + 1):
            after_dep[k:] |= dep[:-k] & (seg[k:] == seg[:-k])
        low = C.after_low_bes[i][1:-1]                          # Ctx.after_low_bes on the grid, verbatim
        c = np.full(m, 6, np.int8)
        c[after_dep] = 5
        c[dn] = 4
        c[up] = 3
        c[low] = 2
        c[after > 0] = 1
        # alternative reading kept for the report: class of the undefined rows if they were not group 7
        alt = c.copy()
        c[~defined] = 7
        cls = np.zeros(n, np.int8)
        cls[1:-1] = c
        cls_alt = np.zeros(n, np.int8)
        cls_alt[1:-1] = alt
        self.cache[name] = (cls, cls_alt)
        return self.cache[name]


# ------------------------------------------------------------------------------ one split
def run_split(s, C, valid, Cl, b39, rng_check):
    runs = {k: load_run(DATA / p.format(s=s)) for k, p in NETS.items()}
    res = {"check_a": check_a(runs, s), "check_b": check_b(runs, s, b39)}
    ref = runs["full"]
    res["check_c"] = check_c(C, valid, ref, s, rng_check)
    log(f"split {s}: checks passed (c: {res['check_c']['share_compacted']:.4f} compacted, "
        f"{res['check_c']['share_grid']:.4f} grid)")
    shot, row = ref["shot"], ref["row"]
    N = len(shot)
    cls = np.zeros(N, np.int8)
    cls_alt = np.zeros(N, np.int8)
    for i in np.unique(shot):
        sel = np.flatnonzero(shot == i)
        name = valid[int(i)]
        gi = compact_index(C.grid[name])
        g = gi[row[sel]]
        a, b = Cl.of_file(name)
        cls[sel], cls_alt[sel] = a[g], b[g]
    if (cls == 0).any():
        raise SystemExit(f"FATAL: split {s}: a scored row maps to the first or last row of its file")
    u, f = np.unique(shot, return_inverse=True)
    n = len(u)
    W = np.random.default_rng(SEED + s).multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)
    masks = {str(k): cls == k for k in range(1, 8)}
    masks["out12"] = np.isin(cls, (3, 4, 5, 6))
    masks["in12"] = np.isin(cls, (1, 2))
    masks["all"] = np.ones(N, bool)
    se = {k: runs[k]["model"] for k in NETS}
    se["gp_causal"], se["persistence"] = ref["gp_causal"], ref["persistence"]
    fs = lambda x: np.bincount(f, x, n)                                         # noqa: E731
    cnt_all = fs(np.ones(N))
    red_all = {k: fs(se["twin"] - se[k]) for k in ("bes", "full")}
    out = {"rows": int(N), "discharges": int(n), "groups": {}}
    for gname in GROUPS:
        m = masks[gname].astype(np.float64)
        cm = fs(m)
        o = {"rows": int(m.sum()), "discharges_with_rows": int((cm > 0).sum()),
             "row_share": [float(m.sum() / N), *ci((W @ cm) / (W @ cnt_all))]}
        if m.sum() > 0:
            sums = {k: fs(se[k] * m) for k in se}
            o["skill"] = {}
            for a, b in SKILLS:
                with np.errstate(invalid="ignore", divide="ignore"):
                    bs = 1 - (W @ sums[a]) / (W @ sums[b])
                o["skill"][f"{a}_over_{b}"] = [float(1 - sums[a].sum() / sums[b].sum()), *ci(bs)]
            o["share_of_reduction"] = {}
            for k in ("bes", "full"):
                dm = fs((se["twin"] - se[k]) * m)
                o["share_of_reduction"][k] = [float(dm.sum() / red_all[k].sum()), *ci((W @ dm) / (W @ red_all[k]))]
            with np.errstate(invalid="ignore", divide="ignore"):
                bh = np.sqrt((W @ sums["persistence"]) / (W @ cm))
            o["rmse_hold_eV"] = [float(np.sqrt(sums["persistence"].sum() / m.sum())), *ci(bh)]
            o["rmse_eV"] = {k: float(np.sqrt(sums[k].sum() / m.sum())) for k in se}
        out["groups"][gname] = o
    g = out["groups"]
    out["R1"] = g["out12"]["skill"]["bes_over_twin"]
    out["R2"] = g["6"]["skill"]["bes_over_twin"]
    out["R3"] = g["1"]["skill"]["ecei_over_twin"]
    for k in ("R1", "R2", "R3"):
        out[k + "_holds"] = bool(out[k][1] > 0)
    out["reduction_total_eV2"] = {k: float(red_all[k].sum()) for k in red_all}
    # alternative reading (after the fact): undefined-departure rows keep their class 1 / 2
    und = cls == 7
    out["group7_by_other_reading"] = {str(k): int((und & (cls_alt == k)).sum()) for k in range(1, 7)}
    m1 = ((cls == 1) | (und & (cls_alt == 1))).astype(np.float64)
    if (und & (cls_alt == 1)).any():
        a_, b_ = fs(se["ecei"] * m1), fs(se["twin"] * m1)
        out["after_fact_R3_with_undefined_rows_in_class1"] = [float(1 - a_.sum() / b_.sum()),
                                                              *ci(1 - (W @ a_) / (W @ b_))]
    log(f"split {s}: R1 {out['R1']} R2 {out['R2']} R3 {out['R3']}")
    res.update(out)
    return res


# ------------------------------------------------------------------------------ expectations
def expectations(res):
    S = [str(s) for s in SEEDS]
    sp = res["splits"]
    E = []

    def share_rng(gn, lo, hi, text):
        v = [sp[s]["groups"][gn]["row_share"][0] for s in S]
        E.append((text, ", ".join(f"{x:.3f}" for x in v), all(lo <= x <= hi for x in v)))

    share_rng("1", 0.03, 0.08, "share of rows, class 1: 0.03 to 0.08")
    share_rng("2", 0.0, 0.04, "share of rows, class 2: below 0.04")
    share_rng("3", 0.01, 0.04, "share of rows, class 3: 0.01 to 0.04")
    share_rng("4", 0.01, 0.04, "share of rows, class 4: 0.01 to 0.04")
    share_rng("5", 0.03, 0.10, "share of rows, class 5: 0.03 to 0.10")
    share_rng("6", 0.75, 0.90, "share of rows, quiet: 0.75 to 0.90")
    for k, lo, hi, cnts, text in (("R1", 0.02, 0.09, (3, 4), "R1: +0.02 to +0.09, holds on three or four splits"),
                                  ("R2", 0.00, 0.05, (2, 3), "R2: 0.00 to +0.05, holds on two or three splits"),
                                  ("R3", 0.03, 0.20, (2, 3, 4), "R3: +0.03 to +0.20, holds on two to four splits")):
        v = [sp[s][k][0] for s in S]
        nh = res["verdicts"][k]["splits"]
        E.append((text, f"points {', '.join(f'{x:+.3f}' for x in v)}; holds on {nh} of 4",
                  all(lo <= x <= hi for x in v) and nh in cnts))
    v = [sp[s]["groups"]["in12"]["share_of_reduction"]["bes"][0] for s in S]
    E.append(("share of the squared-error reduction of the BES-only network in classes 1 and 2 together: 0.25 to 0.60",
              ", ".join(f"{x:.3f}" for x in v), all(0.25 <= x <= 0.60 for x in v)))
    return [{"expectation": a, "observed": b, "met": bool(c)} for a, b, c in E]


# ------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def f3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


CHOICES = [
    "Rows: every row of comparison_errors_test.npz (T_i), i.e. the scored test rows of the B.39 harness; no further "
    "population filter.",
    "Meaning of CES_TI_row (check c): the index into the per-file array of dataset.py, which keeps only the rows with "
    "at least one of CES_TI / CES_VT finite after the 3 keV cut and the removal of held values (the CSV after dropna on "
    "the inputs, then spike cut, then held removal, then rows without any target dropped). It is mapped to the grid row "
    "of seq_data.load_grid_files(DATA, True, 3000.0) through the finite-target rows of that grid; the grid row minus one "
    "is the index of posthoc_b32.channel_departures. Hold of the harness (window W = 2) is the T_i of the previous kept "
    "row in the same block (step < 0.5 s).",
    "Check (c): five test discharges per split drawn with numpy default_rng(20261118) (one generator for the four splits, "
    "in the order 42, 1, 7, 123); a row matches when the absolute error of hold and the measured value both agree with "
    "the stored ones within 0.5 eV (the tolerance of B.30, amendment 1). The other reading (row = grid row, hold = last "
    "finite T_i before it in the segment) is reported next to it.",
    "Absent frame: a step of more than 15 and less than 45 ms (a 20 to 40 ms step on the 10 ms grid) inside a segment "
    "(steps of 0.5 s or more end a segment); class 1 = the `after` counter of channel_departures is 1 to 10.",
    "Class 2: Ctx.after_low_bes of b41_events/run_b41.py, verbatim (BES level = mean of the nine raw channels of the grid "
    "row; rows with a stamp from t_j to t_j + 30 ms of a row j whose level is below half the median level of the whole "
    "grid file).",
    "Classes 3 and 4: departure of channel_departures (per-discharge z-score of each BES channel, minus the mean of the "
    "ten rows before in the same segment); class 3 when any channel is above +3, class 4 when any is below -3 (and none "
    "above +3).",
    "Class 5: a row with any channel beyond +3 or -3 (whatever its own class) among the three grid rows before, in the "
    "same segment.",
    "Group 7: every scored row whose departure is not defined (fewer than ten rows of the segment before) is put in "
    "group 7 before classes 1 to 6 are assigned, also when it is one of the ten rows after an absent frame or a low-beam "
    "row (the literal reading of 'enter no statement'). The number of such rows that the other reading would have put in "
    "classes 1 and 2 is reported.",
    "Statement sets: R1 = classes 3 to 6 (group 7 excluded); R2 = class 6; R3 = class 1.",
    "Intervals: 2,000 multinomial resamples of the test discharges of the split (numpy default_rng(20261118 + split "
    "seed)); the same resamples for every quantity of the split. A resample in which a denominator is zero is dropped "
    "from the interval of that quantity.",
    "Share of the rows and of the reduction: over all scored rows of the split, group 7 included.",
    "Expectations (section 7): a range is met when the point value of every one of the four splits lies inside it; "
    "for R1 to R3 the count of splits must also be in the expected set.",
]


def render(res):
    S = [str(s) for s in SEEDS]
    sp = res["splits"]
    hs = lambda h: "holds" if h else "does not hold"                                       # noqa: E731
    v = res["verdicts"]
    L = [f"# B.45 -- in which rows each fast-channel group carries the increment (registered: {REG})", "",
         "CPU, nothing trained, no network run. T_i, cut population, the four file-level splits, test rows of the B.39 "
         "harness. Networks (epoch cap 80, initialisation = split seed): full causal variant, ECEI-only, BES-only and "
         "no-Mirnov variants (B.39 arm A) and the history-only twin (B.29). Skill of A over R = 1 - SSE_A / SSE_R on the "
         "same rows. Classes (exclusive, first that applies): 1 after an absent frame (one of the ten rows after a "
         "20 to 40 ms step inside a segment); 2 low beam emission (within 30 ms after a row whose BES level is below half "
         "the discharge median); 3 upward departure (a BES channel more than 3 SD above its mean over the ten rows "
         "before); 4 downward departure (more than 3 SD below); 5 after a departure (a row with a channel beyond +-3 SD in "
         "the three rows before); 6 quiet. Group 7: departure not defined (fewer than ten rows of the segment before), in "
         "no statement. Share of the reduction in a class = sum over its rows of (twin error squared minus network error "
         "squared) over the same sum over all rows. Intervals: 95 %, 2,000 resamples of the test discharges. A statement "
         "holds on a split when its interval is above zero, and holds when that is so on at least three of four splits.",
         "", "## Checks before the readout", "",
         "(a) The five runs of each split are scored on the same rows with the same reference errors (causal GP, hold, "
         "measured value): " + ", ".join(f"split {s} {'yes' if all(x['same_rows'] and x['reference_errors_agree'] for x in sp[s]['check_a'].values()) else 'NO'}" for s in S) + ".", "",
         "(b) Skill over the twin on all rows against docs/paper/b39_primary.md (to three decimals):", "",
         "| split | full (mine / B.39) | BES-only | ECEI-only | no-Mirnov |", "|---|---|---|---|---|"]
    for s in S:
        b = sp[s]["check_b"]
        L.append(f"| {s} | " + " | ".join(f"{b[k]['mine']:+.4f} / {b[k]['b39']:+.4f} ({'equal' if b[k]['equal_to_3_decimals'] else 'DIFFERENT'})"
                                          for k in ("full", "bes", "ecei", "nomirnov")) + " |")
    L += ["", "(c) Meaning of CES_TI_row: hold recomputed from the 10 ms file of five test discharges per split. "
          "Convention found: **the row index of the per-file array of dataset.py (rows with at least one of CES_TI / CES_VT "
          "finite after the 3 keV cut and the removal of held values); hold = T_i of the previous kept row in the same "
          "block (window W = 2)**.", "",
          "| split | discharges | rows | matched, that convention | matched, row = grid row (other reading) | after the fact: all test discharges, that convention |",
          "|---|---|---|---|---|---|"]
    for s in S:
        c = sp[s]["check_c"]
        L.append(f"| {s} | {', '.join(x.replace('.csv', '') for x in c['discharges'])} | {c['rows']:,} | "
                 f"{c['share_compacted']:.4f} | {c['share_grid']:.4f} | {c['after_fact_all_discharges_share_compacted']:.4f} |")
    cc = res["csv_check"]
    L += ["", f"BES read from the CSV by channel_departures against the grid: {cc['files']} files, row-count mismatches "
          f"{cc['row_mismatch']}, largest time difference {cc['time_maxdiff_s']:.2e} s.", "",
          "## Registered statements", ""]
    for k, text in (("R1", "classes 3 to 6 (no beam interruption behind them): BES-only over the twin"),
                    ("R2", "class 6, quiet: BES-only over the twin"),
                    ("R3", "class 1, after an absent frame: ECEI-only over the twin")):
        L.append(f"- **{k}** ({text}, interval above zero): {v[k]['splits']} of 4 splits -> **{hs(v[k]['holds'])}**. "
                 + "; ".join(f"split {s} {s3(sp[s][k])}" for s in S) + ".")
    L += ["", "## Reported, per class and split", "",
          "### Rows and skills over the twin", "",
          "| split | group | rows | share of rows | full over twin | BES-only over twin | ECEI-only over twin | no-Mirnov over twin |",
          "|---|---|---|---|---|---|---|---|"]
    for s in S:
        for gn in GROUPS:
            g = sp[s]["groups"][gn]
            if "skill" not in g:
                L.append(f"| {s} | {GLAB[gn]} | 0 | {f3(g['row_share'])} | | | | |")
                continue
            k = g["skill"]
            L.append(f"| {s} | {GLAB[gn]} | {g['rows']:,} | {f3(g['row_share'])} | {s3(k['full_over_twin'])} | "
                     f"{s3(k['bes_over_twin'])} | {s3(k['ecei_over_twin'])} | {s3(k['nomirnov_over_twin'])} |")
    L += ["", "### Networks against each other, the causal GP, and hold", "",
          "| split | group | full over BES-only | full over ECEI-only | BES-only over causal GP | RMSE of hold (eV) |",
          "|---|---|---|---|---|---|"]
    for s in S:
        for gn in GROUPS:
            g = sp[s]["groups"][gn]
            if "skill" not in g:
                continue
            k = g["skill"]
            h = g["rmse_hold_eV"]
            L.append(f"| {s} | {GLAB[gn]} | {s3(k['full_over_bes'])} | {s3(k['full_over_ecei'])} | "
                     f"{s3(k['bes_over_gp_causal'])} | {h[0]:.1f} [{h[1]:.1f}, {h[2]:.1f}] |")
    L += ["", "### Share of the total squared-error reduction against the twin", "",
          "A share above 1 or below 0 means the other rows have a net increase. Total reduction (sum over all rows of "
          "twin error squared minus network error squared, eV^2): "
          + "; ".join(f"split {s} BES-only {sp[s]['reduction_total_eV2']['bes']:.3e}, full {sp[s]['reduction_total_eV2']['full']:.3e}"
                      for s in S) + ".", "",
          "| split | group | share of rows | BES-only | full |", "|---|---|---|---|---|"]
    for s in S:
        for gn in GROUPS:
            g = sp[s]["groups"][gn]
            if gn == "all" or "share_of_reduction" not in g:
                continue
            r = g["share_of_reduction"]
            L.append(f"| {s} | {GLAB[gn]} | {g['row_share'][0]:.3f} | {s3(r['bes'])} | {s3(r['full'])} |")
    L += ["", "### RMSE of every estimator per group (eV, point values)", "",
          "| split | group | hold | causal GP | twin | full | BES-only | ECEI-only | no-Mirnov |", "|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        for gn in GROUPS:
            g = sp[s]["groups"][gn]
            if "rmse_eV" not in g:
                continue
            r = g["rmse_eV"]
            L.append(f"| {s} | {GLAB[gn]} | " + " | ".join(f"{r[k]:.1f}" for k in
                                                         ("persistence", "gp_causal", "twin", "full", "bes", "ecei", "nomirnov")) + " |")
    L += ["", "## Comparison with the expectations (section 7)", "", "| expectation | observed (splits 42, 1, 7, 123) | met |",
          "|---|---|---|"]
    for e in res["expectations"]:
        L.append(f"| {e['expectation']} | {e['observed']} | {'yes' if e['met'] else 'no'} |")
    nm = sum(e["met"] for e in res["expectations"])
    L += ["", f"{nm} of {len(res['expectations'])} expectations met.", "",
          "## Choices made where the registration left room", ""]
    L += [f"- {c}" for c in CHOICES]
    L += ["", "Group 7 rows that the other reading would have put in classes 1 to 6: "
          + "; ".join(f"split {s}: " + ", ".join(f"class {k} {x}" for k, x in sp[s]["group7_by_other_reading"].items())
                      for s in S) + "."]
    af = [s for s in S if "after_fact_R3_with_undefined_rows_in_class1" in sp[s]]
    if af:
        L += ["", "After the fact (not registered): R3 with those rows kept in class 1: "
              + "; ".join(f"split {s} {s3(sp[s]['after_fact_R3_with_undefined_rows_in_class1'])}" for s in af) + "."]
    L += ["", "## Deviations", "", "None from the registered definitions, thresholds, populations or statements.", "",
          "## Limits", "", "One initialisation per network. The classes are defined from BES, which the ECEI-only network "
          "does not read. Classes 1 to 4 hold few rows, so their intervals are wide. The test rows of these splits were read "
          "before by D1, D2 and earlier registered readouts.", "", f"Run time: {res['runtime_s']:.0f} s."]
    return "\n".join(L) + "\n"


def main():
    C = B41.Ctx()                       # pins the CES_* env, loads the cut grid, computes after_low_bes
    log(f"grid loaded: {len(C.names)} files")
    valid = [nm for nm in C.names if np.isfinite(C.grid[nm][:, 1:3]).any()]
    log(f"files with a target (dataset.py valid_files): {len(valid)}")
    b39 = json.loads(B39_JSON.read_text(encoding="utf-8"))
    Cl = Classes(C)
    rng_check = np.random.default_rng(SEED)
    res = {"registration": REG, "seed": SEED, "n_boot": N_BOOT, "valid_files": len(valid), "splits": {}}
    for s in SEEDS:
        res["splits"][str(s)] = run_split(s, C, valid, Cl, b39, rng_check)
    res["csv_check"] = Cl.csv_check
    S = [str(s) for s in SEEDS]
    res["verdicts"] = {}
    for k in ("R1", "R2", "R3"):
        nh = sum(res["splits"][s][k + "_holds"] for s in S)
        res["verdicts"][k] = {"splits": int(nh), "holds": bool(nh >= 3)}
    res["expectations"] = expectations(res)
    res["choices"] = CHOICES
    res["runtime_s"] = time.time() - T0
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"verdicts {res['verdicts']}; written {OUT_MD.name}")


if __name__ == "__main__":
    main()
