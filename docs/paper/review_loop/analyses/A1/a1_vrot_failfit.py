"""Review-loop round 1, analysis A1 (2026-11-13): V_rot fit-failure sensitivity, data-domain rule.

Question: does the registered B.27c V_rot mean-error skill over PCHIP (+0.313 [+0.072, +0.701], cut)
survive a pre-stated symmetric physical-range rule for V_rot fit failures, and what share of each
arm's squared error do the dropped rows carry?

Rule F(T), stated before computing: a scored V_rot row is dropped from EVERY arm when any of
the target value, the persistence anchor (last observed held-free V_rot before the target in the
same contiguous segment), or the PCHIP bracketing anchors (up to two observed neighbours on each
side in the segment) has |V_rot| > T.  T = 1000 km/s primary; 500 and 2000 km/s sensitivity.
Segments: consecutive time deltas < 0.5 s (baselines_interpolation.GAP_SECONDS), on the same
held-free (CES_DROP_STUCK_TARGETS=1) file arrays the harness scored. Anchors are re-derived from
the CSVs and checked against the stored squared errors (persistence and PCHIP must reproduce).

CPU only, existing artifacts only. Run from the repo root:
    py docs/paper/review_loop/analyses/A1/a1_vrot_failfit.py
ASCII-only prints (cp949 console).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
DATA = REPO / "data"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "ces_prediction"))
sys.path.insert(0, str(REPO / "ces_prediction" / "experiments" / "b27_kfold"))

import baselines_interpolation as BI  # noqa: E402
import analyze_b27c as A  # noqa: E402  (registered hier(), B = 20,000, seed 2728)

TS = (1000.0, 500.0, 2000.0)          # primary first
REFS = ("pchip", "persistence", "gp_causal", "gp")
ARMS = ("model",) + REFS
PRIMARY = tuple(range(2, 10))
SPLIT_SEEDS = (42, 1, 7, 123)
POPS = {"cut": "3000", "incl": "0"}


def build_dataset(pop):
    # Pin every data-treatment variable explicitly (the confirmed protocol's env).
    os.environ["CES_DROP_STUCK_TARGETS"] = "1"
    os.environ["CES_TI_SPIKE_CUT_EV"] = POPS[pop]
    os.environ["CES_PER_SHOT_NORM"] = "1"
    os.environ["CES_TEMPORAL_SUBSETS"] = "0"
    os.environ["CES_WINDOW_SIZE"] = "2"
    from dataset import KSTAR_CES_Dataset  # noqa: PLC0415
    ds = KSTAR_CES_Dataset(data_dir=DATA, window_size=2, temporal_subset_augmentation=False,
                           drop_stuck_targets=True, per_shot_input_norm=True,
                           ti_spike_cut_ev=float(POPS[pop]))
    if len(ds.valid_files) != 641:
        raise SystemExit(f"FATAL: expected 641 valid files, got {len(ds.valid_files)}")
    return ds


class Anchors:
    def __init__(self, ds):
        self.ds = ds
        self.tc = int(ds._column_slices["time"])
        self.vc = int(ds._column_slices["target"][1])
        self.cache = {}
        self.n_pers_mismatch = 0
        self.n_pchip_mismatch = 0
        self.n_checked = 0

    def get(self, shot, row):
        key = (int(shot), int(row))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        fa = self.ds.file_arrays[key[0]]
        times, vals, tt = BI.build_neighbor_set(fa, self.tc, self.vc, key[1])
        y = float(fa[key[1], self.vc])
        past, fut = times < tt, times > tt
        pt, pv = times[past], vals[past]
        ft, fv = times[fut], vals[fut]
        po = np.argsort(pt)[::-1][:2]
        fo = np.argsort(ft)[:2]
        pers = float(pv[po[0]]) if len(po) else np.nan
        anchors = np.concatenate([[y], pv[po], fv[fo]])
        pchip = BI.predict_pchip(times, vals, tt)
        hit = (y, pers, float(np.max(np.abs(anchors))), pchip)
        self.cache[key] = hit
        return hit


def load_npz(path):
    if not path.exists():
        raise SystemExit(f"FATAL: missing {path}")
    z = np.load(path)
    d = {c: z[f"CES_VT_{c}"] for c in ("shot", "row", "y_true") if f"CES_VT_{c}" in z.files}
    for a in ARMS:
        d[a] = z[f"CES_VT_se_{a}"].astype(np.float64)
    return d


def annotate(d, anc, tag):
    n = len(d["shot"])
    amax = np.empty(n)
    for i in range(n):
        y, pers, mx, pchip = anc.get(d["shot"][i], d["row"][i])
        amax[i] = mx
        yt = float(d["y_true"][i])
        if abs(y - yt) > 1e-3 * max(1.0, abs(y)):
            raise SystemExit(f"FATAL {tag}: target mismatch at shot {d['shot'][i]} row {d['row'][i]}")
        anc.n_checked += 1
        if not np.isclose((pers - yt) ** 2, d["persistence"][i], rtol=1e-3, atol=1e-2):
            anc.n_pers_mismatch += 1
        if not np.isclose((pchip - yt) ** 2, d["pchip"][i], rtol=1e-3, atol=1e-2):
            anc.n_pchip_mismatch += 1
    d["amax"] = amax
    return d


def annotate_by_match(d, anc, tag):
    """Canonical-split npz files predate the `row` key: recover each row by matching
    (shot, target value, stored PCHIP squared error) against every observed V_rot row of
    that shot. A row counts as matched only if all candidate rows agree on the anchor max."""
    n = len(d["shot"])
    amax = np.empty(n)
    per_shot = {}
    n_amb = 0
    for i in range(n):
        s = int(d["shot"][i])
        if s not in per_shot:
            fa = anc.ds.file_arrays[s]
            rows = np.flatnonzero(~np.isnan(fa[:, anc.vc]))
            info = np.array([anc.get(s, r_) for r_ in rows], dtype=np.float64)  # y, pers, amax, pchip
            per_shot[s] = info
        info = per_shot[s]
        yt = float(d["y_true"][i])
        se_p = (info[:, 3] - yt) ** 2
        cand = (np.abs(info[:, 0] - yt) <= 1e-3 * max(1.0, abs(yt))) & \
            np.isclose(se_p, d["pchip"][i], rtol=1e-3, atol=1e-2)
        if not cand.any():
            raise SystemExit(f"FATAL {tag}: no row match for shot {s} y {yt}")
        vals = info[cand, 2]
        if np.ptp(vals) > 0:
            n_amb += 1
        amax[i] = vals.max()           # conservative: drop if any candidate is flagged
        anc.n_checked += 1
    d["amax"] = amax
    d["n_ambiguous"] = n_amb
    return d


def shot_sums(shot, m, b):
    u, inv = np.unique(shot, return_inverse=True)
    return np.bincount(inv, weights=m, minlength=len(u)), np.bincount(inv, weights=b, minlength=len(u))


def cluster_ci(d, keep, ref, B=10_000, seed=12345):
    M, R = shot_sums(d["shot"][keep], d["model"][keep], d[ref][keep])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(M), size=(B, len(M)))
    lo, hi = np.percentile(1 - M[idx].sum(1) / R[idx].sum(1), [2.5, 97.5])
    return {"point": float(1 - M.sum() / R.sum()), "ci95": [float(lo), float(hi)],
            "excludes_0": bool(lo > 0 or hi < 0)}


def main():
    out = {"rule": "drop row in every arm if |V_rot| > T (km/s) for the target, the persistence anchor, "
                   "or up to 2 observed PCHIP neighbours on each side (held-free series, 0.5 s segments)",
           "T_primary": 1000.0, "T_sensitivity": [500.0, 2000.0],
           "two_level": "analyze_b27c.hier, repeats r2-r9, B=20000, seed 2728",
           "per_split": "shot-cluster bootstrap B=10000 seed 12345; cut = .b1_seqv2_s{s}_i{s}, incl = .b5i_seqv2_s{s}",
           "pops": {}}
    for pop in POPS:
        ds = build_dataset(pop)
        anc = Anchors(ds)
        P = {}
        for r in PRIMARY:
            parts = [annotate(load_npz(DATA / f".b27_seqv2_{pop}_r{r}_f{k}" / "comparison_errors_test.npz"),
                              anc, f"{pop} r{r} f{k}") for k in range(5)]
            P[r] = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
        canon = {}
        for s in SPLIT_SEEDS:
            sub = f".b1_seqv2_s{s}_i{s}" if pop == "cut" else f".b5i_seqv2_s{s}"
            d = load_npz(DATA / sub / "comparison_errors_test.npz")
            canon[s] = annotate(d, anc, f"{pop} s{s}") if "row" in d else annotate_by_match(d, anc, f"{pop} s{s}")
        res = {"anchor_check": {"rows_checked": anc.n_checked, "persistence_mismatch": anc.n_pers_mismatch,
                                "pchip_mismatch": anc.n_pchip_mismatch,
                                "canonical_rows_ambiguous": {f"s{s}": int(canon[s].get("n_ambiguous", 0))
                                                             for s in SPLIT_SEEDS}}}
        print(f"[{pop}] rows checked {anc.n_checked}, persistence mismatch {anc.n_pers_mismatch}, "
              f"pchip mismatch {anc.n_pchip_mismatch}")
        for T in (None,) + TS:
            tk = "none" if T is None else f"T{int(T)}"
            cell = {"two_level": {}, "per_split": {}, "dropped_rows_per_repeat": {}, "sse_share_dropped": {}}
            keeps = {r: (np.ones(len(P[r]["shot"]), bool) if T is None else P[r]["amax"] <= T)
                     for r in PRIMARY}
            for r in PRIMARY:
                k = keeps[r]
                cell["dropped_rows_per_repeat"][f"r{r}"] = {
                    "n_dropped": int((~k).sum()), "n_rows": int(len(k)),
                    "n_shots_touched": int(len(np.unique(P[r]["shot"][~k])))}
                cell["sse_share_dropped"][f"r{r}"] = {a: float(P[r][a][~k].sum() / P[r][a].sum()) for a in ARMS}
            for ref in REFS:
                sums = {r: shot_sums(P[r]["shot"][keeps[r]], P[r]["model"][keeps[r]], P[r][ref][keeps[r]])
                        for r in PRIMARY}
                h = A.hier(sums, PRIMARY, np.random.default_rng(A.SEED))
                h["per_repeat_pooled"] = {f"r{r}": float(1 - sums[r][0].sum() / sums[r][1].sum())
                                          for r in PRIMARY}
                cell["two_level"][ref] = h
                cell["per_split"][ref] = {}
                for s in SPLIT_SEEDS:
                    d = canon[s]
                    k = np.ones(len(d["shot"]), bool) if T is None else d["amax"] <= T
                    c = cluster_ci(d, k, ref)
                    c["n_dropped"] = int((~k).sum())
                    cell["per_split"][ref][f"s{s}"] = c
                sp = cell["per_split"][ref]
                print(f"[{pop}] {tk:6s} vs {ref:11s} two-level median {h['median']:+.3f} "
                      f"[{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] {h['verdict']:9s} | splits "
                      + " ".join(f"{sp[f's{s}']['point']:+.3f}[{sp[f's{s}']['ci95'][0]:+.2f},"
                                 f"{sp[f's{s}']['ci95'][1]:+.2f}]" for s in SPLIT_SEEDS))
            if T is not None:
                nd = [v["n_dropped"] for v in cell["dropped_rows_per_repeat"].values()]
                sh = {a: [cell["sse_share_dropped"][f"r{r}"][a] for r in PRIMARY] for a in ARMS}
                cell["sse_share_dropped_range"] = {a: [min(v), max(v)] for a, v in sh.items()}
                print(f"[{pop}] {tk}: dropped rows/repeat {min(nd)}-{max(nd)}; SSE share dropped "
                      + " ".join(f"{a} {min(v):.3f}-{max(v):.3f}" for a, v in sh.items()))
            res[tk] = cell
        out["pops"][pop] = res
    txt = json.dumps(out, indent=1)
    (OUT / "a1_vrot_failfit.json").write_text(txt, encoding="utf-8")
    (REPO / "docs" / "paper" / "vrot_failfit_sensitivity.json").write_text(txt, encoding="utf-8")
    print("wrote A1/a1_vrot_failfit.json and docs/paper/vrot_failfit_sensitivity.json")


if __name__ == "__main__":
    main()
