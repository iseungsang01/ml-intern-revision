"""Round-1 chair checks (2026-11-13), read-only on existing artifacts. Run from the repo root.

(1) V_rot fit-failure anchors in the registered B.27c two-level readout (hostile critique sig-1).
    Rows dropped per repeat (proxy rule on stored squared errors, threshold 1,000 km/s):
      anchor_only: PCHIP or persistence error > 1,000 km/s while |true V_rot| <= 1,000 km/s
      symmetric:   anchor_only rows plus rows whose true |V_rot| > 1,000 km/s
    Same bootstrap as experiments/b27_kfold/analyze_b27c.py (repeats r2..r9, B = 20,000, seed 2728).
    This is a proxy: anchors are inferred from the stored errors, not re-derived from the CSVs.
    Analysis A1 of round 1 replaces it with a data-domain rule.
(2) The stage-4 20-parameter surrogate (b28h) as a standalone estimator: shot-cluster skill vs each
    reference on the eight headline cells (B = 10,000, seed 12345, the protocol's convention).
ASCII-only prints (cp949 console).
"""
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
THR = 1000.0
PRIMARY = tuple(range(2, 10))


def load_b27(pop, r, target="CES_VT"):
    cols = {}
    for k in range(5):
        z = np.load(DATA / f".b27_seqv2_{pop}_r{r}_f{k}" / "comparison_errors_test.npz")
        for c in ("shot", "se_model", "se_pchip", "se_persistence", "se_gp_causal", "se_gp", "y_true"):
            cols.setdefault(c, []).append(z[f"{target}_{c}"])
    return {c: np.concatenate(v) for c, v in cols.items()}


def shot_sums(shot, m, b):
    u, inv = np.unique(shot, return_inverse=True)
    return np.bincount(inv, weights=m, minlength=len(u)), np.bincount(inv, weights=b, minlength=len(u))


def hier(S, B=20_000, seed=2728):
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.array(PRIMARY), size=B)
    out = np.empty(B)
    for r in PRIMARY:
        sel = np.flatnonzero(draws == r)
        m, b = S[r]
        idx = rng.integers(0, len(m), size=(len(sel), len(m)))
        out[sel] = 1.0 - m[idx].sum(1) / b[idx].sum(1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(np.median(out)), float(lo), float(hi)


def check_vrot():
    for pop in ("cut", "incl"):
        D = {r: load_b27(pop, r) for r in PRIMARY}
        for variant in ("none", "anchor_only", "symmetric"):
            for ref in ("pchip", "persistence", "gp_causal", "gp"):
                S, per, nd = {}, [], []
                for r in PRIMARY:
                    d = D[r]
                    y = np.abs(d["y_true"].astype(np.float64))
                    anchor_bad = ((d["se_pchip"] > THR ** 2) | (d["se_persistence"] > THR ** 2)) & (y <= THR)
                    keep = {"none": np.ones_like(anchor_bad), "anchor_only": ~anchor_bad,
                            "symmetric": ~(anchor_bad | (y > THR))}[variant]
                    S[r] = shot_sums(d["shot"][keep], d["se_model"][keep].astype(np.float64), d[f"se_{ref}"][keep])
                    per.append(1 - S[r][0].sum() / S[r][1].sum())
                    nd.append(int((~keep).sum()))
                med, lo, hi = hier(S)
                print(f"{pop:4s} {variant:11s} V_rot vs {ref:11s} median {med:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                      f" | per-repeat {min(per):+.3f}..{max(per):+.3f} | rows dropped/repeat {min(nd)}-{max(nd)}")
        for r in PRIMARY:
            d = D[r]
            y = np.abs(d["y_true"].astype(np.float64))
            anchor_bad = ((d["se_pchip"] > THR ** 2) | (d["se_persistence"] > THR ** 2)) & (y <= THR)
            print(f"{pop} r{r}: n rows {len(y)}, anchor-failure rows {int(anchor_bad.sum())} carry "
                  f"{d['se_pchip'][anchor_bad].sum() / d['se_pchip'].sum():.2f} of PCHIP SSE; "
                  f"target > 1000 km/s rows {int((y > THR).sum())} carry "
                  f"{d['se_model'][y > THR].sum() / d['se_model'].sum():.2f} of the model SSE")


def check_surrogate():
    for pop in ("cut", "incl"):
        for s in (42, 1, 7, 123):
            z = np.load(DATA / f".b28_b28h_{pop}_s{s}" / "comparison_errors_test.npz")
            m, shot = z["CES_TI_se_model"].astype(np.float64), z["CES_TI_shot"]
            cells = []
            for ref in ("gp_causal", "pchip", "persistence", "gp"):
                M, Rf = shot_sums(shot, m, z[f"CES_TI_se_{ref}"])
                rng = np.random.default_rng(12345)
                idx = rng.integers(0, len(M), size=(10_000, len(M)))
                lo, hi = np.percentile(1 - M[idx].sum(1) / Rf[idx].sum(1), [2.5, 97.5])
                cells.append(f"{ref} {1 - M.sum() / Rf.sum():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"surrogate {pop} s{s}: " + " | ".join(cells))


if __name__ == "__main__":
    check_vrot()
    check_surrogate()
