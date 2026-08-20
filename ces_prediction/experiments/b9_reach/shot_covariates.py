"""Which discharges does the model win on? — the question §8am left open for `V_rot`.

§8am retired "`V_rot` is underpowered at ~96 discharges". The replacement is sharper and
less comfortable: the model beats the causal GP on **48%** of discharges, and deleting ten
of ~197 flips the pooled sign. So the open question is no longer *how many* discharges but
*which* — and until that has an answer, "not shot-general" is a description without a
mechanism, which is the shape of conclusion §8j forbids.

This asks the frozen artifacts directly. Every run already stored per-row squared errors,
the discharge id, the gap length, the peak flag and the truth, so a per-discharge table can
be built with no training at all:

    campaign position   the shot number itself (2022 campaign, monotone in time)
    rows                scored rows in that discharge
    gap                 mean and max Δt to the previous observation
    peak fraction       share of rows flagged as high-variability
    target level/spread mean and sd of the truth in that discharge
    baseline difficulty  RMSE of the causal GP there -- how hard the discharge is at all

`CES_TI` is carried as the control. Its win is shot-general (69.5%), so a covariate that
explains `V_rot` and not `T_i` is about the target; one that explains both is about the
discharge.

**Exploratory, and labelled as such.** No decision rule was pre-registered for this, so
nothing here promotes or demotes a claim. Significance is by permutation (10,000 shuffles
of the response, which respects the covariates' own distribution) and reported both raw and
Bonferroni-adjusted across the covariates tested; the adjusted column is the one to read.

Usage (repo root):
  py ces_prediction/experiments/b9_reach/shot_covariates.py
  py ces_prediction/experiments/b9_reach/shot_covariates.py --arm v2r63
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data"
SEEDS = (42, 1, 7, 123)
TARGETS = ("CES_TI", "CES_VT")
N_PERM = 10000
PERM_SEED = 20260821


def per_shot_table(arm, target):
    """One row per physical discharge, pooled over the four splits."""
    cols = {k: [] for k in ("shot", "dt", "peak", "y", "sm", "sb")}
    for seed in SEEDS:
        path = DATA / f".b9_{arm}_s{seed}" / "comparison_errors_test.npz"
        if not path.exists():
            return None
        d = np.load(path, allow_pickle=True)
        sm, sb = d[f"{target}_se_model"], d[f"{target}_se_gp_causal"]
        ok = np.isfinite(sm) & np.isfinite(sb)
        cols["shot"].append(d[f"{target}_shot"][ok])
        cols["dt"].append(d[f"{target}_dt_ms"][ok])
        cols["peak"].append(d[f"{target}_is_peak"][ok].astype(float))
        cols["y"].append(d[f"{target}_y_true"][ok])
        cols["sm"].append(sm[ok])
        cols["sb"].append(sb[ok])
    c = {k: np.concatenate(v) for k, v in cols.items()}

    shots = np.unique(c["shot"])
    rows = []
    for s in shots:
        m = c["shot"] == s
        if c["sb"][m].sum() <= 0 or m.sum() < 20:      # too few rows to score a discharge
            continue
        rows.append({
            "shot": float(s),
            "skill": float(1.0 - c["sm"][m].sum() / c["sb"][m].sum()),
            "campaign_position": float(s),
            "rows": float(m.sum()),
            "gap_mean_ms": float(np.mean(c["dt"][m])),
            "gap_max_ms": float(np.max(c["dt"][m])),
            "peak_fraction": float(np.mean(c["peak"][m])),
            "target_level": float(np.mean(c["y"][m])),
            "target_spread": float(np.std(c["y"][m])),
            "baseline_rmse": float(np.sqrt(np.mean(c["sb"][m]))),
        })
    return rows


COVARIATES = ("campaign_position", "rows", "gap_mean_ms", "gap_max_ms",
              "peak_fraction", "target_level", "target_spread", "baseline_rmse")


def rank(a):
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=float)
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else 0.0


def permutation_p(x, y, rng):
    """Two-sided p by shuffling the response, which keeps each covariate's own shape."""
    obs = abs(spearman(x, y))
    ry = rank(y)
    rx = rank(x) - rank(x).mean()
    hits = 0
    for _ in range(N_PERM):
        perm = rng.permutation(ry)
        perm = perm - perm.mean()
        d = np.sqrt((rx @ rx) * (perm @ perm))
        if d > 0 and abs(rx @ perm / d) >= obs:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="v2r63")
    args = ap.parse_args()
    rng = np.random.default_rng(PERM_SEED)
    out = {"arm": args.arm, "n_perm": N_PERM, "exploratory": True, "targets": {}}

    for target in TARGETS:
        rows = per_shot_table(args.arm, target)
        if not rows:
            raise SystemExit(f"FATAL: no artifacts for {args.arm}")
        skill = np.array([r["skill"] for r in rows])
        won = (skill > 0).mean()
        print("\n" + "=" * 88)
        print(f"{args.arm} {target}: {len(rows)} discharges, model wins {won:.1%}, "
              f"median per-discharge skill {np.median(skill):+.4f}")
        print("covariate".rjust(19) + "rho".rjust(8) + "p(perm)".rjust(10)
              + "p(Bonf)".rjust(10) + "  reading")
        node = {"n_shots": len(rows), "win_rate": float(won),
                "median_skill": float(np.median(skill)), "covariates": {}}
        for cov in COVARIATES:
            x = np.array([r[cov] for r in rows])
            if np.allclose(x, x[0]):
                continue
            rho = spearman(x, skill)
            p = permutation_p(x, skill, rng)
            padj = min(1.0, p * len(COVARIATES))
            note = ("higher -> model wins more" if rho > 0 else
                    "higher -> model wins less") if padj < 0.05 else ""
            node["covariates"][cov] = {"spearman_rho": rho, "p_perm": p, "p_bonferroni": padj}
            print(cov.rjust(19) + f"{rho:+.3f}".rjust(8) + f"{p:.4f}".rjust(10)
                  + f"{padj:.4f}".rjust(10) + "  " + note)
        out["targets"][target] = node

    dst = DATA / f".b9_shot_covariates_{args.arm}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"\n[cov] wrote {dst}")
    print("[cov] exploratory: no pre-registered rule, so nothing here promotes a claim.")


if __name__ == "__main__":
    main()
