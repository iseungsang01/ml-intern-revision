# -*- coding: utf-8 -*-
"""How many test shots does our gate need before it can detect anything?

`bootstrap_compare._bootstrap` resamples SHOTS, so the number of test shots is the number
of bootstrap clusters. Ten shots is a small number of clusters, and "small" needed to be a
measurement rather than a worry -- so this replays the real gate on the real per-shot errors
of the paper's main model, drawing k of the 96 test shots at a time.

Two comparisons, because they have very different variance:
  * model vs PCHIP        -- different methods; large per-shot spread
  * seq_v2 vs W=2 control -- two model arms on identical rows, which is the shape of the
                             question "does a microsecond-MC feature help?"

Two resampling units:
  * shot                  -- the protocol's unit
  * shot x 500 ms block   -- what a microsecond dataset makes available

And an attenuation sweep (alpha scales the arm difference), because a microsecond-MC
feature is unlikely to buy as much as a whole architecture change.

Usage (repo root):  py ces_prediction/experiments/hires_shots/power_analysis.py
Writes: power_analysis.json next to this file.

Headline results (2026-08-17):
  * Attenuation barely matters -- skill is a ratio, so shrinking the effect shrinks the
    spread with it. A quarter-size effect keeps essentially the same power.
  * What decides power is CONSISTENCY: the fraction of shots where the arm actually wins.
    CES_TI sits at 0.66 -> ~30 % power at any k we can afford; CES_VT at 0.88 -> ~70-90 %.
  * k = 2 is not "weaker", it is misleading: with two clusters half of all resamples repeat
    one shot, the CI collapses, and the measured pass rate rises above k = 3. Those extra
    passes are false positives. k = 3 is the smallest honest size.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("CES_DATA_DIR", REPO_ROOT / "data"))
HERE = Path(__file__).resolve().parent

B_DRAWS = 4000          # the gate itself uses 10000; 4000 keeps this sweep affordable
N_TRIALS = 400          # random shot subsets per configuration
KS = (2, 3, 4, 5, 6, 7, 8, 10, 12)
ALPHAS = (1.0, 0.5, 0.25)
BLOCK_MS = 500.0
ARM_A = DATA / ".b1_seqv2_s42_i42" / "comparison_errors_test.npz"
ARM_B = DATA / ".b1_w2cut_s42" / "comparison_errors_test.npz"


def group_sums(key, se):
    order = np.argsort(key, kind="stable")
    k, v = key[order], se[order]
    uniq, start = np.unique(k, return_index=True)
    return uniq, np.add.reduceat(v, start)


def gate(sm, sb, rng):
    """The statistic of record: skill = 1 - sum(SE_variant)/sum(SE_base), 95 % CI > 0."""
    n = len(sm)
    idx = rng.integers(0, n, size=(B_DRAWS, n))
    skill = 1.0 - sm[idx].sum(axis=1) / sb[idx].sum(axis=1)
    f = np.isfinite(skill)
    return bool(np.percentile(skill[f], 2.5) > 0.0) if f.any() else False


def sweep(shot, se_variant, se_base, rng, label, out, row_idx=None):
    for alpha in ALPHAS:
        se_eff = se_base + alpha * (se_variant - se_base)
        uniq, sm = group_sums(shot, se_eff)
        _, sb = group_sums(shot, se_base)
        per_shot = 1.0 - sm / sb
        rec = {"comparison": label, "alpha": alpha,
               "full_skill": float(1.0 - sm.sum() / sb.sum()),
               "per_shot_skill_median": float(np.median(per_shot)),
               "frac_shots_positive": float(np.mean(per_shot > 0)),
               "power_shot": {}, "power_block": {}}
        for k in KS:
            if k > len(uniq):
                continue
            hits = sum(gate(sm[p], sb[p], rng) for p in
                       (rng.choice(len(uniq), size=k, replace=False) for _ in range(N_TRIALS)))
            rec["power_shot"][str(k)] = hits / N_TRIALS
        if row_idx is not None:
            blk = np.zeros(len(shot), dtype=np.int64)
            for u in np.unique(shot):
                m = shot == u
                r = row_idx[m].astype(float)
                blk[m] = u * 100000 + ((r - r.min()) * 10.0 // BLOCK_MS).astype(np.int64)
            shots_all = np.unique(shot)
            for k in (2, 3, 6, 10):
                if k > len(shots_all):
                    continue
                hits = 0
                for _ in range(N_TRIALS):
                    pick = rng.choice(shots_all, size=k, replace=False)
                    m = np.isin(shot, pick)
                    _, smb = group_sums(blk[m], se_eff[m])
                    _, sbb = group_sums(blk[m], se_base[m])
                    hits += gate(smb, sbb, rng)
                rec["power_block"][str(k)] = hits / N_TRIALS
        out.append(rec)
        line = (f"  alpha={alpha:4.2f}  skill={rec['full_skill']:+.4f}  "
                f"frac_shots_positive={rec['frac_shots_positive']:.2f}")
        print(line)
        print("      shot : " + "  ".join(f"k={k}:{v * 100:5.1f}%"
                                          for k, v in rec["power_shot"].items()))
        if rec["power_block"]:
            print("      block: " + "  ".join(f"k={k}:{v * 100:5.1f}%"
                                              for k, v in rec["power_block"].items()))


def main():
    if not ARM_A.exists() or not ARM_B.exists():
        raise SystemExit(f"need {ARM_A} and {ARM_B} (run the B.1 gate batch first)")
    a, b = np.load(ARM_A, allow_pickle=True), np.load(ARM_B, allow_pickle=True)
    rng = np.random.default_rng(20260817)
    out = []
    for target in ("CES_TI", "CES_VT"):
        shot = a[f"{target}_shot"]
        row_idx = a[f"{target}_row"] if f"{target}_row" in a else np.arange(len(shot))
        print(f"\n{'=' * 96}\n{target}: model vs PCHIP  (rows={len(shot)}, "
              f"shots={len(np.unique(shot))})")
        sweep(shot, a[f"{target}_se_model"].astype(float),
              a[f"{target}_se_pchip"].astype(float), rng, f"{target}|model_vs_pchip", out)
        if np.array_equal(shot, b[f"{target}_shot"]):
            print(f"\n{target}: seq_v2 vs W=2 control (paired arms, identical rows)")
            sweep(shot, a[f"{target}_se_model"].astype(float),
                  b[f"{target}_se_model"].astype(float), rng,
                  f"{target}|seqv2_vs_w2control", out, row_idx=row_idx)
        else:
            print(f"\n{target}: arm rows differ -- paired-arm sweep skipped")
    (HERE / "power_analysis.json").write_text(
        json.dumps({"B_draws": B_DRAWS, "n_trials": N_TRIALS, "block_ms": BLOCK_MS,
                    "results": out}, indent=1), encoding="utf-8")
    print(f"\nwrote {HERE / 'power_analysis.json'}")


if __name__ == "__main__":
    main()
