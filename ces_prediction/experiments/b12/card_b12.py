"""B.12 evaluation card, axes 1 and 3 (PREREGISTRATION_B12.md §5-§6). CPU only.

For each population (cut) x target, over the fixed TEST hidden points:
  skill_vs_{pchip,gp}  = 1 - MSE_arm / MSE_baseline        (physical units)
  delta_vs_control     = (MSE_bilstm - MSE_arm) / MSE_pchip  (> 0: the arm is better)
with a shot-clustered bootstrap 95% CI, per split and pooled (clusters = physical shot),
and the §6 verdict: pooled delta > PRACTICAL_EPS, CI low > 0, same sign on >= 3/4 splits.
Strata (axis 3): the shot's observed V_rot label count (1-20 / > 20) and the length of
the hidden run a point sits in (1 / 2-10 / > 10 steps).

Rows are joined on (shot, row, target), and the script refuses to score an arm whose
point set differs from the baselines' -- the same guard as paired_model_compare.py.

Label-sparsified sets (test_sparse{k}: k V_rot labels kept per dense shot) are carded
the same way; there `mean` (train mean) is the zero-context reference.

Usage: py ces_prediction/experiments/b12/card_b12.py [--arms bilstm np tm_pre tm_scratch]
Output: data/.b12_card.json (+ a printed table).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parents[2] / "data"
SEEDS = (42, 1, 7, 123)
CUTS = (3000, 0)
PRACTICAL_EPS = 0.02
N_BOOT = 2000
CONTROL = "bilstm"


def load_points(path):
    z = np.load(path, allow_pickle=False)
    key = np.char.add(np.char.add(z["shot"].astype(str), ":"),
                      np.char.add(z["row"].astype(str), np.char.add(":", z["target"].astype(str))))
    order = np.argsort(key)
    return key[order], {k: z[k][order] for k in z.files}


def run_lengths(base):
    """Length of the hidden run each point sits in (consecutive hidden rows, same shot/target)."""
    out = np.ones(len(base["row"]), dtype=np.int64)
    idx = np.lexsort((base["row"], base["target"], base["shot"]))
    s, r, t = base["shot"][idx], base["row"][idx], base["target"][idx]
    start = 0
    for i in range(1, len(idx) + 1):
        if i == len(idx) or s[i] != s[i - 1] or t[i] != t[i - 1] or r[i] != r[i - 1] + 1:
            out[idx[start:i]] = i - start
            start = i
    return out


def vt_label_counts():
    """Shot file -> observed (held-free, cut-independent) V_rot label count."""
    path = DATA / ".b12_vt_label_counts.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    sys.path[:0] = [str(HERE.parents[0] / "seq"), str(HERE.parents[1])]
    from b12_data import load_population
    grid, _ = load_population(DATA, 0.0)
    counts = {n: int((~np.isnan(a[:, 2])).sum()) for n, a in grid.items()}
    path.write_text(json.dumps(counts), encoding="utf-8")
    return counts


def boot_ci(clusters, num, den, rng):
    """Ratio-of-sums statistic sum(num)/sum(den) with a cluster bootstrap 95% CI."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    cn = np.bincount(inv, weights=num, minlength=len(uniq))
    cd = np.bincount(inv, weights=den, minlength=len(uniq))
    est = cn.sum() / cd.sum()
    draws = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    reps = cn[draws].sum(1) / cd[draws].sum(1)
    return float(est), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def card(arms, split="test"):
    rng = np.random.default_rng(0)
    counts = vt_label_counts()
    result = {}
    for cut in CUTS:
        for t, tname in enumerate(("TI", "VT")):
            rows = {}
            pooled = {}
            for seed in SEEDS:
                bpath = DATA / f".b12_base_cut{cut}_s{seed}" / f"{split}_points.npz"
                if not bpath.exists():
                    continue
                bkey, base = load_points(bpath)
                sel = base["target"] == t
                runlen = run_lengths(base)
                preds = {m: base[m] for m in ("mean", "linear", "pchip", "gp")}
                for arm in arms:
                    p = DATA / f".b12_{arm}_cut{cut}_s{seed}" / f"{split}_points.npz"
                    if not p.exists():
                        continue
                    akey, a = load_points(p)
                    if not np.array_equal(akey, bkey):
                        raise SystemExit(f"FATAL: {p} scored a different point set than the baselines")
                    preds[arm] = a["pred"]
                if sel.sum() == 0:
                    continue
                valid = sel & np.all([np.isfinite(v) for v in preds.values()], axis=0)
                truth = base["truth"]
                shots = base["shot"]
                vtc = np.array([counts.get(s, 0) for s in shots])
                strata = {"all": valid,
                          "vt_labels_1_20": valid & (vtc <= 20),
                          "vt_labels_gt20": valid & (vtc > 20),
                          "gap_1": valid & (runlen == 1),
                          "gap_2_10": valid & (runlen >= 2) & (runlen <= 10),
                          "gap_gt10": valid & (runlen > 10)}
                se = {m: (v - truth) ** 2 for m, v in preds.items()}
                for st, m_ in strata.items():
                    if m_.sum() == 0:
                        continue
                    pooled.setdefault(st, []).append((shots[m_], {k: v[m_] for k, v in se.items()}))
                    entry = {"n": int(m_.sum())}
                    for k in se:
                        if k in ("pchip",):
                            continue
                        entry[f"{k}_skill_vs_pchip"] = boot_ci(shots[m_], se["pchip"][m_] - se[k][m_],
                                                               se["pchip"][m_], rng)
                        entry[f"{k}_skill_vs_gp"] = boot_ci(shots[m_], se["gp"][m_] - se[k][m_], se["gp"][m_], rng) \
                            if k != "gp" else None
                        if CONTROL in se and k not in (CONTROL, "mean", "linear", "gp"):
                            entry[f"{k}_delta_vs_control"] = boot_ci(shots[m_], se[CONTROL][m_] - se[k][m_],
                                                                     se["pchip"][m_], rng)
                    rows.setdefault(st, {})[str(seed)] = entry
            for st, parts in pooled.items():
                shots = np.concatenate([p[0] for p in parts])
                keys = set.intersection(*[set(p[1]) for p in parts])
                se = {k: np.concatenate([p[1][k] for p in parts]) for k in keys}
                entry = {"n": int(len(shots)), "n_splits": len(parts)}
                for k in se:
                    if k == "pchip":
                        continue
                    entry[f"{k}_skill_vs_pchip"] = boot_ci(shots, se["pchip"] - se[k], se["pchip"], rng)
                    if k != "gp":
                        entry[f"{k}_skill_vs_gp"] = boot_ci(shots, se["gp"] - se[k], se["gp"], rng)
                    if CONTROL in se and k not in (CONTROL, "mean", "linear", "gp"):
                        d = boot_ci(shots, se[CONTROL] - se[k], se["pchip"], rng)
                        per = [rows[st][s].get(f"{k}_delta_vs_control") for s in rows[st]]
                        signs = sum(1 for v in per if v and np.sign(v[0]) == np.sign(d[0]))
                        dg = boot_ci(shots, se["gp"] - se[k], se["pchip"], rng)
                        gp_ok = dg[0] > PRACTICAL_EPS and dg[1] > 0
                        entry[f"{k}_delta_vs_control"] = d
                        entry[f"{k}_verdict"] = (
                            "IMPROVE" if (d[0] > PRACTICAL_EPS and d[1] > 0 and signs >= 3 and gp_ok)
                            else "WORSE" if (d[0] < -PRACTICAL_EPS and d[2] < 0)
                            else "TIE" if abs(d[0]) <= PRACTICAL_EPS else "INCONCLUSIVE")
                rows.setdefault(st, {})["pooled"] = entry
            if rows:
                result[f"{split}/cut{cut}_{tname}"] = rows
    return result


def fmt(v):
    return "      -      " if v is None else f"{v[0]:+.3f}[{v[1]:+.2f},{v[2]:+.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["bilstm", "np", "tm_pre", "tm_scratch"])
    args = ap.parse_args()
    res = {}
    for split in ("test", *(f"test_sparse{k}" for k in (0, 3, 10))):
        res.update(card(args.arms, split))
    (DATA / ".b12_card.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for pop, rows in res.items():
        print(f"\n== {pop} (pooled; skill vs PCHIP, CI) ==")
        for st, by in rows.items():
            e = by.get("pooled")
            if not e:
                continue
            cols = [f"{k.replace('_skill_vs_pchip', '')}={fmt(v)}" for k, v in e.items() if k.endswith("_skill_vs_pchip")]
            print(f"  {st:16s} n={e['n']:6d}  " + "  ".join(cols))
            for k, v in e.items():
                if k.endswith("_verdict"):
                    arm = k[:-8]
                    print(f"  {'':16s} {arm}: delta vs {CONTROL} {fmt(e[arm + '_delta_vs_control'])} -> {v}")
    print(f"\nwrote {DATA / '.b12_card.json'}")


if __name__ == "__main__":
    main()
