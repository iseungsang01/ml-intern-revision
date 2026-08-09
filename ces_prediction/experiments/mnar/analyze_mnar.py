"""Quantify the MNAR optimism bound instead of merely disclaiming it.

The paper's largest stated caveat is that skill is measured on CES points that
happen to be OBSERVED, while the points a nowcaster exists to fill are MISSING --
and CES missingness is not random (drop-out at low S/N, ELMs, transitions). Until
now the paper said only "this is an optimistic bound", with the magnitude unknown.
An unquantified caveat is the easiest thing in the paper to attack.

We cannot measure error where there is no ground truth. What we CAN do is measure
how skill varies over covariates that are computable at BOTH observed and missing
rows, then reweight the observed points to the covariate distribution of the
genuinely missing ones. Two covariates carry the missingness mechanism here:

  dt   time since the last genuinely observed value of that target -- a missing
       point sits, by construction, deeper into a drop-out than a typical
       observed point;
  act  the input-only local-activity flag of peak_analysis (large neighbour-bracket
       slope and/or high local neighbour variance), computed from the neighbour set
       that EXCLUDES the row itself -- so it is defined at a missing row too. This
       is the ELM/transition axis of MNAR.

Post-stratifying on (dt x act) gives an estimate of skill at the missing points
under the assumption that, WITHIN a stratum, missing and observed rows are
exchangeable. That assumption is not free, and it is stated rather than hidden --
but it replaces an unbounded caveat with a number and a direction.

The scored side comes from the frozen npz (its own dt_ms and is_peak), and the
weights from the same dataset construction compare_baselines uses, so the two sides
share a universe. Alignment is verified before any reweighting is reported.

Usage (repo root):  py ces_prediction/experiments/mnar/analyze_mnar.py
Writes: data/.mnar_analysis.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ces_prediction"))
import baselines_interpolation as B  # noqa: E402
import peak_analysis as PK  # noqa: E402
from dataset import (  # noqa: E402
    STUCK_GAP_SECONDS,
    TARGET_COLUMNS,
    TIME_COLUMN,
    KSTAR_CES_Dataset,
)

SEEDS = (42, 1, 7, 123)
SPLIT_SRC = {
    42: REPO_ROOT / "data" / ".improve_split" / "w4",
    1: REPO_ROOT / "data" / ".ms_split_1",
    7: REPO_ROOT / "data" / ".ms_split_7",
    123: REPO_ROOT / "data" / ".ms_split_123",
}
RUN_DIR = {
    42: REPO_ROOT / "data" / ".vt_repro_out",
    1: REPO_ROOT / "data" / ".vt_repro_ms_1",
    7: REPO_ROOT / "data" / ".vt_repro_ms_7",
    123: REPO_ROOT / "data" / ".vt_repro_ms_123",
}
NPZ_SUFFIX = "__test_genuine_persist"   # genuine treatment + the causal arm
DATA_DIR = Path(os.getenv("CES_DATA_DIR", REPO_ROOT / "data"))
BASELINES = ("pchip", "persistence")
DT_EDGES = (15.0, 25.0, 45.0)           # -> 4 dt strata, matching the paper's bins
WINDOW = 4                              # the model's history window (CES_WINDOW_SIZE)
MIN_SCORED_PER_STRATUM = 30             # below this a stratum's mean SE is not usable


def dt_bin(dt_ms):
    return np.digitize(dt_ms, DT_EDGES)


def load_grid(path):
    """Rows exactly as compare_baselines sees them, but WITHOUT dropping rows whose
    targets are all missing -- those rows are the population we are reweighting to."""
    df = pd.read_csv(path)
    cols = list(df.columns)
    bes = [c for c in cols if c.startswith("BES")]
    ecei = [c for c in cols if c.startswith("ECEI")]
    mc = [c for c in cols if c.startswith("MC")]
    need = [TIME_COLUMN, *bes, *ecei, *mc]
    df = df.dropna(subset=need).reset_index(drop=True)
    if df.empty:
        return None
    # Held/forward-filled targets are not measurements: NaN them, exactly as the
    # genuine-evaluation dataset does (same helper, same gap constant).
    time = df[TIME_COLUMN].to_numpy(dtype=np.float64)
    if len(time) >= 2:
        block = np.concatenate(([0], np.cumsum(np.diff(time) >= STUCK_GAP_SECONDS)))
        for col in TARGET_COLUMNS:
            v = df[col].to_numpy(dtype=np.float64, copy=True)
            stuck = KSTAR_CES_Dataset._stuck_repeat_mask(v, block)
            v[stuck] = np.nan
            df[col] = v
    usecols = [TIME_COLUMN, *TARGET_COLUMNS, *bes, *ecei, *mc]
    arr = df.loc[:, usecols].to_numpy(dtype=np.float64)
    slices = {
        "time": 0,
        "CES_TI": 1,
        "CES_VT": 2,
        "bes": list(range(3, 3 + len(bes))),
        "ecei": list(range(3 + len(bes), 3 + len(bes) + len(ecei))),
    }
    return arr, slices


def row_covariates(arr, slices, target, window=WINDOW):
    """(dt_ms, activity, observed, in_domain) per row for one target.

    All are computable at a MISSING row, because the neighbour set excludes the row
    itself. `in_domain` is the decisive one: `compare_baselines` scores a sample only
    when an observed value of that target exists INSIDE the model's window, so a
    missing row whose nearest past observation is further back than `window - 1` rows
    is not merely hard -- the model cannot be applied to it at all. Counting those
    rows as 'missing points we should be able to fill' would silently invent a
    population the method never claimed.
    """
    tcol = slices[target]
    time_col = slices["time"]
    n = len(arr)
    dt_ms = np.full(n, np.inf)
    has_past = np.zeros(n, dtype=bool)
    in_domain = np.zeros(n, dtype=bool)
    observed = np.isfinite(arr[:, tcol])
    times = arr[:, time_col].astype(np.float64)
    block = PK._block_id_array(times, B.GAP_SECONDS)
    # nearest past observed ROW index of this target, within the same block
    last_obs_row = -1
    last_block = None
    for i in range(n):
        if last_block != block[i]:
            last_obs_row, last_block = -1, block[i]
        if last_obs_row >= 0:
            has_past[i] = True
            dt_ms[i] = (times[i] - times[last_obs_row]) * 1000.0
            in_domain[i] = (i - last_obs_row) <= (window - 1)
        if observed[i]:
            last_obs_row = i
    act = PK.detect_peak_rows_input_only(
        arr, time_col, tcol, bes_cols=slices["bes"], ecei_cols=slices["ecei"]
    )["ces_activity"]
    return dt_ms, act, observed, has_past, in_domain


def stratum_key(dt_ms, act):
    return dt_bin(dt_ms) * 2 + act.astype(int)


N_STRATA = (len(DT_EDGES) + 1) * 2


def weights_from_grid(seed, target):
    """Stratum distribution of OBSERVED vs GENUINELY-MISSING rows on the test shots.

    Returns the missing distribution twice: over all missing rows that have any past
    observation, and over the IN-DOMAIN subset the model can actually be applied to.
    """
    manifest = json.loads((SPLIT_SRC[seed] / "split_manifest.json").read_text(encoding="utf-8"))
    obs = np.zeros(N_STRATA)
    mis = np.zeros(N_STRATA)
    mis_dom = np.zeros(N_STRATA)
    n_files = 0
    n_missing_any = n_missing_dom = 0
    for path in manifest["test_files"]:
        p = Path(path)
        if not p.exists():
            p = DATA_DIR / Path(path).name
        if not p.exists():
            continue
        loaded = load_grid(p)
        if loaded is None:
            continue
        arr, slices = loaded
        dt_ms, act, observed, has_past, in_domain = row_covariates(arr, slices, target)
        keep = has_past & np.isfinite(dt_ms)
        k = stratum_key(dt_ms[keep], act[keep])
        o = observed[keep]
        d = in_domain[keep]
        np.add.at(obs, k[o], 1.0)
        np.add.at(mis, k[~o], 1.0)
        np.add.at(mis_dom, k[~o & d], 1.0)
        n_missing_any += int((~o).sum())
        n_missing_dom += int((~o & d).sum())
        n_files += 1
    return {"obs": obs, "mis": mis, "mis_dom": mis_dom, "n_files": n_files,
            "n_missing_any": n_missing_any, "n_missing_dom": n_missing_dom}


def scored_strata(seed, target, sample_mask=None):
    """Per-stratum squared-error sums from the frozen npz (the scored population)."""
    z = _npz(seed)
    k = stratum_key(z[f"{target}_dt_ms"], z[f"{target}_is_peak"].astype(bool))
    if sample_mask is not None:
        k = k[sample_mask]
    out = {"n": np.zeros(N_STRATA)}
    np.add.at(out["n"], k, 1.0)
    for arm in ("model", *BASELINES):
        se = z[f"{target}_se_{arm}"]
        if sample_mask is not None:
            se = se[sample_mask]
        s = np.zeros(N_STRATA)
        np.add.at(s, k, se)
        out[arm] = s
    return out


_NPZ_CACHE = {}


def _npz(seed):
    if seed not in _NPZ_CACHE:
        _NPZ_CACHE[seed] = dict(
            np.load(RUN_DIR[seed] / f"comparison_errors_test{NPZ_SUFFIX}.npz")
        )
    return _NPZ_CACHE[seed]


def bootstrap_reweighted(seed, target, w, base, n_boot=2000, boot_seed=12345):
    """Shot-clustered CI on the reweighted skill.

    Whole shots are resampled on the SCORED side; the stratum weights `w` are held
    fixed. So the interval covers error variability, not the (second-order)
    sampling error of the weights themselves -- stated, not glossed over.
    """
    z = _npz(seed)
    shots = z[f"{target}_shot"]
    uniq, inv = np.unique(shots, return_inverse=True)
    by_shot = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(boot_seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([by_shot[i] for i in pick])
        mask = np.zeros(len(shots), dtype=bool)
        counts = np.bincount(idx, minlength=len(shots))
        mask[:] = counts > 0
        # weight-aware recompute: use repeat counts so a shot drawn twice counts twice
        k = stratum_key(z[f"{target}_dt_ms"], z[f"{target}_is_peak"].astype(bool))
        n = np.zeros(N_STRATA)
        np.add.at(n, k, counts.astype(float))
        sums = {}
        for arm in ("model", base):
            s = np.zeros(N_STRATA)
            np.add.at(s, k, z[f"{target}_se_{arm}"] * counts)
            sums[arm] = s
        ok = (n >= MIN_SCORED_PER_STRATUM) & (w > 0)
        if not ok.any():
            continue
        p = w[ok] / w[ok].sum()
        mse_m = (sums["model"][ok] / n[ok] * p).sum()
        mse_b = (sums[base][ok] / n[ok] * p).sum()
        if mse_b > 0:
            vals.append(1.0 - mse_m / mse_b)
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"ci95": [float(lo), float(hi)], "pass": bool(lo > 0)}


def alignment_check(seed, target, obs_w):
    """The scored side takes its activity flag from the npz; the weights take theirs
    from a grid that also keeps all-missing rows, so the top-percentile thresholds
    are computed over slightly different populations. If the two disagree on the
    OBSERVED rows, post-stratification is comparing different variables."""
    z = _npz(seed)
    npz_rate = float(z[f"{target}_is_peak"].astype(bool).mean())
    grid_rate = float(obs_w[1::2].sum() / obs_w.sum()) if obs_w.sum() else float("nan")
    return {"activity_rate_npz": npz_rate, "activity_rate_grid_observed": grid_rate,
            "abs_diff": abs(npz_rate - grid_rate)}


def reweighted_skill(scored, w, base, min_n=MIN_SCORED_PER_STRATUM):
    """Skill with each stratum's MEAN squared error reweighted to distribution w.

    Strata with fewer than `min_n` scored samples are DROPPED rather than trusted,
    and the returned coverage says how much of the target distribution's mass the
    surviving strata actually carry. A reweighted number with 5% coverage is not a
    weaker estimate, it is a different question -- so coverage is reported next to
    every estimate rather than buried.
    """
    n = scored["n"]
    ok = (n >= min_n) & (w > 0)
    if not ok.any():
        return None, 0.0
    p = w[ok] / w[ok].sum()
    mse_m = (scored["model"][ok] / n[ok] * p).sum()
    mse_b = (scored[base][ok] / n[ok] * p).sum()
    covered = w[ok].sum() / w.sum() if w.sum() else 0.0
    return (1.0 - mse_m / mse_b if mse_b > 0 else None), float(covered)


def main():
    results = {"_design": {
        "covariates": "dt since last genuine observation x input-only local-activity flag",
        "dt_edges_ms": list(DT_EDGES),
        "assumption": ("within a stratum, missing and observed rows are exchangeable "
                       "(MAR given these covariates); stated, not assumed away"),
        "npz": f"comparison_errors_test{NPZ_SUFFIX}.npz (genuine treatment)",
    }, "per_seed": {}}

    for seed in SEEDS:
        entry = {"targets": {}}
        for target in TARGET_COLUMNS:
            g = weights_from_grid(seed, target)
            obs_w, mis_w, dom_w = g["obs"], g["mis"], g["mis_dom"]
            entry["n_test_files_read"] = g["n_files"]
            scored = scored_strata(seed, target)
            tgt = {
                "n_observed_rows": float(obs_w.sum()),
                "n_missing_rows": float(mis_w.sum()),
                "missing_fraction": float(mis_w.sum() / max(obs_w.sum() + mis_w.sum(), 1)),
                "n_missing_in_domain": g["n_missing_dom"],
                "in_domain_fraction_of_missing": (
                    g["n_missing_dom"] / g["n_missing_any"] if g["n_missing_any"] else None),
                "n_scored": float(scored["n"].sum()),
            }
            tgt["alignment"] = alignment_check(seed, target, obs_w)
            for base in BASELINES:
                sk_obs, cov_obs = reweighted_skill(scored, obs_w, base)
                sk_all, cov_all = reweighted_skill(scored, mis_w, base)
                sk_dom, cov_dom = reweighted_skill(scored, dom_w, base)
                tgt[f"vs_{base}"] = {
                    "skill_observed_weighted": sk_obs,
                    "skill_missing_all": sk_all,
                    "coverage_missing_all": cov_all,
                    "skill_missing_in_domain": sk_dom,
                    "coverage_missing_in_domain": cov_dom,
                    "optimism_in_domain": (None if (sk_obs is None or sk_dom is None)
                                           else sk_obs - sk_dom),
                    "stratum_coverage_observed": cov_obs,
                    "in_domain_boot": bootstrap_reweighted(seed, target, dom_w, base),
                }
            tgt["stratum_table"] = [
                {"dt_bin": int(s // 2), "activity": bool(s % 2),
                 "w_observed": float(obs_w[s]), "w_missing": float(mis_w[s]),
                 "w_missing_in_domain": float(dom_w[s]),
                 "n_scored": float(scored["n"][s]),
                 "usable": bool(scored["n"][s] >= MIN_SCORED_PER_STRATUM),
                 "rmse_model": (float(np.sqrt(scored["model"][s] / scored["n"][s]))
                                if scored["n"][s] else None)}
                for s in range(N_STRATA)
            ]
            entry["targets"][target] = tgt
        results["per_seed"][seed] = entry
        print(f"[mnar] seed {seed}: {entry['n_test_files_read']} test files read")

    out = REPO_ROOT / "data" / ".mnar_analysis.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}\n")

    for target in TARGET_COLUMNS:
        print(f"=== {target}: does scoring observed points flatter the model?")
        t0 = results["per_seed"][SEEDS[0]]["targets"][target]
        print(f"  missing rows: {t0['missing_fraction']*100:.1f}% of the grid; of those "
              f"{(t0['in_domain_fraction_of_missing'] or 0)*100:.1f}% are IN-DOMAIN "
              f"(an observed value inside the W={WINDOW} window) -- the rest cannot be "
              f"nowcast by this model at all.")
        print(f"{'seed':>5} {'vs':>12} {'observed':>10} {'in-domain':>11} "
              f"{'95% CI':>22} {'gate':>5} {'optimism':>9} {'cover':>7}")
        for seed in SEEDS:
            t = results["per_seed"][seed]["targets"][target]
            for base in BASELINES:
                e = t[f"vs_{base}"]
                if e["skill_observed_weighted"] is None or e["skill_missing_in_domain"] is None:
                    continue
                b = e.get("in_domain_boot") or {}
                ci = b.get("ci95", [float("nan")] * 2)
                gate = "PASS" if b.get("pass") else "n.s."
                print(f"{seed:>5} {base:>12} {e['skill_observed_weighted']:>+10.3f} "
                      f"{e['skill_missing_in_domain']:>+11.3f} "
                      f"[{ci[0]:+.3f}, {ci[1]:+.3f}]".rjust(22)
                      + f" {gate:>5} {e['optimism_in_domain']:>+9.3f}"
                      + f" {e['coverage_missing_in_domain']*100:>6.0f}%")
        al = t0["alignment"]
        print(f"      alignment (activity rate, npz vs grid-observed): "
              f"{al['activity_rate_npz']:.3f} vs {al['activity_rate_grid_observed']:.3f} "
              f"(|diff| {al['abs_diff']:.3f})")
        print()


if __name__ == "__main__":
    main()
