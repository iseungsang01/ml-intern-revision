"""B.39 read-outs (PREREGISTRATION_B39.md).  CPU; reads the comparison_errors_test.npz of the runs.

    py ces_prediction/experiments/b39_primary/analyze_b39.py
Output: docs/paper/b39_primary.json / .md   (arm P is read only when its 80 runs are complete)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_b39 as B  # noqa: E402
from run_b39 import B27, DATA  # noqa: E402

REPO = HERE.parents[2]
OUT_JSON = REPO / "docs" / "paper" / "b39_primary.json"
OUT_MD = REPO / "docs" / "paper" / "b39_primary.md"
T = "CES_TI"
REFS = {"gp_causal": "causal GP", "persistence": "hold", "pchip": "PCHIP", "gp": "offline GP"}
N_BOOT, SEED, B_POOLED, SEED_POOLED = 2000, 20261117, 20_000, 2728


def load(run):
    z = np.load(run / "comparison_errors_test.npz")
    d = {"shot": z[f"{T}_shot"], "row": z[f"{T}_row"], "model": z[f"{T}_se_model"].astype(np.float64)}
    for r in REFS:
        d[r] = z[f"{T}_se_{r}"].astype(np.float64)
    return d


def same_rows(a, b, what):
    if len(a["shot"]) != len(b["shot"]) or not (np.array_equal(a["shot"], b["shot"]) and np.array_equal(a["row"], b["row"])):
        raise SystemExit(f"FATAL: {what}: the two runs were not scored on the same rows")
    if not np.allclose(a["gp_causal"], b["gp_causal"], rtol=1e-6, atol=1e-6, equal_nan=True):
        raise SystemExit(f"FATAL: {what}: the reference errors differ between the two runs")


def sums(shot, v):
    u, inv = np.unique(shot, return_inverse=True)
    return np.bincount(inv, weights=v, minlength=len(u))


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def two_level(A, Bm, rng):
    """A (ia, n), Bm (ib, n): per-discharge sums; an initialisation of each drawn independently."""
    n = A.shape[1]
    idx = rng.integers(0, n, size=(N_BOOT, n))
    ia, ib = rng.integers(0, A.shape[0], N_BOOT), rng.integers(0, Bm.shape[0], N_BOOT)
    a = np.take_along_axis(A[ia], idx, 1).sum(1)
    b = np.take_along_axis(Bm[ib], idx, 1).sum(1)
    return [float(1 - A.sum(1).mean() / Bm.sum(1).mean()), *ci(1 - a / b)]


def med(v):
    return float(np.median(np.sqrt(v)))


def arm_a(rng):
    out = {}
    for s in B.SEEDS:
        ini = B.inits(s)
        need = [B.split_run_dir("v2", s, i) for i in ini] + [B.twin_dir(s, i) for i in ini] + \
               [B.split_run_dir(m, s, s) for m in B.ABLATIONS]
        have1 = all((d / "comparison_errors_test.npz").exists() for d in
                    [B.split_run_dir("v2", s, s), B.twin_dir(s, s)] + [B.split_run_dir(m, s, s) for m in B.ABLATIONS])
        if not have1:
            continue
        full = all((d / "comparison_errors_test.npz").exists() for d in need)
        use = ini if full else (s,)
        c = {i: load(B.split_run_dir("v2", s, i)) for i in use}
        t = {i: load(B.twin_dir(s, i)) for i in use}
        ab = {m: load(B.split_run_dir(m, s, s)) for m in B.ABLATIONS}
        ref = c[s]
        for what, d in [(f"s{s} v2 i{i}", c[i]) for i in use] + [(f"s{s} v2h i{i}", t[i]) for i in use] + \
                       [(f"s{s} {m}", ab[m]) for m in ab]:
            same_rows(ref, d, what)
        shot = ref["shot"]
        S = lambda v: sums(shot, v)                                             # noqa: E731
        C = np.array([S(c[i]["model"]) for i in use])
        Tw = np.array([S(t[i]["model"]) for i in use])
        R = {r: S(ref[r])[None, :] for r in REFS}
        o = {"rows": int(len(shot)), "discharges": int(C.shape[1]), "initialisations": list(use), "three_initialisations": bool(full),
             "S1_causal_over_gp": two_level(C, R["gp_causal"], rng), "S2_causal_over_twin": two_level(C, Tw, rng),
             "twin_over_gp": two_level(Tw, R["gp_causal"], rng),
             "causal_over": {r: two_level(C, R[r], rng) for r in REFS},
             "per_initialisation": {str(i): {"causal_over_gp": float(1 - c[i]["model"].sum() / ref["gp_causal"].sum()),
                                             "twin_over_gp": float(1 - t[i]["model"].sum() / ref["gp_causal"].sum()),
                                             "causal_over_twin": float(1 - c[i]["model"].sum() / t[i]["model"].sum())} for i in use},
             "median_abs_error_eV": {"hold": med(ref["persistence"]), "causal GP": med(ref["gp_causal"]),
                                     "causal variant": float(np.mean([med(c[i]["model"]) for i in use])),
                                     "twin": float(np.mean([med(t[i]["model"]) for i in use])),
                                     **{m: med(ab[m]["model"]) for m in ab}},
             "rmse_eV": {"hold": float(np.sqrt(ref["persistence"].mean())), "causal GP": float(np.sqrt(ref["gp_causal"].mean())),
                         "causal variant": float(np.mean([np.sqrt(c[i]["model"].mean()) for i in use])),
                         "twin": float(np.mean([np.sqrt(t[i]["model"].mean()) for i in use])),
                         **{m: float(np.sqrt(ab[m]["model"].mean())) for m in ab}}}
        v = [x["causal_over_gp"] for x in o["per_initialisation"].values()]
        o["spread_causal_over_gp"] = float(max(v) - min(v))
        one = lambda a, b: two_level(S(a)[None, :], S(b)[None, :], rng)        # noqa: E731
        tw, cs = t[s]["model"], c[s]["model"]
        o["D1_ecei_over_twin"] = one(ab["v2e"]["model"], tw)
        o["D2_bes_over_twin"] = one(ab["v2b"]["model"], tw)
        o["nomirnov_over_twin"] = one(ab["v2n"]["model"], tw)
        o["full_over_twin_same_initialisation"] = one(cs, tw)
        o["full_over_ecei"] = one(cs, ab["v2e"]["model"])
        o["full_over_bes"] = one(cs, ab["v2b"]["model"])
        o["full_over_nomirnov"] = one(cs, ab["v2n"]["model"])
        o["ablation_over_gp"] = {m: one(ab[m]["model"], ref["gp_causal"]) for m in ab}
        o["epochs_run"] = {}
        for name, d in [(f"v2 i{i}", B.split_run_dir("v2", s, i)) for i in use] + [(f"v2h i{i}", B.twin_dir(s, i)) for i in use] + \
                       [(m, B.split_run_dir(m, s, s)) for m in ab]:
            mm = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            o["epochs_run"][name] = [mm["epochs_run"], mm["best_epoch"]]
        for k in ("S1_causal_over_gp", "S2_causal_over_twin", "D1_ecei_over_twin", "D2_bes_over_twin"):
            o[k.split("_")[0] + "_holds"] = bool(o[k][1] > 0)
        out[str(s)] = o
    return out


def arm_p():
    runs = [(m, r, k) for m in ("v2", "v2h") for r in B.REPEATS for k in range(B27.K)]
    if not all((B.pooled_run_dir(*x) / "comparison_errors_test.npz").exists() for x in runs):
        done = sum((B.pooled_run_dir(*x) / "comparison_errors_test.npz").exists() for x in runs)
        return {"complete": False, "runs_done": int(done), "runs_total": len(runs)}
    rng = np.random.default_rng(SEED_POOLED)
    per, fold = {}, {"gp": [], "twin": []}
    ratios = []
    for r in B.REPEATS:
        cat = {k: [] for k in ("shot", "c", "t", *REFS, "w")}
        for k in range(B27.K):
            c, t = load(B.pooled_run_dir("v2", r, k)), load(B.pooled_run_dir("v2h", r, k))
            same_rows(c, t, f"r{r} f{k}")
            w = None
            wd = DATA / f".b27_seqv2_cut_r{r}_f{k}" / "comparison_errors_test.npz"
            if wd.exists():
                zz = np.load(wd)
                if len(zz[f"{T}_shot"]) == len(c["shot"]) and np.array_equal(zz[f"{T}_row"], c["row"]):
                    w = zz[f"{T}_se_model"].astype(np.float64)
            cat["shot"].append(c["shot"]); cat["c"].append(c["model"]); cat["t"].append(t["model"])
            cat["w"].append(w if w is not None else np.full(len(c["shot"]), np.nan))
            for q in REFS:
                cat[q].append(c[q])
            fold["gp"].append(float(1 - c["model"].sum() / c["gp_causal"].sum()))
            fold["twin"].append(float(1 - c["model"].sum() / t["model"].sum()))
            mm = json.loads((B27.split_dir(r, k) / "split_manifest.json").read_text(encoding="utf-8"))
            ratios.append(mm["test_file_count"] / mm["train_file_count"])
        shot = np.concatenate(cat["shot"])
        per[r] = {k: sums(shot, np.concatenate(v)) for k, v in cat.items() if k != "shot"}
        per[r]["abs_c"] = np.sqrt(np.concatenate(cat["c"])); per[r]["abs_h"] = np.sqrt(np.concatenate(cat["persistence"]))
        per[r]["abs_t"] = np.sqrt(np.concatenate(cat["t"]))

    def hier(a, b):
        draws = rng.choice(np.array(B.REPEATS), size=B_POOLED)
        o = np.empty(B_POOLED)
        for r in B.REPEATS:
            sel = np.flatnonzero(draws == r)
            m, bb = per[r][a], per[r][b]
            idx = rng.integers(0, len(m), size=(len(sel), len(m)))
            o[sel] = 1.0 - m[idx].sum(1) / bb[idx].sum(1)
        point = float(np.mean([1 - per[r][a].sum() / per[r][b].sum() for r in B.REPEATS]))
        return [point, *ci(o)], {str(r): float(1 - per[r][a].sum() / per[r][b].sum()) for r in B.REPEATS}

    def nb(vals):
        v = np.asarray(vals)
        n, mean, var = len(v), float(v.mean()), float(v.var(ddof=1))
        half = float(sps.t.ppf(0.975, n - 1)) * np.sqrt((1.0 / n + float(np.mean(ratios))) * var)
        return [mean, mean - half, mean + half]

    out = {"complete": True, "repeats": list(B.REPEATS)}
    out["P1_causal_over_gp"], out["P1_per_repeat"] = hier("c", "gp_causal")
    out["P2_causal_over_twin"], out["P2_per_repeat"] = hier("c", "t")
    out["twin_over_gp"], _ = hier("t", "gp_causal")
    out["causal_over"] = {q: hier("c", q)[0] for q in REFS}
    if all(np.isfinite(per[r]["w"]).all() for r in B.REPEATS):
        out["causal_over_whole_discharge_variant_cap30"], _ = hier("c", "w")
        out["whole_discharge_variant_cap30_over_gp"], _ = hier("w", "gp_causal")
    out["P1_fold_level_nadeau_bengio"] = nb(fold["gp"])
    out["P2_fold_level_nadeau_bengio"] = nb(fold["twin"])
    out["median_abs_error_eV"] = {"hold": float(np.mean([np.median(per[r]["abs_h"]) for r in B.REPEATS])),
                                  "causal variant": float(np.mean([np.median(per[r]["abs_c"]) for r in B.REPEATS])),
                                  "twin": float(np.mean([np.median(per[r]["abs_t"]) for r in B.REPEATS]))}
    out["P1_holds"] = bool(out["P1_causal_over_gp"][1] > 0)
    out["P2_holds"] = bool(out["P2_causal_over_twin"][1] > 0)
    return out


def render(res):
    s3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                   # noqa: E731
    A, P = res["A"], res["P"]
    done = [s for s in B.SEEDS if str(s) in A]
    cnt = lambda k: sum(A[str(s)][k] for s in done)                             # noqa: E731
    L = ["# B.39 — the causal variant as the primary estimator, and which diagnostic carries the increment", "",
         "Registered: PREREGISTRATION_B39.md. GPU runs; read-outs on a CPU. T_i, cut population, epoch cap 80 for every "
         "network, fast channels standardized with the statistics of the training discharges. Skill = 1 - sum of squared "
         "errors of A over that of R on the same test rows.", "",
         f"Splits read: {len(done)} of 4" + ("" if all(A[str(s)]["three_initialisations"] for s in done) else
                                            " (some with the initialisation = split seed only; the further initialisations are not yet complete)") + ".", ""]
    if done:
        L += [f"- **S1** (causal variant over the causal GP; interval over initialisations and discharges): {cnt('S1_holds')} of {len(done)}.",
              f"- **S2** (causal variant over the history-only twin): {cnt('S2_holds')} of {len(done)}.",
              f"- **D1** (ECEI-only variant over the twin): {cnt('D1_holds')} of {len(done)}.",
              f"- **D2** (BES-only variant over the twin): {cnt('D2_holds')} of {len(done)}.", "",
              "## Arm A: four file-level splits", "",
              "| split | rows | initialisations | S1: causal variant over causal GP | S2: over the twin | twin over causal GP | "
              "spread of the skill over GP between initialisations | over hold | over PCHIP | over the offline GP |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in done:
            o = A[str(s)]
            L.append(f"| {s} | {o['rows']:,} | {', '.join(map(str, o['initialisations']))} | {s3(o['S1_causal_over_gp'])} | "
                     f"{s3(o['S2_causal_over_twin'])} | {s3(o['twin_over_gp'])} | {o['spread_causal_over_gp']:.3f} | "
                     f"{s3(o['causal_over']['persistence'])} | {s3(o['causal_over']['pchip'])} | {s3(o['causal_over']['gp'])} |")
        L += ["", "Ablations of the fast-channel groups (initialisation = split seed for every network; 2,000 resamples of discharges):", "",
              "| split | D1: ECEI-only over twin | D2: BES-only over twin | no-Mirnov over twin | full over twin | full over ECEI-only | "
              "full over BES-only | full over no-Mirnov |", "|---|---|---|---|---|---|---|---|"]
        for s in done:
            o = A[str(s)]
            L.append(f"| {s} | {s3(o['D1_ecei_over_twin'])} | {s3(o['D2_bes_over_twin'])} | {s3(o['nomirnov_over_twin'])} | "
                     f"{s3(o['full_over_twin_same_initialisation'])} | {s3(o['full_over_ecei'])} | {s3(o['full_over_bes'])} | "
                     f"{s3(o['full_over_nomirnov'])} |")
        L += ["", "| split | over the causal GP: ECEI-only | BES-only | no-Mirnov | skill of each initialisation over the causal GP: causal variant / twin | "
              "median absolute error (eV): hold / causal variant / twin / ECEI-only / BES-only / no-Mirnov | epochs run (best) |", "|---|---|---|---|---|---|---|"]
        for s in done:
            o = A[str(s)]
            m = o["median_abs_error_eV"]
            pi = "; ".join(f"i{i}: {x['causal_over_gp']:+.3f} / {x['twin_over_gp']:+.3f}" for i, x in o["per_initialisation"].items())
            ep = ", ".join(f"{k} {v[0]} ({v[1]})" for k, v in o["epochs_run"].items())
            L.append(f"| {s} | {s3(o['ablation_over_gp']['v2e'])} | {s3(o['ablation_over_gp']['v2b'])} | {s3(o['ablation_over_gp']['v2n'])} | {pi} | "
                     f"{m['hold']:.1f} / {m['causal variant']:.1f} / {m['twin']:.1f} / {m['v2e']:.1f} / {m['v2b']:.1f} / {m['v2n']:.1f} | {ep} |")
    L += ["", "## Arm P: pooled cross-validation (repeats r2 to r9, five folds each)", ""]
    if not P.get("complete"):
        L.append(f"Not read yet: {P.get('runs_done', 0)} of {P.get('runs_total', 80)} runs are complete.")
    else:
        L += [f"- **P1** (causal variant over the causal GP): {s3(P['P1_causal_over_gp'])} -> {'holds' if P['P1_holds'] else 'does not hold'}.",
              f"- **P2** (causal variant over the history-only twin): {s3(P['P2_causal_over_twin'])} -> {'holds' if P['P2_holds'] else 'does not hold'}.", "",
              "| reference | skill of the causal variant (two-level interval: repeat, then discharges) |", "|---|---|"]
        for q, lab in REFS.items():
            L.append(f"| {lab} | {s3(P['causal_over'][q])} |")
        L.append(f"| history-only twin | {s3(P['P2_causal_over_twin'])} |")
        if "causal_over_whole_discharge_variant_cap30" in P:
            L.append(f"| whole-discharge variant of the manuscript (cap 30, same rows) | {s3(P['causal_over_whole_discharge_variant_cap30'])} |")
            L.append(f"| (that variant over the causal GP on these rows) | {s3(P['whole_discharge_variant_cap30_over_gp'])} |")
        L += ["", f"Twin over the causal GP: {s3(P['twin_over_gp'])}. Fold-level interval with the Nadeau-Bengio correction (40 folds): "
              f"over the causal GP {s3(P['P1_fold_level_nadeau_bengio'])}, over the twin {s3(P['P2_fold_level_nadeau_bengio'])}.",
              "Per repeat, over the causal GP: " + ", ".join(f"{v:+.3f}" for v in P["P1_per_repeat"].values()) +
              "; over the twin: " + ", ".join(f"{v:+.3f}" for v in P["P2_per_repeat"].values()) + ".",
              f"Median absolute error (eV, mean over repeats): hold {P['median_abs_error_eV']['hold']:.1f}, causal variant "
              f"{P['median_abs_error_eV']['causal variant']:.1f}, twin {P['median_abs_error_eV']['twin']:.1f}."]
    return "\n".join(L) + "\n"


def main():
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B39.md",
           "A": arm_a(np.random.default_rng(SEED)), "P": arm_p()}
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print("written", OUT_MD, "| splits:", list(res["A"]), "| pooled complete:", res["P"].get("complete"))


if __name__ == "__main__":
    main()
