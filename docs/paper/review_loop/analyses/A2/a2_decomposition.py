"""Review-loop round 1, analysis A2 (2026-11-13). CPU, existing artifacts only, real data.

(a) Decomposition of the backbone's T_i margin over PCHIP into the share a CES-only causal
    smoother already gives (EWMA of past observed T_i, alpha* tuned on VAL files, no fitted
    parameter on TEST) and the increment above it. Reuses the frozen-row path of
    ces_prediction/experiments/b28_equation/stage1_score.py (re-derives and verifies the npz
    row population), then adds the shot-cluster skill of the backbone vs EWMA(alpha*).
(b) The stage-4 20-parameter surrogate (b28h, data/.b28_b28h_{pop}_s{s}) as a standalone
    estimator: shot-cluster skill vs causal GP, PCHIP, persistence, offline GP, and RMSE.
Bootstrap: B = 10,000, seed 12345 (fresh generator per comparison). ASCII prints only.
Run from the repo root:  py docs/paper/review_loop/analyses/A2/a2_decomposition.py [--part a|b|all]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
DATA = REPO / "data"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "ces_prediction" / "experiments" / "b28_equation"))
import stage1_score as S1  # noqa: E402  (pins env per split via S1.pin_env)

SEEDS = (42, 1, 7, 123)
REFS = ("gp_causal", "pchip", "persistence", "gp")
B, SEED = 10_000, 12345


def skill(shot, se_a, se_ref):
    return S1.boot_skill(shot, se_a.astype(np.float64), se_ref.astype(np.float64), np.random.default_rng(SEED), B)


def part_a(pop, seed):
    p = S1.POPS[pop]
    run = DATA / p["run"].format(s=seed)
    cm = DATA / p["control_metrics"].format(s=seed) / "metrics.json"
    man_p = DATA / p["split"].format(s=seed) / "split_manifest.json"
    for f in (run / "comparison_errors_test.npz", run / "metrics.json", cm, man_p):
        if not f.exists():
            raise SystemExit(f"FATAL: missing {f}")
    S1.pin_env(p["cut"])
    t0 = time.time()
    npz = np.load(run / "comparison_errors_test.npz")
    rows = S1.frozen_rows(cm, man_p, seed, npz)
    from seq_data import load_grid_files  # noqa: E402
    from evaluate import _load_stats  # noqa: E402
    my_stats = _load_stats(json.loads((run / "metrics.json").read_text(encoding="utf-8")))
    grid, dims = load_grid_files(DATA, drop_stuck_targets=True, ti_spike_cut_ev=float(p["cut"]))
    man = json.loads(man_p.read_text(encoding="utf-8"))
    val_files = [f for f in man["val_files"] if f in grid]
    test_files = sorted(set(rows["file"]))
    tables = {f: S1.ewma_table(grid[f], dims, my_stats, p["cut"]) for f in val_files + test_files}
    vkeys = [(f, t) for f in val_files for t, r in tables[f].items() if r["obs"] and r["has"]]
    Ev, Pv, _, _, Yv, _ = S1.stack(tables, vkeys)
    mse_pv = ((Yv - Pv) ** 2).mean()
    vs = [float(1 - ((Yv - Ev[:, j]) ** 2).mean() / mse_pv) for j in range(len(S1.ALPHAS))]
    j = int(np.argmax(vs)); alpha = S1.ALPHAS[j]
    tkeys = list(zip(rows["file"], rows["time"].tolist()))
    Et, _, _, _, Yt, _ = S1.stack(tables, tkeys)
    if float(np.max(np.abs(Yt - rows["y"]))) > 1e-2:
        raise SystemExit("FATAL: grid y does not match frozen y_true")
    y = rows["y"].astype(np.float64)
    shot = rows["shot"]
    se = {"model": npz["CES_TI_se_model"].astype(np.float64), "ewma": (Et[:, j] - y) ** 2}
    for r in REFS:
        se[r] = npz[f"CES_TI_se_{r}"].astype(np.float64)
    rec = {"pop": pop, "seed": seed, "run": run.name, "n_rows": int(len(y)), "alpha_star": alpha,
           "row_check": rows["check"],
           "backbone_vs_ewma": skill(shot, se["model"], se["ewma"]),
           "ewma_vs_pchip": skill(shot, se["ewma"], se["pchip"]),
           "backbone_vs_pchip": skill(shot, se["model"], se["pchip"]),
           "backbone_vs_gp_causal": skill(shot, se["model"], se["gp_causal"]),
           "ewma_vs_gp_causal": skill(shot, se["ewma"], se["gp_causal"]),
           "rmse_eV": {k: float(np.sqrt(v.mean())) for k, v in se.items()},
           "minutes": round((time.time() - t0) / 60, 2)}
    # additive decomposition on the PCHIP SSE scale: 1 - M/P = (1 - E/P) + (E - M)/P
    sp = se["pchip"].sum()
    rec["decomp_vs_pchip"] = {"total": float(1 - se["model"].sum() / sp),
                              "denoising_share_ewma": float(1 - se["ewma"].sum() / sp),
                              "increment_above_ewma": float((se["ewma"].sum() - se["model"].sum()) / sp)}
    b = rec["backbone_vs_ewma"]
    print(f"[A2a] {pop} s{seed}: n={len(y)} alpha*={alpha} bb/ewma {b['skill_point']:+.3f} "
          f"[{b['skill_ci95'][0]:+.3f},{b['skill_ci95'][1]:+.3f}] ewma/pchip {rec['ewma_vs_pchip']['skill_point']:+.3f} "
          f"bb/pchip {rec['backbone_vs_pchip']['skill_point']:+.3f} ({rec['minutes']} min)", flush=True)
    return rec


def part_b(pop, seed):
    sur = np.load(DATA / f".b28_b28h_{pop}_s{seed}" / "comparison_errors_test.npz")
    bb = np.load(DATA / S1.POPS[pop]["run"].format(s=seed) / "comparison_errors_test.npz")
    shot = sur["CES_TI_shot"]
    same_rows = bool(np.array_equal(shot, bb["CES_TI_shot"]) and np.allclose(sur["CES_TI_dt_ms"], bb["CES_TI_dt_ms"]))
    se_s = sur["CES_TI_se_model"].astype(np.float64)
    rec = {"pop": pop, "seed": seed, "n_rows": int(len(shot)), "same_rows_as_backbone": same_rows,
           "rmse_eV": {"surrogate_b28h": float(np.sqrt(se_s.mean()))},
           "surrogate_vs": {r: skill(shot, se_s, sur[f"CES_TI_se_{r}"]) for r in REFS}}
    for r in REFS:
        rec["rmse_eV"][r] = float(np.sqrt(sur[f"CES_TI_se_{r}"].astype(np.float64).mean()))
    if same_rows:
        rec["rmse_eV"]["backbone"] = float(np.sqrt(bb["CES_TI_se_model"].astype(np.float64).mean()))
        rec["surrogate_vs"]["backbone"] = skill(shot, se_s, bb["CES_TI_se_model"])
    s = rec["surrogate_vs"]
    print(f"[A2b] {pop} s{seed}: same_rows={same_rows} rmse {rec['rmse_eV']['surrogate_b28h']:.1f} | "
          + " ".join(f"{r} {s[r]['skill_point']:+.3f}[{s[r]['skill_ci95'][0]:+.3f},{s[r]['skill_ci95'][1]:+.3f}]" for r in s), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="all")
    a = ap.parse_args()
    out = HERE / "a2_decomposition.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    res["protocol"] = {"B": B, "seed_boot": SEED, "skill": "1 - sum SE_arm / sum SE_ref, shot-cluster bootstrap",
                       "ewma": "alpha* in {0.25,0.35,0.45} chosen on VAL files (stage1_score.py)"}
    if a.part in ("b", "all"):
        res["b_surrogate"] = [part_b(p, s) for p in ("cut", "incl") for s in SEEDS]
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a.part in ("a", "all"):
        res["a_decomposition"] = []
        for p in ("cut", "incl"):
            for s in SEEDS:
                res["a_decomposition"].append(part_a(p, s))
                out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[A2] wrote {out}")


if __name__ == "__main__":
    main()
