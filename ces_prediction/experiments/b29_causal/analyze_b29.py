"""B.29 analysis -- applies the PREREGISTRATION_B29.md sec. 3 rules to whatever runs are finished.

CPU only; reads the per-run JSON the registered tools wrote (`bootstrap_summary.json`,
`paired_vs_backbone.json`, `paired_backbone_vs_arm.json`, `metrics.json`) plus the frozen
backbone's `bootstrap_summary.json`. Nothing is recomputed and nothing under `data/` is written.

Output: docs/paper/b29_causal.json (every number + the rule counts) and
        docs/paper/b29_causal.md   (tables; point estimate [95 % interval], * = interval excludes 0).

A rule's verdict is written only when all four cells of its family are present; a partial batch
gets counts and `"complete": false`.

Usage (repo root):  py ces_prediction/experiments/b29_causal/analyze_b29.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_b29 import ARMS, DATA, POPS, REPO_ROOT, SEEDS, arm_dir, controls  # noqa: E402

OUT_JSON = REPO_ROOT / "docs" / "paper" / "b29_causal.json"
OUT_MD = REPO_ROOT / "docs" / "paper" / "b29_causal.md"
TARGETS = ("CES_TI", "CES_VT")
REFS = ("pchip", "persistence", "gp_causal", "gp")
ARM_LABEL = {"v2h": "A history-only seq_v2", "norm0": "B train-file standardization",
             "expand": "C expanding-window standardization", "mlp22": "D memoryless MLP"}


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _cell(res):
    lo, hi = res["skill_ci95"]
    return {"skill": res["skill_point"], "ci95": [lo, hi], "above_0": bool(lo > 0.0), "below_0": bool(hi < 0.0)}


def read_run(arm, family, pop, s):
    out = arm_dir(arm, family, pop, s)
    backbone = controls(family, pop, s)[0]
    need = [out / "bootstrap_summary.json", out / "paired_vs_backbone.json",
            out / "paired_backbone_vs_arm.json", out / "metrics.json", backbone / "bootstrap_summary.json"]
    if not all(p.exists() for p in need):
        return None
    bs = _load(need[0])["splits"]["test"]
    pa = _load(need[1])["targets"]
    pb = _load(need[2])["targets"]
    m = _load(need[3])
    cb = _load(need[4])["splits"]["test"]
    row = {"dir": out.name, "backbone_dir": backbone.name, "epochs_run": m["epochs_run"],
           "best_epoch": m["best_epoch"], "n_params": m["n_params"],
           "per_shot_input_norm": m.get("per_shot_input_norm"),
           "per_shot_norm_expanding": m.get("per_shot_norm_expanding"), "seq_model": m.get("seq_model")}
    for t in TARGETS:
        row[t] = {
            "n_rows": pa[t]["n_rows"], "n_shots": pa[t]["n_shots"],
            "arm_vs": {r: _cell(bs[t][r]) for r in REFS if r in bs[t]},
            "backbone_vs": {r: _cell(cb[t][r]) for r in REFS if r in cb[t]},
            "arm_vs_backbone": _cell(pa[t]),      # > 0: the arm's error is lower
            "backbone_vs_arm": _cell(pb[t]),      # > 0: the backbone's error is lower
        }
    return row


def count(rows, fn):
    return int(sum(bool(fn(r)) for r in rows.values()))


def rules(cells):
    """cells[arm][family][pop] = {seed: row}. Returns the sec. 3 rule read-outs."""
    def fam(arm, family, pop):
        return cells.get(arm, {}).get(family, {}).get(pop, {})

    def rule(arm, family, pop, fn, values):
        rows = fam(arm, family, pop)
        return {"n_runs": len(rows), "complete": len(rows) == len(SEEDS), "count": count(rows, fn),
                "per_seed": {s: values(r) for s, r in rows.items()}}

    def ti(key, ref=None):
        if ref is None:
            return (lambda r: r["CES_TI"][key]["above_0"]), (lambda r: [r["CES_TI"][key]["skill"], *r["CES_TI"][key]["ci95"]])
        return (lambda r: r["CES_TI"][key][ref]["above_0"]), (lambda r: [r["CES_TI"][key][ref]["skill"], *r["CES_TI"][key][ref]["ci95"]])

    v = {}
    for pop in POPS:
        v[f"H29-1 {pop}: backbone vs history-only, T_i, lower bound > 0"] = rule("v2h", "split", pop, *ti("backbone_vs_arm"))
        v[f"H29-1b {pop}: history-only vs causal GP, T_i, lower bound > 0"] = rule("v2h", "split", pop, *ti("arm_vs", "gp_causal"))
        for arm, h in (("norm0", "H29-2"), ("expand", "H29-3")):
            v[f"{h} {pop}: {arm} vs causal GP, T_i, lower bound > 0 (file-level)"] = rule(arm, "split", pop, *ti("arm_vs", "gp_causal"))
            v[f"{h}c {pop}: {arm} vs causal GP, T_i, lower bound > 0 (temporal split)"] = rule(arm, "camp", pop, *ti("arm_vs", "gp_causal"))
            v[f"{h} cost {pop}: backbone vs {arm}, T_i, lower bound > 0 (file-level)"] = rule(arm, "split", pop, *ti("backbone_vs_arm"))
            v[f"{h}c cost {pop}: backbone vs {arm}, T_i, lower bound > 0 (temporal split)"] = rule(arm, "camp", pop, *ti("backbone_vs_arm"))
        v[f"H29-4 {pop}: backbone vs memoryless MLP, T_i, lower bound > 0"] = rule("mlp22", "split", pop, *ti("backbone_vs_arm"))
        for arm in ARMS:
            for family in ARMS[arm]["families"]:
                v[f"guard {pop}: backbone vs {arm}, V_rot, lower bound > 0 ({family})"] = rule(
                    arm, family, pop, lambda r: r["CES_VT"]["backbone_vs_arm"]["above_0"],
                    lambda r: [r["CES_VT"]["backbone_vs_arm"]["skill"], *r["CES_VT"]["backbone_vs_arm"]["ci95"]])

    def passed(key):
        return v[key]["complete"] and v[key]["count"] >= 3

    def complete(*keys):
        return all(v[k]["complete"] for k in keys)

    verdict = {}
    k1 = "H29-1 cut: backbone vs history-only, T_i, lower bound > 0"
    if complete(k1):
        verdict["H29-1"] = ("title kept: fast-channel increment interval lower bound > 0 on >= 3/4 cut splits"
                            if passed(k1) else
                            "retitle to noise-limited estimation of the CES past: fewer than 3/4 cut splits")
        ki = "H29-1 incl: backbone vs history-only, T_i, lower bound > 0"
        if passed(k1) and complete(ki) and not passed(ki):
            verdict["H29-1"] += "; increment stated as conditional on the cut (inclusive < 3/4)"
    kb = "H29-2 cut: norm0 vs causal GP, T_i, lower bound > 0 (file-level)"
    kbc = "H29-2c cut: norm0 vs causal GP, T_i, lower bound > 0 (temporal split)"
    kc = "H29-3 cut: expand vs causal GP, T_i, lower bound > 0 (file-level)"
    kcc = "H29-3c cut: expand vs causal GP, T_i, lower bound > 0 (temporal split)"
    for name, key in (("H29-2", kb), ("H29-2c", kbc), ("H29-3", kc), ("H29-3c", kcc)):
        if complete(key):
            verdict[name] = "supported (>= 3/4)" if passed(key) else "not supported (< 3/4)"
    if complete(kb, kbc, kc, kcc):
        if passed(kb) and passed(kbc):
            named = "B (train-file standardization): passes the file-level and the temporal rule"
        elif passed(kc) and passed(kcc):
            named = "C (expanding-window standardization): passes the file-level and the temporal rule"
        elif passed(kb):
            named = "B (train-file standardization): file-level rule only, temporal-split failure to be stated"
        elif passed(kc):
            named = "C (expanding-window standardization): file-level rule only, temporal-split failure to be stated"
        else:
            named = "none: 'causal' stays out of the title and abstract"
        verdict["causal estimator named"] = named
    k4 = "H29-4 cut: backbone vs memoryless MLP, T_i, lower bound > 0"
    if complete(k4):
        verdict["H29-4"] = ("recurrent state adds to a memoryless regression (>= 3/4 cut)" if passed(k4)
                            else "memoryless row not separated from the backbone (< 3/4 cut)")
    return v, verdict


def fmt(c):
    star = "*" if (c["above_0"] or c["below_0"]) else ""
    return f"{c['skill']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]{star}"


def markdown(cells, v, verdict, n_done, n_total):
    L = ["# B.29 -- fast-channel increment on the main model and causal standardization", "",
         f"Generated by `ces_prediction/experiments/b29_causal/analyze_b29.py` from {n_done}/{n_total} finished runs. "
         "Rules: `ces_prediction/experiments/PREREGISTRATION_B29.md` sec. 3. Every cell is a skill "
         "1 - SSE_X/SSE_Y on the paired TEST rows with its shot-clustered 95 % interval (10,000 resamples); "
         "`*` = the interval excludes 0. \"backbone vs arm\" > 0 means the per-discharge-standardized "
         "seq_v2 backbone has the lower error.", ""]
    for arm in ARMS:
        if arm not in cells:
            continue
        for family in ARMS[arm]["families"]:
            for pop in POPS:
                rows = cells[arm].get(family, {}).get(pop, {})
                if not rows:
                    continue
                fam = "file-level splits" if family == "split" else "temporal split (seed = initialisation)"
                L += [f"## {ARM_LABEL[arm]} -- {pop}, {fam}", "",
                      "| seed | T_i arm vs PCHIP | T_i arm vs causal GP | T_i arm vs persistence | T_i backbone vs causal GP | "
                      "T_i backbone vs arm | V_rot backbone vs arm | epochs (best) |",
                      "|---|---|---|---|---|---|---|---|"]
                for s, r in rows.items():
                    ti, vt = r["CES_TI"], r["CES_VT"]
                    L.append(f"| {s} | {fmt(ti['arm_vs']['pchip'])} | {fmt(ti['arm_vs']['gp_causal'])} | "
                             f"{fmt(ti['arm_vs']['persistence'])} | {fmt(ti['backbone_vs']['gp_causal'])} | "
                             f"{fmt(ti['backbone_vs_arm'])} | {fmt(vt['backbone_vs_arm'])} | "
                             f"{r['epochs_run']} ({r['best_epoch']}) |")
                L.append("")
    L += ["## Registered rule read-outs", "", "| rule | splits with the condition | complete |", "|---|---|---|"]
    for k, r in v.items():
        L.append(f"| {k} | {r['count']}/{r['n_runs']} | {'yes' if r['complete'] else 'no'} |")
    L += ["", "## Verdicts (written only for complete families)", ""]
    L += [f"- **{k}**: {x}" for k, x in verdict.items()] or ["- none yet"]
    return "\n".join(L) + "\n"


def main():
    cells, n_done, n_total = {}, 0, 0
    for arm in ARMS:
        for family in ARMS[arm]["families"]:
            for pop in POPS:
                for s in SEEDS:
                    n_total += 1
                    row = read_run(arm, family, pop, s)
                    if row is None:
                        continue
                    n_done += 1
                    cells.setdefault(arm, {}).setdefault(family, {}).setdefault(pop, {})[str(s)] = row
    if not n_done:
        raise SystemExit(f"no finished B.29 run under {DATA} (looked for .b29_*); nothing to analyze.")
    v, verdict = rules(cells)
    result = {"registration": "ces_prediction/experiments/PREREGISTRATION_B29.md",
              "runs_done": n_done, "runs_total": n_total,
              "statistic": "skill = 1 - SSE_X/SSE_Y, shot-clustered paired bootstrap, B = 10000, seed 12345",
              "arms": {a: {k: ARMS[a][k] for k in ("model", "per_shot", "expanding")} for a in ARMS},
              "cells": cells, "rules": v, "verdict": verdict}
    OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(cells, v, verdict, n_done, n_total), encoding="utf-8")
    print(f"[b29] {n_done}/{n_total} runs analyzed")
    for k, x in verdict.items():
        print(f"[b29] {k}: {x}")
    print(f"[b29] wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
