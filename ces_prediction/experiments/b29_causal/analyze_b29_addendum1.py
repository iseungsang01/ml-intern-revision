"""B.29 addendum 1 analysis -- PREREGISTRATION_B29_ADDENDUM1.md (epoch cap 30 -> 80, arm A and
the backbone, four cut splits).

CPU only; reads the JSON the registered tools wrote under `data/.b29a1_*`, the first arm A runs
(`data/.b29_v2h_cut_s*`) and the frozen backbone. Nothing is recomputed, nothing under `data/`
is written. Before any statistic the cap-80 backbone's `metrics.json` is compared with the
frozen backbone's on every field that describes the model and the training data (fatal on a
mismatch): the two runs may differ in the epoch cap only.

Output: docs/paper/b29_addendum1.json and docs/paper/b29_addendum1.md
        (point estimate [95 % interval], * = the interval excludes 0).

Usage (repo root):  py ces_prediction/experiments/b29_causal/analyze_b29_addendum1.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_b29 import _cell, _load, fmt  # noqa: E402
from run_b29 import ADD1_EPOCH_CAP, DATA, REPO_ROOT, SEEDS, add1_dir, arm_dir, controls  # noqa: E402

OUT_JSON = REPO_ROOT / "docs" / "paper" / "b29_addendum1.json"
OUT_MD = REPO_ROOT / "docs" / "paper" / "b29_addendum1.md"
TARGETS = ("CES_TI", "CES_VT")
# fields of metrics.json that must be equal between the frozen backbone and its cap-80 retrain
IDENTITY_FIELDS = ("seq_model", "n_params", "per_shot_input_norm", "feature_dims", "normalization",
                   "train_blocks", "train_labels", "drop_stuck_targets", "fixed_epochs", "seed",
                   "init_seed", "loss")


def identity_mismatches(frozen_metrics, retrain_metrics):
    """Names of the identity fields on which the two metrics.json records differ."""
    return [k for k in IDENTITY_FIELDS if frozen_metrics.get(k) != retrain_metrics.get(k)]


def read_split(s):
    arm80, bb80 = add1_dir("v2h", s), add1_dir("v2", s)
    backbone = controls("split", "cut", s)[0]
    first = arm_dir("v2h", "split", "cut", s)
    need = {
        "i": arm80 / "paired_backbone_vs_arm.json",          # frozen backbone vs cap-80 arm A
        "ii": bb80 / "paired_vs_cap80_arm.json",              # cap-80 backbone vs cap-80 arm A
        "arm80_vs_first": arm80 / "paired_vs_first_arm.json",
        "bb80_vs_frozen": bb80 / "paired_vs_backbone.json",
        "first": first / "paired_backbone_vs_arm.json",       # registered primary (cap 30)
        "arm80_bs": arm80 / "bootstrap_summary.json",
        "bb80_bs": bb80 / "bootstrap_summary.json",
        "arm80_m": arm80 / "metrics.json", "bb80_m": bb80 / "metrics.json",
        "first_m": first / "metrics.json", "frozen_m": backbone / "metrics.json",
    }
    if not all(p.exists() for p in need.values()):
        return None
    d = {k: _load(p) for k, p in need.items()}
    bad = identity_mismatches(d["frozen_m"], d["bb80_m"])
    if bad:
        raise SystemExit(f"FATAL: split {s}: the cap-80 backbone differs from the frozen backbone on {bad}; "
                         "the retrain is not the backbone with a longer cap.")
    if d["arm80_m"].get("seq_model") != "v2h" or d["arm80_m"].get("n_params") != d["first_m"].get("n_params"):
        raise SystemExit(f"FATAL: split {s}: the cap-80 arm is not the registered history-only model.")
    cap = int(ADD1_EPOCH_CAP)
    row = {"epochs": {k: {"epochs_run": d[m]["epochs_run"], "best_epoch": d[m]["best_epoch"]}
                      for k, m in (("arm_cap30", "first_m"), ("arm_cap80", "arm80_m"),
                                   ("backbone_frozen", "frozen_m"), ("backbone_cap80", "bb80_m"))},
           "cap80_bound": {"arm": d["arm80_m"]["epochs_run"] >= cap, "backbone": d["bb80_m"]["epochs_run"] >= cap}}
    for t in TARGETS:
        row[t] = {
            "n_rows": d["ii"]["targets"][t]["n_rows"], "n_shots": d["ii"]["targets"][t]["n_shots"],
            "first_backbone_vs_arm": _cell(d["first"]["targets"][t]),
            "i_frozen_backbone_vs_cap80_arm": _cell(d["i"]["targets"][t]),
            "ii_cap80_backbone_vs_cap80_arm": _cell(d["ii"]["targets"][t]),
            "iii_cap80_arm_vs_first_arm": _cell(d["arm80_vs_first"]["targets"][t]),
            "iii_cap80_backbone_vs_frozen": _cell(d["bb80_vs_frozen"]["targets"][t]),
            "cap80_arm_vs_causal_gp": _cell(d["arm80_bs"]["splits"]["test"][t]["gp_causal"]),
            "cap80_backbone_vs_causal_gp": _cell(d["bb80_bs"]["splits"]["test"][t]["gp_causal"]),
        }
    return row


def verdict(rows):
    """Section 4 of the addendum; written only when all four splits are present."""
    if len(rows) != len(SEEDS):
        return {}
    n_i = sum(r["CES_TI"]["i_frozen_backbone_vs_cap80_arm"]["above_0"] for r in rows.values())
    n_ii = sum(r["CES_TI"]["ii_cap80_backbone_vs_cap80_arm"]["above_0"] for r in rows.values())
    v = {"count_i": int(n_i), "count_ii": int(n_ii)}
    v["rule"] = ("R-a: lower bound of (ii) > 0 on >= 3/4 cut splits -- the cut fast-channel increment is quoted "
                 "as final with the (ii) numbers" if n_ii >= 3 else
                 "R-b: lower bound of (ii) > 0 on fewer than 3/4 cut splits -- the cut increment is sensitive to "
                 "training length; H29-1 re-read as failed (retitle to noise-limited estimation of the CES past)")
    if (n_i >= 3) != (n_ii >= 3):
        v["R-c"] = f"(i) {n_i}/4 and (ii) {n_ii}/4 disagree on the 3-of-4 count; (ii) decides"
    bound = [f"{who} s{s}" for s, r in rows.items() for who, b in r["cap80_bound"].items() if b]
    v["R-d"] = ("cap 80 reached by: " + ", ".join(bound) + " (no further extension)") if bound else "no run reached cap 80"
    v["guard_vrot_ii_above_0"] = int(sum(r["CES_VT"]["ii_cap80_backbone_vs_cap80_arm"]["above_0"] for r in rows.values()))
    return v


def markdown(rows, v):
    L = ["# B.29 addendum 1 -- epoch cap 30 -> 80, arm A and the backbone, cut splits", "",
         f"Generated by `ces_prediction/experiments/b29_causal/analyze_b29_addendum1.py` from {len(rows)}/{len(SEEDS)} "
         "splits. Rules: `ces_prediction/experiments/PREREGISTRATION_B29_ADDENDUM1.md` sec. 4. Every cell is a skill "
         "1 - SSE_X/SSE_Y on the paired TEST rows with its shot-clustered 95 % interval; `*` = the interval "
         "excludes 0. \"X vs Y\" > 0 means X has the lower error.", "",
         "## T_i", "",
         "| split | cap 30 (registered): backbone vs arm A | (i) frozen backbone vs cap-80 arm A | "
         "(ii) cap-80 backbone vs cap-80 arm A | cap-80 arm A vs first arm A | cap-80 backbone vs frozen backbone | "
         "cap-80 arm A vs causal GP |", "|---|---|---|---|---|---|---|"]
    for s, r in rows.items():
        t = r["CES_TI"]
        L.append(f"| {s} | {fmt(t['first_backbone_vs_arm'])} | {fmt(t['i_frozen_backbone_vs_cap80_arm'])} | "
                 f"{fmt(t['ii_cap80_backbone_vs_cap80_arm'])} | {fmt(t['iii_cap80_arm_vs_first_arm'])} | "
                 f"{fmt(t['iii_cap80_backbone_vs_frozen'])} | {fmt(t['cap80_arm_vs_causal_gp'])} |")
    L += ["", "## V_rot (guard) and training length", "",
          "| split | (ii) V_rot cap-80 backbone vs cap-80 arm A | arm A cap 30: epochs (best) | arm A cap 80 | "
          "backbone frozen | backbone cap 80 |", "|---|---|---|---|---|---|"]
    for s, r in rows.items():
        e = r["epochs"]
        ep = [f"{e[k]['epochs_run']} ({e[k]['best_epoch']})" for k in ("arm_cap30", "arm_cap80", "backbone_frozen", "backbone_cap80")]
        L.append(f"| {s} | {fmt(r['CES_VT']['ii_cap80_backbone_vs_cap80_arm'])} | " + " | ".join(ep) + " |")
    L += ["", "## Verdict (written only when all four splits are present)", ""]
    L += [f"- **{k}**: {x}" for k, x in v.items()] or ["- none yet"]
    return "\n".join(L) + "\n"


def main():
    rows = {}
    for s in SEEDS:
        row = read_split(s)
        if row is not None:
            rows[str(s)] = row
    if not rows:
        raise SystemExit(f"no finished addendum-1 split under {DATA} (looked for .b29a1_*); nothing to analyze.")
    v = verdict(rows)
    result = {"registration": "ces_prediction/experiments/PREREGISTRATION_B29_ADDENDUM1.md",
              "epoch_cap": int(ADD1_EPOCH_CAP), "splits_done": len(rows), "splits_total": len(SEEDS),
              "identity_fields_checked": list(IDENTITY_FIELDS),
              "statistic": "skill = 1 - SSE_X/SSE_Y, shot-clustered paired bootstrap (paired_model_compare.py)",
              "splits": rows, "verdict": v}
    OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(rows, v), encoding="utf-8")
    print(f"[b29a1] {len(rows)}/{len(SEEDS)} splits analyzed")
    for k, x in v.items():
        print(f"[b29a1] {k}: {x}")
    print(f"[b29a1] wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
