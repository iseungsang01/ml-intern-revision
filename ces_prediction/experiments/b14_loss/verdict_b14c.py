"""B.14 sec. 10 verdict (PREREGISTRATION_B14.md sec. 10) over finished artifacts -- no training.

Row A: cut population, checkpoint on the arm's own val objective (`.b14s_{arm}_cut_s{s}`, cap 60).
Row B: inclusive population, same rule, epoch cap 100 (`.b14c_{arm}_incl_s{s}`).
Both against the frozen MSE backbone. M1 (median-AE skill) from rescore_b13 (`{arm}_vs_model`,
shot-cluster bootstrap); MSE guard = paired_vs_control.json `b_better` (control significantly
better), the same guard sec. 8bh used. Median / persistence = 1 - M1 skill vs persistence.

Inputs:  data/.b14s_cut_rescore.json, data/.b14c_rescore.json, the arm dirs.
Output:  data/.b14c_verdict.json.   Usage (repo root):  py ces_prediction/experiments/b14_loss/verdict_b14c.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
SEEDS = ("42", "1", "7", "123")
TARGETS = ("CES_TI", "CES_VT")
ROWS = {
    "A_cut_select_loss": {"pop": "cut", "rescore": ".b14s_cut_rescore.json", "tag": "_s", "dir": ".b14s_{arm}_cut_s{s}"},
    "B_incl_cap100": {"pop": "incl", "rescore": ".b14c_rescore.json", "tag": "_c", "dir": ".b14c_{arm}_incl_s{s}"},
}
ARMS = ("huber", "tnll")


def passes(cell):
    """Registered rule (sec. 9 case 1, reused in sec. 10): MSE significant deficits 0/4 and M1 CI > 0 in >= 3/4."""
    return cell["mse_sig_deficits"] == 0 and cell["m1_splits_pos"] >= 3


def joint(a_ok, b_ok):
    if a_ok and b_ok:
        return "unconditional: the robust rung holds in both populations under one selection rule (val own objective)"
    if a_ok:
        return "cut-conditional: loss selection keeps the cut rung; the inclusive guard still fails at cap 100"
    if b_ok:
        return "population-specific rule: inclusive passes at cap 100 but loss selection breaks the cut rung"
    return "cut-conditional under MSE selection (sec. 8ay unchanged); loss selection does not carry the cut rung"


def cell(row, arm, t):
    r = ROWS[row]
    res = json.loads((DATA / r["rescore"]).read_text(encoding="utf-8"))["results"][r["pop"]]
    key = f"{arm}{r['tag']}"
    out = {"per_split": {}, "m1_splits_pos": 0, "mse_sig_deficits": 0}
    for s in SEEDS:
        ps = res["per_seed"][s][t]
        m1 = ps[f"{key}_vs_model"]["all"]["m1_medae"]
        vp = ps[f"{key}_vs_persistence"]["all"]["m1_medae"]
        d = DATA / r["dir"].format(arm=arm, s=s)
        pv = json.loads((d / "paired_vs_control.json").read_text(encoding="utf-8"))["targets"][t]
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        out["per_split"][s] = {"m1": m1["skill_point"], "m1_ci": m1["skill_ci95"],
                               "mse_paired": pv["skill_point"], "mse_ci": pv["skill_ci95"],
                               "mse_control_better": bool(pv["b_better"]),
                               "median_over_persistence": 1.0 - vp["skill_point"],
                               "epochs": [m["epochs_run"], m["best_epoch"]]}
        out["m1_splits_pos"] += int(m1["skill_ci95"][0] > 0)
        out["mse_sig_deficits"] += int(bool(pv["b_better"]))
    po = res["pooled"][t]
    out["m1_pooled"] = po[f"{key}_vs_model"]["all"]["m1_medae"]
    out["mse_pooled"] = po[f"{key}_vs_model"]["all"]["mse"]
    out["median_over_persistence_pooled"] = 1.0 - po[f"{key}_vs_persistence"]["all"]["m1_medae"]["skill_point"]
    out["pass"] = passes(out)
    return out


def main():
    v = {}
    for row in ROWS:
        v[row] = {arm: {t: cell(row, arm, t) for t in TARGETS} for arm in ARMS}
        for arm in ARMS:
            for t in TARGETS:
                c = v[row][arm][t]
                ps = c["per_split"]
                print(f"[b14c] {row} {arm:5s} {t}: M1 " + " ".join(f"{ps[s]['m1']:+.3f}" for s in SEEDS)
                      + f" (CI>0 {c['m1_splits_pos']}/4, pooled {c['m1_pooled']['skill_point']:+.3f})"
                      + " | MSE " + " ".join(f"{ps[s]['mse_paired']:+.3f}" for s in SEEDS)
                      + f" (sig worse {c['mse_sig_deficits']}/4)"
                      + f" | med/pers {c['median_over_persistence_pooled']:.3f}"
                      + " | ep " + " ".join(f"{ps[s]['epochs'][0]}/{ps[s]['epochs'][1]}" for s in SEEDS)
                      + f" | pass {c['pass']}", flush=True)
    a_ok = v["A_cut_select_loss"]["huber"]["CES_TI"]["pass"]
    b_ok = v["B_incl_cap100"]["huber"]["CES_TI"]["pass"]
    v["rules"] = {"A_huber_ti_pass": a_ok, "B_huber_ti_pass": b_ok, "verdict": joint(a_ok, b_ok)}
    print(json.dumps(v["rules"], indent=1))
    (DATA / ".b14c_verdict.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
    print(f"wrote {DATA / '.b14c_verdict.json'}")


if __name__ == "__main__":
    main()
