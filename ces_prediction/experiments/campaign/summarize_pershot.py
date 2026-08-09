"""§8s: does per-shot input standardization repair the campaign-shift transfer loss?

§8n measured why the strictly-temporal (campaign) split kills the offline advantage: between
campaigns the fast diagnostics drift 1.22 sigma while the quantity being predicted drifts
0.115, so the input distribution slides out from under a model trained on earlier discharges.
The repair §8n named -- and the only named repair for Q10 -- is to standardize the fast
diagnostics within each discharge. This scores that arm against §8n's own runs.

Pairing: both arms use the SAME fixed temporal manifest, caps, treatment
(`CES_DROP_STUCK_TARGETS=1`) and architecture; the single controlled variable is
`CES_PER_SHOT_NORM`. Per-shot standardization touches the model's INPUTS only, so the
targets, the keep mask and every baseline are untouched -- which
`paired_model_compare.py` verifies row by row before computing anything.

Pre-registered verdict rule (fixed before looking): KEEP_CANDIDATE iff the paired
`CES_TI` skill of per_shot over base is positive on >= 3 of 4 init seeds AND at least one
paired CI excludes 0; REJECT if it is negative on >= 3 of 4.

Usage (repo root):
  py ces_prediction/experiments/campaign/run_campaign_shift.py --arm per_shot
  py ces_prediction/experiments/campaign/summarize_pershot.py
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
PAIRED = CES_DIR / "experiments" / "paired_model_compare.py"
SEEDS = (42, 1, 7, 123)
TARGETS = ("CES_TI", "CES_VT")
OUT_PATH = REPO_ROOT / "data" / ".pershot_summary.json"


def own_metrics(run_dir):
    """That arm's own vs-PCHIP headline on the campaign test split."""
    cm = json.loads((run_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
    bs = json.loads((run_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
    out = {}
    for t in TARGETS:
        pt, b = cm["per_target"][t], bs["splits"]["test"][t]["pchip"]
        out[t] = {"n": pt["n"], "rmse_model": pt["rmse_model"],
                  "rmse_pchip": pt["baselines"]["pchip"]["rmse"],
                  "rmse_persistence": pt["baselines"]["persistence"]["rmse"],
                  "skill_vs_pchip": pt["skill_vs_pchip"],
                  "ci95": b["skill_ci95"], "pass": b["pass"], "n_shots": b["n_shots"]}
    return out


def paired(seed):
    ps_dir = REPO_ROOT / "data" / f".campaign_ps_s{seed}"
    base_dir = REPO_ROOT / "data" / f".campaign_s{seed}"
    out_path = ps_dir / "paired_vs_base.json"
    proc = subprocess.run(
        [sys.executable, str(PAIRED),
         "--a", str(ps_dir / "comparison_errors_test.npz"),
         "--b", str(base_dir / "comparison_errors_test.npz"),
         "--out", str(out_path)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": (proc.stdout + proc.stderr).strip().splitlines()[-1]}
    # paired_model_compare nests the per-target stats under "targets"
    return json.loads(out_path.read_text(encoding="utf-8"))["targets"]


def main():
    rows = []
    for seed in SEEDS:
        ps_dir = REPO_ROOT / "data" / f".campaign_ps_s{seed}"
        base_dir = REPO_ROOT / "data" / f".campaign_s{seed}"
        if not (ps_dir / "bootstrap_summary.json").exists():
            rows.append({"seed": seed, "status": "per_shot run missing"})
            continue
        if not (base_dir / "bootstrap_summary.json").exists():
            rows.append({"seed": seed, "status": "base run missing"})
            continue
        rows.append({"seed": seed, "status": "ok",
                     "base": own_metrics(base_dir), "per_shot": own_metrics(ps_dir),
                     "paired": paired(seed)})

    ok = [r for r in rows if r["status"] == "ok" and "error" not in r["paired"]]
    verdict = {}
    for t in TARGETS:
        pts = [r["paired"][t]["skill_point"] for r in ok if t in r["paired"]]
        favor = sum(1 for r in ok if r["paired"].get(t, {}).get("skill_point", 0) > 0)
        sig_favor = sum(1 for r in ok if r["paired"].get(t, {}).get("pass"))
        verdict[t] = {
            "paired_skill_points": pts,
            "mean": (sum(pts) / len(pts)) if pts else None,
            "favor_seeds": f"{favor}/{len(ok)}",
            "significant_favor_seeds": f"{sig_favor}/{len(ok)}",
        }
    ti = verdict["CES_TI"]
    n_ok = len(ok)
    favor_ti = int(ti["favor_seeds"].split("/")[0]) if n_ok else 0
    sig_ti = int(ti["significant_favor_seeds"].split("/")[0]) if n_ok else 0
    verdict["overall"] = (
        "KEEP_CANDIDATE" if (n_ok == 4 and favor_ti >= 3 and sig_ti >= 1)
        else "REJECT" if (n_ok == 4 and favor_ti <= 1)
        else "INCONCLUSIVE" if n_ok == 4 else "INCOMPLETE")

    summary = {
        "design": ("campaign (strictly temporal) split; single controlled variable "
                   "CES_PER_SHOT_NORM (fast diagnostics z-scored within each shot, "
                   "targets untouched); control = the §8n runs"),
        "verdict_rule": ("KEEP_CANDIDATE iff paired CES_TI skill > 0 on >=3/4 init seeds "
                         "and >=1 paired CI excludes 0"),
        "runs": rows, "verdict": verdict,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}\n")

    for t in TARGETS:
        print(f"=== {t} (campaign test, vs PCHIP) ===")
        print(f"{'seed':>5} {'base':>18} {'per_shot':>18} {'paired ps-vs-base':>26}")
        for r in rows:
            if r["status"] != "ok":
                print(f"{r['seed']:>5} {r['status']}")
                continue
            b, p = r["base"][t], r["per_shot"][t]
            pr = r["paired"].get(t, {})
            def g(e):
                return f"{e['skill_vs_pchip']:+.4f}{'*' if e['pass'] else ' '}"
            pstr = ("error" if "error" in r["paired"] else
                    f"{pr['skill_point']:+.4f} [{pr['skill_ci95'][0]:+.3f},"
                    f"{pr['skill_ci95'][1]:+.3f}]{'*' if pr['pass'] else ' '}")
            print(f"{r['seed']:>5} {g(b):>18} {g(p):>18} {pstr:>26}")
        v = verdict[t]
        print(f"   favor {v['favor_seeds']}, significant {v['significant_favor_seeds']}, "
              f"mean paired {v['mean']:+.4f}" if v["mean"] is not None else "   (no runs)")
        print()
    print(f"VERDICT: {verdict['overall']}")


if __name__ == "__main__":
    main()
