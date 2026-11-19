"""B.70 phase 2 readout: every phase-2 readout on D1 next to its old value (PREREGISTRATION_B70.md, Addendum 2).

Reads the D1 outputs in data/.b70p2/paper/ (written by the batches' own code, p2_*.py) and the old docs/paper/b*.json,
and writes docs/paper/b70_d1_phase2.{json,md}: per batch, the verdict of every original statement on D1 beside the old
one, and the main-text numbers (the keys docs/paper/ppcf/make_tables.py reads) beside their old values.

    py ces_prediction/experiments/b70_d1/p2_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_common as PC  # noqa: E402

OLD, NEW = PC.PAPER_OLD, PC.PAPER_D1
OUT_JSON, OUT_MD = OLD / "b70_d1_phase2.json", OLD / "b70_d1_phase2.md"

TREAT_OLD = {
    "b54": "r2-r9, B.39 rows (random 40,000-sample subset per fold)",
    "b55": "r2-r9 networks, B.39 rows (40,000 subset); rows after an absent frame by B.55's own rule",
    "b56": "filters fitted on all non-test discharges, held values kept; pooled set r3-r9 (confirmation), r2 development; "
           "B.39 rows (40,000 subset); KX6 rows = not within ten rows after an absent frame",
    "b57": "r2-r9, B.39 rows (40,000 subset); filters of B.56; quiet = not within ten rows after an absent frame",
    "b60": "r2-r9 (filters' addendum r3-r9), B.39 rows (40,000 subset)",
    "b61": "part A r2-r9 (filters r3-r9), B.39 rows (40,000 subset)",
    "b66": "r2-r9 (filters r3-r9), B.39 rows (40,000 subset)",
    "b65": "inclusive population (values above 3 keV kept in training, scoring and references), r2-r4, 40,000 subset",
    "b48": "four file-level splits, B.39 four-split network (cap 80, initialisation = split seed), B.30/B.42 row sets; "
           "a statement held on at least three of four splits",
}
TREAT_D1 = "r2-r4 pooled, every eligible row (no cap), refitted filters (training discharges, held removed)"


def get(o, path):
    for p in path:
        if o is None:
            return None
        o = o.get(p) if isinstance(o, dict) else None
    return o


def tri(v):
    return isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)


def fmt(v):
    if tri(v):
        return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" if abs(v[0]) < 5 else f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"
    if isinstance(v, (int, float)):
        return f"{v:+.4f}" if abs(v) < 5 else f"{v:.1f}"
    return "-" if v is None else str(v)


def point(v):
    return v[0] if tri(v) else (v if isinstance(v, (int, float)) else None)


def load(name, where):
    p = where / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ------------------------------------------------------------------------------------------ main-text keys
def keys_b54():
    th = (("low", "coldest"), ("middle", "middle"), ("high", "hottest"))
    out = [(f"thirds {t}: network over causal GP", ("reported", "thirds", t, "gp")) for t, _ in th]
    out += [(f"thirds {t}: network over twin", ("reported", "thirds", t, "t")) for t, _ in th]
    out += [("rows not flagged: increment", ("reported", "plain_rows", "t")),
            ("rows flagged: increment", ("reported", "flagged_rows", "t")),
            ("p90 |e| ratio to causal GP", ("reported", "p90_ratio", "causal GP", "value")),
            ("median |e| ratio to causal GP", ("reported", "median_ratio", "causal GP", "value")),
            ("overall: network over causal GP", ("reported", "overall", "gp")),
            ("overall: network over twin", ("reported", "overall", "t"))]
    return out


def keys_b55():
    return [("inputs as trained: over twin", ("partD", "all_discharges", "skill_file_over_twin")),
            ("inputs as trained: over causal GP", ("partD", "all_discharges", "skill_file_over_gp"))] + \
        [(f"{v}: over {r}", ("partD", "all_discharges", "variants", v, f"skill_over_{r}"))
         for v in ("MCmean", "Gmc", "Gecei", "Gbes") for r in ("twin", "gp")] + \
        [("all fast channels one frame old (G): over twin", ("partG", "variants", "G", "skill_over_twin")),
         ("G: over causal GP", ("partG", "variants", "G", "skill_over_gp")),
         ("G: change skill", ("partG", "variants", "G", "change_skill")),
         ("G, rows 1-3 after an absent frame: change skill", ("partG_rows_1_to_3_after_absent", "variants", "G", "change_skill"))]


def keys_b56():
    out = [(f"confirm: {k}", ("confirm", k)) for k in
           ("kfx_over_kf0", "kg_over_kf0", "A_over_kfx", "kc_over_kf0", "Tw_over_kf0", "kfx_over_Tw", "kfx_over_G",
            "kfx_over_O", "kf0_over_G", "A_over_G", "A_over_Tw", "kf0_over_G_90", "median_ratio_kfx_over_hold")]
    out += [("rows not after an absent frame: kfx_over_kf0", ("confirm_quiet", "kfx_over_kf0")),
            ("rows not flagged: kfx_over_kf0", ("confirm_plain", "kfx_over_kf0"))]
    for t in ("coldest", "middle", "hottest"):
        out += [(f"third {t}: kfx_over_kf0", (f"confirm_third_{t}", "kfx_over_kf0")),
                (f"third {t}: A_over_kfx", (f"confirm_third_{t}", "A_over_kfx"))]
    out += [(f"median |e| {e} (eV)", ("confirm", "median_abs_eV", e)) for e in ("kfx", "kf0", "H", "A", "G")]
    return out


def keys_b57():
    k = ["rms", "mse_over_s2_A", "mse_over_s2_Tw", "mse_over_s2_G", "mse_over_s2_H", "mse_over_s2_O", "mse_over_s2_KX",
         "mse_over_s2_K0", "gap_closure_A", "gap_closure_Tw", "gap_closure_KX", "net_twin_rho1", "kfx_kf0_rho1",
         "twin_kf0_rho1", "net_twin_beta0", "kfx_kf0_beta0", "removable_A_over_G", "removable_A_over_Tw",
         "removable_KX_over_G", "net_twin_rho3_minus_rho1"]
    return [(f"all rows: {x}", ("all", x)) for x in k]


def keys_b60():
    return [("low third: within 20 %, network - GP", ("cells", "low third by median T_i", "diff", "c-gp|0.20")),
            ("rows 1-3 after an absent frame: within 20 %, network - GP",
             ("cells", "rows 1 to 3 after an absent frame", "diff", "c-gp|0.20"))] + \
        [(f"all rows: share {q}", ("cells", "all rows", "value", q)) for q in ("c|0.20", "gp|0.20", "c|0.10", "gp|0.10")] + \
        [(f"filters, all rows: share {q}", ("addendum1", "cells_r3_r9", "all rows", "value", q)) for q in ("kfx|0.20", "kfx|0.10")] + \
        [(f"filters, {t}: within 20 %, KFX - GP", ("addendum1", "cells_r3_r9", t, "diff", "kfx-gp|0.20"))
         for t in ("low third by median T_i", "middle third by median T_i", "high third by median T_i")]


def keys_b61():
    a = ("part_a_r2_r9",)
    out = [(f"part A: {k}", a + (k,)) for k in ("rmse|c", "rmse|gp", "rmse_minus_gp_eV|c", "rmse_minus_gp_pct|c",
                                                 "rmse_minus_gp_eV|t", "mae_minus_gp_eV|c")]
    out += [(f"part A: {k}", a + (k,)) for k in ("skill_huber|c|hold", "skill_trim|c|hold", "skill_huber|c|gp",
                                                 "skill_trim|c|gp", "skill_huber|c|t", "skill_trim|c|t",
                                                 "skill_huber|c|off", "skill_trim|c|off")]
    out += [(f"filters: {k}", ("part_a_r3_r9_with_filters", k)) for k in ("skill_huber|kfx|kf0", "skill_trim|kfx|kf0",
                                                                          "skill_huber|kfx|gp", "skill_trim|kfx|gp", "rmse|kfx", "rmse|kf0")]
    out += [("part B: share of the reduction held by the 64 discharges", ("part_b", "64", "share_mean"))]
    return out


def keys_b66():
    out = []
    for c in "ABC":
        out += [(f"class {c}: increment", ("reported", "class", c, "inc")), (f"class {c}: over causal GP", ("reported", "class", c, "gp")),
                (f"class {c}: discharges", ("counts", "classes", c, "discharges"))]
    out += [("between bursts (B, C): increment", ("reported", "phase", "inter_BC", "inc")),
            ("between bursts (B, C): over causal GP", ("reported", "phase", "inter_BC", "gp"))]
    return out


BATCHES = [("b54", "B.54 row readouts", "b54_pooled_rows.json", keys_b54),
           ("b55", "B.55 sample timing", "b55_sample_timing.json", keys_b55),
           ("b56", "B.56 filters (refitted per D1 item 2.9)", "b56_kfx.json", keys_b56),
           ("b57", "B.57 yardsticks", "b57_yardsticks.json", keys_b57),
           ("b60", "B.60 relative errors", "b60_relative_error.json", keys_b60),
           ("b61", "B.61 absolute and robust scores", "b61_absolute_robust.json", keys_b61),
           ("b66", "B.66 regimes", "b66_regimes.json", keys_b66)]

B56_RULES = {"KX1": ("above", 0.0), "KX2": ("above", 0.0), "KX3": ("above", 0.0), "KX4": ("inside", 0.02),
             "KX5": ("below", 1.0), "KX6": ("above", 0.0)}


def verdict(rule, v):
    kind, ref = rule
    if kind == "above":
        return bool(v[1] > ref)
    if kind == "below":
        return bool(v[2] < ref)
    return bool(v[1] > -ref and v[2] < ref)


def statements(tag, old, new):
    out = []
    if tag == "b55":
        for k in old.get("verdicts", {}):
            out.append({"statement": k, "old_holds": old["verdicts"][k], "d1_holds": new["verdicts"].get(k)})
        od, nd = get(old, ("partD", "verdicts")), get(new, ("partD", "verdicts"))
        for k in (od or {}):
            out.append({"statement": f"part D {k}", "old_holds": od[k], "d1_holds": (nd or {}).get(k)})
        return out
    if tag == "b56":
        for k, rule in B56_RULES.items():
            ov = old["statements"][k]["value"]
            nv = new["statements_pooled"][k]["value"]
            out.append({"statement": k, "rule": old["statements"][k]["rule"], "old": ov, "old_holds": old["statements"][k]["holds"],
                        "d1": nv, "d1_holds": verdict(rule, nv)})
        return out
    for k, s in (old.get("statements") or {}).items():
        n = (new.get("statements") or {}).get(k, {})
        ov, nv = s.get("value"), n.get("value")
        rec = {"statement": k, "old_holds": s.get("holds"), "d1_holds": n.get("holds"), "text": s.get("text") or s.get("rule")}
        if tri(ov):
            rec["old"], rec["d1"] = ov, nv
        elif isinstance(ov, dict) and all(tri(x) for x in ov.values()):
            rec["old"] = {kk: x for kk, x in ov.items()}
            rec["d1"] = nv
        out.append(rec)
    return out


def main():
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B70.md (Addendum 2)", "d1_treatment": TREAT_D1,
           "batches": {}}
    for tag, title, fname, kf in BATCHES:
        old, new = load(fname, OLD), load(fname, NEW)
        if new is None:
            res["batches"][tag] = {"title": title, "missing": True}
            continue
        rows = []
        for label, path in kf():
            o, n = get(old, path), get(new, path)
            rows.append({"quantity": label, "key": "/".join(path), "old": o, "d1": n,
                         "diff": (point(n) - point(o)) if point(n) is not None and point(o) is not None else None})
        res["batches"][tag] = {"title": title, "old_treatment": TREAT_OLD[tag], "statements": statements(tag, old, new),
                               "main_text": rows}
    # B.65 on the cut population
    o65, n65 = load("b65_inclusive.json", OLD), load("b65_inclusive_cut.json", NEW)
    if n65:
        res["batches"]["b65"] = {"title": "B.65: inclusive networks scored with the cut harness on the D1 rows",
                                 "old_treatment": TREAT_OLD["b65"],
                                 "statements": [{"statement": k, "text": n65["statements"][k]["text"],
                                                 "old": o65["statements"][k]["value"], "old_holds": o65["statements"][k]["holds"],
                                                 "d1": n65["statements"][k]["value"], "d1_holds": n65["statements"][k]["holds"]}
                                                for k in ("I1", "I2", "I3", "I4")],
                                 "main_text": [{"quantity": f"{a} over {b}, MSE skill", "key": f"all_rows/mse|{a}|{b}",
                                                "old": None, "d1": n65["all_rows"].get(f"mse|{a}|{b}"), "diff": None}
                                               for a, b in (("c", "c39"), ("t", "t39"), ("c39", "gp"), ("c39", "t39"))]}
    # replays
    rp, b48 = load("b70_replay.json", NEW), load("b48_reread.json", OLD)
    if rp:
        sp42 = b48["b42"]["splits"]
        sp30 = b48["b30"]["cells"]
        seeds = list(sp42)

        def old_split(src, L, key):
            v = [src[s][L]["skill"][key] for s in seeds]
            return {"per_split": {s: x for s, x in zip(seeds, v)}, "mean_point": float(np.mean([x[0] for x in v])),
                    "splits_above_0": int(sum(x[1] > 0 for x in v))}
        st = rp["statements"]
        rows = [{"statement": "T1 (L = 1)", "d1": st["T1_L1"]["value"], "d1_holds": st["T1_L1"]["holds"],
                 "old": old_split(sp42, "1", "causal_vs_gp"), "old_holds": b48["b42"]["verdicts"]["T1"]["holds"]},
                {"statement": "T1 (L = 2)", "d1": st["T1_L2"]["value"], "d1_holds": st["T1_L2"]["holds"],
                 "old": old_split(sp42, "2", "causal_vs_gp"), "old_holds": b48["b42"]["verdicts"]["T1"]["holds"]},
                {"statement": "T2 (L = 2, over the twin)", "d1": st["T2"]["value"], "d1_holds": st["T2"]["holds"],
                 "old": old_split(sp42, "2", "causal_vs_twin"), "old_holds": b48["b42"]["verdicts"]["T2"]["holds"]},
                {"statement": "H1 (k = 5, over the causal GP)", "d1": st["H1"]["value"], "d1_holds": st["H1"]["holds"],
                 "old": old_split(sp30, "5", "causal_vs_gp"), "old_holds": b48["b30"]["H1"]["holds"]},
                {"statement": "H2 (k = 5, over the twin)", "d1": st["H2"]["value"], "d1_holds": st["H2"]["holds"],
                 "old": old_split(sp30, "5", "causal_vs_hist80"), "old_holds": b48["b30"]["H2"]["holds"]}]
        res["batches"]["replay"] = {"title": "Delay replay (L = 1, 2) and 50 ms thinning replay", "old_treatment": TREAT_OLD["b48"],
                                    "statements": rows, "checks": rp["checks"], "main_text": []}
    changed = []
    for tag, b in res["batches"].items():
        for s in b.get("statements", []):
            if s.get("old_holds") is not None and s.get("d1_holds") is not None and bool(s["old_holds"]) != bool(s["d1_holds"]):
                changed.append(f"{tag} {s['statement']}: {'holds' if s['old_holds'] else 'did not hold'} -> "
                               f"{'holds' if s['d1_holds'] else 'does not hold'}")
    res["verdicts_changed"] = changed
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print("written", OUT_MD, "| verdicts changed:", changed, flush=True)


def render(res):
    L = ["# B.70 phase 2 -- the other readouts the paper cites, on D1", "",
         f"Registered: {res['registration']}. {res['d1_treatment']}. Each readout is the batch's own code pointed at the D1 "
         "rows (Addendum 2, C1); its outputs are in `data/.b70p2/paper/`. Old values: `docs/paper/b*.json`, with the treatment "
         "named per batch. Checks: `docs/paper/b70_checks.md` (phase 2 section).", "",
         "## Verdicts that change", ""]
    L += [f"- {c}" for c in res["verdicts_changed"]] or ["- none"]
    for tag, b in res["batches"].items():
        L += ["", f"## {b['title']}", ""]
        if b.get("missing"):
            L.append("Not computed.")
            continue
        L += [f"Old: {b['old_treatment']}. D1: {res['d1_treatment']}.", "",
              "| statement | old | old verdict | D1 | D1 verdict |", "|---|---|---|---|---|"]
        for s in b["statements"]:
            ov = s.get("old")
            if isinstance(ov, dict) and "per_split" in ov:
                ovs = "; ".join(f"s{k} {fmt(v)}" for k, v in ov["per_split"].items()) + f" ({ov['splits_above_0']} of 4 above 0)"
            elif isinstance(ov, dict):
                ovs = "; ".join(f"{k} {fmt(v)}" for k, v in ov.items())
            else:
                ovs = fmt(ov)
            nv = s.get("d1")
            nvs = "; ".join(f"{k} {fmt(v)}" for k, v in nv.items()) if isinstance(nv, dict) else fmt(nv)
            yn = lambda x: "-" if x is None else ("holds" if x else "does not hold")      # noqa: E731
            L.append(f"| {s['statement']}{(': ' + s['text']) if s.get('text') else ''} | {ovs} | {yn(s.get('old_holds'))} | "
                     f"{nvs} | {yn(s.get('d1_holds'))} |")
        if b.get("main_text"):
            L += ["", "| main-text quantity | old | D1 | difference |", "|---|---|---|---|"]
            for q in b["main_text"]:
                d = q["diff"]
                L.append(f"| {q['quantity']} | {fmt(q['old'])} | {fmt(q['d1'])} | {'-' if d is None else (f'{d:+.3f}' if abs(d) < 5 else f'{d:+.1f}')} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
