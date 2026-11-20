"""B.80: docs/paper/b80_ecei_gate.md from the readout dictionary (readout80.py).  No number is recomputed here."""
from __future__ import annotations

DES = ("V1", "V2", "V3", "V4a", "V4b")
DLAB = {"V1": "V1 random (r2-r4)", "V2": "V2 cluster-blocked (3 assignments)", "V3": "V3 rolling origin (fifths 2-5 pooled)",
        "V4a": "V4 chronological: fit to 31991, score from 32312", "V4b": "V4 chronological: fit to 32310, score from 32312"}
RLAB = {"S": "rule S (S set)", "D650": "rule D650 (common set)"}
BLOCKS = ("block up to 31991", "block 32002-32310", "block from 32312", "blocks up to 32310")
SK = ("TX/T0", "TXN/TN", "TB/T0", "TE/T0")
SKLAB = {"TX/T0": "TX over T0", "TXN/TN": "TXN over TN", "TB/T0": "TB over T0", "TE/T0": "TE over T0"}


def s3(x):
    return f"{x:+.3f}"


def iv(v):
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def cellf(v):
    if v is None:
        return "-"
    out = f"{s3(v['point'])}<br>d {iv(v['discharge'])}<br>c {iv(v['cluster'])}"
    nf = v.get("draws_not_finite")
    if nf and (nf.get("discharge") or nf.get("cluster")):
        out += f"<br>(draws left out: d {nf['discharge']}, c {nf['cluster']})"
    return out


def cnt(c):
    return f"{c['rows']:,} / {c['discharges']} / {c['clusters']}"


def class_table(G, prefix, title_rows=True):
    """Rows: all (or the block), classes, no class, high minus low."""
    L = ["| stratum | rows / discharges / clusters | RMSE T0 | RMSE TX | " + " | ".join(SKLAB[k] for k in SK) + " |",
         "|---|---|---|---|" + "---|" * len(SK)]
    head = prefix[:-1] if prefix else "all"
    keys = [(head, head if prefix else "all")] + [(f"class {c}", f"{prefix}class {c}") for c in ("low", "middle", "high")] + \
        [("no class", f"{prefix}no class")]
    for lab, k in keys:
        if k not in G:
            continue
        o = G[k]
        rm = o["rmse_eV"]
        L.append(f"| {lab} | {cnt(o['counts'])} | {rm['T0']:.1f} | {rm['TX']:.1f} | " +
                 " | ".join(cellf(o.get(s)) for s in SK) + " |")
    k = f"{prefix}low+high"
    if k in G:
        o = G[k]
        L.append(f"| high minus low (rows of low and high: {cnt(o['counts'])}) | | | | " +
                 " | ".join(cellf(o.get(f"high minus low: {s}")) for s in SK) + " |")
    return L


def render(o):
    D = o["distribution"]["D1 discharges"]
    L = ["# B.80 — the coherence of the ECEI channels as an input-only condition for the fast-channel increment", "",
         f"Registered: `{o['registration']}`; implementation choices in `{o['addendum']}`, written before any readout. "
         "CPU only; **no model is fitted**: the stored estimates of B.75 (T0, TX, TB, TE; `data/.b75/fits/`) and of the main "
         "run of B.79 (TN, TXN, beam columns as delivered; `data/.b79/fits/`), target L5, rule S on B.75's S set and rule "
         "D650 on B.75's common set, and input-only measures of the four raw ECEI channels. Exploratory on used data "
         "(registration section 1).", "",
         "**C_E** = the mean of the six pairwise Pearson correlations of the four raw ECEI channels over the grid rows of "
         "the discharge (first and last row of the file excluded; discharges with at least 50 grid rows; undefined where a "
         "channel is constant). Classes: low C_E < 0.7, middle 0.7 to below 0.9, high 0.9 and above; the class of a row is "
         "the class of its discharge; a discharge without C_E belongs to no class. **TG** = TX where C_E is 0.7 or above, "
         "T0 otherwise (also without C_E), assembled row by row from the stored estimates; no fit.", "",
         "Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows. Each value: point, then 95 % intervals "
         "with discharges resampled (d) and with shot clusters resampled (c), 20,000 draws, seed 20261180 (V1: a "
         "re-partition drawn first; V2: an assignment drawn first; V3, V4: the scored units). Verdicts on the cluster form. "
         "Points under V1 and V2 = mean over the three splits. A skill within a class resamples the units with rows in "
         "that class; a difference high minus low is formed draw by draw on draws over the units with rows in the low or "
         "the high class (a draw with no unit of one class is left out; the count is given where it is not zero). RMSE in "
         "eV.", "",
         "Order of work. Before this readout, these results existed: the input-only measures (their distribution, "
         "section 'The measures'), and the check of registration section 8. No skill, RMSE or statement by class and no "
         "gated estimate had been computed.", ""]
    # ---- statements
    L += ["## Statements (cluster interval)", "",
          "| Id | statement | value: point; d interval; c interval | verdict | registered expectation | expectation met | note |",
          "|---|---|---|---|---|---|---|"]
    for k, s in o["statements"].items():
        v = s["value"]
        if v is None:
            val, verdict = "not formed", "not formed"
        else:
            val = f"{s3(v['point'])}; d {iv(v['discharge'])}; c {iv(v['cluster'])}"
            nf = v.get("draws_not_finite")
            if nf and (nf["discharge"] or nf["cluster"]):
                val += f" (draws left out: d {nf['discharge']}, c {nf['cluster']})"
            verdict = f"**{'holds' if s['holds'] else 'does not hold'}**"
        met = {True: "yes", False: "no", None: "n/a (uncertain)"}[o["expectations"][k]["verdict_as_expected"]]
        if v is None:
            met = "n/a (not formed)"
        L.append(f"| {k} | {s['text']} | {val} | {verdict} | {s['expected']} | {met} | {s.get('note', '')} |")
    L += ["", "G1, G4 and G5 are differences of two class skills (TX over T0, high class minus low class). G2 holds when the "
              "lower end of the cluster interval is at or below zero; G9 holds when its upper end is at or above zero. 'V4' "
              "in G8 is V4a (fit to 31991); V4b is in the note.", ""]
    # ---- check
    L += ["## Check of registration section 8", "",
          "TX over T0 over all rows (all classes), target L5, recomputed from the stored estimates read by this batch, "
          "against the values printed in `docs/paper/b75_availability.md`. Pass rule: equal to the third decimal.", "",
          "| design | rule | rows | recomputed | published | difference to the JSON of B.75 | pass |", "|---|---|---|---|---|---|---|"]
    for c in o["check_stage"]["cells"]:
        L.append(f"| {DLAB[c['design']]} | {c['rule']} | {c['rows']:,} | {c['recomputed']:+.6f} | {c['published_md']:+.3f} | "
                 f"{c['difference_to_json']:+.1e} | {'yes' if c['pass'] else 'no'} |")
    L += ["", f"All cells pass: {'yes' if o['check_stage']['all_pass'] else 'no'} (stage `check80.py`); recomputed in the readout "
              f"pass: {'yes' if o['check_in_readout']['all_pass'] else 'no'}.", ""]
    # ---- measures
    L += ["## The measures, per block of shots", "",
          "Inputs only; population: the D1 discharges. C_E early: the same measure over the first 200 grid rows of the "
          "discharge (equal to C_E for discharges with at most 200 grid rows). N_E: the median over the absent steps of the "
          "discharge with five rows of the block before them of the ECEI array mean in the first row after the step over its "
          "median over the five rows before (at least three steps). No statement uses C_E early or N_E.", "",
          "| block | D1 discharges | with C_E | without C_E: fewer than 50 grid rows / constant channel | low / middle / high | share low | C_E 10 / 25 / 50 / 75 / 90 % |",
          "|---|---|---|---|---|---|---|"]
    for b in ("all blocks",) + BLOCKS[:3]:
        x = D[b]
        q = x["C_E quantiles"]
        w = x["without C_E, by reason"]
        pc = x["per class"]
        L.append(f"| {b} | {x['discharges']} | {x['with C_E']} | {w['fewer than 50 grid rows']} / {w['constant channel']} | "
                 f"{pc['low']} / {pc['middle']} / {pc['high']} | {x['share low among those with C_E']:.3f} | "
                 + " / ".join(f"{q[p]:.3f}" for p in ("0.1", "0.25", "0.5", "0.75", "0.9")) + " |")
    L += ["", "| block | C_E early defined | same class as C_E (share) | same class, discharges with more than 200 grid rows | "
              "discharges with at most 200 grid rows | C_E early median | N_E defined | N_E 10 / 50 / 90 % | absent steps used | steps with a reference not above zero |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for b in ("all blocks",) + BLOCKS[:3]:
        x = D[b]
        qn = x["N_E quantiles"]
        sl = x["C_E early: same class, discharges with at least 200 grid rows only"]
        ne = " / ".join(f"{qn[p]:.3f}" for p in ("0.1", "0.5", "0.9")) if qn["0.5"] is not None else "-"
        L.append(f"| {b} | {x['C_E early: defined']} | {x['C_E early: same class as C_E (share of discharges with both)']:.3f} "
                 f"({x['C_E early: discharges with both']}) | {'-' if sl is None else f'{sl:.3f}'} | "
                 f"{x['C_E early: discharges with fewer than 200 grid rows (early = whole)']} | {x['C_E early quantiles']['0.5']:.3f} | "
                 f"{x['N_E: defined']} | {ne} | {x['N_E: absent steps used (sum)']} | "
                 f"{x['N_E: steps with five rows before, reference not above zero (sum)']} |")
    A = o["distribution"]["all grid files"]["all blocks"]
    L += ["", f"All 641 grid files: {A['with C_E']} with C_E (low / middle / high {A['per class']['low']} / {A['per class']['middle']} / "
              f"{A['per class']['high']}; without C_E: {A['without C_E, by reason']['fewer than 50 grid rows']} with fewer than 50 grid "
              f"rows, {A['without C_E, by reason']['constant channel']} with a constant channel); median C_E {A['C_E quantiles']['0.5']:.3f}.", ""]
    # ---- by class, per design and rule
    L += ["## By class of C_E, per design and rule", "",
          "Cells: rows / discharges / clusters scored; RMSE (points); skills with intervals. TB and TE exist for V1 and V4 "
          "only. The last row is the difference of the class skills, high class minus low class.", ""]
    for r in ("S", "D650"):
        for d in DES:
            L += [f"### {DLAB[d]}, {RLAB[r]}", ""] + class_table(o["cells"][f"{d}|{r}"], "") + [""]
    # ---- V1 by block
    L += ["## V1 by chronological block of the test discharge and class of C_E", ""]
    for r in ("S", "D650"):
        G = o["cells"][f"V1|{r}"]
        for b in BLOCKS:
            if b in G:
                L += [f"### V1, {RLAB[r]}, {b}", ""] + class_table(G, b + "|") + [""]
    # ---- gated estimator
    L += ["## The gated estimator TG (TX where C_E is 0.7 or above, T0 otherwise)", "",
          "| design | rule | stratum | rows / discharges / clusters | rows gated to TX (middle and high class) | RMSE T0 / TX / TG | TG over T0 | TG over TX |",
          "|---|---|---|---|---|---|---|---|"]
    for r in ("S", "D650"):
        for d in DES:
            G = o["cells"][f"{d}|{r}"]
            sts = ["all"] + ([b for b in BLOCKS if b in G] if d == "V1" else [])
            for st in sts:
                x = G[st]
                pre = "" if st == "all" else st + "|"
                ng = sum(G[f"{pre}class {c}"]["counts"]["rows"] for c in ("middle", "high") if f"{pre}class {c}" in G)
                rm = x["rmse_eV"]
                L.append(f"| {DLAB[d]} | {r} | {st} | {cnt(x['counts'])} | {ng:,} | {rm['T0']:.1f} / {rm['TX']:.1f} / {rm['TG']:.1f} | "
                         f"{cellf(x['TG/T0'])} | {cellf(x['TG/TX'])} |")
    L.append("")
    # ---- Spearman
    sp = o["spearman"]
    L += ["## Spearman correlation between C_E and the per-discharge skill of TX over T0 (V1, rule S)", "",
          "Per-discharge skill = 1 - SSE(TX) / SSE(T0) over the discharge's S-set rows, per re-partition; discharges with C_E "
          "and at least 50 S-set rows. Point = mean over the three re-partitions of the Spearman correlation; intervals: a "
          "re-partition drawn first, then discharges (d) or shot clusters (c) resampled. Registered: all blocks; the rows by "
          "block are added (Addendum 1 section 4).", "",
          f"Discharges with C_E but fewer than 50 S-set rows: {sp['discharges with C_E but fewer than 50 S-set rows']}; with "
          f"at least 50 S-set rows but no C_E: {sp['discharges with at least 50 S-set rows but no C_E']}.", "",
          "| discharges of | discharges / clusters | Spearman (point) | per re-partition (r2, r3, r4) | d interval | c interval | median per-discharge skill |",
          "|---|---|---|---|---|---|---|"]
    for st in ("all blocks",) + BLOCKS[:3]:
        x = sp[st]
        L.append(f"| {st} | {x['discharges']} / {x['clusters']} | {s3(x['point'])} | " + ", ".join(s3(p) for p in x["per re-partition"]) +
                 f" | {iv(x['discharge'])} | {iv(x['cluster'])} | {x['median per-discharge skill (mean over re-partitions)']:+.3f} |")
    L += ["", f"Readout run time {o['runtime_readout_s']:.0f} s.", ""]
    L += ["## Limits (registration section 7, unchanged)", "",
          "- Exploratory on used data; the measure was chosen after the block comparison had been seen.",
          "- C_E is computed over the whole discharge. No field, ECEI setting or channel radius is in the files, so what "
          "lowers the coherence is not decided here.",
          "- The cuts 0.7 and 0.9 are fixed by hand before the distribution was seen; the tables by class show what other "
          "cuts would give only coarsely.",
          "- TB and TE exist for V1 and V4 only; no new fit is made, so there is no estimator refitted within a class.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    # re-render the markdown from the readout JSON (no number is recomputed)
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common80 as M
    M.save_text(M.OUT_MD, render(json.loads(M.OUT_JSON.read_text(encoding="utf-8"))))
    print("written", M.OUT_MD)
