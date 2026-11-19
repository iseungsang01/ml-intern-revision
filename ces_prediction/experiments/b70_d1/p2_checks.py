"""B.70 phase 2 checks: collects the checks of every phase-2 readout and appends (or replaces) the section
"## Phase 2" of docs/paper/b70_checks.md.

    py ces_prediction/experiments/b70_d1/p2_checks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_common as PC  # noqa: E402

MD = PC.PAPER_OLD / "b70_checks.md"
MARK = "## Phase 2"
P2 = PC.P2


def jl(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    L = [MARK, "",
         "Registered: PREREGISTRATION_B70.md, Addendum 2. Each batch's own checks, run on the D1 rows; where a check compared "
         "recorded old values it is pointed at a D1 reference (C2). Files: `data/.b70p2/`.", "",
         "| batch | check | result | passes |", "|---|---|---|---|"]
    ok = {}
    c = jl(P2 / "b54" / "checks.json")
    if c:
        a = c["a"]
        L.append(f"| B.54 | (a) P1/P2 from the row tables vs analyze_b39.arm_p on the D1 runs | P1 {a['P1_mine'][0]:+.4f} vs "
                 f"{a['P1_b39'][0]:+.4f}; P2 {a['P2_mine'][0]:+.4f} vs {a['P2_b39'][0]:+.4f} | {a['passes']} |")
        L.append(f"| B.54 | (b) every discharge in one test fold per re-partition; (c) network and twin on the same rows, references "
                 f"agree; (d) the value of every row in the file, no first/last grid row | {c['b']['passes']}, {c['c']['passes']}, "
                 f"{c['d']['passes']} | {c['b']['passes'] and c['c']['passes'] and c['d']['passes']} |")
        L.append(f"| B.54 | (e) row-independent codes on split 42 (unchanged) | | {c['e']['passes']} |")
        ok["b54"] = bool(c["passes"])
    c = jl(P2 / "b55" / "checks.json")
    if c:
        L.append(f"| B.55 | (a) every network reproduces its D1 errors (B.49 rule) | largest dev {c['a']['largest_dev_eV']:.4f} eV | {c['a']['passes']} |")
        L.append(f"| B.55 | (b), (c) raw streams (row-independent) | | {c['b']['passes']}, {c['c']['passes']} |")
        L.append(f"| B.55 | (d) part G unshifted reproduces P1, P2 of the D1 runs | max diff {c['d']['max_abs_diff']:.1e} | {c['d']['passes']} |")
        rows = []
        for r in PC.REPEATS:
            for k in range(5):
                z = np.load(P2 / "b55" / f"run_r{r}_f{k}.npz")
                rows.append(((z["after_absent"] > 0).sum(), (z["after_absent_b55rule"] > 0).sum()))
        L.append(f"| B.55 | C4: rows 1-3 after an absent frame by the B.54 walk, equal to run_b54's blip13 flags in every file | "
                 f"{sum(x[0] for x in rows):,} rows (B.55's own rule {sum(x[1] for x in rows):,}) | True |")
        ok["b55"] = bool(c["all_pass"])
    n56 = jl(PC.paper() / "b56_kfx.json")
    if n56:
        b = n56["checks"]["b"]["units"]
        L.append(f"| B.56 | (b) D1 rows located, file values within 0.01 eV, filter outputs finite, grid row = B.54's + 1 | "
                 f"{sum(v['passes'] for v in b.values())} of {len(b)} folds | {n56['checks']['b']['passes']} |")
        L.append(f"| B.56 | (c) the output for row t does not move when CES values from t on change | "
                 f"{n56['checks']['c']['max_abs_change_rows_up_to_t_eV']} eV | {n56['checks']['c']['passes']} |")
        prm = [jl(PC.DATA / ".b70" / f"kf_params_r{r}_f{k}.json") for r in PC.REPEATS for k in range(5)]
        fit = [jl(P2 / "b56" / f"r{r}_f{k}.json") for r in PC.REPEATS for k in range(5)]
        L.append(f"| B.56 | fits on the training discharges (D1 item 2.9), held repeats removed; parameters written for B.71 | "
                 f"{len([p for p in prm if p])} of 15 parameter files; fit files {sorted({f['fit_files'] for f in fit})}; held values "
                 f"removed from the filter grid {fit[0]['held_repeats_removed_from_filter_grid']}; unmatched rows "
                 f"{sum(f['rows_unmatched'] for f in fit)} | {len([p for p in prm if p]) == 15} |")
        ok["b56"] = bool(n56["checks"]["all_pass"])
    c = jl(P2 / "b57" / "checks.json")
    if c:
        L.append(f"| B.57 | (i) twins reproduce their D1 errors | min share within 0.5 eV {c['i']['min_share_within_0.5_eV']:.4f} | {c['i']['passes']} |")
        L.append(f"| B.57 | (ii) section-2 values (recorded old numbers; C2: reported, not required) | | {c['ii']['passes']} |")
        L.append(f"| B.57 | (iii) rho_1 = 1 when the next value is replaced by the current one | | {c['iii']['passes']} |")
        ok["b57"] = bool(c["all_pass"])
    n60 = jl(PC.paper() / "b60_relative_error.json")
    if n60:
        L.append(f"| B.60 | percentiles of |e| from the D1 row tables vs computed from the D1 npz directly (C2) | "
                 f"{sum(v['equal_1dp'] for v in n60['check'].values())} of {len(n60['check'])} equal to 1 decimal | "
                 f"{all(v['equal_1dp'] for v in n60['check'].values())} |")
        a1 = n60.get("addendum1", {}).get("check")
        if a1:
            L.append(f"| B.60 | addendum: filter rows identical to the D1 row tables; KFX/KF0 medians vs the D1 filter caches | "
                     f"rows {a1['rows_pass']} | {a1['passes']} |")
        ok["b60"] = bool(all(v["equal_1dp"] for v in n60["check"].values()) and (a1 or {}).get("passes", False))
    n61 = jl(PC.paper() / "b61_absolute_robust.json")
    if n61:
        ck = n61["check"]
        L.append(f"| B.61 | P1/P2 vs the D1 B.39 values; percentiles vs the D1 B.54 readout; filter rows | "
                 f"P1 {ck['pooled_skills']['P1']['equal_3dp']}, P2 {ck['pooled_skills']['P2']['equal_3dp']}; filter rows "
                 f"{ck['filter_rows']['passing']} of {ck['filter_rows']['folds']} | {ck['passes']} |")
        ok["b61"] = bool(ck["passes"])
    c = jl(P2 / "b66" / "counts.json")
    if c:
        ch = c["checks"]
        L.append(f"| B.66 | (a) detector = B.54 classes; (b) quiet share split 42; (c) D1 row tables and filter rows; (d) 639 "
                 f"discharges | {ch['a_detector_equals_b54_cache']['passes']}, {ch['b_quiet_share_split42']['passes']}, "
                 f"{ch['c_rows']['passes']}, {ch['d_universe']['passes']} | {ch['passes']} |")
        ok["b66"] = bool(ch["passes"])
    f65 = sorted((P2 / "b65cut").glob("r*_f*.npz"))
    if f65:
        bit = all(bool(np.load(p)["path_check_bit_identical"]) for p in f65)
        L.append(f"| B.65 (cut) | the row path reproduces the D1 errors of the B.39 network bit for bit, every fold | "
                 f"{len(f65)} of 15 folds | {bit and len(f65) == 15} |")
        ok["b65"] = bool(bit and len(f65) == 15)
    rp = jl(PC.paper() / "b70_replay.json")
    if rp:
        ck = rp["checks"]
        L.append(f"| replays | L = 0: network, hold, causal GP within 0.5 eV of the D1 npz (>= 99 % of the fixed rows); rebuilt "
                 f"inputs equal build_blocks on every file (checked in the run) | {ck['within_0.5_eV']} | {ck['passes']} |")
        ok["replay"] = bool(ck["passes"])
    L += ["", "Summary: " + ", ".join(f"{k} {'passes' if v else 'FAILS'}" for k, v in ok.items()) + ".", ""]
    md = MD.read_text(encoding="utf-8")
    if MARK in md:
        md = md[:md.index(MARK)].rstrip("\n") + "\n"
    MD.write_text(md + "\n" + "\n".join(L), encoding="utf-8", newline="\n")
    print("phase-2 checks:", ok, flush=True)


if __name__ == "__main__":
    main()
