"""B.80 check of registration section 8 (Addendum 1 section 6): TX over T0 over all rows, target L5, under D650 (common
set) and S (S set), recomputed from the stored estimates read by this batch, against the published values of B.75.

    py ces_prediction/experiments/b80_ecei_gate/check80.py
Writes data/.b80/check.json; prints pass / fail per cell (ASCII).
"""
from __future__ import annotations

import json
import re

import common80 as M

import numpy as np  # noqa: E402

MD_SECTION = "## The fast-channel increment TX over T0 by rule and design, target L5"
MD_ROW = {"V1": "V1 random (r2-r4)", "V2": "V2 cluster-blocked (3 assignments)", "V3": "V3 rolling origin (fifths 2-5 pooled)",
          "V4a": "V4 chronological: fit to 31991, score from 32312", "V4b": "V4 chronological: fit to 32310, score from 32312"}
MD_COL = {"D650": 4, "S": 6}          # columns after the design: D10 D50 D100 D300 D650 D1000 S D10S


def published_md():
    txt = M.B75_MD.read_text(encoding="utf-8")
    sec = txt[txt.index(MD_SECTION):]
    sec = sec[:sec.index("\n### ")]
    out = {}
    for d, lab in MD_ROW.items():
        line = next(ln for ln in sec.splitlines() if ln.startswith(f"| {lab} |"))
        cells = [c.strip() for c in line.strip().strip("|").split("|")][1:]
        for r, k in MD_COL.items():
            out[(d, r)] = float(re.match(r"([+-]\d\.\d{3})", cells[k]).group(1))
    return out


def published_json():
    o = json.loads(M.B75_JSON.read_text(encoding="utf-8"))
    return {(d, r): o["main"][f"{d}|L5|{kind}"]["all"][r]["TX/T0"]["point"]
            for d in M.DESIGNS for kind, r in (("common", "D650"), ("S", "S"))}


def points(R, sets, us):
    """{(design, rule): (point, rows)}: B.75's point rule (mean over splits of the per-split skill)."""
    n = len(R["y"])
    y = R["tg_L5"]
    out = {}
    for d in M.DESIGNS:
        subs = M.subs_of(us, d)
        scored = np.isin(R["fidx"], np.concatenate([u["test"] for u in subs[0]]))
        for kind, r in (("common", "D650"), ("S", "S")):
            rows = np.flatnonzero(scored & sets[f"{kind}_L5"])
            sk = []
            for units in subs:
                t0 = M.load_est(units, "T0", r, n)[rows]
                tx = M.load_est(units, "TX", r, n)[rows]
                if not (np.all(np.isfinite(t0)) and np.all(np.isfinite(tx))):
                    raise SystemExit(f"FATAL: {d} {r}: a scored row without an estimate")
                sk.append(1.0 - np.sum((tx - y[rows]) ** 2) / np.sum((t0 - y[rows]) ** 2))
            out[(d, r)] = (float(np.mean(sk)), int(len(rows)))
    return out


def run():
    R = M.rows()
    sets = M.sets()
    us, _, _ = M.DS.units(R)
    pts = points(R, sets, us)
    md, js = published_md(), published_json()
    cells = []
    for (d, r), (p, nrows) in pts.items():
        ok = round(p, 3) == md[(d, r)] or f"{p:+.3f}" == f"{md[(d, r)]:+.3f}"
        cells.append({"design": d, "rule": r, "rows": nrows, "recomputed": p, "published_md": md[(d, r)],
                      "published_json": js[(d, r)], "difference_to_json": p - js[(d, r)], "pass": bool(ok)})
    res = {"what": "TX over T0, target L5, all rows; recomputed from the stored estimates against b75_availability.{md,json}",
           "pass_rule": "point rounded to three decimals equals the value printed in b75_availability.md",
           "cells": cells, "all_pass": all(c["pass"] for c in cells)}
    return res


def main():
    res = run()
    M.save_json(M.CACHE / "check.json", res)
    for c in res["cells"]:
        print(f"  {c['design']:4s} {c['rule']:5s} rows {c['rows']:7d}  recomputed {c['recomputed']:+.6f}  md {c['published_md']:+.3f}  "
              f"json diff {c['difference_to_json']:+.2e}  {'pass' if c['pass'] else 'FAIL'}")
    print("all pass" if res["all_pass"] else "CHECK FAILED")


if __name__ == "__main__":
    main()
