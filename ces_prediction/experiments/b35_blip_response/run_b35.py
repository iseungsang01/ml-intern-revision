"""B.35 (PREREGISTRATION_B35.md): the beam blip as a repeated perturbation.

CPU only; the 10 ms files, no model.  Run from the repo root:
    py ces_prediction/experiments/b35_blip_response/run_b35.py
Output: docs/paper/b35_blip_response.json / .md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
OUT_JSON = REPO / "docs" / "paper" / "b35_blip_response.json"
OUT_MD = REPO / "docs" / "paper" / "b35_blip_response.md"
SEED, N_BOOT, K = 20261116, 2000, 15


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def events():
    files = sorted(DATA.glob("s*.csv"))
    if len(files) != 641:
        raise SystemExit(f"FATAL: expected 641 CSV files, found {len(files)}")
    ev = []            # file, absent frames, level before, T_i ratios (K), ECEI ratios (K)
    for fi, f in enumerate(files):
        df = pd.read_csv(f).iloc[1:-1].reset_index(drop=True)
        if len(df) < 12:
            continue
        t = df["time"].to_numpy(float)
        ti = df["CES_TI"].to_numpy(float)
        ti = np.where(ti <= 3000, ti, np.nan)
        ec = df[[c for c in df.columns if c.startswith("ECEI_")]].to_numpy(float).mean(1)
        dt = np.diff(t)
        n = len(t)
        for j in np.flatnonzero((dt > 0.015) & (dt < 0.045)):
            if j < 5 or not np.all(np.abs(dt[j - 5:j] - 0.010) < 1e-4):
                continue
            pre = ti[j - 5:j + 1]
            pre = pre[np.isfinite(pre)]
            if len(pre) < 3:
                continue
            lvl, elvl = float(np.median(pre)), float(np.median(ec[j - 5:j + 1]))
            rt, re_ = np.full(K, np.nan), np.full(K, np.nan)
            for k in range(1, K + 1):
                i = j + k
                if i >= n or (k > 1 and abs(dt[i - 1] - 0.010) > 1e-4):
                    break
                rt[k - 1], re_[k - 1] = ti[i] / lvl, ec[i] / elvl
            ev.append((fi, int(round(dt[j] / 0.010)) - 1, lvl, rt, re_))
    return ev


def main():
    ev = events()
    fidx = np.array([e[0] for e in ev])
    absent = np.array([e[1] for e in ev])
    lvl = np.array([e[2] for e in ev])
    RT, RE = np.array([e[3] for e in ev]), np.array([e[4] for e in ev])
    uf, f = np.unique(fidx, return_inverse=True)
    rng = np.random.default_rng(SEED)
    reps = [rng.integers(0, len(uf), len(uf)) for _ in range(N_BOOT)]
    by_file = [np.flatnonzero(f == i) for i in range(len(uf))]

    def stat(fn):
        """fn(indices of events) -> float; point and bootstrap over discharges."""
        p = fn(np.arange(len(ev)))
        b = [fn(np.concatenate([by_file[i] for i in r])) for r in reps]
        return [float(p), *ci(b)]

    def trace(R_, idx):
        return np.array([np.nanmedian(R_[idx, k]) if np.isfinite(R_[idx, k]).any() else np.nan for k in range(K)])

    def tint(R_, idx):
        d = 1 - trace(R_, idx)
        return 10.0 * np.nansum(d) / d[0] if np.isfinite(d[0]) and d[0] > 0 else np.nan

    one, more = absent == 1, absent >= 2
    q1, q2 = np.quantile(lvl[one], [1 / 3, 2 / 3])
    low, high = one & (lvl <= q1), one & (lvl > q2)

    def first(R_, m, idx):
        v = R_[idx[m[idx]], 0]
        return np.nanmedian(v) if np.isfinite(v).any() else np.nan

    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B35.md",
           "events": int(len(ev)), "discharges": int(len(uf)),
           "events_one_absent_frame": int(one.sum()), "events_two_or_three": int(more.sum()),
           "level_thirds_eV": [float(q1), float(q2)],
           "events_lowest_third": int(low.sum()), "events_highest_third": int(high.sum())}
    for name, R_ in (("T_i", RT), ("ECEI", RE)):
        c = {"ratio_k1_one": stat(lambda i: first(R_, one, i)), "ratio_k1_two_or_three": stat(lambda i: first(R_, more, i)),
             "N1_difference": stat(lambda i: first(R_, more, i) - first(R_, one, i)),
             "N1_in_lowest_third": stat(lambda i: first(R_, more & (lvl <= q1), i) - first(R_, low, i)),
             "N1_in_highest_third": stat(lambda i: first(R_, more & (lvl > q2), i) - first(R_, high, i)),
             "integral_time_ms_all": stat(lambda i: tint(R_, i)),
             "integral_time_ms_lowest_third": stat(lambda i: tint(R_, i[low[i]])),
             "integral_time_ms_highest_third": stat(lambda i: tint(R_, i[high[i]])),
             "N2_difference_ms": stat(lambda i: tint(R_, i[high[i]]) - tint(R_, i[low[i]])),
             "ratio_k1_lowest_third": stat(lambda i: first(R_, low, i)), "ratio_k1_highest_third": stat(lambda i: first(R_, high, i)),
             "trace_one": [float(x) for x in trace(R_, np.flatnonzero(one))],
             "trace_two_or_three": [float(x) for x in trace(R_, np.flatnonzero(more))],
             "trace_lowest_third": [float(x) for x in trace(R_, np.flatnonzero(low))],
             "trace_highest_third": [float(x) for x in trace(R_, np.flatnonzero(high))]}
        res[name] = c
        print(name, "done", flush=True)
    res["N1_holds"] = bool(res["T_i"]["N1_difference"][2] < 0)
    res["N2_holds"] = bool(res["T_i"]["N2_difference_ms"][1] > 0)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    f3 = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"                      # noqa: E731
    s3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                   # noqa: E731
    m1 = lambda x: f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"                      # noqa: E731
    sm = lambda x: f"{x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]"                   # noqa: E731
    T, E = res["T_i"], res["ECEI"]
    L = ["# B.35 — the beam blip as a repeated perturbation (registered: PREREGISTRATION_B35.md)", "",
         f"CPU, the 10 ms files, no model. {res['events']:,} blips of {res['discharges']} discharges with T_i observed in at "
         f"least three of the six rows before: {res['events_one_absent_frame']:,} with one absent frame, "
         f"{res['events_two_or_three']} with two or three. Ratio = value over the median of the six rows before; "
         "integral recovery time = 10 ms x sum of the deficit over rows 1 to 15 / deficit of row 1 (median traces). "
         "Intervals: 95 %, 2,000 resamples of discharges.", "",
         f"- **N1** (dose: the T_i ratio in the first row is lower after two or three absent frames than after one): "
         f"{f3(T['ratio_k1_two_or_three'])} against {f3(T['ratio_k1_one'])}, difference {s3(T['N1_difference'])} -> "
         f"{'holds' if res['N1_holds'] else 'does not hold'}.",
         f"- **N2** (time scale: the integral recovery time of T_i is longer in the highest third of the level before than in the lowest; "
         f"thirds at {res['level_thirds_eV'][0]:.0f} and {res['level_thirds_eV'][1]:.0f} eV, {res['events_lowest_third']} and "
         f"{res['events_highest_third']} events): {m1(T['integral_time_ms_highest_third'])} ms against "
         f"{m1(T['integral_time_ms_lowest_third'])} ms, difference {sm(T['N2_difference_ms'])} ms -> "
         f"{'holds' if res['N2_holds'] else 'does not hold'}.", "",
         "## N3 (reported)", "",
         "| quantity | T_i | ECEI level |", "|---|---|---|",
         f"| ratio in the first row, one absent frame | {f3(T['ratio_k1_one'])} | {f3(E['ratio_k1_one'])} |",
         f"| ratio in the first row, two or three | {f3(T['ratio_k1_two_or_three'])} | {f3(E['ratio_k1_two_or_three'])} |",
         f"| difference (dose) | {s3(T['N1_difference'])} | {s3(E['N1_difference'])} |",
         f"| dose difference within the lowest third | {s3(T['N1_in_lowest_third'])} | {s3(E['N1_in_lowest_third'])} |",
         f"| dose difference within the highest third | {s3(T['N1_in_highest_third'])} | {s3(E['N1_in_highest_third'])} |",
         f"| ratio in the first row, lowest third | {f3(T['ratio_k1_lowest_third'])} | {f3(E['ratio_k1_lowest_third'])} |",
         f"| ratio in the first row, highest third | {f3(T['ratio_k1_highest_third'])} | {f3(E['ratio_k1_highest_third'])} |",
         f"| integral recovery time, all events (ms) | {m1(T['integral_time_ms_all'])} | {m1(E['integral_time_ms_all'])} |",
         f"| integral recovery time, lowest third (ms) | {m1(T['integral_time_ms_lowest_third'])} | {m1(E['integral_time_ms_lowest_third'])} |",
         f"| integral recovery time, highest third (ms) | {m1(T['integral_time_ms_highest_third'])} | {m1(E['integral_time_ms_highest_third'])} |",
         f"| highest minus lowest (ms) | {sm(T['N2_difference_ms'])} | {sm(E['N2_difference_ms'])} |", "",
         "## Median traces (ratio by row after the blip)", "",
         "| group | " + " | ".join(str(k) for k in range(1, K + 1)) + " |", "|---|" + "---|" * K]
    for lab, key in (("one absent frame", "trace_one"), ("two or three", "trace_two_or_three"),
                     ("lowest third of the level", "trace_lowest_third"), ("highest third", "trace_highest_third")):
        L.append(f"| T_i, {lab} | " + " | ".join(f"{x:.3f}" for x in T[key]) + " |")
    for lab, key in (("one absent frame", "trace_one"), ("two or three", "trace_two_or_three")):
        L.append(f"| ECEI, {lab} | " + " | ".join(f"{x:.3f}" for x in E[key]) + " |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT_MD, res["N1_holds"], res["N2_holds"])


if __name__ == "__main__":
    main()
