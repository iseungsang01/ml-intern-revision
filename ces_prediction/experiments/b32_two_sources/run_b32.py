"""B.32 (PREREGISTRATION_B32.md): two sources of a large fast-channel excursion.

A  KSTAR, all files: in the first row after a beam blip, does the T_i change scale with the BES
   departure of that row?  (A1, A2)
B  KSTAR, twelve discharges with a raw stream: where do the departures beyond three standard
   deviations sit relative to the edge bursts?  (B1, B2)
C  MAST: T_i after a step of the SW beam while the viewed SS beam stays on.  (C1, C2, C3)

CPU only; no model.  Run from the repo root:
    py ces_prediction/experiments/b32_two_sources/run_b32.py
Output: docs/paper/b32_two_sources.json / .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "b31_bursts"))
sys.path.insert(0, str(HERE.parent / "mast"))
DATA = REPO / "data"
OUT_JSON = REPO / "docs" / "paper" / "b32_two_sources.json"
OUT_MD = REPO / "docs" / "paper" / "b32_two_sources.md"
N_BOOT, SEED = 2000, 20261116
GAIN_EV_PER_SD = 50.0
SEG_BREAK_S = 0.5


def ci(b):
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def wmedian(v, f, W):
    """Point median and bootstrap medians; v values, f index of the unit of each value, W weights."""
    o = np.argsort(v)
    v, f = v[o], f[o]
    c = np.cumsum(W[:, f], 1)
    tot = c[:, -1:]
    k = (c >= tot / 2).argmax(1)
    b = v[k]
    b[tot[:, 0] == 0] = np.nan
    return float(np.median(v)), b


def kstar_file(path):
    """Grid rows of one discharge: time, T_i, mean BES departure, rows-since-blip, segment id."""
    df = pd.read_csv(path).iloc[1:-1].reset_index(drop=True)
    t = df["time"].to_numpy(float)
    n = len(t)
    if n < 12:
        return None
    ti = df["CES_TI"].to_numpy(float)
    X = df[[c for c in df.columns if c.startswith("BES_")]].to_numpy(float)
    z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    dt = np.diff(t)
    seg = np.concatenate(([0], np.cumsum(dt >= SEG_BREAK_S)))
    cs = np.vstack([np.zeros((1, z.shape[1])), np.cumsum(z, 0)])
    d = np.full(n, np.nan)
    for i in range(10, n):
        if seg[i] == seg[i - 10]:
            d[i] = float((z[i] - (cs[i] - cs[i - 10]) / 10.0).mean())
    after = np.zeros(n, int)
    first = []
    for j in np.flatnonzero((dt > 0.015) & (dt < 0.045)):
        for k in range(1, 11):
            i = j + k
            if i < n and seg[i] == seg[j + 1] and after[i] == 0:
                after[i] = k
        if j >= 5 and np.all(np.abs(np.diff(t[j - 5:j + 1]) - 0.010) < 1e-4):
            first.append(j)
    return {"t": t, "ti": ti, "d": d, "after": after, "first": first}


# ------------------------------------------------------------------------------------------ A
def part_a(rng):
    files = sorted(DATA.glob("s*.csv"))
    if len(files) != 641:
        raise SystemExit(f"FATAL: expected 641 CSV files, found {len(files)}")
    rows = []          # (file index, d, ratio, change eV)
    for fi, f in enumerate(files):
        k = kstar_file(f)
        if k is None:
            continue
        for j in k["first"]:
            i = j + 1
            pre = k["ti"][j - 5:j + 1]
            pre = pre[np.isfinite(pre) & (pre <= 3000)]
            y = k["ti"][i]
            if len(pre) < 3 or not np.isfinite(y) or y > 3000 or not np.isfinite(k["d"][i]):
                continue
            m = float(np.median(pre))
            rows.append((fi, k["d"][i], y / m, y - m))
    r = np.array(rows)
    uf, inv = np.unique(r[:, 0].astype(int), return_inverse=True)
    W = rng.multinomial(len(uf), np.full(len(uf), 1.0 / len(uf)), size=N_BOOT).astype(float)
    groups = {"D1 (d <= -3)": r[:, 1] <= -3, "D2 (-3 < d <= -2)": (r[:, 1] > -3) & (r[:, 1] <= -2),
              "D3 (d > -2)": r[:, 1] > -2}
    out = {"rows": int(len(r)), "discharges": int(len(uf)), "groups": {}}
    bm = {}
    for name, m in groups.items():
        p, b = wmedian(r[m, 2], inv[m], W)
        bm[name] = (p, b)
        w = W[:, inv[m]]
        mean_change = (w * r[m, 3]).sum(1) / w.sum(1)
        mean_d = (w * r[m, 1]).sum(1) / w.sum(1)
        out["groups"][name] = {"rows": int(m.sum()), "median_ratio": p, "median_ratio_ci95": ci(b[np.isfinite(b)]),
                               "mean_departure_sd": float(r[m, 1].mean()),
                               "mean_change_eV": float(r[m, 3].mean()), "mean_change_ci95": ci(mean_change),
                               "proportional_reading_eV": float(GAIN_EV_PER_SD * r[m, 1].mean()),
                               "measured_minus_proportional_eV": float(r[m, 3].mean() - GAIN_EV_PER_SD * r[m, 1].mean()),
                               "measured_minus_proportional_ci95": ci(mean_change - GAIN_EV_PER_SD * mean_d)}
    d13 = bm["D1 (d <= -3)"][1] - bm["D3 (d > -2)"][1]
    d13 = d13[np.isfinite(d13)]
    diff = bm["D1 (d <= -3)"][0] - bm["D3 (d > -2)"][0]
    out["A1"] = {"D1_minus_D3": diff, "ci95": ci(d13),
                 "holds": bool(not (np.quantile(d13, 0.975) < 0 and diff < -0.05))}
    g1 = out["groups"]["D1 (d <= -3)"]
    out["A2"] = {"measured_minus_proportional_eV": g1["measured_minus_proportional_eV"],
                 "ci95": g1["measured_minus_proportional_ci95"],
                 "holds": bool(g1["measured_minus_proportional_eV"] >= 30 and g1["measured_minus_proportional_ci95"][0] > 0)}
    return out


# ------------------------------------------------------------------------------------------ B
def part_b(rng):
    import run_b31 as B31
    per = []
    for shot in B31.SHOTS:
        k = kstar_file(DATA / f"s{shot}.csv")
        raw = B31.bursts_and_blips(shot)
        t, d, after = k["t"], k["d"], k["after"]
        inside = (t >= raw["t0"] + 0.02) & (t <= raw["t_end"] - 0.02) & (after == 0) & np.isfinite(d)
        ev = raw["bursts"]
        near = (np.searchsorted(ev, t + 0.001, side="right") - np.searchsorted(ev, t - 0.001, side="left")) > 0
        t1, lvl = raw["level_1ms"]
        run = pd.Series(lvl).rolling(200, center=True, min_periods=50).median().to_numpy()
        idx = np.clip(np.searchsorted(t1, t), 0, len(t1) - 1)
        dip = lvl[idx] < 0.8 * run[idx]
        per.append({"shot": shot, "rows": int(inside.sum()), "bursts": int(len(ev)),
                    "u": int((inside & (d > 3)).sum()), "u_near": int((inside & (d > 3) & near).sum()),
                    "base_near": int((inside & near).sum()), "low": int((inside & (d < -3)).sum()),
                    "low_dip": int((inside & (d < -3) & dip).sum())})
    a = np.array([[p["rows"], p["u"], p["u_near"], p["base_near"], p["low"], p["low_dip"]] for p in per], float)
    tot = a.sum(0)
    out = {"per_discharge": per, "rows": int(tot[0]), "u_rows": int(tot[1]), "u_rows_with_burst": int(tot[2]),
           "share_u": float(tot[2] / tot[1]) if tot[1] else None, "base_share": float(tot[3] / tot[0]),
           "low_rows": int(tot[4]), "low_rows_with_dip": int(tot[5]),
           "B2_share_low_with_dip": float(tot[5] / tot[4]) if tot[4] else None}
    if tot[1] < 20:
        out["B1"] = {"testable": False}
        return out
    W = rng.multinomial(len(per), np.full(len(per), 1.0 / len(per)), size=N_BOOT).astype(float)
    s = W @ a
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_b = (s[:, 2] / s[:, 1]) / (s[:, 3] / s[:, 0])
    ratio_b = ratio_b[np.isfinite(ratio_b)]
    ratio = (tot[2] / tot[1]) / (tot[3] / tot[0])
    out["B1"] = {"testable": True, "ratio": float(ratio), "ci95": ci(ratio_b),
                 "holds": bool(ratio >= 3 and np.quantile(ratio_b, 0.025) > 1)}
    return out


# ------------------------------------------------------------------------------------------ C
def part_c(rng):
    import run_coupling as rc
    import run_gate as rg
    shots = [int(x) for x in rc.SHOTS_TXT.read_text().split()]
    cat = pd.read_parquet(rg.META, columns=["shot_id", "radii_magnetic__radius_efit", "radii_router_radius_efit",
                                            "radii_router_radius_d_alpha"]).set_index("shot_id")
    K = np.arange(-4, 17)
    ev = {"down": [], "up": []}          # (shot, region, ratios over K)
    status = {}
    for s in shots:
        d = rg.load_shot(s)
        if d is None:
            status["no_file"] = status.get("no_file", 0) + 1
            continue
        p = rg.prepare(d, "T_i", rg.POPS["G"])
        if p["status"] != "ok" or p["n_bad_dt"] > 0 or not p["t_finite"]:
            status["no_block_or_irregular"] = status.get("no_block_or_irregular", 0) + 1
            continue
        with np.load(rg.ACT_DIR / f"{s}.npz") as z:
            if not {"anb_time", "anb_ss_sum_power", "anb_sw_sum_power"} <= set(z.files):
                status["no_beam_trace"] = status.get("no_beam_trace", 0) + 1
                continue
            tb = z["anb_time"].astype(float) * 1000.0
            ss, sw = z["anb_ss_sum_power"].astype(float), z["anb_sw_sum_power"].astype(float)
        if ss.shape != tb.shape or sw.shape != tb.shape:
            status["no_beam_trace"] = status.get("no_beam_trace", 0) + 1
            continue
        a, _b = rc.block_of(d, "T_i", rg.POPS["G"])
        t_a = float(d["time_ms"][a])
        L, G, Y = p["L"], p["G"], p["Y0"]
        fss = rc.frame_means(tb, ss, t_a, 0, L - 1)
        fsw = rc.frame_means(tb, sw, t_a, 0, L - 1)
        try:
            ax = float(cat.loc[s, "radii_magnetic__radius_efit"])
            ro = float(cat.loc[s, "radii_router_radius_efit"])
            if not (np.isfinite(ro) and ro > ax):
                ro = float(cat.loc[s, "radii_router_radius_d_alpha"])
            rho = (p["R"] - ax) / (ro - ax) if np.isfinite(ax) and np.isfinite(ro) and ro > ax else np.full(len(p["R"]), np.nan)
        except KeyError:
            rho = np.full(len(p["R"]), np.nan)
        status["used"] = status.get("used", 0) + 1
        for n in range(4, L - 16):
            pre_sw, post_sw = fsw[n - 4:n], fsw[n + 1:n + 17]
            win_ss = fss[n - 4:n + 17]
            if not (np.isfinite(pre_sw).all() and np.isfinite(post_sw).all() and np.isfinite(win_ss).all()):
                continue
            m_ss = win_ss[:4].mean()
            if not (m_ss >= 0.3 and np.all(win_ss >= 0.3) and np.all(np.abs(win_ss - m_ss) <= 0.2 * m_ss)):
                continue
            kind = None
            if pre_sw.mean() >= 0.3 and post_sw.mean() <= 0.2 * pre_sw.mean():
                kind = "down"
            elif pre_sw.mean() <= 0.05 and post_sw.mean() >= 0.3:
                kind = "up"
            if kind is None:
                continue
            gpre = G[n - 4:n]
            okc = gpre.sum(0) >= 3
            if not okc.any():
                continue
            base = np.array([np.median(Y[n - 4:n, c][gpre[:, c]]) for c in np.flatnonzero(okc)])
            cols = np.flatnonzero(okc)
            for region, sel in (("all", np.ones(len(cols), bool)), ("core", rho[cols] < 0.5), ("edge", rho[cols] >= 0.7)):
                if not sel.any():
                    continue
                rat = np.full(len(K), np.nan)
                for q, kk in enumerate(K):
                    g = G[n + kk, cols[sel]]
                    if g.any():
                        rat[q] = np.median(Y[n + kk, cols[sel]][g] / base[sel][g])
                ev[kind].append((s, region, rat))
    uniq = sorted({e[0] for k in ev for e in ev[k]})
    sidx = {s: i for i, s in enumerate(uniq)}
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=N_BOOT).astype(float)
    out = {"status": status, "frames": [int(k) for k in K], "kinds": {}}
    for kind in ("down", "up"):
        out["kinds"][kind] = {}
        for region in ("all", "core", "edge"):
            E = [e for e in ev[kind] if e[1] == region]
            if not E:
                continue
            f = np.array([sidx[e[0]] for e in E])
            R = np.array([e[2] for e in E])
            n_ev = len({(e[0], i) for i, e in enumerate(E)})
            med, boots = [], []
            for q in range(len(K)):
                okq = np.isfinite(R[:, q])
                pq, bq = wmedian(R[okq, q], f[okq], W)
                med.append(pq)
                boots.append(bq)
            med, boots = np.array(med), np.array(boots)
            i1 = int(np.flatnonzero(K == 1)[0])
            late = (K >= 10) & (K <= 16)
            r_late, b_late = float(med[late].mean()), np.nanmean(boots[late], 0)
            stat = abs(1 - r_late) - 2 * abs(1 - med[i1])
            b_stat = np.abs(1 - b_late) - 2 * np.abs(1 - boots[i1])
            half = None
            for q in np.flatnonzero(K >= 1):
                if abs(1 - med[q]) >= 0.5 * abs(1 - r_late):
                    half = int(K[q])
                    break
            cell = {"events": int(len(E)), "shots": int(len(set(f))), "median_ratio": [float(x) for x in med],
                    "ci95": [ci(b[np.isfinite(b)]) for b in boots], "R_1": float(med[i1]),
                    "R_1_ci95": ci(boots[i1][np.isfinite(boots[i1])]), "R_late": r_late,
                    "R_late_ci95": ci(b_late[np.isfinite(b_late)]),
                    "delay_statistic": float(stat), "delay_ci95": ci(b_stat[np.isfinite(b_stat)]),
                    "half_response_frame": half, "testable": bool(len(E) >= 30)}
            lo, hi = cell["R_late_ci95"]
            cell["C1_direction"] = bool((hi < 1) if kind == "down" else (lo > 1))
            cell["C2_delay"] = bool(cell["delay_ci95"][0] > 0)
            out["kinds"][kind][region] = cell
    return out


def render(res):
    A, B, C = res["A"], res["B"], res["C"]
    L = ["# B.32 — two sources of a large fast-channel excursion (registered: PREREGISTRATION_B32.md)", "",
         "CPU, no model. Intervals: 95 %, 2,000 resamples of discharges (shots on MAST).", "",
         "## A. KSTAR, all files: the first row after a beam blip, by the BES departure of that row", "",
         f"{A['rows']:,} blips with T_i observed in the first row after and in at least three of the six rows before "
         f"({A['discharges']} discharges). d = mean over the nine BES channels of the departure from the 100 ms mean, "
         "in per-discharge standard deviations. Proportional reading = 50 eV per standard deviation.", "",
         "| group | rows | mean d (sd) | median T_i ratio to the level before | mean T_i change (eV) | proportional reading (eV) | measured minus proportional (eV) |",
         "|---|---|---|---|---|---|---|"]
    for name, g in A["groups"].items():
        L.append(f"| {name} | {g['rows']:,} | {g['mean_departure_sd']:+.2f} | {g['median_ratio']:.3f} "
                 f"[{g['median_ratio_ci95'][0]:.3f}, {g['median_ratio_ci95'][1]:.3f}] | {g['mean_change_eV']:+.1f} "
                 f"[{g['mean_change_ci95'][0]:+.1f}, {g['mean_change_ci95'][1]:+.1f}] | {g['proportional_reading_eV']:+.1f} | "
                 f"{g['measured_minus_proportional_eV']:+.1f} [{g['measured_minus_proportional_ci95'][0]:+.1f}, "
                 f"{g['measured_minus_proportional_ci95'][1]:+.1f}] |")
    L += ["", f"- **A1** (the ratio of D1 is not below that of D3): D1 minus D3 = {A['A1']['D1_minus_D3']:+.3f} "
          f"[{A['A1']['ci95'][0]:+.3f}, {A['A1']['ci95'][1]:+.3f}] -> {'holds' if A['A1']['holds'] else 'does not hold'}.",
          f"- **A2** (in D1 the measured change is above the proportional reading by at least 30 eV): "
          f"{A['A2']['measured_minus_proportional_eV']:+.1f} [{A['A2']['ci95'][0]:+.1f}, {A['A2']['ci95'][1]:+.1f}] eV -> "
          f"{'holds' if A['A2']['holds'] else 'does not hold'}.", "",
          "## B. KSTAR, twelve discharges with a raw stream: departures beyond three standard deviations and bursts", "",
          f"Rows inside the raw window and not within ten rows after a blip: {B['rows']:,}. Rows with a mean BES departure "
          f"above +3: {B['u_rows']}, of which {B['u_rows_with_burst']} have a detected burst within 1 ms of the time stamp "
          + (f"(share {B['share_u']:.3f})" if B["share_u"] is not None else "") + f"; base share over all rows {B['base_share']:.3f}.", ""]
    if B["B1"].get("testable"):
        L.append(f"- **B1**: ratio of the two shares {B['B1']['ratio']:.1f} [{B['B1']['ci95'][0]:.1f}, {B['B1']['ci95'][1]:.1f}] -> "
                 f"{'holds' if B['B1']['holds'] else 'does not hold'}.")
    else:
        L.append("- **B1**: not testable (fewer than 20 rows above +3).")
    L.append(f"- **B2** (reported): rows with a mean departure below -3: {B['low_rows']}; with the raw BES level at the stamp "
             f"below 0.8 of its 200 ms running median: {B['low_rows_with_dip']}"
             + (f" (share {B['B2_share_low_with_dip']:.3f})." if B["B2_share_low_with_dip"] is not None else "."))
    L += ["", "| discharge | rows | bursts | rows above +3 | of which with a burst | rows below -3 | of which with a dip |",
          "|---|---|---|---|---|---|---|"]
    for p in B["per_discharge"]:
        L.append(f"| {p['shot']} | {p['rows']} | {p['bursts']} | {p['u']} | {p['u_near']} | {p['low']} | {p['low_dip']} |")
    L += ["", "## C. MAST: T_i after a step of the SW beam while the viewed SS beam stays on", "",
          f"Shots: {C['status']}. Ratio of T_i to its level in the four frames before the step (5 ms frames; "
          "median over events of the median over chords).", ""]
    for kind, lab in (("down", "step down"), ("up", "step up")):
        for region in ("all", "core", "edge"):
            c = C["kinds"].get(kind, {}).get(region)
            if not c:
                continue
            L.append(f"- **{lab}, {region}**: {c['events']} events in {c['shots']} shots"
                     + ("" if c["testable"] else " (not testable: fewer than 30)")
                     + f". R_1 = {c['R_1']:.3f} [{c['R_1_ci95'][0]:.3f}, {c['R_1_ci95'][1]:.3f}], R_late (50 to 80 ms) = "
                     f"{c['R_late']:.3f} [{c['R_late_ci95'][0]:.3f}, {c['R_late_ci95'][1]:.3f}]; C1 direction "
                     f"{'holds' if c['C1_direction'] else 'does not hold'}; C2 delay statistic {c['delay_statistic']:+.3f} "
                     f"[{c['delay_ci95'][0]:+.3f}, {c['delay_ci95'][1]:+.3f}] -> {'holds' if c['C2_delay'] else 'does not hold'}; "
                     f"half response at frame {c['half_response_frame']}.")
    L += ["", "| frame after the step | " + " | ".join(f"{k:+d}" for k in C["frames"]) + " |",
          "|---|" + "---|" * len(C["frames"])]
    for kind, lab in (("down", "step down, all radii"), ("up", "step up, all radii")):
        c = C["kinds"].get(kind, {}).get("all")
        if c:
            L.append(f"| {lab} | " + " | ".join(f"{x:.3f}" for x in c["median_ratio"]) + " |")
    return "\n".join(L) + "\n"


def main():
    rng = np.random.default_rng(SEED)
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B32.md"}
    res["A"] = part_a(rng)
    print("A done", res["A"]["A1"], res["A"]["A2"], flush=True)
    res["B"] = part_b(rng)
    print("B done", res["B"]["B1"], flush=True)
    res["C"] = part_c(rng)
    print("C done", flush=True)
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print("written", OUT_MD)


if __name__ == "__main__":
    main()
