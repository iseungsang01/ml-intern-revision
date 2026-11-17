"""B.42 (PREREGISTRATION_B42.md): frozen estimators when the CES value arrives L frames late.

At grid row t a CES sample measured at row u is available if u <= t - 1 - L (row indices inside one
block of seq_data.build_blocks; L = 0 is the present protocol).  The six CES-derived input channels
(last available value, log1p staleness, observed flag; T_i and V_rot) are rebuilt from the available
samples only; the fast channels and the time step are those of build_blocks.  Two staleness modes:
  true    - time[t] - time[u]   (the estimator is told how old the sample is)
  nodelay - time[t] - time[u+L] (the age the sample would show if it had arrived without delay:
            the sample that becomes available at row u+1+L is presented as if measured at row u+L)
Baselines on the same available samples: hold (last available value) and the causal GP
(baselines_interpolation.predict_gp_causal, the 16 last available samples, at the stamp of row t).
Networks (frozen, CPU): causal variant data/.b29_norm0_cut_s{s} (v2, train-file standardization)
and the history-only twin data/.b29a1_v2h_cut_s{s} (v2h, per-discharge standardization as trained).
Scored rows: T_i measured at row t, at least one available T_i sample in the block.

Checks before the readout (the run stops if either fails):
  1. at L = 0 the rebuilt inputs equal build_blocks (np.allclose) on every test file of every split;
  2. at L = 0 the causal variant, hold and the GP agree with the frozen row table
     data/.b30/rows_cut_s{s}.npz (k = 1) within 0.5 eV on >= 99 % of the rows of five test files.

CPU only, nothing is trained.  Run from the repo root:
    py ces_prediction/experiments/b42_delay/run_b42.py            # rows + readout
    py ces_prediction/experiments/b42_delay/run_b42.py check      # the two checks only (five files per split)
    py ces_prediction/experiments/b42_delay/run_b42.py analyze    # readout from the cached rows
Row cache: data/.b42/rows_s{seed}.npz (gitignored).  Output: docs/paper/b42_delay.json / .md
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b30_thinned"))
sys.path.insert(0, str(HERE.parent / "b37_plan"))
sys.path.insert(0, str(HERE.parent / "b34_error_model"))
import run_b30 as R  # noqa: E402
from run_b37 import Ctx  # noqa: E402
from run_b34 import ci, wq  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b42_delay.json"
OUT_MD = R.REPO / "docs" / "paper" / "b42_delay.md"
CACHE = R.DATA / ".b42"
LS = (0, 1, 2, 4)
MODES = ("true", "nodelay")
N_BOOT, BOOT_SEED = 2000, 20261118
CHECK_EV, CHECK_SHARE, CHECK_FILES = 0.5, 0.99, 5
EST = ("hold", "gp", "twin", "causal", "twin_nd", "causal_nd")


def twin_net(C, s):
    run = R.DATA / f".b29a1_v2h_cut_s{s}"
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if m["seq_model"] != "v2h" or not bool(m["per_shot_input_norm"]) or m.get("per_shot_norm_expanding") \
            or m.get("zero_mc") or m.get("extra_vt_channels") or m.get("extra_ti_channels") or m.get("train_ctx"):
        raise SystemExit(f"FATAL: {run.name} is not the history-only twin")
    stats = C.load_stats(m)
    model = C.MODELS["v2h"]()
    model.load_state_dict(C.torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
    model.eval()
    return model, stats


def ces_channels(time, tgt_z, obs, L, mode):
    """(n, 6) CES channels of one block under delay L (the rule of seq_data.build_blocks, lines
    295-311, with 'rows < i' replaced by 'rows <= i - 1 - L')."""
    n = len(time)
    out = np.zeros((n, 6))
    i = np.arange(n)
    for t in range(2):
        o = np.flatnonzero(obs[:, t])
        if o.size == 0:
            continue
        p = np.searchsorted(o, i - L, side="left") - 1          # last u with u < i - L
        has = p >= 0
        u = o[np.where(has, p, 0)]
        ref = u if mode == "true" else u + L
        out[:, 3 * t] = np.where(has, tgt_z[u, t], 0.0)
        out[:, 3 * t + 1] = np.where(has, np.log1p(np.maximum(time[i] - time[np.minimum(ref, n - 1)], 0.0)), 0.0)
        out[:, 3 * t + 2] = has.astype(float)
    return out


def file_inputs(C, arr, net, per_shot):
    """build_blocks blocks of one file plus what is needed to rebuild the CES channels."""
    model, stats = net
    blks = C.build_blocks(arr, C.dims, stats, per_shot_norm=per_shot, expanding=False)
    tm = arr[:, 0].astype(np.float64)
    spans = R.block_spans(tm, C.GAP)
    if len(blks) != len(spans):
        raise SystemExit("FATAL: block spans do not match build_blocks")
    tgt = arr[:, 1:3].astype(np.float64)
    tz = (tgt - stats["target"]["mean"].astype(np.float64)) / stats["target"]["std"].astype(np.float64)
    return blks, spans, tm, tgt, tz


def run_net(C, net, blks, spans, tm, tgt, tz, L, mode, n):
    model, stats = net
    base = C.dims["n_fast"] + 1
    mean, std = float(stats["target"]["mean"][0]), float(stats["target"]["std"][0])
    out = np.full(n, np.nan)
    with C.torch.no_grad():
        for blk, (a, b) in zip(blks, spans):
            x = blk["x"].copy()
            x[:, base:base + 6] = ces_channels(tm[a:b], tz[a:b], ~np.isnan(tgt[a:b]), L, mode)
            out[a:b] = model(C.torch.from_numpy(x).unsqueeze(0))[0][:, 0].numpy() * std + mean
    return out


def input_check(C, blks, spans, tm, tgt, tz):
    base = C.dims["n_fast"] + 1
    ok = True
    for blk, (a, b) in zip(blks, spans):
        x = blk["x"].copy()
        x[:, base:base + 6] = ces_channels(tm[a:b], tz[a:b], ~np.isnan(tgt[a:b]), 0, "true")
        ok &= bool(np.allclose(x, blk["x"], rtol=1e-6, atol=1e-6))
    return ok


def rows_split(C, s, limit=None):
    t0 = _time.time()
    man = C.manifest(s)
    test = sorted(C.names.index(nm) for nm in man["test_files"] if nm in C.grid)[:limit]
    nets = {"causal": (C.net(s), False), "twin": (twin_net(C, s), True)}
    acc = {L: {c: [] for c in ("file", "row", "y", "age_ms", *EST)} for L in LS}
    in_ok = {"causal": 0, "twin": 0}
    for fi in test:
        arr = C.grid[C.names[fi]]
        n = len(arr)
        pred = {}
        for nm, (net, per_shot) in nets.items():
            blks, spans, tm, tgt, tz = file_inputs(C, arr, net, per_shot)
            if not input_check(C, blks, spans, tm, tgt, tz):
                raise SystemExit(f"FATAL: L = 0 rebuilt inputs differ from build_blocks ({nm}, {C.names[fi]})")
            in_ok[nm] += 1
            for L in LS:
                for mode in MODES:
                    if mode == "nodelay" and L == 0:
                        pred[(nm, L, mode)] = pred[(nm, 0, "true")]
                        continue
                    pred[(nm, L, mode)] = run_net(C, net, blks, spans, tm, tgt, tz, L, mode, n)
        y = tgt[:, 0]
        for L in LS:
            A = acc[L]
            for a, b in spans:
                obs = np.flatnonzero(~np.isnan(y[a:b])) + a
                for r in obs:
                    av = obs[obs <= r - 1 - L]
                    if av.size == 0:
                        continue
                    last = av[-R.GP_SIDE:]
                    A["file"].append(fi); A["row"].append(r); A["y"].append(y[r])
                    A["age_ms"].append((tm[r] - tm[av[-1]]) * 1000.0)
                    A["hold"].append(y[av[-1]])
                    A["gp"].append(C.B.predict_gp_causal(tm[last], y[last], tm[r]))
                    A["causal"].append(pred[("causal", L, "true")][r])
                    A["twin"].append(pred[("twin", L, "true")][r])
                    A["causal_nd"].append(pred[("causal", L, "nodelay")][r])
                    A["twin_nd"].append(pred[("twin", L, "nodelay")][r])
        print(f"[{_time.time() - t0:5.0f}s] s{s} file {fi} done", flush=True)
    out = {"input_check_files": in_ok}
    for L in LS:
        for c, v in acc[L].items():
            out[f"L{L}__{c}"] = np.asarray(v, dtype=np.int32 if c in ("file", "row") else np.float64)
    return out


def table_check(C, s, rows):
    z = np.load(R.ROWS_DIR / f"rows_cut_s{s}.npz")
    tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
    k1 = tab["k"] == 1
    files = sorted(np.unique(tab["file"][k1]).tolist())[:CHECK_FILES]
    mine = {(int(f), int(r)): j for j, (f, r) in enumerate(zip(rows["L0__file"], rows["L0__row"]))}
    cnt, dev = 0, {"causal": 0, "hold": 0, "gp": 0, "twin": 0, "any_of_causal_hold_gp": 0, "missing": 0}
    for f in files:
        m = k1 & (tab["file"] == f)
        for r, c, h, g, tw in zip(tab["row"][m], tab["causal"][m], tab["hold"][m], tab["gp"][m], tab["hist80"][m]):
            cnt += 1
            j = mine.get((int(f), int(r)))
            if j is None:
                dev["missing"] += 1; dev["any_of_causal_hold_gp"] += 1
                continue
            d = {e: abs(rows[f"L0__{e}"][j] - v) for e, v in (("causal", c), ("hold", h), ("gp", g), ("twin", tw))}
            for e in d:
                dev[e] += int(not d[e] <= CHECK_EV)
            dev["any_of_causal_hold_gp"] += int(max(d["causal"], d["hold"], d["gp"]) > CHECK_EV or not np.isfinite(d["gp"]))
    share = 1 - dev["any_of_causal_hold_gp"] / max(cnt, 1)
    return {"files": [int(f) for f in files], "rows": cnt, "deviating": dev, "share_within_0.5_eV": share,
            "passes": bool(cnt > 0 and share >= CHECK_SHARE)}


# ----------------------------------------------------------------------------------- readout
def analyze_split(rows, W, uf):
    pos = {f: k for k, f in enumerate(uf)}
    nf = len(uf)
    res = {}
    for L in LS:
        f = np.array([pos[int(x)] for x in rows[f"L{L}__file"]])
        y = rows[f"L{L}__y"]
        e = {k: rows[f"L{L}__{k}"] - y for k in EST}
        fin = np.all([np.isfinite(v) for v in e.values()], axis=0)
        f, e = f[fin], {k: v[fin] for k, v in e.items()}
        sse = {k: np.bincount(f, v ** 2, nf) for k, v in e.items()}
        cnt = np.bincount(f, None, nf).astype(float)
        bs = {k: W @ v for k, v in sse.items()}
        bn = W @ cnt
        Wr = W[:, f]
        o = {"rows": int(fin.sum()), "rows_nonfinite_dropped": int((~fin).sum()),
             "discharges": int((cnt > 0).sum()),
             "median_age_ms": float(np.median(rows[f"L{L}__age_ms"][fin])),
             "rmse": {k: [float(np.sqrt(sse[k].sum() / cnt.sum())), *ci(np.sqrt(bs[k] / bn))] for k in EST},
             "median_abs": {k: [float(np.median(np.abs(e[k]))), *ci(wq(np.abs(e[k]), Wr))] for k in EST},
             "skill": {}, "_boot": {}}
        for a, b in (("causal", "hold"), ("causal", "gp"), ("causal", "twin"), ("twin", "gp"), ("twin", "hold"),
                     ("gp", "hold"), ("causal_nd", "hold"), ("causal_nd", "gp"), ("causal_nd", "twin_nd"),
                     ("causal_nd", "causal"), ("twin_nd", "twin")):
            b_ = 1 - bs[a] / bs[b]
            o["skill"][f"{a}_vs_{b}"] = [float(1 - sse[a].sum() / sse[b].sum()), *ci(b_)]
            o["_boot"][f"{a}_vs_{b}"] = b_
        o["_boot"]["rmse_causal"] = np.sqrt(bs["causal"] / bn)
        res[L] = o
    for L in LS:
        o, z = res[L], res[0]
        o["increment_minus_L0"] = [o["skill"]["causal_vs_twin"][0] - z["skill"]["causal_vs_twin"][0],
                                   *ci(o["_boot"]["causal_vs_twin"] - z["_boot"]["causal_vs_twin"])]
        o["skill_over_hold_minus_L0"] = [o["skill"]["causal_vs_hold"][0] - z["skill"]["causal_vs_hold"][0],
                                         *ci(o["_boot"]["causal_vs_hold"] - z["_boot"]["causal_vs_hold"])]
        o["rmse_causal_ratio_to_L0"] = [o["rmse"]["causal"][0] / z["rmse"]["causal"][0],
                                        *ci(o["_boot"]["rmse_causal"] / z["_boot"]["rmse_causal"])]
    for L in LS:
        del res[L]["_boot"]
    return {str(L): v for L, v in res.items()}


def analyze(checks):
    rng = np.random.default_rng(BOOT_SEED)
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B42.md", "delays_frames": list(LS),
           "n_boot": N_BOOT, "checks": checks, "splits": {}}
    for s in R.SEEDS:
        z = np.load(CACHE / f"rows_s{s}.npz")
        rows = {k: z[k] for k in z.files}
        uf = np.unique(rows["L0__file"])
        W = rng.multinomial(len(uf), np.full(len(uf), 1.0 / len(uf)), size=N_BOOT).astype(float)
        res["splits"][str(s)] = analyze_split(rows, W, uf)
    sp = res["splits"]
    t1 = {str(L): sum(sp[str(s)][str(L)]["skill"]["causal_vs_gp"][1] > 0 for s in R.SEEDS) for L in (1, 2)}
    t2 = sum(sp[str(s)]["2"]["skill"]["causal_vs_twin"][1] > 0 for s in R.SEEDS)
    res["verdicts"] = {"T1": {"splits_above_0": t1, "holds": bool(all(v >= 3 for v in t1.values()))},
                       "T2": {"splits_above_0": t2, "holds": bool(t2 >= 3)}}
    ex = {}
    r = [sp[str(s)]["2"]["rmse_causal_ratio_to_L0"][0] for s in R.SEEDS]
    ex["rmse_causal_L2_over_L0_in_1.08_1.25"] = {"values": r, "n_in": sum(1.08 <= v <= 1.25 for v in r)}
    for L, lo, hi in ((1, 0.10, 0.22), (2, 0.12, 0.28)):
        v = [sp[str(s)][str(L)]["skill"]["causal_vs_gp"][0] for s in R.SEEDS]
        ex[f"skill_over_gp_L{L}_in_{lo}_{hi}"] = {"values": v, "n_in": sum(lo <= x <= hi for x in v)}
    v = [sp[str(s)]["2"]["skill"]["causal_vs_twin"][0] for s in R.SEEDS]
    ex["increment_L2_in_0.06_0.16"] = {"values": v, "n_in": sum(0.06 <= x <= 0.16 for x in v)}
    v = [sp[str(s)]["2"]["increment_minus_L0"][0] for s in R.SEEDS]
    ex["increment_L2_larger_than_L0_on_3_splits"] = {"values": v, "n_in": sum(x > 0 for x in v)}
    v = [sp[str(s)]["2"]["skill_over_hold_minus_L0"][0] for s in R.SEEDS]
    ex["skill_over_hold_lower_at_L2"] = {"values": v, "n_in": sum(x < 0 for x in v)}
    res["expectations"] = ex
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print(json.dumps(res["verdicts"]), flush=True)


def render(res):
    s3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                   # noqa: E731
    f1 = lambda x: f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"                      # noqa: E731
    f3 = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"                      # noqa: E731
    sp, V = res["splits"], res["verdicts"]
    L = ["# B.42 — frozen estimators with a dense but delayed CES history (registered: PREREGISTRATION_B42.md)", "",
         "CPU, no network trained. T_i, cut population, four file-level splits, test discharges. A CES sample measured at "
         "row u is available at row t if u <= t - 1 - L (L in grid rows of 10 ms; L = 0 is the present protocol). "
         "causal = the causal variant (`.b29_norm0_cut`), twin = the history-only twin (`.b29a1_v2h_cut`), both frozen and "
         "trained without delay; their CES input channels are rebuilt from the available samples with the true staleness. "
         "`_nd` = the same networks with the staleness left as if there were no delay (age of the sample counted from row "
         "u + L). hold = last available value; gp = causal GP on the 16 last available samples. Scored rows: T_i measured at "
         "row t with at least one available sample in the block; the same rows for every estimator at a given L. "
         "Skill a vs b = 1 - SSE_a / SSE_b. Intervals: 95 %, 2,000 resamples of test discharges (the same resamples for every "
         "L of a split, so differences between delays are paired). A statement holds if it holds on at least three of four splits.", "",
         "## Checks before the readout", ""]
    for s in R.SEEDS:
        c = res["checks"][str(s)]
        L.append(f"- split {s}: rebuilt L = 0 inputs equal `build_blocks` (np.allclose) on {c['input']['causal']} test files "
                 f"(causal) and {c['input']['twin']} (twin); row table, files {c['table']['files']}: "
                 f"{c['table']['share_within_0.5_eV']:.4f} of {c['table']['rows']:,} rows within 0.5 eV for causal, hold and GP "
                 f"(deviating: {c['table']['deviating']}) -> {'passes' if c['table']['passes'] else 'FAILS'}.")
    t1 = V["T1"]["splits_above_0"]
    L += ["", "## Registered statements", "",
          f"- **T1** (skill of the causal variant over the causal GP, interval above zero, at L = 1 and at L = 2): "
          f"L = 1 on {t1['1']} of 4, L = 2 on {t1['2']} of 4 -> **{'holds' if V['T1']['holds'] else 'does not hold'}**.",
          f"- **T2** (skill of the causal variant over the history-only twin at L = 2, interval above zero): "
          f"{V['T2']['splits_above_0']} of 4 -> **{'holds' if V['T2']['holds'] else 'does not hold'}**.", "",
          "| split | L | causal vs GP | causal vs twin |", "|---|---|---|---|"]
    for s in R.SEEDS:
        for Lx in ("1", "2"):
            o = sp[str(s)][Lx]["skill"]
            L.append(f"| {s} | {Lx} | {s3(o['causal_vs_gp'])} | {s3(o['causal_vs_twin'])} |")
    L += ["", "## Root-mean-square error (eV) by delay", "",
          "| split | L | rows | median age of the last available sample (ms) | hold | GP | twin | causal | twin_nd | causal_nd | "
          "causal, ratio to L = 0 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        for Lx in map(str, LS):
            o = sp[str(s)][Lx]
            L.append(f"| {s} | {Lx} | {o['rows']:,} | {o['median_age_ms']:.0f} | " +
                     " | ".join(f1(o["rmse"][e]) for e in EST[:4]) + " | " +
                     " | ".join(f1(o["rmse"][e]) for e in EST[4:]) + f" | {f3(o['rmse_causal_ratio_to_L0'])} |")
    L += ["", "## Skill by delay (true staleness)", "",
          "| split | L | causal vs hold | causal vs GP | causal vs twin (fast-channel increment) | increment minus L = 0 | "
          "causal vs hold minus L = 0 | twin vs GP | GP vs hold |", "|---|---|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        for Lx in map(str, LS):
            o = sp[str(s)][Lx]
            k = o["skill"]
            L.append(f"| {s} | {Lx} | {s3(k['causal_vs_hold'])} | {s3(k['causal_vs_gp'])} | {s3(k['causal_vs_twin'])} | "
                     f"{s3(o['increment_minus_L0'])} | {s3(o['skill_over_hold_minus_L0'])} | {s3(k['twin_vs_gp'])} | "
                     f"{s3(k['gp_vs_hold'])} |")
    L += ["", "## Staleness left as if there were no delay (reported)", "",
          "| split | L | causal_nd vs hold | causal_nd vs GP | causal_nd vs twin_nd | causal_nd vs causal | twin_nd vs twin |",
          "|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        for Lx in ("1", "2", "4"):
            k = sp[str(s)][Lx]["skill"]
            L.append(f"| {s} | {Lx} | {s3(k['causal_nd_vs_hold'])} | {s3(k['causal_nd_vs_gp'])} | "
                     f"{s3(k['causal_nd_vs_twin_nd'])} | {s3(k['causal_nd_vs_causal'])} | {s3(k['twin_nd_vs_twin'])} |")
    L += ["", "## Median absolute error (eV)", "",
          "| split | L | hold | GP | twin | causal | twin_nd | causal_nd |", "|---|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        for Lx in map(str, LS):
            o = sp[str(s)][Lx]
            L.append(f"| {s} | {Lx} | " + " | ".join(f1(o["median_abs"][e]) for e in EST) + " |")
    L += ["", "## Expectations of the registration (section 6)", ""]
    for k, v in res["expectations"].items():
        L.append(f"- {k}: " + ", ".join(f"{x:+.3f}" for x in v["values"]) + f" -> {v['n_in']} of 4")
    return "\n".join(L) + "\n"


def main():
    if sys.argv[1:] == ["analyze"]:
        analyze(json.loads((CACHE / "checks.json").read_text(encoding="utf-8")))
        return
    C = Ctx()
    if sys.argv[1:] == ["check"]:          # the checks alone, on the five check files of every split
        for s in R.SEEDS:
            rows = rows_split(C, s, CHECK_FILES)
            print(f"s{s} check: inputs {rows['input_check_files']}, table {json.dumps(table_check(C, s, rows))}",
                  flush=True)
        return
    CACHE.mkdir(exist_ok=True)
    checks = {}
    for s in R.SEEDS:
        rows = rows_split(C, s)
        tc = table_check(C, s, rows)
        print(f"s{s} check: inputs {rows['input_check_files']}, table {json.dumps(tc)}", flush=True)
        checks[str(s)] = {"input": rows.pop("input_check_files"), "table": tc}
        (CACHE / "checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
        if not tc["passes"]:
            raise SystemExit(f"FATAL: row-table check failed on split {s}; the run stops")
        np.savez_compressed(CACHE / f"rows_s{s}.npz", **rows)
    analyze(checks)


if __name__ == "__main__":
    main()
