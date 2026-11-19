"""B.70 phase 2: the delay replay (L = 0, 1, 2) and the 50 ms thinning replay (k = 5) on the D1 rows, as forward runs
of the r2-r4 networks of B.39 (PREREGISTRATION_B70.md section 3 and Addendum 2, C8-C9).

Delay (run_b42's rule and functions, imported): a T_i sample at grid row u is available at row t if u <= t - 1 - L;
the six CES channels are rebuilt by run_b42.ces_channels (true staleness; the no-delay staleness reported);
hold = last available value; causal GP = predict_gp_causal on the 16 last available samples.  Fixed row set (all L):
the D1 rows whose block holds a T_i at u <= t - 3.
Thinning (run_b30.rows at k = 5, its row loop copied): every fifth row of a block is visible (phases o = 0..4); the
networks and the history see the visible values only; GP length scales x 5 (unscaled reported); PCHIP on up to four
visible values each side; hold = last visible value.  Rows: the D1 rows; a row enters a phase when it has a visible
earlier T_i in its block (the B.30 population also needs a visible later one and every estimator finite).
References depend only on the discharge: computed once per discharge, then paired with the network and twin of the fold
in which the discharge is a test discharge, in each of r2, r3, r4.

    py ces_prediction/experiments/b70_d1/p2_replay.py rows [--part i/n]   # per discharge -> data/.b70p2/replay/
    py ces_prediction/experiments/b70_d1/p2_replay.py readout              # -> data/.b70p2/paper/b70_replay.{json,md}
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
for sub in ("b30_thinned", "b37_plan", "b34_error_model", "b42_delay", "b64_two_stage"):
    sys.path.insert(0, str(EXP / sub))
import runs_b70 as RB  # noqa: E402
import run_b30 as R  # noqa: E402
import run_b42 as B42  # noqa: E402

OUT = RB.DATA / ".b70p2" / "replay"
PAPER = RB.DATA / ".b70p2" / "paper"
REPEATS, K = (2, 3, 4), 5
LS = (0, 1, 2)
KTHIN = 5
T = "CES_TI"
CHECK_EV, CHECK_SHARE = 0.5, 0.99
N_BOOT, SEED = 20_000, 20261170
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def d1(model, r, k):
    return RB.OUT_ROOT / RB.B39.pooled_run_dir(model, r, k).name.lstrip(".")


def load_net(C, model, r, k):
    run = d1(model, r, k)
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if m["seq_model"] != model or m.get("per_shot_norm_expanding") or m.get("zero_mc") or m.get("extra_vt_channels") \
            or m.get("extra_ti_channels") or m.get("train_ctx"):
        raise SystemExit(f"FATAL: {run.name} is not the registered {model}")
    model_ = C.MODELS[model]()
    model_.load_state_dict(C.torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
    model_.eval()
    return (model_, C.load_stats(m)), bool(m["per_shot_input_norm"])


def fold_of():
    """file name -> {r: k} (the fold in which the discharge is a test discharge)."""
    out = {}
    for r in REPEATS:
        for k in range(K):
            man = json.loads((RB.B27.split_dir(r, k) / "split_manifest.json").read_text(encoding="utf-8"))
            for n in man["test_files"]:
                out.setdefault(n, {})[r] = k
    return out


def d1_rows(C, valid):
    """file name -> (grid rows of its D1 T_i rows, and per repeat the D1 npz errors of network / twin / references)."""
    rows = {}
    for r in REPEATS:
        for k in range(K):
            zc = np.load(d1("v2", r, k) / "comparison_errors_test.npz")
            zt = np.load(d1("v2h", r, k) / "comparison_errors_test.npz")
            shot, row = zc[f"{T}_shot"], zc[f"{T}_row"]
            for u in np.unique(shot):
                sel = np.flatnonzero(shot == u)
                nm = valid[int(u)]
                gi = np.flatnonzero(np.isfinite(C.grid[nm][:, 1:3]).any(1))
                g = gi[row[sel]]
                rec = rows.setdefault(nm, {"g": g, "rep": {}})
                if not np.array_equal(rec["g"], g):
                    raise SystemExit(f"FATAL: {nm}: D1 rows differ between re-partitions")
                rec["rep"][r] = {"se_c": zc[f"{T}_se_model"][sel], "se_t": zt[f"{T}_se_model"][sel],
                                 "se_hold": zc[f"{T}_se_persistence"][sel], "se_gp": zc[f"{T}_se_gp_causal"][sel],
                                 "y": zc[f"{T}_y_true"][sel]}
    return rows


def one_file(C, nets, nm, rec, folds):
    arr = C.grid[nm]
    n = len(arr)
    tm = arr[:, 0].astype(np.float64)
    y = arr[:, 1].astype(np.float64)
    spans = R.block_spans(tm, C.GAP)
    span_of = np.full(n, -1)
    for j, (a, b) in enumerate(spans):
        span_of[a:b] = j
    G = rec["g"]
    out = {"g": G, "y": y[G]}
    # ------------------------------------------------ delay: fixed rows and references
    fixed, refs = [], {L: {"hold": [], "gp": [], "age_ms": []} for L in LS}
    for i, r in enumerate(G):
        j = span_of[r]
        if j < 0:
            continue
        a, b = spans[j]
        obs = np.flatnonzero(~np.isnan(y[a:b])) + a
        if obs[obs <= r - 3].size == 0:
            continue
        fixed.append(i)
        for L in LS:
            av = obs[obs <= r - 1 - L]
            last = av[-R.GP_SIDE:]
            refs[L]["hold"].append(y[av[-1]])
            refs[L]["gp"].append(C.B.predict_gp_causal(tm[last], y[last], tm[r]))
            refs[L]["age_ms"].append((tm[r] - tm[av[-1]]) * 1000.0)
    fixed = np.array(fixed, np.int64)
    out["d_idx"] = fixed
    for L in LS:
        for q, v in refs[L].items():
            out[f"d_L{L}_{q}"] = np.array(v, np.float64)
    # ------------------------------------------------ thinning: references per phase (B.30 row loop, D1 rows only)
    base_grid = tuple(C.B._GP_LS_GRID)
    th = {c: [] for c in ("i", "o", "age_ms", "has_future", "hold", "gp", "gp_unscaled", "pchip")}
    vis_by_o = {}
    for o in range(KTHIN):
        vis = R.visible_rows(n, spans, KTHIN, o)
        vis_by_o[o] = vis
        for i, r in enumerate(G):
            j = span_of[r]
            if j < 0:
                continue
            a, b = spans[j]
            obs = np.flatnonzero(~np.isnan(y[a:b])) + a
            vo = obs[vis[obs]]
            if vo.size == 0:
                continue
            tv, vv = tm[vo], y[vo]
            pos = int(np.searchsorted(vo, r))
            if pos == 0:
                continue
            fut0 = pos + 1 if (pos < vo.size and vo[pos] == r) else pos
            has_future = fut0 < vo.size
            tr = tm[r]
            lo = max(0, pos - R.GP_SIDE)
            try:
                C.B._GP_LS_GRID = tuple(ls * KTHIN for ls in base_grid)
                gp = C.B.predict_gp_causal(tv[lo:pos], vv[lo:pos], tr)
            finally:
                C.B._GP_LS_GRID = base_grid
            gp0 = C.B.predict_gp_causal(tv[lo:pos], vv[lo:pos], tr)
            pch = np.nan
            if has_future:
                idx = [*range(max(0, pos - 4), pos), *range(fut0, min(vo.size, fut0 + 4))]
                pch = C.B.predict_pchip(tv[idx], vv[idx], tr)
            th["i"].append(i); th["o"].append(o); th["age_ms"].append((tr - tv[pos - 1]) * 1000.0)
            th["has_future"].append(has_future); th["hold"].append(vv[pos - 1]); th["gp"].append(gp)
            th["gp_unscaled"].append(gp0); th["pchip"].append(pch)
    for c, v in th.items():
        out[f"t_{c}"] = np.array(v, np.int64 if c in ("i", "o") else bool if c == "has_future" else np.float64)
    # ------------------------------------------------ networks of each re-partition
    tgt = arr[:, 1:3].astype(np.float64)
    for r in REPEATS:
        k = folds[nm][r]
        for role, model in (("c", "v2"), ("t", "v2h")):
            net, per_shot = nets[(model, r, k)]
            blks, sp2, _tm, _tgt, tz = B42.file_inputs(C, arr, net, per_shot)
            if not B42.input_check(C, blks, sp2, _tm, _tgt, tz):
                raise SystemExit(f"FATAL: {nm} r{r}: rebuilt L = 0 inputs differ from build_blocks")
            for L in LS:
                for mode in ("true", "nodelay"):
                    if mode == "nodelay" and L == 0:
                        continue
                    p = B42.run_net(C, net, blks, sp2, _tm, _tgt, tz, L, mode, n)
                    out[f"d_r{r}_{role}_L{L}{'' if mode == 'true' else '_nd'}"] = p[G[fixed]]
            model_, stats = net
            mean, std = float(stats["target"]["mean"][0]), float(stats["target"]["std"][0])
            pt = np.full(len(th["i"]), np.nan)
            ii, oo = out["t_i"], out["t_o"]
            for o in range(KTHIN):
                arr_h = arr.copy()
                arr_h[~vis_by_o[o], 1:3] = np.nan
                bl = C.build_blocks(arr_h, C.dims, stats, per_shot_norm=per_shot, expanding=False)
                if len(bl) != len(spans):
                    raise SystemExit("FATAL: block spans do not match build_blocks")
                pr = np.full(n, np.nan)
                with C.torch.no_grad():
                    for blk, (a, b) in zip(bl, spans):
                        pr[a:b] = model_(C.torch.from_numpy(blk["x"]).unsqueeze(0))[0][:, 0].numpy() * std + mean
                sel = oo == o
                pt[sel] = pr[G[ii[sel]]]
            out[f"t_r{r}_{role}"] = pt
        # D1 references and errors of the repeat at the fixed rows (check at L = 0)
        for q, v in rec["rep"][r].items():
            out[f"npz_r{r}_{q}"] = np.asarray(v, np.float64)
    return out


def stage_rows(part):
    C = R37Ctx()
    valid = [nm for nm in sorted(C.grid) if np.isfinite(C.grid[nm][:, 1:3]).any()]
    folds = fold_of()
    rows = d1_rows(C, valid)
    names = sorted(rows)
    i, nparts = (int(x) for x in part.split("/"))
    names = names[i::nparts]
    nets = {}
    for r in REPEATS:
        for k in range(K):
            for model in ("v2", "v2h"):
                nets[(model, r, k)] = load_net(C, model, r, k)
    OUT.mkdir(parents=True, exist_ok=True)
    done = 0
    for nm in names:
        p = OUT / f"{nm[:-4]}.npz"
        if p.exists():
            continue
        t0 = _time.time()
        res = one_file(C, nets, nm, rows[nm], folds)
        tmp = OUT / f"{nm[:-4]}.tmp.npz"
        np.savez(tmp, **res)
        os.replace(tmp, p)
        done += 1
        if done % 20 == 0:
            log(f"part {part}: {done} files ({_time.time() - t0:.1f} s for the last)")
    log(f"part {part}: done ({done} written)")


def R37Ctx():
    import run_b37
    C = run_b37.Ctx()
    C.torch.set_num_threads(2)
    return C


# ------------------------------------------------------------------------------------------ readout
def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def stage_readout():
    import stage2_b64 as S2
    files = sorted(OUT.glob("s*.npz"))
    folds = fold_of()
    D_delay, D_thin = {}, {}
    chk = {"rows_L0": 0, "within_0.5_eV": {"c": 0, "hold": 0, "gp": 0}, "fixed_rows_per_repeat": 0}
    for r in REPEATS:
        cols = {}
        tcol = {}
        for p in files:
            z = np.load(p)
            fi = int(p.stem[1:])
            idx = z["d_idx"]
            y = z["y"][idx]
            e = {"y": y, "f": np.full(len(idx), fi)}
            for L in LS:
                e[f"c{L}"] = (z[f"d_r{r}_c_L{L}"] - y) ** 2
                e[f"t{L}"] = (z[f"d_r{r}_t_L{L}"] - y) ** 2
                e[f"hold{L}"] = (z[f"d_L{L}_hold"] - y) ** 2
                e[f"gp{L}"] = (z[f"d_L{L}_gp"] - y) ** 2
                if L > 0:
                    e[f"cnd{L}"] = (z[f"d_r{r}_c_L{L}_nd"] - y) ** 2
                    e[f"tnd{L}"] = (z[f"d_r{r}_t_L{L}_nd"] - y) ** 2
            # check at L = 0 against the D1 npz on the fixed rows
            dc = np.abs(np.abs(z[f"d_r{r}_c_L0"] - y) - np.sqrt(z[f"npz_r{r}_se_c"][idx]))
            dh = np.abs(np.abs(z["d_L0_hold"] - y) - np.sqrt(z[f"npz_r{r}_se_hold"][idx]))
            dg = np.abs(np.abs(z["d_L0_gp"] - y) - np.sqrt(z[f"npz_r{r}_se_gp"][idx]))
            chk["rows_L0"] += len(idx)
            chk["within_0.5_eV"]["c"] += int((dc <= CHECK_EV).sum())
            chk["within_0.5_eV"]["hold"] += int((dh <= CHECK_EV).sum())
            chk["within_0.5_eV"]["gp"] += int((dg <= CHECK_EV).sum())
            for q, v in e.items():
                cols.setdefault(q, []).append(v)
            # thinning: B.30 population (has_future, every estimator finite)
            ii = z["t_i"]
            yt = z["y"][ii]
            et = {"y": yt, "f": np.full(len(ii), fi), "hf": z["t_has_future"],
                  "c": (z[f"t_r{r}_c"] - yt) ** 2, "t": (z[f"t_r{r}_t"] - yt) ** 2, "hold": (z["t_hold"] - yt) ** 2,
                  "gp": (z["t_gp"] - yt) ** 2, "gp0": (z["t_gp_unscaled"] - yt) ** 2, "pchip": (z["t_pchip"] - yt) ** 2}
            for q, v in et.items():
                tcol.setdefault(q, []).append(v)
        d = {q: np.concatenate(v) for q, v in cols.items()}
        u, f = np.unique(d["f"], return_inverse=True)
        D_delay[r] = {"f": f, "n": len(u), "y": d["y"], "se": {q: d[q] for q in d if q not in ("y", "f")}}
        D_delay[r]["keep_trim"] = d["hold0"] < np.quantile(d["hold0"], 0.99)
        t_ = {q: np.concatenate(v) for q, v in tcol.items()}
        pop = t_["hf"] & np.all([np.isfinite(t_[q]) for q in ("c", "t", "hold", "gp", "pchip")], axis=0)
        u2, f2 = np.unique(t_["f"][pop], return_inverse=True)
        D_thin[r] = {"f": f2, "n": len(u2), "y": t_["y"][pop],
                     "se": {q: t_[q][pop] for q in ("c", "t", "hold", "gp", "gp0", "pchip")}}
        D_thin[r]["keep_trim"] = D_thin[r]["se"]["hold"] < np.quantile(D_thin[r]["se"]["hold"], 0.99)
        chk["fixed_rows_per_repeat"] = int(len(d["y"]))
        chk.setdefault("thinning_rows_per_repeat", {})[str(r)] = int(pop.sum())
        chk.setdefault("thinning_rows_all_phases_per_repeat", {})[str(r)] = int(len(t_["y"]))
    for q in chk["within_0.5_eV"]:
        chk["within_0.5_eV"][q] = chk["within_0.5_eV"][q] / max(chk["rows_L0"], 1)
    chk["passes"] = bool(all(v >= CHECK_SHARE for v in chk["within_0.5_eV"].values()))
    log(f"check at L = 0: {chk}")
    dpairs = []
    for L in LS:
        dpairs += [(f"c{L}", f"gp{L}"), (f"c{L}", f"t{L}"), (f"c{L}", f"hold{L}"), (f"t{L}", f"gp{L}"), (f"gp{L}", f"hold{L}")]
        if L > 0:
            dpairs += [(f"cnd{L}", f"gp{L}"), (f"cnd{L}", f"tnd{L}"), (f"cnd{L}", f"c{L}")]
    dests = tuple(sorted({x for p in dpairs for x in p}))
    Ad, Adper = S2.cell(D_delay, lambda r: np.ones(len(D_delay[r]["y"]), bool), dests, dpairs, rng_for("delay"),
                        nboot=N_BOOT, reps=REPEATS)
    log("delay cell done")
    tpairs = [("c", "gp"), ("c", "t"), ("c", "pchip"), ("c", "hold"), ("t", "gp"), ("c", "gp0")]
    At, Atper = S2.cell(D_thin, lambda r: np.ones(len(D_thin[r]["y"]), bool), ("c", "t", "hold", "gp", "gp0", "pchip"),
                        tpairs, rng_for("thin"), nboot=N_BOOT, reps=REPEATS)
    log("thinning cell done")
    st = {"T1_L1": {"text": "delay L = 1: network over the causal GP, interval above zero", "value": Ad["skill_mse|c1|gp1"]},
          "T1_L2": {"text": "delay L = 2: network over the causal GP, interval above zero", "value": Ad["skill_mse|c2|gp2"]},
          "T2": {"text": "delay L = 2: network over the twin, interval above zero", "value": Ad["skill_mse|c2|t2"]},
          "H1": {"text": "thinning to 50 ms (k = 5): network over the causal GP, interval above zero", "value": At["skill_mse|c|gp"]},
          "H2": {"text": "thinning to 50 ms (k = 5): network over the twin, interval above zero", "value": At["skill_mse|c|t"]}}
    for s in st.values():
        s["holds"] = bool(s["value"][1] > 0)
    st["T1"] = {"text": "T1 = T1_L1 and T1_L2", "holds": bool(st["T1_L1"]["holds"] and st["T1_L2"]["holds"])}
    rmse_ratio = {f"L{L}": float(np.mean([np.sqrt(D_delay[r]["se"][f"c{L}"].mean()) / np.sqrt(D_delay[r]["se"]["c0"].mean())
                                           for r in REPEATS])) for L in LS}
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B70.md section 3, Addendum 2 C8-C9",
           "n_boot": N_BOOT, "seed": SEED, "checks": chk, "statements": st,
           "delay": {"cell": Ad, "per_repeat": Adper, "rmse_network_ratio_to_L0": rmse_ratio,
                     "median_age_ms": {f"L{L}": float(np.median(np.concatenate(
                         [np.load(p)[f"d_L{L}_age_ms"] for p in files]))) for L in LS}},
           "thinning": {"cell": At, "per_repeat": Atper}}
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "b70_replay.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    s3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                      # noqa: E731
    L_ = ["# B.70 phase 2 -- delay and thinning replays on the D1 rows (r2-r4)", "",
          f"Check at L = 0 (network, hold, causal GP within 0.5 eV of the D1 npz): {chk['within_0.5_eV']} -> "
          f"{'passes' if chk['passes'] else 'FAILS'}. Fixed rows per re-partition: {chk['fixed_rows_per_repeat']:,}; "
          f"thinning rows (B.30 population, all phases): {chk['thinning_rows_per_repeat']}.", "",
          "| statement | value | verdict |", "|---|---|---|"]
    for k, s in st.items():
        if "value" in s:
            L_.append(f"| {k}: {s['text']} | {s3(s['value'])} | {'holds' if s['holds'] else 'does not hold'} |")
    L_ += ["", "| L | network over GP | over twin | over hold | twin over GP | GP over hold | network RMSE / L = 0 |",
           "|---|---|---|---|---|---|---|"]
    for L in LS:
        L_.append(f"| {L} | {s3(Ad[f'skill_mse|c{L}|gp{L}'])} | {s3(Ad[f'skill_mse|c{L}|t{L}'])} | {s3(Ad[f'skill_mse|c{L}|hold{L}'])} | "
                  f"{s3(Ad[f'skill_mse|t{L}|gp{L}'])} | {s3(Ad[f'skill_mse|gp{L}|hold{L}'])} | {rmse_ratio[f'L{L}']:.3f} |")
    L_ += ["", "Thinning k = 5: " + "; ".join(f"{a} over {b} {s3(At[f'skill_mse|{a}|{b}'])}" for a, b in tpairs) + "."]
    (PAPER / "b70_replay.md").write_text("\n".join(L_) + "\n", encoding="utf-8", newline="\n")
    log("written " + str(PAPER / "b70_replay.json"))


if __name__ == "__main__":
    if sys.argv[1] == "rows":
        part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "0/1"
        stage_rows(part)
    else:
        stage_readout()
    os._exit(0)
