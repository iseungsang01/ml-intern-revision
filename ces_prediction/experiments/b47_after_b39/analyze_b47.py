"""B.47 readouts (PREREGISTRATION_B47.md).  CPU only; frozen networks, nothing is trained.

    py ces_prediction/experiments/b47_after_b39/analyze_b47.py              # readout -> docs/paper/b47_after_b39.json / .md
    py ces_prediction/experiments/b47_after_b39/analyze_b47.py --check-d    # arm D path with one initialisation vs B.39 / B.45 (prints)
    py ces_prediction/experiments/b47_after_b39/analyze_b47.py --check-l [--split 42]   # arm L machinery vs B.42 (prints)

Arm D  three initialisations per network (B.39's runs + arm D's); D1', D2', D3' and the reported pairs with the
       two-level interval of B.39 (analyze_b39.two_level); R1 to R3 of B.45 with the class assignment of run_b45.
Arm L  the row machinery of B.42 (run_b42.file_inputs / run_net / input_check / ces_channels) on the
       delay-trained networks, their L = 0 controls, the undelayed B.39 estimator and the networks of B.42, at
       L = 0, 1, 2, 4; L1 to L3 at L = 2.  Before the readout the networks of B.42 must reproduce the cached rows
       of B.42 (data/.b42/rows_s{seed}.npz) and its skills.
Arm I  S1i, S2i as S1, S2 of analyze_b39 on the inclusive runs.
An arm is read only when all its runs are complete (arm I: by split).  Row cache of arm L: data/.b47_readout/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b39_primary"))
sys.path.insert(0, str(HERE.parent / "b42_delay"))
import run_b47 as P  # noqa: E402
import analyze_b39 as A39  # noqa: E402
import run_b42 as K  # noqa: E402

B39 = P.B39
DATA = P.DATA
REPO = HERE.parents[2]
OUT_JSON = REPO / "docs" / "paper" / "b47_after_b39.json"
OUT_MD = REPO / "docs" / "paper" / "b47_after_b39.md"
B39_JSON = REPO / "docs" / "paper" / "b39_primary.json"
B42_JSON = REPO / "docs" / "paper" / "b42_delay.json"
B45_JSON = REPO / "docs" / "paper" / "b45_ablation_rows.json"
CACHE = DATA / ".b47_readout"
REG = "ces_prediction/experiments/PREREGISTRATION_B47.md"
SEEDS = P.SEEDS
ABL = P.ABLATIONS
SEED_D, SEED_I = 20261118, 20261119          # two_level draws of arms D and I (B.39 used 20261117)
LS = K.LS
CAP = int(P.EPOCH_CAP)
NLAB = {"v2": "full", "v2h": "twin", "v2e": "ECEI-only", "v2b": "BES-only", "v2n": "no-Mirnov"}
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def has(d):
    return (d / "comparison_errors_test.npz").exists()


def epochs(d):
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    return [int(m["epochs_run"]), int(m["best_epoch"]), bool(m["epochs_run"] >= CAP)]


# =============================================================================== arm D
def d_dirs(s, one=False):
    ini = (s,) if one else B39.inits(s)
    nets = {"v2": [B39.split_run_dir("v2", s, i) for i in ini], "v2h": [B39.twin_dir(s, i) for i in ini]}
    for m in ABL:
        nets[m] = [B39.split_run_dir(m, s, s) if i == s else P.d_dir(m, s, i) for i in ini]
    return ini, nets


def d_status():
    own = [P.d_dir(m, s, i) for s in SEEDS for i in B39.inits(s)[1:] for m in ABL]
    need = [d for s in SEEDS for v in d_dirs(s)[1].values() for d in v if d not in own]
    return {"arm_D_runs_done": sum(has(d) for d in own), "arm_D_runs_total": len(own),
            "b39_runs_missing": [d.name for d in need if not has(d)]}


def arm_d(rng, one=False):
    st = d_status()
    if not one and (st["arm_D_runs_done"] < st["arm_D_runs_total"] or st["b39_runs_missing"]):
        return {"complete": False, **st}
    out = {"complete": True, "one_initialisation_check": one, **st, "splits": {}}
    for s in SEEDS:
        ini, nets = d_dirs(s, one)
        L = {k: [A39.load(d) for d in v] for k, v in nets.items()}
        ref = L["v2"][0]
        for k, v in L.items():
            for j, d in enumerate(v):
                A39.same_rows(ref, d, f"arm D s{s} {k} i{ini[j]}")
        shot = ref["shot"]
        M = {k: np.array([A39.sums(shot, d["model"]) for d in v]) for k, v in L.items()}
        gp = A39.sums(shot, ref["gp_causal"])[None, :]
        tl = lambda a, b: A39.two_level(M[a], M[b] if isinstance(b, str) else b, rng)   # noqa: E731
        o = {"rows": int(len(shot)), "discharges": int(M["v2"].shape[1]), "initialisations": list(ini),
             "D1p_ecei_over_twin": tl("v2e", "v2h"), "D2p_bes_over_twin": tl("v2b", "v2h"),
             "D3p_full_over_bes": tl("v2", "v2b"), "nomirnov_over_twin": tl("v2n", "v2h"),
             "full_over_twin": tl("v2", "v2h"), "full_over_nomirnov": tl("v2", "v2n"), "full_over_ecei": tl("v2", "v2e"),
             "over_causal_gp": {k: tl(k, gp) for k in M}}
        tw_mean = M["v2h"].sum(1).mean()
        o["per_initialisation"] = {k: {str(ini[j]): {"over_causal_gp": float(1 - M[k][j].sum() / gp.sum()),
                                                     "over_twin_mean": float(1 - M[k][j].sum() / tw_mean),
                                                     "over_twin_same_init": float(1 - M[k][j].sum() / M["v2h"][j].sum())}
                                       for j in range(len(ini))} for k in M}
        o["spread"] = {k: {q: float(max(x[q] for x in v.values()) - min(x[q] for x in v.values()))
                           for q in ("over_twin_mean", "over_causal_gp")} for k, v in o["per_initialisation"].items()}
        o["median_abs_error_eV"] = {k: float(np.mean([A39.med(d["model"]) for d in v])) for k, v in L.items()}
        o["median_abs_error_eV"].update({"hold": A39.med(ref["persistence"]), "causal GP": A39.med(ref["gp_causal"])})
        o["epochs_run_best_capped"] = {f"{k} i{ini[j]}": epochs(d) for k, v in nets.items() for j, d in enumerate(v)}
        for k in ("D1p_ecei_over_twin", "D2p_bes_over_twin", "D3p_full_over_bes"):
            o[k.split("_")[0] + "_holds"] = bool(o[k][1] > 0)
        out["splits"][str(s)] = o
    for k in ("D1p", "D2p", "D3p"):
        n = sum(out["splits"][str(s)][k + "_holds"] for s in SEEDS)
        out[k] = {"splits": int(n), "holds": bool(n >= 3)}
    return out


def check_d_against_b39(D):
    """Points of the one-initialisation path against docs/paper/b39_primary.json (deterministic)."""
    b39 = json.loads(B39_JSON.read_text(encoding="utf-8"))["A"]
    keys = {"D1p_ecei_over_twin": "D1_ecei_over_twin", "D2p_bes_over_twin": "D2_bes_over_twin",
            "nomirnov_over_twin": "nomirnov_over_twin", "D3p_full_over_bes": "full_over_bes",
            "full_over_twin": "full_over_twin_same_initialisation"}
    res, ok = {}, True
    for s in SEEDS:
        for mine, theirs in keys.items():
            a, b = D["splits"][str(s)][mine], b39[str(s)][theirs]
            eq = round(a[0], 3) == round(b[0], 3)
            ok &= eq
            res[f"{s} {mine}"] = {"mine": a, "b39": b, "point_equal_3dp": eq,
                                  "max_interval_dev": float(max(abs(a[1] - b[1]), abs(a[2] - b[2])))}
    return ok, res


# ------------------------------------------------------------------------------- R1 to R3 of B.45
def r_rows(rng, one=False):
    sys.path.insert(0, str(HERE.parent / "b45_ablation_rows"))
    import run_b45 as B45
    C = B45.B41.Ctx()
    valid = [nm for nm in C.names if np.isfinite(C.grid[nm][:, 1:3]).any()]
    Cl = B45.Classes(C)
    out = {}
    for s in SEEDS:
        ini, nets = d_dirs(s, one)
        runs = {k: [B45.load_run(d) for d in nets[k]] for k in ("v2", "v2h", "v2b", "v2e")}
        ref = runs["v2"][0]
        for k, v in runs.items():
            for d in v:
                A39.same_rows(ref, d, f"R rows s{s} {k}")
        shot, row = ref["shot"], ref["row"]
        cls = np.zeros(len(shot), np.int8)
        for i in np.unique(shot):                       # run_b45.run_split, the class of every scored row
            sel = np.flatnonzero(shot == i)
            name = valid[int(i)]
            g = B45.compact_index(C.grid[name])[row[sel]]
            cls[sel] = Cl.of_file(name)[0][g]
        if (cls == 0).any():
            raise SystemExit(f"FATAL: split {s}: a scored row maps to the first or last row of its file")
        masks = {"R1": np.isin(cls, (3, 4, 5, 6)), "R2": cls == 6, "R3": cls == 1}
        net = {"R1": "v2b", "R2": "v2b", "R3": "v2e"}
        o = {"rows": {k: int(m.sum()) for k, m in masks.items()}}
        for k, m in masks.items():
            A = np.array([A39.sums(shot, d["model"] * m) for d in runs[net[k]]])
            Bm = np.array([A39.sums(shot, d["model"] * m) for d in runs["v2h"]])
            o[k] = A39.two_level(A, Bm, rng)
            o[k + "_holds"] = bool(o[k][1] > 0)
        out[str(s)] = o
    res = {"splits": out}
    for k in ("R1", "R2", "R3"):
        n = sum(out[str(s)][k + "_holds"] for s in SEEDS)
        res[k] = {"splits": int(n), "holds": bool(n >= 3)}
    return res


def check_r_against_b45(R):
    b45 = json.loads(B45_JSON.read_text(encoding="utf-8"))["splits"]
    res, ok = {}, True
    for s in SEEDS:
        for k in ("R1", "R2", "R3"):
            a, b = R["splits"][str(s)][k], b45[str(s)][k]
            eq = round(a[0], 3) == round(b[0], 3)
            ok &= eq
            res[f"{s} {k}"] = {"mine": a, "b45": b, "point_equal_3dp": eq,
                               "max_interval_dev": float(max(abs(a[1] - b[1]), abs(a[2] - b[2])))}
    return ok, res


# =============================================================================== arm L
def l_nets(s):
    """name -> (run dir, model, delay the run was trained with (None = trained by train_seq))."""
    return {"causal_L2": (P.l_dir("v2", s, 2), "v2", 2), "twin_L2": (P.l_dir("v2h", s, 2), "v2h", 2),
            "causal_L0": (P.l_dir("v2", s, 0), "v2", 0), "twin_L0": (P.l_dir("v2h", s, 0), "v2h", 0),
            "causal_b39": (B39.split_run_dir("v2", s, s), "v2", None),
            "twin_b29": (DATA / f".b29a1_v2h_cut_s{s}", "v2h", None),
            "causal_b29": (DATA / f".b29_norm0_cut_s{s}", "v2", None)}


LAB = {"causal_L2": "estimator trained at L = 2", "twin_L2": "twin trained at L = 2", "causal_L0": "estimator, L = 0 control",
       "twin_L0": "twin, L = 0 control", "causal_b39": "estimator trained without delay (B.39)",
       "twin_b29": "twin trained without delay (B.29)", "causal_b29": "B.42's estimator (cap 30)", "hold": "hold", "gp": "causal GP"}
L_PAIRS = [("causal_L2", "gp"), ("causal_L2", "twin_L2"), ("causal_L2", "causal_b39"), ("causal_L2", "hold"),
           ("twin_L2", "gp"), ("twin_L2", "twin_b29"), ("causal_b39", "gp"), ("causal_b39", "twin_b29"),
           ("causal_L0", "gp"), ("causal_L0", "causal_b39"), ("twin_L0", "twin_b29"), ("causal_L0", "twin_L0"),
           ("twin_b29", "gp"), ("gp", "hold"), ("causal_b29", "gp"), ("causal_b29", "twin_b29")]


def l_status():
    runs = [P.l_dir(m, s, L) for s in SEEDS for m in ("v2", "v2h") for L in P.DELAYS]
    return {"runs_done": sum(has(d) and (d / "b47_delay.json").exists() for d in runs), "runs_total": len(runs)}


def load_net(C, run, model, delay):
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    per_shot = bool(m["per_shot_input_norm"])
    if m["seq_model"] != model or per_shot != (model == "v2h") or m.get("per_shot_norm_expanding") or m.get("zero_mc") \
            or m.get("extra_vt_channels") or m.get("extra_ti_channels") or m.get("train_ctx"):
        raise SystemExit(f"FATAL: {run.name} is not the registered {model} network")
    dj = run / "b47_delay.json"
    if delay is None and dj.exists():
        raise SystemExit(f"FATAL: {run.name} carries a b47_delay.json but should be undelayed")
    if delay is not None:
        rec = json.loads(dj.read_text(encoding="utf-8"))
        if rec["delay_frames"] != delay or rec["staleness"] != "true" or rec["wrapper_calls"] < 1:
            raise SystemExit(f"FATAL: {run.name} was trained with delay {rec['delay_frames']}, expected {delay}")
    stats = C.load_stats(m)
    net = C.MODELS[model]()
    net.load_state_dict(C.torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
    net.eval()
    return (net, stats), per_shot


def rows_split(C, s, names, limit=None):
    """B.42's rows_split for an arbitrary set of networks (true staleness only)."""
    spec = l_nets(s)
    nets = {nm: load_net(C, *spec[nm]) for nm in names}
    man = C.manifest(s)
    test = sorted(C.names.index(nm) for nm in man["test_files"] if nm in C.grid)[:limit]
    acc = {L: {c: [] for c in ("file", "row", "y", "age_ms", "hold", "gp", *names)} for L in LS}
    in_ok = {nm: 0 for nm in names}
    t0 = time.time()
    for fi in test:
        arr = C.grid[C.names[fi]]
        n = len(arr)
        pred = {}
        for nm, (net, per_shot) in nets.items():
            blks, spans, tm, tgt, tz = K.file_inputs(C, arr, net, per_shot)
            if not K.input_check(C, blks, spans, tm, tgt, tz):
                raise SystemExit(f"FATAL: L = 0 rebuilt inputs differ from build_blocks ({nm}, {C.names[fi]})")
            in_ok[nm] += 1
            for L in LS:
                pred[(nm, L)] = K.run_net(C, net, blks, spans, tm, tgt, tz, L, "true", n)
        y = tgt[:, 0]
        for L in LS:
            A = acc[L]
            for a, b in spans:
                obs = np.flatnonzero(~np.isnan(y[a:b])) + a
                for r in obs:
                    av = obs[obs <= r - 1 - L]
                    if av.size == 0:
                        continue
                    last = av[-K.R.GP_SIDE:]
                    A["file"].append(fi); A["row"].append(r); A["y"].append(y[r])
                    A["age_ms"].append((tm[r] - tm[av[-1]]) * 1000.0)
                    A["hold"].append(y[av[-1]])
                    A["gp"].append(C.B.predict_gp_causal(tm[last], y[last], tm[r]))
                    for nm in names:
                        A[nm].append(pred[(nm, L)][r])
        log(f"s{s} file {fi} done ({time.time() - t0:.0f}s)")
    out = {"input_check_files": in_ok}
    for L in LS:
        for c, v in acc[L].items():
            out[f"L{L}__{c}"] = np.asarray(v, dtype=np.int32 if c in ("file", "row") else np.float64)
    return out


def skills(rows, ests, pairs, W, uf):
    """B.42's analyze_split for arbitrary estimators and pairs (same resampling rule)."""
    pos = {f: k for k, f in enumerate(uf)}
    nf = len(uf)
    res = {}
    for L in LS:
        f = np.array([pos[int(x)] for x in rows[f"L{L}__file"]])
        y = rows[f"L{L}__y"]
        e = {k: rows[f"L{L}__{k}"] - y for k in ests}
        fin = np.all([np.isfinite(v) for v in e.values()], axis=0)
        f, e = f[fin], {k: v[fin] for k, v in e.items()}
        sse = {k: np.bincount(f, v ** 2, nf) for k, v in e.items()}
        cnt = np.bincount(f, None, nf).astype(float)
        bs = {k: W @ v for k, v in sse.items()}
        bn = W @ cnt
        Wr = W[:, f]
        o = {"rows": int(fin.sum()), "rows_nonfinite_dropped": int((~fin).sum()),
             "median_age_ms": float(np.median(rows[f"L{L}__age_ms"][fin])),
             "rmse": {k: [float(np.sqrt(sse[k].sum() / cnt.sum())), *K.ci(np.sqrt(bs[k] / bn))] for k in ests},
             "median_abs": {k: [float(np.median(np.abs(e[k]))), *K.ci(K.wq(np.abs(e[k]), Wr))] for k in ests},
             "skill": {f"{a}_vs_{b}": [float(1 - sse[a].sum() / sse[b].sum()), *K.ci(1 - bs[a] / bs[b])]
                       for a, b in pairs if a in ests and b in ests}}
        res[str(L)] = o
    return res


def boot_weights():
    """B.42's resamples: one generator (run_b42.BOOT_SEED), one draw per split in the order 42, 1, 7, 123."""
    rng = np.random.default_rng(K.BOOT_SEED)
    W = {}
    for s in SEEDS:
        z = np.load(K.CACHE / f"rows_s{s}.npz")
        uf = np.unique(z["L0__file"])
        W[s] = (rng.multinomial(len(uf), np.full(len(uf), 1.0 / len(uf)), size=K.N_BOOT).astype(float), uf)
    return W


def reproduce_b42(rows, s, W):
    """The networks of B.42 through this machinery against the cached rows and the skills of B.42."""
    z = np.load(K.CACHE / f"rows_s{s}.npz")
    dev = {}
    for L in LS:
        for c in ("file", "row"):
            if not np.array_equal(rows[f"L{L}__{c}"], z[f"L{L}__{c}"]):
                raise SystemExit(f"FATAL: split {s} L = {L}: scored rows differ from B.42 ({c})")
        for mine, theirs in (("y", "y"), ("hold", "hold"), ("gp", "gp"), ("causal_b29", "causal"), ("twin_b29", "twin")):
            dev[f"L{L} {mine}"] = float(np.nanmax(np.abs(rows[f"L{L}__{mine}"] - z[f"L{L}__{theirs}"])))
    b42 = json.loads(B42_JSON.read_text(encoding="utf-8"))["splits"][str(s)]
    mine = skills(rows, ("hold", "gp", "causal_b29", "twin_b29"),
                  [("causal_b29", "gp"), ("causal_b29", "twin_b29"), ("causal_b29", "hold")], *W[s])
    comp, ok = {}, max(dev.values()) <= 1e-6
    for L in map(str, LS):
        for a, b in (("causal_b29_vs_gp", "causal_vs_gp"), ("causal_b29_vs_twin_b29", "causal_vs_twin"),
                     ("causal_b29_vs_hold", "causal_vs_hold")):
            x, y = mine[L]["skill"][a], b42[L]["skill"][b]
            eq = all(round(p, 3) == round(q, 3) for p, q in zip(x, y))
            ok &= eq
            comp[f"L{L} {b}"] = {"mine": x, "b42": y, "equal_3dp_point_and_interval": eq}
    return {"max_abs_dev_eV": dev, "skills": comp, "passes": bool(ok)}


def arm_l(C, fresh=False):
    st = l_status()
    if st["runs_done"] < st["runs_total"]:
        return {"complete": False, **st}
    names = list(l_nets(42))
    W = boot_weights()
    CACHE.mkdir(exist_ok=True)
    out = {"complete": True, **st, "splits": {}, "checks": {}}
    for s in SEEDS:
        cp = CACHE / f"rows_L_s{s}.npz"
        if cp.exists() and not fresh:
            z = np.load(cp)
            rows = {k: z[k] for k in z.files}
            in_ok = json.loads((CACHE / f"input_check_s{s}.json").read_text(encoding="utf-8"))
        else:
            rows = rows_split(C, s, names)
            in_ok = rows.pop("input_check_files")
            np.savez_compressed(cp, **rows)
            (CACHE / f"input_check_s{s}.json").write_text(json.dumps(in_ok), encoding="utf-8")
        rep = reproduce_b42(rows, s, W)
        out["checks"][str(s)] = {"input_check_files": in_ok, "reproduce_b42": rep}
        log(f"split {s}: reproduce B.42 passes {rep['passes']} (max dev {max(rep['max_abs_dev_eV'].values()):.2e} eV)")
        if not rep["passes"]:
            raise SystemExit(f"FATAL: split {s}: the machinery does not reproduce B.42; arm L is not read")
        out["splits"][str(s)] = skills(rows, ("hold", "gp", *names), L_PAIRS, *W[s])
    sp = out["splits"]
    for k, pair in (("L1", "causal_L2_vs_gp"), ("L2", "causal_L2_vs_twin_L2"), ("L3", "causal_L2_vs_causal_b39")):
        n = sum(sp[str(s)]["2"]["skill"][pair][1] > 0 for s in SEEDS)
        out[k] = {"pair": pair, "splits": int(n), "holds": bool(n >= 3)}
    # L = 0 control on the harness of eval_seq (undelayed scoring): the control against the B.39 / B.29 run
    ctrl = {}
    rng = np.random.default_rng(SEED_D + 7)
    for s in SEEDS:
        o = {}
        for nm, mine, ref in (("estimator", P.l_dir("v2", s, 0), B39.split_run_dir("v2", s, s)),
                              ("twin", P.l_dir("v2h", s, 0), DATA / f".b29a1_v2h_cut_s{s}"),
                              ("estimator trained at L = 2 (undelayed scoring, sanity)", P.l_dir("v2", s, 2), B39.split_run_dir("v2", s, s)),
                              ("twin trained at L = 2 (undelayed scoring, sanity)", P.l_dir("v2h", s, 2), DATA / f".b29a1_v2h_cut_s{s}")):
            a, b = A39.load(mine), A39.load(ref)
            A39.same_rows(a, b, f"L control s{s} {nm}")
            ga, gb = float(1 - a["model"].sum() / a["gp_causal"].sum()), float(1 - b["model"].sum() / b["gp_causal"].sum())
            o[nm] = {"run": mine.name, "reference": ref.name, "over_causal_gp": ga, "reference_over_causal_gp": gb,
                     "difference": ga - gb, "within_0.02": bool(abs(ga - gb) <= 0.02),
                     "over_reference": A39.two_level(A39.sums(a["shot"], a["model"])[None, :],
                                                     A39.sums(b["shot"], b["model"])[None, :], rng),
                     "rmse_eV": [float(np.sqrt(a["model"].mean())), float(np.sqrt(b["model"].mean()))],
                     "epochs_run_best_capped": epochs(mine)}
        ctrl[str(s)] = o
    out["L0_control_eval_seq"] = ctrl
    return out


# =============================================================================== arm I
def arm_i(rng):
    out = {"splits": {}, "runs_done": 0, "runs_total": 24}
    for s in SEEDS:
        ini = B39.inits(s)
        dirs = {"v2": [P.i_dir("v2", s, i) for i in ini], "v2h": [P.i_dir("v2h", s, i) for i in ini]}
        n_done = sum(has(d) for v in dirs.values() for d in v)
        out["runs_done"] += n_done
        if n_done < 6:
            continue
        L = {k: [A39.load(d) for d in v] for k, v in dirs.items()}
        ref = L["v2"][0]
        for k, v in L.items():
            for j, d in enumerate(v):
                A39.same_rows(ref, d, f"arm I s{s} {k} i{ini[j]}")
        shot = ref["shot"]
        S = lambda v: A39.sums(shot, v)                                             # noqa: E731
        C = np.array([S(d["model"]) for d in L["v2"]])
        Tw = np.array([S(d["model"]) for d in L["v2h"]])
        R = {r: S(ref[r])[None, :] for r in A39.REFS}
        o = {"rows": int(len(shot)), "discharges": int(C.shape[1]), "initialisations": list(ini),
             "S1i_causal_over_gp": A39.two_level(C, R["gp_causal"], rng), "S2i_causal_over_twin": A39.two_level(C, Tw, rng),
             "twin_over_gp": A39.two_level(Tw, R["gp_causal"], rng),
             "causal_over": {r: A39.two_level(C, R[r], rng) for r in A39.REFS},
             "per_initialisation": {str(ini[j]): {"causal_over_gp": float(1 - L["v2"][j]["model"].sum() / ref["gp_causal"].sum()),
                                                  "twin_over_gp": float(1 - L["v2h"][j]["model"].sum() / ref["gp_causal"].sum()),
                                                  "causal_over_twin": float(1 - L["v2"][j]["model"].sum() / L["v2h"][j]["model"].sum())}
                                    for j in range(3)},
             "median_abs_error_eV": {"hold": A39.med(ref["persistence"]), "causal GP": A39.med(ref["gp_causal"]),
                                     "PCHIP": A39.med(ref["pchip"]), "offline GP": A39.med(ref["gp"]),
                                     "causal variant": float(np.mean([A39.med(d["model"]) for d in L["v2"]])),
                                     "twin": float(np.mean([A39.med(d["model"]) for d in L["v2h"]]))},
             "rmse_eV": {"hold": float(np.sqrt(ref["persistence"].mean())), "causal GP": float(np.sqrt(ref["gp_causal"].mean())),
                         "causal variant": float(np.mean([np.sqrt(d["model"].mean()) for d in L["v2"]])),
                         "twin": float(np.mean([np.sqrt(d["model"].mean()) for d in L["v2h"]]))},
             "epochs_run_best_capped": {f"{k} i{ini[j]}": epochs(d) for k, v in dirs.items() for j, d in enumerate(v)}}
        v = [x["causal_over_gp"] for x in o["per_initialisation"].values()]
        o["spread_causal_over_gp"] = float(max(v) - min(v))
        o["S1i_holds"] = bool(o["S1i_causal_over_gp"][1] > 0)
        o["S2i_holds"] = bool(o["S2i_causal_over_twin"][1] > 0)
        out["splits"][str(s)] = o
    done = list(out["splits"])
    out["complete"] = len(done) == 4
    for k in ("S1i", "S2i"):
        n = sum(out["splits"][s][k + "_holds"] for s in done)
        out[k] = {"splits_read": len(done), "splits": int(n), "holds": bool(n >= 3) if len(done) == 4 else None}
    return out


# =============================================================================== expectations, render
def expectations(res):
    E = []
    D, L, I = res["D"], res["L"], res["I"]
    pt = lambda o, k: [o["splits"][str(s)][k][0] for s in SEEDS]                  # noqa: E731
    if D.get("complete"):
        v = pt(D, "D1p_ecei_over_twin")
        E.append(("D1': +0.00 to +0.05 on three splits and up to +0.09 on the fourth; holds on one or two splits",
                  v, sum(0 <= x <= 0.05 for x in v) >= 3 and max(v) <= 0.09 and D["D1p"]["splits"] in (1, 2), D["D1p"]["splits"]))
        v = pt(D, "D2p_bes_over_twin")
        E.append(("D2': +0.03 to +0.10, holds on four", v, all(0.03 <= x <= 0.10 for x in v) and D["D2p"]["splits"] == 4,
                  D["D2p"]["splits"]))
        v = pt(D, "D3p_full_over_bes")
        E.append(("D3': within 0.02 of zero, does not hold", v, all(abs(x) <= 0.02 for x in v) and not D["D3p"]["holds"],
                  D["D3p"]["splits"]))
        v = [D["splits"][str(s)]["spread"][m]["over_twin_mean"] for s in SEEDS for m in ABL]
        E.append(("spread between initialisations of an ablation network (largest minus smallest skill over the twin): "
                  "0.01 to 0.04 (12 values: split x network)", v, all(0.01 <= x <= 0.04 for x in v), sum(0.01 <= x <= 0.04 for x in v)))
    if L.get("complete"):
        for k, lo, hi, text in (("L1", 0.07, 0.16, "L1: +0.07 to +0.16, holds"), ("L2", 0.04, 0.13, "L2: +0.04 to +0.13, holds")):
            v = [L["splits"][str(s)]["2"]["skill"][L[k]["pair"]][0] for s in SEEDS]
            E.append((text, v, all(lo <= x <= hi for x in v) and L[k]["holds"], L[k]["splits"]))
        v = [L["splits"][str(s)]["2"]["skill"][L["L3"]["pair"]][0] for s in SEEDS]
        E.append(("L3: -0.01 to +0.03, holds on at most two splits (does not hold)", v,
                  all(-0.01 <= x <= 0.03 for x in v) and L["L3"]["splits"] <= 2, L["L3"]["splits"]))
        v = [L["L0_control_eval_seq"][str(s)]["estimator"]["difference"] for s in SEEDS]
        E.append(("L = 0 control: within 0.02 in skill over the causal GP of the B.39 run", v, all(abs(x) <= 0.02 for x in v),
                  sum(abs(x) <= 0.02 for x in v)))
    if I.get("complete"):
        v = pt(I, "S1i_causal_over_gp")
        E.append(("S1i: +0.09 to +0.17, holds on four", v, all(0.09 <= x <= 0.17 for x in v) and I["S1i"]["splits"] == 4, I["S1i"]["splits"]))
        v = pt(I, "S2i_causal_over_twin")
        E.append(("S2i: +0.01 to +0.04, holds on two or three splits", v,
                  all(0.01 <= x <= 0.04 for x in v) and I["S2i"]["splits"] in (2, 3), I["S2i"]["splits"]))
    return [{"expectation": a, "points": [float(x) for x in b], "met": bool(c), "count": int(d)} for a, b, c, d in E]


CHOICES = [
    "Arm D initialisations: the network at initialisation = split seed is B.39's run (`.b39_{v2e,v2b,v2n}_cut_s{seed}_i{seed}`), "
    "the two further ones are arm D's; the full estimator and the twin are B.39's three initialisations (the twin at "
    "initialisation = split seed is `.b29a1_v2h_cut_s{seed}`, as in B.39). Every interval is analyze_b39.two_level: in each of "
    "2,000 draws one initialisation of each of the two networks is drawn uniformly and independently and the test discharges are "
    "resampled (generator 20261118 for arm D and R1 to R3, 20261119 for arm I).",
    "Spread between initialisations: largest minus smallest skill of the network's initialisations over the twin's squared-error "
    "sum averaged over its three initialisations (so the spread is that of the network, not of the twin); the same over the causal "
    "GP is reported beside it.",
    "R1 to R3 with three initialisations: the class of every scored row is that of run_b45 (Classes, compact_index; classes 3 to 6 "
    "for R1, 6 for R2, 1 for R3); the two-level interval is applied to the per-discharge sums inside the class.",
    "Arm L: the machinery of B.42 is reused unchanged (run_b42.file_inputs, input_check, run_net with true staleness, the scoring "
    "rule 'T_i measured at row t with at least one available sample in the block', hold and the causal GP on the 16 last "
    "available samples). Each network is read with its own standardization statistics and its own mode (train-file statistics "
    "for the estimator, per-discharge for the twin, as trained). Resamples: B.42's (generator run_b42.BOOT_SEED, one draw per "
    "split in the order 42, 1, 7, 123), so the intervals of arm L are paired with those of B.42.",
    "Arm L before the readout: B.42's own networks (`.b29_norm0_cut`, `.b29a1_v2h_cut`) are run through the same machinery on "
    "every split; their rows must equal B.42's cached rows (data/.b42/rows_s{seed}.npz) and their skills over the causal GP, "
    "the twin and hold at every L must equal B.42's to three decimals, point and interval.",
    "L = 0 control: read on the harness of eval_seq.py (undelayed, the test rows of B.39) as the skill over the causal GP of "
    "the control minus that of the B.39 run (estimator) or of `.b29a1_v2h_cut` (twin), with the paired skill of the control over "
    "that run; it is also scored by the B.42 machinery at L = 0. The networks trained at L = 2 are scored by eval_seq.py too "
    "(undelayed inputs); that score is a sanity record only.",
    "Arm I: initialisations as in B.39 (run_b39.inits(seed)); a split is read when its six runs are complete. Reported against "
    "hold, PCHIP and the offline GP with the same two-level interval.",
]


def render(res):
    D, L, I = res["D"], res["L"], res["I"]
    o = ["# B.47 — after B.39: the ablation with three initialisations, an estimator trained with a delayed CES history, "
         "and the inclusive population", "",
         f"Registered: {REG}. GPU runs; readouts on a CPU. T_i, epoch cap 80 for every network, fast channels standardized with "
         "the statistics of the training discharges (the twin as trained, per discharge; it reads no fast channel). Skill of A over "
         "R = 1 - sum of squared errors of A over that of R on the same test rows. Intervals 95 %, 2,000 draws; where three "
         "initialisations exist a draw takes one initialisation of each network uniformly and resamples the test discharges. A "
         "statement holds if its interval is above zero on at least three of four splits. Arm L: a CES value measured at row u is "
         "available at row t if u <= t - 1 - L (L in frames of 10 ms); the estimator and its twin are trained with the six "
         "CES-derived inputs built under L = 2 (true staleness).", "", "## Checks before the readout", ""]
    for k, c in res["checks"].items():
        o.append(f"- {k}: {c}")
    if L.get("complete"):
        for s in SEEDS:
            c = L["checks"][str(s)]
            r = c["reproduce_b42"]
            o.append(f"- arm L, split {s}: L = 0 rebuilt inputs equal build_blocks on {c['input_check_files']} test files; B.42's "
                     f"networks through this machinery: largest deviation from B.42's rows {max(r['max_abs_dev_eV'].values()):.1e} eV, "
                     f"skills at every L equal to three decimals: {all(x['equal_3dp_point_and_interval'] for x in r['skills'].values())} "
                     f"-> {'passes' if r['passes'] else 'FAILS'}.")
    o += ["", "## Registered statements", ""]
    if D.get("complete"):
        for k, t in (("D1p", "D1' (ECEI-only over the twin)"), ("D2p", "D2' (BES-only over the twin)"),
                     ("D3p", "D3' (full estimator over BES-only)")):
            o.append(f"- **{t}**: {D[k]['splits']} of 4 -> **{'holds' if D[k]['holds'] else 'does not hold'}**.")
    else:
        o.append(f"- Arm D not read: {D.get('arm_D_runs_done')} of {D.get('arm_D_runs_total')} runs complete; B.39 runs missing: "
                 f"{D.get('b39_runs_missing')}.")
    if L.get("complete"):
        for k, t in (("L1", "L1 (trained at L = 2, over the causal GP, at L = 2)"), ("L2", "L2 (over the twin trained at L = 2)"),
                     ("L3", "L3 (over the estimator trained without delay, read at L = 2)")):
            o.append(f"- **{t}**: {L[k]['splits']} of 4 -> **{'holds' if L[k]['holds'] else 'does not hold'}**.")
    else:
        o.append(f"- Arm L not read: {L.get('runs_done')} of {L.get('runs_total')} runs complete.")
    if I["splits"]:
        for k, t in (("S1i", "S1i (causal variant over the causal GP, inclusive)"), ("S2i", "S2i (over the twin, inclusive)")):
            h = I[k]["holds"]
            o.append(f"- **{t}**: {I[k]['splits']} of {I[k]['splits_read']} splits read -> **"
                     f"{'holds' if h else 'does not hold' if h is not None else 'not decided (splits missing)'}**.")
    if not I.get("complete"):
        o.append(f"- Arm I: {I['runs_done']} of 24 runs complete; splits read: {list(I['splits']) or 'none'}.")
    if D.get("complete"):
        o += ["", "## Arm D: the ablation with three initialisations", "",
              "| split | D1': ECEI-only over twin | D2': BES-only over twin | D3': full over BES-only | no-Mirnov over twin | "
              "full over twin | full over no-Mirnov | full over ECEI-only |", "|---|---|---|---|---|---|---|---|"]
        for s in SEEDS:
            x = D["splits"][str(s)]
            o.append(f"| {s} | {s3(x['D1p_ecei_over_twin'])} | {s3(x['D2p_bes_over_twin'])} | {s3(x['D3p_full_over_bes'])} | "
                     f"{s3(x['nomirnov_over_twin'])} | {s3(x['full_over_twin'])} | {s3(x['full_over_nomirnov'])} | {s3(x['full_over_ecei'])} |")
        o += ["", "Skill of each initialisation over the twin (its three initialisations averaged) and spread (largest minus smallest; "
              "in brackets the spread of the skill over the causal GP):", "",
              "| split | network | initialisations | spread |", "|---|---|---|---|"]
        for s in SEEDS:
            x = D["splits"][str(s)]
            for k, v in x["per_initialisation"].items():
                o.append(f"| {s} | {NLAB[k]} | " + ", ".join(f"i{i}: {y['over_twin_mean']:+.3f}" for i, y in v.items()) +
                         f" | {x['spread'][k]['over_twin_mean']:.3f} ({x['spread'][k]['over_causal_gp']:.3f}) |")
        o += ["", "| split | over the causal GP: full / ECEI-only / BES-only / no-Mirnov / twin | median absolute error (eV): hold / causal GP / "
              "full / twin / ECEI-only / BES-only / no-Mirnov | epochs run (best), cap reached |", "|---|---|---|---|"]
        for s in SEEDS:
            x = D["splits"][str(s)]
            g, m = x["over_causal_gp"], x["median_abs_error_eV"]
            ep = ", ".join(f"{k} {v[0]} ({v[1]}){' cap' if v[2] else ''}" for k, v in x["epochs_run_best_capped"].items())
            o.append(f"| {s} | " + " / ".join(s3(g[k]) for k in ("v2", "v2e", "v2b", "v2n", "v2h")) + " | " +
                     " / ".join(f"{m[k]:.1f}" for k in ("hold", "causal GP", "v2", "v2h", "v2e", "v2b", "v2n")) + f" | {ep} |")
        if "R" in res:
            R = res["R"]
            o += ["", "R1 to R3 of B.45 with three initialisations (classes of run_b45):", "",
                  f"- R1 (BES-only over the twin in classes 3 to 6): {R['R1']['splits']} of 4 -> {'holds' if R['R1']['holds'] else 'does not hold'}; "
                  f"R2 (quiet rows): {R['R2']['splits']} of 4 -> {'holds' if R['R2']['holds'] else 'does not hold'}; "
                  f"R3 (ECEI-only over the twin after an absent frame): {R['R3']['splits']} of 4 -> {'holds' if R['R3']['holds'] else 'does not hold'}.", "",
                  "| split | R1 | R2 | R3 | rows R1 / R2 / R3 |", "|---|---|---|---|---|"]
            for s in SEEDS:
                x = R["splits"][str(s)]
                o.append(f"| {s} | {s3(x['R1'])} | {s3(x['R2'])} | {s3(x['R3'])} | {x['rows']['R1']:,} / {x['rows']['R2']:,} / {x['rows']['R3']:,} |")
    if L.get("complete"):
        o += ["", "## Arm L: trained with a delayed CES history", "",
              "| split | L | L1: trained at 2 over causal GP | L2: over twin trained at 2 | L3: over estimator trained without delay | "
              "over hold | twin trained at 2 over twin trained without delay |", "|---|---|---|---|---|---|---|"]
        for s in SEEDS:
            for Lx in map(str, LS):
                k = L["splits"][str(s)][Lx]["skill"]
                o.append(f"| {s} | {Lx} | {s3(k['causal_L2_vs_gp'])} | {s3(k['causal_L2_vs_twin_L2'])} | "
                         f"{s3(k['causal_L2_vs_causal_b39'])} | {s3(k['causal_L2_vs_hold'])} | {s3(k['twin_L2_vs_twin_b29'])} |")
        o += ["", "Networks trained without delay, frozen, read at the same delays (the B.42 question at cap 80), and the L = 0 controls:", "",
              "| split | L | estimator without delay over causal GP | over twin without delay | L = 0 control over causal GP | "
              "L = 0 control over the B.39 run | twin L = 0 control over twin without delay | causal GP over hold |",
              "|---|---|---|---|---|---|---|---|"]
        for s in SEEDS:
            for Lx in map(str, LS):
                k = L["splits"][str(s)][Lx]["skill"]
                o.append(f"| {s} | {Lx} | {s3(k['causal_b39_vs_gp'])} | {s3(k['causal_b39_vs_twin_b29'])} | {s3(k['causal_L0_vs_gp'])} | "
                         f"{s3(k['causal_L0_vs_causal_b39'])} | {s3(k['twin_L0_vs_twin_b29'])} | {s3(k['gp_vs_hold'])} |")
        o += ["", "Median absolute error (eV):", "", "| split | L | rows | " + " | ".join(LAB[e] for e in ("hold", "gp", "causal_L2", "twin_L2",
              "causal_b39", "twin_b29", "causal_L0", "twin_L0")) + " |", "|---|---|---|" + "---|" * 8]
        for s in SEEDS:
            for Lx in map(str, LS):
                x = L["splits"][str(s)][Lx]
                o.append(f"| {s} | {Lx} | {x['rows']:,} | " + " | ".join(f"{x['median_abs'][e][0]:.1f}" for e in
                         ("hold", "gp", "causal_L2", "twin_L2", "causal_b39", "twin_b29", "causal_L0", "twin_L0")) + " |")
        o += ["", "L = 0 control on the harness of eval_seq (undelayed test rows of B.39):", "",
              "| split | network | over causal GP | reference over causal GP | difference | within 0.02 | over the reference | "
              "RMSE (eV) run / reference | epochs run (best), cap |", "|---|---|---|---|---|---|---|---|---|"]
        for s in SEEDS:
            for nm, x in L["L0_control_eval_seq"][str(s)].items():
                e = x["epochs_run_best_capped"]
                o.append(f"| {s} | {nm} ({x['run']} vs {x['reference']}) | {x['over_causal_gp']:+.3f} | {x['reference_over_causal_gp']:+.3f} | "
                         f"{x['difference']:+.3f} | {x['within_0.02']} | {s3(x['over_reference'])} | "
                         f"{x['rmse_eV'][0]:.1f} / {x['rmse_eV'][1]:.1f} | {e[0]} ({e[1]}){' cap' if e[2] else ''} |")
    if I["splits"]:
        o += ["", "## Arm I: inclusive population (values above 3 keV kept)", "",
              "| split | rows | S1i: over causal GP | S2i: over twin | twin over causal GP | over hold | over PCHIP | over offline GP | "
              "spread over GP | median abs. error (eV): hold / causal GP / PCHIP / offline GP / causal variant / twin | epochs run (best), cap |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s, x in I["splits"].items():
            m = x["median_abs_error_eV"]
            ep = ", ".join(f"{k} {v[0]} ({v[1]}){' cap' if v[2] else ''}" for k, v in x["epochs_run_best_capped"].items())
            o.append(f"| {s} | {x['rows']:,} | {s3(x['S1i_causal_over_gp'])} | {s3(x['S2i_causal_over_twin'])} | {s3(x['twin_over_gp'])} | "
                     f"{s3(x['causal_over']['persistence'])} | {s3(x['causal_over']['pchip'])} | {s3(x['causal_over']['gp'])} | "
                     f"{x['spread_causal_over_gp']:.3f} | " + " / ".join(f"{m[k]:.1f}" for k in m) + f" | {ep} |")
    o += ["", "## Expectations of the registration (section 6)", ""]
    for e in res["expectations"]:
        o.append(f"- {e['expectation']}: " + ", ".join(f"{x:+.3f}" for x in e["points"]) +
                 f" (count {e['count']}) -> {'met' if e['met'] else 'not met'}")
    if not res["expectations"]:
        o.append("- none read yet (no arm complete).")
    o += ["", "## Choices made where the registration left room", ""] + [f"- {c}" for c in CHOICES]
    return "\n".join(o) + "\n"


# =============================================================================== main
def ctx():
    sys.path.insert(0, str(HERE.parent / "b37_plan"))
    from run_b37 import Ctx
    return Ctx()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-d", action="store_true", help="arm D and R1-R3 code path with one initialisation (prints)")
    ap.add_argument("--check-l", action="store_true", help="arm L machinery on B.42's networks for one split (prints)")
    ap.add_argument("--split", type=int, default=42)
    ap.add_argument("--fresh", action="store_true", help="recompute the arm L row cache")
    ap.add_argument("--no-r", action="store_true", help="skip R1 to R3 (they load the B.41 context)")
    args = ap.parse_args()
    if args.check_d:
        D = arm_d(np.random.default_rng(SEED_D), one=True)
        ok, comp = check_d_against_b39(D)
        for k, v in comp.items():
            print(f"{k:28s} mine {s3(v['mine'])}  B.39 {s3(v['b39'])}  point equal (3 dp) {v['point_equal_3dp']}  "
                  f"max interval dev {v['max_interval_dev']:.4f}")
        print(f"arm D one-initialisation path reproduces B.39 points: {ok}")
        if not args.no_r:
            R = r_rows(np.random.default_rng(SEED_D), one=True)
            okr, compr = check_r_against_b45(R)
            for k, v in compr.items():
                print(f"{k:10s} mine {s3(v['mine'])}  B.45 {s3(v['b45'])}  point equal (3 dp) {v['point_equal_3dp']}  "
                      f"max interval dev {v['max_interval_dev']:.4f}")
            print(f"R1-R3 one-initialisation path reproduces B.45 points: {okr}")
        return
    if args.check_l:
        C = ctx()
        rows = rows_split(C, args.split, ["causal_b29", "twin_b29", "causal_b39"])
        print("input check files:", rows.pop("input_check_files"))
        rep = reproduce_b42(rows, args.split, boot_weights())
        for k, v in rep["max_abs_dev_eV"].items():
            print(f"max |mine - B.42 cached row| {k:16s} {v:.2e} eV")
        for k, v in rep["skills"].items():
            print(f"{k:22s} mine {s3(v['mine'])}  B.42 {s3(v['b42'])}  equal (3 dp, point and interval) {v['equal_3dp_point_and_interval']}")
        sk = skills(rows, ("hold", "gp", "causal_b39", "twin_b29"), [("causal_b39", "gp"), ("causal_b39", "twin_b29")],
                    *boot_weights()[args.split])
        for Lx in map(str, LS):
            print(f"(after the fact, cap-80 B.39 estimator frozen) L = {Lx}: over causal GP {s3(sk[Lx]['skill']['causal_b39_vs_gp'])}, "
                  f"over twin {s3(sk[Lx]['skill']['causal_b39_vs_twin_b29'])}")
        print(f"reproduce B.42 on split {args.split}: {'PASSES' if rep['passes'] else 'FAILS'}")
        return
    res = {"registration": REG, "checks": {}}
    D = arm_d(np.random.default_rng(SEED_D))
    if D.get("complete"):
        ok, comp = check_d_against_b39(arm_d(np.random.default_rng(SEED_D), one=True))
        res["checks"]["arm D: the one-initialisation path reproduces the points of docs/paper/b39_primary.md (D1, D2, no-Mirnov, "
                      "full over BES-only, full over twin; 16 values, three decimals)"] = ok
        if not ok:
            raise SystemExit("FATAL: the arm D path does not reproduce B.39")
        if not args.no_r:
            res["R"] = r_rows(np.random.default_rng(SEED_D))
    res["D"] = D
    res["L"] = arm_l(ctx(), args.fresh) if l_status()["runs_done"] == l_status()["runs_total"] else {"complete": False, **l_status()}
    res["I"] = arm_i(np.random.default_rng(SEED_I))
    res["expectations"] = expectations(res)
    res["choices"] = CHOICES
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"written {OUT_MD.name}: D complete {D.get('complete')}, L complete {res['L'].get('complete')}, "
        f"I splits {list(res['I']['splits'])}")


if __name__ == "__main__":
    main()
