"""B.57 (PREREGISTRATION_B57.md with Addendum 1): yardsticks for the size of the gain -- a noise floor, the gap
to the offline estimate, and whether the fast-channel correction lasts into the next frame.  CPU only.

Signed estimates on the scored rows of the pooled cross-validation of B.39 (r = 2..9, k = 0..4):
  network  data/.b55/run_r{r}_f{k}.npz  (`est_base`: the 40 estimator networks run forward on a CPU by B.55)
  twin     data/.b57/twin_r{r}_f{k}.npz (the 40 stored twins `data/.b39p_v2h_cut_r{r}_f{k}` run forward here, with
           the grid, block construction and forward pass of `run_b55.Nets`, imported)
  KF0, KFX data/.b56/r{r}_f{k}.npz      (B.56)
Squared errors of the mean-squared-error quantities (floor ratios, removable-error skills, gap closure) are the
stored ones of the pooled runs (as B.39 and the look of section 2); the filters' are those of B.56.

Usage (repo root):
  py ces_prediction/experiments/b57_yardsticks/run_b57.py twins     # 40 twins forward -> data/.b57/
  py ces_prediction/experiments/b57_yardsticks/run_b57.py checks    # checks (i)-(iii)
  py ces_prediction/experiments/b57_yardsticks/run_b57.py analyze   # statements -> docs/paper/b57_yardsticks.{json,md}
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[2]
DATA = ROOT / "data"
CACHE = DATA / ".b57"
B55C = DATA / ".b55"
B56C = DATA / ".b56"
OUT_JSON = ROOT / "docs" / "paper" / "b57_yardsticks.json"
OUT_MD = ROOT / "docs" / "paper" / "b57_yardsticks.md"
sys.path.insert(0, str(EXP / "b55_sample_timing"))
REPEATS = tuple(range(2, 10))
K = 5
N_BOOT = 2000
SEED = 57
CHECK_EV, CHECK_SHARE = 0.5, 0.99
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def q(b, lo=2.5, hi=97.5):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.percentile(b, lo)), float(np.percentile(b, hi))]


# ------------------------------------------------------------------ twins forward (B.55 machinery)
def stage_twins():
    import run_b55 as B55
    N = B55.Nets()
    torch = N.torch
    torch.set_num_threads(2)
    from seq_models import SEQ_MODELS
    log(f"grid: {len(N.names)} files; treatment {N.tr}")
    CACHE.mkdir(exist_ok=True)
    for r in REPEATS:
        for k in range(K):
            out = CACHE / f"twin_r{r}_f{k}.npz"
            if out.exists():
                continue
            t0 = time.time()
            run = B55.B39.pooled_run_dir("v2h", r, k)
            m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            env = B55.B39.pooled_env("v2h", r, k, run)
            for key in ("CES_DROP_STUCK_TARGETS", "CES_TI_SPIKE_CUT_EV"):
                if env.get(key) != N.tr.get(key):
                    raise SystemExit(f"FATAL: r{r} f{k}: twin treatment {key}={env.get(key)} differs from {N.tr.get(key)}")
            if m["seq_model"] != "v2h" or str(m["seed"]) != env["CES_SEED"] or str(m["init_seed"]) != env["CES_INIT_SEED"] \
                    or m.get("extra_vt_channels") or m.get("extra_ti_channels") or m.get("train_ctx") or m.get("zero_mc") \
                    or m.get("per_shot_norm_expanding"):
                raise SystemExit(f"FATAL: {run.name} is not the registered twin of the runner")
            stats = N.load_stats(m)
            model = SEQ_MODELS["v2h"]()
            model.load_state_dict(torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
            model.eval()
            psn = bool(m["per_shot_input_norm"])
            z = np.load(B55C / f"run_r{r}_f{k}.npz")
            names, grow = z["names"], z["grow"]
            est = np.full(len(grow), np.nan)
            mean, std = float(stats["target"]["mean"][0]), float(stats["target"]["std"][0])
            for nm in np.unique(names):
                sel = np.flatnonzero(names == nm)
                arr = N.grid[str(nm)]
                tm = arr[:, 0]
                pred = np.full(len(tm), np.nan)
                spans = B55.block_spans(tm.astype(np.float64))
                blks = N.build_blocks(arr, N.dims, stats, per_shot_norm=psn, expanding=False)
                if len(blks) != len(spans):
                    raise SystemExit("FATAL: block spans do not match build_blocks")
                with torch.no_grad():
                    for blk, (a, b) in zip(blks, spans):
                        if not np.array_equal(blk["time"], arr[a:b, 0]):
                            raise SystemExit("FATAL: block times do not match the grid")
                        pred[a:b] = model(torch.from_numpy(blk["x"]).unsqueeze(0))[0][:, 0].numpy() * std + mean
                est[sel] = pred[grow[sel]]
            np.savez(out, shot=z["shot"], row=z["row"], grow=grow, est_twin=est, per_shot_input_norm=psn)
            log(f"twin r{r} f{k}: {len(grow)} rows ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------ assembled rows
def load_fold(r, k):
    a = np.load(B55C / f"run_r{r}_f{k}.npz")
    t = np.load(CACHE / f"twin_r{r}_f{k}.npz")
    f = np.load(B56C / f"r{r}_f{k}.npz")
    p = np.load(DATA / f".b39p_v2_cut_r{r}_f{k}" / "comparison_errors_test.npz")
    for name, z in (("twin", t), ("B.56", f)):
        if not (np.array_equal(z["shot"], a["shot"]) and np.array_equal(z["row"], a["row"])):
            raise SystemExit(f"FATAL: r{r} f{k}: {name} rows differ from B.55's")
    if not np.array_equal(p["CES_TI_shot"], a["shot"]) or not np.array_equal(p["CES_TI_row"], a["row"]):
        raise SystemExit(f"FATAL: r{r} f{k}: stored rows differ")
    if not np.all(a["grow"] - f["gi"] == 1):
        raise SystemExit(f"FATAL: r{r} f{k}: B.55 and B.56 locate the rows differently")
    y = a["y"].astype(float)
    d = {"shot": a["shot"], "grow": a["grow"], "y": y, "dt": p["CES_TI_dt_ms"].astype(float),
         "after_absent10": None,
         # stored squared errors (pooled runs)
         "A": a["se_stored"], "Tw": a["se_twin"], "G": a["se_gp"], "H": a["se_hold"], "O": a["se_gpoff"], "P": a["se_pchip"],
         "K0": (f["est_kf0"] - y) ** 2, "KX": (f["est_kfx"] - y) ** 2,
         # signed estimates
         "sA": a["est_base"], "sTw": t["est_twin"], "sK0": f["est_kf0"], "sKX": f["est_kfx"],
         "se_twin_stored": a["se_twin"], "gi": f["gi"]}
    return d


_TIMES = {}


def after_absent10(d):
    """B.56's definition: the ten grid rows starting at the first row after a step of 15 ms to 0.5 s."""
    sys.path.insert(0, str(EXP / "b56_kfx"))
    import run_b56 as B56
    out = np.zeros(len(d["gi"]), bool)
    for fi in np.unique(d["shot"]):
        t = B56.grid_time(int(fi))
        step = np.diff(t, prepend=np.nan)
        blip = (step >= 0.015) & (step < 0.5)
        near = np.zeros(len(t), bool)
        for j in np.flatnonzero(blip):
            near[j:j + 10] = True
        m = d["shot"] == fi
        out[m] = near[d["gi"][m]]
    return out


def chains(d, h):
    """Index pairs (i, j): rows t and t+h, every frame t..t+h scored, consecutive grid rows, each step < 15 ms."""
    o = np.lexsort((d["grow"], d["shot"]))
    sh, gr, dt = d["shot"][o], d["grow"][o], d["dt"][o]
    n = len(o)
    ok = np.ones(n, bool)
    if h == 0:
        return o, o
    ok[n - h:] = False
    for s in range(1, h + 1):
        ok[:n - s] &= (sh[s:] == sh[:n - s]) & (gr[s:] == gr[:n - s] + s) & (dt[s:] < 15)
    i = np.flatnonzero(ok)
    return o[i], o[i + h]


def triples(d):
    """Rows t with t-1 and t+1 scored, consecutive grid rows, stored time step < 15 ms at t and t+1 (Addendum 1)."""
    o = np.lexsort((d["grow"], d["shot"]))
    sh, gr, dt = d["shot"][o], d["grow"][o], d["dt"][o]
    ok = np.zeros(len(o), bool)
    ok[1:-1] = (sh[2:] == sh[1:-1]) & (sh[1:-1] == sh[:-2]) & (gr[2:] == gr[1:-1] + 1) & (gr[1:-1] == gr[:-2] + 1) \
        & (dt[2:] < 15) & (dt[1:-1] < 15)
    i = np.flatnonzero(ok)
    return o[i - 1], o[i], o[i + 1]


EST = ("A", "Tw", "G", "H", "O", "P", "K0", "KX")


class Rep:
    """Per re-partition: per-discharge sums for the floor (on triples), the MSEs on triples and on all rows, and
    the persistence moments for each pair and lag."""

    PAIRS = {"net_twin": ("sA", "sTw"), "kfx_kf0": ("sKX", "sK0"), "twin_kf0": ("sTw", "sK0")}

    def __init__(self, r, mask_fn=None, y_override=None):
        D = [load_fold(r, k) for k in range(K)]
        d = {key: np.concatenate([x[key] for x in D]) for key in D[0] if D[0][key] is not None}
        if y_override is not None:
            d = y_override(d)
        self.d = d
        msk = mask_fn(d) if mask_fn else np.ones(len(d["y"]), bool)
        self.uf = np.unique(d["shot"])
        ix = lambda sel: np.searchsorted(self.uf, d["shot"][sel])                       # noqa: E731
        n = len(self.uf)
        bc = lambda sel, v: np.bincount(ix(sel), v, n)                                   # noqa: E731
        S = {}
        # floor on triples
        im, i0, ip = triples(d)
        keep = msk[i0]
        im, i0, ip = im[keep], i0[keep], ip[keep]
        dy1, dy2 = d["y"][i0] - d["y"][im], d["y"][ip] - d["y"][i0]
        S["tri_n"] = bc(i0, np.ones(len(i0)))
        S["tri_c"], S["tri_m1"], S["tri_m2"] = bc(i0, dy1 * dy2), bc(i0, dy1), bc(i0, dy2)
        for e in EST:
            S[f"tri_{e}"] = bc(i0, d[e][i0])
        allr = np.flatnonzero(msk)
        S["all_n"] = bc(allr, np.ones(len(allr)))
        for e in EST:
            S[f"all_{e}"] = bc(allr, d[e][allr])
        # persistence
        for h in (0, 1, 3):
            I, J = chains(d, h)
            keep = msk[I]
            I, J = I[keep], J[keep]
            S[f"n_h{h}"] = bc(I, np.ones(len(I)))
            for pn, (a, b) in self.PAIRS.items():
                dd = d[a][I] - d[b][I]
                e0 = d["y"][I] - d[b][I]
                eh = d["y"][J] - d[b][I]
                S[f"{pn}_h{h}_d"] = bc(I, dd)
                S[f"{pn}_h{h}_dd"] = bc(I, dd * dd)
                S[f"{pn}_h{h}_e0"] = bc(I, e0)
                S[f"{pn}_h{h}_eh"] = bc(I, eh)
                S[f"{pn}_h{h}_de0"] = bc(I, dd * e0)
                S[f"{pn}_h{h}_deh"] = bc(I, dd * eh)
        self.S = S
        self.n = n

    @staticmethod
    def stats(S, w):
        W = lambda k: (S[k] * w).sum(-1)                                               # noqa: E731
        o = {}
        nt = W("tri_n")
        s2 = -(W("tri_c") / nt - (W("tri_m1") / nt) * (W("tri_m2") / nt))
        o["s2"] = s2
        o["rms"] = np.sqrt(np.maximum(s2, 0))
        mt = {e: W(f"tri_{e}") / nt for e in EST}
        for e in EST:
            o[f"mse_over_s2_{e}"] = mt[e] / s2
        for a in ("A", "KX"):
            for r_ in ("G", "Tw", "H"):
                o[f"removable_{a}_over_{r_}"] = (mt[r_] - mt[a]) / (mt[r_] - s2)
        ma = {e: W(f"all_{e}") for e in EST}
        for a in ("A", "Tw", "KX", "K0"):
            o[f"gap_closure_{a}"] = (ma["G"] - ma[a]) / (ma["G"] - ma["O"])
        for pn in Rep.PAIRS:
            beta = {}
            for h in (0, 1, 3):
                nn = W(f"n_h{h}")
                md = W(f"{pn}_h{h}_d") / nn
                var = W(f"{pn}_h{h}_dd") / nn - md ** 2
                c0 = W(f"{pn}_h{h}_de0") / nn - md * W(f"{pn}_h{h}_e0") / nn
                ch = W(f"{pn}_h{h}_deh") / nn - md * W(f"{pn}_h{h}_eh") / nn
                beta[h] = (c0 / var, ch / var)       # beta_0 and beta_h on the rows valid for lag h
            o[f"{pn}_beta0"] = beta[0][0]
            o[f"{pn}_rho1"] = beta[1][1] / beta[1][0]
            o[f"{pn}_rho3"] = beta[3][1] / beta[3][0]
            o[f"{pn}_rho3_minus_rho1"] = o[f"{pn}_rho3"] - o[f"{pn}_rho1"]
        return o


def pooled(reps, rng):
    """Point: mean over re-partitions; interval: a re-partition drawn, then its discharges resampled."""
    pts = [Rep.stats(R.S, np.ones(R.n)) for R in reps]
    keys = list(pts[0])
    out = {k: [float(np.mean([p[k] for p in pts]))] for k in keys}
    draws = rng.integers(0, len(reps), N_BOOT)
    boot = {k: np.empty(N_BOOT) for k in keys}
    for j, R in enumerate(reps):
        sel = np.flatnonzero(draws == j)
        if not len(sel):
            continue
        W = rng.multinomial(R.n, np.full(R.n, 1.0 / R.n), size=len(sel)).astype(float)
        st = Rep.stats({k: v[None, :] for k, v in R.S.items()}, W)
        for k in keys:
            boot[k][sel] = st[k]
    for k in keys:
        out[k] += q(boot[k])
    out["_per_repeat"] = {k: [float(p[k]) for p in pts] for k in keys}
    return out


# ------------------------------------------------------------------ checks
def checks():
    res = {}
    # (i) twins reproduce their stored squared errors (B.55 rule)
    per = {}
    for r in REPEATS:
        for k in range(K):
            t = np.load(CACHE / f"twin_r{r}_f{k}.npz")
            a = np.load(B55C / f"run_r{r}_f{k}.npz")
            dev = np.abs(np.abs(t["est_twin"] - a["y"]) - np.sqrt(a["se_twin"]))
            per[f"r{r}_f{k}"] = {"share_within_0.5_eV": float(np.mean(dev <= CHECK_EV)), "median_dev_eV": float(np.median(dev)),
                                 "max_dev_eV": float(np.nanmax(dev)), "finite": bool(np.isfinite(t["est_twin"]).all())}
    smin = min(v["share_within_0.5_eV"] for v in per.values())
    res["i"] = {"per_twin": per, "min_share_within_0.5_eV": smin,
                "median_dev_eV": float(np.median([v["median_dev_eV"] for v in per.values()])),
                "passes": bool(smin >= CHECK_SHARE and all(v["finite"] for v in per.values()))}
    # (ii) floor and removable-error skills of section 2 (point estimates, three significant figures)
    reps = [Rep(r) for r in REPEATS]
    pts = [Rep.stats(R.S, np.ones(R.n)) for R in reps]
    m = lambda k: float(np.mean([p[k] for p in pts]))                                  # noqa: E731
    mine = {"floor_eV2": m("s2"), "mse_over_floor_net": m("mse_over_s2_A"), "mse_over_floor_gp": m("mse_over_s2_G"),
            "mse_over_floor_offline_gp": m("mse_over_s2_O"), "removable_net_over_gp": m("removable_A_over_G"),
            "removable_net_over_twin": m("removable_A_over_Tw")}
    look = {"floor_eV2": 18414.7, "mse_over_floor_net": 1.3312, "mse_over_floor_gp": 1.4717, "mse_over_floor_offline_gp": 1.2792,
            "removable_net_over_gp": 0.2977, "removable_net_over_twin": 0.2017}
    sig3 = lambda x: float(f"{x:.3g}")                                                   # noqa: E731
    res["ii"] = {k: {"look": look[k], "batch": mine[k], "same_3_sig": sig3(look[k]) == sig3(mine[k])} for k in look}
    res["ii"]["passes"] = bool(all(v["same_3_sig"] for k, v in res["ii"].items() if k != "passes"))
    # (iii) with y_{t+1} replaced by y_t, rho_1 = 1
    # direct: recompute rho_1 with eh := e0 on the same chains
    vals = []
    for R in reps[:2]:
        S = dict(R.S)
        for pn in Rep.PAIRS:
            S[f"{pn}_h1_deh"] = S[f"{pn}_h1_de0"]
            S[f"{pn}_h1_eh"] = S[f"{pn}_h1_e0"]
        st = Rep.stats(S, np.ones(R.n))
        vals += [float(st[f"{pn}_rho1"]) for pn in Rep.PAIRS]
    res["iii"] = {"rho1_values": vals, "passes": bool(np.allclose(vals, 1.0, atol=1e-12))}
    res["all_pass"] = bool(res["i"]["passes"] and res["ii"]["passes"] and res["iii"]["passes"])
    CACHE.mkdir(exist_ok=True)
    (CACHE / "checks.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"check (i) {res['i']['passes']} (min share {smin:.4f}, median dev {res['i']['median_dev_eV']:.3g} eV); "
        f"(ii) {res['ii']['passes']}; (iii) {res['iii']['passes']}")
    for k in look:
        log(f"  (ii) {k}: look {look[k]} batch {mine[k]:.5g}")
    return res


# ------------------------------------------------------------------ analysis
def analyze():
    chk = json.loads((CACHE / "checks.json").read_text(encoding="utf-8"))
    if not chk["all_pass"]:
        raise SystemExit("FATAL: a check failed; no statement is read")
    rng = np.random.default_rng(SEED)
    log("all rows ...")
    reps = [Rep(r) for r in REPEATS]
    P = pooled(reps, rng)
    log("rows not after an absent frame ...")
    repq = [Rep(r, mask_fn=lambda d: ~after_absent10(d)) for r in REPEATS]
    Pq = pooled(repq, rng)
    res = {"checks": chk, "all": P, "quiet": Pq,
           "rows": {str(r): int(R.S["all_n"].sum()) for r, R in zip(REPEATS, reps)},
           "triple_rows": {str(r): int(R.S["tri_n"].sum()) for r, R in zip(REPEATS, reps)},
           "chain_rows_h1": {str(r): int(R.S["n_h1"].sum()) for r, R in zip(REPEATS, reps)},
           "chain_rows_h3": {str(r): int(R.S["n_h3"].sum()) for r, R in zip(REPEATS, reps)}}
    n1 = {e: P[f"mse_over_s2_{e}"] for e in EST}
    st = {"N1": {"value": n1, "rule": "every interval of MSE over the floor above 1", "holds": all(v[1] > 1 for v in n1.values())},
          "N2": {"value": P["gap_closure_A"], "rule": "interval above 0.5", "holds": P["gap_closure_A"][1] > 0.5},
          "N3": {"value": P["net_twin_rho1"], "rule": "interval above 0.5", "holds": P["net_twin_rho1"][1] > 0.5},
          "N4": {"value": P["kfx_kf0_rho1"], "rule": "interval above 0.5", "holds": P["kfx_kf0_rho1"][1] > 0.5},
          "N5": {"value": P["net_twin_rho3_minus_rho1"], "rule": "interval inside [-0.15, +0.15]",
                 "holds": P["net_twin_rho3_minus_rho1"][1] > -0.15 and P["net_twin_rho3_minus_rho1"][2] < 0.15}}
    for k in st:
        st[k]["holds"] = bool(st[k]["holds"])
    res["statements"] = st
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"written {OUT_JSON} and {OUT_MD}")
    for k, v in st.items():
        if k == "N1":
            log(f"N1: " + ", ".join(f"{e} {x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]" for e, x in v["value"].items()) + f" -> {v['holds']}")
        else:
            x = v["value"]
            log(f"{k}: {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}] -> {'holds' if v['holds'] else 'does not hold'}")


ENAME = {"A": "network", "Tw": "twin", "G": "causal GP", "H": "hold", "O": "offline GP", "P": "PCHIP", "K0": "KF0", "KX": "KFX"}
EXPECT = {"N1": "holds; network 1.25 to 1.45", "N2": "0.60 to 0.85, interval reaching down to about 0.45; may fail",
          "N3": "rho_1 0.70 to 0.90, holds", "N4": "rho_1 0.70 to 0.95, holds", "N5": "-0.10 to +0.10, holds"}
ERANGE = {"N2": (0.60, 0.85), "N3": (0.70, 0.90), "N4": (0.70, 0.95), "N5": (-0.10, 0.10)}


def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def r3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def render(res):
    P, Pq, st, ch = res["all"], res["quiet"], res["statements"], res["checks"]
    L = ["# B.57: a noise floor, the gap to the offline estimate, and the persistence of the fast-channel correction", "",
         "Registration: `ces_prediction/experiments/PREREGISTRATION_B57.md` with Addendum 1. Runner: "
         "`ces_prediction/experiments/b57_yardsticks/run_b57.py`. CPU only; the 40 stored twins of the pooled cross-validation "
         "were run forward for their signed estimates; nothing was trained. T_i, cut population, the rows of the pooled "
         "cross-validation of B.39, re-partitions r2 to r9.", "",
         "Sources: squared errors of the network, twin, hold, causal GP, offline GP and PCHIP are the stored ones of the pooled runs; "
         "the filters' (KF0, KFX) are from B.56 (`data/.b56/`). Signed estimates for the persistence: the network from B.55 "
         "(`data/.b55/run_r{r}_f{k}.npz`, `est_base`), the twin from this batch (`data/.b57/twin_r{r}_f{k}.npz`), the filters from B.56. "
         "Point estimates are the mean over the eight re-partitions; intervals: a re-partition drawn, then its discharges "
         "resampled, 2,000 draws. beta_h = cov(d_t, y_{t+h} - B_t) / var(d_t) (centred, pooled over the rows of a re-partition), "
         "and rho_h = beta_h / beta_0 with both computed on the rows whose frames t to t+h are consecutive, scored and 10 ms apart.", "",
         "## Checks", ""]
    ci_ = ch["i"]
    L.append(f"- (i) twins run on the CPU reproduce their stored squared errors: {'pass' if ci_['passes'] else 'FAIL'}; the smallest share "
             f"of rows within 0.5 eV over the 40 twins is {ci_['min_share_within_0.5_eV']:.4f} (required 0.99); median deviation "
             f"{ci_['median_dev_eV']:.3g} eV.")
    L.append(f"- (ii) section 2 reproduced on point estimates (three significant figures): {'pass' if ch['ii']['passes'] else 'FAIL'}; " +
             "; ".join(f"{k} look {v['look']} batch {v['batch']:.5g}" for k, v in ch["ii"].items() if k != "passes") + ".")
    L.append(f"- (iii) with the value of frame t+1 replaced by that of frame t, rho_1 = {', '.join(f'{x:.12f}' for x in ch['iii']['rho1_values'])}: "
             f"{'pass' if ch['iii']['passes'] else 'FAIL'}.")
    L += ["", "## Statements", "", "| | Statement | Value [95 %] | Verdict | Expected | Point inside the expected range |", "|---|---|---|---|---|---|"]
    for k, v in st.items():
        if k == "N1":
            val = "; ".join(f"{ENAME[e]} {r3(x)}" for e, x in v["value"].items())
            inside = "network " + ("yes" if 1.25 <= v["value"]["A"][0] <= 1.45 else "no")
        else:
            val = s3(v["value"]) if k == "N5" else r3(v["value"])
            lo, hi = ERANGE[k]
            inside = "yes" if lo <= v["value"][0] <= hi else "no"
        L.append(f"| {k} | {v['rule']} | {val} | {'holds' if v['holds'] else 'does not hold'} | {EXPECT[k]} | {inside} |")
    L += ["", "N1 MSE of every estimator over the floor; N2 gap closure of the network, (MSE_GP - MSE_net) / (MSE_GP - MSE_offline GP); "
          "N3 rho_1 of the network minus its twin; N4 rho_1 of KFX minus KF0; N5 rho_3 - rho_1 of the network minus its twin.", "",
          "## Reported", "",
          f"Floor (lower bound of the mean white variance, on {sum(res['triple_rows'].values()):,} triple rows over the eight re-partitions): "
          f"{P['s2'][0]:,.0f} [{P['s2'][1]:,.0f}, {P['s2'][2]:,.0f}] eV^2, root mean square {r3(P['rms'])} eV (expected 126 to 145 eV).", "",
          "| | All rows | Rows not after an absent frame |", "|---|---|---|"]
    for e in EST:
        L.append(f"| MSE over the floor, {ENAME[e]} | {r3(P[f'mse_over_s2_{e}'])} | {r3(Pq[f'mse_over_s2_{e}'])} |")
    for a in ("A", "KX"):
        for r_ in ("G", "Tw", "H"):
            L.append(f"| removable-error skill, {ENAME[a]} over {ENAME[r_]} | {s3(P[f'removable_{a}_over_{r_}'])} | "
                     f"{s3(Pq[f'removable_{a}_over_{r_}'])} |")
    for a in ("A", "Tw", "KX", "K0"):
        L.append(f"| gap closure, {ENAME[a]} | {r3(P[f'gap_closure_{a}'])} | {r3(Pq[f'gap_closure_{a}'])} |")
    for pn, lab in (("net_twin", "network - twin"), ("kfx_kf0", "KFX - KF0"), ("twin_kf0", "twin - KF0 (reference)")):
        for qn in ("beta0", "rho1", "rho3", "rho3_minus_rho1"):
            f_ = s3 if qn == "rho3_minus_rho1" else r3
            L.append(f"| {qn}, {lab} | {f_(P[f'{pn}_{qn}'])} | {f_(Pq[f'{pn}_{qn}'])} |")
    L += ["", "Expected: floor 126 to 145 eV; MSE over the floor 1.25 to 1.45 for the network; removable-error skill over the causal GP "
          "+0.19 to +0.41; beta_0 0.6 to 0.9.", "",
          "Rows: all " + ", ".join(f"r{r} {n:,}" for r, n in res["rows"].items()) + "; chains h = 1 " +
          ", ".join(f"{n:,}" for n in res["chain_rows_h1"].values()) + "; h = 3 " + ", ".join(f"{n:,}" for n in res["chain_rows_h3"].values()) + ".", "",
          "Rows not after an absent frame: B.56's definition (the ten grid rows starting at the first row after a step of 15 ms to 0.5 s); "
          "for the persistence the row t must be outside them.", ""]
    return "\n".join(L)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "twins":
        stage_twins()
    elif cmd == "checks":
        checks()
    elif cmd == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
