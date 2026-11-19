"""B.71 (PREREGISTRATION_B71.md, addendum PREREGISTRATION_B71_ADDENDUM1.md): frozen estimators when the CES updates stop.

Replays of the D1 rows (B.70 section 2) of the pooled r2-r4 folds under four CES-availability scenarios:
  U  every sample measured at a row before t is available (the present protocol)
  F  from a cut c on, samples measured at or after c are not available; the network is told their true age
  R  as F, but from the row after the cut on the six CES channels hold the estimator's own estimate of the
     previous row (T_i and V_rot, standardised, staleness = the time step, flag 1); stepped row by row
  G  samples measured in [c, c + g) are not available (g = 0.1, 0.3, 1.0 s); afterwards every sample again
Cuts: c_j = block start + j x 1 s; kept when >= 3 T_i samples lie before c in the block and the block runs
>= 0.5 s after c.  The six CES channels are rebuilt from the available samples by run_b42.ces_channels (true
staleness), whose `obs` argument is the availability mask here.  F, G and R start from the LSTM state of the
U pass at the row before the cut; rows before the cut are those of U.

Estimators: network `.b39p_v2_cut_r{r}_f{k}` (v2, train-file standardisation) and twin `.b39p_v2h_cut_r{r}_f{k}`
(v2h, per-discharge standardisation as stored), frozen, CPU; hold and the causal GP on the available samples;
the B.56 filters (KF0, KFX) once B.70 phase 2 has delivered their parameters (`kf` command).

CPU only (CUDA hidden, two threads), nothing is trained.  Run from the repo root:
    py ces_prediction/experiments/b71_scenarios/run_b71.py check  [--fold r2f0] [--files 5]
    py ces_prediction/experiments/b71_scenarios/run_b71.py timing [--fold r2f0] [--files 5]
    py ces_prediction/experiments/b71_scenarios/run_b71.py rows   [--folds r2f0 r2f1 ...]     (default: all 15)
    py ces_prediction/experiments/b71_scenarios/run_b71.py kf     [--params PATH]
    py ces_prediction/experiments/b71_scenarios/run_b71.py analyze
Rows: data/.b71/rows_r{r}_f{k}.npz (gitignored).  Checks: docs/paper/b71_checks.md.
Readout: docs/paper/b71_scenarios.{md,json} (analyze_b71.py).

Filter parameters (B.70 phase 2, item 2.9): data/.b70/kf_params_r{r}_f{k}.json, written by b70_d1/p2_b56_fit.py
(keys KF0, KFX, clip, use_x).  If a fold's file is absent the filter rows are left out and the readout says so.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"                         # the CPU is shared with B.70

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
if __name__ == "__main__":                       # checks_b71 / kf_b71 `import run_b71` must get this module (PRECISION)
    sys.modules.setdefault("run_b71", sys.modules[__name__])
sys.path.insert(0, str(EXP / "b27_kfold"))
sys.path.insert(0, str(EXP / "b42_delay"))
import run_b42 as D42  # noqa: E402  (puts b30_thinned, b37_plan, b34_error_model on sys.path)
import run_b27 as B27  # noqa: E402
from run_b42 import R, Ctx, ces_channels, file_inputs, input_check  # noqa: E402

DATA = R.DATA
CACHE = DATA / ".b71"
B70_ROOT = DATA / ".b70_rescore"
KF_PARAMS = DATA / ".b70" / "kf_params_r{r}_f{k}.json"      # written by b70_d1/p2_b56_fit.py
CHECKS_MD = R.REPO / "docs" / "paper" / "b71_checks.md"

REPEATS = (2, 3, 4)
K = B27.K
TQ = 10_000                          # time quanta per second: comparisons in integer units of 0.1 ms
CUT_STEP_Q = 1 * TQ                  # c_j = block start + j x 1.0 s
MIN_TI_BEFORE = 3                    # >= 3 T_i samples available before the cut
MIN_AFTER_Q = TQ // 2                # the block continues >= 0.5 s after the cut
F_HORIZON_Q = 3 * TQ                 # F and R: rows with c <= time <= c + 3 s (or the block end)
GAPS_MS = (100, 300, 1000)           # G: g = 0.1, 0.3, 1.0 s
G_AFTER_Q = 1 * TQ                   # G: rows with c <= time <= c + g + 1 s
CHECK_FILES = 5
PRECISION = "float64"               # arithmetic of the replays (addendum 1, C16); --precision float32 for the check of C16


def log(t0, msg):
    print(f"[{_time.time() - t0:6.0f}s] {msg}", flush=True)


def tq_of(time):
    return np.rint(np.asarray(time, dtype=np.float64) * TQ).astype(np.int64)


def pin_env():
    """Every CES_* variable this batch depends on (run_b30.pin_env plus the D1 items it does not set)."""
    R.pin_env()
    os.environ.update({"CES_TI_SPIKE_CUT_EV": "3000", "CES_DROP_STUCK_TARGETS": "1", "CES_PER_SHOT_NORM": "0",
                       "CES_SEQ_NORM_EXPANDING": "0", "CES_SEQ_ZERO_MC": "0", "CES_SEQ_EXTRA_VT": "",
                       "CES_SEQ_EXTRA_TI": "", "CES_SEQ_DEVICE": "cpu", "CES_WINDOW_SIZE": "2",
                       "CES_MAX_VAL_SAMPLES": "0"})


# ------------------------------------------------------------------------------------------ networks
class Net:
    """A frozen network.  `precision` = the arithmetic of the forward passes (addendum 1, C16): the inputs are always
    build_blocks' float32 values; with "float64" the weights and the inputs are cast to float64 before the forward."""

    def __init__(self, kind, model, stats, per_shot, run, precision, torch):
        self.kind, self.model, self.stats, self.per_shot, self.run = kind, model, stats, per_shot, run
        self.precision = precision
        self.np_dtype = np.float64 if precision == "float64" else np.float32
        self.tdtype = torch.float64 if precision == "float64" else torch.float32
        if precision == "float64":
            self.model = model.double()
        self.mean32 = np.asarray(stats["target"]["mean"], dtype=np.float32)
        self.std32 = np.asarray(stats["target"]["std"], dtype=np.float32)

    def phys(self, z):
        """normalised (..., 2) -> physical units: eval_seq.build_pred_lookup's float32 arithmetic, or the same
        float32 statistics in float64 arithmetic for float64 outputs."""
        if self.precision == "float64":
            return np.asarray(z, dtype=np.float64) * self.std32.astype(np.float64) + self.mean32.astype(np.float64)
        return np.asarray(z, dtype=np.float32) * self.std32 + self.mean32


def fwd(T, net, x, state=None):
    """seq_v2 / seq_v2h forward (model_seq_v2.forward, model_seq_v2h.forward) with an initial LSTM state.
    x (B, L, 22) float32 tensor; state = ((h, c) of lstm_ti, (h, c) of lstm_vt) or None (zeros, as forward)."""
    m = net.model
    x_ti = x[..., :m.n_base] if net.kind == "v2" else x[..., m.n_fast:m.n_base]
    x_vt = x[..., m.n_fast:m.n_base]
    o_ti, s_ti = m.lstm_ti(x_ti, None if state is None else state[0])
    o_vt, s_vt = m.lstm_vt(x_vt, None if state is None else state[1])
    return T.cat([m.head_ti(m.norm_ti(o_ti)), m.head_vt(m.norm_vt(o_vt))], dim=-1), (s_ti, s_vt)


def u_states(T, net, x, t0s):
    """LSTM state of the U pass after the rows [0, t0) of one block, for every t0 (block-relative)."""
    out, state, prev = {}, None, 0
    xt = T.from_numpy(x).to(net.tdtype).unsqueeze(0)
    for t0 in sorted(set(int(v) for v in t0s)):
        if t0 > prev:
            _, state = fwd(T, net, xt[:, prev:t0], state)
            prev = t0
        out[t0] = state
    return out


def stack_states(T, sts):
    return tuple((T.cat([s[i][0] for s in sts], 1), T.cat([s[i][1] for s in sts], 1)) for i in range(2))


def pad(xs, width):
    H = max((len(x) for x in xs), default=0)
    X = np.zeros((len(xs), H, width), dtype=np.float32)
    for i, x in enumerate(xs):
        X[i, :len(x)] = x
    return X


def run_cont(T, net, xs, states):
    """F and G: one batched forward over the replay rows of every cut, from the U state at the row before t0.
    xs: list of (n_i, 22) float32; returns list of (n_i, 2) normalised outputs."""
    if not xs or max(len(x) for x in xs) == 0:
        return [np.zeros((len(x), 2), dtype=net.np_dtype) for x in xs]
    out, _ = fwd(T, net, T.from_numpy(pad(xs, xs[0].shape[1])).to(net.tdtype), stack_states(T, states))
    out = out.numpy()
    return [out[i, :len(x)] for i, x in enumerate(xs)]


def run_r(T, net, xs, dts, states, base, feedback=True):
    """R: row by row from the U state at the row before t0, batched over cuts.  Row 0 of a replay (t0) takes the
    inputs given (those of U and F); from row 1 on the six CES channels hold the network's own normalised estimate
    of the previous row for both targets, log1p(time step) as staleness and flag 1.  feedback=False steps the
    inputs as given (check 4)."""
    if not xs or max(len(x) for x in xs) == 0:
        return [np.zeros((len(x), 2), dtype=net.np_dtype) for x in xs]
    X = pad(xs, xs[0].shape[1])
    ldt = np.zeros(X.shape[:2], dtype=np.float32)
    for i, d in enumerate(dts):
        ldt[i, :len(d)] = np.log1p(np.asarray(d, dtype=np.float64))
    st = stack_states(T, states)
    outs = np.zeros((X.shape[0], X.shape[1], 2), dtype=net.np_dtype)
    prev = None
    for j in range(X.shape[1]):
        xj = T.from_numpy(X[:, j:j + 1].copy()).to(net.tdtype)
        if feedback and j > 0:
            lj = T.from_numpy(ldt[:, j]).to(net.tdtype)
            for t in range(2):
                xj[:, 0, base + 3 * t] = prev[:, t]
                xj[:, 0, base + 3 * t + 1] = lj
                xj[:, 0, base + 3 * t + 2] = 1.0
        o, st = fwd(T, net, xj, st)
        prev = o[:, 0]
        outs[:, j] = prev.numpy()
    return [outs[i, :len(x)] for i, x in enumerate(xs)]


def scen_x(x, tm, tz, avail, t0, end, base):
    """Inputs of the replay rows [t0, end) of one block: x of build_blocks with the six CES channels rebuilt from
    the available samples (run_b42.ces_channels, L = 0, true staleness)."""
    xs = x[t0:end].copy()
    if end > t0:
        xs[:, base:base + 6] = ces_channels(tm[:end], tz[:end], avail[:end], 0, "true")[t0:end]
    return xs


# ------------------------------------------------------------------------------------------ cuts
def block_cuts(tq, ti_obs):
    """Every c_j = start + j s with c_j <= the last row of the block; `kept` per the registration (section 3)."""
    out, start, last = [], int(tq[0]), int(tq[-1])
    j = 1
    while start + j * CUT_STEP_Q <= last:
        cq = start + j * CUT_STEP_Q
        before = np.flatnonzero(ti_obs & (tq < cq))
        reason = "lt3_ti" if len(before) < MIN_TI_BEFORE else ("tail_lt_0.5s" if last - cq < MIN_AFTER_Q else "")
        c = {"j": j, "cq": cq, "t0": int(np.searchsorted(tq, cq, "left")), "n_ti_before": int(len(before)),
             "kept": not reason, "reason": reason, "hF": int(np.searchsorted(tq, cq + F_HORIZON_Q, "right")),
             "hG": {g: int(np.searchsorted(tq, cq + g * TQ // 1000 + G_AFTER_Q, "right")) for g in GAPS_MS}}
        c["u_last"] = int(before[-1]) if len(before) else -1
        out.append(c)
        j += 1
    return out


def avail_F(obs, tq, cq):
    return obs & (tq < cq)[:, None]


def avail_G(obs, tq, cq, g_ms):
    return obs & ~((tq >= cq) & (tq < cq + g_ms * TQ // 1000))[:, None]


# ------------------------------------------------------------------------------------------ causal GP, fixed samples
def gp_causal_fixed(B, times, values, targets):
    """baselines_interpolation.predict_gp_causal(times, values, tt) for every tt in `targets`, when every sample
    lies before every target (F: the available samples do not change after the cut).  The 16 nearest past samples
    and the hyperparameters (log marginal likelihood over the fixed grid) do not depend on tt, so the fit is done
    once and the posterior mean evaluated at every target; the arithmetic is _gp_fit_predict's (check 5)."""
    targets = np.asarray(targets, dtype=np.float64)
    if len(targets) == 0:
        return np.zeros(0)
    if not np.all(times[None, :] < targets[:, None]):
        raise SystemExit("FATAL: gp_causal_fixed needs every sample before every target")
    hold = float(values[int(np.argmax(times))])          # predict_persistence
    if not B._HAVE_GP:
        return np.full(len(targets), np.nan)
    keep = np.argsort(targets[0] - times)[:B._GP_MAX_SIDE]
    t, v = times[keep], values[keep]
    if len(t) < 2:
        return np.full(len(targets), hold)
    mu, sd = float(np.mean(v)), float(np.std(v))
    if sd <= 0.0:
        return np.full(len(targets), mu)
    y = (v - mu) / sd
    n = len(t)
    dist = np.abs(t[:, None] - t[None, :])
    dist_star = np.abs(t[None, :] - targets[:, None])
    eye = np.eye(n)
    best_lml, best = -np.inf, None
    for ls in B._GP_LS_GRID:
        k0 = B._matern32(dist, ls)
        for noise in B._GP_NOISE_GRID:
            try:
                c, low = cho_factor(k0 + (noise + B._GP_JITTER) * eye, lower=True)
            except np.linalg.LinAlgError:  # pragma: no cover
                continue
            alpha = cho_solve((c, low), y)
            lml = (-0.5 * float(y @ alpha) - float(np.sum(np.log(np.diag(c)))) - 0.5 * n * np.log(2.0 * np.pi))
            if lml > best_lml:
                best_lml, best = lml, (ls, alpha)
    if best is None:
        return np.full(len(targets), hold)
    pred = np.array([float(k @ best[1]) for k in B._matern32(dist_star, best[0])])
    out = mu + sd * pred
    return np.where(np.isfinite(pred), out, hold)


# ------------------------------------------------------------------------------------------ context
class B71:
    def __init__(self, precision=None):
        t0 = _time.time()
        self.precision = precision or PRECISION
        self.C = Ctx()                                   # run_b30.pin_env + the D1 grid (cut 3000, held removed)
        pin_env()
        self.T = self.C.torch
        self.T.set_num_threads(2)
        self.base = self.C.dims["n_fast"] + 1
        grid = self.C.grid
        all_csv = sorted(p.name for p in DATA.glob("*.csv"))
        if set(all_csv) != {p.name for p in DATA.glob("s*.csv")}:
            raise SystemExit("FATAL: data/ holds CSVs that are not shot files; the dataset order cannot be rebuilt")
        # the order and rows of evaluate.build_clean_val_subset's dataset (dataset.py _preload_data): sorted CSVs,
        # files with >= 1 target after the D1 treatment, rows with >= 1 target
        self.lab_rows = {nm: np.flatnonzero(np.isfinite(grid[nm][:, 1:3]).any(1)) for nm in grid}
        self.ds_names = [nm for nm in all_csv if nm in grid and len(self.lab_rows[nm])]
        self.idx = {nm: i for i, nm in enumerate(self.C.names)}
        log(t0, f"grid: {len(grid)} files, dataset order {len(self.ds_names)} files")

    def manifest(self, r, k):
        return json.loads((B27.split_dir(r, k) / "split_manifest.json").read_text(encoding="utf-8"))

    def test_files(self, r, k, limit=None):
        man = self.manifest(r, k)
        return sorted(nm for nm in man["test_files"] if nm in self.C.grid)[:limit]

    def net(self, kind, r, k):
        run = DATA / f".b39p_{kind}_cut_r{r}_f{k}"
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        variant, per_shot = ("v2", False) if kind == "v2" else ("v2h", True)
        if (m["seq_model"] != variant or bool(m["per_shot_input_norm"]) != per_shot or m.get("per_shot_norm_expanding")
                or m.get("zero_mc") or m.get("extra_vt_channels") or m.get("extra_ti_channels") or m.get("train_ctx")
                or not m.get("drop_stuck_targets") or int(m["init_seed"]) != B27.init_seed(r, k)
                or int(m["seed"]) != B27.REPEAT_SEEDS[r]):
            raise SystemExit(f"FATAL: {run.name} is not the registered {kind} network of r{r} f{k}")
        stats = self.C.load_stats(m)
        model = self.C.MODELS[variant]()
        model.load_state_dict(self.T.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
        model.eval()
        if model.n_extra_vt or model.n_extra_ti or model.n_fast != self.C.dims["n_fast"]:
            raise SystemExit(f"FATAL: {run.name}: unexpected channel layout")
        return Net(kind, model, stats, per_shot, run.name, self.precision, self.T)

    def nets(self, r, k):
        return {"net": self.net("v2", r, k), "twin": self.net("v2h", r, k)}

    def d1_rows(self, r, k):
        """The D1 T_i rows of the fold (B.70 phase 1 re-scoring of the network) as grid rows per file, with the
        stored y_true and squared error (check 2)."""
        p = B70_ROOT / f"b39p_v2_cut_r{r}_f{k}" / "comparison_errors_test.npz"
        if not p.exists():
            raise SystemExit(f"FATAL: missing B.70 phase-1 output {p}; B.71 runs after B.70 phase 1")
        z = np.load(p)
        shot, row = z["CES_TI_shot"], z["CES_TI_row"]
        yt, se = z["CES_TI_y_true"].astype(np.float64), z["CES_TI_se_model"].astype(np.float64)
        out, max_dev = {}, 0.0
        for s in np.unique(shot):
            nm = self.ds_names[int(s)]
            m = shot == s
            g = self.lab_rows[nm][row[m]]
            dev = np.abs(self.C.grid[nm][g, 1].astype(np.float64) - yt[m])
            if not np.all(dev <= 0.01):
                raise SystemExit(f"FATAL: r{r} f{k}: D1 rows of {nm} do not map onto the grid (max dev {np.nanmax(dev)})")
            max_dev = max(max_dev, float(dev.max()))
            o = np.argsort(g)
            if len(np.unique(g)) != len(g):
                raise SystemExit(f"FATAL: r{r} f{k}: duplicate D1 rows in {nm}")
            out[nm] = {"rows": g[o], "y_true": yt[m][o], "se": se[m][o]}
        return out, {"rows": int(len(shot)), "files": len(out), "max_y_dev_eV": max_dev}

    def train_range(self, r, k):
        """T_i staleness input (true staleness, ms) on the fold's training discharges: every grid row whose T_i flag
        is 1, and the rows with T_i measured."""
        C = self.C
        allv, meas = [], []
        for nm in self.manifest(r, k)["train_files"]:
            if nm not in C.grid:
                continue
            arr = C.grid[nm]
            tm, ti = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
            for a, b in R.block_spans(tm, C.GAP):
                o = np.flatnonzero(np.isfinite(ti[a:b]))
                p = np.searchsorted(o, np.arange(b - a), "left") - 1
                has = p >= 0
                st = (tm[a:b][has] - tm[a:b][o[p[has]]]) * 1000.0
                allv.append(st)
                meas.append(st[np.isfinite(ti[a:b][has])])

        def q(v):
            v = np.concatenate(v)
            return {"n": int(len(v)), "p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99)),
                    "p99.9": float(np.percentile(v, 99.9)), "max": float(v.max())}
        return {"flag_rows": q(allv), "measured_rows": q(meas)}

    # -------------------------------------------------------------------------------------- one file
    def file_prep(self, nm, nets):
        arr = self.C.grid[nm]
        P = {}
        for key, net in nets.items():
            blks, spans, tm, tgt, tz = file_inputs(self.C, arr, (net.model, net.stats), net.per_shot)
            P[key] = {"blks": blks, "tz": tz}
        tq = tq_of(tm)
        cuts = []
        for bi, (a, b) in enumerate(spans):
            for c in block_cuts(tq[a:b], ~np.isnan(tgt[a:b, 0])):
                c.update(block=bi, a=a, b=b)
                cuts.append(c)
        return {"P": P, "spans": spans, "tm": tm, "tq": tq, "tgt": tgt, "obs": ~np.isnan(tgt), "cuts": cuts}

    def u_pass(self, net, blks, spans, n):
        """U: one forward per block (eval_seq.build_pred_lookup); normalised (n, 2), NaN outside blocks."""
        U = np.full((n, 2), np.nan, dtype=net.np_dtype)
        with self.T.no_grad():
            for blk, (a, b) in zip(blks, spans):
                U[a:b] = net.model(self.T.from_numpy(blk["x"]).to(net.tdtype).unsqueeze(0))[0].numpy()
        return U

    def replay(self, F, net, key, cuts, scen=("F", "R", "G")):
        """Normalised outputs of the replay rows [t0, horizon) of every cut: {"F": [...], "R": [...], "G": {g: [...]}}."""
        T, base = self.T, self.base
        blks, tz = F["P"][key]["blks"], F["P"][key]["tz"]
        xsF, dtsR, sts, xsG = [], [], [], {g: [] for g in GAPS_MS}
        with T.no_grad():
            for bi, (blk, (a, b)) in enumerate(zip(blks, F["spans"])):
                kc = [c for c in cuts if c["block"] == bi]
                if not kc:
                    continue
                x = blk["x"]
                st = u_states(T, net, x, [c["t0"] for c in kc])
                tm, tzl, obs, tq = F["tm"][a:b], tz[a:b], F["obs"][a:b], F["tq"][a:b]
                for c in kc:
                    t0 = c["t0"]
                    sts.append(st[t0])
                    xsF.append(scen_x(x, tm, tzl, avail_F(obs, tq, c["cq"]), t0, c["hF"], base))
                    dtsR.append(np.diff(tm[t0 - 1:c["hF"]]) if c["hF"] > t0 else np.zeros(0))
                    for g in GAPS_MS:
                        xsG[g].append(scen_x(x, tm, tzl, avail_G(obs, tq, c["cq"], g), t0, c["hG"][g], base))
            out = {}
            if "F" in scen:
                out["F"] = run_cont(T, net, xsF, sts)
            if "R" in scen:
                out["R"] = run_r(T, net, xsF, dtsR, sts, base)
            if "G" in scen:
                out["G"] = {g: run_cont(T, net, xsG[g], sts) for g in GAPS_MS}
        return out

    def file_units(self, nm, nets, d1):
        """Every (scored row, cut) unit of one test file with the T_i estimate of every estimator (eV)."""
        F = self.file_prep(nm, nets)
        n = len(F["tm"])
        kept = [c for c in F["cuts"] if c["kept"]]
        is_d1 = np.zeros(n, dtype=bool)
        if nm in d1:
            is_d1[d1[nm]["rows"]] = True
        y = F["tgt"][:, 0]
        U, RP = {}, {}
        for key, net in nets.items():
            U[key] = net.phys(self.u_pass(net, F["P"][key]["blks"], F["spans"], n))[:, 0]
            RP[key] = self.replay(F, net, key, kept, ("F", "R", "G") if key == "net" else ("F", "R"))
        fu = {c: [] for c in ("cut", "row", "e_q", "y", "hold_F", "gp_F", *(f"{k}_{s}" for k in nets for s in "UFR"))}
        gu = {c: [] for c in ("cut", "row", "g_ms", "s_q", "y", "net_U", "net_G")}
        for ci, c in enumerate(kept):
            a, t0 = c["a"], c["t0"]
            rows = np.arange(a + t0, a + c["hF"])
            sel = is_d1[rows]
            r_ = rows[sel]
            uq = F["tq"][a + c["u_last"]]
            o = np.flatnonzero(F["obs"][a:a + t0, 0] & (F["tq"][a:a + t0] < c["cq"])) + a
            fu["cut"].append(np.full(len(r_), ci)); fu["row"].append(r_)
            fu["e_q"].append(F["tq"][r_] - uq); fu["y"].append(y[r_].astype(np.float64))
            fu["hold_F"].append(np.full(len(r_), float(y[o[-1]])))
            fu["gp_F"].append(gp_causal_fixed(self.C.B, F["tm"][o], y[o].astype(np.float64), F["tm"][r_]))
            for key, net in nets.items():
                fu[f"{key}_U"].append(U[key][r_].astype(np.float64))
                for s in "FR":
                    fu[f"{key}_{s}"].append(net.phys(RP[key][s][ci])[sel, 0].astype(np.float64))
            for g in GAPS_MS:
                rows = np.arange(a + t0, a + c["hG"][g])
                s_q = F["tq"][rows] - (c["cq"] + g * TQ // 1000)
                sel = is_d1[rows] & (s_q > 0)
                r_ = rows[sel]
                gu["cut"].append(np.full(len(r_), ci)); gu["row"].append(r_); gu["g_ms"].append(np.full(len(r_), g))
                gu["s_q"].append(s_q[sel]); gu["y"].append(y[r_].astype(np.float64))
                gu["net_U"].append(U["net"][r_].astype(np.float64))
                gu["net_G"].append(nets["net"].phys(RP["net"]["G"][g][ci])[sel, 0].astype(np.float64))
        ints = ("cut", "row", "e_q", "s_q", "g_ms")
        cat = lambda d: {k: (np.concatenate(v) if v else np.zeros(0)).astype(np.int64 if k in ints else np.float64)  # noqa: E731
                         for k, v in d.items()}
        return cat(fu), cat(gu), F["cuts"], U

    # -------------------------------------------------------------------------------------- one fold
    def rows_fold(self, r, k, limit=None, t_start=None, check2=True):
        t0 = t_start or _time.time()
        nets = self.nets(r, k)
        test = self.test_files(r, k, limit)
        d1, d1info = self.d1_rows(r, k)
        FU, GU, CUTS = [], [], []
        c1 = {"files": 0, "failed": []}
        c2 = {"rows": 0, "within": 0, "max_dev_eV": 0.0, "missing_rows": 0}
        ncut = 0
        for fi, nm in enumerate(test):
            arr = self.C.grid[nm]
            for key, net in nets.items():              # check 1 (registration 6.1) on every test file
                blks, spans, tm, tgt, tz = file_inputs(self.C, arr, (net.model, net.stats), net.per_shot)
                if not input_check(self.C, blks, spans, tm, tgt, tz):
                    c1["failed"].append(f"{key} {nm}")
            c1["files"] += 1
            fu, gu, cuts, U = self.file_units(nm, nets, d1)
            if check2 and nm in d1:                    # check 2 (registration 6.2): U vs the B.70 D1 re-scoring
                rows, yt, se = d1[nm]["rows"], d1[nm]["y_true"], d1[nm]["se"]
                p = U["net"][rows]
                dev = np.abs(np.abs(p.astype(np.float64) - yt) - np.sqrt(se))
                c2["rows"] += len(rows)
                c2["missing_rows"] += int((~np.isfinite(p)).sum())
                c2["within"] += int((dev <= 0.5).sum())
                c2["max_dev_eV"] = max(c2["max_dev_eV"], float(np.nanmax(dev)) if len(dev) else 0.0)
            fid = self.idx[nm]
            fu["file"] = np.full(len(fu["row"]), fid); gu["file"] = np.full(len(gu["row"]), fid)
            fu["cut"] = fu["cut"] + ncut; gu["cut"] = gu["cut"] + ncut
            kept = [c for c in cuts if c["kept"]]
            ncut += len(kept)
            FU.append(fu); GU.append(gu)
            for c in cuts:
                CUTS.append((fid, c["block"], c["a"], c["b"], c["j"], c["cq"], c["t0"], c["n_ti_before"],
                             int(c["kept"]), {"": 0, "lt3_ti": 1, "tail_lt_0.5s": 2}[c["reason"]], c["hF"]))
            if (fi + 1) % 10 == 0 or fi + 1 == len(test):
                log(t0, f"r{r} f{k}: {fi + 1}/{len(test)} files, {ncut} kept cuts")
        if c1["failed"]:
            raise SystemExit(f"FATAL: check 1 failed (rebuilt inputs differ from build_blocks): {c1['failed'][:5]}")
        c2["share_within_0.5_eV"] = c2["within"] / max(c2["rows"], 1)
        c2["passes"] = bool(c2["rows"] > 0 and c2["missing_rows"] == 0 and c2["share_within_0.5_eV"] >= 0.999)
        out = {f"F__{c}": np.concatenate([u[c] for u in FU]) for c in FU[0]}
        out.update({f"G__{c}": np.concatenate([u[c] for u in GU]) for c in GU[0]})
        out["cuts"] = np.array(CUTS, dtype=np.int64).reshape(-1, 11)
        out["cut_cols"] = np.array(["file", "block", "a", "b", "j", "cq", "t0", "n_ti_before", "kept", "reason", "hF"])
        tset = set(test)
        out["d1_files"] = np.array(sorted(self.idx[nm] for nm in d1 if nm in tset), dtype=np.int64)
        out["d1_rows_per_file"] = np.array([len(d1[self.C.names[i]]["rows"]) for i in out["d1_files"]], dtype=np.int64)
        meta = {"repeat": r, "fold": k, "test_files": len(test), "limit": limit, "nets": {kk: v.run for kk, v in nets.items()},
                "d1": d1info, "check1": c1, "check2": c2, "kept_cuts": ncut, "train_range": self.train_range(r, k),
                "precision": self.precision}
        out["meta"] = np.array(json.dumps(meta))
        return out, meta


def fold_list(args):
    if args:
        return [(int(f[1:].split("f")[0]), int(f.split("f")[1])) for f in args]
    return [(r, k) for r in REPEATS for k in range(K)]


def main():
    global PRECISION
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "timing", "rows", "kf", "analyze"))
    ap.add_argument("--fold", default="r2f0")
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--files", type=int, default=CHECK_FILES)
    ap.add_argument("--params", default=str(KF_PARAMS))
    ap.add_argument("--out", default=None, help="check: where to keep the check rows (default: not kept)")
    ap.add_argument("--precision", choices=("float64", "float32"), default=PRECISION)
    args = ap.parse_args()
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    PRECISION = args.precision
    if args.cmd == "analyze":
        import analyze_b71
        analyze_b71.main()
        return
    if args.cmd in ("check", "timing"):
        import checks_b71
        (checks_b71.run_checks if args.cmd == "check" else checks_b71.run_timing)(args)
        return
    if args.cmd == "kf":
        import kf_b71
        kf_b71.main(args.params, fold_list(args.folds))
        return
    X = B71()
    CACHE.mkdir(exist_ok=True)
    t0 = _time.time()
    import checks_b71
    for r, k in fold_list(args.folds):
        p = CACHE / f"rows_r{r}_f{k}.npz"
        if p.exists():
            log(t0, f"r{r} f{k}: {p.name} exists, skipped")
            continue
        tf = _time.time()
        out, meta = X.rows_fold(r, k, t_start=t0)
        if not meta["check2"]["passes"]:
            raise SystemExit(f"FATAL: check 2 failed on r{r} f{k}: {meta['check2']}")
        c34 = checks_b71.fold_checks(X, r, k, CHECK_FILES)      # checks 3, 4, 4b, 5 on the first five test files
        if not c34["passes"]:
            raise SystemExit(f"FATAL: checks 3-5 failed on r{r} f{k}: {json.dumps(c34)[:2000]}")
        meta["checks_3_to_5"] = c34
        meta["seconds"] = round(_time.time() - tf, 1)
        out["meta"] = np.array(json.dumps(meta))
        tmp = CACHE / f"rows_r{r}_f{k}.partial.npz"         # written whole, then renamed: an interrupted save
        np.savez_compressed(tmp, **out)                      # never leaves a file that the resume would skip
        os.replace(tmp, p)
        log(t0, f"r{r} f{k}: {len(out['F__row'])} F units, {len(out['G__row'])} G units, {meta['kept_cuts']} cuts -> {p.name}")


if __name__ == "__main__":
    main()
    os._exit(0)
