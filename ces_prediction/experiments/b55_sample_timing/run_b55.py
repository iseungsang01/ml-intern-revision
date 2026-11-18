"""B.55 (PREREGISTRATION_B55.md): the instant of the fast-channel sample.

The 40 estimator networks of the pooled cross-validation of B.39 (`data/.b39p_v2_cut_r{r}_f{k}`, causal variant,
statistics of the training discharges) are run forward on the CPU over the discharges of their own test fold with
changed fast-channel inputs.  Nothing is trained; no shared code is edited.

  Part R  twelve discharges with raw streams: inside the raw window, BES, ECEI and Mirnov of every grid row are
          replaced by the raw samples taken d before the sample the file holds (d = one raw sample, 0.1, 0.5, 1, 2,
          5, 10 ms), or by the mean of the raw samples in the 0.1 ms / 1 ms up to the stamp.
  Part G  every discharge: the fast channels of a row are those of the previous row of its block.

Stages (repo root; each is resumable):
    py ces_prediction/experiments/b55_sample_timing/run_b55.py raw       # raw streams -> data/.b55/raw_s{shot}.npz, checks (b), (c)
    py ces_prediction/experiments/b55_sample_timing/run_b55.py nets      # 40 networks forward -> data/.b55/run_r{r}_f{k}.npz
    py ces_prediction/experiments/b55_sample_timing/run_b55.py checks    # checks (a)-(d) -> data/.b55/checks.json
    py ces_prediction/experiments/b55_sample_timing/run_b55.py readout   # parts R and G -> docs/paper/b55_sample_timing.{json,md}
    py ces_prediction/experiments/b55_sample_timing/run_b55.py all
    py ces_prediction/experiments/b55_sample_timing/run_b55.py render    # .md from the stored .json
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the GPU belongs to the running training batch

import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
CES_DIR = REPO / "ces_prediction"
SEQ_DIR = EXP / "seq"
DATA = REPO / "data"
sys.path.insert(0, str(EXP / "b39_primary"))
sys.path.insert(0, str(CES_DIR))
import run_b39 as B39  # noqa: E402
import analyze_file_rules as AFR  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_B55.md"
CACHE = DATA / ".b55"
OUT_JSON = REPO / "docs" / "paper" / "b55_sample_timing.json"
OUT_MD = REPO / "docs" / "paper" / "b55_sample_timing.md"
B39_JSON = REPO / "docs" / "paper" / "b39_primary.json"
FILE_RULES_JSON = REPO / "docs" / "paper" / "file_rules.json"
T = "CES_TI"
RAW_SHOTS = AFR.RAW_SHOTS
DIAGS = ("BES", "ECEI", "MC")
REPEATS = B39.REPEATS
K = B39.B27.K
PRINT_TOL = AFR.PRINT_TOL
# variants of part R: (name, kind, value); kind "back" = samples d before the held one, "mean" = mean over a window
VARIANTS = [("1samp", "back", None), ("0.1ms", "back", 0.1e-3), ("0.5ms", "back", 0.5e-3), ("1ms", "back", 1e-3),
            ("2ms", "back", 2e-3), ("5ms", "back", 5e-3), ("10ms", "back", 10e-3),
            ("mean0.1ms", "mean", 0.1e-3), ("mean1ms", "mean", 1e-3)]
VNAMES = [v[0] for v in VARIANTS]
LABEL = {"1samp": "one raw sample earlier", "0.1ms": "0.1 ms earlier", "0.5ms": "0.5 ms earlier", "1ms": "1 ms earlier",
         "2ms": "2 ms earlier", "5ms": "5 ms earlier", "10ms": "10 ms earlier",
         "mean0.1ms": "mean of the samples in the 0.1 ms up to the stamp",
         "mean1ms": "mean of the samples in the 1 ms up to the stamp"}
MARGIN = 0.01
N_BOOT_R, N_BOOT_G = 2000, 20_000
SEED_R, SEED_G = 20261118, 20261119
SEED_B39P = B39_SEED_POOLED = 2728          # analyze_b39.SEED_POOLED: check (d) replays its generator
CHECK_A_EV, CHECK_A_SHARE = 0.5, 0.99          # addendum 1 (registered 0.05 eV max replaced by the B.49 rule)
CHECK_D_TOL = 0.0005
CHECK_C_SLACK = 0.0005
N_FAST = 15
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


# ------------------------------------------------------------------------------------------ environment
def run_dirs(r, k):
    return B39.pooled_run_dir("v2", r, k), B39.pooled_run_dir("v2h", r, k)


def treatment():
    """The data treatment of the pooled estimator runs, read from the runner (identical for all 40)."""
    keys = ("CES_DROP_STUCK_TARGETS", "CES_TI_SPIKE_CUT_EV", "CES_PER_SHOT_NORM", "CES_SEQ_NORM_EXPANDING",
            "CES_SEQ_ZERO_MC", "CES_SEQ_MODEL", "CES_SEQ_TRAIN_CTX", "CES_SEQ_EVAL_CTX")
    seen = None
    for r in REPEATS:
        for k in range(K):
            env = B39.pooled_env("v2", r, k, run_dirs(r, k)[0])
            tr = {x: env.get(x) for x in keys}
            for x in ("CES_SEQ_EXTRA_VT", "CES_SEQ_EXTRA_TI", "CES_SEQ_ENSEMBLE_DIRS"):
                if env.get(x, "").strip():
                    raise SystemExit(f"FATAL: r{r} f{k}: {x} is set in the runner environment")
            if seen is None:
                seen = tr
            elif tr != seen:
                raise SystemExit(f"FATAL: r{r} f{k}: treatment {tr} differs from {seen}")
    if seen["CES_PER_SHOT_NORM"] != "0" or seen["CES_SEQ_NORM_EXPANDING"] != "0" or seen["CES_SEQ_ZERO_MC"] != "0" \
            or seen["CES_SEQ_MODEL"] != "v2" or seen["CES_SEQ_TRAIN_CTX"] != "0" or seen["CES_SEQ_EVAL_CTX"] != "0":
        raise SystemExit(f"FATAL: unexpected treatment {seen}")
    return seen


def pin_env(tr):
    for k in [k for k in os.environ if k.startswith("CES_")]:
        del os.environ[k]
    os.environ.update({"CES_DATA_DIR": str(DATA), **{k: v for k, v in tr.items() if v is not None}})


class Nets:
    def __init__(self):
        self.tr = treatment()
        pin_env(self.tr)
        sys.path.insert(0, str(SEQ_DIR))
        import torch
        from evaluate import _load_stats
        from seq_data import load_grid_files, build_blocks, infer_columns
        from seq_models import SEQ_MODELS
        torch.set_num_threads(4)
        self.torch, self.load_stats, self.build_blocks, self.MODELS = torch, _load_stats, build_blocks, SEQ_MODELS
        self.grid, self.dims = load_grid_files(DATA, self.tr["CES_DROP_STUCK_TARGETS"] == "1",
                                               float(self.tr["CES_TI_SPIKE_CUT_EV"]))
        if self.dims["n_fast"] != N_FAST or self.dims["extra_vt"] or self.dims["extra_ti"]:
            raise SystemExit(f"FATAL: unexpected dims {self.dims}")
        self.names = sorted(self.grid)
        self.valid = [nm for nm in self.names if np.isfinite(self.grid[nm][:, 1:3]).any()]
        bes, ecei, mc = infer_columns(sorted(DATA.glob("s*.csv"))[0])
        self.fast_cols = [*bes, *ecei, *mc]

    def load(self, run, r, k):
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        env = B39.pooled_env("v2", r, k, run)
        if m["seq_model"] != "v2" or bool(m["per_shot_input_norm"]) or m.get("per_shot_norm_expanding") \
                or m.get("zero_mc") or m.get("extra_vt_channels") or m.get("extra_ti_channels") or m.get("train_ctx") \
                or str(m["seed"]) != env["CES_SEED"] or str(m["init_seed"]) != env["CES_INIT_SEED"]:
            raise SystemExit(f"FATAL: {run.name} is not the registered causal variant of the runner")
        stats = self.load_stats(m)
        model = self.MODELS["v2"]()
        model.load_state_dict(self.torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
        model.eval()
        return model, stats, env

    def predict(self, arr, net, shift=False):
        """T_i estimate (eV) at every grid row of one file (NaN in blocks build_blocks skips)."""
        model, stats, _ = net
        tm = arr[:, 0]
        out = np.full(len(tm), np.nan)
        mean, std = float(stats["target"]["mean"][0]), float(stats["target"]["std"][0])
        a = 0
        spans = block_spans(tm.astype(np.float64))
        blks = self.build_blocks(arr, self.dims, stats, per_shot_norm=False, expanding=False)
        if len(blks) != len(spans):
            raise SystemExit("FATAL: block spans do not match build_blocks")
        with self.torch.no_grad():
            for blk, (a, b) in zip(blks, spans):
                if not np.array_equal(blk["time"], arr[a:b, 0]):
                    raise SystemExit("FATAL: block times do not match the grid")
                x = blk["x"]
                if shift:                       # part G: fast channels of the previous row of the block
                    x = x.copy()
                    x[1:, :N_FAST] = blk["x"][:-1, :N_FAST]
                out[a:b] = model(self.torch.from_numpy(x).unsqueeze(0))[0][:, 0].numpy() * std + mean
        return out


def block_spans(time):
    from dataset import STUCK_GAP_SECONDS
    bounds = [0]
    for i in range(1, len(time)):
        d = time[i] - time[i - 1]
        if d >= STUCK_GAP_SECONDS or d <= 0:
            bounds.append(i)
    bounds.append(len(time))
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]


def compact_index(arr):
    """Grid rows that dataset.py keeps (at least one target finite); CES_TI_row indexes these (B.45, B.49)."""
    return np.flatnonzero(np.isfinite(arr[:, 1:3]).any(1))


def load_npz(run):
    z = np.load(run / "comparison_errors_test.npz")
    d = {"shot": z[f"{T}_shot"].astype(np.int64), "row": z[f"{T}_row"].astype(np.int64),
         "se": z[f"{T}_se_model"].astype(np.float64), "y": z[f"{T}_y_true"].astype(np.float64),
         "gp_causal": z[f"{T}_se_gp_causal"].astype(np.float64)}
    for q in ("persistence", "pchip", "gp"):
        d[q] = z[f"{T}_se_{q}"].astype(np.float64)
    return d


# ------------------------------------------------------------------------------------------ raw stage
def held_index(x, ci_, idx, v):
    """Index of the raw sample the file holds: offset 0, else -1, else +1 (all channels within the print
    tolerance); rows without a match keep the nearest sample.  Returns (held, offset used or 99 = no match)."""
    held = idx.copy()
    off = np.full(len(idx), 99, dtype=np.int64)
    for s in (0, -1, 1):
        hit = np.abs(x[idx + s][:, ci_].astype(np.float64) - v).max(1) <= PRINT_TOL
        new = hit & (off == 99)
        held[new] = idx[new] + s
        off[new] = s
    return held, off


def stage_raw():
    CACHE.mkdir(parents=True, exist_ok=True)
    fr = json.loads(FILE_RULES_JSON.read_text(encoding="utf-8"))["raw_match"]
    for shot in RAW_SHOTS:
        out = CACHE / f"raw_s{shot}.npz"
        if out.exists():
            continue
        t_s = _time.time()
        csv = DATA / f"s{shot}.csv"
        full = pd.read_csv(csv)
        tfull = full["time"].to_numpy(np.float64)
        n = len(tfull)
        df, _f, _l, _t = AFR.load(csv)          # check (b) on the rows of the file rules
        t = df["time"].to_numpy(float)
        res = {"time": tfull, "file": full[[c for c in full.columns if c.startswith(("BES_", "ECEI_", "MC"))]].to_numpy(np.float64)}
        fast_cols = [c for c in full.columns if c.startswith(("BES_", "ECEI_", "MC"))]
        if len(fast_cols) != N_FAST:
            raise SystemExit(f"FATAL: s{shot}: {len(fast_cols)} fast columns")
        vals = {v: np.full((n, N_FAST), np.nan) for v in VNAMES}
        ok = np.ones(n, dtype=bool)
        win = []
        chk = {"b": {}, "c": {}, "held_offsets": {}, "scatter": {}}
        for diag in DIAGS:
            x, fs, t0, chans = AFR.raw_npz(shot, diag)
            cols = [c for c in fast_cols if c.startswith(diag if diag != "BES" else "BES_")]
            if diag == "MC":
                cols = [c for c in fast_cols if c.startswith("MC")]
            if any(c not in chans for c in cols):
                raise SystemExit(f"FATAL: s{shot} {diag}: raw channels {chans} lack {cols}")
            ci_ = [chans.index(c) for c in cols]
            pos = [fast_cols.index(c) for c in cols]
            N = x.shape[0]
            win.append((t0, t0 + (N - 1) / fs))
            # ---- check (b): exactly the rule of analyze_file_rules.raw_side
            v = df[cols].to_numpy(float)
            idx = np.rint((t - t0) * fs).astype(np.int64)
            okr = (idx >= 1) & (idx < N - 1)
            hit = np.stack([np.abs(x[idx[okr] + s][:, ci_].astype(np.float64) - v[okr]).max(1) <= PRINT_TOL
                            for s in (-1, 0, 1)], 1)
            rec = fr[str(shot)][diag]
            mine = {"rows": int(okr.sum()), "share_equal_within_one_sample": float(hit.any(1).mean()),
                    "share_at_offset_-1_0_+1": [float(h) for h in hit.mean(0)]}
            chk["b"][diag] = {"mine": mine, "recorded": {k_: rec[k_] for k_ in mine},
                              "nearest_sample_share": float(hit[:, 1].mean()),
                              "equal": bool(mine["rows"] == rec["rows"]
                                            and mine["share_equal_within_one_sample"] == rec["share_equal_within_one_sample"]
                                            and mine["share_at_offset_-1_0_+1"] == rec["share_at_offset_-1_0_+1"])}
            # ---- check (c): d = 10 ms, rows whose previous row is 10 ms earlier, against the previous row's file value
            n10 = int(round(10e-3 * fs))
            prev10 = np.zeros(len(t), dtype=bool)
            prev10[1:] = np.abs(np.diff(t) - 0.010) < 1e-4
            sel = okr & prev10 & (idx - n10 - 1 >= 0)
            iv = np.flatnonzero(sel)
            vprev = v[iv - 1]
            hc = np.stack([np.abs(x[idx[iv] - n10 + s][:, ci_].astype(np.float64) - vprev).max(1) <= PRINT_TOL
                           for s in (-1, 0, 1)], 1)
            hcur, _o = held_index(x, ci_, idx[iv], v[iv])
            strict = np.abs(x[hcur - n10][:, ci_].astype(np.float64) - vprev).max(1) <= PRINT_TOL
            chk["c"][diag] = {"rows": int(len(iv)), "share_equal_within_one_sample": float(hc.any(1).mean()),
                              "share_strict_held_minus_10ms": float(strict.mean()),
                              "recorded_share_b": rec["share_equal_within_one_sample"]}
            # ---- the variants on every grid row of the full file
            idx_f = np.rint((tfull - t0) * fs).astype(np.int64)
            nback = {nm: (1 if val is None else int(round(val * fs))) for nm, kind, val in VARIANTS}
            need = max(max(nback.values()), int(round(1e-3 * fs))) + 1
            inside = (idx_f - need >= 0) & (idx_f + 1 < N)
            ok &= inside
            ii = np.flatnonzero(inside)
            held, off = held_index(x, ci_, idx_f[ii], res["file"][ii][:, pos])
            chk["held_offsets"][diag] = {str(o): int((off == o).sum()) for o in (-1, 0, 1, 99)}
            for nm, kind, val in VARIANTS:
                if kind == "back":
                    vals[nm][np.ix_(ii, pos)] = x[held - nback[nm]][:, ci_].astype(np.float64)
                else:
                    w = int(round(val * fs))
                    cs = np.zeros((len(ii), len(ci_)))
                    for j in range(w):                       # samples held - w + 1 .. held
                        cs += x[held - j][:, ci_].astype(np.float64)
                    vals[nm][np.ix_(ii, pos)] = cs / w
            # sample-to-sample scatter of the raw stream at the held sample (raw units, per channel)
            d1 = x[held][:, ci_].astype(np.float64) - x[held - 1][:, ci_].astype(np.float64)
            chk["scatter"][diag] = {"channels": cols, "rms_one_sample_difference": [float(a) for a in np.sqrt((d1 ** 2).mean(0))],
                                    "fs_hz": fs, "samples_back": nback}
            del x
        w0, w1 = max(a for a, _ in win), min(b for _, b in win)
        ok &= (tfull >= w0) & (tfull <= w1)
        for nm in VNAMES:
            vals[nm][~ok] = np.nan
        chk["window_s"] = [w0, w1]
        chk["windows_by_diag_s"] = {d: list(w) for d, w in zip(DIAGS, win)}
        chk["rows_replaced"] = int(ok.sum())
        np.savez(out, time=tfull, filev=res["file"], ok=ok, **{f"v_{nm}": vals[nm] for nm in VNAMES})
        (CACHE / f"raw_s{shot}.json").write_text(json.dumps(chk, indent=1), encoding="utf-8")
        log(f"raw s{shot}: {int(ok.sum())} rows replaceable, window {w0:.4f}-{w1:.4f} s "
            f"({_time.time() - t_s:.0f} s); (b) " + ", ".join(f"{d} {chk['b'][d]['equal']}" for d in DIAGS))


# ------------------------------------------------------------------------------------------ nets stage
def absent_rows(tm):
    """Grid rows 1 to 3 after an absent frame (step of 20-40 ms); rows 2 and 3 follow in 10 ms steps."""
    dt = np.diff(tm)
    out = np.zeros(len(tm), dtype=np.int8)
    for j in np.flatnonzero((dt > 0.015) & (dt < AFR.ABSENT_MAX_S)):
        a = j + 1
        for kk in range(3):
            i = a + kk
            if i >= len(tm) or abs(tm[i] - tm[a] - 0.010 * kk) > 1e-4:
                break
            if out[i] == 0 or out[i] > kk + 1:
                out[i] = kk + 1
    return out


def stage_nets():
    N = Nets()
    log(f"grid loaded: {len(N.names)} files, {len(N.valid)} with a target; treatment {N.tr}")
    raw = {}
    for shot in RAW_SHOTS:
        z = np.load(CACHE / f"raw_s{shot}.npz")
        raw[f"s{shot}.csv"] = {k_: z[k_] for k_ in z.files}
        arr = N.grid[f"s{shot}.csv"]
        if len(arr) != len(z["time"]) or np.max(np.abs(arr[:, 0].astype(np.float64) - z["time"])) > 1e-5 \
                or np.max(np.abs(arr[:, 3:3 + N_FAST].astype(np.float64) - z["filev"])) > 1e-5:
            raise SystemExit(f"FATAL: s{shot}: raw cache rows do not match the grid")
        if list(N.fast_cols) != [c for c in pd.read_csv(DATA / f"s{shot}.csv", nrows=0).columns
                                 if c.startswith(("BES_", "ECEI_", "MC"))]:
            raise SystemExit(f"FATAL: s{shot}: column order differs")
    for r in REPEATS:
        for k in range(K):
            out = CACHE / f"run_r{r}_f{k}.npz"
            if out.exists():
                continue
            t_s = _time.time()
            cdir, tdir = run_dirs(r, k)
            c, tw = load_npz(cdir), load_npz(tdir)
            if not (np.array_equal(c["shot"], tw["shot"]) and np.array_equal(c["row"], tw["row"])
                    and np.allclose(c["gp_causal"], tw["gp_causal"], rtol=1e-6, atol=1e-6)):
                raise SystemExit(f"FATAL: r{r} f{k}: estimator and twin not scored on the same rows")
            net = N.load(cdir, r, k)
            man = json.loads((Path(net[2]["CES_SPLIT_DIR"]) / "split_manifest.json").read_text(encoding="utf-8"))
            test = set(man["test_files"])
            n = len(c["shot"])
            grow = np.empty(n, np.int64)
            names = np.empty(n, dtype=object)
            for i in np.unique(c["shot"]):
                sel = np.flatnonzero(c["shot"] == i)
                nm = N.valid[int(i)]
                if nm not in test:
                    raise SystemExit(f"FATAL: r{r} f{k}: scored discharge {nm} is not in the test fold")
                grow[sel] = compact_index(N.grid[nm])[c["row"][sel]]
                names[sel] = nm
            ygrid = np.array([N.grid[nm][g, 1] for nm, g in zip(names, grow)], dtype=np.float64)
            ydev = float(np.nanmax(np.abs(ygrid - c["y"])))
            est = {"base": np.full(n, np.nan), "G": np.full(n, np.nan)}
            for v in VNAMES:
                est[f"R_{v}"] = np.full(n, np.nan)
            rmask = np.zeros(n, dtype=bool)
            aft = np.zeros(n, dtype=np.int8)
            raw_test = sorted(nm for nm in raw if nm in test)
            for nm in np.unique(names):
                sel = np.flatnonzero(names == nm)
                arr = N.grid[nm]
                g = grow[sel]
                est["base"][sel] = N.predict(arr, net)[g]
                est["G"][sel] = N.predict(arr, net, shift=True)[g]
                aft[sel] = absent_rows(arr[:, 0].astype(np.float64))[g]
                if nm in raw:
                    rz = raw[nm]
                    okr = rz["ok"]
                    rmask[sel] = okr[g]
                    for v in VNAMES:
                        a2 = arr.copy()
                        a2[okr, 3:3 + N_FAST] = rz[f"v_{v}"][okr].astype(np.float32)
                        est[f"R_{v}"][sel] = N.predict(a2, net)[g]
            np.savez(out, shot=c["shot"], row=c["row"], grow=grow, names=names.astype(str), y=c["y"],
                     se_stored=c["se"], se_gp=c["gp_causal"], se_twin=tw["se"], se_hold=c["persistence"],
                     se_pchip=c["pchip"], se_gpoff=c["gp"], rmask=rmask, after_absent=aft,
                     y_grid_max_dev=ydev, raw_test_files=np.array(raw_test), test_files=np.array(sorted(test)),
                     **{f"est_{k_}": v for k_, v in est.items()})
            log(f"r{r} f{k}: {n} rows, {len(np.unique(names))} discharges, raw test discharges {len(raw_test)}, "
                f"part R rows {int(rmask.sum())}, y dev {ydev:.2g} ({_time.time() - t_s:.0f} s)")


def load_run(r, k):
    z = np.load(CACHE / f"run_r{r}_f{k}.npz", allow_pickle=False)
    return {k_: z[k_] for k_ in z.files}


# ------------------------------------------------------------------------------------------ checks
def hier_b39(per, a, b, rng):
    """analyze_b39.arm_p.hier, verbatim."""
    draws = rng.choice(np.array(REPEATS), size=B39_N)
    o = np.empty(B39_N)
    for r in REPEATS:
        sel = np.flatnonzero(draws == r)
        m, bb = per[r][a], per[r][b]
        idx = rng.integers(0, len(m), size=(len(sel), len(m)))
        o[sel] = 1.0 - m[idx].sum(1) / bb[idx].sum(1)
    point = float(np.mean([1 - per[r][a].sum() / per[r][b].sum() for r in REPEATS]))
    return [point, *ci(o)]


B39_N = 20_000


def sums(shot, v):
    u, inv = np.unique(shot, return_inverse=True)
    return np.bincount(inv, weights=v, minlength=len(u))


def stage_checks():
    res = {}
    # (a) every network reproduces its stored squared errors row by row
    a, alldev = {}, []
    for r in REPEATS:
        for k in range(K):
            z = load_run(r, k)
            dev = np.abs(np.abs(z["est_base"] - z["y"]) - np.sqrt(z["se_stored"]))
            a[f"r{r}_f{k}"] = {"rows": int(len(dev)), "max_dev_eV": float(np.nanmax(dev)),
                               "share_within_0.5_eV": float(np.mean(dev <= CHECK_A_EV)),
                               "finite": bool(np.isfinite(z["est_base"]).all()),
                               "y_grid_max_dev_eV": float(z["y_grid_max_dev"])}
            alldev.append(dev)
    amax = max(v["max_dev_eV"] for v in a.values())
    dv = np.concatenate(alldev)
    smin = min(v["share_within_0.5_eV"] for v in a.values())
    res["a"] = {"per_run": a, "largest_dev_eV": amax, "threshold_eV": CHECK_A_EV, "required_share": CHECK_A_SHARE,
                "min_share_within_0.5_eV": smin, "rows": int(len(dv)),
                "dev_quantiles_eV": {"median": float(np.median(dv)), "p99": float(np.quantile(dv, 0.99)),
                                     "max": float(dv.max())},
                "rows_above_0.5_eV": int((dv > CHECK_A_EV).sum()), "rows_above_0.05_eV": int((dv > 0.05).sum()),
                "rule": "addendum 1 (B.49 rule): per network, |error| within 0.5 eV of the stored one on >= 99 % of rows",
                "passes": bool(smin >= CHECK_A_SHARE and all(v["finite"] for v in a.values()))}
    # (b), (c) from the raw stage
    fr = {}
    for shot in RAW_SHOTS:
        fr[str(shot)] = json.loads((CACHE / f"raw_s{shot}.json").read_text(encoding="utf-8"))
    res["b"] = {"per_discharge": {s: v["b"] for s, v in fr.items()},
                "passes": bool(all(v["b"][d]["equal"] for v in fr.values() for d in DIAGS))}
    cpass = all(v["c"][d]["share_equal_within_one_sample"] >= v["c"][d]["recorded_share_b"] - CHECK_C_SLACK
                for v in fr.values() for d in DIAGS)
    tot = {d: [sum(v["c"][d]["rows"] * v["c"][d]["share_equal_within_one_sample"] for v in fr.values()),
               sum(v["c"][d]["rows"] for v in fr.values()),
               sum(v["c"][d]["rows"] * v["c"][d]["share_strict_held_minus_10ms"] for v in fr.values())] for d in DIAGS}
    res["c"] = {"per_discharge": {s: v["c"] for s, v in fr.items()},
                "all_discharges": {d: {"rows": int(x[1]), "share_within_one_sample": x[0] / x[1],
                                       "share_strict": x[2] / x[1]} for d, x in tot.items()},
                "slack": CHECK_C_SLACK, "passes": bool(cpass)}
    # (d) part G with no shift reproduces P1 and P2 of B.39 (generator of analyze_b39 replayed)
    per = {}
    for r in REPEATS:
        cat = {"shot": [], "c": [], "t": [], "gp_causal": []}
        for k in range(K):
            z = load_run(r, k)
            cat["shot"].append(z["shot"])
            cat["c"].append((z["est_base"] - z["y"]) ** 2)
            cat["t"].append(z["se_twin"])
            cat["gp_causal"].append(z["se_gp"])
        shot = np.concatenate(cat["shot"])
        per[r] = {kk: sums(shot, np.concatenate(v)) for kk, v in cat.items() if kk != "shot"}
    rng = np.random.default_rng(SEED_B39P)
    p1 = hier_b39(per, "c", "gp_causal", rng)
    p2 = hier_b39(per, "c", "t", rng)
    b39 = json.loads(B39_JSON.read_text(encoding="utf-8"))["P"]
    d1 = [abs(x - y) for x, y in zip(p1, b39["P1_causal_over_gp"])]
    d2 = [abs(x - y) for x, y in zip(p2, b39["P2_causal_over_twin"])]
    res["d"] = {"P1_mine": p1, "P1_b39": b39["P1_causal_over_gp"], "P2_mine": p2, "P2_b39": b39["P2_causal_over_twin"],
                "max_abs_diff": max(d1 + d2), "tolerance": CHECK_D_TOL, "passes": bool(max(d1 + d2) <= CHECK_D_TOL)}
    res["all_pass"] = bool(all(res[x]["passes"] for x in "abcd"))
    (CACHE / "checks.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"check (a) largest dev {amax:.4f} eV -> {res['a']['passes']}; (b) {res['b']['passes']}; (c) {res['c']['passes']} "
        f"{res['c']['all_discharges']}; (d) P1 {p1} vs {b39['P1_causal_over_gp']}, P2 {p2} vs {b39['P2_causal_over_twin']} "
        f"-> {res['d']['passes']}")
    if not res["all_pass"]:
        raise SystemExit("FATAL: a check fails; the readout stops")
    return res


# ------------------------------------------------------------------------------------------ readout
class TwoLevel:
    """Two-level resamples: a re-partition (repeat) is drawn, then its discharges are resampled with replacement.
    `per[r][key]` are per-discharge sums (same discharge order for every key of a repeat); every quantity of a part
    uses the same draws."""

    def __init__(self, per, n_boot, seed):
        rng = np.random.default_rng(seed)
        self.per = per
        self.draws = rng.choice(np.array(REPEATS), size=n_boot)
        self.idx = {}
        for r in REPEATS:
            sel = np.flatnonzero(self.draws == r)
            n = len(next(iter(per[r].values())))
            self.idx[r] = (sel, rng.integers(0, n, size=(len(sel), n)))
        self.n_boot = n_boot
        self.cache = {}

    def boot(self, key):
        if key not in self.cache:
            o = np.empty(self.n_boot)
            for r in REPEATS:
                sel, idx = self.idx[r]
                o[sel] = self.per[r][key][idx].sum(1)
            self.cache[key] = o
        return self.cache[key]

    def point(self, fn):
        return float(np.mean([fn({k_: v.sum() for k_, v in self.per[r].items()}) for r in REPEATS]))

    def q(self, fn, keys):
        pt = self.point(fn)
        b = fn({k_: self.boot(k_) for k_ in keys})
        return [pt, *ci(b)]


def skill(a, b, keys=None):
    return (lambda S: 1.0 - S[a] / S[b]), [a, b]


def readout_part(runs, sel_fn, variants, n_boot, seed, by_discharge=False, extra=None):
    """Per-discharge sums per repeat over the rows chosen by sel_fn(z) for estimates est_<variant> and the base."""
    per = {}
    allshots = set()
    for r in REPEATS:
        rows = {"shot": [], "base": [], "gp": [], "twin": [], "cnt": [], **{v: [] for v in variants},
                **{f"chg_{v}": [] for v in variants}}
        for k in range(K):
            z = runs[(r, k)]
            m = sel_fn(z)
            y = z["y"][m]
            b = z["est_base"][m]
            rows["shot"].append(z["names"][m])
            rows["base"].append((b - y) ** 2)
            rows["gp"].append(z["se_gp"][m])
            rows["twin"].append(z["se_twin"][m])
            rows["cnt"].append(np.ones(int(m.sum())))
            for v in variants:
                e = z[f"est_{v}"][m]
                if not np.isfinite(e).all():
                    raise SystemExit(f"FATAL: r{r} f{k}: non-finite estimate in variant {v}")
                rows[v].append((e - y) ** 2)
                rows[f"chg_{v}"].append((e - b) ** 2)
        shot = np.concatenate(rows["shot"])
        allshots.update(shot.tolist())
        u, inv = np.unique(shot, return_inverse=True)
        per[r] = {kk: np.bincount(inv, weights=np.concatenate(v), minlength=len(u)) for kk, v in rows.items() if kk != "shot"}
        per[r]["_names"] = u
    names_per_r = {r: per[r].pop("_names") for r in REPEATS}
    TL = TwoLevel(per, n_boot, seed)
    out = {"rows_per_repeat": [int(per[r]["cnt"].sum()) for r in REPEATS],
           "discharges_per_repeat": [int(len(names_per_r[r])) for r in REPEATS], "discharges": int(len(allshots)),
           "rmse_file_eV": TL.q(lambda S: np.sqrt(S["base"] / S["cnt"]), ["base", "cnt"]),
           "skill_file_over_gp": TL.q(*skill("base", "gp")), "skill_file_over_twin": TL.q(*skill("base", "twin")),
           "variants": {}}
    for v in variants:
        out["variants"][v] = {
            "rms_change_eV": TL.q(lambda S, v=v: np.sqrt(S[f"chg_{v}"] / S["cnt"]), [f"chg_{v}", "cnt"]),
            "rmse_eV": TL.q(lambda S, v=v: np.sqrt(S[v] / S["cnt"]), [v, "cnt"]),
            "change_skill": TL.q(*skill(v, "base")),
            "skill_over_gp": TL.q(*skill(v, "gp")),
            "skill_over_twin": TL.q(*skill(v, "twin")),
            "increment_change_over_twin": TL.q(lambda S, v=v: (1 - S[v] / S["twin"]) - (1 - S["base"] / S["twin"]),
                                               [v, "base", "twin"])}
        cs = out["variants"][v]["change_skill"]
        out["variants"][v]["inside_margin"] = bool(cs[1] >= -MARGIN and cs[2] <= MARGIN)
    for name, fn, keys in (extra or []):          # part D: further quantities from the same draws
        out[name] = TL.q(fn, keys)
    if by_discharge:
        d = {}
        for nm in sorted(allshots):
            acc = {kk: 0.0 for kk in per[REPEATS[0]]}
            nets = 0
            for r in REPEATS:
                j = np.flatnonzero(names_per_r[r] == nm)
                if len(j):
                    nets += 1
                    for kk in acc:
                        acc[kk] += float(per[r][kk][j[0]])
            rows_ = acc["cnt"] / max(nets, 1)
            d[nm] = {"networks": nets, "rows": int(round(rows_)), "rmse_file_eV": float(np.sqrt(acc["base"] / acc["cnt"])),
                     "variants": {v: {"rms_change_eV": float(np.sqrt(acc[f"chg_{v}"] / acc["cnt"])),
                                      "change_skill": float(1 - acc[v] / acc["base"])} for v in variants}}
        out["per_discharge"] = d
    return out


def scatter_summary(stats_std):
    """Sample-to-sample scatter of the raw streams at the held sample against the training standard deviation."""
    out = {}
    for diag in DIAGS:
        ratios = []
        for shot in RAW_SHOTS:
            j = json.loads((CACHE / f"raw_s{shot}.json").read_text(encoding="utf-8"))["scatter"][diag]
            for ch, rms in zip(j["channels"], j["rms_one_sample_difference"]):
                ratios.append(rms / np.sqrt(2.0) / stats_std[ch])
        ratios = np.array(ratios)
        out[diag] = {"median": float(np.median(ratios)), "min": float(ratios.min()), "max": float(ratios.max()),
                     "n": int(len(ratios))}
    return out


def change_in_z(stats_std, fast_cols):
    """RMS change of the fast-channel inputs per diagnostic, in training standard deviations, on replaced rows."""
    out = {}
    std = np.array([stats_std[c] for c in fast_cols])
    groups = {d: [i for i, c in enumerate(fast_cols) if c.startswith("BES_" if d == "BES" else d)] for d in DIAGS}
    for v in VNAMES:
        acc = {d: [0.0, 0] for d in DIAGS}
        for shot in RAW_SHOTS:
            z = np.load(CACHE / f"raw_s{shot}.npz")
            ok = z["ok"]
            dz = (z[f"v_{v}"][ok] - z["filev"][ok]) / std
            for d, gi in groups.items():
                acc[d][0] += float((dz[:, gi] ** 2).sum())
                acc[d][1] += dz[:, gi].size
        out[v] = {d: float(np.sqrt(a / max(n, 1))) for d, (a, n) in acc.items()}
    return out


def stage_readout():
    checks = json.loads((CACHE / "checks.json").read_text(encoding="utf-8"))
    if not checks["all_pass"]:
        raise SystemExit("FATAL: a check fails; no statement is read")
    runs = {(r, k): load_run(r, k) for r in REPEATS for k in range(K)}
    # every raw discharge is a test discharge of one fold in each re-partition
    cover = {}
    for shot in RAW_SHOTS:
        nm = f"s{shot}.csv"
        cover[nm] = [sum(nm in set(runs[(r, k)]["raw_test_files"].tolist()) for k in range(K)) for r in REPEATS]
    log("readout part R")
    partR = readout_part(runs, lambda z: z["rmask"].astype(bool), [f"R_{v}" for v in VNAMES], N_BOOT_R, SEED_R,
                         by_discharge=True)
    log("readout part G")
    partG = readout_part(runs, lambda z: np.ones(len(z["y"]), dtype=bool), ["G"], N_BOOT_G, SEED_G)
    partG_abs = readout_part(runs, lambda z: (z["after_absent"] >= 1) & (z["after_absent"] <= 3), ["G"], N_BOOT_G, SEED_G)
    partG_abs_rows = {}
    for kk in (1, 2, 3):
        partG_abs_rows[str(kk)] = readout_part(runs, lambda z, kk=kk: z["after_absent"] == kk, ["G"], N_BOOT_G, SEED_G)
    partG_rest = readout_part(runs, lambda z: z["after_absent"] == 0, ["G"], N_BOOT_G, SEED_G)
    # training standard deviations, mean over the 40 networks
    N_cols = [c for c in pd.read_csv(DATA / f"s{RAW_SHOTS[0]}.csv", nrows=0).columns if c.startswith(("BES_", "ECEI_", "MC"))]
    stds = []
    for r in REPEATS:
        for k in range(K):
            m = json.loads((run_dirs(r, k)[0] / "metrics.json").read_text(encoding="utf-8"))["normalization"]["stats"]
            stds.append(np.concatenate([np.asarray(m[g]["std"], float) for g in ("bes", "ecei", "mc")]))
    std_mean = dict(zip(N_cols, np.mean(stds, 0).tolist()))
    R = partR["variants"]
    V = {"R1": R["R_1samp"]["inside_margin"], "R2": R["R_0.5ms"]["inside_margin"], "R3": R["R_2ms"]["inside_margin"],
         "G1": bool(partG["variants"]["G"]["skill_over_twin"][1] > 0), "G2": partG["variants"]["G"]["inside_margin"]}
    res = {"registration": REG, "margin": MARGIN, "n_boot": {"R": N_BOOT_R, "G": N_BOOT_G},
           "seeds": {"R": SEED_R, "G": SEED_G, "check_d": SEED_B39P}, "checks": checks,
           "raw_discharge_fold_cover": cover, "partR": partR, "partG": partG, "partG_rows_1_to_3_after_absent": partG_abs,
           "partG_by_row_after_absent": partG_abs_rows, "partG_other_rows": partG_rest,
           "raw_scatter_over_training_sd": scatter_summary(std_mean),
           "input_change_in_training_sd": change_in_z(std_mean, N_cols),
           "raw_windows": {str(s): {k_: json.loads((CACHE / f"raw_s{s}.json").read_text(encoding="utf-8"))[k_]
                                    for k_ in ("window_s", "windows_by_diag_s", "rows_replaced", "held_offsets")}
                           for s in RAW_SHOTS},
           "verdicts": V}
    res["expectations"] = expectations(res)
    res["choices"] = CHOICES
    rt = json.loads((CACHE / "runtime.json").read_text()) if (CACHE / "runtime.json").exists() else {}
    res["runtime_s"] = rt
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"verdicts {V}; written {OUT_MD.name}")


def expectations(res):
    R, G = res["partR"]["variants"], res["partG"]["variants"]["G"]
    E = []

    def rng_(text, val, lo, hi):
        E.append({"expectation": text, "found": val, "inside": bool(lo <= val <= hi)})

    for v, lo, hi in (("1samp", 2, 8), ("0.5ms", 3, 10), ("2ms", 4, 12), ("10ms", 10, 30)):
        rng_(f"RMS change of the estimate, {LABEL[v]}: {lo} to {hi} eV", R[f"R_{v}"]["rms_change_eV"][0], lo, hi)
    E.append({"expectation": "RMS error near 200 eV (part R, file values; read as 150 to 250 eV)",
              "found": res["partR"]["rmse_file_eV"][0], "inside": bool(150 <= res["partR"]["rmse_file_eV"][0] <= 250)})
    E.append({"expectation": "RMS error near 200 eV (part G, file values; read as 150 to 250 eV)",
              "found": res["partG"]["rmse_file_eV"][0], "inside": bool(150 <= res["partG"]["rmse_file_eV"][0] <= 250)})
    for key, v, lo, hi in (("R1", "1samp", -0.003, 0.003), ("R2", "0.5ms", -0.004, 0.004), ("R3", "2ms", -0.006, 0.004)):
        rng_(f"{key} change skill {lo:+.3f} to {hi:+.3f}", R[f"R_{v}"]["change_skill"][0], lo, hi)
        E.append({"expectation": f"{key} holds", "found": "holds" if res["verdicts"][key] else "does not hold",
                  "inside": bool(res["verdicts"][key])})
    rng_("G2 change skill -0.030 to -0.005", G["change_skill"][0], -0.030, -0.005)
    E.append({"expectation": "G2 does not hold", "found": "holds" if res["verdicts"]["G2"] else "does not hold",
              "inside": bool(not res["verdicts"]["G2"])})
    rng_("G1 skill over the twin +0.020 to +0.045", G["skill_over_twin"][0], 0.020, 0.045)
    E.append({"expectation": "G1 holds", "found": "holds" if res["verdicts"]["G1"] else "does not hold",
              "inside": bool(res["verdicts"]["G1"])})
    rng_("rows 1 to 3 after an absent frame, fast channels one row old: change skill -0.15 to -0.03",
         res["partG_rows_1_to_3_after_absent"]["variants"]["G"]["change_skill"][0], -0.15, -0.03)
    return E


CHOICES = [
    "Treatment of every network: read from `run_b39.pooled_env` (identical for the 40 runs: drop of held values on, T_i "
    "spike cut 3000 eV, fast channels standardized with the statistics of the training discharges, no expanding "
    "standardizer, no zeroed Mirnov, v2); each metrics.json is checked against it (model, standardization, seed and "
    "initialisation seed of the runner).",
    "Scored rows: `CES_TI_row` indexes the grid rows with at least one finite target (`compact_index` of B.45/B.49); "
    "`CES_TI_shot` indexes the discharges with any finite target in the cut population. The grid T_i at the mapped row is "
    "compared with the stored `y_true` (largest difference listed under check (a)).",
    "The sample the file holds: for each row and diagnostic, the raw sample at offset 0, -1 or +1 from the stamp (nearest "
    "sample = rint((stamp - t0) fs)) whose values equal the file values of all channels of the diagnostic at the five "
    "printed decimals, tried in the order 0, -1, +1; a row without such a sample (Mirnov only) keeps the nearest sample. "
    "'d before' = that index minus round(d fs) samples; one raw sample = 1 sample of each diagnostic (2 us for BES and "
    "ECEI, 1 us for the BES of 31747, 0.5 us for Mirnov).",
    "The two means: the mean of the round(w fs) raw samples ending at the held sample (w = 0.1 ms or 1 ms), i.e. the "
    "samples in the w up to and including the stamp.",
    "Raw window of a discharge: the intersection of the windows of BES, ECEI and Mirnov (BES, ECEI and Mirnov are moved "
    "together, so a row is replaced only where all three streams exist). A grid row is replaced in every variant only "
    "if all samples of every variant exist (10 ms plus one sample before the stamp), so the replaced rows are the same "
    "in every variant; rows outside keep the file values. Every replaced row has its stamp minus 10 ms inside the "
    "window; the scored rows of part R are the replaced rows that the pooled run scored.",
    "Part G: the 15 standardized fast channels of row i of a block are those of row i - 1 of the block; the first row "
    "of each block (blocks of `build_blocks`, split at steps of at least 0.5 s) keeps its own. The time step and the CES "
    "channels are unchanged.",
    "Rows 1 to 3 after an absent frame: the first grid row after a step of 20 to 40 ms and the next two rows if they "
    "follow in 10 ms steps (the rule of `analyze_file_rules`).",
    "Quantities: the change skill is 1 - SSE(moved) / SSE(file values) with both estimates computed on the CPU here; "
    "the skills over the causal GP and the twin use their stored squared errors of the same rows. Point values are "
    "the mean over the eight re-partitions of the value of each re-partition (the point rule of `analyze_b39.hier`); "
    "the root-mean-square quantities likewise. Intervals: a re-partition is drawn, then its discharges (part R: the "
    "twelve discharges; part G: all scored discharges of the re-partition) are resampled with replacement; the same "
    "draws for every quantity and variant of a part (seeds 20261118 for part R, 20261119 for part G; the subsets of "
    "part G reuse the generator seed of part G).",
    "Check (a), as fixed by addendum 1: for every network, |estimate - y_true| computed here lies within 0.5 eV of the "
    "square root of the stored squared error on at least 99 % of its rows (the registered rule, largest difference "
    "below 0.05 eV, failed before any statement was read; see Deviations).",
    "Check (b): the rule of `analyze_file_rules.raw_side` run again; it passes when the number of rows and the shares "
    "(within one sample, and at offsets -1, 0, +1) equal those stored in `docs/paper/file_rules.json` for every "
    "discharge and diagnostic. The share at the nearest sample alone is also given.",
    "Check (c): on the rows of the file rules whose previous row is 10 ms earlier, the raw samples 10 ms (round(0.010 "
    "fs) samples) before the nearest sample, within one raw sample, are compared with the file values of the previous "
    "row (all channels, five decimals); it passes when, for every discharge and diagnostic, this share is at least the "
    "share recorded in the file rules minus 0.0005. The strict share (held index of the row minus 10 ms, no tolerance "
    "of one sample) is reported beside it.",
    "Check (d): the per-discharge sums of the squared errors computed here (file values) replace the stored ones in "
    "`analyze_b39.arm_p.hier`, replayed with its generator (seed 2728, 20,000 draws, P1 then P2); every number (point "
    "and both ends of the interval) must be within 0.0005 of `docs/paper/b39_primary.json`.",
    "Expectation 'RMS error near 200 eV' is read as 150 to 250 eV.",
]


# ------------------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"


def s3s(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def f1(x):
    return f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"


def hs(h):
    return "holds" if h else "does not hold"


def render(res):
    ck, P, G = res["checks"], res["partR"], res["partG"]
    V = res["verdicts"]
    L = ["# B.55 — the instant of the fast-channel sample (registered: PREREGISTRATION_B55.md)", "",
         "CPU, nothing trained. T_i, cut population. The 40 estimator networks of the pooled cross-validation of B.39 "
         "(`data/.b39p_v2_cut_r{r}_f{k}`, causal variant, fast channels standardized with the statistics of the training "
         "discharges), each run forward over the discharges of its own test fold with changed fast-channel inputs. The "
         "twin reads no fast channel and the causal GP none either: their errors are the stored errors of the same rows. "
         "Change skill = 1 - SSE(moved samples) / SSE(file values) on the same rows. Equivalence margin 0.01: "
         "'equivalent' means the 95 % interval of the change skill lies inside [-0.01, +0.01]. Intervals: two-level (a "
         "re-partition is drawn, then its discharges are resampled), "
         f"{res['n_boot']['R']:,} draws in part R, {res['n_boot']['G']:,} in part G.", "",
         "## Checks before any statement", ""]
    a = ck["a"]
    q = a["dev_quantiles_eV"]
    L.append(f"- **(a)** (rule of addendum 1, the B.49 rule) file values, every network against its stored squared errors: "
             f"the absolute error computed here lies within 0.5 eV of the stored one on at least "
             f"{100 * a['min_share_within_0.5_eV']:.3f} % of the rows of every network (required 99 %). Distribution of the "
             f"difference over {a['rows']:,} rows of 40 networks: median {q['median']:.3f} eV, 99th percentile "
             f"{q['p99']:.3f} eV, maximum {q['max']:.3f} eV; {a['rows_above_0.5_eV']} rows above 0.5 eV "
             f"({a['rows_above_0.05_eV']:,} above 0.05 eV, the registered threshold that failed before the addendum). The "
             "stored errors come from the GPU, these from the CPU; the relative size (about 1e-4) points to the arithmetic "
             "of the two devices (inferred, not verified). Every change of an estimate and every change skill below "
             "compares two estimates computed on the CPU, so the device difference does not enter them. Grid T_i against "
             f"stored y_true, largest difference {max(v['y_grid_max_dev_eV'] for v in a['per_run'].values()):.2g} eV "
             f"-> **{'passes' if a['passes'] else 'FAILS'}**.")
    b = ck["b"]
    L.append("- **(b)** nearest raw sample (within one raw sample) against the file value, rule of the file rules run "
             "again; rows and shares identical to `file_rules.json` on all 36 (discharge, diagnostic) pairs: "
             f"**{'passes' if b['passes'] else 'FAILS'}**.")
    L += ["", "| discharge | BES rows, share (recorded) / at nearest sample | ECEI | Mirnov |", "|---|---|---|---|"]
    for s, v in b["per_discharge"].items():
        L.append(f"| {s} | " + " | ".join(
            f"{v[d]['mine']['rows']}, {v[d]['mine']['share_equal_within_one_sample']:.3f} "
            f"({v[d]['recorded']['share_equal_within_one_sample']:.3f}) / {v[d]['nearest_sample_share']:.3f}" for d in DIAGS) + " |")
    c = ck["c"]
    L += ["", f"- **(c)** at d = 10 ms, rows whose previous row is 10 ms earlier: the raw sample 10 ms before the stamp "
          "(within one raw sample) equals the file value of the previous row: " + "; ".join(
              f"{d} {x['share_within_one_sample']:.4f} of {x['rows']:,} rows (strict, held sample minus 10 ms: "
              f"{x['share_strict']:.4f})" for d, x in c["all_discharges"].items())
          + f" -> **{'passes' if c['passes'] else 'FAILS'}** (every discharge and diagnostic at least its recorded share minus "
          f"{c['slack']}).", "",
          "| discharge | BES share (c) / recorded (b) | ECEI | Mirnov |", "|---|---|---|---|"]
    for s, v in c["per_discharge"].items():
        L.append(f"| {s} | " + " | ".join(f"{v[d]['share_equal_within_one_sample']:.3f} / {v[d]['recorded_share_b']:.3f} "
                                         f"({v[d]['rows']})" for d in DIAGS) + " |")
    d = ck["d"]
    L += ["", f"- **(d)** part G with no shift (estimates of this run, file values) in the pooled readout of B.39: "
          f"P1 {s3(d['P1_mine'])} against {s3(d['P1_b39'])}; P2 {s3(d['P2_mine'])} against {s3(d['P2_b39'])}; largest "
          f"difference {d['max_abs_diff']:.2e} -> **{'passes' if d['passes'] else 'FAILS'}**.", "",
          "All four checks pass." if ck["all_pass"] else "A CHECK FAILS: no statement is read.", "",
          "## Registered statements", "",
          f"- **R1** (one raw sample earlier, equivalent): change skill {s3(P['variants']['R_1samp']['change_skill'])} -> **{hs(V['R1'])}**.",
          f"- **R2** (0.5 ms earlier, equivalent): change skill {s3(P['variants']['R_0.5ms']['change_skill'])} -> **{hs(V['R2'])}**.",
          f"- **R3** (2 ms earlier, equivalent): change skill {s3(P['variants']['R_2ms']['change_skill'])} -> **{hs(V['R3'])}**.",
          f"- **G1** (fast channels one row old, skill over the twin with the interval above zero): "
          f"{s3(G['variants']['G']['skill_over_twin'])} -> **{hs(V['G1'])}**.",
          f"- **G2** (fast channels one row old, equivalent): change skill {s3(G['variants']['G']['change_skill'])} -> **{hs(V['G2'])}**.", "",
          "## Part R — raw streams, twelve discharges", "",
          f"Scored rows per re-partition: {', '.join(f'{x:,}' for x in P['rows_per_repeat'])} (discharges per re-partition: "
          f"{', '.join(map(str, P['discharges_per_repeat']))}); each discharge is estimated by eight networks, one per "
          "re-partition. Fold cover of the twelve discharges (test folds per re-partition): "
          + ("one in every re-partition for all twelve." if all(x == [1] * len(REPEATS) for x in res["raw_discharge_fold_cover"].values())
             else json.dumps(res["raw_discharge_fold_cover"])) + "", "",
          f"With the file values: RMS error {f1(P['rmse_file_eV'])} eV, skill over the causal GP {s3s(P['skill_file_over_gp'])}, "
          f"over the twin {s3s(P['skill_file_over_twin'])}.", "",
          "| variant | RMS change of the estimate (eV) | RMS error (eV) | change skill | inside [-0.01, +0.01] | skill over the causal GP | skill over the twin |",
          "|---|---|---|---|---|---|---|",
          f"| file values (sample at the stamp) | 0 | {f1(P['rmse_file_eV'])} | 0 | - | {s3s(P['skill_file_over_gp'])} | {s3s(P['skill_file_over_twin'])} |"]
    for v in VNAMES:
        x = P["variants"][f"R_{v}"]
        L.append(f"| {LABEL[v]} | {f1(x['rms_change_eV'])} | {f1(x['rmse_eV'])} | {s3(x['change_skill'])} | "
                 f"{'yes' if x['inside_margin'] else 'no'} | {s3s(x['skill_over_gp'])} | {s3s(x['skill_over_twin'])} |")
    L += ["", "Change of the fast-channel inputs on the replaced rows, RMS in training standard deviations (mean of the 40 "
          "networks' training statistics):", "", "| variant | BES | ECEI | Mirnov |", "|---|---|---|---|"]
    for v in VNAMES:
        x = res["input_change_in_training_sd"][v]
        L.append(f"| {LABEL[v]} | {x['BES']:.3f} | {x['ECEI']:.3f} | {x['MC']:.3f} |")
    sc = res["raw_scatter_over_training_sd"]
    L += ["", "Sample-to-sample scatter of the raw streams at the held sample (RMS of the one-sample difference / sqrt 2), "
          "in training standard deviations, median (range) over the channels of the twelve discharges: " + "; ".join(
              f"{dg} {x['median']:.3f} ({x['min']:.3f} to {x['max']:.3f})" for dg, x in sc.items()) + ".", "",
          "### Per discharge (pooled over its eight networks)", "",
          "| discharge | rows | RMS error, file values (eV) | " + " | ".join(f"{v}: RMS change (eV) / change skill" for v in VNAMES) + " |",
          "|---|---|---|" + "---|" * len(VNAMES)]
    for nm, x in P["per_discharge"].items():
        L.append(f"| {nm.replace('.csv', '')} | {x['rows']} | {x['rmse_file_eV']:.1f} | " + " | ".join(
            f"{x['variants'][f'R_{v}']['rms_change_eV']:.1f} / {x['variants'][f'R_{v}']['change_skill']:+.4f}" for v in VNAMES) + " |")
    rw = res["raw_windows"]
    L += ["", "Raw windows (intersection of the three diagnostics) and replaced rows: " + "; ".join(
        f"{s} {w['window_s'][0]:.3f}-{w['window_s'][1]:.3f} s, {w['rows_replaced']} rows" for s, w in rw.items()) + ".",
          "Held BES sample, all twelve discharges: " + ", ".join(
              f"offset {o} on {sum(w['held_offsets']['BES'][o] for w in rw.values()):,} rows" for o in ("-1", "0", "1", "99"))
          + f" of {sum(sum(w['held_offsets']['BES'].values()) for w in rw.values()):,} (99 = no match).",
          "Offset of the sample the file holds (rows of the full file inside the window; 99 = no match, nearest kept): " + "; ".join(
              f"{s} " + ", ".join(f"{dg} {json.dumps(w['held_offsets'][dg])}" for dg in DIAGS) for s, w in rw.items()) + ".", "",
          "## Part G — fast channels one row old, all discharges", ""]
    gv = G["variants"]["G"]
    L += [f"Rows per re-partition: {', '.join(f'{x:,}' for x in G['rows_per_repeat'])}; discharges {G['discharges']}.", "",
          "| rows | rows per re-partition | RMS change (eV) | RMS error, file values (eV) | RMS error, one row old (eV) | change skill | inside [-0.01, +0.01] | "
          "skill over causal GP: file / one row old | skill over the twin: file / one row old |",
          "|---|---|---|---|---|---|---|---|---|"]

    def grow_(lab, part):
        x = part["variants"]["G"]
        L.append(f"| {lab} | {part['rows_per_repeat'][0]:,} | {f1(x['rms_change_eV'])} | {f1(part['rmse_file_eV'])} | {f1(x['rmse_eV'])} | "
                 f"{s3(x['change_skill'])} | {'yes' if x['inside_margin'] else 'no'} | {s3s(part['skill_file_over_gp'])} / "
                 f"{s3s(x['skill_over_gp'])} | {s3s(part['skill_file_over_twin'])} / {s3s(x['skill_over_twin'])} |")

    grow_("all scored rows", G)
    grow_("rows 1 to 3 after an absent frame", res["partG_rows_1_to_3_after_absent"])
    for kk in ("1", "2", "3"):
        grow_(f"row {kk} after an absent frame", res["partG_by_row_after_absent"][kk])
    grow_("all other rows", res["partG_other_rows"])
    L += ["", f"Change of the skill over the twin (one row old minus file values), all rows: {s3(gv['increment_change_over_twin'])}.", "",
          "## Expectations of the registration (section 6)", "", "| expectation | found | inside |", "|---|---|---|"]
    for e in res["expectations"]:
        f = e["found"]
        fs = f if isinstance(f, str) else (f"{f:+.4f}" if abs(f) < 1 else f"{f:.1f}")
        L.append(f"| {e['expectation']} | {fs} | {'inside' if e['inside'] else 'outside'} |")
    L += ["", "## Choices made where the registration left room", ""] + [f"- {x}" for x in res["choices"]]
    L += ["", "## Deviations", "", "Check (a) as registered (largest difference of the estimate below 0.05 eV) failed "
          "(largest difference 2.57 eV, 23.6 % of the rows above 0.05 eV) before any statement was read; addendum 1 of "
          "the registration replaced it with the B.49 rule, which is the rule applied above. The implementation choices "
          "listed above are recorded in addendum 1 as implemented. No other deviation from the registered definitions, "
          "margin, intervals or rules.", "",
          "## Limits stated in advance (section 7)", "",
          "The twelve discharges with raw streams were chosen for other purposes and the raw windows cover part of each. "
          "The networks were trained with samples at the stamp: a difference shows how these networks respond to another "
          "instant, not what a network trained with it would reach. The start of the exposure is inferred from beam blips "
          "to within about 2 ms.", "",
          "Run time (s): " + ", ".join(f"{k_} {v:.0f}" for k_, v in res["runtime_s"].items()) + "."]
    if "partD" in res:
        L += render_d(res["partD"])
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------ part D (addendum 2)
D_DS = ("1samp", "0.5ms", "2ms")
D_R = [(f"Dmc_{v}", "MC", v) for v in D_DS] + [(f"Dbe_{v}", "BE", v) for v in D_DS]
D_G = ("Gbes", "Gecei", "Gmc", "MCmean")
D_LABEL = {**{f"Dmc_{v}": f"only Mirnov moved, {LABEL[v]}" for v in D_DS},
           **{f"Dbe_{v}": f"BES and ECEI moved, {LABEL[v]}; Mirnov at the file values" for v in D_DS},
           "Gbes": "only BES one row old", "Gecei": "only ECEI one row old", "Gmc": "only Mirnov one row old",
           "MCmean": "Mirnov at its training mean (0 after standardization) at every row"}


def fast_groups(cols):
    g = {"BES": [i for i, c in enumerate(cols) if c.startswith("BES_")],
         "ECEI": [i for i, c in enumerate(cols) if c.startswith("ECEI_")],
         "MC": [i for i, c in enumerate(cols) if c.startswith("MC")]}
    if sorted(sum(g.values(), [])) != list(range(N_FAST)):
        raise SystemExit(f"FATAL: fast columns {cols} do not split into BES, ECEI, Mirnov")
    return g


def predict_fn(N, arr, net, fn=None):
    """Nets.predict with a transform of the standardized block inputs (fn(x) -> new x)."""
    model, stats, _ = net
    tm = arr[:, 0]
    out = np.full(len(tm), np.nan)
    mean, std = float(stats["target"]["mean"][0]), float(stats["target"]["std"][0])
    spans = block_spans(tm.astype(np.float64))
    blks = N.build_blocks(arr, N.dims, stats, per_shot_norm=False, expanding=False)
    if len(blks) != len(spans):
        raise SystemExit("FATAL: block spans do not match build_blocks")
    with N.torch.no_grad():
        for blk, (a, b) in zip(blks, spans):
            x = blk["x"] if fn is None else fn(blk["x"])
            out[a:b] = model(N.torch.from_numpy(x).unsqueeze(0))[0][:, 0].numpy() * std + mean
    return out


def shift_cols(cols):
    def f(x):
        y = x.copy()
        y[1:, cols] = x[:-1, cols]
        return y
    return f


def zero_cols(cols):
    def f(x):
        y = x.copy()
        y[:, cols] = 0.0
        return y
    return f


def stage_nets_d():
    N = Nets()
    gr = fast_groups(N.fast_cols)
    log(f"part D: grid loaded; groups {gr}")
    raw = {}
    for shot in RAW_SHOTS:
        z = np.load(CACHE / f"raw_s{shot}.npz")
        raw[f"s{shot}.csv"] = {k_: z[k_] for k_ in z.files}
    fns = {"Gbes": shift_cols(gr["BES"]), "Gecei": shift_cols(gr["ECEI"]), "Gmc": shift_cols(gr["MC"]),
           "MCmean": zero_cols(gr["MC"])}
    rcols = {"MC": gr["MC"], "BE": gr["BES"] + gr["ECEI"]}
    for r in REPEATS:
        for k in range(K):
            out = CACHE / f"runD_r{r}_f{k}.npz"
            if out.exists():
                continue
            t_s = _time.time()
            z = load_run(r, k)
            net = N.load(run_dirs(r, k)[0], r, k)
            names, grow = z["names"], z["grow"]
            n = len(names)
            est = {v: np.full(n, np.nan) for v in (*D_G, *[x[0] for x in D_R])}
            base = np.full(n, np.nan)
            for nm in np.unique(names):
                sel = np.flatnonzero(names == nm)
                arr = N.grid[nm]
                g = grow[sel]
                base[sel] = predict_fn(N, arr, net)[g]
                for v, fn in fns.items():
                    est[v][sel] = predict_fn(N, arr, net, fn)[g]
                if nm in raw:
                    rz = raw[nm]
                    okr = np.flatnonzero(rz["ok"])
                    for v, grp, d in D_R:
                        a2 = arr.copy()
                        cc = np.array(rcols[grp])
                        a2[np.ix_(okr, 3 + cc)] = rz[f"v_{d}"][np.ix_(okr, cc)].astype(np.float32)
                        est[v][sel] = predict_fn(N, a2, net)[g]
            dev = float(np.max(np.abs(base - z["est_base"])))
            if dev > 1e-6:
                raise SystemExit(f"FATAL: r{r} f{k}: file-value estimate differs from the part R/G run by {dev} eV")
            np.savez(out, base_recomputed_max_dev=dev, **{f"est_{v}": e for v, e in est.items()})
            log(f"part D r{r} f{k}: done, base reproduced to {dev:.1e} eV ({_time.time() - t_s:.0f} s)")


D_CHOICES = [
    "Same rows, generators and draws as parts R and G: the twelve-discharge variants use the draws of part R (seed "
    "20261118, 2,000), the all-discharge variants those of part G (seed 20261119, 20,000); the subset of rows 1 to 3 "
    "after an absent frame reuses the seed of part G. The file-value estimate is recomputed and must equal the one of "
    "parts R and G (largest difference given above).",
    "Moved samples on the twelve discharges: the raw values of the variant of part R for the moved channels, the file "
    "values for the others, on the same replaced rows.",
    "One row old for one diagnostic: as in part G, only on the standardized channels of that diagnostic.",
    "Mirnov at its training mean: the two standardized Mirnov channels set to 0 at every row of every block.",
    "D1: ratio of the root-mean-square change with only Mirnov moved to that with BES and ECEI moved, both at 0.5 ms; "
    "point = mean over the re-partitions of the ratio of each, interval from the draws of part R; holds if the lower "
    "end is above 1. D3 holds if the upper end of the change skill is below zero.",
    "Expectation 'interval about 0.02 wide on each side' of D2 is read as each side between 0.01 and 0.03.",
]


def stage_readout_d():
    res = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    if not res["checks"]["all_pass"]:
        raise SystemExit("FATAL: a check fails")
    runs, devs = {}, []
    for r in REPEATS:
        for k in range(K):
            z = load_run(r, k)
            zd = np.load(CACHE / f"runD_r{r}_f{k}.npz")
            devs.append(float(zd["base_recomputed_max_dev"]))
            z.update({k_: zd[k_] for k_ in zd.files if k_.startswith("est_")})
            runs[(r, k)] = z

    def ratio(S):
        return np.sqrt(S["chg_Dmc_0.5ms"] / S["cnt"]) / np.sqrt(S["chg_Dbe_0.5ms"] / S["cnt"])

    log("part D readout, twelve discharges")
    pr = readout_part(runs, lambda z: z["rmask"].astype(bool), [x[0] for x in D_R], N_BOOT_R, SEED_R, by_discharge=True,
                      extra=[("D1_ratio", ratio, ["chg_Dmc_0.5ms", "chg_Dbe_0.5ms", "cnt"])])
    log("part D readout, all discharges")
    pg = readout_part(runs, lambda z: np.ones(len(z["y"]), dtype=bool), list(D_G), N_BOOT_G, SEED_G)
    pga = readout_part(runs, lambda z: (z["after_absent"] >= 1) & (z["after_absent"] <= 3), list(D_G), N_BOOT_G, SEED_G)
    rv, gv = pr["variants"], pg["variants"]
    d1 = pr["D1_ratio"]
    V = {"D1": bool(d1[1] > 1), "D2": rv["Dbe_2ms"]["inside_margin"], "D3": bool(gv["Gbes"]["change_skill"][2] < 0),
         "D4": gv["Gecei"]["inside_margin"], "D5": gv["Gmc"]["inside_margin"],
         "D6": bool(gv["MCmean"]["skill_over_twin"][1] > 0), "D7": gv["MCmean"]["inside_margin"]}
    E = []

    def ex(text, val, lo, hi):
        E.append({"expectation": text, "found": val, "inside": bool(lo <= val <= hi)})

    def exv(key, want):
        E.append({"expectation": f"{key} {'holds' if want else 'does not hold'}",
                  "found": "holds" if V[key] else "does not hold", "inside": bool(V[key] == want)})

    ex("D1 ratio 2 to 6", d1[0], 2, 6)
    exv("D1", True)
    cs = rv["Dbe_2ms"]["change_skill"]
    ex("D2 change skill -0.010 to +0.005", cs[0], -0.010, 0.005)
    E.append({"expectation": "D2 interval about 0.02 wide on each side (read as 0.01 to 0.03 below and above the point)",
              "found": f"{cs[0] - cs[1]:.4f} below, {cs[2] - cs[0]:.4f} above",
              "inside": bool(0.01 <= cs[0] - cs[1] <= 0.03 and 0.01 <= cs[2] - cs[0] <= 0.03)})
    exv("D2", False)
    ex("D3 change skill -0.09 to -0.04", gv["Gbes"]["change_skill"][0], -0.09, -0.04)
    exv("D3", True)
    ex("D4 change skill -0.010 to +0.002", gv["Gecei"]["change_skill"][0], -0.010, 0.002)
    exv("D4", True)
    ex("D5 change skill -0.015 to +0.005", gv["Gmc"]["change_skill"][0], -0.015, 0.005)
    exv("D5", False)
    ex("D6 skill over the twin +0.030 to +0.055", gv["MCmean"]["skill_over_twin"][0], 0.030, 0.055)
    exv("D6", True)
    ex("D7 change skill -0.020 to +0.010", gv["MCmean"]["change_skill"][0], -0.020, 0.010)
    exv("D7", False)
    rt = json.loads((CACHE / "runtime.json").read_text()) if (CACHE / "runtime.json").exists() else {}
    res["partD"] = {"registration": "addendum 2 of " + REG + " (registered after parts R and G were read)",
                    "twelve_discharges": pr, "all_discharges": pg, "all_discharges_rows_1_to_3_after_absent": pga,
                    "file_value_estimate_reproduced_max_dev_eV": max(devs), "verdicts": V, "expectations": E,
                    "labels": D_LABEL, "choices": D_CHOICES, "runtime_s": {"netsD": rt.get("netsD")}}
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"part D verdicts {V}; D1 ratio {d1}")


def render_d(D):
    V, lab = D["verdicts"], D["labels"]
    pr, pg, pga = D["twelve_discharges"], D["all_discharges"], D["all_discharges_rows_1_to_3_after_absent"]
    d1 = pr["D1_ratio"]
    rnames = [x[0] for x in D_R]
    L = ["", "## Part D — by diagnostic (addendum 2; registered after parts R and G were read)", "",
         "Same 40 networks, rows, quantities and intervals as parts R and G (same draws). The estimate with the file "
         f"values was recomputed and equals that of parts R and G to {D['file_value_estimate_reproduced_max_dev_eV']:.1e} eV.", "",
         "### Statements", "",
         f"- **D1** (0.5 ms: RMS change with only Mirnov moved over that with BES and ECEI moved, interval above 1): "
         f"ratio {d1[0]:.2f} [{d1[1]:.2f}, {d1[2]:.2f}] -> **{hs(V['D1'])}**.",
         f"- **D2** (BES and ECEI 2 ms earlier, Mirnov at the file values, equivalent): change skill "
         f"{s3(pr['variants']['Dbe_2ms']['change_skill'])} -> **{hs(V['D2'])}**.",
         f"- **D3** (only BES one row old, change skill with the interval below zero): "
         f"{s3(pg['variants']['Gbes']['change_skill'])} -> **{hs(V['D3'])}**.",
         f"- **D4** (only ECEI one row old, equivalent): {s3(pg['variants']['Gecei']['change_skill'])} -> **{hs(V['D4'])}**.",
         f"- **D5** (only Mirnov one row old, equivalent): {s3(pg['variants']['Gmc']['change_skill'])} -> **{hs(V['D5'])}**.",
         f"- **D6** (Mirnov at its training mean, skill over the twin with the interval above zero): "
         f"{s3(pg['variants']['MCmean']['skill_over_twin'])} -> **{hs(V['D6'])}**.",
         f"- **D7** (Mirnov at its training mean, equivalent): {s3(pg['variants']['MCmean']['change_skill'])} -> "
         f"**{hs(V['D7'])}**.", "",
         "### Twelve discharges (raw streams)", "",
         f"With the file values: RMS error {f1(pr['rmse_file_eV'])} eV, skill over the causal GP "
         f"{s3s(pr['skill_file_over_gp'])}, over the twin {s3s(pr['skill_file_over_twin'])}. Rows per re-partition: "
         f"{', '.join(f'{x:,}' for x in pr['rows_per_repeat'])}.", "",
         "| variant | RMS change of the estimate (eV) | RMS error (eV) | change skill | inside [-0.01, +0.01] | "
         "skill over the causal GP | skill over the twin |", "|---|---|---|---|---|---|---|"]
    for v in rnames:
        x = pr["variants"][v]
        L.append(f"| {lab[v]} | {f1(x['rms_change_eV'])} | {f1(x['rmse_eV'])} | {s3(x['change_skill'])} | "
                 f"{'yes' if x['inside_margin'] else 'no'} | {s3s(x['skill_over_gp'])} | {s3s(x['skill_over_twin'])} |")
    L += ["", "Per discharge (pooled over its eight networks), RMS change (eV) / change skill:", "",
          "| discharge | rows | " + " | ".join(rnames) + " |", "|---|---|" + "---|" * len(rnames)]
    for nm, x in pr["per_discharge"].items():
        L.append(f"| {nm.replace('.csv', '')} | {x['rows']} | " + " | ".join(
            f"{x['variants'][v]['rms_change_eV']:.1f} / {x['variants'][v]['change_skill']:+.4f}" for v in rnames) + " |")
    L += ["", "### All discharges", "",
          "| variant | rows | RMS change (eV) | RMS error, file values (eV) | RMS error, variant (eV) | change skill | "
          "inside [-0.01, +0.01] | skill over causal GP: file / variant | skill over the twin: file / variant |",
          "|---|---|---|---|---|---|---|---|---|"]
    for part, rows_lab in ((pg, "all scored rows"), (pga, "rows 1 to 3 after an absent frame")):
        for v in D_G:
            x = part["variants"][v]
            L.append(f"| {lab[v]} | {rows_lab} | {f1(x['rms_change_eV'])} | {f1(part['rmse_file_eV'])} | "
                     f"{f1(x['rmse_eV'])} | {s3(x['change_skill'])} | {'yes' if x['inside_margin'] else 'no'} | "
                     f"{s3s(part['skill_file_over_gp'])} / {s3s(x['skill_over_gp'])} | "
                     f"{s3s(part['skill_file_over_twin'])} / {s3s(x['skill_over_twin'])} |")
    L += ["", "### Expectations of addendum 2", "", "| expectation | found | inside |", "|---|---|---|"]
    for e in D["expectations"]:
        f = e["found"]
        fs = f if isinstance(f, str) else (f"{f:+.4f}" if abs(f) < 1 else f"{f:.2f}")
        L.append(f"| {e['expectation']} | {fs} | {'inside' if e['inside'] else 'outside'} |")
    L += ["", "### Choices made for part D", ""] + [f"- {c}" for c in D["choices"]]
    L += ["", "Limit added by addendum 2: a network that was trained with the Mirnov channels and is given their mean "
          "is not a network trained without them.", "",
          "Run time of part D (s): " + ", ".join(f"{k_} {v:.0f}" for k_, v in D["runtime_s"].items() if v is not None) + "."]
    return L

def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "render":
        OUT_MD.write_text(render(json.loads(OUT_JSON.read_text(encoding="utf-8"))), encoding="utf-8", newline="\n")
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    rt_path = CACHE / "runtime.json"
    for name, fn in (("raw", stage_raw), ("nets", stage_nets), ("checks", stage_checks), ("readout", stage_readout),
                     ("netsD", stage_nets_d), ("readoutD", stage_readout_d)):
        if stage in (name, "all") or (stage == "partD" and name in ("netsD", "readoutD")):
            t = _time.time()
            fn()
            rt = json.loads(rt_path.read_text()) if rt_path.exists() else {}
            rt[name] = rt.get(name, 0.0) + round(_time.time() - t, 1) if name in ("raw", "nets", "netsD") else round(_time.time() - t, 1)
            rt_path.write_text(json.dumps(rt))
    log("done")


if __name__ == "__main__":
    main()
