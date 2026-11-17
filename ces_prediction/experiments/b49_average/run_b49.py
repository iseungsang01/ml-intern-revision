"""B.49 (PREREGISTRATION_B49.md): the average of the T_i estimates of three initialisations as one estimator.

For every file-level split (42, 1, 7, 123) the three initialisations of the causal variant of B.39 and the
three of the history-only twin are run forward on the CPU over the TEST discharges; the T_i estimate at every
scored row of `comparison_errors_test.npz` is checked against the stored squared error (absolute error within
0.5 eV on at least 99 % of the rows, per member), then
    E   = mean of the three causal-variant estimates at the row,
    E_h = mean of the three twin estimates at the row,
and E1 (E over the causal GP) and E2 (E over E_h) are read with 2,000 resamples of the test discharges.

CPU only; nothing is trained.  Run from the repo root:
    py ces_prediction/experiments/b49_average/run_b49.py
Output: docs/paper/b49_average.json / .md;  per-split estimate cache in data/.b49/ (gitignored).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b30_thinned"))
sys.path.insert(0, str(HERE.parent / "b39_primary"))
import run_b30 as R  # noqa: E402
import run_b39 as B39  # noqa: E402

REPO = R.REPO
DATA = R.DATA
CACHE = DATA / ".b49"
OUT_JSON = REPO / "docs" / "paper" / "b49_average.json"
OUT_MD = REPO / "docs" / "paper" / "b49_average.md"
B39_JSON = REPO / "docs" / "paper" / "b39_primary.json"
REG = "ces_prediction/experiments/PREREGISTRATION_B49.md"
SEEDS = B39.SEEDS
SEED, N_BOOT = 20261118, 2000          # resamples of split s drawn from default_rng(SEED + s)
T = "CES_TI"
CHECK_ABS_EV, CHECK_SHARE = 0.5, 0.99
REFS = {"persistence": "hold", "gp_causal": "causal GP", "pchip": "PCHIP", "gp": "offline GP"}
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def compact_index(arr):
    """Grid rows that dataset.py keeps (at least one target finite); CES_TI_row indexes these (B.45)."""
    return np.flatnonzero(np.isfinite(arr[:, 1:3]).any(1))


def load_npz(run):
    z = np.load(run / "comparison_errors_test.npz")
    d = {"shot": z[f"{T}_shot"].astype(np.int64), "row": z[f"{T}_row"].astype(np.int64),
         "se_model": z[f"{T}_se_model"].astype(np.float64), "y": z[f"{T}_y_true"].astype(np.float64)}
    for r in REFS:
        d[r] = z[f"{T}_se_{r}"].astype(np.float64)
    return d


class Nets:
    def __init__(self):
        R.pin_env()
        (self.torch, _B, self.load_stats, load_grid_files, self.build_blocks, self.GAP, self.MODELS) = R.imports()
        self.torch.set_num_threads(3)
        self.grid, self.dims = load_grid_files(DATA, True, R.POPS["cut"])
        self.names = sorted(self.grid)
        self.valid = [nm for nm in self.names if np.isfinite(self.grid[nm][:, 1:3]).any()]

    def load(self, run, variant):
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if m["seq_model"] != variant or m.get("per_shot_norm_expanding") or m.get("zero_mc") \
                or m.get("extra_vt_channels") or m.get("extra_ti_channels") or m.get("train_ctx"):
            raise SystemExit(f"FATAL: {run.name} is not the registered {variant} treatment")
        if variant == "v2" and bool(m["per_shot_input_norm"]):
            raise SystemExit(f"FATAL: {run.name}: the causal variant must use training-discharge statistics")
        stats = self.load_stats(m)
        model = self.MODELS[variant]()
        model.load_state_dict(self.torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
        model.eval()
        return model, stats, bool(m["per_shot_input_norm"]), int(m["init_seed"])

    def predict(self, name, net):
        model, stats, per_shot, _ = net
        arr = self.grid[name]
        tm = arr[:, 0].astype(np.float64)
        spans = R.block_spans(tm, self.GAP)
        blks = self.build_blocks(arr, self.dims, stats, per_shot_norm=per_shot, expanding=False)
        if len(blks) != len(spans):
            raise SystemExit("FATAL: block spans do not match build_blocks")
        out = np.full(len(tm), np.nan)
        mean, std = float(stats["target"]["mean"][0]), float(stats["target"]["std"][0])
        with self.torch.no_grad():
            for blk, (a, b) in zip(blks, spans):
                if not np.array_equal(blk["time"], arr[a:b, 0]):
                    raise SystemExit(f"FATAL: {name}: block times do not match the grid")
                out[a:b] = model(self.torch.from_numpy(blk["x"]).unsqueeze(0))[0][:, 0].numpy() * std + mean
        return out


def members(s):
    ini = B39.inits(s)
    return [("v2", i, B39.split_run_dir("v2", s, i)) for i in ini] + [("v2h", i, B39.twin_dir(s, i)) for i in ini]


def estimates(N, s):
    """(stored npz per member, CPU estimate per member at the scored rows), cached in data/.b49/."""
    mem = members(s)
    st = {f"{v} i{i}": load_npz(d) for v, i, d in mem}
    ref = st[f"v2 i{s}"]
    for k, d in st.items():
        if not (np.array_equal(d["shot"], ref["shot"]) and np.array_equal(d["row"], ref["row"])):
            raise SystemExit(f"FATAL: split {s} {k}: not scored on the same rows")
        if not np.array_equal(d["y"], ref["y"]):
            raise SystemExit(f"FATAL: split {s} {k}: y_true differs")
        for r in REFS:
            if not np.allclose(d[r], ref[r], rtol=1e-6, atol=1e-6):
                raise SystemExit(f"FATAL: split {s} {k}: stored {r} errors differ")
    cache = CACHE / f"s{s}.npz"
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["shot"], ref["shot"]) and np.array_equal(z["row"], ref["row"]) \
                and all(k.replace(" ", "_") in z for k in st):
            log(f"split {s}: estimates from cache {cache.name}")
            return st, {k: z[k.replace(" ", "_")] for k in st}, {"y_grid_max_dev_eV": float(z["y_grid_max_dev_eV"])}
    shot, row = ref["shot"], ref["row"]
    grow = np.empty(len(shot), np.int64)
    ygrid = np.empty(len(shot))
    files = {}
    for i in np.unique(shot):
        sel = np.flatnonzero(shot == i)
        name = N.valid[int(i)]
        arr = N.grid[name]
        g = compact_index(arr)[row[sel]]
        grow[sel] = g
        ygrid[sel] = arr[g, 1]
        files[name] = sel
    ydev = float(np.nanmax(np.abs(ygrid - ref["y"])))
    if not np.isfinite(ygrid).all() or ydev > 0.05:
        raise SystemExit(f"FATAL: split {s}: grid T_i at the mapped rows differs from y_true by up to {ydev:.3f} eV")
    est = {}
    for v, i, d in mem:
        net = N.load(d, v)
        if net[3] != i:
            raise SystemExit(f"FATAL: {d.name}: init_seed {net[3]} != {i}")
        e = np.full(len(shot), np.nan)
        for name, sel in files.items():
            e[sel] = N.predict(name, net)[grow[sel]]
        est[f"{v} i{i}"] = e
        log(f"split {s}: {v} i{i} ({d.name}) forward pass done")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, shot=shot, row=row, y_grid_max_dev_eV=ydev, **{k.replace(" ", "_"): v for k, v in est.items()})
    return st, est, {"y_grid_max_dev_eV": ydev}


def check(st, est, s):
    out = {}
    for k in st:
        dev = np.abs(np.abs(est[k] - st[k]["y"]) - np.sqrt(st[k]["se_model"]))
        share = float(np.mean(dev <= CHECK_ABS_EV))
        out[k] = {"rows": int(len(dev)), "share_within_0.5_eV": share, "max_dev_eV": float(np.nanmax(dev)),
                  "finite": bool(np.isfinite(est[k]).all()), "passed": bool(share >= CHECK_SHARE and np.isfinite(est[k]).all())}
        if not out[k]["passed"]:
            raise SystemExit(f"FATAL: check split {s} {k}: share {share:.4f}, max dev {out[k]['max_dev_eV']:.3f} eV")
    return out


def wquant(v, inv, W, q):
    """Quantile q of v with row weights W[:, inv] (inverted CDF), per resample."""
    order = np.argsort(v, kind="stable")
    vs, io = v[order], inv[order]
    out = np.empty(W.shape[0])
    for a in range(0, W.shape[0], 200):
        w = W[a:a + 200][:, io].astype(np.float32)
        c = np.cumsum(w, axis=1)
        k = (c < q * c[:, -1:]).sum(1)
        out[a:a + 200] = vs[np.minimum(k, len(vs) - 1)]
    return out


def run_split(N, s, b39):
    st, est, extra = estimates(N, s)
    chk = check(st, est, s)
    log(f"split {s}: check passed, shares " + ", ".join(f"{k} {c['share_within_0.5_eV']:.4f}" for k, c in chk.items()))
    ini = B39.inits(s)
    ref = st[f"v2 i{s}"]
    y, shot = ref["y"], ref["shot"]
    u, inv = np.unique(shot, return_inverse=True)
    n = len(u)
    W = np.random.default_rng(SEED + s).multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)
    fs = lambda x: np.bincount(inv, x, n)                                       # noqa: E731
    C = {i: est[f"v2 i{i}"] for i in ini}
    H = {i: est[f"v2h i{i}"] for i in ini}
    E = np.mean([C[i] for i in ini], axis=0)
    Eh = np.mean([H[i] for i in ini], axis=0)
    se = {"E": (E - y) ** 2, "E_h": (Eh - y) ** 2}
    for i in ini:
        se[f"v2 i{i}"] = (C[i] - y) ** 2
        se[f"v2h i{i}"] = (H[i] - y) ** 2
    for a, b in itertools.combinations(ini, 2):
        se[f"E2 {a}+{b}"] = ((C[a] + C[b]) / 2 - y) ** 2
        se[f"E2h {a}+{b}"] = ((H[a] + H[b]) / 2 - y) ** 2
    for r in REFS:
        se[r] = ref[r]
    D = {k: fs(v) for k, v in se.items()}                                       # per-discharge sums
    BS = {k: W @ v for k, v in D.items()}                                       # resampled sums

    def sk(a, b):
        return [float(1 - D[a].sum() / D[b].sum()), *ci(1 - BS[a] / BS[b])]

    o = {"rows": int(len(y)), "discharges": int(n), "initialisations": list(ini), "check": chk, **extra}
    o["E1_E_over_causal_gp"] = sk("E", "gp_causal")
    o["E2_E_over_twin_average"] = sk("E", "E_h")
    o["E1_holds"] = bool(o["E1_E_over_causal_gp"][1] > 0)
    o["E2_holds"] = bool(o["E2_E_over_twin_average"][1] > 0)
    o["E_over"] = {r: sk("E", r) for r in REFS}
    o["E_h_over_causal_gp"] = sk("E_h", "gp_causal")
    o["E_over_member"] = {str(i): sk("E", f"v2 i{i}") for i in ini}
    mem_b = np.mean([1 - BS["E"] / BS[f"v2 i{i}"] for i in ini], axis=0)
    o["E_over_member_mean_of_three"] = [float(np.mean([o["E_over_member"][str(i)][0] for i in ini])), *ci(mem_b)]
    # gain from averaging on the scale of the skill over the causal GP: skill(E) - mean_i skill(member i)
    g_pt = (np.mean([D[f"v2 i{i}"].sum() for i in ini]) - D["E"].sum()) / D["gp_causal"].sum()
    g_b = (np.mean([BS[f"v2 i{i}"] for i in ini], axis=0) - BS["E"]) / BS["gp_causal"]
    o["E_over_gp_minus_mean_member_over_gp"] = [float(g_pt), *ci(g_b)]
    o["member_over_causal_gp_cpu"] = {f"v2 i{i}": sk(f"v2 i{i}", "gp_causal") for i in ini}
    o["member_over_causal_gp_cpu"].update({f"v2h i{i}": sk(f"v2h i{i}", "gp_causal") for i in ini})
    o["member_over_causal_gp_stored"] = {f"v2 i{i}": float(1 - st[f"v2 i{i}"]["se_model"].sum() / ref["gp_causal"].sum()) for i in ini}
    o["b39_single_training"] = {"per_initialisation_over_gp": {str(i): b39["A"][str(s)]["per_initialisation"][str(i)]["causal_over_gp"] for i in ini},
                                "S1": b39["A"][str(s)]["S1_causal_over_gp"], "S2": b39["A"][str(s)]["S2_causal_over_twin"]}
    o["two_member"] = {}
    for a, b in itertools.combinations(ini, 2):
        o["two_member"][f"{a}+{b}"] = {"over_causal_gp": sk(f"E2 {a}+{b}", "gp_causal"),
                                       "over_twin_average_of_same_two": sk(f"E2 {a}+{b}", f"E2h {a}+{b}")}
    tm = o["two_member"].values()
    o["two_member_range"] = {"over_causal_gp": [min(x["over_causal_gp"][0] for x in tm), max(x["over_causal_gp"][0] for x in tm)],
                             "over_twin_average": [min(x["over_twin_average_of_same_two"][0] for x in tm),
                                                   max(x["over_twin_average_of_same_two"][0] for x in tm)]}
    # absolute errors: median, 90th percentile, ratio of E to the causal GP at each
    ab = {"E": np.sqrt(se["E"]), "E_h": np.sqrt(se["E_h"]), "hold": np.sqrt(se["persistence"]),
          "causal GP": np.sqrt(se["gp_causal"])}
    o["abs_error_eV"] = {}
    qb = {}
    for k, v in ab.items():
        qb[k] = {q: wquant(v, inv, W, q) for q in (0.5, 0.9)}
        o["abs_error_eV"][k] = {"median": [float(np.median(v)), *ci(qb[k][0.5])],
                                "p90": [float(np.percentile(v, 90)), *ci(qb[k][0.9])]}
    o["ratio_E_to_causal_gp"] = {
        "median": [o["abs_error_eV"]["E"]["median"][0] / o["abs_error_eV"]["causal GP"]["median"][0], *ci(qb["E"][0.5] / qb["causal GP"][0.5])],
        "p90": [o["abs_error_eV"]["E"]["p90"][0] / o["abs_error_eV"]["causal GP"]["p90"][0], *ci(qb["E"][0.9] / qb["causal GP"][0.9])]}
    o["rmse_eV"] = {k: float(np.sqrt(se[k].mean())) for k in ["E", "E_h", *[f"v2 i{i}" for i in ini], *[f"v2h i{i}" for i in ini], *REFS]}
    log(f"split {s}: E1 {o['E1_E_over_causal_gp']}  E2 {o['E2_E_over_twin_average']}")
    return o


def expectations(sp):
    S = [str(s) for s in SEEDS]
    inr = lambda x, lo, hi: lo <= x <= hi                                       # noqa: E731
    items = []

    def per_split(text, key_fn, lo, hi, need_all=True):
        vals = {s: key_fn(sp[s]) for s in S}
        k = sum(inr(v, lo, hi) for v in vals.values())
        items.append({"expectation": text, "values": vals, "in_range": int(k), "met": bool(k == 4 if need_all else k >= 3)})

    per_split("E over the causal GP: +0.08 to +0.16 (each split)", lambda o: o["E1_E_over_causal_gp"][0], 0.08, 0.16)
    per_split("E over the causal GP above the mean of its members by +0.005 to +0.03 on each split",
              lambda o: o["E_over_gp_minus_mean_member_over_gp"][0], 0.005, 0.03)
    n1 = sum(sp[s]["E1_holds"] for s in S)
    items.append({"expectation": "E1 holds on three or four splits", "values": n1, "met": bool(n1 >= 3)})
    per_split("E over E_h: +0.04 to +0.10 (each split)", lambda o: o["E2_E_over_twin_average"][0], 0.04, 0.10)
    n2 = sum(sp[s]["E2_holds"] for s in S)
    items.append({"expectation": "E2 holds on three or four splits", "values": n2, "met": bool(n2 >= 3)})
    per_split("E_h over the causal GP: +0.02 to +0.08 (each split)", lambda o: o["E_h_over_causal_gp"][0], 0.02, 0.08)
    per_split("ratio of the median absolute error of E to that of the causal GP: 0.97 to 1.10 (each split)",
              lambda o: o["ratio_E_to_causal_gp"]["median"][0], 0.97, 1.10)
    return items


CHOICES = [
    "Scored rows: `CES_TI_row` is the index into the per-file array of dataset.py (grid rows with at least one finite "
    "target, `compact_index` of B.45); the estimate of a member at a scored row is its output at that grid row. The grid "
    "T_i at the mapped row equals the stored `y_true` within the tolerance printed under the checks.",
    "Errors of E, E_h, the members and the two-member averages are computed against the stored `y_true`; the "
    "references (hold, causal GP, PCHIP, offline GP) are the stored squared errors of the same rows.",
    "Each network runs with the standardization recorded in its own metrics.json (causal variant: statistics of the "
    "training discharges; twin: `per_shot_input_norm` as recorded, which it does not read since it sees no fast channel).",
    "Members are compared with E using their CPU estimates (the same arithmetic as E); the stored GPU errors of the "
    "members give the B.39 single-training values, listed beside them.",
    "\"E over each of its members and the mean of those three\": the mean of the three skills of E over a member, with "
    "the interval of that mean over the same resamples. The expectation \"above the mean of its members by +0.005 to "
    "+0.03\" is read as skill(E over the causal GP) minus the mean of the three skills of the members over the causal "
    "GP, with its own interval.",
    "Resamples: discharges drawn with replacement, 2,000 times, from `default_rng(20261118 + split seed)`; the same "
    "draws for every quantity of a split. Percentiles under resampling are weighted quantiles (inverted CDF, weight = "
    "the number of times the discharge is drawn); the point values are the plain median and 90th percentile.",
    "An expectation stated as a range is counted as met when the point value lies in the range on all four splits; "
    "the counts of splits in range are given.",
]


def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def render(res):
    S = [str(s) for s in SEEDS]
    sp = res["splits"]
    v = res["verdicts"]
    L = ["# B.49 — the average of three initialisations as the estimator", "",
         "Registered: PREREGISTRATION_B49.md, after S1 and S2 of B.39 had failed and before any of these readouts was "
         "computed. CPU only; nothing is trained. T_i, cut population, the four file-level splits, the test rows of "
         "`eval_seq`. E is the mean of the T_i estimates of the three initialisations of the causal variant of B.39 at "
         "each scored row; E_h is the same mean for the three initialisations of the history-only twin. Hold, the causal "
         "GP, PCHIP and the offline GP are the stored errors of the same rows. Skill of A over R = 1 - sum of squared "
         "errors of A over that of R on the same rows. Intervals: 95 %, 2,000 resamples of the test discharges; no "
         "initialisation is drawn, since E is one estimator. A statement holds on a split when the whole interval is "
         "above zero, and holds when it does on at least three of the four splits. These statements concern an estimator "
         "other than the single training of S1 and S2 and were registered after those had failed.", "",
         "## Checks before the readout", "",
         "Each member was run forward on the CPU over the test discharges; its absolute error at every scored row is "
         "compared with the square root of its stored squared error (tolerance 0.5 eV, required share 99 %). All six "
         "members of a split were scored on the same rows with the same `y_true`, and the stored reference errors agree.", "",
         "| split | member | run | rows | share within 0.5 eV | largest deviation (eV) |", "|---|---|---|---|---|---|"]
    for s in S:
        o = sp[s]
        for k, c in o["check"].items():
            v_, i_ = k.split(" i")
            run = (B39.split_run_dir("v2", int(s), int(i_)) if v_ == "v2" else B39.twin_dir(int(s), int(i_))).name
            L.append(f"| {s} | {k} | `{run}` | {c['rows']:,} | {c['share_within_0.5_eV']:.4f} | {c['max_dev_eV']:.3f} |")
    L += ["", "Largest difference between the grid T_i at the mapped row and the stored `y_true` (eV): " +
          ", ".join(f"split {s} {sp[s]['y_grid_max_dev_eV']:.2g}" for s in S) + ".", "",
          "## Registered statements", "",
          f"- **E1** (E over the causal GP): {v['E1']['splits']} of 4 -> **{'holds' if v['E1']['holds'] else 'does not hold'}**.",
          f"- **E2** (E over the twin average E_h): {v['E2']['splits']} of 4 -> **{'holds' if v['E2']['holds'] else 'does not hold'}**.", "",
          "| split | rows | discharges | initialisations | E1: E over causal GP | interval above zero | E2: E over E_h | interval above zero |",
          "|---|---|---|---|---|---|---|---|"]
    for s in S:
        o = sp[s]
        L.append(f"| {s} | {o['rows']:,} | {o['discharges']} | {', '.join(map(str, o['initialisations']))} | {s3(o['E1_E_over_causal_gp'])} | "
                 f"{'yes' if o['E1_holds'] else 'no'} | {s3(o['E2_E_over_twin_average'])} | {'yes' if o['E2_holds'] else 'no'} |")
    L += ["", "## E beside the single trainings of B.39", "",
          "Single-training values from `docs/paper/b39_primary.json` (stored GPU errors, point values; S1 and S2 with the "
          "registered interval that also draws the initialisation).", "",
          "| split | each member over causal GP (B.39) | S1 (B.39) | E over causal GP | S2 (B.39) | E over E_h |", "|---|---|---|---|---|---|"]
    for s in S:
        o = sp[s]
        b = o["b39_single_training"]
        pm = "; ".join(f"i{i}: {x:+.3f}" for i, x in b["per_initialisation_over_gp"].items())
        L.append(f"| {s} | {pm} | {s3(b['S1'])} | {s3(o['E1_E_over_causal_gp'])} | {s3(b['S2'])} | {s3(o['E2_E_over_twin_average'])} |")
    L += ["", "## Reported", "", "E over the references:", "",
          "| split | over hold | over PCHIP | over the offline GP | E_h over causal GP |", "|---|---|---|---|---|"]
    for s in S:
        o = sp[s]
        L.append(f"| {s} | {s3(o['E_over']['persistence'])} | {s3(o['E_over']['pchip'])} | {s3(o['E_over']['gp'])} | {s3(o['E_h_over_causal_gp'])} |")
    L += ["", "Gain from averaging (members on the CPU, same arithmetic as E):", "",
          "| split | E over each member | mean of those three | skill of E over causal GP minus mean skill of the members over causal GP | members over causal GP (CPU): causal variant / twin |",
          "|---|---|---|---|---|"]
    for s in S:
        o = sp[s]
        ini = o["initialisations"]
        em = "; ".join(f"i{i}: {s3(o['E_over_member'][str(i)])}" for i in ini)
        mm = "; ".join(f"i{i}: {o['member_over_causal_gp_cpu'][f'v2 i{i}'][0]:+.3f} / {o['member_over_causal_gp_cpu'][f'v2h i{i}'][0]:+.3f}" for i in ini)
        L.append(f"| {s} | {em} | {s3(o['E_over_member_mean_of_three'])} | {s3(o['E_over_gp_minus_mean_member_over_gp'])} | {mm} |")
    L += ["", "Averages of two members (the dependence on which trainings enter):", "",
          "| split | pair | average of two causal-variant members over causal GP | over the average of the two twins of the same initialisations |", "|---|---|---|---|"]
    for s in S:
        for p, x in sp[s]["two_member"].items():
            L.append(f"| {s} | i{p.replace('+', ' + i')} | {s3(x['over_causal_gp'])} | {s3(x['over_twin_average_of_same_two'])} |")
    L += ["", "| split | range over the three pairs: over causal GP | over the twin average |", "|---|---|---|"]
    for s in S:
        r = sp[s]["two_member_range"]
        L.append(f"| {s} | {r['over_causal_gp'][0]:+.3f} to {r['over_causal_gp'][1]:+.3f} | {r['over_twin_average'][0]:+.3f} to {r['over_twin_average'][1]:+.3f} |")
    L += ["", "Absolute error (eV; point with 95 % interval over the resamples):", "",
          "| split | median: E | E_h | hold | causal GP | ratio E / causal GP | 90th percentile: E | E_h | hold | causal GP | ratio E / causal GP |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    f1 = lambda x: f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"                        # noqa: E731
    f3 = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"                        # noqa: E731
    for s in S:
        a = sp[s]["abs_error_eV"]
        rr = sp[s]["ratio_E_to_causal_gp"]
        L.append(f"| {s} | " + " | ".join(f1(a[k]["median"]) for k in ("E", "E_h", "hold", "causal GP")) + f" | {f3(rr['median'])} | " +
                 " | ".join(f1(a[k]["p90"]) for k in ("E", "E_h", "hold", "causal GP")) + f" | {f3(rr['p90'])} |")
    L += ["", "Root-mean-square error (eV):", "", "| split | E | E_h | causal-variant members | twin members | hold | causal GP | PCHIP | offline GP |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        o = sp[s]
        r = o["rmse_eV"]
        ini = o["initialisations"]
        L.append(f"| {s} | {r['E']:.1f} | {r['E_h']:.1f} | " + ", ".join(f"{r[f'v2 i{i}']:.1f}" for i in ini) + " | " +
                 ", ".join(f"{r[f'v2h i{i}']:.1f}" for i in ini) +
                 f" | {r['persistence']:.1f} | {r['gp_causal']:.1f} | {r['pchip']:.1f} | {r['gp']:.1f} |")
    L += ["", "## Expectations of section 6", "", "| expectation | values | met |", "|---|---|---|"]
    for e in res["expectations"]:
        vals = e["values"]
        vs = ", ".join(f"{s}: {x:+.3f}" if abs(x) < 0.5 else f"{s}: {x:.3f}" for s, x in vals.items()) if isinstance(vals, dict) else f"{vals} of 4"
        if isinstance(vals, dict):
            vs += f" ({e['in_range']} of 4 in range)"
        L.append(f"| {e['expectation']} | {vs} | {'met' if e['met'] else 'missed'} |")
    L += ["", "## Choices made where the registration left room", ""] + [f"- {c}" for c in res["choices"]]
    L += ["", "## Deviations", "", "- None from the registered design."]
    L += ["", "Limits stated in advance (section 7): three members; the test rows were read by every earlier readout; the "
          "members were trained independently, not as an ensemble.", "", f"Run time: {res['runtime_s']:.0f} s."]
    return "\n".join(L) + "\n"


def main():
    N = Nets()
    log(f"grid loaded: {len(N.names)} files, {len(N.valid)} with a target")
    b39 = json.loads(B39_JSON.read_text(encoding="utf-8"))
    res = {"registration": REG, "seed": SEED, "n_boot": N_BOOT, "splits": {}}
    for s in SEEDS:
        res["splits"][str(s)] = run_split(N, s, b39)
    S = [str(s) for s in SEEDS]
    res["verdicts"] = {}
    for k in ("E1", "E2"):
        nh = sum(res["splits"][s][k + "_holds"] for s in S)
        res["verdicts"][k] = {"splits": int(nh), "holds": bool(nh >= 3)}
    res["expectations"] = expectations(res["splits"])
    res["choices"] = CHOICES
    res["runtime_s"] = time.time() - T0
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"verdicts {res['verdicts']}; written {OUT_MD.name}")


if __name__ == "__main__":
    main()
