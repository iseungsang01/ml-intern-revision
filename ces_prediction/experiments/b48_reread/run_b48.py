"""B.48 (PREREGISTRATION_B48.md): the row-level readouts of B.41, B.42, B.44 and B.30 (k = 5) repeated with
the estimator of B.39 (causal variant, 80-epoch cap, initialisation = split seed) in place of the 30-epoch one.

CPU only; nothing is trained, frozen networks are run forward.  Nothing in the earlier runners is edited: their
functions are imported and pointed at the new row table / the new network by overriding module attributes here.

Stages (run from the repo root, in this order):
    py ces_prediction/experiments/b48_reread/run_b48.py rows      # B.30 machinery at k = 1, 5; checks (a), (b)
    py ces_prediction/experiments/b48_reread/run_b48.py delay     # B.42 machinery at L = 0, 1, 2, 4; check (c)
    py ces_prediction/experiments/b48_reread/run_b48.py readout   # B.41, B.42, B.44, B.30 k = 5 readouts + figure
    py ces_prediction/experiments/b48_reread/run_b48.py all
Cache: data/.b48/ (gitignored).  Output: docs/paper/b48_reread.json / .md, docs/paper/b48_events_fig.{pdf,png,json}
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the GPU belongs to the running B.39 batch

import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for sub in ("b30_thinned", "b37_plan", "b34_error_model", "b31_bursts", "b32_two_sources", "b33_clusters",
            "b41_events", "b42_delay", "b44_strata"):
    sys.path.insert(0, str(EXP / sub))
import run_b30 as R  # noqa: E402
import run_b37 as R37  # noqa: E402
import run_b41 as B41  # noqa: E402
import run_b42 as B42  # noqa: E402
import run_b44 as B44  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_B48.md"
NEW = ".b39_v2_cut_s{s}_i{s}"
TWIN = ".b29a1_v2h_cut_s{s}"
OLD_ROWS = R.DATA / ".b30"
CACHE = R.DATA / ".b48"
RAW_ROWS = CACHE / "b30_machinery"
DELAY = CACHE / "b42"
PAPER = R.REPO / "docs" / "paper"
OUT_JSON = PAPER / "b48_reread.json"
OUT_MD = PAPER / "b48_reread.md"
FIG = PAPER / "b48_events_fig"
KS = (1, 5)
TOL_EQ = 1e-3            # eV / km/s: "equal" for columns recomputed on the CPU and stored as float32
CHECK_EV, CHECK_SHARE = 0.5, 0.99
NETS_ORIG = {p: dict(v) for p, v in R.NETS.items()}
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


# at most 4 threads (R.imports sets 1; the GPU batch needs the rest of the CPU for data loading)
_imports_orig = R.imports


def _imports4():
    out = _imports_orig()
    out[0].set_num_threads(4)
    return out


R.imports = _imports4


def check_new_metrics(s):
    run = R.DATA / NEW.format(s=s)
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    bad = (m["seq_model"] != "v2" or bool(m["per_shot_input_norm"]) or m.get("per_shot_norm_expanding")
           or m.get("zero_mc") or m.get("extra_vt_channels") or m.get("extra_ti_channels") or m.get("train_ctx")
           or int(m.get("seed", -1)) != s or int(m.get("init_seed", -1)) != s)
    if bad:
        raise SystemExit(f"FATAL: {run.name} is not the causal variant with initialisation = split seed")
    return {"run": run.name, "epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"], "n_params": m["n_params"]}


def load_tab(path, t="CES_TI"):
    z = np.load(path)
    return {c.split("__")[1]: z[c] for c in z.files if c.startswith(t + "__")}


# ------------------------------------------------------------------------------------------ rows
def stage_rows():
    """The B.30 row machinery (run_b30.rows, imported, not re-derived) at k = 1 and 5 with two networks:
    the new estimator (passed under the name `bb`, because run_b30.rows checks the network called `bb` against
    the stored errors of the run named in CONTROLS -- here the run of the new estimator, i.e. check (b)) and
    the twin (recomputed for check (a)).  The final table takes every column from the frozen table except
    `causal`, which is the new estimator."""
    CACHE.mkdir(exist_ok=True)
    RAW_ROWS.mkdir(exist_ok=True)
    checks = {}
    for s in R.SEEDS:
        meta = check_new_metrics(s)
        raw = RAW_ROWS / f"rows_cut_s{s}.npz"
        if not raw.exists():
            saved = (R.ROWS_DIR, R.KS, dict(R.NETS["cut"]), R.CONTROLS["cut"])
            R.ROWS_DIR, R.KS = RAW_ROWS, KS
            R.NETS["cut"] = {"bb": (NEW, "v2", False), "hist80": (TWIN, "v2h", True)}
            R.CONTROLS["cut"] = (NEW, ".b1_w2cut_split_s{s}")
            try:
                R.rows("cut", s)              # raises SystemExit if check (b) fails
            finally:
                R.ROWS_DIR, R.KS = saved[0], saved[1]
                R.NETS["cut"] = saved[2]
                R.CONTROLS["cut"] = saved[3]
        chk_b = json.loads((RAW_ROWS / f"check_cut_s{s}.json").read_text())
        zn, zo = np.load(raw), np.load(OLD_ROWS / f"rows_cut_s{s}.npz")
        out, ca = {}, {"max_abs_diff": {}, "nan_mismatch": {}, "rows": {}}
        ok = True
        for t in R.TARGETS:
            ko = np.isin(zo[f"{t}__k"], KS)
            ca["rows"][t] = int(ko.sum())
            if int(ko.sum()) != len(zn[f"{t}__k"]):
                raise SystemExit(f"FATAL: s{s} {t}: {len(zn[f'{t}__k'])} rows rebuilt, {int(ko.sum())} in the frozen table")
            for c in ("file", "k", "o", "row", "has_future"):
                if not np.array_equal(zo[f"{t}__{c}"][ko], zn[f"{t}__{c}"]):
                    raise SystemExit(f"FATAL: s{s} {t}: column {c} is not the frozen one")
            for c in ("y", "age_ms", "hold", "gp", "gp_unscaled", "pchip", "hist80"):
                a = zo[f"{t}__{c}"][ko].astype(np.float64)
                b = zn[f"{t}__{c}"].astype(np.float64)
                nm = int((np.isnan(a) != np.isnan(b)).sum())
                fin = np.isfinite(a) & np.isfinite(b)
                d = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
                ca["max_abs_diff"][f"{t}|{c}"] = d
                ca["nan_mismatch"][f"{t}|{c}"] = nm
                ok &= (nm == 0 and d <= TOL_EQ)
            for c in zo.files:
                if c.startswith(t + "__"):
                    out[c] = zo[c][ko]
            out[f"{t}__causal"] = zn[f"{t}__bb"].astype(np.float32)
        ca["passes"] = bool(ok)
        # check (b) restated: share of the stored scored T_i rows of the run reproduced within 0.5 eV
        chk_b["passes"] = bool(chk_b["CES_TI"]["model_abs"] >= CHECK_SHARE)
        checks[str(s)] = {"network": meta, "a": ca, "b": chk_b}
        log(f"s{s}: check (a) {'passes' if ok else 'FAILS'} {ca['max_abs_diff']}; check (b) T_i model_abs "
            f"{chk_b['CES_TI']['model_abs']:.4f}")
        if not ok:
            (CACHE / "checks_rows.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
            raise SystemExit(f"FATAL: check (a) fails on split {s}; the run stops")
        if not chk_b["passes"]:
            raise SystemExit(f"FATAL: check (b) fails on split {s}; the run stops")
        np.savez_compressed(CACHE / f"rows_cut_s{s}.npz", **out)
    (CACHE / "checks_rows.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------------------ delay
class Ctx48(R37.Ctx):
    """run_b37.Ctx with `net` pointed at the estimator of B.39 (the loader otherwise unchanged)."""

    def net(self, s):
        check_new_metrics(s)
        run = R.DATA / NEW.format(s=s)
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        stats = self.load_stats(m)
        model = self.MODELS["v2"]()
        model.load_state_dict(self.torch.load(run / "weights" / "seq_lstm.pth", map_location="cpu"))
        model.eval()
        return model, stats


def stage_delay():
    DELAY.mkdir(parents=True, exist_ok=True)
    R.ROWS_DIR = CACHE                              # table_check reads the new table
    C = Ctx48()
    checks = {}
    for s in R.SEEDS:
        p = DELAY / f"rows_s{s}.npz"
        if p.exists():
            z = np.load(p)
            rows = {k: z[k] for k in z.files}
            inp = json.loads((DELAY / f"input_s{s}.json").read_text())
        else:
            rows = B42.rows_split(C, s)
            inp = rows.pop("input_check_files")
            (DELAY / f"input_s{s}.json").write_text(json.dumps(inp))
        tc = B42.table_check(C, s, rows)            # the B.42 check (five files), now against the new table
        # check (c): every k = 1 row of the new table
        tab = load_tab(CACHE / f"rows_cut_s{s}.npz")
        k1 = tab["k"] == 1
        mine = {(int(f), int(r)): j for j, (f, r) in enumerate(zip(rows["L0__file"], rows["L0__row"]))}
        dev = {"causal": 0, "hold": 0, "gp": 0, "twin": 0, "missing": 0}
        maxd = {"causal": 0.0, "hold": 0.0, "gp": 0.0, "twin": 0.0}
        n = 0
        for f, r, c, h, g, tw in zip(tab["file"][k1], tab["row"][k1], tab["causal"][k1], tab["hold"][k1],
                                     tab["gp"][k1], tab["hist80"][k1]):
            n += 1
            j = mine.get((int(f), int(r)))
            if j is None:
                dev["missing"] += 1
                continue
            for e, v in (("causal", c), ("hold", h), ("gp", g), ("twin", tw)):
                d = abs(float(rows[f"L0__{e}"][j]) - float(v))
                if not d <= CHECK_EV:
                    dev[e] += 1
                if np.isfinite(d):
                    maxd[e] = max(maxd[e], d)
        share = 1 - (dev["causal"] + dev["missing"]) / max(n, 1)
        # the twin, hold and GP at every L against the rows of the earlier B.42 run (estimator-independent)
        zo = np.load(R.DATA / ".b42" / f"rows_s{s}.npz")
        same = {}
        for L in B42.LS:
            eq = len(zo[f"L{L}__row"]) == len(rows[f"L{L}__row"]) and np.array_equal(zo[f"L{L}__row"], rows[f"L{L}__row"])
            same[str(L)] = {"same_rows": bool(eq)}
            if eq:
                for e in ("hold", "gp", "twin", "twin_nd"):
                    a, b = zo[f"L{L}__{e}"], rows[f"L{L}__{e}"]
                    fin = np.isfinite(a) & np.isfinite(b)
                    same[str(L)][e] = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        checks[str(s)] = {"input": inp, "table": tc,
                          "c": {"rows": n, "deviating": dev, "max_abs_dev_eV": maxd, "share_within_0.5_eV": share,
                                "passes": bool(n > 0 and share >= CHECK_SHARE)},
                          "against_b42_cache": same}
        log(f"s{s}: check (c) {share:.5f} of {n} rows; dev {dev}; B.42 five-file check {tc['share_within_0.5_eV']:.4f}; "
            f"vs B.42 cache {same}")
        (DELAY / "checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
        if not checks[str(s)]["c"]["passes"] or not tc["passes"]:
            raise SystemExit(f"FATAL: check (c) fails on split {s}; the run stops")
        if not p.exists():
            np.savez_compressed(p, **rows)


# ------------------------------------------------------------------------------------------ readout
def readout_b41():
    R.ROWS_DIR = CACHE
    B41.OUT_JSON, B41.OUT_MD, B41.OUT_FIG = CACHE / "b41.json", CACHE / "b41.md", FIG.with_suffix(".json")
    B41.main()
    fig = json.loads(B41.OUT_FIG.read_text(encoding="utf-8"))
    fig["note"] = "data of docs/paper/b48_events_fig.pdf; written by b48_reread/run_b48.py (run_b41.main on the B.48 table)"
    B41.OUT_FIG.write_text(json.dumps(fig, indent=1), encoding="utf-8")
    return json.loads(B41.OUT_JSON.read_text(encoding="utf-8")), B41.OUT_MD.read_text(encoding="utf-8")


def readout_b42():
    checks = json.loads((DELAY / "checks.json").read_text(encoding="utf-8"))
    B42.CACHE = DELAY
    B42.OUT_JSON, B42.OUT_MD = CACHE / "b42.json", CACHE / "b42.md"
    B42.analyze({s: {"input": c["input"], "table": c["table"]} for s, c in checks.items()})
    return json.loads(B42.OUT_JSON.read_text(encoding="utf-8")), B42.OUT_MD.read_text(encoding="utf-8")


def readout_b44():
    R.ROWS_DIR = CACHE
    B44.OUT_JSON, B44.OUT_MD = CACHE / "b44.json", CACHE / "b44.md"
    B44.T0 = _time.time()
    B44.main()
    return json.loads(B44.OUT_JSON.read_text(encoding="utf-8")), B44.OUT_MD.read_text(encoding="utf-8")


PAIRS5 = [("causal", "gp"), ("causal", "hist80"), ("causal", "pchip"), ("causal", "hold")]


def readout_b30():
    """k = 1 and k = 5 cells of run_b30.cell on the B.30 population (has_future, PCHIP and every network
    finite), with the pairs of H1, H2 and the PCHIP pair; one generator seeded as run_b30.analyze."""
    R.NETS["cut"] = dict(NETS_ORIG["cut"])
    saved = list(R.PAIRS["cut"])
    R.PAIRS["cut"] = PAIRS5
    rng = np.random.default_rng(R.BOOT_SEED)
    cache = {}

    def weights(n):
        if n not in cache:
            cache[n] = rng.multinomial(n, np.full(n, 1.0 / n), size=R.N_BOOT).astype(np.float64)
        return cache[n]

    res = {}
    try:
        for s in R.SEEDS:
            tab = load_tab(CACHE / f"rows_cut_s{s}.npz")
            ests = ["hold", "gp", "pchip", *R.NETS["cut"]]
            fin = tab["has_future"] & np.all([np.isfinite(tab[e]) for e in ests], axis=0)
            res[str(s)] = {str(k): R.cell(tab, fin & (tab["k"] == k), "cut", weights) for k in KS}
    finally:
        R.PAIRS["cut"] = saved
    h1 = sum(res[str(s)]["5"]["skill"]["causal_vs_gp"][1] > 0 for s in R.SEEDS)
    h2 = sum(res[str(s)]["5"]["skill"]["causal_vs_hist80"][1] > 0 for s in R.SEEDS)
    hp = sum(res[str(s)]["5"]["skill"]["causal_vs_pchip"][1] > 0 for s in R.SEEDS)
    return {"cells": res, "H1": {"splits": int(h1), "holds": bool(h1 >= 3)},
            "H2": {"splits": int(h2), "holds": bool(h2 >= 3)},
            "pchip_reported": {"splits": int(hp)}}


def figure():
    import fig_events as FE
    FE.FIG = FIG
    FE.STYLE = dict(FE.STYLE)
    FE.STYLE["causal"] = ("recurrent estimator", *FE.STYLE["causal"][1:])
    FE.main()
    for p in (FIG.with_suffix(".pdf"), FIG.with_suffix(".png")):
        if not p.exists():
            raise SystemExit(f"FATAL: figure {p} not written")


def stage_readout():
    out = {"registration": REG, "estimator": NEW, "twin": TWIN,
           "checks": {"rows": json.loads((CACHE / "checks_rows.json").read_text(encoding="utf-8")),
                      "delay": json.loads((DELAY / "checks.json").read_text(encoding="utf-8"))}}
    log("B.41 readout")
    out["b41"], md41 = readout_b41()
    log("B.42 readout")
    out["b42"], md42 = readout_b42()
    log("B.44 readout")
    out["b44"], md44 = readout_b44()
    log("B.30 k = 5 readout")
    out["b30"] = readout_b30()
    log("figure")
    figure()
    old = {"b41": json.loads((PAPER / "b41_events.json").read_text(encoding="utf-8")),
           "b42": json.loads((PAPER / "b42_delay.json").read_text(encoding="utf-8")),
           "b44": json.loads((PAPER / "b44_strata.json").read_text(encoding="utf-8")),
           "b30": json.loads((PAPER / "b30_thinned.json").read_text(encoding="utf-8")),
           "b30_posthoc": json.loads((PAPER / "b30_posthoc.json").read_text(encoding="utf-8"))}
    out["statements"] = statements(out, old)
    out["old_vs_new"] = old_vs_new(out, old)
    out["expectations"] = expectations(out)
    out["runtime_readout_s"] = round(_time.time() - T0, 1)
    rt = json.loads((CACHE / "runtime.json").read_text()) if (CACHE / "runtime.json").exists() else {}
    out["runtime_rows_s"], out["runtime_delay_s"] = rt.get("rows", "n/a"), rt.get("delay", "n/a")
    out["deviations_text"] = DEVIATIONS
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (CACHE / "embedded_md.json").write_text(json.dumps({"b41": md41, "b42": md42, "b44": md44}), encoding="utf-8")
    OUT_MD.write_text(render(out, md41, md42, md44), encoding="utf-8", newline="\n")
    log("written " + str(OUT_MD))


# ------------------------------------------------------------------------------------------ statements
def statements(o, old):
    S = [str(s) for s in R.SEEDS]
    st = {}
    G, Go = o["b41"]["G"], old["b41"]["G"]
    for key, field, rule in (("G1", ("plain", "share_of_reduction_vs_twin"), "lo>0.5"),
                             ("G2", ("plain", "share_of_reduction_vs_gp"), "lo>0.5")):
        st[key] = {"new": [G[s][field[0]][field[1]] for s in S], "old": [Go[s][field[0]][field[1]] for s in S],
                   "n": int(o["b41"]["verdicts"][key]["splits"]), "holds": bool(o["b41"]["verdicts"][key]["holds"]),
                   "n_old": int(old["b41"]["verdicts"][key]["splits"]), "holds_old": bool(old["b41"]["verdicts"][key]["holds"])}
    st["Q1"] = {"new": {"gp": [G[s]["Q1_p90_ratio_vs_gp"] for s in S], "hold": [G[s]["Q1_p90_ratio_vs_hold"] for s in S]},
                "old": {"gp": [Go[s]["Q1_p90_ratio_vs_gp"] for s in S], "hold": [Go[s]["Q1_p90_ratio_vs_hold"] for s in S]},
                "n": int(o["b41"]["verdicts"]["Q1"]["splits"]), "holds": bool(o["b41"]["verdicts"]["Q1"]["holds"]),
                "n_old": int(old["b41"]["verdicts"]["Q1"]["splits"]), "holds_old": bool(old["b41"]["verdicts"]["Q1"]["holds"])}
    P, Po = o["b42"]["splits"], old["b42"]["splits"]
    st["T1"] = {"new": {L: [P[s][L]["skill"]["causal_vs_gp"] for s in S] for L in ("1", "2")},
                "old": {L: [Po[s][L]["skill"]["causal_vs_gp"] for s in S] for L in ("1", "2")},
                "n": o["b42"]["verdicts"]["T1"]["splits_above_0"], "holds": o["b42"]["verdicts"]["T1"]["holds"],
                "n_old": old["b42"]["verdicts"]["T1"]["splits_above_0"], "holds_old": old["b42"]["verdicts"]["T1"]["holds"]}
    st["T2"] = {"new": [P[s]["2"]["skill"]["causal_vs_twin"] for s in S],
                "old": [Po[s]["2"]["skill"]["causal_vs_twin"] for s in S],
                "n": o["b42"]["verdicts"]["T2"]["splits_above_0"], "holds": o["b42"]["verdicts"]["T2"]["holds"],
                "n_old": old["b42"]["verdicts"]["T2"]["splits_above_0"], "holds_old": old["b42"]["verdicts"]["T2"]["holds"]}
    for w, ref in (("W1", "hist80"), ("W2", "gp")):
        th = lambda d: {s: {t: d["splits"][s]["strata"]["median_ti"]["thirds"][t]["skill_causal_vs"][ref]   # noqa: E731
                            for t in B44.THIRDS} for s in S}
        st[w] = {"new": th(o["b44"]), "old": th(old["b44"]),
                 "per_split_new": {s: bool(o["b44"]["splits"][s][f"{w}_holds"]) for s in S},
                 "per_split_old": {s: bool(old["b44"]["splits"][s][f"{w}_holds"]) for s in S},
                 "n": o["b44"]["verdicts"][w]["splits"], "holds": o["b44"]["verdicts"][w]["holds"],
                 "n_old": old["b44"]["verdicts"][w]["splits"], "holds_old": old["b44"]["verdicts"][w]["holds"]}
    C = o["b30"]["cells"]
    oc = old["b30"]["cells"]
    oph = old["b30_posthoc"]["causal_share"]
    st["H1"] = {"new": [C[s]["5"]["skill"]["causal_vs_gp"] for s in S],
                "old": [oc[f"cut|s{s}|CES_TI"]["by_k"]["5"]["skill"]["causal_vs_gp"] for s in S],
                "n": o["b30"]["H1"]["splits"], "holds": o["b30"]["H1"]["holds"]}
    st["H1"]["n_old"] = int(sum(x[1] > 0 for x in st["H1"]["old"]))
    st["H1"]["holds_old"] = bool(st["H1"]["n_old"] >= 3)
    st["H2"] = {"new": [C[s]["5"]["skill"]["causal_vs_hist80"] for s in S],
                "old": [oph[f"cut|s{s}"]["5"]["causal_vs_hist"] for s in S],
                "n": o["b30"]["H2"]["splits"], "holds": o["b30"]["H2"]["holds"]}
    st["H2"]["n_old"] = int(sum(x[1] > 0 for x in st["H2"]["old"]))
    st["H2"]["holds_old"] = bool(st["H2"]["n_old"] >= 3)
    st["PCHIP_k5_reported"] = {"new": [C[s]["5"]["skill"]["causal_vs_pchip"] for s in S],
                               "old": [oph[f"cut|s{s}"]["5"]["causal_vs_pchip"] for s in S],
                               "n": o["b30"]["pchip_reported"]["splits"]}
    st["PCHIP_k5_reported"]["n_old"] = int(sum(x[1] > 0 for x in st["PCHIP_k5_reported"]["old"]))
    for k, v in st.items():
        if "holds" in v:
            v["verdict_differs"] = bool(v["holds"] != v["holds_old"])
    return st


def old_vs_new(o, old):
    S = [str(s) for s in R.SEEDS]
    G, Go = o["b41"]["G"], old["b41"]["G"]
    P, Po = o["b42"]["splits"], old["b42"]["splits"]
    W, Wo = o["b44"]["splits"], old["b44"]["splits"]
    C, Co = o["b30"]["cells"], old["b30_posthoc"]["causal_share"]
    rows = {
        "increment over the twin, all rows (k = 1)": ([W[s]["overall"]["hist80"] for s in S], [Wo[s]["overall"]["hist80"] for s in S]),
        "skill over the causal GP, all rows (k = 1)": ([W[s]["overall"]["gp"] for s in S], [Wo[s]["overall"]["gp"] for s in S]),
        "skill over hold, all rows (k = 1)": ([W[s]["overall"]["hold"] for s in S], [Wo[s]["overall"]["hold"] for s in S]),
        "increment over the twin, plain rows": ([G[s]["plain"]["skill_causal_vs_twin"] for s in S],
                                                [Go[s]["plain"]["skill_causal_vs_twin"] for s in S]),
        "increment over the twin, flagged rows": ([G[s]["flagged"]["skill_causal_vs_twin"] for s in S],
                                                  [Go[s]["flagged"]["skill_causal_vs_twin"] for s in S]),
        "p90 ratio over the causal GP": ([G[s]["Q1_p90_ratio_vs_gp"] for s in S], [Go[s]["Q1_p90_ratio_vs_gp"] for s in S]),
        "p90 ratio over hold": ([G[s]["Q1_p90_ratio_vs_hold"] for s in S], [Go[s]["Q1_p90_ratio_vs_hold"] for s in S]),
        "p90 ratio over the twin": ([G[s]["ratio_causal_over"]["hist80"]["90"] for s in S],
                                    [Go[s]["ratio_causal_over"]["hist80"]["90"] for s in S]),
        "increment over the twin, rows 1 to 3 after an absent frame": (
            [o["b41"]["B"][s]["skill_causal_vs_twin_rows_1-3"] for s in S],
            [old["b41"]["B"][s]["skill_causal_vs_twin_rows_1-3"] for s in S]),
        "skill over the causal GP at L = 2": ([P[s]["2"]["skill"]["causal_vs_gp"] for s in S],
                                              [Po[s]["2"]["skill"]["causal_vs_gp"] for s in S]),
        "increment over the twin at L = 2": ([P[s]["2"]["skill"]["causal_vs_twin"] for s in S],
                                             [Po[s]["2"]["skill"]["causal_vs_twin"] for s in S]),
        "increment over the twin at L = 0": ([P[s]["0"]["skill"]["causal_vs_twin"] for s in S],
                                             [Po[s]["0"]["skill"]["causal_vs_twin"] for s in S]),
        "skill over the causal GP at k = 5": ([C[s]["5"]["skill"]["causal_vs_gp"] for s in S],
                                              [Co[f"cut|s{s}"]["5"]["causal_vs_gp"] for s in S]),
        "increment over the twin at k = 5": ([C[s]["5"]["skill"]["causal_vs_hist80"] for s in S],
                                             [Co[f"cut|s{s}"]["5"]["causal_vs_hist"] for s in S]),
    }
    for t in B44.THIRDS:
        rows[f"increment over the twin, {t} third of the median T_i"] = (
            [W[s]["strata"]["median_ti"]["thirds"][t]["skill_causal_vs"]["hist80"] for s in S],
            [Wo[s]["strata"]["median_ti"]["thirds"][t]["skill_causal_vs"]["hist80"] for s in S])
    return {k: {"new": v[0], "old": v[1], "new_minus_old": [float(a[0] - b[0]) for a, b in zip(*v)]}
            for k, v in rows.items()}


def expectations(o):
    st, ov = o["statements"], o["old_vs_new"]
    E = []
    want = {"G1": False, "G2": False, "Q1": True, "T1": True, "T2": True, "W1": False, "W2": False, "H1": True, "H2": True}
    for k, w in want.items():
        E.append({"expectation": f"{k} {'holds' if w else 'does not hold'}",
                  "observed": f"{k} {'holds' if st[k]['holds'] else 'does not hold'} ({st[k]['n']} of 4)",
                  "met": bool(st[k]["holds"] == w)})
    for key in ("increment over the twin, all rows (k = 1)", "increment over the twin, plain rows",
                "increment over the twin at L = 2", "increment over the twin at k = 5"):
        d = ov[key]["new_minus_old"]
        E.append({"expectation": f"{key}: within 0.03 of the earlier readout on each split",
                  "observed": ", ".join(f"{x:+.3f}" for x in d), "met": bool(all(abs(x) <= 0.03 for x in d))})
    v = [x[0] for x in ov["p90 ratio over the causal GP"]["new"]]
    E.append({"expectation": "90th-percentile ratio to the causal GP 0.90 to 0.95",
              "observed": ", ".join(f"{x:.3f}" for x in v), "met": bool(all(0.90 <= x <= 0.95 for x in v))})
    v = [x[0] for x in ov["skill over the causal GP at L = 2"]["new"]]
    E.append({"expectation": "skill over the causal GP at L = 2 +0.07 to +0.17",
              "observed": ", ".join(f"{x:+.3f}" for x in v), "met": bool(all(0.07 <= x <= 0.17 for x in v))})
    return E


# ------------------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def f3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def hs(h):
    return "holds" if h else "does not hold"


def ex0(x):
    return "excl. 0" if not (x[1] <= 0 <= x[2]) else "incl. 0"


def ex1(x):
    return "excl. 1" if not (x[1] <= 1 <= x[2]) else "incl. 1"


CHOICES = [
    "Row table: `run_b30.rows` is called with `KS = (1, 5)`, two networks and `CONTROLS['cut']` pointed at the run of "
    "the new estimator. The new estimator is passed under the network name `bb`, because `run_b30.rows` checks the "
    "network named `bb` against the stored errors of the run named in `CONTROLS` (that check is check (b)); its "
    "column is renamed `causal` in the final table. The twin is recomputed (for check (a)); `bb` and `bb80` "
    "(networks not read by any readout here) and every other column are copied from the frozen table. The rows of "
    "k = 2, 3 and 10 are omitted.",
    "Check (a): 'equal' is read as the same NaN positions and an absolute difference of at most 1e-3 eV (km/s for "
    "V_rot) on every row of k = 1 and k = 5, for y, age, hold, the causal GP (scaled and unscaled grid), PCHIP and the "
    "twin, both targets; file, k, phase, row and has_future must be identical.",
    "Check (b): the rule of `run_b30.rows` (nearest stored row by measured value within the discharge, absolute-error "
    "deviation within 0.5 eV), on the T_i rows stored by `eval_seq` for the new estimator; the stage stops below 0.99.",
    "Check (c): every k = 1 row of the new table (not only the five files of the B.42 table check), matched by "
    "(file, row) to the L = 0 rows of the B.42 machinery; the B.42 five-file check is also run and reported. In "
    "addition hold, the GP and the twin at every L are compared with the cached rows of the earlier B.42 run.",
    "Readouts: `run_b41.main`, `run_b42.analyze` and `run_b44.main` are run unchanged with `run_b30.ROWS_DIR` "
    "pointed at `data/.b48/` (and, for B.42, the cache pointed at the rows built with the new network); their "
    "outputs are written under `data/.b48/` and embedded below. Generators, seeds and resample counts are those of "
    "the earlier runners, so for B.41, B.42 and B.44 the discharge resamples are the same as in the earlier readouts "
    "(paired comparison of old and new).",
    "B.30 at k = 5: `run_b30.cell` on the B.30 population (has_future, PCHIP and every network finite) at k = 1 and 5, "
    "with the pairs causal vs causal GP (H1), vs twin (H2), vs PCHIP of the visible values and vs hold; one generator "
    "seeded as `run_b30.analyze` (20261116), weights cached by number of discharges. The earlier k = 5 numbers came "
    "from two runs (`b30_thinned.json` for the GP pair, `b30_posthoc.json` for the twin and PCHIP pairs), whose "
    "generators drew in a different order, so the resamples here are not guaranteed to be identical to those.",
    "H1 and H2 hold on a split when the lower end of the 95 % interval is above zero; the statement holds on at least "
    "three of four splits. The earlier verdicts of H1 and H2 are read from the earlier intervals with the same rule "
    "(the earlier H1 is H30-4 of B.30; the earlier H2 was an after-the-fact readout).",
    "Figure: `fig_events.main` with the output path and the legend entry of the estimator changed ('recurrent "
    "estimator'); no legend of that figure says 'persistence', so nothing else was renamed.",
    "Expectation 'increments over the twin within 0.03 of the earlier ones on each split' is read on four increments: "
    "all rows at k = 1, plain rows, L = 2 and k = 5.",
]


DEVIATIONS = "None from the registered definitions, rows, intervals, seeds or rules."


def demote(md, repl):
    out = []
    for line in md.splitlines():
        for a, b in repl:
            line = line.replace(a, b)
        out.append("##" + line if line.startswith("#") else line)
    return "\n".join(out)


def render(o, md41, md42, md44):
    S = [str(s) for s in R.SEEDS]
    st, ck = o["statements"], o["checks"]
    L = ["# B.48 -- the row-level readouts repeated with the estimator of B.39 (registered: "
         "PREREGISTRATION_B48.md)", "",
         "CPU, nothing trained; frozen networks run forward. T_i, cut population, the four file-level splits (42, 1, 7, "
         "123), test discharges. The only change from the earlier readouts (B.41, B.42, B.44, B.30 at k = 5) is the "
         "estimator: the causal variant of B.39 (model v2, statistics of the training discharges, 80-epoch cap, "
         "initialisation = split seed, `data/.b39_v2_cut_s{seed}_i{seed}`) in place of the causal variant of B.29 "
         "(30-epoch cap, `data/.b29_norm0_cut_s{seed}`). Twin = the history-only twin `data/.b29a1_v2h_cut_s{seed}` "
         "(80-epoch cap; column `hist80`); hold = last visible measured value; causal GP = the GP on the 16 last "
         "visible values; PCHIP = offline interpolation of the visible values (k = 5 only). Skill a vs b = 1 - SSE_a / "
         "SSE_b. Intervals: 95 %, discharge resamples as in each earlier registration (2,000). A statement holds if it "
         "holds on at least three of the four splits. In the embedded tables of B.41, B.42 and B.44 below, "
         "'causal variant' / 'causal' is the estimator of B.39.", "",
         "## Checks before the readout", "",
         "| split | network (epochs run, best) | (a) max abs diff, k = 1 and 5 (T_i hold / GP / GP unscaled / PCHIP / twin) | "
         "(a) NaN mismatches | (a) | (b) share within 0.5 eV (T_i) | (b) p99 dev (eV) | (b) hold / GP shares | (b) | "
         "(c) share within 0.5 eV | (c) rows | (c) max dev (eV) | (c) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        a, b, c = ck["rows"][s]["a"], ck["rows"][s]["b"], ck["delay"][s]["c"]
        n = ck["rows"][s]["network"]
        md = a["max_abs_diff"]
        L.append(f"| {s} | {n['run']} ({n['epochs_run']}, {n['best_epoch']}) | "
                 + " / ".join(f"{md['CES_TI|' + c_]:.1e}" for c_ in ("hold", "gp", "gp_unscaled", "pchip", "hist80"))
                 + f" | {sum(a['nan_mismatch'].values())} | {'passes' if a['passes'] else 'FAILS'} | "
                 f"{b['CES_TI']['model_abs']:.4f} | {b['CES_TI']['model_dev_p99']:.3f} | "
                 f"{b['CES_TI']['persistence_abs']:.4f} / {b['CES_TI']['gp_causal_abs']:.4f} | "
                 f"{'passes' if b['passes'] else 'FAILS'} | {c['share_within_0.5_eV']:.5f} | {c['rows']:,} | "
                 f"{c['max_abs_dev_eV']['causal']:.2e} | {'passes' if c['passes'] else 'FAILS'} |")
    L += ["", "V_rot columns of check (a) (max abs diff over hold, GP, GP unscaled, PCHIP, twin, km/s): "
          + "; ".join(f"split {s}: " + f"{max(v for k, v in ck['rows'][s]['a']['max_abs_diff'].items() if k.startswith('CES_VT')):.1e}"
                      for s in S) + ". Check (b) on the stored V_rot rows (reported, not a stop rule): "
          + "; ".join(f"{s}: {ck['rows'][s]['b']['CES_VT']['model_abs']:.4f}" for s in S) + ".", "",
          "B.42 machinery, further: rebuilt L = 0 inputs equal `build_blocks` on "
          + "; ".join(f"split {s}: {ck['delay'][s]['input']}" for s in S)
          + " test files; the B.42 five-file table check: "
          + "; ".join(f"{s}: {ck['delay'][s]['table']['share_within_0.5_eV']:.4f}" for s in S)
          + ". Hold, GP, twin and twin_nd at L = 0, 1, 2, 4 against the cached rows of the earlier B.42 run (same rows; "
          "largest absolute difference over the four columns): "
          + "; ".join(f"{s}: " + ", ".join(
              f"L{Lx} " + (f"{max(v for k, v in d.items() if k != 'same_rows'):.1e}" if d['same_rows'] else "rows differ")
              for Lx, d in ck['delay'][s]['against_b42_cache'].items()) for s in S) + ".", "",
          "All three registered checks pass on every split." if all(
              ck["rows"][s]["a"]["passes"] and ck["rows"][s]["b"]["passes"] and ck["delay"][s]["c"]["passes"] for s in S)
          else "A check FAILS (see above).", "",
          "## Registered statements", "",
          "Per-split values in the order 42, 1, 7, 123; `n of 4` = splits on which the statement holds; the earlier "
          "readout (30-epoch estimator) beside it.", ""]

    def line(key, desc, new_txt, old_txt):
        d = st[key]
        flag = " **Verdict differs from the earlier readout.**" if d["verdict_differs"] else ""
        L.append(f"- **{key}** ({desc}): {new_txt} -> **{hs(d['holds'])}**. "
                 f"Earlier: {old_txt} -> {hs(d['holds_old'])}.{flag}")

    for key, desc in (("G1", "share of the reduction against the twin held by the plain rows; interval above 0.5"),
                      ("G2", "the same against the causal GP")):
        d = st[key]
        line(key, desc, "; ".join(f3(x) for x in d["new"]) + f"; {d['n']} of 4",
             "; ".join(f3(x) for x in d["old"]) + f", {d['n_old']} of 4")
    d = st["Q1"]
    line("Q1", "90th percentile of the absolute error, estimator over causal GP and over hold, both intervals below 1",
         "over GP " + "; ".join(f"{f3(x)} ({ex1(x)})" for x in d["new"]["gp"]) + "; over hold "
         + "; ".join(f"{f3(x)} ({ex1(x)})" for x in d["new"]["hold"]) + f"; {d['n']} of 4",
         "over GP " + "; ".join(f3(x) for x in d["old"]["gp"]) + "; over hold "
         + "; ".join(f3(x) for x in d["old"]["hold"]) + f", {d['n_old']} of 4")
    d = st["T1"]
    line("T1", "skill over the causal GP with the interval above zero, at L = 1 and at L = 2",
         "; ".join(f"L = {Lx}: " + "; ".join(f"{s3(x)} ({ex0(x)})" for x in d["new"][Lx]) for Lx in ("1", "2"))
         + f"; L = 1 on {d['n']['1']} of 4, L = 2 on {d['n']['2']} of 4",
         "; ".join(f"L = {Lx}: " + "; ".join(s3(x) for x in d["old"][Lx]) for Lx in ("1", "2"))
         + f", L = 1 on {d['n_old']['1']}, L = 2 on {d['n_old']['2']} of 4")
    d = st["T2"]
    line("T2", "skill over the twin at L = 2, interval above zero",
         "; ".join(f"{s3(x)} ({ex0(x)})" for x in d["new"]) + f"; {d['n']} of 4",
         "; ".join(s3(x) for x in d["old"]) + f", {d['n_old']} of 4")
    for w, ref in (("W1", "the twin"), ("W2", "the causal GP")):
        d = st[w]
        line(w, f"in every third of the median T_i, skill over {ref} with the interval above zero",
             " | ".join(f"split {s}: " + "; ".join(f"{t} {s3(d['new'][s][t])}" for t in B44.THIRDS)
                        + f" ({'holds' if d['per_split_new'][s] else 'not'})" for s in S) + f"; {d['n']} of 4",
             ", ".join(f"split {s} {'holds' if d['per_split_old'][s] else 'not'}" for s in S) + f"; {d['n_old']} of 4")
    for key, desc in (("H1", "value visible every fifth frame: skill over the causal GP, interval above zero"),
                      ("H2", "the same over the twin")):
        d = st[key]
        line(key, desc, "; ".join(f"{s3(x)} ({ex0(x)})" for x in d["new"]) + f"; {d['n']} of 4",
             "; ".join(s3(x) for x in d["old"]) + f", {d['n_old']} of 4")
    d = st["PCHIP_k5_reported"]
    L.append(f"- Reported with H1 and H2 (no rule registered): k = 5, estimator over PCHIP of the visible values "
             + "; ".join(f"{s3(x)} ({ex0(x)})" for x in d["new"]) + f" (interval above zero on {d['n']} of 4); earlier "
             + "; ".join(s3(x) for x in d["old"]) + f" ({d['n_old']} of 4).")
    diff = [k for k, v in st.items() if v.get("verdict_differs")]
    L += ["", "Verdicts that differ from the earlier readouts: " + (", ".join(diff) if diff else "none") + ".", ""]

    # ---------- old vs new
    L += ["## Old estimator against new", "",
          "Per split: earlier value (30-epoch estimator, `.b29_norm0_cut`) -> new value (`.b39_v2_cut_s{s}_i{s}`), "
          "difference new minus old of the point values. Same rows, same twin, same references; for B.41, B.42 and "
          "B.44 also the same discharge resamples.", "",
          "| quantity | " + " | ".join(f"split {s}" for s in S) + " |", "|---|" + "---|" * len(S)]
    for k, v in o["old_vs_new"].items():
        fm = f3 if "ratio" in k else s3
        L.append(f"| {k} | " + " | ".join(f"{fm(a)} -> {fm(b)} ({d_:+.3f})"
                                           for a, b, d_ in zip(v["old"], v["new"], v["new_minus_old"])) + " |")
    L += ["", "## Expectations of the registration (section 6)", "", "| expectation | observed | met |", "|---|---|---|"]
    for e in o["expectations"]:
        L.append(f"| {e['expectation']} | {e['observed']} | {'yes' if e['met'] else 'no'} |")
    L += ["", f"{sum(e['met'] for e in o['expectations'])} of {len(o['expectations'])} expectations met.", ""]

    # ---------- B.30 tables
    C = o["b30"]["cells"]
    ests = ["hold", "gp", "pchip", "hist80", "causal"]
    L += ["## Full tables", "",
          "### B.30 replay at k = 1 and k = 5 (value visible every k-th frame, every phase)", "",
          "B.30 population (a later visible value exists; PCHIP and every network finite). RMSE in eV.", "",
          "| split | k | rows | discharges | " + " | ".join(f"RMSE {e}" for e in ests) + " | "
          + " | ".join(f"{a} vs {b}" for a, b in PAIRS5) + " |", "|---|---|---|---|" + "---|" * (len(ests) + len(PAIRS5))]
    for s in S:
        for k in ("1", "5"):
            c = C[s][k]
            L.append(f"| {s} | {k} | {c['rows']:,} | {c['discharges']} | " + " | ".join(f"{c['rmse'][e]:.1f}" for e in ests)
                     + " | " + " | ".join(f"{s3(c['skill'][f'{a}_vs_{b}'])} ({ex0(c['skill'][f'{a}_vs_{b}'])})" for a, b in PAIRS5) + " |")
    L += ["", "Top 1 % of the hold squared error removed (co-reported as in B.30):", "",
          "| split | k | " + " | ".join(f"{a} vs {b}" for a, b in PAIRS5) + " |", "|---|---|" + "---|" * len(PAIRS5)]
    for s in S:
        for k in ("1", "5"):
            c = C[s][k]
            L.append(f"| {s} | {k} | " + " | ".join(s3(c["skill_trim1"][f"{a}_vs_{b}"]) for a, b in PAIRS5) + " |")
    L += ["", "Median absolute error (eV):", "", "| split | k | " + " | ".join(ests) + " |", "|---|---|" + "---|" * len(ests)]
    for s in S:
        for k in ("1", "5"):
            c = C[s][k]
            L.append(f"| {s} | {k} | " + " | ".join(f"{c['median_abs'][e]:.1f}" for e in ests) + " |")
    L.append("")
    tabnote = ("B.48 row tables (`data/.b48/rows_cut_s*.npz`, column `causal` = the estimator of B.39; every other "
               "column the frozen B.30 one)")
    L += ["### B.41 readout (run_b41.main on the B.48 table)", "",
          demote(md41, [("frozen B.30 row tables", tabnote), ("docs/paper/b41_events_fig", "docs/paper/b48_events_fig")]), "",
          "### B.42 readout (run_b42 machinery with the estimator of B.39)", "",
          demote(md42, [("the causal variant (`.b29_norm0_cut`)", "the estimator of B.39 (`.b39_v2_cut_s{seed}_i{seed}`)"),
                        ("## Expectations of the registration (section 6)",
                         "## Expectations of PREREGISTRATION_B42.md section 6 (for reference; not the B.48 expectations)")]), "",
          "### B.44 readout (run_b44.main on the B.48 table)", "",
          demote(md44, [("frozen B.30 row tables", tabnote),
                        ("docs/paper/b30_thinned.json:", "docs/paper/b30_thinned.json (the B.30 column is the 30-epoch "
                         "estimator; the 'mine' column is the estimator of B.39, so the two differ by design):"),
                        ("## Expectations of the registration (section 6)",
                         "## Expectations of PREREGISTRATION_B44.md section 6 (for reference; not the B.48 expectations)")]), ""]
    L += ["## Choices made where the registration left room", ""] + [f"- {c}" for c in CHOICES]
    L += ["", "## Deviations", "", o.get("deviations_text", "None."), "",
          "## Limits", "", "One initialisation (the split seed). The test rows of these splits have been read by every "
          "earlier readout, and the definitions were fixed with the earlier estimator. The estimator was trained without "
          "delay and at the natural cadence.", "",
          f"Run time: rows stage {o.get('runtime_rows_s', 'n/a')} s, delay stage {o.get('runtime_delay_s', 'n/a')} s, "
          f"readout stage {o['runtime_readout_s']:.0f} s."]
    return "\n".join(L) + "\n"


def rerender():
    o = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    md = json.loads((CACHE / "embedded_md.json").read_text(encoding="utf-8"))
    OUT_MD.write_text(render(o, md["b41"], md["b42"], md["b44"]), encoding="utf-8", newline="\n")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "render":
        rerender()
        return
    rt_path = CACHE / "runtime.json"
    for name, fn in (("rows", stage_rows), ("delay", stage_delay)):
        if stage in (name, "all"):
            t = _time.time()
            fn()
            rt = json.loads(rt_path.read_text()) if rt_path.exists() else {}
            rt[name] = round(_time.time() - t, 1)
            rt_path.write_text(json.dumps(rt))
    if stage in ("readout", "all"):
        stage_readout()
    log("done")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
