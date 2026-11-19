"""B.70 checks before and after the full re-scoring (task steps 2 (a)-(d)); writes docs/paper/b70_checks.md.

    py ces_prediction/experiments/b70_d1/checks_b70.py counts     # (b) independent counts per fold -> data/.b70/counts.json
    py ces_prediction/experiments/b70_d1/checks_b70.py report     # (a)-(d) -> docs/paper/b70_checks.md / data/.b70/checks.json

(a) the cap-40000 re-scoring of .b39p_v2_cut_r2_f0 (data/.b70_rescore/b39p_v2_cut_r2_f0_cap40000) against the stored
    comparison_errors_test.npz, key by key;
(b) the eligible-sample count of every fold under CES_MAX_VAL_SAMPLES=0 (eval_samples of the D1 run) against the
    ledger's range (docs/paper/dataset_ledger.md 1.4: 40,666-52,744) and against an independent recount from the grid
    (seq_data.load_grid_files, cut + held removed: kept rows = rows with a finite target; a sample = a kept row whose
    previous kept row is less than 0.5 s before it); the scored T_i rows against the recount (T_i finite on the row
    and on the previous kept row); the old 40,000-sample rows are a subset of the D1 rows;
(c) every arm of a fold has the same (shot, row) list and bit-identical reference errors;
(d) the refitted second stage against the stored B.64 predictions on the old rows (stage2_b70 writes the deviations).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs_b70 as RB  # noqa: E402

B39, B27, DATA = RB.B39, RB.B27, RB.DATA
REPO = HERE.parents[2]
CACHE = DATA / ".b70"
OUT_MD = REPO / "docs" / "paper" / "b70_checks.md"
OUT_JSON = CACHE / "checks.json"
T = "CES_TI"
LEDGER_RANGE = (40666, 52744)


def grid():
    sys.path.insert(0, str(RB.SEQ_DIR))
    sys.path.insert(1, str(REPO / "ces_prediction"))
    from seq_data import load_grid_files
    g, _ = load_grid_files(DATA, True, 3000.0)
    return g


def counts():
    g = grid()
    per_file = {}
    for nm, arr in g.items():
        fin = np.isfinite(arr[:, 1:3])
        kept = np.flatnonzero(fin.any(1))
        if len(kept) < 2:
            per_file[nm] = (0, 0, 0)
            continue
        t = arr[kept, 0].astype(np.float64)
        cont = np.diff(t) < 0.5
        ti = fin[kept, 0]
        samples = int(cont.sum())
        ti_rows = int((cont & ti[1:] & ti[:-1]).sum())
        per_file[nm] = (samples, ti_rows, len(kept))
    out = {}
    for r in RB.REPEATS:
        for k in range(RB.K):
            man = json.loads((B27.split_dir(r, k) / "split_manifest.json").read_text(encoding="utf-8"))
            test = man["test_files"]
            s = [per_file.get(n, (0, 0, 0)) for n in test]
            out[f"r{r}f{k}"] = {"test_files": len(test), "eligible_samples": int(sum(x[0] for x in s)),
                                "ti_rows": int(sum(x[1] for x in s)),
                                "discharges_with_ti_rows": int(sum(x[1] > 0 for x in s)),
                                "max_ti_rows_one_discharge": int(max(x[1] for x in s))}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "counts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    tot = sum(v["eligible_samples"] for v in out.values())
    print("eligible samples per fold:", {k: v["eligible_samples"] for k, v in out.items()}, "total", tot, flush=True)
    return out


def npz(path):
    z = np.load(path)
    return {kk: z[kk] for kk in z.files}


def check_a():
    new = npz(RB.new_dir("b39", "v2", 2, 0, cap=40000) / "comparison_errors_test.npz")
    old = npz(B39.pooled_run_dir("v2", 2, 0) / "comparison_errors_test.npz")
    rows = {}
    for kk in sorted(set(new) | set(old)):
        if kk not in new or kk not in old:
            rows[kk] = {"present": [kk in new, kk in old]}
            continue
        a, b = new[kk], old[kk]
        same_shape = a.shape == b.shape
        rec = {"shape": list(a.shape), "same_shape": same_shape}
        if same_shape:
            rec["bit_identical"] = bool(np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b))
            if a.dtype.kind == "f" and not rec["bit_identical"]:
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                rec["max_abs_diff"] = float(np.nanmax(d))
                if kk.endswith("_se_model"):
                    yk = kk.replace("_se_model", "_y_true")
                    ae = np.abs(np.sqrt(a.astype(np.float64)) - np.sqrt(b.astype(np.float64)))
                    rec["max_abs_error_diff_eV_or_kms"] = float(np.max(ae))
                    rec["median_abs_error_diff"] = float(np.median(ae))
                    rec["p99_abs_error_diff"] = float(np.quantile(ae, 0.99))
                    rec["rows_abs_error_diff_above_1e-3"] = int((ae > 1e-3).sum())
                    rec["rows_abs_error_diff_above_0.5"] = int((ae > 0.5).sum())
                    rec["relative_mse_diff"] = float(abs(a.mean() - b.mean()) / b.mean())
                    rec["y_key"] = yk
        rows[kk] = rec
    model_keys = [kk for kk in rows if kk.endswith("_se_model")]
    ref_ok = all(rows[kk].get("bit_identical") for kk in rows if kk not in model_keys)
    return {"keys": rows, "everything_but_the_network_bit_identical": bool(ref_ok),
            "network_bit_identical": bool(all(rows[kk].get("bit_identical") for kk in model_keys))}


def load_ti(path):
    z = np.load(path)
    return {kk: z[kk] for kk in z.files if kk.startswith(T + "_")}


def check_bc():
    C = json.loads((CACHE / "counts.json").read_text(encoding="utf-8"))
    b, c = {}, {}
    for r in RB.REPEATS:
        for k in range(RB.K):
            key = f"r{r}f{k}"
            ref_dir = RB.new_dir("b39", "v2", r, k)
            cm = json.loads((ref_dir / "comparison_metrics_test.json").read_text(encoding="utf-8"))
            ref = load_ti(ref_dir / "comparison_errors_test.npz")
            old = load_ti(B39.pooled_run_dir("v2", r, k) / "comparison_errors_test.npz")
            pair_new = set(zip(ref[f"{T}_shot"].tolist(), ref[f"{T}_row"].tolist()))
            pair_old = set(zip(old[f"{T}_shot"].tolist(), old[f"{T}_row"].tolist()))
            n_rows = len(ref[f"{T}_shot"])
            u, cnt = np.unique(ref[f"{T}_shot"], return_counts=True)
            b[key] = {"eligible_samples_d1": cm["eval_samples"], "eligible_samples_recount": C[key]["eligible_samples"],
                      "in_ledger_range": bool(LEDGER_RANGE[0] <= cm["eval_samples"] <= LEDGER_RANGE[1]),
                      "ti_rows_d1": n_rows, "ti_rows_recount": C[key]["ti_rows"],
                      "discharges_d1": int(len(u)), "discharges_recount": C[key]["discharges_with_ti_rows"],
                      "max_rows_one_discharge_d1": int(cnt.max()), "max_rows_recount": C[key]["max_ti_rows_one_discharge"],
                      "old_rows": len(pair_old), "old_rows_subset_of_d1": bool(pair_old <= pair_new),
                      "d1_rows_unique": bool(len(pair_new) == n_rows)}
            b[key]["passes"] = bool(b[key]["eligible_samples_d1"] == b[key]["eligible_samples_recount"]
                                    and b[key]["ti_rows_d1"] == b[key]["ti_rows_recount"]
                                    and b[key]["old_rows_subset_of_d1"] and b[key]["d1_rows_unique"])
            arms = {}
            for fam, arms_ in RB.FAMILIES.items():
                for arm in arms_:
                    if (fam, arm) == ("b39", "v2"):
                        continue
                    d = RB.new_dir(fam, arm, r, k)
                    p = d / "comparison_errors_test.npz"
                    if not p.exists():
                        arms[f"{fam} {arm}"] = "missing"
                        continue
                    z = load_ti(p)
                    same = (np.array_equal(z[f"{T}_shot"], ref[f"{T}_shot"]) and np.array_equal(z[f"{T}_row"], ref[f"{T}_row"]))
                    refs = same and all(np.array_equal(z[f"{T}_se_{q}"], ref[f"{T}_se_{q}"])
                                        for q in ("persistence", "gp_causal", "gp", "pchip", "linear"))
                    yy = same and np.array_equal(z[f"{T}_y_true"], ref[f"{T}_y_true"])
                    arms[f"{fam} {arm}"] = {"same_rows": bool(same), "references_bit_identical": bool(refs),
                                            "y_bit_identical": bool(yy)}
            c[key] = {"arms": arms, "passes": bool(all(isinstance(v, dict) and v["same_rows"] and v["references_bit_identical"]
                                                       and v["y_bit_identical"] for v in arms.values()))}
    return b, c


def check_d():
    out = {}
    for r in RB.REPEATS:
        for k in range(RB.K):
            p = CACHE / f"stage2_r{r}_f{k}.npz"
            if not p.exists():
                out[f"r{r}f{k}"] = "missing"
                continue
            info = json.loads(str(np.load(p)["info"]))
            out[f"r{r}f{k}"] = {a: {"max_dev_eV": v["check_d_max_dev_eV"], "passes": v["check_d_passes"],
                                    "fit_rows_equal_b64": v["fit_rows_equal_b64"]} for a, v in info.items()}
    ok = all(isinstance(v, dict) and all(x["passes"] and x["fit_rows_equal_b64"] for x in v.values()) for v in out.values())
    return out, ok


def env_pins():
    """Every stored run: metrics.json agrees with the env it is re-scored under (model, init seed, input norm)."""
    bad = []
    for x in RB.plan():
        m = json.loads((RB.stored_dir(*x) / "metrics.json").read_text(encoding="utf-8"))
        env = RB.rescore_env(*x, RB.new_dir(*x))
        if m["seq_model"] != env["CES_SEQ_MODEL"] or int(m["init_seed"]) != int(env["CES_INIT_SEED"]) \
                or bool(m["per_shot_input_norm"]) != (env["CES_PER_SHOT_NORM"] == "1"):
            bad.append((RB.stored_dir(*x).name, m["seq_model"], m.get("init_seed"), m["per_shot_input_norm"]))
    return {"runs": len(RB.plan()), "mismatches": bad, "passes": not bad}


def report():
    res = {"a": check_a()}
    res["b"], res["c"] = check_bc()
    res["d"], d_ok = check_d()
    res["env"] = env_pins()
    vp = DATA / ".b70_rescore_check" / "validation.json"
    res["shared_scorer"] = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else {"passes": False, "missing": True}
    res["passes"] = {"shared_scorer_equals_own_scripts_r2f0": bool(res["shared_scorer"]["passes"]),
                     "a_references": res["a"]["everything_but_the_network_bit_identical"],
                     "b": all(v["passes"] for v in res["b"].values()),
                     "c": all(v["passes"] for v in res["c"].values()), "d": d_ok, "env": res["env"]["passes"]}
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print("checks:", res["passes"], flush=True)


def render(res):
    A = res["a"]
    L = ["# B.70 checks (phase 1)", "",
         "Registered: ces_prediction/experiments/PREREGISTRATION_B70.md (Addendum 1). Written by "
         "`ces_prediction/experiments/b70_d1/checks_b70.py report`; the numbers are in `data/.b70/checks.json`.", "",
         "## (a) The re-scoring path reproduces the stored B.39 network on the old subset", "",
         "`.b39p_v2_cut_r2_f0` re-scored by its own scorer (seq/eval_seq.py) under `run_b39.pooled_env` with "
         "CES_MAX_VAL_SAMPLES=40000 (the stored value), CES_SEQ_DEVICE=cpu and a new output directory, against the stored "
         "`comparison_errors_test.npz` (scored on the GPU).", "",
         "| key | shape | bit-identical | max abs difference | notes |", "|---|---|---|---|---|"]
    for kk, v in A["keys"].items():
        if "present" in v:
            L.append(f"| {kk} | - | key present (new, stored): {v['present']} | | |")
            continue
        note = ""
        if "max_abs_error_diff_eV_or_kms" in v:
            note = (f"absolute error differs by median {v['median_abs_error_diff']:.1e}, 99th percentile "
                    f"{v['p99_abs_error_diff']:.1e}, max {v['max_abs_error_diff_eV_or_kms']:.2f} (eV or km/s); rows above 0.5: "
                    f"{v['rows_abs_error_diff_above_0.5']}; relative MSE difference {v['relative_mse_diff']:.1e}")
        L.append(f"| {kk} | {v['shape']} | {v.get('bit_identical')} | {v.get('max_abs_diff', 0):.3g} | {note} |")
    L += ["", f"Everything except the network's squared errors is bit-identical: **{A['everything_but_the_network_bit_identical']}**. "
          f"The network's squared errors are bit-identical: {A['network_bit_identical']} (the stored run scored the LSTM on CUDA, "
          "this one on the CPU; the difference is float32 rounding of the forward pass).", "",
          "## (b) Every eligible row is kept with CES_MAX_VAL_SAMPLES=0", "",
          "`dataset.select_seeded_random_indices` returns the list unchanged when the cap is <= 0 (dataset.py:547-549). "
          "Eligible samples (eval_samples of the D1 run) against an independent recount from the grid (cut + held removed; "
          "a sample = a kept row whose previous kept row is < 0.5 s before it); scored T_i rows against the recount (T_i finite "
          "on the row and on the previous kept row); the old 40,000-sample rows are a subset of the D1 rows. Ledger range "
          f"(1.4, r2/r5/r9 checked): {LEDGER_RANGE[0]:,}-{LEDGER_RANGE[1]:,}.", "",
          "| fold | eligible samples (D1 / recount) | in ledger range | T_i rows (D1 / recount) | discharges (D1 / recount) | "
          "max rows of one discharge (D1 / recount) | old rows, subset of D1 | passes |", "|---|---|---|---|---|---|---|---|"]
    for key, v in res["b"].items():
        L.append(f"| {key} | {v['eligible_samples_d1']:,} / {v['eligible_samples_recount']:,} | {v['in_ledger_range']} | "
                 f"{v['ti_rows_d1']:,} / {v['ti_rows_recount']:,} | {v['discharges_d1']} / {v['discharges_recount']} | "
                 f"{v['max_rows_one_discharge_d1']:,} / {v['max_rows_recount']:,} | {v['old_rows']:,}, {v['old_rows_subset_of_d1']} | "
                 f"{v['passes']} |")
    L += ["", "The ledger's range was measured on r2, r5 and r9 only; a fold of r3 or r4 outside it (r3 f2) is not a failure. "
          "The check passes when the D1 counts equal the recount and the old rows are a subset of the D1 rows."]
    L += ["", "## (c) Every arm of a fold is scored on identical rows", "",
          "Against the D1 re-scoring of the B.39 network of the fold: same (shot, row) list, bit-identical T_i values and "
          "bit-identical reference errors (hold, causal GP, offline GP, PCHIP, linear).", "",
          "| fold | arms compared | all same rows, values and references |", "|---|---|---|"]
    for key, v in res["c"].items():
        L.append(f"| {key} | {len(v['arms'])} | {v['passes']} |")
    L += ["", "## (d) The refitted second stage reproduces the stored B.64 predictions on the old rows", "",
          "The B.64 second-stage models were not saved (stage2_b64.fit_fold stores predictions only). They were refitted with "
          "stage2_b64's code, model and seed on the unchanged R-fit64 rows and applied to the old B.39 rows exactly as in "
          "fit_fold. Tolerance 1e-6 eV.", "",
          "| fold | network: max deviation (two-stage / own missing-input handling / first stage), eV | fit rows = B.64 | "
          "twin: max deviation, eV | fit rows = B.64 | passes |", "|---|---|---|---|---|---|"]
    for key, v in res["d"].items():
        if not isinstance(v, dict):
            L.append(f"| {key} | {v} | | | | |")
            continue
        n, t = v["v2"], v["v2h"]
        f = lambda x: f"{x['max_dev_eV']['est2']:.1e} / {x['max_dev_eV']['est2nan']:.1e} / {x['max_dev_eV']['est1']:.1e}"  # noqa: E731
        L.append(f"| {key} | {f(n)} | {n['fit_rows_equal_b64']} | {f(t)} | {t['fit_rows_equal_b64']} | "
                 f"{n['passes'] and t['passes']} |")
    v = res["shared_scorer"]
    L += ["", "## The shared-reference scorer (Addendum 1, A6) against the arms' own scoring scripts", "",
          "Fold r2 f0 was scored twice: each arm by its own scoring script (seq/eval_seq.py, eval_b59L.py, eval_b67.py; "
          "data/.b70_rescore/) and by `score_b70.py` (rows and references once, the network lookup per arm; "
          "data/.b70_rescore_check/). Every key of every arm's npz (both targets) and the metrics json:", "",
          "| arm | keys | keys not bit-identical | metrics json equal |", "|---|---|---|---|"]
    for kk, x in v.get("arms", {}).items():
        L.append(f"| {kk} | {x['keys']} | {', '.join(x['keys_not_bit_identical']) or 'none'} | {x['metrics_json_equal']} |")
    L += ["", f"Passes: **{v['passes']}**. The other 14 folds were scored by `score_b70.py`; r2 f0 keeps the outputs of the "
          "arms' own scripts."]
    e = res["env"]
    L += ["", "## Environment of the re-scored runs", "",
          f"All {e['runs']} runs are re-scored under their runner's own env function with exactly CES_MAX_VAL_SAMPLES, "
          "CES_SEQ_DEVICE and CES_OUTPUT_DIR changed (runs_b70.execute refuses any other difference); the stored "
          f"metrics.json agrees with the env on the model, the initialisation seed and the input standardisation: "
          f"{e['passes']}" + (f" (mismatches: {e['mismatches']})" if e["mismatches"] else "") + ".", "",
          "## Summary", "", "| check | passes |", "|---|---|"]
    for kk, v in res["passes"].items():
        L.append(f"| {kk} | {v} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("counts", "report"):
        raise SystemExit("usage: checks_b70.py counts|report")
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    counts() if sys.argv[1] == "counts" else report()
