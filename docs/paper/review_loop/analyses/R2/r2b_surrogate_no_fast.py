"""R2-b (REGISTRATION.md): the stage-4 surrogate with every fast-channel coefficient a_c = 0,
i.e. the 4-parameter gap-dependent-gain CES-only filter, on the frozen T_i rows.

S0-zero: trained checkpoint with a_c := 0. S0-refit: K0, K_inf, s_K, b0 refit on TRAIN files
only (masked MSE on observed T_i rows with a past observation, L-BFGS, CPU).
Run from the repo root:  py docs/paper/review_loop/analyses/R2/r2b_surrogate_no_fast.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import r2_common as C
from evaluate import _load_stats
from seq_data import build_blocks, load_grid_files
from seq_models import SEQ_MODELS

torch.set_num_threads(8)


def predict(model, grid, dims, stats, names):
    """{name: {float(time): T_i eV}} -- eval_seq.build_pred_lookup's path, T_i only."""
    mean, std = float(stats["target"]["mean"][0]), float(stats["target"]["std"][0])
    out = {}
    model.eval()
    with torch.no_grad():
        for nm in names:
            tab = {}
            for b in build_blocks(grid[nm], dims, stats, per_shot_norm=True):
                p = model(torch.from_numpy(b["x"]).unsqueeze(0))[0, :, 0].numpy() * std + mean
                for i, t in enumerate(b["time"]):
                    tab[float(t)] = float(p[i])
            out[nm] = tab
    return out


def refit(grid, dims, stats, train_names):
    """4 parameters on TRAIN files: masked MSE (normalized units) on observed rows with a past obs."""
    blocks = [b for nm in train_names if nm in grid
              for b in build_blocks(grid[nm], dims, stats, per_shot_norm=True)]
    L = max(b["x"].shape[0] for b in blocks)
    x = torch.zeros(len(blocks), L, blocks[0]["x"].shape[1])
    y = torch.zeros(len(blocks), L)
    w = torch.zeros(len(blocks), L)
    for i, b in enumerate(blocks):
        n = b["x"].shape[0]
        x[i, :n] = torch.from_numpy(b["x"])
        y[i, :n] = torch.from_numpy(b["y"][:, 0])
        w[i, :n] = torch.from_numpy(b["mask"][:, 0])
    model = SEQ_MODELS["b28h"]()
    w = w * (x[..., model.col_ti_has] > 0.5).float()
    params = [model.k0_logit, model.kinf_logit, model.sk_raw, model.b0]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=200, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")
    n_obs = float(w.sum())
    trace = []

    def closure():
        opt.zero_grad()
        m, _ = model.ti_terms(x)
        loss = (((m + model.b0 - y) ** 2) * w).sum() / n_obs
        loss.backward()
        trace.append(float(loss))
        return loss

    opt.step(closure)
    with torch.no_grad():
        if float(model.a.abs().max()) != 0.0:
            raise SystemExit("FATAL: a_c moved during the 4-parameter refit")
    return model, {"n_train_blocks": len(blocks), "n_train_rows": int(n_obs), "loss_first": trace[0],
                   "loss_last": trace[-1], "n_closure_calls": len(trace)}


def main():
    t0 = time.time()
    out = {"protocol": {"B": C.B_BOOT, "seed_boot": C.SEED_BOOT,
                        "skill": "1 - sum SE_arm / sum SE_ref on the frozen T_i rows, shot-cluster bootstrap"},
           "cells": []}
    for pop in C.POPS:
        ds = C.build_dataset(pop)
        tc = int(ds._column_slices["time"])
        grid, dims = load_grid_files(C.DATA, drop_stuck_targets=True, ti_spike_cut_ev=float(C.POPS[pop]["cut"]))
        for s in C.SEEDS:
            d = C.canonical(pop, s, "CES_TI")
            sur_dir = C.DATA / f".b28_b28h_{pop}_s{s}"
            stats_sur = _load_stats(json.loads((sur_dir / "metrics.json").read_text(encoding="utf-8")))
            stats_bb = _load_stats(json.loads((C.DATA / C.POPS[pop]["run"].format(s=s) / "metrics.json").read_text(encoding="utf-8")))
            names = [Path(ds.valid_files[int(i)]).name for i in d["shot"]]
            keys = [float(ds.file_arrays[int(i)][int(r), tc]) for i, r in zip(d["shot"], d["row"])]
            test_names = sorted(set(names))
            man = C.manifest(pop, s)
            if set(test_names) - set(man["test_files"]):
                raise SystemExit("FATAL: frozen rows outside the manifest TEST files")

            def rows(lookup):
                return np.array([lookup[n][k] for n, k in zip(names, keys)])

            full = SEQ_MODELS["b28h"]()
            full.load_state_dict(torch.load(sur_dir / "weights" / "seq_lstm.pth", map_location="cpu"))
            p_full = rows(predict(full, grid, dims, stats_sur, test_names))
            # reproduction check: |pred - y| must equal sqrt(stored se) to < 0.05 eV
            repro = float(np.max(np.abs(np.abs(p_full - d["y"]) - np.sqrt(d["surrogate"]))))
            if repro > 0.05:
                raise SystemExit(f"FATAL: {pop} s{s}: surrogate checkpoint not reproduced (max {repro:.3g} eV)")
            with torch.no_grad():
                full.a.zero_()
            p_zero = rows(predict(full, grid, dims, stats_sur, test_names))
            ph_zero = full.physics(stats_sur["target"]["std"])

            model, fitrec = refit(grid, dims, stats_bb, man["train_files"])
            p_refit = rows(predict(model, grid, dims, stats_bb, test_names))
            ph = model.physics(stats_bb["target"]["std"])

            se = dict(d)
            se["s0_zero"] = (p_zero - d["y"]) ** 2
            se["s0_refit"] = (p_refit - d["y"]) ** 2
            sh = d["shot"]
            rec = {"pop": pop, "seed": s, "n_rows": int(len(sh)), "repro_max_abs_eV": repro,
                   "refit": {**fitrec, "K0": ph["K0"], "K_inf": ph["K_inf"], "s_K_ms": ph["s_K_ms"], "b0_eV": ph["b0_eV"]},
                   "zero": {"K0": ph_zero["K0"], "K_inf": ph_zero["K_inf"], "s_K_ms": ph_zero["s_K_ms"], "b0_eV": ph_zero["b0_eV"]},
                   "rmse_eV": {a: float(np.sqrt(se[a].mean())) for a in
                               ("s0_zero", "s0_refit", "surrogate", "model", "gp_causal", "pchip", "persistence", "gp")},
                   "skill": {}}
            for a, b in (("s0_refit", "gp_causal"), ("s0_zero", "gp_causal"),
                         ("surrogate", "s0_refit"), ("surrogate", "s0_zero"),
                         ("model", "s0_refit"), ("model", "s0_zero"),
                         ("s0_refit", "pchip"), ("s0_refit", "persistence"), ("s0_refit", "gp"),
                         ("s0_zero", "pchip"), ("s0_zero", "persistence"),
                         ("surrogate", "gp_causal"), ("model", "gp_causal"), ("surrogate", "pchip"), ("model", "pchip"),
                         ("model", "surrogate")):
                rec["skill"][f"{a}_vs_{b}"] = C.skill(sh, se[a], se[b])
            sk = rec["skill"]
            rec["retention_s0_refit"] = {
                "vs_pchip": sk["s0_refit_vs_pchip"]["point"] / sk["model_vs_pchip"]["point"],
                "vs_gp_causal": sk["s0_refit_vs_gp_causal"]["point"] / sk["model_vs_gp_causal"]["point"]}
            rec["retention_surrogate"] = {
                "vs_pchip": sk["surrogate_vs_pchip"]["point"] / sk["model_vs_pchip"]["point"],
                "vs_gp_causal": sk["surrogate_vs_gp_causal"]["point"] / sk["model_vs_gp_causal"]["point"]}
            out["cells"].append(rec)
            (C.HERE / "r2b_surrogate_no_fast.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
            print(f"[R2b] {pop} s{s}: repro {repro:.2g} eV; refit K0 {ph['K0']:.3f} Kinf {ph['K_inf']:.3f} sK {ph['s_K_ms']:.1f} ms "
                  f"b0 {ph['b0_eV']:+.2f} eV ({fitrec['n_closure_calls']} evals) | rmse S0r {rec['rmse_eV']['s0_refit']:.1f} "
                  f"S0z {rec['rmse_eV']['s0_zero']:.1f} sur {rec['rmse_eV']['surrogate']:.1f} bb {rec['rmse_eV']['model']:.1f} "
                  f"gpc {rec['rmse_eV']['gp_causal']:.1f}", flush=True)
            for k in ("s0_refit_vs_gp_causal", "surrogate_vs_s0_refit", "model_vs_s0_refit", "s0_zero_vs_gp_causal",
                      "surrogate_vs_s0_zero", "s0_refit_vs_pchip", "s0_refit_vs_persistence"):
                print(f"      {k:26s} {C.fmt(sk[k])}", flush=True)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (C.HERE / "r2b_surrogate_no_fast.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote r2b_surrogate_no_fast.json ({out['minutes']} min)")


if __name__ == "__main__":
    main()
