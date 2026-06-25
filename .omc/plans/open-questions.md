# Open Questions

## CES Peak Reconstruction Eval (ces-peak-recon-plan) - 2026-06-23

Resolved in plan draft v2 (consensus iteration 2):
- [x] Peak-detection default thresholds → conservative defaults pinned in `PEAK_PARAMS` (input-only `slope_z=2.5`/`neigh_var_pct=0.10`; target `diff_z=2.5`/`roll_window=5`/`top_pct=0.10`) + 2–3 setting sensitivity sweep recorded in `HANDOFF.md`.
- [x] AC8 exposure strategy → Option A standalone `peak_analysis.py` + minimal Option-B npz hook (`{name}_is_peak`); peak block routed into `eval_metrics.json` + `build_briefing` string + `program.md` steering paragraph (the researcher's actual read path).
- [x] `N_MIN` for peak bootstrap → `N_MIN_PEAK_SHOTS=15` (binding shot-clustered CI gate) + `N_MIN_PEAK_ROWS=200` (secondary); below either → `insufficient_*` and `pass` forced false.
- [x] `matplotlib` dependency → optional `[viz]` extra, lazy guarded `Agg` import, figures skippable with `--no-figures`; never a core dependency.

Updated in FINAL (consensus approved; status pending approval):
- [x] AC8 exposure refined → in-loop in-memory merge in `run_evaluation` + `build_briefing` string + `program.md` (an on-disk `eval_metrics.json` write is never re-read by the briefing). The in-loop block is the CHEAP headline peak skill/CI (reuses the val-split npz, no extra forward pass).

Open for the USER (non-blocking, surfaced in the final plan):
- [ ] (a) Loop-cost tolerance — resolved by flag-gating the AC7 ablation (`CES_PEAK_ABLATION=1`, out-of-loop) and keeping only the cheap headline peak metric in-loop. Confirm this matches loop-budget expectations; if the ablation should run every iteration, the flag can be lifted.
- [ ] (b) "the loop auto-recommends" intent — satisfied via two channels (live per-iteration peak skill/CI in the briefing + a standing steering rule in `program.md`; researcher proposes the peak-weighted loss, loop does not silently flip it). Confirm this combined reading matches the spec's intent or whether a stronger/automatic mechanism is expected.

Standing finding to report (not a question):
- CES_VT may be non-significant at peaks (known T_i/V_rot asymmetry). A null there is a valid, reportable scientific result, NOT a failure to fix. Headline rests on input-defined (Family i) peaks; observed-target (Family ii) peaks carry a selection-bias caveat.
