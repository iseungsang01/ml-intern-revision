# Proof scripts behind docs/theory_results_physics.md (section 8bb)

One folder per proof, P01-P10. Each holds the script(s) an independent agent wrote and the JSON /
log it produced; every number quoted in the document comes from these outputs. The scripts are
read-only audits in the class of `analyze_foundations.py`: they read `data/*.csv` and the frozen
TEST npz under `data/.b1_seqv2_s*`, `data/.b5i_seqv2_s*`, `data/.b15_ens4_cut_s*`; they train,
score or select nothing. Run each from the repo root with `py <script>`; the few that import the
baselines patch `sys.path` themselves. Outputs were written next to the scripts.

| folder | proof | main script |
|---|---|---|
| p01 | T_i headline = transient physics | p01_transient_proof.py, p01_part2.py |
| p02 | V_rot tie from the momentum balance | p02_vrot_momentum.py, p02_part_d.py |
| p03 | 50 ms saturation: filter convergence vs tau_eq vs ELM recovery | p03_proof.py |
| p04 | campaign drift = instrument gain g(D)s(t)+c(D) | p04_campaign_drift.py, p04_extra.py |
| p05 | 3 keV cut = pedestal pressure limit; spike = fit failure | p05_pressure_limit.py, p05_fit_mc.py |
| p06 | mutual-information bound on the fast-diagnostic gain | p06_mi_bound.py, p06_timedomain.py |
| p07 | MNAR = photon-budget physics | p07_mnar_check.py, p07_crb_background.py, p07_event_study.py |
| p08 | per-discharge win-rate asymmetry | p08_winrate_asymmetry.py |
| p09 | MAST transfer null by dimensionless similarity | p09_check.py |
| p10 | real-time latency classes from plasma time scales | p10_latency_classes.py |
