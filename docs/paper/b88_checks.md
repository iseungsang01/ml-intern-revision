# B.88 implementation checks

- `checks88.py`: all four designs and eight rules, 32 real-data cases. Original 22 channels, normalized labels and boolean masks equal the frozen builder. Appended channels use the exact latest visible measurement source and existing normalization.
- Independent sampled timestamp/visibility derivation plus all reference-cache supports; strict prior, file and segment bounds.
- Whole-prefix invariance to unavailable CES value mutation/removal and later fast-channel mutation, and positive selected-anchor fast mutation.
- Matched A/C initialization and 356754 parameters; C appended-anchor invariance; absent-history agreement; real-prefix streaming/full agreement and recurrent reset.
- Adapter CPU inspection: `.b88` callback/cache isolation; `common85.CACHE` stays `.b85`; repeated activation returns the same local trainer; current B.85 dependency correctly returns false.
- Premature `readout88.py` fails at the dependency guard before loading predictions or calculating errors. The plan enumerates all 90 registered fits.
- `py -m compileall -q ces_prediction/experiments/b88_anchor` passed; `py -m pytest -q`: 89 passed, three pre-existing warnings.

CPU checks do not substitute for registered real-data CUDA smoke. Each inherited training child runs finite forward/backward, streaming/full, segment reset and save/load checks with preserved training RNG. The supervisor requires separate fresh A/C smoke completions before the 90-fit batch, runner-observed exit attestations and prediction label-invariance checks before readout.
