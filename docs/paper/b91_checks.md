# B.91 checks

All 21 fresh children passed numerical, real-input and seven-rule validator checks and exited normally with supervisor-observed code 0. Parent exit attestations, child result hashes, source/software/input bindings were rechecked. runtime91.verify_decision() passed against current sources and actual training inputs after the matrix. See b91_runtime.md for measured point timings and data/.b91/probes for full artifacts.

These checks use dropout 0 and do not establish production dropout 0.1 teardown safety. No production fitting or scientific test scoring was performed.
