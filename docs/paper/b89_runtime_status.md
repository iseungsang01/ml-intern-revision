# B.89 runtime scout status

Implementation and real-input CPU preflight passed. Two actual training prefixes (33 and 128 rows) retain real labels/masks; CPU forward/backward, recurrent streaming, reset and save/load guards passed with child exit 0. Sources, inherited source pins, software and input files are hash-bound in `data/.b89/preflight.json` and `runtime_manifest.json`.

No B.89 GPU benchmark has run. The scout waits for verified completion of B.85 and all 90 clean fit attestations before launching any CUDA child. It writes a recommendation only and cannot activate a backend for B.88 or B.90. Current B.85 execution is not modified or interrupted.

The supported cache environment spelling is `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=-1`, confirmed in the official [PyTorch CUDA environment documentation](https://docs.pytorch.org/docs/main/cuda_environment_variables.html). These API controls may not resolve the RNN teardown problem; no success or speedup is presumed. Parent-observed clean exit is mandatory even if a timing artifact was written.
