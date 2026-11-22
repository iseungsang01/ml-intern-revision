# B.91 independent artifact/timing audit

{
  "passes": true,
  "sources": {
    "ces_prediction/experiments/b85_unified/common85.py": "33fe0dbc5381aceffbcb268daac5d67290c3d78fef2029ee4705a2fef6fa2360",
    "ces_prediction/experiments/b85_unified/data85.py": "eb9af716bf7860a05eeb1015ee141e3eaff353d5ade493c6ccb91762f3141755",
    "ces_prediction/experiments/b85_unified/model85.py": "2979353ce7b3945191be1a32f132c91b78a97f91b77fd084b7b6515533640456",
    "ces_prediction/experiments/b85_unified/train85.py": "c9fa94a17b21cff2d96e25b52498e1fdea552c6040fc8a6eed75638cf616ecde",
    "ces_prediction/experiments/b85_unified/readout85.py": "b7f0f33706222c850a5d6a4f97c405a1e58bce29dd7024d878496db218f25a83",
    "ces_prediction/experiments/b73_targets/common.py": "5d63e579b1f24c560ba97b73bd76345ac8eae04a5b706a4670326d4ccc87555a",
    "ces_prediction/experiments/b73_targets/designs.py": "06ef0874049ca8ced4a5f1059013e4e92631f46c887cff1cda799fecaac0a095",
    "ces_prediction/experiments/seq/model_seq_v2.py": "4ecba72554169ba6365f8528d583bc475199c0ce60ad3854e2e7bf798a4d4929",
    "ces_prediction/experiments/seq/seq_data.py": "60f1a43cce460c5653d12dcf704228d13ab4ea072187fd1e7a486bb70cb49038",
    "ces_prediction/dataset.py": "44c034d30e460eb7264fe12e0c7d629dafd055dc4def45cad92f56bbadf78782",
    "ces_prediction/experiments/b82_localize/boot82.py": "b8036aaa63aebe47fb709dd33d5ad8a3d721cd89b338e66321df8d01a4a63593",
    "ces_prediction/experiments/b82_localize/common82.py": "bedaa2a0e6f5a0cf69f4d4e67c6a54f8a35d6741a2d195aaaf95fc231b1d576b",
    "ces_prediction/experiments/PREREGISTRATION_B91.md": "62531f6261ed9fe45b7d39588d86f595aa2b2f8a58b5be399c1b23d3bf5db0f4",
    "ces_prediction/experiments/b88_anchor/frozen_b85_sources.json": "e198f7d4202f5e9547d6705f30cbd7289108ca42428e42ee3109f20b022de1c2",
    "ces_prediction/experiments/b91_fast/runtime91.py": "b102df311a5010bd0b3683ad224ea13796f64ee6c89d6b0a6313edd90661a67f",
    "ces_prediction/experiments/b91_fast/probe91.py": "22c6d0075887f01edc8153f63bb84f7e659d77929ce89020848f9b945393b681",
    "ces_prediction/experiments/b91_fast/validation91.py": "c41abbd60598e6be13f3b9bfda375f12d345c5c17bfed27abbdb0aa10b7f9109"
  },
  "clean_children": 21,
  "selected_path": "cuda_dense_cudnn",
  "batch_speed_factor": 11.030222978499308,
  "median_batch_seconds": {
    "cuda_packed_native": 0.6254710000066552,
    "cuda_dense_native": 0.534944999992149,
    "cuda_dense_cudnn": 0.05670519999694079,
    "cuda_packed_cudnn": 0.08646809999481775,
    "cpu_packed_native2": 1.2854195999971125,
    "cpu_dense_mkldnn2": 0.2413100999838207,
    "cpu_dense_mkldnn8": 0.11681120001594536
  },
  "scope": "dropout0 real-prefix batch; no full-epoch or production-dropout stability claim"
}
