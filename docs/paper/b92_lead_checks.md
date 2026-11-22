# B.92 independent model/loss/objective audit

{
  "passes": true,
  "sources": {
    "checks92.py": "a9ef7a45fc5da3fd855fcd9d46ad5fdf44abce8c6a055f8f231320446e0b3e5d",
    "common92.py": "bc7cb553b0640d8efac9c830f606a8fe4a489b8dfcfb491e832f7d8378aa6279",
    "model92.py": "64951153796374940ec82e0a341d106397045dc2234bdd3c37ee9eec5af113d5",
    "predict92.py": "603cd85e6ddde7f005558361b36604c6cb8e930277d5d9acd1d1ec2c8ac0285c",
    "prepare92.py": "a955ece8b7a02e449a94fd5f0d850cef6975db919010bb5f76a6392ce79b01d8",
    "queue92.py": "9bc53cf8f4a5f273c7c90280ddf637038cd4c27ff93d53cb119c9eb08dfcb12e",
    "readout92.py": "5a5950235a56b2b27a743a0fd17425247e6e1bcffe6bb70e05caa321ca8492e9",
    "run92.py": "84620d76fd696dcc2c83e02cd6c1ab0d1bd558fa733b3585be7864e26b41894b",
    "runtime92.py": "089593e7b28e0246ac7c4710bc19123e5128bc1e9a6fe1907168910aa6c3c5cc",
    "select92.py": "dae81bcd1279a7b5cee090014d6aac41fe3dfac4597ed5559e9c987ff71193c0",
    "train92.py": "722f6f83e3b7d5f17a01e838b9ad9416bc848c636679c79e4a207159e3791987"
  },
  "frozen": {
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
    "ces_prediction/experiments/b88_anchor/adapter88.py": "368fe2325d34cbef8129aee59c9ae62d8510dd0883a01cf296da88b01447bd10",
    "ces_prediction/experiments/b88_anchor/checks88.py": "9129356db67e89510b321cdd34a1d6b2618d2ae9c5bedc2ba0fdddb46130ef6f",
    "ces_prediction/experiments/b88_anchor/common88.py": "588a9b02ccc238f82dec69f4acd0b41e1b4d37623ef640e81153201d05c5f31f",
    "ces_prediction/experiments/b88_anchor/data88.py": "88e7d12872f3373b4e059e6e45cb8086a5a34ad4f8de0bfd7d18a9cc08eeaa63",
    "ces_prediction/experiments/b88_anchor/model88.py": "ebbb4d4cb464047486618e238de64686f8bc415605c50af5b1d598810be36371",
    "ces_prediction/experiments/b88_anchor/predict88.py": "922f8c9647ffca346e4d2e2c75d412929a32c219b0ce48630343efe2a2e0a4a9",
    "ces_prediction/experiments/b88_anchor/queue88.py": "ea16acaec2e037b8c25673a1819fb1801f65bfd62d0ac24be6a66c3474e94d7c",
    "ces_prediction/experiments/b88_anchor/readout88.py": "c98a4897eeaa9c47477914cf23894ad8d07fa8126245427622de1376bb218bbc",
    "ces_prediction/experiments/b88_anchor/run88.py": "28268346d30bf82d35b166b654740abd70aece37affb2d6f1fb1559e078c8e7d",
    "ces_prediction/experiments/b88_anchor/train88.py": "8156511200fa3e36902de3f45c1ea09f5e641a4265934b5492298ef930abe2f8",
    "ces_prediction/experiments/b88_anchor/frozen_b85_sources.json": "e198f7d4202f5e9547d6705f30cbd7289108ca42428e42ee3109f20b022de1c2",
    "ces_prediction/experiments/b90_methods/models90.py": "8c1b7b228aff58e3e9655afffa062d9e0ea97a7ac39b5c772b01f340fd3ef4ea",
    "ces_prediction/experiments/b91_fast/runtime91.py": "b102df311a5010bd0b3683ad224ea13796f64ee6c89d6b0a6313edd90661a67f",
    "ces_prediction/experiments/b91_fast/probe91.py": "22c6d0075887f01edc8153f63bb84f7e659d77929ce89020848f9b945393b681",
    "ces_prediction/experiments/b91_fast/validation91.py": "c41abbd60598e6be13f3b9bfda375f12d345c5c17bfed27abbdb0aa10b7f9109"
  },
  "real_train_val_only": true,
  "checks": [
    "original22/batching",
    "paired A/C",
    "A positive and C anchor invariance",
    "dense packed valid-prefix",
    "original and NumPy loss/gradient",
    "T-H fast invariance",
    "stream/reset",
    "NumPy seven-rule selector",
    "premature TEST guard"
  ]
}
