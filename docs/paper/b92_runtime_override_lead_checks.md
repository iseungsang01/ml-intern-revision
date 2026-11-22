# B.92 explicit CPU runtime revision audit

{
  "passes": true,
  "sources": {
    "checks92.py": "a9ef7a45fc5da3fd855fcd9d46ad5fdf44abce8c6a055f8f231320446e0b3e5d",
    "common92.py": "53133944d7d351262ea41eb949bc13de193d055d23b65f4a9a71a6cc72320669",
    "model92.py": "64951153796374940ec82e0a341d106397045dc2234bdd3c37ee9eec5af113d5",
    "predict92.py": "603cd85e6ddde7f005558361b36604c6cb8e930277d5d9acd1d1ec2c8ac0285c",
    "prepare92.py": "a955ece8b7a02e449a94fd5f0d850cef6975db919010bb5f76a6392ce79b01d8",
    "queue92.py": "9bc53cf8f4a5f273c7c90280ddf637038cd4c27ff93d53cb119c9eb08dfcb12e",
    "readout92.py": "5a5950235a56b2b27a743a0fd17425247e6e1bcffe6bb70e05caa321ca8492e9",
    "run92.py": "cda460f872981c2c405c43e3bb68da0a79f4b164b8446b75f3ef4fbf0d69d33a",
    "runtime92.py": "e3346a0c96b7695c431adf760bc1d4033d6e943abcf0ca37cc197bb0f349f446",
    "select92.py": "dae81bcd1279a7b5cee090014d6aac41fe3dfac4597ed5559e9c987ff71193c0",
    "train92.py": "1d00454c3dfa3f376662055db115449a324b132a497a8916e9cb35457f986e21"
  },
  "production_override_sha256": "081d7e2dc8041ddf90115503f773911043e4b997775562bfdf0247bf76825c50",
  "selected_path": "cpu_dense_mkldnn8",
  "batch_speed_factor": 5.3545464811702566,
  "qualification": "dropout0 timing and preserved failed cuDNN dropout.1 smoke; new CPU dropout.1 smoke still required"
}
