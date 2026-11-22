# B.90 independent model and loss audit

{
  "passes": true,
  "execution_sources": {
    "adapter90.py": "35e98883c1e219853e1c25715059829dc1a011ad70b77d68f12f53380db9a308",
    "checks90.py": "cbd7eb32553066442b809073d82c0eac3f6b110af97cfecc5bf8438e1a49eec0",
    "common90.py": "188e0c5c89620f9db8ff0e44f16f94f1cbc94955d838cd460611764c9736aa72",
    "model_checks90.py": "3cdceee5de83163ccc5d3c316de0fdda28797e82f11ee2b2a2e5558fbee6f90a",
    "models90.py": "8c1b7b228aff58e3e9655afffa062d9e0ea97a7ac39b5c772b01f340fd3ef4ea",
    "predict90.py": "7d33e4bdcde1bc249ebdbebaf9667c2c47e9adb72ea860b5ca860ff637e99064",
    "queue90.py": "bf491ba6031a3d5b32c2b282344fbd8915453c2addd4739bde1d8124dc7922ba",
    "readout90.py": "4d5d124dd035842f68d20271875eee899731a5cd50b3b9cf28954eaf5327d7ee",
    "run90.py": "36d32933bb776a5181d3a6d2dc0f32a9639e148a37700b2b3bba5cc1db79af64",
    "train90.py": "4ce097f2f113f5a4c1a6702ee3ff6a2fe00c880ac2fb6b47242cbe3237be8e51"
  },
  "scope": "real held-out inputs and labels; untrained equations and gradients only, no performance inference",
  "TCN_remote_past_rows": 160,
  "methods": {
    "G": {
      "parameters": 265218,
      "paired": true,
      "causal_streaming": true,
      "independent_equations": true,
      "numpy_masked_loss": true
    },
    "T": {
      "parameters": 317058,
      "paired": true,
      "causal_streaming": true,
      "independent_equations": false,
      "numpy_masked_loss": true
    },
    "R": {
      "parameters": 24066,
      "paired": true,
      "causal_streaming": true,
      "independent_equations": true,
      "numpy_masked_loss": true
    },
    "C": {
      "parameters": 24066,
      "paired": true,
      "causal_streaming": true,
      "independent_equations": true,
      "numpy_masked_loss": true
    },
    "U": {
      "parameters": 344898,
      "paired": true,
      "causal_streaming": true,
      "independent_equations": false,
      "numpy_masked_loss": true
    },
    "V": {
      "parameters": 344898,
      "paired": true,
      "causal_streaming": true,
      "independent_equations": false,
      "numpy_masked_loss": true
    },
    "N": {
      "parameters": 365636,
      "paired": true,
      "causal_streaming": true,
      "independent_equations": false,
      "numpy_masked_loss": true
    }
  }
}
