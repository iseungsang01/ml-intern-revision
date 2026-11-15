"""Temperature level at the KSTAR CES chord: per-discharge median T_i (descriptive).

All 641 discharge files; T_i values that are observed, positive and below 3 keV (the cut
population); discharges with at least 20 such values. Output: kstar_level.json.
Run from the repo root:  py docs/paper/ppcf/kstar_level.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
data_dir = os.environ.get("CES_DATA_DIR", "data")
files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
med = []
for f in files:
    t = pd.read_csv(f, usecols=["CES_TI"])["CES_TI"].dropna().to_numpy()
    t = t[(t > 0) & (t < 3000)]
    if len(t) >= 20:
        med.append(float(np.median(t)))
med = np.array(med)
out = {
    "n_files": len(files),
    "n_discharges": int(len(med)),
    "median_of_medians_eV": float(np.median(med)),
    "p10_eV": float(np.percentile(med, 10)),
    "p90_eV": float(np.percentile(med, 90)),
    "nominal_rho": (2.198 - 1.8) / 0.5,
    "note": "nominal rho from R = 2.198 m, R0 = 1.8 m, a = 0.5 m; not an equilibrium mapping",
}
(HERE / "kstar_level.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(out)
