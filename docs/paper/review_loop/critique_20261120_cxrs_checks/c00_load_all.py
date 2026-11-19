import os, sys, glob
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np, pandas as pd
data_dir = sys.argv[1]; out = sys.argv[2]
files = sorted(glob.glob(os.path.join(data_dir, "s*.csv")))
frames = []
for i, f in enumerate(files):
    df = pd.read_csv(f)
    shot = int(os.path.basename(f)[1:-4])
    df = df.iloc[1:-1].copy()  # drop first (t=0) and last (t=30 s) rows as file_rules.md says
    df.insert(0, "shot", shot)
    frames.append(df)
all_ = pd.concat(frames, ignore_index=True)
print("files", len(files), "grid rows", len(all_))
print(all_.columns.tolist())
all_.to_pickle(out)
