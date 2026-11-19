# Read-only: loads the 641 KSTAR 10 ms files into one npz in the scratchpad (no tracked file is touched).
import os, sys, glob, numpy as np, pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "2")
data_dir = sys.argv[1]; out = sys.argv[2]
files = sorted(glob.glob(os.path.join(data_dir, "s*.csv")))
cols = None; rows = []; shot_ids = []
for f in files:
    df = pd.read_csv(f)
    if cols is None:
        cols = list(df.columns)
    assert list(df.columns) == cols, f
    # drop the first (t = 0) and last (t = 30 s) rows: not grid rows
    df = df.iloc[1:-1]
    if len(df) == 0:
        continue
    shot = int(os.path.basename(f)[1:6])
    rows.append(df.to_numpy(dtype=np.float64))
    shot_ids.append(np.full(len(df), shot, dtype=np.int32))
X = np.concatenate(rows); S = np.concatenate(shot_ids)
np.savez_compressed(out, X=X, shot=S, cols=np.array(cols))
print("files", len(files), "rows", X.shape, "cols", cols)
