"""B87 matched-delay experiment; all commands run from the repository root."""
from pathlib import Path
import os
import sys
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'b73_targets'))
import common as C
import designs as DS
import stage_trees as T
import numpy as np

for k in list(os.environ):
    if k.startswith('CES_'):
        del os.environ[k]
os.environ.update(CES_DATA_DIR=str(C.DATA), CES_DROP_STUCK_TARGETS='1', CES_TI_SPIKE_CUT_EV='3000',
                  CES_PER_SHOT_NORM='0', CES_SEQ_NORM_EXPANDING='0', CES_SEQ_ZERO_MC='0', CUDA_VISIBLE_DEVICES='')
CACHE = C.DATA / '.b87'
ARMS = tuple(f'{a}{j}' for j in (0,1,2,5) for a in ('H','X'))

def rng(name):
    return np.random.default_rng([20261187, zlib.crc32(name.encode())])

def future(G, gi):
    """Find exact future timestamps without crossing file or segment boundaries."""
    out = np.full((len(gi),5), np.nan)
    source = np.full((len(gi),5), -1, dtype=np.int64)
    for f in np.unique(G['fidx'][gi]):
        q = np.flatnonzero(G['fidx'][gi] == f)
        a,b = G['off'][f:f+2]
        tm = G['time'][a:b]
        for j in range(1,6):
            want = G['time'][gi[q]] + .01*j
            p = np.searchsorted(tm, want-.003)+a
            pc = np.minimum(p,b-1)
            ok = (p < b) & (np.abs(G['time'][pc]-want)<.003) & (G['seg'][pc]==G['seg'][gi[q]])
            qq = q[ok]
            source[qq,j-1] = pc[ok]
            out[qq,j-1] = G['ti'][pc[ok]]
    return out, source

def matrix(H, fast, slots, arm):
    s = slots.copy()
    s[:,int(arm[1:]):] = np.nan
    return np.column_stack([H,fast,s]) if arm[0]=='X' else np.column_stack([H,s])

def groups(units, design):
    if design in ('V1','V2'):
        keys = C.REPEATS if design=='V1' else C.V2_SEEDS
        return [[u for u in units if u['design']==design and u['sub']==s] for s in keys]
    if design=='V3':
        return [[u for u in units if u['design']=='V3']]
    return [[u for u in units if u['name']==design]]
