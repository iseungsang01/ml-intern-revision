# Claude authored the implementation
# ces_prediction/experiments/b85_unified/common85.py
import os as _os
_cuda_prior = _os.environ.get('CUDA_VISIBLE_DEVICES', None)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Careful sys.path to avoid collisions
_b73_path = str(ROOT / 'ces_prediction' / 'experiments' / 'b73_targets')
if _b73_path not in sys.path:
    sys.path.insert(0, _b73_path)

import importlib
_common_mod = importlib.import_module('common')
C = _common_mod

_designs_mod = importlib.import_module('designs')
DS = _designs_mod

# Restore CUDA_VISIBLE_DEVICES that common.py may have cleared
if _cuda_prior is not None:
    _os.environ['CUDA_VISIBLE_DEVICES'] = _cuda_prior
elif 'CUDA_VISIBLE_DEVICES' in _os.environ:
    del _os.environ['CUDA_VISIBLE_DEVICES']

import numpy as np
import json
import hashlib
import zlib
from functools import lru_cache

DATA = ROOT / 'data'
CACHE = DATA / '.b85'
PAPER = ROOT / 'docs' / 'paper'

CACHE.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)

TRAIN_RULES = ['D10', 'D50', 'D100', 'D300', 'D650', 'D1000', 'S']
RULES = ['D10', 'D50', 'D100', 'D300', 'D650', 'D1000', 'S', 'K10']
INITS = [1701, 1702, 1703]
VARIANTS = ['X', 'H']


def select_units():
    """Return list of 15 selected unit dicts."""
    units = []
    for _ in [None]:
        all_u = DS.units()
        ulist, tab, fifth = all_u
        for u in ulist:
            name = u['name']
            if name in [
                'V1_r2_k0', 'V1_r2_k1', 'V1_r2_k2', 'V1_r2_k3', 'V1_r2_k4',
                'V2_s1_k0', 'V2_s1_k1', 'V2_s1_k2', 'V2_s1_k3', 'V2_s1_k4',
                'V3_q1', 'V3_q2', 'V3_q3', 'V3_q4',
                'V4a',
            ]:
                units.append(u)
    assert len(units) == 15, f'Expected 15 units, got {len(units)}'
    return units


@lru_cache(maxsize=1)
def load_sources():
    """Load G and R from .b73 cache."""
    b73_cache = DATA / '.b73'
    G = C.load_npz(b73_cache / 'grid.npz')
    R = C.load_npz(b73_cache / 'rows.npz')
    return G, R


def unit_design(name):
    """Return design string from unit name."""
    if name.startswith('V1'):
        return 'V1'
    elif name.startswith('V2'):
        return 'V2'
    elif name.startswith('V3'):
        return 'V3'
    elif name.startswith('V4'):
        return 'V4'
    return 'unknown'


def deterministic_seed(unit_name, init, epoch, segment_idx):
    """Deterministic seed for rule sampling."""
    s = f'{unit_name}_{init}_{epoch}_{segment_idx}'
    return zlib.crc32(s.encode()) & 0xFFFFFFFF


def rule_for(unit_name, init, epoch, segment_idx):
    """Uniform deterministic paired X/H rule selection per segment."""
    seed = deterministic_seed(unit_name, init, epoch, segment_idx)
    return TRAIN_RULES[seed % 7]


def val_split_seed(unit_name):
    """Seed for validation split."""
    return [20261185, zlib.crc32(unit_name.encode()) & 0xFFFFFFFF]


def make_val_split(unit, fit_ids):
    """Split fit_ids into train and val. Returns (train_ids, val_ids)."""
    name = unit['name']
    design = unit_design(name)
    fit_ids = sorted(fit_ids)
    n_val = max(1, int(np.ceil(0.15 * len(fit_ids))))

    if design in ('V1', 'V2'):
        seeds = val_split_seed(name)
        rng = np.random.default_rng(seeds)
        perm = rng.permutation(len(fit_ids))
        val_indices = sorted(perm[:n_val])
        val_ids = [fit_ids[i] for i in val_indices]
        train_ids = [f for f in fit_ids if f not in val_ids]
    else:
        # V3/V4: latest shot numbers -> need shot info from G
        G, R = load_sources()
        # Map file ids to shots
        file_shots = {}
        for fid in fit_ids:
            mask = G['fidx'] == fid
            if np.any(mask):
                shots = np.unique(R['shot'][np.isin(R['fidx'], [fid])] if hasattr(R, '__getitem__') else [])
                # Actually use G to get shot - but G doesn't have shot directly
                # R has shot for rows; use R.fidx
                rmask = R['fidx'] == fid
                if np.any(rmask):
                    s = R['shot'][rmask]
                    if len(s) > 0:
                        file_shots[fid] = int(s[0])
                    else:
                        file_shots[fid] = fid
                else:
                    file_shots[fid] = fid
            else:
                file_shots[fid] = fid
        # Sort by shot descending, take n_val
        sorted_by_shot = sorted(fit_ids, key=lambda f: file_shots.get(f, 0), reverse=True)
        val_ids = sorted_by_shot[:n_val]
        train_ids = [f for f in fit_ids if f not in val_ids]

    return sorted(train_ids), sorted(val_ids)


def stats_hash(stats):
    """Hash of stats dict for verification."""
    s = json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in sorted(stats.items())}, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def manifest_hash(manifest):
    """Hash of manifest for verification."""
    s = json.dumps(manifest, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()
