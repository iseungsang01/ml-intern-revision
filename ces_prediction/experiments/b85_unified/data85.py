# Claude authored the implementation
# ces_prediction/experiments/b85_unified/data85.py
import sys
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'ces_prediction' / 'experiments' / 'b85_unified'))

from common85 import (
    CACHE, DATA, RULES, TRAIN_RULES, load_sources, select_units,
    make_val_split, rule_for, unit_design
)


def unit_manifest(u):
    """Return dict with train, val, test integer file-id lists."""
    fit_ids = [int(i) for i in u['fit']]
    test_ids = [int(i) for i in u['test']]
    train_ids, val_ids = make_val_split(u, fit_ids)
    return {
        'train': sorted(train_ids),
        'val': sorted(val_ids),
        'test': sorted(test_ids),
        'name': u['name'],
    }


def fit_stats(train_file_ids, G=None, R=None):
    """Compute normalization stats from training files only.
    Returns dict: fast_mu(13), fast_sd(13), ti_mu, ti_sd, train_files."""
    if G is None or R is None:
        from common85 import load_sources
        G, R = load_sources()

    mask = np.isin(G['fidx'], train_file_ids)
    fast = G['fast'][mask]  # (N, 13)
    ti = G['ti'][mask]

    # Only finite values
    fast_finite = np.where(np.isfinite(fast), fast, np.nan)
    fast_mu = np.nanmean(fast_finite, axis=0).astype(np.float32)
    fast_sd = np.nanstd(fast_finite, axis=0).astype(np.float32)
    fast_sd = np.maximum(fast_sd, 1e-6)

    ti_finite = ti[np.isfinite(ti)]
    if not len(ti_finite):
        raise RuntimeError('No finite training Ti values')
    ti_mu = float(np.mean(ti_finite))
    ti_sd = float(np.std(ti_finite))
    ti_sd = max(ti_sd, 1e-6)

    return {
        'fast_mu': fast_mu.tolist(),
        'fast_sd': fast_sd.tolist(),
        'ti_mu': float(ti_mu),
        'ti_sd': float(ti_sd),
        'train_files': sorted(int(i) for i in train_file_ids),
    }


def _load_frames():
    """Load .b78/frames.npz for K10 rule."""
    p = DATA / '.b78' / 'frames.npz'
    if p.exists():
        return dict(np.load(p, allow_pickle=True))
    raise FileNotFoundError(p)


_ANCHOR_CACHE = {}


def anchors(rule, G=None, file_ids=None):
    """Latest finite strictly previous observation under the registered cutoff.

    Local contract repair of Claude source: vectorized search, including S cutoffs
    whose own row has no Ti. Modified-grid audit calls never reuse cached anchors.
    """
    base, _ = load_sources()
    G = base if G is None else G
    if G is base and rule in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[rule]
    if rule not in RULES:
        raise ValueError(rule)
    tm, seg, ti = G['time'], G['seg'], G['ti']
    out = np.full(len(tm), -1, dtype=np.int64)
    fr = _load_frames()['fr'] if rule == 'K10' else None
    if fr is not None and len(fr) != len(tm):
        raise RuntimeError('K10 frame cache length mismatch')
    starts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1]
    ends = np.r_[starts[1:], len(tm)]
    for a, b in zip(starts, ends):
        ii = np.arange(a, b)
        obs = ii[np.isfinite(ti[a:b])]
        if rule == 'K10':
            obs = obs[(fr[obs] >= 0) & (fr[obs] % 10 == 0)]
            pos = np.searchsorted(obs, ii, side='left') - 1
        elif rule == 'S':
            dt = np.r_[0., np.diff(tm[a:b])]
            cutoff = np.maximum.accumulate(np.where((dt > .015) & (dt < .045), ii-1, -1))
            pos = np.searchsorted(obs, cutoff, side='right') - 1
        else:
            lim = tm[a:b] - int(rule[1:])/1000. + .003
            pos = np.searchsorted(tm[obs], lim, side='right') - 1
            pos = np.minimum(pos, np.searchsorted(obs, ii, side='left') - 1)
        valid = pos >= 0
        if len(obs):
            out[ii[valid]] = obs[pos[valid]]
    if G is base:
        _ANCHOR_CACHE[rule] = out
    return out



def _build_x_row(gi, anc_gi, G, stats, seg_start_time, row_time, has_visible, visible_ti, age_s):
    """Build single x row (22,) float32."""
    x = np.zeros(22, dtype=np.float32)
    # Slots 0:13 - standardized fast
    fast = G['fast'][gi].astype(np.float32)
    x[0:13] = (fast - stats['fast_mu']) / stats['fast_sd']
    # Slots 13:15 - zero MC
    x[13:15] = 0.0
    # Slot 15 - log1p(dt_sec), first segment row = 0
    if gi == seg_start_time:
        x[15] = 0.0
    else:
        dt = row_time - seg_start_time
        x[15] = np.log1p(max(dt, 0.0)).astype(np.float32)
    # Slot 16 - normalized visible Ti
    if has_visible:
        x[16] = (visible_ti - stats['ti_mu']) / stats['ti_sd']
    else:
        x[16] = 0.0
    # Slot 17 - log1p(age_sec)
    x[17] = np.log1p(max(age_s, 0.0)).astype(np.float32) if has_visible else 0.0
    # Slot 18 - has visible
    x[18] = 1.0 if has_visible else 0.0
    # Slots 19:22 - zero VT history
    x[19:22] = 0.0
    return x


def build_segments(file_ids, stats, rule, G=None, R=None):
    """Build list of segment dicts for given file_ids and rule.
    Each dict: x(n,22)f32, y(n,2)f32, mask(n,2)bool, gi(n)int, fidx int, segment int."""
    if G is None or R is None:
        from common85 import load_sources
        G, R = load_sources()

    anc = anchors(rule, G)
    segments = []

    # Build R lookup
    r_gi_set = set(R['gi'].tolist()) if 'gi' in R else set()
    r_lookup = {}
    if 'gi' in R:
        for idx in range(len(R['gi'])):
            r_lookup[int(R['gi'][idx])] = idx

    for fid in file_ids:
        fmask = G['fidx'] == fid
        findices = np.where(fmask)[0]
        if len(findices) == 0:
            continue

        # Group by segment
        file_segs = G['seg'][findices]
        for s in np.unique(file_segs):
            smask = file_segs == s
            sidx = findices[smask]
            sidx = np.sort(sidx)
            n = len(sidx)
            if n == 0:
                continue

            x = np.zeros((n, 22), dtype=np.float32)
            y = np.zeros((n, 2), dtype=np.float32)
            mask = np.zeros((n, 2), dtype=bool)
            gi_arr = sidx.copy()

            seg_start_time = float(G['time'][sidx[0]])

            for i, gi in enumerate(sidx):
                row_time = float(G['time'][gi])
                a = anc[gi]
                has_vis = False
                vis_ti = 0.0
                age_s = 0.0

                if a >= 0 and a < gi:  # strictly earlier
                    has_vis = True
                    vis_ti = float(G['ti'][a])
                    age_s = row_time - float(G['time'][a])

                # dt from previous row in segment
                if i == 0:
                    dt_prev = 0.0
                else:
                    dt_prev = row_time - float(G['time'][sidx[i-1]])

                x_row = np.zeros(22, dtype=np.float32)
                # Slots 0:13 standardized fast (context: past/current fast)
                fast_vals = G['fast'][gi].astype(np.float32)
                x_row[0:13] = (fast_vals - stats['fast_mu']) / stats['fast_sd']
                x_row[13:15] = 0.0  # zero MC
                x_row[15] = np.float32(np.log1p(dt_prev)) if i > 0 else 0.0
                if has_vis:
                    x_row[16] = np.float32((vis_ti - stats['ti_mu']) / stats['ti_sd'])
                    x_row[17] = np.float32(np.log1p(age_s))
                    x_row[18] = 1.0
                else:
                    x_row[16:19] = 0.0
                x_row[19:22] = 0.0
                x[i] = x_row

                # Labels from R
                if gi in r_lookup:
                    ri = r_lookup[gi]
                    # F target
                    yf = R['y'][ri] if 'y' in R else np.nan
                    # Handle y shape
                    if hasattr(yf, '__len__'):
                        y[i, 0] = float(yf[0]) if len(yf) > 0 else np.nan
                    else:
                        y[i, 0] = float(yf)

                    # L5 target - tg_L5
                    if 'tg_L5' in R:
                        yl5 = R['tg_L5'][ri]
                        y[i, 1] = float(yl5)
                    else:
                        y[i, 1] = np.nan

                    # F target from tg_F if available
                    if 'tg_F' in R:
                        y[i, 0] = float(R['tg_F'][ri])

                    mask[i, 0] = np.isfinite(y[i, 0])
                    mask[i, 1] = np.isfinite(y[i, 1])
                else:
                    y[i] = 0.0
                    mask[i] = False

            # Protocol compatibility fix: both heads use the same normalized Ti scale.
            y = np.where(mask, (y - stats['ti_mu']) / stats['ti_sd'], 0).astype(np.float32)
            segments.append({
                'x': x,
                'y': y,
                'mask': mask,
                'gi': gi_arr,
                'fidx': int(fid),
                'segment': int(s),
            })

    return segments


def feature_helper(file_ids, stats, rule, G=None, R=None, mutate_ti_fn=None):
    """Build segments with optional Ti mutation for audit.
    mutate_ti_fn(G_copy) -> modified G with mutated ti.
    Uses frozen stats regardless of mutation."""
    if G is None or R is None:
        from common85 import load_sources
        G, R = load_sources()

    if mutate_ti_fn is not None:
        import copy
        G_mut = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in G.items()}
        G_mut = mutate_ti_fn(G_mut)
        return build_segments(file_ids, stats, rule, G=G_mut, R=R)
    return build_segments(file_ids, stats, rule, G=G, R=R)
