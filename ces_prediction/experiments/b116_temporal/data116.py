"""Read-only reuse of frozen real sequence data and anonymous common support."""
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'data'
sys.path.insert(0, str(HERE.parent / 'b105_merit'))
import common105 as C105
import data88

UNITS = C105.UNITS
TRAIN_RULES = C105.TRAIN_RULES
RULES = C105.RULES


def source_guard():
    frozen = C105.guard()
    C105.inputs_guard()
    return frozen


@lru_cache(maxsize=1)
def load_prepared(unit):
    assert unit in UNITS
    z = C105.prepared(unit)
    assert z['identity']['width'] == 35
    m = z['identity']['manifest']
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not set(m[a]) & set(m[b])
    for role in ('train', 'val'):
        assert set(z[role]) == set(TRAIN_RULES)
        for segments in z[role].values():
            for s in segments:
                n = len(s['gi'])
                assert s['x'].shape == (n, 35)
                assert s['y'].shape == s['mask'].shape == (n, 2)
                assert s['x'].dtype == s['y'].dtype == np.float32
                assert s['mask'].dtype == np.bool_
                assert np.isfinite(s['x']).all() and np.isfinite(s['y']).all()
    return z


@lru_cache(maxsize=1)
def raw_sources():
    return C105.BASEM.load_sources()


def test_segments(unit, rule):
    assert rule in RULES
    z = load_prepared(unit)
    g, r = raw_sources()
    return data88.build_segments(z['identity']['manifest']['test'],
                                 z['identity']['stats'], rule, G=g, R=r)


def batches(segments, device, batch_size=8, shuffle_seed=None):
    """Yield full segments, padding only after their final real row."""
    assert batch_size > 0
    order = list(range(len(segments)))
    if shuffle_seed is not None:
        random.Random(int(shuffle_seed)).shuffle(order)
    for start in range(0, len(order), batch_size):
        chosen = [segments[i] for i in order[start:start + batch_size]]
        lengths = np.asarray([len(s['gi']) for s in chosen], dtype=np.int64)
        b, t = len(chosen), int(lengths.max())
        x = np.zeros((b, t, 35), dtype=np.float32)
        y = np.zeros((b, t), dtype=np.float32)
        mask = np.zeros((b, t), dtype=bool)
        gi = np.full((b, t), -1, dtype=np.int64)
        for i, s in enumerate(chosen):
            n = lengths[i]
            x[i, :n], y[i, :n], mask[i, :n] = s['x'], s['y'][:, 0], s['mask'][:, 0]
            gi[i, :n] = s['gi']
        yield dict(x=torch.from_numpy(x).to(device), y=torch.from_numpy(y).to(device),
                   mask=torch.from_numpy(mask).to(device), lengths=torch.from_numpy(lengths),
                   gi=torch.from_numpy(gi), fidx=[int(s['fidx']) for s in chosen],
                   segment=[int(s['segment']) for s in chosen])


@lru_cache(maxsize=1)
def _row_mapping():
    p = DATA / 'colab_b111/private_map.json'
    private = json.loads(p.read_text(encoding='utf8'))
    bind = json.loads((DATA / 'colab_b112/private_map.json').read_text(encoding='utf8'))
    assert C105.sha(p) == bind['mapping_source_sha256']
    codes = np.asarray(private['row_codes'], dtype=np.int64)
    inverse = np.asarray(private['row_inverse'], dtype=np.int64)
    assert np.array_equal(codes[inverse], np.arange(len(codes)))
    g, r = raw_sources()
    lookup = np.full(len(g['time']), -1, dtype=np.int64)
    lookup[r['gi']] = np.arange(len(r['gi']))
    assert len(codes) == len(r['gi'])
    return lookup, codes


def anonymous_indices(gi):
    gi = np.asarray(gi, dtype=np.int64)
    lookup, codes = _row_mapping()
    out = np.full(gi.shape, -1, dtype=np.int64)
    valid = (gi >= 0) & (gi < len(lookup))
    rows = lookup[gi[valid]]
    mapped = np.full(rows.shape, -1, dtype=np.int64)
    mapped[rows >= 0] = codes[rows[rows >= 0]]
    out[valid] = mapped
    return out


def common_row_support(unit, rule, split='test'):
    assert unit in UNITS and rule in RULES and split in ('train', 'val', 'test')
    assert rule != 'K10' or split == 'test'
    tag = 'U' + str(UNITS.index(unit))
    base = DATA / 'colab_b112/bundle'
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf8'))
    assert manifest['units'][tag]['name'] == unit
    with np.load(base / f'{tag}_{rule}.npz', allow_pickle=False) as z:
        keep = z['split'] == {'train': 0, 'val': 1, 'test': 2}[split]
        result = {k: z[k][keep].copy() for k in ('idx', 'cluster', 'discharge')}
        result['truth'], result['gp'] = z['y'][keep].copy(), z['gp'][keep].copy()
    if split == 'test':
        with np.load(base / f'{tag}_{rule}_reference.npz', allow_pickle=False) as z:
            ref = {k: z[k].copy() for k in z.files}
        for key in ('idx', 'truth', 'cluster', 'discharge'):
            assert np.array_equal(ref[key], result[key])
        result.update(ref)
    assert len(np.unique(result['idx'])) == len(result['idx'])
    return result


def _align(unit, rule, split, segments, values):
    assert len(segments) == len(values)
    ids, arrays = [], []
    for s, v in zip(segments, values):
        v = np.asarray(v)
        assert len(v) == len(s['gi'])
        idx = anonymous_indices(s['gi'])
        keep = idx >= 0
        ids.append(idx[keep])
        arrays.append(v[keep])
    idx, values = np.concatenate(ids), np.concatenate(arrays)
    assert len(np.unique(idx)) == len(idx)
    order = np.argsort(idx)
    ref = common_row_support(unit, rule, split)
    pos = np.searchsorted(idx[order], ref['idx'])
    assert (pos < len(order)).all()
    assert np.array_equal(idx[order[pos]], ref['idx']), 'Missing original common-support row'
    return ref, values[order[pos]]


def predictions_to_rows(unit, rule, role, segments, preds, normalized=False):
    ref, pred = _align(unit, rule, role, segments, preds)
    if normalized:
        stats = load_prepared(unit)['identity']['stats']
        pred = pred * stats['ti_sd'] + stats['ti_mu']
    assert pred.ndim == 1 and np.isfinite(pred).all()
    return {**ref, 'prediction': pred}


def aligned_inputs(unit, rule, split):
    segments = (test_segments(unit, rule) if split == 'test'
                else load_prepared(unit)[split][rule])
    ref, x = _align(unit, rule, split, segments, [s['x'] for s in segments])
    return {**ref, 'x': x}


def check_cluster_contract():
    """Require stable cluster identities across every overlapping anonymous row."""
    seen, discharge_cluster = {}, {}
    for unit in UNITS:
        for rule in RULES:
            splits = ('test',) if rule == 'K10' else ('train', 'val', 'test')
            for split in splits:
                r = common_row_support(unit, rule, split)
                for idx, cluster, discharge in zip(r['idx'].tolist(), r['cluster'].tolist(), r['discharge'].tolist()):
                    if idx in seen:
                        assert seen[idx] == (cluster, discharge), 'Inconsistent global group identity'
                    if discharge in discharge_cluster:
                        assert discharge_cluster[discharge] == cluster, 'Discharge spans conflicting clusters'
                    seen[idx] = (cluster, discharge)
                    discharge_cluster[discharge] = cluster
    return {'unique_rows': len(seen), 'consistent': True,
            'row_cluster_discharge_consistent': True, 'discharge_cluster_consistent': True}
