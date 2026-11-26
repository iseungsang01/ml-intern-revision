"""Independent CPU equations, provenance and numerical readout audit for B116.

No fitting and no producer model import. Recurrent inference replay is explicitly
bounded to the first two complete eligible segments per unit/rule/model/seed;
input, target, support and saved-array checks cover every reported row.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import hashlib
import json
import time
import zlib
from functools import lru_cache
import numpy as np
import common116 as C
import data116 as D

ARCHS = ('lstm', 'decay_gru', 'innovation_gru')
ATOL, RTOL = .05, 1e-4


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checked_files(root, files):
    for rel, expected in files.items():
        p = (root / rel).resolve()
        assert p.is_relative_to(root.resolve()) and C.sha(p) == expected, rel


def recurrent_predict(weights, arch, x):
    """Functional CPU gate equations, deliberately independent of models116."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    p = {k.removeprefix('model.'): torch.as_tensor(v, dtype=torch.float32) for k, v in weights.items()}
    assert all(torch.isfinite(v).all() for v in p.values())
    def lin(v, name):
        return F.linear(v, p[name + '.weight'], p[name + '.bias'])
    def head(v):
        v = F.layer_norm(v, (64,), p['readout.0.weight'], p['readout.0.bias'], 1e-5)
        return lin(F.gelu(lin(v, 'readout.1'), approximate='none'), 'readout.3')
    def gru(v, h):
        a = F.linear(v, p['recurrent.weight_ih'], p['recurrent.bias_ih'])
        b = F.linear(h, p['recurrent.weight_hh'], p['recurrent.bias_hh'])
        ar, az, an = a.chunk(3, -1)
        br, bz, bn = b.chunk(3, -1)
        r, z = torch.sigmoid(ar + br), torch.sigmoid(az + bz)
        n = torch.tanh(an + r * bn)
        return n + z * (h - n)
    with torch.no_grad():
        xx = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        h, c = torch.zeros(1, 64), torch.zeros(1, 64)
        old_age, old_has = torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.bool)
        result = []
        for row in xx:
            v = row[None]
            if arch == 'lstm':
                gates = F.linear(v, p['recurrent.weight_ih_l0'], p['recurrent.bias_ih_l0'])
                gates += F.linear(h, p['recurrent.weight_hh_l0'], p['recurrent.bias_hh_l0'])
                i, f, g, o = gates.chunk(4, -1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                h = torch.sigmoid(o) * torch.tanh(c)
            elif arch == 'decay_gru':
                dt = torch.expm1(v[:, 15:16]).clamp_min(0)
                decay = torch.exp(-(F.softplus(lin(v[:, [15, 17, 18]], 'rate')) * dt).clamp_max(80))
                h = gru(v, h * decay)
            else:
                assert arch == 'innovation_gru'
                has = v[:, 18:19] > .5
                age = torch.expm1(v[:, 17:18]).clamp_min(0)
                dt = torch.expm1(v[:, 15:16]).clamp_min(0)
                transition = v.clone()
                transition[:, 16] = 0
                prior = gru(transition, h)
                event = has & (~old_has | (dt + old_age - age > 1e-5))
                gain = torch.sigmoid(lin(torch.cat((prior, v[:, [15, 17, 18]]), -1), 'gain'))
                h = prior + event.float() * gain * torch.tanh(lin(v[:, 16:17] - head(prior), 'innovation'))
                old_age, old_has = age, has
            result.append(float(head(h)[0, 0]))
    result = np.asarray(result, dtype=np.float64)
    assert np.isfinite(result).all()
    return result


@lru_cache(maxsize=8)
def independent_anchors(rule):
    g, _ = D.raw_sources()
    tm, seg, ti = g['time'], g['seg'], g['ti']
    out = np.full(len(tm), -1, dtype=np.int64)
    fr = C.npz(C.ROOT / 'data/.b78/frames.npz')['fr'] if rule == 'K10' else None
    starts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1]
    for start, end in zip(starts, np.r_[starts[1:], len(tm)]):
        rows = np.arange(start, end)
        observed = rows[np.isfinite(ti[start:end])]
        if rule == 'K10':
            observed = observed[(fr[observed] >= 0) & (fr[observed] % 10 == 0)]
            pos = np.searchsorted(observed, rows, side='left') - 1
        elif rule == 'S':
            dt = np.r_[0., np.diff(tm[start:end])]
            cutoff = np.maximum.accumulate(np.where((dt > .015) & (dt < .045), rows - 1, -1))
            pos = np.searchsorted(observed, cutoff, side='right') - 1
        else:
            pos = np.searchsorted(tm[observed], tm[start:end] - int(rule[1:]) / 1000 + .003, side='right') - 1
            pos = np.minimum(pos, np.searchsorted(observed, rows, side='left') - 1)
        ok = pos >= 0
        if len(observed):
            out[rows[ok]] = observed[pos[ok]]
    return out


def verify_normalization(prepared):
    g, _ = D.raw_sources()
    identity = prepared['identity']
    train = identity['manifest']['train']
    fit = np.isin(g['fidx'], train)
    fast = np.where(np.isfinite(g['fast'][fit]), g['fast'][fit], np.nan)
    ti = g['ti'][fit]
    ti = ti[np.isfinite(ti)]
    expected = {'fast_mu': np.nanmean(fast, axis=0).astype(np.float32).tolist(),
                'fast_sd': np.maximum(np.nanstd(fast, axis=0).astype(np.float32), 1e-6).tolist(),
                'ti_mu': float(np.mean(ti)), 'ti_sd': float(max(np.std(ti), 1e-6)),
                'train_files': sorted(int(i) for i in train)}
    assert expected == identity['stats'], 'Normalization differs from TRAIN-only raw recomputation'
    return digest(expected)


def verify_segments(unit, rule, split, segments, require_full_support=True):
    """Reconstruct all35 input columns and F labels independently from raw data."""
    prepared = D.load_prepared(unit)
    stats = prepared['identity']['stats']
    g, r = D.raw_sources()
    lookup = np.full(len(g['time']), -1, dtype=np.int64)
    lookup[r['gi']] = np.arange(len(r['gi']))
    anc = independent_anchors(rule)
    total = labels = 0
    support = D.common_row_support(unit, rule, split)
    observed_ids, observed_truth = [], []
    for s in segments:
        gi, x = np.asarray(s['gi']), s['x']
        assert s['fidx'] in prepared['identity']['manifest'][split]
        assert np.all(g['fidx'][gi] == s['fidx']) and np.all(g['seg'][gi] == s['segment'])
        assert np.all(np.diff(gi) > 0)
        a = anc[gi]
        has = a >= 0
        assert np.all(a[has] < gi[has])
        assert np.array_equal(g['seg'][a[has]], g['seg'][gi[has]])
        assert np.array_equal(g['fidx'][a[has]], g['fidx'][gi[has]])
        exp = np.zeros_like(x)
        exp[:, :13] = (g['fast'][gi].astype(np.float32) - stats['fast_mu']) / stats['fast_sd']
        exp[:, 15] = np.log1p(np.r_[0., np.diff(g['time'][gi])]).astype(np.float32)
        exp[has, 16] = ((g['ti'][a[has]].astype(np.float64) - stats['ti_mu']) / stats['ti_sd']).astype(np.float32)
        exp[has, 17] = np.log1p(g['time'][gi[has]] - g['time'][a[has]]).astype(np.float32)
        exp[:, 18] = has
        exp[has, 22:] = (g['fast'][a[has]].astype(np.float32) - stats['fast_mu']) / stats['fast_sd']
        assert np.allclose(x, exp, rtol=0, atol=2e-6), 'Raw input reconstruction mismatch'
        ri = lookup[gi]
        truth = np.full(len(gi), np.nan)
        present = ri >= 0
        raw_y = r['tg_F'] if 'tg_F' in r else r['y']
        if np.asarray(raw_y).ndim == 2:
            raw_y = raw_y[:, 0]
        truth[present] = np.asarray(raw_y[ri[present]], dtype=np.float32)
        mask = np.isfinite(truth)
        assert np.array_equal(mask, s['mask'][:, 0])
        yy = np.where(mask, (truth.astype(np.float32) - stats['ti_mu']) / stats['ti_sd'], 0).astype(np.float32)
        assert np.array_equal(yy, s['y'][:, 0]), 'F target reconstruction mismatch'
        ids = D.anonymous_indices(gi)
        selected = mask & np.isin(ids, support['idx'])
        observed_ids.extend(ids[selected].tolist())
        observed_truth.extend(np.asarray(raw_y[ri[selected]], dtype=np.float64).tolist())
        total += len(gi)
        labels += int(mask.sum())
    observed_ids, observed_truth = np.asarray(observed_ids), np.asarray(observed_truth)
    order = np.argsort(observed_ids)
    if require_full_support:
        assert np.array_equal(observed_ids[order], np.sort(support['idx'])), 'Missing/duplicated observed F common-support row'
        where = np.searchsorted(observed_ids[order], support['idx'])
        assert np.array_equal(observed_truth[order][where], support['truth']), 'Common-support raw F truth mismatch'
    return {'segments': len(segments), 'context_rows': total, 'F_labels': labels, 'common_rows': len(support['idx'])}


def verify_history(unit, arch, seed, prepared):
    folder = C.OUT / 'fits' / unit / arch / ('s' + str(seed))
    p, h = C.read(folder / 'params.json'), C.read(folder / 'history.json')
    assert p['normal'] and 1 <= len(h) <= 60 and len(h) == p['completed_epochs']
    assert p['unit'] == unit and p['architecture'] == arch and p['seed'] == seed
    assert p['best_sha256'] == C.sha(folder / 'best.npz')
    from datetime import datetime
    start = C.read(folder / 'started.json')
    assert datetime.fromisoformat(start['utc']) <= datetime.fromisoformat(p['completed_utc'])
    if 'history_sha256' in p:
        assert p['history_sha256'] == C.sha(folder / 'history.json')
    assert p['stats_sha256'] == digest(prepared['identity']['stats'])
    assert p['manifest_sha256'] == digest(prepared['identity']['manifest'])
    assert p['prepared_artifact_sha256'] == prepared['artifact_sha256']
    assert (p['budget_seconds'], p['max_epochs'], p['batch_size']) == (900, 60, 16)
    assert p['target'] == 'F' and p['precision'] == 'float32' and not any(p[k] for k in ('cudnn', 'tf32', 'amp'))
    assert p['threads'] == 2
    assert p['optimizer'] == {'name': 'AdamW', 'initial_lr': .001, 'weight_decay': .0001, 'clip_grad_norm': 1.0}
    assert p['scheduler'] == {'name': 'ReduceLROnPlateau', 'patience': 3, 'factor': .5, 'min_lr': 1e-5}
    scores = np.asarray([v['seven_rule_mean_mse'] for v in h])
    assert p['best_epoch'] == int(np.argmin(scores)) + 1
    assert p['best_val_mse'] == float(scores.min())
    lr, best, bad = .001, float('inf'), 0
    # Plateau uses relative threshold1e-4 (PyTorch default), unlike strict checkpoint selection.
    for epoch, entry in enumerate(h, 1):
        assert entry['epoch'] == epoch and np.isclose(entry['lr_used'], lr, rtol=0, atol=1e-14)
        assert set(entry['val']) == set(C.TRAIN_RULES)
        vals = []
        for rule, v in entry['val'].items():
            assert v['count'] == len(D.common_row_support(unit, rule, 'val')['idx'])
            assert np.isclose(v['mse'], v['sse'] / v['count'], rtol=1e-12)
            vals.append(v['mse'])
        score = float(np.mean(vals))
        assert np.isclose(score, entry['seven_rule_mean_mse'], rtol=1e-12)
        if score < best * (1 - 1e-4):
            best, bad = score, 0
        else:
            bad += 1
        if bad > 3:
            new_lr = max(lr * .5, 1e-5)
            if lr - new_lr > 1e-8:
                lr = new_lr
            bad = 0
        assert np.isclose(entry['lr_next'], lr, rtol=0, atol=1e-14)
        assert entry['train_count'] > 0 and np.isfinite(entry['train_normalized_sse'])
        lookups = {(s['fidx'], s['segment']) for s in prepared['train'][C.TRAIN_RULES[0]]}
        keys = sorted(lookups)
        rng = np.random.default_rng([seed, epoch, zlib.crc32(unit.encode())])
        choices = rng.integers(0, len(C.TRAIN_RULES), len(keys))
        assignment = [(f, s, C.TRAIN_RULES[int(i)]) for (f, s), i in zip(keys, choices)]
        assert entry['assignment_sha256'] == digest(assignment)
    assert p['total_updates'] == sum(v['updates'] for v in h) + p['discarded_partial_updates']
    assert len(h) == 60 or p['actual_seconds'] >= 900
    assert p['stop_reason'] in ('max_epochs', 'deadline_before_epoch', 'deadline_in_incomplete_epoch')
    return p


def replay_segments(unit, rule, segments, arr, states):
    stats = D.load_prepared(unit)['identity']['stats']
    wanted = set(arr['idx'].tolist())
    chosen = []
    for s in segments:
        ids = D.anonymous_indices(s['gi'])
        keep = np.asarray(s['mask'][:, 0]) & np.isin(ids, arr['idx'])
        if keep.any():
            chosen.append((s, ids, keep))
        if len(chosen) == 2:
            break
    assert len(chosen) == 2, 'Need two complete real eligible segments'
    order = np.argsort(arr['idx'])
    result = {}
    for (arch, seed), state in states.items():
        name = 'Seq_' + arch + '_s' + str(seed)
        maximum = 0.
        count = 0
        for s, ids, keep in chosen:
            pred = recurrent_predict(state, arch, s['x']) * stats['ti_sd'] + stats['ti_mu']
            positions = order[np.searchsorted(arr['idx'][order], ids[keep])]
            expected = arr[name][positions]
            assert np.allclose(pred[keep], expected, atol=ATOL, rtol=RTOL), unit + '/' + rule + '/' + name
            maximum = max(maximum, float(np.max(np.abs(pred[keep] - expected))))
            count += int(keep.sum())
        result[name] = {'max_abs_eV': maximum, 'scored_rows': count,
                        'complete_segments': 2, 'context_rows': sum(len(s['gi']) for s, _, _ in chosen)}
    return result


def compare_nested(actual, expected, path=''):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, path + '/' + key
            compare_nested(actual[key], value, path + '/' + key)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare_nested(a, b, path + '/' + str(i))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-6), path
    else:
        assert actual == expected, path

ENSEMBLES = tuple('Seq_' + a + '_ensemble' for a in ('decay_gru', 'innovation_gru', 'lstm'))
SINGLES = tuple('Seq_' + a + '_s' + str(s) for a in ('decay_gru', 'innovation_gru', 'lstm') for s in C.SEEDS)
SELECTED, GP = 'Seq_selected_ensemble', 'GP_fast_boost'
MODELS = ENSEMBLES + SINGLES + (SELECTED, GP)
PAIRS = tuple((a, GP) for a in ENSEMBLES + SINGLES + (SELECTED,)) + ((SELECTED, 'Seq_lstm_ensemble'),)


def independent_statistics(cells, key, models=MODELS, pairs=PAIRS, groupings=('cluster', 'discharge')):
    """Separate bincount aggregation and shared global-cluster bootstrap."""
    arrays = list(cells.values())
    losses = [np.column_stack([(np.asarray(a[n], dtype=np.float64) - a['truth']) ** 2 for n in models]) for a in arrays]
    point = np.mean([a.mean(0) for a in losses], 0)
    result = {'cells': list(cells), 'cell_count': len(cells), 'models': {}, 'pairs': {}, 'bootstrap': {}}
    for j, name in enumerate(models):
        result['models'][name] = {'MSE': float(point[j]), 'RMSE': float(np.sqrt(point[j])),
            'bias': float(np.mean([np.mean(a[name] - a['truth']) for a in arrays]))}
    for a, b in pairs:
        result['pairs'][a + '_over_' + b] = {'point': float(1 - point[models.index(a)] / point[models.index(b)])}
    for grouping in groupings:
        labels = np.unique(np.concatenate([a[grouping] for a in arrays]))
        ng = len(labels)
        count, sums = [], []
        for a, loss in zip(arrays, losses):
            inv = np.searchsorted(labels, a[grouping])
            count.append(np.bincount(inv, minlength=ng))
            sums.append(np.column_stack([np.bincount(inv, weights=loss[:, j], minlength=ng) for j in range(len(models))]))
        seed = [20261126, zlib.crc32((key + '/' + grouping).encode())]
        rng = np.random.default_rng(seed)
        sampled, accepted, generated, rejected = [], 0, 0, 0
        while accepted < 20000:
            n = min(500, 20000 - accepted)
            w = rng.multinomial(ng, np.full(ng, 1 / ng), size=n)
            denominators = np.column_stack([np.dot(w, c) for c in count])
            keep = (denominators > 0).all(1)
            generated += n
            rejected += int((~keep).sum())
            assert generated < 2000000
            if keep.any():
                m = sum(np.dot(w[keep], s) / denominators[keep, i, None] for i, s in enumerate(sums)) / len(arrays)
                sampled.append(m)
                accepted += len(m)
        draws = np.concatenate(sampled)
        q = np.quantile(draws, [.025, .975], axis=0)
        for j, name in enumerate(models):
            result['models'][name][grouping + '_MSE95'] = q[:, j].tolist()
            result['models'][name][grouping + '_RMSE95'] = np.sqrt(q[:, j]).tolist()
        for a, b in pairs:
            interval = np.quantile(1 - draws[:, models.index(a)] / draws[:, models.index(b)], [.025, .975])
            result['pairs'][a + '_over_' + b][grouping] = {'ci95': interval.tolist(),
                'excludes_zero': bool(interval[0] > 0 or interval[1] < 0)}
        result['bootstrap'][grouping] = {'groups': ng, 'accepted': accepted, 'generated': generated,
            'zero_support_redraws': rejected, 'seed': seed}
    return result


def input_features(x):
    x = np.asarray(x, dtype=np.float64)
    return {'age': np.expm1(x[:, 17]), 'fastchange': np.sqrt(np.mean((x[:, :13] - x[:, 22:]) ** 2, 1)),
            'fastlevel': np.sqrt(np.mean(x[:, :13] ** 2, 1))}


def independent_cuts(unit):
    features = {k: [] for k in ('age', 'fastchange', 'fastlevel')}
    weights = {k: [] for k in features}
    for rule in C.TRAIN_RULES:
        x = D.aligned_inputs(unit, rule, 'train')['x']
        for key, value in input_features(x).items():
            keep = x[:, 18] >= .5 if key == 'age' else np.ones(len(x), dtype=bool)
            value = value[keep]
            features[key].append(value)
            weights[key].append(np.full(len(value), 1 / (7 * len(value))))
    result = {}
    for key, pieces in features.items():
        v = np.concatenate(pieces)
        w = np.concatenate(weights[key])
        order = np.argsort(v, kind='stable')
        result[key] = v[order][np.searchsorted(np.cumsum(w[order]) / w.sum(), [1 / 3, 2 / 3])].tolist()
    return result


def verify_failure(arr, x, cuts, reported, key):
    masks = {}
    for name, value in input_features(x).items():
        lo, hi = cuts[name]
        eligible = x[:, 18] >= .5 if name == 'age' else np.ones(len(x), dtype=bool)
        masks[name + '/low'] = eligible & (value <= lo)
        masks[name + '/middle'] = eligible & (value > lo) & (value <= hi)
        masks[name + '/high'] = eligible & (value > hi)
    masks['history/absent'], masks['history/present'] = x[:, 18] < .5, x[:, 18] >= .5
    masks['dt/normal'], masks['dt/large'] = np.expm1(x[:, 15]) <= .015, np.expm1(x[:, 15]) > .015
    assert set(masks) == set(reported)
    for label, mask in masks.items():
        row = reported[label]
        ng = len(np.unique(arr['cluster'][mask]))
        assert row['rows'] == int(mask.sum()) and row['clusters'] == ng and row['interval_available'] == (ng >= 8)
        assert row['discharges'] == len(np.unique(arr['discharge'][mask]))
        if mask.any():
            mses = {}
            for name in (SELECTED, GP):
                error = arr[name][mask] - arr['truth'][mask]
                mses[name] = float(np.mean(error ** 2))
                compare_nested(row['models'][name], {'MSE': mses[name], 'RMSE': float(np.sqrt(mses[name])), 'bias': float(np.mean(error))})
            compare_nested(row['pairs'][SELECTED + '_over_' + GP], {'point': 1 - mses[SELECTED] / mses[GP]})
        if ng >= 8:
            sub = {k: np.asarray(v)[mask] for k, v in arr.items() if np.asarray(v).ndim == 1 and len(v) == len(mask)}
            expected = independent_statistics({key: sub}, key + '/' + label, (SELECTED, GP), ((SELECTED, GP),), ('cluster',))
            for field in ('models', 'pairs', 'bootstrap'):
                compare_nested(row[field], expected[field], key + '/' + label + '/' + field)


def main():
    started = time.time()
    sources = C.source_guard()
    C.verify_inputs()
    fit = C.read(C.OUT / 'fit_complete.json')
    complete = C.read(C.OUT / 'predictions_complete.json')
    pred_start = C.read(C.OUT / 'predict_started.json')
    selection = C.read(C.OUT / 'selection.json')
    report = C.read(C.OUT / 'readout.json')
    staged = C.read(C.OUT / 'readout_complete.json')
    assert fit['all_pass'] and len(fit['fits']) == 36 and not fit['test_accessed']
    assert complete['all_pass'] and len(complete['files']) == 32 and staged['all_pass'] and not staged['published']
    assert fit['sources'] == complete['sources'] == selection['sources'] == report['sources'] == sources
    fit_sha, selection_sha = C.sha(C.OUT / 'fit_complete.json'), C.sha(C.OUT / 'selection.json')
    assert selection['fit_complete_sha256'] == complete['fit_complete_sha256'] == pred_start['fit_complete_sha256'] == fit_sha
    assert complete['selection_sha256'] == pred_start['selection_sha256'] == staged['selection_sha256'] == report['selection_sha256'] == selection_sha
    assert not selection['TEST_used']
    # Hash barriers establish all36fits -> frozen VAL choice -> TEST access.
    checked_files(C.OUT, fit['files'])
    checked_files(C.OUT, complete['files'])
    checked_files(C.OUT, staged['files'])
    checked_files(C.OUT, selection['val_files'])
    for receipt in fit['fits']:
        assert receipt['normal'] and C.sha(C.OUT / receipt['path']) == receipt['sha256']
    seen_receipts = {(v['unit'], v['architecture'], v['seed']) for v in fit['fits']}
    assert seen_receipts == {(u, a, s) for u in C.UNITS for a in ARCHS for s in C.SEEDS}
    from datetime import datetime, timezone
    stamp = lambda x: datetime.fromisoformat(x).timestamp()
    assert stamp(fit['utc']) <= selection['created_unix'] <= stamp(pred_start['utc']) <= stamp(complete['utc'])
    smoke = C.read(C.OUT / 'smoke_independent.json')
    assert smoke['all_pass'] and smoke['sources'] == sources
    arrays, replay, training, data_checks = {}, {}, {}, {}
    global_ids, discharge_clusters = {}, {}
    for unit in C.UNITS:
        prepared = D.load_prepared(unit)
        normalization_sha = verify_normalization(prepared)
        cuts = independent_cuts(unit)
        compare_nested(selection['input_cuts'][unit], cuts)
        states, receipts = {}, {}
        for arch in ARCHS:
            for seed in C.SEEDS:
                receipt = verify_history(unit, arch, seed, prepared)
                assert receipt['sources'] == sources
                assert stamp(receipt['completed_utc']) <= stamp(fit['utc'])
                receipts[arch, seed] = receipt
                states[arch, seed] = C.npz(C.OUT / 'fits' / unit / arch / ('s' + str(seed)) / 'best.npz')
                training[unit + '/' + arch + '/' + str(seed)] = receipt
        scores = {a: [] for a in sorted(ARCHS)}
        seed_scores = {(a, s): [] for a in ARCHS for s in C.SEEDS}
        for split, rules in (('train', C.TRAIN_RULES), ('val', C.TRAIN_RULES), ('test', C.RULES)):
            for rule in rules:
                segments = D.test_segments(unit, rule) if split == 'test' else prepared[split][rule]
                key = unit + '/' + rule
                data_checks[key + '/' + split] = verify_segments(unit, rule, split, segments)
                if split == 'train':
                    continue
                arr = C.npz(C.OUT / split / (unit + '_' + rule + '.npz'))
                ref = D.common_row_support(unit, rule, split)
                for name, values in ref.items():
                    assert np.array_equal(arr[name], values), key + '/' + name
                for idx, cluster, discharge in zip(arr['idx'], arr['cluster'], arr['discharge']):
                    if int(discharge) in discharge_clusters:
                        assert discharge_clusters[int(discharge)] == int(cluster), 'A discharge cannot split across global shot clusters'
                    discharge_clusters[int(discharge)] = int(cluster)
                    identity = (int(cluster), int(discharge))
                    if int(idx) in global_ids:
                        assert global_ids[int(idx)] == identity
                    global_ids[int(idx)] = identity
                for arch in ARCHS:
                    ens = np.mean([arr['Seq_' + arch + '_s' + str(s)] for s in C.SEEDS], axis=0)
                    assert np.array_equal(ens, arr['Seq_' + arch + '_ensemble'])
                    if split == 'val':
                        scores[arch].append(float(np.mean((ens - arr['truth']) ** 2)))
                        for seed in C.SEEDS:
                            seed_scores[arch, seed].append(float(np.mean((arr['Seq_' + arch + '_s' + str(seed)] - arr['truth']) ** 2)))
                replay[key + '/' + split] = replay_segments(unit, rule, segments, arr, states)
                if split == 'test':
                    arr[SELECTED] = arr['Seq_' + selection['by_unit'][unit]['architecture'] + '_ensemble']
                    arrays[key] = arr
                    compare_nested(report['cells'][key], independent_statistics({key: arr}, key))
                    assert report['cells'][key]['rows'] == len(arr['truth'])
                    x = D.aligned_inputs(unit, rule, split)
                    assert np.array_equal(x['idx'], arr['idx'])
                    verify_failure(arr, x['x'], cuts, report['failure_strata'][key], key)
        means = {a: float(np.mean(v)) for a, v in scores.items()}
        chosen = min(means, key=lambda a: (means[a], a))
        compare_nested(selection['by_unit'][unit], {'architecture': chosen, 'score': means[chosen], 'candidate_scores': means, 'rule_scores': scores})
        for pair, values in seed_scores.items():
            assert np.isclose(np.mean(values), receipts[pair]['best_val_mse'], rtol=1e-10, atol=1e-5)
        data_checks[unit + '/normalization_sha256'] = normalization_sha
    scopes = {'overall_8rules': arrays, 'overall_D10': {k: v for k, v in arrays.items() if k.endswith('/D10')}}
    scopes.update({u + '/8rules': {k: v for k, v in arrays.items() if k.startswith(u + '/')} for u in C.UNITS})
    for scope, cells in scopes.items():
        compare_nested(report['averages'][scope], independent_statistics(cells, 'average/' + scope))
    expected_statements = {'R1': report['averages']['overall_D10']['pairs'][SELECTED + '_over_' + GP]['cluster']['ci95'][0] > 0,
        'R2': report['averages']['overall_8rules']['pairs'][SELECTED + '_over_' + GP]['cluster']['ci95'][0] > 0,
        'R3': report['averages']['overall_8rules']['pairs'][SELECTED + '_over_Seq_lstm_ensemble']['cluster']['ci95'][0] > 0}
    compare_nested(report['statements'], expected_statements)
    assert C.source_guard() == sources
    audit = {'all_pass': True, 'sources': sources, 'selection_sha256': selection_sha,
        'fit_complete_sha256': fit_sha, 'readout_sha256': C.sha(C.OUT / 'readout.json'),
        'staged_readout': staged['files'], 'training': training, 'data_checks': data_checks,
        'recurrent_replay': replay, 'replay_scope': 'First two complete eligible real segments per VAL/TEST unit/rule/architecture/seed; not every saved recurrent prediction',
        'full_row_scope': 'All35 inputs, F masks/targets, support IDs, baseline arrays, ensemble arithmetic, readout numerical cells and global macro intervals',
        'strata_scope': 'Independent TRAIN cuts, all TEST stratum counts/MSE/RMSE/bias/skill points, and all eligible cluster intervals',
        'atol_eV': ATOL, 'rtol': RTOL, 'global_rows': len(global_ids), 'global_discharges': len(discharge_clusters), 'seconds': time.time() - started}
    C.save(C.OUT / 'audit.json', audit)
    print('B116 independent audit passed')




def smoke_audit():
    """Qualify the actual post-update GPU smoke by separate CPU equations."""
    sources = C.source_guard()
    C.verify_inputs()
    smoke = C.read(C.OUT / 'smoke.json')
    assert smoke['all_pass'] and smoke['sources'] == sources and not smoke['test_accessed']
    assert smoke['train_segments'] == 2 and smoke['max_rows'] == 64
    unit, rule = smoke['unit'], smoke['rule']
    prepared = D.load_prepared(unit)
    verify_normalization(prepared)
    candidates = []
    for seg in prepared['train'][rule]:
        if np.asarray(seg['mask'])[:64, 0].any():
            candidates.append(seg)
        if len(candidates) == 2:
            break
    assert len(candidates) == 2
    checks = {}
    for arch in ARCHS:
        checked_files(C.OUT, smoke['architectures'][arch]['files'])
        weights = C.npz(C.OUT / 'smoke' / (arch + '_weights.npz'))
        prefix = C.npz(C.OUT / 'smoke' / (arch + '_prefix.npz'))
        maximum = 0.
        for i, segment in enumerate(candidates):
            n = min(64, len(segment['gi']))
            assert int(prefix['lengths'][i]) == n
            assert prefix['fidx'][i] == segment['fidx'] and prefix['segment'][i] == segment['segment']
            for name, source in [('x', segment['x'][:n]), ('gi', segment['gi'][:n]),
                                 ('y', segment['y'][:n, 0]), ('mask', segment['mask'][:n, 0])]:
                assert np.array_equal(prefix[name][i, :n], source), arch + '/smoke/' + name
            pred = recurrent_predict(weights, arch, segment['x'][:n])
            physical = pred * prepared['identity']['stats']['ti_sd'] + prepared['identity']['stats']['ti_mu']
            expected = prefix['pred'][i, :n].astype(np.float64) * prepared['identity']['stats']['ti_sd'] + prepared['identity']['stats']['ti_mu']
            assert np.allclose(physical, expected, atol=ATOL, rtol=RTOL), arch + '/independent_smoke'
            maximum = max(maximum, float(np.max(np.abs(physical - expected))))
            k = min(8, n)
            shorter = recurrent_predict(weights, arch, segment['x'][:k])
            assert np.array_equal(shorter, pred[:k]), 'Independent prefix causality'
        checks[arch] = {'all_pass': True, 'max_abs_eV': maximum, 'real_segments': 2}
    assert C.source_guard() == sources
    C.save(C.OUT / 'smoke_independent.json', {'all_pass': True, 'sources': sources,
        'smoke_sha256': C.sha(C.OUT / 'smoke.json'), 'architectures': checks,
        'scope': 'Actual TRAIN prefixes, post-one-update GPU saved predictions versus CPU functional recurrent equations',
        'atol_eV': ATOL, 'rtol': RTOL, 'TEST_accessed': False})
    print('B116 independent TRAIN smoke passed')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    (smoke_audit if args.smoke else main)()
