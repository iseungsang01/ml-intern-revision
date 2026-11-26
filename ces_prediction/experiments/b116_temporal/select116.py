"""Freeze within-unit VAL architecture choices and TRAIN-only input strata."""
import time
import numpy as np
import common116 as C
import data116 as D

ARCHES = ('decay_gru', 'innovation_gru', 'lstm')


def features(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == 35 and np.isfinite(x).all()
    return {'age': np.expm1(x[:, 17]),
            'fastchange': np.sqrt(np.mean((x[:, :13] - x[:, 22:35]) ** 2, axis=1)),
            'fastlevel': np.sqrt(np.mean(x[:, :13] ** 2, axis=1))}


def weighted_quantiles(values, weights):
    order = np.argsort(values, kind='stable')
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    assert np.isfinite(v).all() and (w > 0).all()
    # Left inverse of weighted empirical CDF; ties remain in the lower bin.
    return v[np.searchsorted(np.cumsum(w) / w.sum(), [1 / 3, 2 / 3])].tolist()


def strata_masks(x, cuts):
    result = {}
    for feature, values in features(x).items():
        lo, hi = cuts[feature]
        eligible = x[:, 18] >= .5 if feature == 'age' else np.ones(len(x), dtype=bool)
        result[feature + '/low'] = eligible & (values <= lo)
        result[feature + '/middle'] = eligible & (values > lo) & (values <= hi)
        result[feature + '/high'] = eligible & (values > hi)
    result['history/absent'] = x[:, 18] < .5
    result['history/present'] = x[:, 18] >= .5
    result['dt/normal'] = np.expm1(x[:, 15]) <= .015
    result['dt/large'] = np.expm1(x[:, 15]) > .015
    return result


def main():
    sources = C.source_guard()
    C.verify_inputs()
    assert not (C.OUT / 'selection.json').exists(), 'Never replace frozen selection'
    assert not (C.OUT / 'predictions_complete.json').exists()
    receipt = C.read(C.OUT / 'fit_complete.json')
    assert receipt['all_pass'] and len(receipt['fits']) == 36
    assert receipt['sources'] == sources and not receipt['test_accessed']
    for fit in receipt['fits']:
        assert fit['normal'] and C.sha(C.OUT / fit['path']) == fit['sha256']
    for rel, digest in receipt['files'].items():
        assert C.sha(C.OUT / rel) == digest
    result = {'sources': sources, 'created_unix': time.time(), 'TEST_used': False,
              'fit_complete_sha256': C.sha(C.OUT / 'fit_complete.json'),
              'criterion': 'within-unit equal seven-rule ensemble physical VAL MSE; lexical ties',
              'strata': 'TRAIN input-only per-unit weighted empirical terciles; equal total weight per rule',
              'by_unit': {}, 'input_cuts': {}, 'val_files': {}}
    for unit in C.UNITS:
        scores = {a: [] for a in ARCHES}
        values = {f: [] for f in ('age', 'fastchange', 'fastlevel')}
        weights = {f: [] for f in values}
        for rule in C.TRAIN_RULES:
            path = C.OUT / 'val' / f'{unit}_{rule}.npz'
            arr = C.npz(path)
            reference = D.common_row_support(unit, rule, 'val')
            for k in reference:
                assert np.array_equal(arr[k], reference[k]), (unit, rule, k)
            for arch in ARCHES:
                ensemble = np.mean([arr[f'Seq_{arch}_s{s}'] for s in C.SEEDS], axis=0)
                assert np.allclose(ensemble, arr[f'Seq_{arch}_ensemble'], rtol=0, atol=1e-12)
                scores[arch].append(float(np.mean((ensemble - arr['truth']) ** 2)))
            result['val_files'][path.relative_to(C.OUT).as_posix()] = C.sha(path)
            train = D.aligned_inputs(unit, rule, 'train')
            feats = features(train['x'])
            for name in values:
                eligible = train['x'][:, 18] >= .5 if name == 'age' else np.ones(len(train['x']), dtype=bool)
                value = feats[name][eligible]
                assert len(value), 'TRAIN rule lacks eligible input rows: ' + unit + '/' + rule + '/' + name
                values[name].append(value)
                weights[name].append(np.full(len(value), 1 / (len(C.TRAIN_RULES) * len(value))))
        means = {a: float(np.mean(v)) for a, v in scores.items()}
        chosen = min(ARCHES, key=lambda a: (means[a], a))
        result['by_unit'][unit] = {'architecture': chosen, 'score': means[chosen],
                                   'candidate_scores': means, 'rule_scores': scores}
        result['input_cuts'][unit] = {f: weighted_quantiles(np.concatenate(v), np.concatenate(weights[f]))
                                       for f, v in values.items()}
    C.save(C.OUT / 'selection.json', result)
    print('B116 selection frozen; TEST not used')


if __name__ == '__main__':
    main()
