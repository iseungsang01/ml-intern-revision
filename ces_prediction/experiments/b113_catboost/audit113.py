"""Independent NumPy CatBoost JSON replay and fixed-fit B113 readout; no fitting."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'b112_residual'))
from common112 import RULES, TRAIN_RULES, SEEDS, npz

PRIOR = ROOT / 'data/colab_b112'
OUT = ROOT / 'data/.b113'
FORMS = ('direct', 'residual')
BASELINES = ('HGB_direct', 'HGB_residual', 'GP_fast_boost', 'GP_fair_mixture', 'GP_only')
ENSEMBLES = tuple('CatBoost_' + f + '_ensemble' for f in FORMS)
SINGLES = tuple('CatBoost_' + f + '_s' + str(s) for f in FORMS for s in SEEDS)
MODELS = ENSEMBLES + SINGLES + BASELINES
PAIRS = ((ENSEMBLES[1], ENSEMBLES[0]),) + tuple((a, b) for a in ENSEMBLES + SINGLES for b in BASELINES)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ah(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def checked_files(root, files):
    for rel, digest in files.items():
        target = (root / rel).resolve()
        assert target.is_relative_to(root.resolve()), 'Path escaped source root'
        assert sha(target) == digest, 'Digest mismatch: ' + rel


def portable_predict(model, x):
    """Replay symmetric numeric trees without importing CatBoost or producer code.

    CatBoost converts numerical inputs to float32 before border comparisons.
    The first JSON split contributes the least significant leaf-index bit.
    """
    assert x.ndim == 2 and np.isfinite(x).all()
    features = model['features_info']
    assert not features.get('categorical_features') and not features.get('text_features')
    assert not features.get('embedding_features')
    mapping = {int(f['feature_index']): int(f['flat_feature_index'])
               for f in features['float_features']}
    scale, bias = model.get('scale_and_bias', [1.0, [0.0]])
    assert len(bias) == 1
    values = np.zeros(len(x), dtype=np.float64)
    xx = np.asarray(x, dtype=np.float32)
    trees = model['oblivious_trees']
    assert trees
    for tree in trees:
        splits = tree.get('splits') or []
        leaf = np.zeros(len(xx), dtype=np.int64)
        for bit, split in enumerate(splits):
            assert split['split_type'] == 'FloatFeature', 'Unsupported nonnumeric split'
            col = mapping[int(split['float_feature_index'])]
            assert 0 <= col < xx.shape[1]
            leaf |= (xx[:, col] > np.float32(split['border'])).astype(np.int64) << bit
        leaves = np.asarray(tree['leaf_values'], dtype=np.float64)
        assert leaves.shape == (2 ** len(splits),) and np.isfinite(leaves).all()
        values += leaves[leaf]
    return values * float(scale) + float(bias[0])


def statistics(arr, key):
    """Group via add.at; resample groups while preserving row-weighted MSE."""
    squared = np.stack([(np.asarray(arr[n], dtype=np.float64) - arr['truth']) ** 2
                        for n in MODELS], axis=1)
    assert np.isfinite(squared).all()
    point = squared.sum(axis=0) / len(squared)
    assert (point > 0).all()
    models = {n: {'MSE': float(point[j]), 'RMSE': float(np.sqrt(point[j]))}
              for j, n in enumerate(MODELS)}
    pairs = {a + '_over_' + b: {'point': float(1 - point[MODELS.index(a)] / point[MODELS.index(b)])}
             for a, b in PAIRS}
    counts = {}
    for grouping in ('cluster', 'discharge'):
        labels, inv = np.unique(arr[grouping], return_inverse=True)
        ng = len(labels)
        assert ng >= 2
        counts[grouping] = ng
        sums = np.zeros((ng, len(MODELS)), dtype=np.float64)
        count = np.zeros(ng, dtype=np.int64)
        np.add.at(sums, inv, squared)
        np.add.at(count, inv, 1)
        rng = np.random.default_rng([20261126, zlib.crc32((key + '/' + grouping).encode())])
        draws = []
        for _ in range(40):
            weight = rng.multinomial(ng, np.repeat(1 / ng, ng), size=500)
            draws.append(np.einsum('bg,gm->bm', weight, sums) /
                         np.einsum('bg,g->b', weight, count)[:, None])
        draws = np.concatenate(draws)
        assert draws.shape == (20000, len(MODELS)) and np.isfinite(draws).all() and (draws > 0).all()
        ci = np.percentile(draws, [2.5, 97.5], axis=0)
        for j, n in enumerate(MODELS):
            models[n][grouping + '_MSE95'] = ci[:, j].tolist()
            models[n][grouping + '_RMSE95'] = np.sqrt(ci[:, j]).tolist()
        for a, b in PAIRS:
            values = 1 - draws[:, MODELS.index(a)] / draws[:, MODELS.index(b)]
            q = np.percentile(values, [2.5, 97.5])
            pairs[a + '_over_' + b][grouping] = {
                'ci95': q.tolist(), 'excludes_zero': bool(q[0] > 0 or q[1] < 0)}
    return {'rows': len(squared), 'clusters': counts['cluster'], 'discharges': counts['discharge'],
            'models': models, 'pairs': pairs}


def render(report):
    lines = ['# B113: matched GPU CatBoost scalar comparison', '',
             'Exploratory reused-TEST fixed-fit comparison; F target; all four validation designs.',
             'Same B112 scalar inputs and seven-rule equal-weight TRAIN mixture; K10 is evaluation only.',
             'Pointwise 95% intervals resample shot clusters; discharge intervals and each seed are in JSON.',
             'Intervals omit refitting, model selection and initialization uncertainty. Seed ranges are descriptive.',
             'Only the regression learner changes; fixed budgets and direct/residual parametrizations are reported together.', '',
             '| Unit / rule | Model | Rows | MSE [95% cluster] | RMSE |',
             '|---|---|---:|---|---:|']
    for key, cell in report['cells'].items():
        for n, m in cell['models'].items():
            lo, hi = m['cluster_MSE95']
            lines.append(f"| {key} | {n} | {cell['rows']} | {m['MSE']:.3f} [{lo:.3f}, {hi:.3f}] | {m['RMSE']:.3f} |")
    lines += ['', '| Unit / rule | Paired MSE skill | Point [95% cluster] | Excludes zero |',
              '|---|---|---|---|']
    for key, cell in report['cells'].items():
        for pair, p in cell['pairs'].items():
            if '_ensemble_over_' not in pair:
                continue
            lo, hi = p['cluster']['ci95']
            lines.append(f"| {key} | {pair} | {p['point']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {p['cluster']['excludes_zero']} |")
    lines += ['', 'Statements: ' + json.dumps(report['statements']), '',
              'A matched comparison on unused observations could overturn an unfavorable result.',
              'No architecture-class impossibility, new physics, or independent confirmation is established.']
    return '\n'.join(lines) + '\n'


def main():
    assert read(PRIOR / 'independent_audit.json')['all_pass']
    prior_report = read(ROOT / 'docs/paper/b112_comparison.json')
    assert prior_report['independent_audit']['all_pass']
    assert sha(PRIOR / 'b112_results.zip') == prior_report['independent_audit']['archive_sha256']
    prior_manifest = read(PRIOR / 'bundle/manifest.json')
    checked_files(PRIOR / 'bundle', prior_manifest['files'])
    prior_hashes = read(PRIOR / 'retrieved/result_hashes.json')
    started = read(OUT / 'started.json')
    complete = read(OUT / 'complete.json')
    training = read(OUT / 'training_complete.json')
    assert complete['all_pass'] and complete['fits'] == 24 and complete['cells'] == 32
    assert training['all_pass'] and training['fits'] == 24
    assert training['VAL_fit_rows'] == training['TEST_fit_rows'] == 0
    assert len(training['files']) == 24 and len(complete['estimates']) == 32
    assert started['sources'] == complete['sources']
    checked_files(ROOT, started['sources'])
    assert started['manifest_sha256'] == sha(PRIOR / 'bundle/manifest.json')
    checked_files(OUT, training['files'])
    checked_files(OUT, complete['estimates'])
    units = prior_manifest['units']
    assert len(units) == 4
    cells, replay = {}, {}
    for tag, unit in units.items():
        models = {}
        for form in FORMS:
            for seed in SEEDS:
                label = form + '_s' + str(seed)
                receipt = read(OUT / 'models' / tag / (label + '_complete.json'))
                for ext in ('cbm', 'json'):
                    assert sha(OUT / 'models' / tag / (label + '.' + ext)) == receipt['model_sha256'][ext]
                assert receipt['tree_count'] == 1000 and receipt['params']['task_type'] == 'GPU'
                models[(form, seed)] = read(OUT / 'models' / tag / (label + '.json'))
                assert len(models[(form, seed)]['oblivious_trees']) == 1000
                assert receipt['VAL_fit_rows'] == receipt['TEST_fit_rows'] == 0
        coverage, train_parts = {}, []
        for rule in RULES:
            original = npz(PRIOR / 'bundle' / (tag + '_' + rule + '.npz'))
            assert np.isfinite(original['x']).all() and np.isfinite(original['y']).all()
            assert set(np.unique(original['split'])).issubset({0, 1, 2})
            if rule == 'K10':
                assert (original['split'] == 2).all()
            else:
                mask = original['split'] == 0
                assert mask.any()
                coverage[rule] = {'rows': int(mask.sum()), **{
                    n + '_sha256': ah(original[n][mask]) for n in ('idx', 'x', 'y', 'gp')}}
                train_parts.append(tuple(original[n][mask] for n in ('x', 'y', 'gp')))
            test = original['split'] == 2
            arr = npz(OUT / 'estimates' / (tag + '_' + rule + '.npz'))
            for n, source in (('idx', 'idx'), ('truth', 'y'), ('cluster', 'cluster'), ('discharge', 'discharge')):
                assert np.array_equal(arr[n], original[source][test]), tag + '/' + rule + '/' + n
            prior_rel = 'estimates/' + tag + '_' + rule + '.npz'
            assert sha(PRIOR / 'retrieved' / prior_rel) == prior_hashes[prior_rel]
            old = npz(PRIOR / 'retrieved' / prior_rel)
            for n in ('idx', 'truth', 'cluster', 'discharge'):
                assert np.array_equal(arr[n], old[n])
            for n in BASELINES:
                if n in arr:
                    assert np.array_equal(arr[n], old[n])
                arr[n] = old[n]
            errors = {}
            for form in FORMS:
                names = []
                for seed in SEEDS:
                    name = 'CatBoost_' + form + '_s' + str(seed)
                    pred = portable_predict(models[(form, seed)], original['x'][test])
                    if form == 'residual':
                        pred += original['gp'][test]
                    assert np.isfinite(arr[name]).all() and pred.shape == arr[name].shape
                    errors[name] = float(np.max(np.abs(pred - arr[name])))
                    assert np.allclose(pred, arr[name], rtol=1e-7, atol=1e-5), 'Portable replay mismatch: ' + name
                    names.append(name)
                ensemble = np.mean(np.stack([arr[n] for n in names]), axis=0)
                assert np.array_equal(ensemble, arr['CatBoost_' + form + '_ensemble'])
            key = unit['name'] + '/' + rule + '/F'
            replay[key] = errors
            cells[key] = statistics(arr, key)
            print('B113 independent audit ' + key, flush=True)
        assert set(coverage) == set(TRAIN_RULES)
        pooled_x, pooled_y, pooled_gp = [np.concatenate([p[j] for p in train_parts]) for j in range(3)]
        n_train = len(pooled_y)
        weights = np.concatenate([np.full(len(p[1]), n_train / (len(TRAIN_RULES) * len(p[1])))
                                  for p in train_parts])
        for form in FORMS:
            for seed in SEEDS:
                receipt = read(OUT / 'models' / tag / (form + '_s' + str(seed) + '_complete.json'))
                assert receipt['TRAIN_coverage'] == coverage
                assert receipt['TRAIN_rows'] == n_train
                target = pooled_y if form == 'direct' else pooled_y - pooled_gp
                expected = dict(x=pooled_x, y=pooled_y, gp=pooled_gp, weights=weights, target=target)
                assert receipt['pooled_hashes'] == {k: ah(v) for k, v in expected.items()}
    assert len(cells) == 32
    audit = {'all_pass': True, 'completed': time.time(), 'cells': 32, 'fits': 24,
             'portable_replay_max_abs_ev': replay, 'sources': started['sources'],
             'original_b112_audit_sha256': sha(PRIOR / 'independent_audit.json'),
             'started_sha256': sha(OUT / 'started.json'), 'complete_sha256': sha(OUT / 'complete.json'),
             'independent_numpy_json_replay': True, 'bootstrap_draws': 20000,
             'seed': 20261126, 'local_fits_in_audit': 0}
    report = {'registration': 'PREREGISTRATION_B113.md', 'exploratory': True,
              'target': 'current_report_F', 'cells': cells, 'seeds': list(SEEDS),
              'sources': started['sources'], 'independent_audit': audit,
              'limits': ['Reused TEST exploratory comparison.', 'Fixed-fit pointwise intervals omit refitting and selection uncertainty.',
                         'Seeds are reported separately; ensembles do not remove initialization uncertainty.',
                         'GPU floating-point training may be nondeterministic.']}
    fresh = [v for k,v in cells.items() if '/D10/' in k]
    assert len(fresh) == 4
    report['statements'] = {name: all(c['pairs'][pair]['cluster']['ci95'][0] > 0 for c in fresh) for name,pair in [('R1','CatBoost_residual_ensemble_over_HGB_residual'),('R2','CatBoost_direct_ensemble_over_HGB_direct'),('R3','CatBoost_residual_ensemble_over_GP_fast_boost')]}
    for form in FORMS:
        for baseline in ('HGB_residual', 'GP_fast_boost'):
            pair = 'CatBoost_' + form + '_ensemble_over_' + baseline
            report['statements'][form + '_over_' + baseline + '_all_D10'] = all(
                c['pairs'][pair]['cluster']['ci95'][0] > 0 for k, c in cells.items() if '/D10/' in k)
            report['statements'][form + '_over_' + baseline + '_all_scenarios'] = all(
                c['pairs'][pair]['cluster']['ci95'][0] > 0 for c in cells.values())
    save(OUT / 'audit.json', audit)
    save(ROOT / 'docs/paper/b113_comparison.json', report)
    (ROOT / 'docs/paper/b113_comparison.md').write_text(render(report), encoding='utf8')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
