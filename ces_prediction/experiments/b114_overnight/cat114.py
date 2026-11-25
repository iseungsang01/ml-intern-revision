"""Bounded GPU CatBoost candidates: TRAIN fits, VAL export, frozen-choice TEST.

The parent workflow owns preregistration, the global fit barrier and selection.
No automatic retry, no CPU fitting and no TEST inference/scoring in smoke/fit.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import common114 as C

FORMS = ('direct', 'residual')
CONFIGS = {
    'depth4': dict(depth=4, iterations=1000, l2_leaf_reg=3),
    'depth8': dict(depth=8, iterations=1000, l2_leaf_reg=3),
    'depth6_long4000': dict(depth=6, iterations=4000, l2_leaf_reg=3),
    'depth6_l2_10': dict(depth=6, iterations=1000, l2_leaf_reg=10),
}
BASE_PARAMS = dict(learning_rate=0.03, loss_function='RMSE', border_count=128,
                   task_type='GPU', devices='0', thread_count=1, verbose=500,
                   allow_writing_files=False, use_best_model=False, gpu_ram_part=0.6)
OUT = C.OUT / 'cat'


def ah(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def column(config, form, seed):
    return f'Cat_{config}_{form}_s{seed}'


def ensemble_column(config, form):
    return f'Cat_{config}_{form}_ensemble'


def checked_context():
    from catboost.utils import get_gpu_device_count
    sources = C.source_guard()
    manifest = C.verify_inputs()
    assert all(os.environ.get(k) == v for k, v in C.ENV_PINS.items()), 'Explicit CES pins required'
    assert get_gpu_device_count() > 0, 'GPU device required; no CPU fitting fallback'
    return sources, manifest


def exclusive_start(dest, sources, manifest, mode):
    import catboost
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'started.json').open('x', encoding='utf8') as stream:
        json.dump({'mode': mode, 'started': time.time(), 'pid': os.getpid(),
                   'sources': sources, 'manifest_sha256': C.sha(C.PRIOR / 'manifest.json'),
                   'units': manifest['units'], 'catboost': catboost.__version__,
                   'env': C.ENV_PINS, 'configs': CONFIGS, 'base_params': BASE_PARAMS,
                   'TRAIN_rules': list(C.TRAIN_RULES), 'seeds': list(C.SEEDS),
                   'TEST_fit_rows': 0, 'VAL_fit_rows': 0}, stream, indent=2)


def fit_one(tag, config, form, seed, x, y, gp, weights, coverage, dest, smoke=False):
    from catboost import CatBoostRegressor
    dest.mkdir(parents=True, exist_ok=True)
    label = f'{form}_s{seed}'
    params = {**BASE_PARAMS, **CONFIGS[config], 'random_seed': int(seed)}
    if smoke:
        params['iterations'] = 10
    target = y if form == 'direct' else y - gp
    receipt = {'unit': tag, 'config': config, 'form': form, 'seed': int(seed),
               'params': params, 'started': time.time(), 'TRAIN_rows': len(y),
               'TRAIN_coverage': coverage, 'VAL_fit_rows': 0, 'TEST_fit_rows': 0,
               'pooled_hashes': {k: ah(v) for k, v in
                                 dict(x=x, y=y, gp=gp, weights=weights, target=target).items()}}
    with (dest / (label + '_started.json')).open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, indent=2)
    model = CatBoostRegressor(**params)
    model.fit(x, target, sample_weight=weights)
    assert model.tree_count_ == params['iterations'], 'Fixed full budget not completed'
    for ext, format in [('cbm', 'cbm'), ('json', 'json')]:
        model.save_model(str(dest / (label + '.' + ext)), format=format)
    receipt.update(seconds=time.time() - receipt['started'], tree_count=model.tree_count_,
                   all_params=model.get_all_params(),
                   model_sha256={ext: C.sha(dest / (label + '.' + ext)) for ext in ('cbm', 'json')})
    C.save(dest / (label + '_complete.json'), receipt)
    return model


def smoke():
    sources, manifest = checked_context()
    dest = OUT / 'smoke'
    exclusive_start(dest, sources, manifest, 'smoke')
    tag = next(iter(manifest['units']))
    _, x, y, gp, weights, _, _, coverage = C.load_unit(tag)
    assert len(x) >= 512
    fit_one(tag, 'depth4', 'direct', C.SEEDS[0], x[:512], y[:512], gp[:512],
            weights[:512], coverage, dest, smoke=True)
    C.save(dest / 'complete.json', {'all_pass': True, 'sources': sources, 'fits': 1,
                                   'iterations': 10, 'TRAIN_rows': 512, 'TEST_predictions': 0})


def fit():
    sources, manifest = checked_context()
    smoke_record = C.read(OUT / 'smoke/complete.json')
    assert smoke_record['all_pass'] and smoke_record['sources'] == sources
    exclusive_start(OUT, sources, manifest, 'fit')
    val_dir = OUT / 'val'
    val_dir.mkdir(exist_ok=False)
    receipts, val_hashes = {}, {}
    expected = len(manifest['units']) * len(CONFIGS) * len(FORMS) * len(C.SEEDS)
    for tag, unit in manifest['units'].items():
        data, x, y, gp, weights, _, _, coverage = C.load_unit(tag)
        val = {}
        for rule in C.TRAIN_RULES:
            z = data[rule]
            mask = z['split'] == 1
            assert mask.sum() >= 30
            val[rule] = {name: z[key][mask].copy() for name, key in
                         [('idx', 'idx'), ('truth', 'y'), ('cluster', 'cluster'),
                          ('discharge', 'discharge'), ('gp', 'gp')]}
            val[rule]['GP_only'] = val[rule]['gp'].copy()
        for config in CONFIGS:
            dest = OUT / 'models' / tag / config
            for form in FORMS:
                for seed in C.SEEDS:
                    C.save(OUT / 'progress.json', {'state': 'fitting', 'unit': tag,
                           'name': unit['name'], 'config': config, 'form': form, 'seed': int(seed),
                           'completed_fits': len(receipts), 'total_fits': expected, 'updated': time.time()})
                    model = fit_one(tag, config, form, seed, x, y, gp, weights, coverage, dest)
                    for rule in C.TRAIN_RULES:
                        z = data[rule]
                        mask = z['split'] == 1
                        pred = model.predict(z['x'][mask], thread_count=1, task_type='CPU')
                        if form == 'residual':
                            pred += z['gp'][mask]
                        assert np.isfinite(pred).all()
                        val[rule][column(config, form, seed)] = pred
                    receipt = dest / f'{form}_s{seed}_complete.json'
                    receipts[receipt.relative_to(OUT).as_posix()] = C.sha(receipt)
                    del model
                for rule in C.TRAIN_RULES:
                    val[rule][ensemble_column(config, form)] = np.mean(
                        [val[rule][column(config, form, seed)] for seed in C.SEEDS], axis=0)
        for rule, arr in val.items():
            path = val_dir / f'{tag}_{rule}.npz'
            np.savez_compressed(path, **arr)
            val_hashes[path.relative_to(OUT).as_posix()] = C.sha(path)
        del data, x, y, gp, weights, val
        gc.collect()
    assert len(receipts) == expected == 96
    assert C.source_guard() == sources
    C.save(OUT / 'fit_complete.json', {'all_pass': True, 'fits': expected, 'sources': sources,
           'receipts': receipts, 'val_files': val_hashes, 'completed': time.time(),
           'VAL_fit_rows': 0, 'TEST_fit_rows': 0, 'TEST_predictions': 0})
    C.save(OUT / 'progress.json', {'state': 'fit_complete', 'completed_fits': expected,
                                 'total_fits': expected, 'updated': time.time()})


def predict():
    from catboost import CatBoostRegressor
    sources = C.source_guard()
    manifest = C.verify_inputs()
    fit_record = C.read(OUT / 'fit_complete.json')
    assert fit_record['all_pass'] and fit_record['sources'] == sources
    barrier = C.read(C.OUT / 'all_fits_complete.json')
    assert barrier['all_pass'], 'Global suite fit barrier required before TEST inference'
    selection_path = C.OUT / 'selection.json'
    selected = C.read(selection_path)['cat']
    assert set(selected['by_unit']) == set(manifest['units'])
    dest = OUT / 'estimates'
    exclusive_start(dest, sources, manifest, 'predict')
    files = {}
    # No fitted model is selected using any TEST error.
    for tag in manifest['units']:
        choice = selected['by_unit'][tag]
        config, form = choice['config'], choice['form']
        assert config in CONFIGS and form in FORMS
        assert np.isfinite(choice['score'])
        models = {}
        for seed in C.SEEDS:
            path = OUT / 'models' / tag / config / f'{form}_s{seed}.cbm'
            receipt_path = path.with_name(f'{form}_s{seed}_complete.json')
            assert C.sha(receipt_path) == fit_record['receipts'][receipt_path.relative_to(OUT).as_posix()]
            receipt = C.read(receipt_path)
            assert C.sha(path) == receipt['model_sha256']['cbm']
            models[seed] = CatBoostRegressor().load_model(str(path))
        data, *_unused = C.load_unit(tag)
        for rule in C.RULES:
            z = data[rule]
            mask = z['split'] == 2
            arr = C.npz(C.PRIOR / f'{tag}_{rule}_reference.npz')
            for name, key in [('idx', 'idx'), ('truth', 'y'), ('cluster', 'cluster'), ('discharge', 'discharge')]:
                assert np.array_equal(arr[name], z[key][mask]), 'TEST support mismatch'
            for seed in C.SEEDS:
                pred = models[seed].predict(z['x'][mask], thread_count=1, task_type='CPU')
                if form == 'residual':
                    pred += z['gp'][mask]
                assert np.isfinite(pred).all()
                arr[f'Cat_selected_s{seed}'] = pred
            arr['Cat_selected_ensemble'] = np.mean([arr[f'Cat_selected_s{s}'] for s in C.SEEDS], axis=0)
            path = dest / f'{tag}_{rule}.npz'
            np.savez_compressed(path, **arr)
            files[path.relative_to(OUT).as_posix()] = C.sha(path)
            C.save(OUT / 'progress.json', {'state': 'predicting', 'completed_cells': len(files),
                                         'total_cells': 32, 'updated': time.time()})
        del data, models, _unused
        gc.collect()
    assert len(files) == 32 and C.source_guard() == sources
    C.save(OUT / 'prediction_complete.json', {'all_pass': True, 'sources': sources,
           'selection': selected, 'selection_sha256': C.sha(selection_path), 'cells': 32,
           'files': files, 'completed': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--fit', action='store_true')
    mode.add_argument('--predict', action='store_true')
    args = parser.parse_args()
    try:
        smoke() if args.smoke else fit() if args.fit else predict()
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        C.save(OUT / ('smoke_failed.json' if args.smoke else 'fit_failed.json' if args.fit else 'prediction_failed.json'),
               {'error': repr(exc), 'traceback': traceback.format_exc(), 'time': time.time(),
                'no_automatic_retry': True})
        raise
