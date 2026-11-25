"""Registered local GPU CatBoost comparison on the exact B112 scalar inputs.

No retries, tuning or validation-based selection. Prediction may use CPU;
every fit, including the real-data smoke, explicitly requires GPU device 0.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = ROOT / 'data/colab_b112/bundle'
OUT = ROOT / 'data/.b113'
sys.path.insert(0, str(HERE.parent / 'b112_residual'))
import common112 as C
import run112 as B
B.W = PRIOR
FORMS = ('direct', 'residual')
ENV_PINS = {**C.ENV_PINS, 'CES_B113_TARGET': 'F', 'CES_B113_RULE': 'MIXTURE7'}
PARAMS = dict(iterations=1000, depth=6, learning_rate=0.03, loss_function='RMSE',
              l2_leaf_reg=3, border_count=128, task_type='GPU', devices='0',
              thread_count=1, verbose=100, allow_writing_files=False,
              use_best_model=False, gpu_ram_part=0.6)


def ah(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save(path, value):
    C.write(path, value)


def source_guard():
    reg = HERE.parent / 'PREREGISTRATION_B113.md'
    assert reg.is_file(), 'B113 registration required before any fit'
    paths = [*sorted(HERE.glob('*.py')), *sorted(HERE.parent.glob('PREREGISTRATION_B113*.md'))]
    result = {}
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        committed = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT,
                                   capture_output=True, check=True).stdout
        assert hashlib.sha256(committed).hexdigest() == C.sha(path), 'Uncommitted source: ' + rel
        result[rel] = C.sha(path)
    return result


def verify_inputs():
    manifest = C.read(PRIOR / 'manifest.json')
    assert manifest['registration'] == 'PREREGISTRATION_B112.md'
    assert manifest['target'] == 'current_report_F'
    assert [u['name'] for u in manifest['units'].values()] == list(C.UNITS)
    assert len(manifest['units']) == 4
    for name, digest in manifest['files'].items():
        assert Path(name).name == name
        assert C.sha(PRIOR / name) == digest, 'B112 input changed: ' + name
    for rel, digest in manifest['sources'].items():
        assert C.sha(ROOT / rel) == digest, 'Inherited scientific source changed: ' + rel
    audit = C.read(ROOT / 'data/colab_b112/independent_audit.json')
    assert audit['all_pass'], 'B112 independent audit required'
    return manifest


def fit_one(tag, form, seed, x, y, gp, weights, coverage, dest, iterations=1000):
    from catboost import CatBoostRegressor
    label = form + '_s' + str(seed)
    params = {**PARAMS, 'iterations': iterations, 'random_seed': seed}
    target = y if form == 'direct' else y - gp
    started = time.time()
    save(dest / (label + '_started.json'), {'unit': tag, 'form': form, 'seed': seed,
         'params': params, 'started': started, 'TRAIN_coverage': coverage,
         'TRAIN_rows': len(y), 'VAL_fit_rows': 0, 'TEST_fit_rows': 0,
         'pooled_hashes': {k: ah(v) for k, v in dict(x=x, y=y, gp=gp, weights=weights, target=target).items()}})
    model = CatBoostRegressor(**params)
    model.fit(x, target, sample_weight=weights)
    assert model.tree_count_ == iterations, 'Fixed fit budget was not completed'
    model.save_model(str(dest / (label + '.cbm')))
    model.save_model(str(dest / (label + '.json')), format='json')
    record = C.read(dest / (label + '_started.json'))
    record.update(seconds=time.time() - started, tree_count=model.tree_count_,
                  all_params=model.get_all_params(),
                  model_sha256={ext: C.sha(dest / (label + '.' + ext)) for ext in ('cbm', 'json')})
    save(dest / (label + '_complete.json'), record)
    return model


def execute(smoke=False):
    import catboost
    from catboost.utils import get_gpu_device_count
    sources = source_guard()
    manifest = verify_inputs()
    assert all(os.environ.get(k) == v for k, v in ENV_PINS.items()), 'Explicit CES pins required'
    assert get_gpu_device_count() > 0, 'Actual local CatBoost CUDA device required; no CPU fallback'
    dest = OUT / 'smoke' if smoke else OUT
    dest.mkdir(parents=True, exist_ok=True)
    # Exclusive receipt also prevents duplicate scientific execution and smoke overwrites.
    with (dest / 'started.json').open('x', encoding='utf8') as f:
        json.dump({'started': time.time(), 'pid': os.getpid(), 'smoke': smoke,
                   'sources': sources, 'manifest_sha256': C.sha(PRIOR / 'manifest.json'),
                   'catboost': catboost.__version__, 'gpu_devices': get_gpu_device_count(),
                   'env': ENV_PINS, 'params': PARAMS, 'exploratory_reused_TEST': True}, f, indent=2)
    try:
        if smoke:
            tag = next(iter(manifest['units']))
            _, x, y, gp, weights, _, _, coverage = B.load_unit(tag)
            fit_one(tag, 'direct', C.SEEDS[0], x[:512], y[:512], gp[:512], weights[:512],
                    coverage, dest, iterations=10)
            save(dest / 'complete.json', {'all_pass': True, 'fits': 1, 'iterations': 10,
                                         'TRAIN_rows': min(512, len(y)), 'TEST_predictions': 0})
            return
        smoke_receipt = C.read(OUT / 'smoke/complete.json')
        assert smoke_receipt['all_pass'], 'Successful registered GPU smoke required'
        assert C.read(OUT / 'smoke/started.json')['sources'] == sources, 'Smoke source mismatch'
        fits = {}
        # Global fit barrier: all registered fits finish before any TEST prediction.
        for tag, unit in manifest['units'].items():
            data, x, y, gp, weights, _, _, coverage = B.load_unit(tag)
            model_dir = dest / 'models' / tag
            model_dir.mkdir(parents=True, exist_ok=False)
            for form in FORMS:
                for seed in C.SEEDS:
                    save(dest / 'progress.json', {'state': 'fitting', 'unit': tag, 'name': unit['name'],
                         'form': form, 'seed': seed, 'completed_fits': len(fits), 'total_fits': 24, 'updated': time.time()})
                    model = fit_one(tag, form, seed, x, y, gp, weights, coverage, model_dir)
                    receipt = model_dir / (form + '_s' + str(seed) + '_complete.json')
                    fits[receipt.relative_to(dest).as_posix()] = C.sha(receipt)
                    del model
            del data, x, y, gp, weights
            gc.collect()
        assert len(fits) == 24
        assert source_guard() == sources, 'Frozen sources changed before TEST prediction'
        save(dest / 'training_complete.json', {'all_pass': True, 'fits': 24, 'files': fits,
             'sources': sources, 'VAL_fit_rows': 0, 'TEST_fit_rows': 0, 'completed': time.time()})
        from catboost import CatBoostRegressor
        estimates = dest / 'estimates'
        estimates.mkdir(exist_ok=False)
        files = {}
        for tag in manifest['units']:
            models = {}
            for form in FORMS:
                for seed in C.SEEDS:
                    m = CatBoostRegressor()
                    m.load_model(str(dest / 'models' / tag / (form + '_s' + str(seed) + '.cbm')))
                    models[form, seed] = m
            for rule in C.RULES:
                z = C.npz(PRIOR / (tag + '_' + rule + '.npz'))
                mask = z['split'] == 2
                arr = C.npz(PRIOR / (tag + '_' + rule + '_reference.npz'))
                for k, key in [('idx', 'idx'), ('truth', 'y'), ('cluster', 'cluster'), ('discharge', 'discharge')]:
                    assert np.array_equal(arr[k], z[key][mask]), 'Reference support changed'
                for form in FORMS:
                    names = []
                    for seed in C.SEEDS:
                        name = 'CatBoost_' + form + '_s' + str(seed)
                        arr[name] = models[form, seed].predict(z['x'][mask], thread_count=1, task_type='CPU')
                        if form == 'residual':
                            arr[name] += z['gp'][mask]
                        assert np.isfinite(arr[name]).all()
                        names.append(name)
                    arr['CatBoost_' + form + '_ensemble'] = np.mean([arr[n] for n in names], axis=0)
                path = estimates / (tag + '_' + rule + '.npz')
                np.savez_compressed(path, **arr)
                files[path.relative_to(dest).as_posix()] = C.sha(path)
                save(dest / 'progress.json', {'state': 'predicting', 'completed_cells': len(files),
                     'total_cells': 32, 'updated': time.time()})
        assert len(files) == 32
        assert source_guard() == sources, 'Frozen sources changed during execution'
        save(dest / 'complete.json', {'all_pass': True, 'fits': 24, 'cells': 32,
             'sources': sources, 'estimates': files, 'completed': time.time(),
             'note': 'Predictions complete; independent audit and statistical readout still required.'})
        save(dest / 'progress.json', {'state': 'predictions_complete', 'completed_cells': 32,
                                     'total_cells': 32, 'updated': time.time()})
    except BaseException as e:
        save(dest / 'failed.json', {'time': time.time(), 'error': repr(e), 'traceback': traceback.format_exc(),
                                   'no_automatic_retry': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    execute(smoke=args.smoke)
