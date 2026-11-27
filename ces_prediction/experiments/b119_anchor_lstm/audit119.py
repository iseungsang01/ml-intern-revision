"""B119 independent no-update qualification, VAL gate, and fixed TEST audit.

Only new learned output is a LSTM correction to the available causal CES anchor
(or zero normalized TRAIN mean when absent). Frozen independent LSTM gates are
reused without importing a producer model. No fitting is performed here.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path
import numpy as np
import common119 as C
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'b118_lstm_eval'))
import audit118 as PREVIOUS
A, D, F = PREVIOUS.A, PREVIOUS.D, PREVIOUS.FROZEN
UNIT, SEED = 'V1_r2_k0', 1701
TRAIN_RULES = ('D10', 'D50', 'D100', 'D300', 'D650', 'D1000', 'S')
RULES = TRAIN_RULES + ('K10',)
MODELS = ('Anchor_LSTM', 'Direct_LSTM', 'GP_fast_boost')
PAIRS = (('Anchor_LSTM', 'GP_fast_boost'), ('Anchor_LSTM', 'Direct_LSTM'))
ATOL, RTOL = .05, 1e-4
DIRECT = C.ROOT / 'data/.b118'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def statistics(cells, key, groupings=('cluster', 'discharge')):
    # Frozen independent bincount bootstrap, prospectively distinct B119 CRC scope.
    return PREVIOUS.statistics(cells, 'B119/' + key, models=MODELS, pairs=PAIRS, groupings=groupings)


def anchor_prediction(weights, x):
    delta = F.functional_batch(weights, x)
    baseline = np.where(np.asarray(x)[:, :, 18] > .5, np.asarray(x)[:, :, 16], 0).astype(np.float64)
    return delta + baseline


def initial_weights_guard(weights):
    original_path = C.ROOT / 'data/.b117_focused/smoke/weights.npz'
    original_receipt = read(C.ROOT / 'data/.b117_focused/smoke.json')
    independent = read(C.ROOT / 'data/.b117_focused/smoke_independent.json')
    assert original_receipt['all_pass'] and independent['all_pass']
    assert original_receipt['training_updates'] == independent['training_updates'] == 0
    assert not original_receipt['TEST_accessed'] and not independent['TEST_accessed']
    assert independent['smoke_sha256'] == C.sha(C.ROOT / 'data/.b117_focused/smoke.json')
    assert original_receipt['sources'] == independent['sources'] == C.P.P.source_guard()
    A.checked_files(C.ROOT / 'data/.b117_focused', original_receipt['files'])
    assert C.sha(original_path) == original_receipt['files']['smoke/weights.npz']
    old = C.npz(original_path)
    assert set(old) == set(weights)
    head = {'model.readout.3.weight', 'model.readout.3.bias'}
    for key in weights:
        assert np.isfinite(weights[key]).all()
        if key in head:
            assert np.count_nonzero(weights[key]) == 0, 'Correction head is not initialized to exact zero'
        else:
            assert np.array_equal(weights[key], old[key]), 'Non-head initialization changed: ' + key
    return {'all_pass': True, 'original_initial_weights_sha256': C.sha(original_path),
        'unchanged_nonhead_arrays': len(weights) - 2, 'zero_head_arrays': sorted(head)}


def history_checks(history, params, prepared):
    assert (params['unit'], params['seed']) == (UNIT, SEED)
    assert len(history) == params['completed_epochs'] and 1 <= len(history) <= 200
    assert params['inputs']['inherited']['inherited']['prepared_artifact_sha256'] == prepared['artifact_sha256']
    scores = [h['seven_rule_mean_mse'] for h in history]
    assert np.isfinite(scores).all()
    assert params['best_epoch'] == int(np.argmin(scores)) + 1 and params['best_val_mse'] == min(scores)
    best, bad, lr = float('inf'), 0, .001
    previous_elapsed = 0.
    keys = sorted((s['fidx'], s['segment']) for s in prepared['train'][TRAIN_RULES[0]])
    for epoch, h in enumerate(history, 1):
        assert h['epoch'] == epoch and set(h['val']) == set(TRAIN_RULES)
        assert np.isclose(h['lr_used'], lr, rtol=0, atol=1e-14)
        for rule in TRAIN_RULES:
            v = h['val'][rule]
            assert v['count'] == len(D.common_row_support(UNIT, rule, 'val')['idx'])
            assert np.isclose(v['mse'], v['sse'] / v['count'], rtol=1e-12)
        assert np.isclose(h['seven_rule_mean_mse'], np.mean([h['val'][r]['mse'] for r in TRAIN_RULES]), rtol=1e-14)
        score = h['seven_rule_mean_mse']
        if score < best * .9999:
            best, bad = score, 0
        else:
            bad += 1
        if bad > 3:
            proposed = max(lr * .5, 1e-5)
            if lr - proposed > 1e-8:
                lr = proposed
            bad = 0
        assert np.isclose(h['lr_next'], lr, rtol=0, atol=1e-14)
        assert h['best_epoch'] == int(np.argmin(scores[:epoch])) + 1
        assert h['updates'] > 0 and h['train_count'] > 0 and np.isfinite(h['train_normalized_sse'])
        assert np.isclose(h['train_normalized_mse'], h['train_normalized_sse'] / h['train_count'], rtol=1e-14)
        assert previous_elapsed <= h['elapsed_seconds']
        previous_elapsed = h['elapsed_seconds']
        rng = np.random.default_rng([SEED, epoch, zlib.crc32(UNIT.encode())])
        choices = rng.integers(0, len(TRAIN_RULES), len(keys))
        assignments = [(f, s, TRAIN_RULES[int(i)]) for (f, s), i in zip(keys, choices)]
        assert h['assignment_sha256'] == F.digest(assignments)
    tracking = F.stop_tracking(history)
    for h, t in zip(history, tracking):
        assert h['significant_best'] == t['meaningful_reference_mse']
        assert h['last_significant_epoch'] == t['last_meaningful_epoch']
        assert h['epochs_since_significant'] == t['epochs_without_meaningful_improvement']
        assert h['low_lr_streak'] == t['low_lr_streak']
    first = [i for i, t in enumerate(tracking, 1) if t['converged']]
    assert not first or first[0] == len(history)
    converged = tracking[-1]['converged']
    assert params['converged'] == converged
    assert params['status'] == ('CONVERGED' if converged else 'SAFETY_CAPPED_NOT_CONVERGED')
    assert params['total_updates'] == sum(h['updates'] for h in history) + params['discarded_partial_updates']
    assert (params['min_epochs'], params['max_epochs'], params['budget_seconds']) == (50, 200, 21600)
    if converged:
        assert params['stop_reason'] == 'registered_statistical_stop'
    else:
        assert params['stop_reason'] in ('max_epochs_safety', 'wall_safety', 'wall_safety_incomplete_epoch',
            'wall_safety_incomplete_validation', 'wall_safety_after_complete_validation')
        assert len(history) == 200 or params['actual_seconds'] >= 21600
    return tracking


def full_replay(rule, split, arr, weights, segments, prepared, prediction_name):
    ref = D.common_row_support(UNIT, rule, split)
    for key in ('idx', 'truth', 'cluster', 'discharge'):
        assert np.array_equal(arr[key], ref[key]), split + '/' + rule + '/' + key
    ids, values = [], []
    stats = prepared['identity']['stats']
    for batch in D.batches(segments, 'cpu', batch_size=16):
        pred = anchor_prediction(weights, batch['x'].numpy())
        anon = D.anonymous_indices(batch['gi'].numpy())
        take = batch['mask'].numpy() & np.isin(anon, ref['idx'])
        ids.extend(anon[take].tolist())
        values.extend((pred[take] * stats['ti_sd'] + stats['ti_mu']).tolist())
    ids, values = np.asarray(ids), np.asarray(values)
    order = np.argsort(ids)
    assert np.array_equal(ids[order], np.sort(ref['idx']))
    independent = values[order][np.searchsorted(ids[order], ref['idx'])]
    saved = arr[prediction_name]
    assert np.allclose(independent, saved, atol=ATOL, rtol=RTOL), split + '/' + rule + '/full independent replay'
    mse = float(np.mean((independent - ref['truth']) ** 2))
    saved_mse = float(np.mean((saved - ref['truth']) ** 2))
    bound = float(np.mean(np.abs(independent - saved) * (np.abs(independent - ref['truth']) + np.abs(saved - ref['truth']))))
    assert abs(mse - saved_mse) <= bound + 1e-7
    return {'all_pass': True, 'rows': len(saved), 'complete_segments': len(segments),
        'max_abs_eV': float(np.max(np.abs(independent - saved))), 'independent_MSE': mse,
        'saved_prediction_MSE': saved_mse, 'MSE_absolute_difference_bound': bound,
        'atol_eV': ATOL, 'rtol': RTOL}


# Independent recovery-state checks retained from the frozen B117 verifier.

def verify_recovery(history, params):
    import torch
    torch.set_num_threads(2)
    latest = torch.load(C.FIT / 'latest.pt', map_location='cpu', weights_only=False)
    assert latest['epoch'] == len(history) and latest['next_epoch'] == len(history) + 1 and latest['validated']
    assert latest['sources'] == params['sources'] and latest['inputs'] == params['inputs']
    assert latest['history_sha256'] == C.sha(C.FIT / 'history.json')
    assert latest['history'] == history
    assert latest['best_epoch'] == params['best_epoch'] and latest['best_val_mse'] == params['best_val_mse']
    for key in ('significant_best', 'last_significant_epoch', 'low_lr_streak'):
        assert latest[key] == history[-1][key]
    weights = C.npz(C.FIT / 'best.npz')
    assert set(latest['best_model_state']) == set(weights)
    for key, value in weights.items():
        assert np.array_equal(np.asarray(latest['best_model_state'][key]), value)
    recovery = read(C.FIT / 'recovery.json')
    for key, filename in [('latest_sha256', 'latest.pt'), ('best_sha256', 'best.npz'), ('history_sha256', 'history.json')]:
        assert recovery[key] == C.sha(C.FIT / filename)
    model = latest['model']
    assert {'model.' + k for k in model} == set(weights)
    for key, value in model.items():
        assert value.shape == weights['model.' + key].shape and torch.isfinite(value).all()
        if params['best_epoch'] == len(history):
            assert np.array_equal(value.numpy(), weights['model.' + key])
    updates = sum(h['updates'] for h in history)
    assert latest['global_updates'] == updates
    optimizer = latest['optimizer']
    assert len(optimizer['param_groups']) == 1 and len(optimizer['state']) == len(model)
    group = optimizer['param_groups'][0]
    assert group['lr'] == history[-1]['lr_next'] and group['weight_decay'] == .0001
    assert tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8
    for state in optimizer['state'].values():
        assert int(state['step']) == updates
        for key in ('exp_avg', 'exp_avg_sq'):
            assert torch.isfinite(state[key]).all()
        assert (state['exp_avg_sq'] >= 0).all()
    scheduler = latest['scheduler']
    assert scheduler['last_epoch'] == len(history) and scheduler['patience'] == 3
    assert scheduler['factor'] == .5 and scheduler['min_lrs'] == [1e-5]
    assert scheduler['threshold'] == 1e-4 and scheduler['threshold_mode'] == 'rel' and scheduler['mode'] == 'min'
    reference, bad = float('inf'), 0
    for h in history:
        if h['seven_rule_mean_mse'] < reference * .9999:
            reference, bad = h['seven_rule_mean_mse'], 0
        else:
            bad += 1
        if bad > 3:
            bad = 0
    assert scheduler['best'] == reference and scheduler['num_bad_epochs'] == bad
    torch.Generator().set_state(latest['rng_cpu'])
    assert latest['rng_cuda'] and all(v.dtype == torch.uint8 and v.ndim == 1 for v in latest['rng_cuda'])
    np.random.RandomState().set_state(latest['rng_numpy'])
    random.Random().setstate(latest['rng_python'])
    assert all(np.isfinite(v).all() for v in weights.values())
    return {'all_pass': True, 'epoch': len(history), 'next_epoch': latest['next_epoch'],
            'validated_updates': updates, 'model_parameters': len(model), 'optimizer_states': len(optimizer['state']),
            'rng': ['torch_cpu', 'torch_cuda', 'numpy', 'python'],
            'latest_sha256': C.sha(C.FIT / 'latest.pt'), 'best_sha256': C.sha(C.FIT / 'best.npz')}


def smoke():
    sources, inputs = C.source_guard(), C.verify_inputs()
    receipt = read(C.OUT / 'smoke.json')
    assert receipt['all_pass'] and receipt['sources'] == sources and receipt['inputs'] == inputs
    assert receipt['training_updates'] == receipt['optimizer_steps'] == 0 and not receipt['TEST_accessed']
    assert all(receipt[k] for k in ('backward_finite', 'initial_output_exact_anchor', 'nonhead_seeded_core_identical', 'prefix_causal', 'future_causal'))
    assert receipt['step_max_abs'] <= 2e-5
    A.checked_files(C.OUT, receipt['files'])
    weights = C.npz(C.OUT / 'smoke/weights.npz')
    initialization = initial_weights_guard(weights)
    saved = C.npz(C.OUT / 'smoke/prefix.npz')
    prepared = D.load_prepared(UNIT)
    candidates = []
    for segment in prepared['train'][TRAIN_RULES[0]]:
        if segment['mask'][:64, 0].any():
            candidates.append(segment)
        if len(candidates) == 2:
            break
    assert len(candidates) == 2
    sd, mu = prepared['identity']['stats']['ti_sd'], prepared['identity']['stats']['ti_mu']
    pred = anchor_prediction(weights, saved['x'])
    expected = np.where(saved['x'][:, :, 18] > .5, saved['x'][:, :, 16], 0)
    assert np.array_equal(saved['expected_anchor'], expected) and np.array_equal(saved['pred'], expected)
    maximum, rows = 0., 0
    for i, segment in enumerate(candidates):
        n = min(64, len(segment['gi']))
        assert saved['lengths'][i] == n and saved['fidx'][i] == segment['fidx'] and saved['segment'][i] == segment['segment']
        for key, value in [('x', segment['x'][:n]), ('gi', segment['gi'][:n]), ('mask', segment['mask'][:n, 0])]:
            assert np.array_equal(saved[key][i, :n], value)
        actual = saved['pred'][i, :n].astype(np.float64) * sd + mu
        assert np.allclose(pred[i, :n] * sd + mu, actual, atol=ATOL, rtol=RTOL)
        delta = A.recurrent_predict(weights, 'lstm', saved['x'][i, :n])
        base = expected[i, :n].astype(np.float64)
        assert np.allclose((delta + base) * sd + mu, actual, atol=ATOL, rtol=RTOL)
        k = min(8, n)
        shorter = anchor_prediction(weights, saved['x'][i:i + 1, :k])[0]
        assert np.array_equal(shorter, pred[i, :k])
        maximum = max(maximum, float(np.max(np.abs(pred[i, :n] * sd + mu - actual))))
        rows += n
    C.save(C.OUT / 'smoke_independent.json', {'all_pass': True, 'sources': sources, 'inputs': inputs,
        'smoke_sha256': C.sha(C.OUT / 'smoke.json'), 'training_updates': 0, 'optimizer_steps': 0,
        'TEST_accessed': False, 'initialization': initialization, 'real_TRAIN_segments': 2,
        'real_TRAIN_rows': rows, 'max_abs_eV': maximum, 'atol_eV': ATOL, 'rtol': RTOL,
        'absent_history_baseline': 'zero normalized Ti, hence physical TRAIN mean',
        'finite_backward_evidence': 'Frozen GPU source assertion and unchanged exact initialization after backward; no optimizer step'})
    print('B119 independent no-update initialized anchor qualification passed')


def val_gate(write=True):
    started = time.time()
    sources, inputs = C.source_guard(), C.verify_inputs()
    if write:
        assert not (C.OUT / 'predict_started.json').exists() and not (C.OUT / 'predictions_complete.json').exists()
        assert not (C.OUT / 'test').exists() and not (C.OUT / 'val_audit.json').exists()
    complete = read(C.OUT / 'fit_complete.json')
    assert complete['all_pass'] and complete['production_fits'] == 1 and not complete['TEST_accessed']
    assert complete['sources'] == sources
    A.checked_files(C.OUT, complete['files'])
    params, history = read(C.FIT / 'params.json'), read(C.FIT / 'history.json')
    assert C.sha(C.FIT / 'params.json') == complete['params_sha256']
    assert params['sources'] == sources and params['inputs'] == inputs and params['architecture'] == 'anchor_lstm'
    assert params['env'] == C.ENV_PINS
    assert params['runtime']['cuda'] and params['runtime']['device'] and params['runtime']['torch']
    A.checked_files(C.FIT, params['files'])
    assert [p.name for p in (C.OUT / 'fits').iterdir()] == [C.FIT.name]
    assert not (C.OUT / 'selection.json').exists()
    qualified = read(C.OUT / 'smoke_independent.json')
    assert qualified['all_pass'] and qualified['sources'] == sources and qualified['inputs'] == inputs
    assert qualified['training_updates'] == qualified['optimizer_steps'] == 0 and not qualified['TEST_accessed']
    assert qualified['smoke_sha256'] == C.sha(C.OUT / 'smoke.json')
    initialization = initial_weights_guard(C.npz(C.OUT / 'smoke/weights.npz'))
    A.compare_nested(qualified['initialization'], initialization)
    prepared = D.load_prepared(UNIT)
    normalization = A.verify_normalization(prepared)
    data_checks = {}
    for split in ('train', 'val'):
        for rule in TRAIN_RULES:
            data_checks[split + '/' + rule] = A.verify_segments(UNIT, rule, split, prepared[split][rule])
    tracking = history_checks(history, params, prepared)
    assert complete['converged'] == params['converged'] and complete['status'] == params['status']
    recovery = verify_recovery(history, params)
    weights = C.npz(C.FIT / 'best.npz')
    val = {}
    for rule in TRAIN_RULES:
        path = C.OUT / 'val' / (UNIT + '_' + rule + '.npz')
        assert C.sha(path) == complete['files'][path.relative_to(C.OUT).as_posix()]
        arr = C.npz(path)
        val[rule] = full_replay(rule, 'val', arr, weights, prepared['val'][rule], prepared, 'prediction')
        target = history[params['best_epoch'] - 1]['val'][rule]['mse']
        assert np.isclose(val[rule]['saved_prediction_MSE'], target, rtol=1e-8, atol=.01)
    assert np.isclose(np.mean([v['saved_prediction_MSE'] for v in val.values()]), params['best_val_mse'], rtol=1e-8, atol=.01)
    assert C.source_guard() == sources and C.verify_inputs() == inputs
    result = {'all_pass': True, 'sources': sources, 'inputs': inputs, 'TEST_accessed': False,
        'fit_complete_sha256': C.sha(C.OUT / 'fit_complete.json'), 'params_sha256': C.sha(C.FIT / 'params.json'),
        'best_sha256': C.sha(C.FIT / 'best.npz'), 'history_sha256': C.sha(C.FIT / 'history.json'),
        'latest_sha256': C.sha(C.FIT / 'latest.pt'), 'converged': params['converged'], 'status': params['status'],
        'initialization': initialization, 'normalization_sha256': normalization, 'data_checks': data_checks,
        'recovery': recovery, 'stopping': tracking, 'full_VAL_replay': val,
        'production_fits': 1, 'created_unix': time.time(), 'seconds': time.time() - started,
        'replay_scope': 'ALL common VAL rows, all seven rules, complete segment context, independent delta LSTM plus visible CES/TRAIN-mean baseline'}
    if write:
        C.save(C.OUT / 'val_audit.json', result)
        print('B119 independent VAL gate passed; converged=' + str(params['converged']))
    return result

def check_strata(arr, x, cuts, reported, key):
    masks = PREVIOUS.stratum_masks(x, cuts)
    assert set(masks) == set(reported)
    for label, mask in masks.items():
        row = reported[label]
        n, ng = int(mask.sum()), len(np.unique(arr['cluster'][mask]))
        assert row['rows'] == n and row['clusters'] == ng
        assert row['discharges'] == len(np.unique(arr['discharge'][mask]))
        assert row['interval_available'] == (ng >= 8)
        if not n:
            assert not row['models'] and not row['pairs']
            continue
        mses = {}
        for name in MODELS:
            error = np.asarray(arr[name][mask], dtype=np.float64) - arr['truth'][mask]
            mses[name] = float(np.mean(error ** 2))
            A.compare_nested(row['models'][name], {'MSE': mses[name], 'RMSE': float(np.sqrt(mses[name])), 'bias': float(np.mean(error))})
        for a, b in PAIRS:
            A.compare_nested(row['pairs'][a + '_over_' + b], {'point': 1 - mses[a] / mses[b]})
        if ng >= 8:
            sub = {k: np.asarray(v)[mask] for k, v in arr.items() if np.asarray(v).ndim == 1 and len(v) == len(mask)}
            expected = statistics({key: sub}, key + '/' + label, groupings=('cluster',))
            for field in ('models', 'pairs', 'bootstrap'):
                A.compare_nested(row[field], expected[field], key + '/' + label + '/' + field)


def main():
    started = time.time()
    sources, inputs = C.source_guard(), C.verify_inputs()
    gate_sha = C.sha(C.OUT / 'val_audit.json')
    gate = read(C.OUT / 'val_audit.json')
    assert gate['all_pass'] and gate['converged'] and not gate['TEST_accessed']
    assert gate['sources'] == sources and gate['inputs'] == inputs
    repeat = val_gate(write=False)
    for key in ('all_pass', 'sources', 'inputs', 'fit_complete_sha256', 'params_sha256', 'best_sha256',
                'history_sha256', 'latest_sha256', 'converged', 'status', 'initialization',
                'normalization_sha256', 'data_checks', 'recovery', 'stopping', 'full_VAL_replay'):
        A.compare_nested(gate[key], repeat[key], 'VAL_reaudit/' + key)
    launch = read(C.OUT / 'workflow/started.json')
    begin = read(C.OUT / 'predict_started.json')
    complete = read(C.OUT / 'predictions_complete.json')
    assert launch['sources'] == begin['sources'] == complete['sources'] == sources
    assert launch['inputs'] == begin['inputs'] == complete['inputs'] == inputs
    assert launch['production_fits'] == 1
    assert launch['GPU_end'] == launch['started'] + 28800 and launch['CPU_end'] == launch['started'] + 36000
    assert launch['started'] <= gate['created_unix'] <= begin['started'] <= complete['completed'] <= launch['GPU_end']
    assert begin['optimization_steps'] == complete['optimization_steps'] == 0
    assert begin['val_audit_sha256'] == complete['val_audit_sha256'] == gate_sha
    assert begin['fit_complete_sha256'] == complete['fit_complete_sha256'] == gate['fit_complete_sha256'] == C.sha(C.OUT / 'fit_complete.json')
    assert begin['best_sha256'] == gate['best_sha256'] == C.sha(C.FIT / 'best.npz')
    assert complete['all_pass'] and complete['scenarios'] == len(complete['files']) == 8 and complete['model_state_unchanged']
    before_after = {'best': gate['best_sha256'], 'latest': gate['latest_sha256']}
    assert complete['checkpoint_before'] == complete['checkpoint_after'] == before_after
    A.checked_files(C.OUT, complete['files'])
    expected_files = {'test/' + UNIT + '_' + rule + '.npz' for rule in RULES}
    assert set(complete['files']) == expected_files
    assert {p.relative_to(C.OUT).as_posix() for p in (C.OUT / 'test').iterdir() if p.is_file()} == expected_files
    report, staged = read(C.OUT / 'readout.json'), read(C.OUT / 'readout_complete.json')
    assert staged['all_pass'] and not staged['published']
    A.checked_files(C.OUT, staged['files'])
    assert report['sources'] == sources and report['inputs'] == inputs and report['unit'] == UNIT
    assert report['new_fits'] == 1 and report['architecture_searches'] == 0 and report['primary_validation_design'] is None
    assert report['validation_designs'] == {u: ('EVALUATED: one fixed single fit' if u == UNIT else 'NOT RUN') for u in D.UNITS}
    assert report['val_audit_sha256'] == gate_sha and report['fit_complete_sha256'] == gate['fit_complete_sha256']
    assert report['predictions_complete_sha256'] == staged['predictions_complete_sha256'] == C.sha(C.OUT / 'predictions_complete.json')
    assert complete['completed'] <= report['created_unix'] <= launch['CPU_end']
    old_audit, old_report = read(DIRECT / 'audit.json'), read(DIRECT / 'readout.json')
    old_complete = read(DIRECT / 'predictions_complete.json')
    assert old_audit['all_pass'] and C.sha(DIRECT / 'readout.json') == old_audit['readout_sha256']
    assert C.sha(DIRECT / 'audit.json') == inputs['B118_audit_sha256']
    assert C.sha(DIRECT / 'predictions_complete.json') == old_report['predictions_complete_sha256'] == inputs['B118_predictions_complete_sha256']
    selection_path = C.ROOT / 'data/.b116/selection.json'
    original_audit = read(C.ROOT / 'data/.b116/audit.json')
    selection_sha = C.sha(selection_path)
    assert selection_sha == original_audit['selection_sha256'] == old_report['input_cuts_source_sha256']
    cuts = read(selection_path)['input_cuts'][UNIT]
    assert report['input_cuts'] == cuts and report['input_cuts_source_sha256'] == selection_sha
    prepared = D.load_prepared(UNIT)
    weights = C.npz(C.FIT / 'best.npz')
    params = read(C.FIT / 'params.json')
    direct_fit = C.ROOT / 'data/.b117_focused/fits/U0_lstm_s1701'
    direct_params, direct_history = read(direct_fit / 'params.json'), read(direct_fit / 'history.json')
    assert report['fixed_checkpoint'] == {'architecture': 'anchor_lstm', 'seed': SEED,
        'anchor_best_epoch': params['best_epoch'], 'direct_best_epoch': direct_params['best_epoch'],
        'anchor_best_sha256': gate['best_sha256']}
    arrays, data_checks, replays, identities, discharge_cluster = {}, {}, {}, {}, {}
    for rule in RULES:
        key, rel = UNIT + '/' + rule, 'test/' + UNIT + '_' + rule + '.npz'
        arr, old = C.npz(C.OUT / rel), C.npz(DIRECT / rel)
        assert set(arr) == {'idx', 'truth', 'cluster', 'discharge', 'time', 'segment', *MODELS}
        assert all(np.asarray(v).shape == arr['truth'].shape for v in arr.values())
        assert all(np.isfinite(arr[n]).all() for n in MODELS)
        assert C.sha(DIRECT / rel) == old_complete['files'][rel] == inputs['direct_control_files'][rel]
        ref = D.common_row_support(UNIT, rule, 'test')
        for name in ('idx', 'truth', 'cluster', 'discharge', 'GP_fast_boost'):
            assert np.array_equal(arr[name], ref[name]) and np.array_equal(old[name], ref[name])
        assert np.array_equal(arr['Direct_LSTM'], old['LSTM_focused'])
        _, times, ordinal = PREVIOUS.independent_coordinates(arr['idx'], arr['discharge'])
        assert np.array_equal(arr['time'], times) and np.array_equal(arr['segment'], ordinal)
        for idx, cluster, discharge in zip(arr['idx'], arr['cluster'], arr['discharge']):
            identity = (int(cluster), int(discharge))
            if int(idx) in identities:
                assert identities[int(idx)] == identity
            identities[int(idx)] = identity
            if int(discharge) in discharge_cluster:
                assert discharge_cluster[int(discharge)] == int(cluster)
            discharge_cluster[int(discharge)] = int(cluster)
        segments = D.test_segments(UNIT, rule)
        data_checks[rule] = A.verify_segments(UNIT, rule, 'test', segments)
        replays[rule] = full_replay(rule, 'test', arr, weights, segments, prepared, 'Anchor_LSTM')
        expected = statistics({key: arr}, key)
        expected.update(rows=len(arr['truth']), clusters=len(np.unique(arr['cluster'])), discharges=len(np.unique(arr['discharge'])))
        A.compare_nested(report['cells'][key], expected, key)
        x = D.aligned_inputs(UNIT, rule, 'test')
        assert np.array_equal(x['idx'], arr['idx'])
        check_strata(arr, x['x'], cuts, report['failure_strata'][key], key)
        arrays[key] = arr
    scopes = {'overall_8rules': arrays, 'overall_D10': {UNIT + '/D10': arrays[UNIT + '/D10']}}
    assert set(report['averages']) == set(scopes)
    for scope, cells in scopes.items():
        A.compare_nested(report['averages'][scope], statistics(cells, 'average/' + scope), scope)
    low = report['failure_strata'][UNIT + '/D10']['fastchange/low']
    low_interval = low['pairs'].get('Anchor_LSTM_over_Direct_LSTM', {}).get('cluster', {}).get('ci95')
    statements = {
        'R1': report['averages']['overall_8rules']['pairs']['Anchor_LSTM_over_GP_fast_boost']['cluster']['ci95'][0] > 0,
        'R2': report['averages']['overall_8rules']['pairs']['Anchor_LSTM_over_Direct_LSTM']['cluster']['ci95'][0] > 0,
        'R3': bool(low['interval_available'] and low_interval is not None and low_interval[0] > 0)}
    assert report['statements'] == statements
    assert report['statement_interval_available'] == {'R1': True, 'R2': True, 'R3': low['interval_available']}
    A.compare_nested(report['global_group_contract'], {'overlapping_row_identities_consistent': True, 'unique_rows': len(identities)})
    trace = PREVIOUS.verify_trace(arrays[UNIT + '/D10'], report['trace'])
    for key in ('anonymous_discharge', 'rows', 'anonymous_idx_ordered', 'time_seconds_ordered', 'segment_ordered', 'support_counts'):
        assert report['trace'][key] == old_report['trace'][key], 'Changed frozen B118 trace: ' + key
    assert report['trace']['B118_readout_sha256'] == C.sha(DIRECT / 'readout.json')
    assert report['trace']['original_B118_selection'] == old_report['trace']['selection']
    sd = prepared['identity']['stats']['ti_sd']
    F.check_summary(report['learning']['direct_B117'], direct_history, direct_params, sd)
    F.check_summary(report['learning']['anchor_B119'], read(C.FIT / 'history.json'), params, sd)
    assert report['learning_sources'] == {'anchor_history_sha256': C.sha(C.FIT / 'history.json'),
        'anchor_params_sha256': C.sha(C.FIT / 'params.json'),
        'direct_B117_readout_sha256': C.sha(C.ROOT / 'data/.b117_focused/readout.json')}
    assert C.sha(C.OUT / 'val_audit.json') == gate_sha
    assert C.sha(C.FIT / 'best.npz') == gate['best_sha256'] and C.sha(C.FIT / 'latest.pt') == gate['latest_sha256']
    assert C.source_guard() == sources and C.verify_inputs() == inputs
    audit = {'all_pass': True, 'sources': sources, 'inputs': inputs,
        'readout_sha256': C.sha(C.OUT / 'readout.json'), 'staged_readout': staged['files'],
        'production_fits': 1, 'TEST_optimization_steps': 0, 'unit': UNIT,
        'val_audit_sha256': gate_sha, 'val_gate_reaudit': repeat,
        'full_TEST_replay': replays, 'TEST_data_checks': data_checks,
        'replay_scope': 'ALL seven common VAL and ALL eight common TEST rule supports, full segment context; independent LSTM correction plus causal anchor',
        'bootstrap_scope': 'All scalar/cell/macro/eligible-stratum paired20k intervals, independent bincount with B119 CRC prefix',
        'trace': trace, 'statements': statements, 'global_unique_rows': len(identities),
        'atol_eV': ATOL, 'rtol': RTOL, 'seconds': time.time() - started}
    C.save(C.OUT / 'audit.json', audit)
    print('B119 independent anchored-LSTM final audit passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--val', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.val:
        val_gate()
    else:
        main()
