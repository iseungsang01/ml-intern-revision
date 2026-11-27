"""B117 independent TRAIN/VAL-only audit; no fitting or TEST model prediction.

Functional gate arithmetic is reused from frozen B116 audit116, never from the
producer model. Full raw inputs/masks and scalar history/VAL arithmetic are
checked. Independent recurrent replay covers ALL common VAL rows and all
seven rules, with complete segment context. No TEST prediction is computed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '2'
import argparse
import hashlib
import json
import random
import sys
import time
import zlib
from pathlib import Path
import numpy as np
import common117 as C
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'b116_temporal'))
import data116 as D
import audit116 as A

UNIT, SEED = 'V1_r2_k0', 1701
RULES = ('D10', 'D50', 'D100', 'D300', 'D650', 'D1000', 'S')
BASE = C.ROOT / 'data/.b116/fits' / UNIT / 'lstm/s1701'
ATOL, RTOL = .05, 1e-4


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stop_tracking(history):
    """Independent minimum budget / meaningful improvement / terminal LR rule."""
    significant, last, low_streak = float('inf'), 0, 0
    output = []
    for epoch, entry in enumerate(history, 1):
        score = float(entry['seven_rule_mean_mse'])
        improved = not np.isfinite(significant) or score < significant * .999
        if improved:
            significant, last = score, epoch
        low_streak = low_streak + 1 if entry['lr_used'] <= 1e-5 else 0
        no_significant = epoch - last
        converged = epoch >= 50 and low_streak >= 10 and no_significant >= 20 and entry['elapsed_seconds'] < 21600
        output.append({'epoch': epoch, 'meaningful_reference_mse': significant,
                       'last_meaningful_epoch': last, 'epochs_without_meaningful_improvement': no_significant,
                       'low_lr_streak': low_streak, 'converged': converged})
    return output


def verify_prefix(history):
    previous = read(BASE / 'history.json')
    oldparams = read(BASE / 'params.json')
    assert len(previous) == oldparams['completed_epochs'] == 17
    assert len(history) >= 17, 'Cannot assess the registered extended training below original17epochs'
    maximum = {'train_normalized_MSE': 0., 'seven_rule_mean_mse': 0.}
    for old, new in zip(previous, history):
        for k in ('epoch', 'train_count', 'updates', 'assignment_sha256', 'lr_used', 'lr_next'):
            assert old[k] == new[k], 'Original trajectory prefix mismatch: ' + k
        old_loss = old['train_normalized_sse'] / old['train_count']
        new_loss = new['train_normalized_sse'] / new['train_count']
        assert np.isclose(old_loss, new_loss, rtol=1e-6, atol=1e-5)
        assert np.isclose(old['seven_rule_mean_mse'], new['seven_rule_mean_mse'], rtol=1e-6, atol=.01)
        maximum['train_normalized_MSE'] = max(maximum['train_normalized_MSE'], abs(old_loss - new_loss))
        maximum['seven_rule_mean_mse'] = max(maximum['seven_rule_mean_mse'], abs(old['seven_rule_mean_mse'] - new['seven_rule_mean_mse']))
        for rule in RULES:
            assert old['val'][rule]['count'] == new['val'][rule]['count']
            assert np.isclose(old['val'][rule]['mse'], new['val'][rule]['mse'], rtol=1e-6, atol=.01)
    return {'all_pass': True, 'epochs': 17, 'old_history_sha256': C.sha(BASE / 'history.json'),
            'old_params_sha256': C.sha(BASE / 'params.json'), 'maximum': maximum,
            'rtol': 1e-6, 'normalized_MSE_atol': 1e-5, 'physical_VAL_MSE_atol': .01}


def verify_history(history, params, prepared):
    assert 1 <= len(history) <= 200 and len(history) == params['completed_epochs']
    scores = np.asarray([h['seven_rule_mean_mse'] for h in history])
    assert np.isfinite(scores).all()
    assert params['best_epoch'] == int(np.argmin(scores)) + 1
    assert params['best_val_mse'] == float(scores.min())
    assert params['unit'] == UNIT and params['seed'] == SEED and params['architecture'] == 'lstm'
    assert params['inputs']['prepared_artifact_sha256'] == prepared['artifact_sha256']
    if 'manifest_sha256' in params:
        assert params['manifest_sha256'] == digest(prepared['identity']['manifest'])
        assert params['stats_sha256'] == digest(prepared['identity']['stats'])
    lr, best, bad = .001, float('inf'), 0
    elapsed = 0.
    keys = sorted((s['fidx'], s['segment']) for s in prepared['train'][RULES[0]])
    for epoch, h in enumerate(history, 1):
        assert h['epoch'] == epoch
        assert np.isclose(h['lr_used'], lr, rtol=0, atol=1e-14)
        assert set(h['val']) == set(RULES)
        vals = []
        for rule in RULES:
            v = h['val'][rule]
            assert v['count'] == len(D.common_row_support(UNIT, rule, 'val')['idx'])
            assert np.isclose(v['mse'], v['sse'] / v['count'], rtol=1e-12)
            vals.append(v['mse'])
        score = float(np.mean(vals))
        assert np.isclose(score, h['seven_rule_mean_mse'], rtol=1e-14, atol=1e-10)
        if score < best * (1 - 1e-4):
            best, bad = score, 0
        else:
            bad += 1
        if bad > 3:
            new_lr = max(lr * .5, 1e-5)
            if lr - new_lr > 1e-8:
                lr = new_lr
            bad = 0
        assert np.isclose(h['lr_next'], lr, rtol=0, atol=1e-14)
        assert np.isfinite(h['train_normalized_sse']) and h['train_count'] > 0 and h['updates'] > 0
        assert elapsed <= h['elapsed_seconds']
        elapsed = h['elapsed_seconds']
        rng = np.random.default_rng([SEED, epoch, zlib.crc32(UNIT.encode())])
        choices = rng.integers(0, len(RULES), len(keys))
        assignment = [(f, s, RULES[int(i)]) for (f, s), i in zip(keys, choices)]
        assert h['assignment_sha256'] == digest(assignment)
    tracking = stop_tracking(history)
    for h, t in zip(history, tracking):
        assert h['significant_best'] == t['meaningful_reference_mse']
        assert h['last_significant_epoch'] == t['last_meaningful_epoch']
        assert h['epochs_since_significant'] == t['epochs_without_meaningful_improvement']
        assert h['low_lr_streak'] == t['low_lr_streak']
        assert h['best_epoch'] == int(np.argmin(scores[:h['epoch']])) + 1
    assert params['total_updates'] == sum(h['updates'] for h in history) + params['discarded_partial_updates']
    reached = [i for i, t in enumerate(tracking, 1) if t['converged']]
    assert not reached or reached[0] == len(history), 'Fit continued after first registered convergence'
    return tracking


def functional_batch(weights, x):
    """Independent explicit LSTM gates, vectorized only over real segment batches."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    p = {k.removeprefix('model.'): torch.as_tensor(v, dtype=torch.float32) for k, v in weights.items()}
    xx = torch.as_tensor(x, dtype=torch.float32)
    h, c = torch.zeros(len(xx), 64), torch.zeros(len(xx), 64)
    output = []
    with torch.no_grad():
        for t in range(xx.shape[1]):
            gates = F.linear(xx[:, t], p['recurrent.weight_ih_l0'], p['recurrent.bias_ih_l0'])
            gates += F.linear(h, p['recurrent.weight_hh_l0'], p['recurrent.bias_hh_l0'])
            i, f, g, o = gates.chunk(4, -1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            v = F.layer_norm(h, (64,), p['readout.0.weight'], p['readout.0.bias'], 1e-5)
            v = F.gelu(F.linear(v, p['readout.1.weight'], p['readout.1.bias']), approximate='none')
            output.append(F.linear(v, p['readout.3.weight'], p['readout.3.bias'])[:, 0])
    return torch.stack(output, 1).numpy().astype(np.float64)


def verify_all_val(rule, arr, weights, prepared):
    ref = D.common_row_support(UNIT, rule, 'val')
    for name, expected in ref.items():
        assert np.array_equal(arr[name], expected), rule + '/VAL/' + name
    stats = prepared['identity']['stats']
    ids, predictions = [], []
    for batch in D.batches(prepared['val'][rule], 'cpu', batch_size=16):
        normalized = functional_batch(weights, batch['x'].numpy())
        anon = D.anonymous_indices(batch['gi'].numpy())
        keep = batch['mask'].numpy() & np.isin(anon, ref['idx'])
        ids.extend(anon[keep].tolist())
        predictions.extend((normalized[keep] * stats['ti_sd'] + stats['ti_mu']).tolist())
    ids, predictions = np.asarray(ids), np.asarray(predictions)
    order = np.argsort(ids)
    assert np.array_equal(ids[order], np.sort(ref['idx']))
    pred = predictions[order][np.searchsorted(ids[order], ref['idx'])]
    actual = arr['prediction']
    assert np.allclose(pred, actual, atol=ATOL, rtol=RTOL), 'Independent full VAL prediction mismatch: ' + rule
    mse = float(np.mean((pred - ref['truth']) ** 2))
    saved = float(np.mean((actual - ref['truth']) ** 2))
    # A direct identity bounds any CPU/GPU MSE difference from verified prediction differences.
    bound = float(np.mean(np.abs(pred - actual) * (np.abs(pred - ref['truth']) + np.abs(actual - ref['truth']))))
    assert abs(mse - saved) <= bound + 1e-7
    return {'all_pass': True, 'rows': len(pred), 'complete_segments': len(prepared['val'][rule]),
            'max_abs_eV': float(np.max(np.abs(pred - actual))), 'independent_MSE': mse,
            'saved_prediction_MSE': saved, 'MSE_absolute_difference_bound': bound,
            'atol_eV': ATOL, 'rtol': RTOL}


def verify_checkpoint(history, params):
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
    weights = npz(C.FIT / 'best.npz')
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
    sources = C.source_guard()
    inputs = C.verify_inputs()
    receipt = read(C.OUT / 'smoke.json')
    assert receipt['all_pass'] and receipt['sources'] == sources and receipt['inputs'] == inputs
    assert receipt['training_updates'] == 0 and not receipt['TEST_accessed']
    A.checked_files(C.OUT, receipt['files'])
    prepared = D.load_prepared(UNIT)
    candidates = []
    for segment in prepared['train'][RULES[0]]:
        if segment['mask'][:64, 0].any():
            candidates.append(segment)
        if len(candidates) == 2:
            break
    assert len(candidates) == 2
    saved = npz(C.OUT / 'smoke/prefix.npz')
    weights = npz(C.OUT / 'smoke/weights.npz')
    sd = prepared['identity']['stats']['ti_sd']
    mu = prepared['identity']['stats']['ti_mu']
    batched = functional_batch(weights, saved['x'])
    maximum, rows = 0., 0
    for i, segment in enumerate(candidates):
        n = min(64, len(segment['gi']))
        assert saved['lengths'][i] == n
        assert np.array_equal(saved['x'][i, :n], segment['x'][:n])
        assert np.array_equal(saved['gi'][i, :n], segment['gi'][:n])
        assert np.array_equal(saved['mask'][i, :n], segment['mask'][:n, 0])
        assert int(saved['fidx'][i]) == segment['fidx'] and int(saved['segment'][i]) == segment['segment']
        pred = A.recurrent_predict(weights, 'lstm', saved['x'][i, :n])
        expected = saved['pred'][i, :n].astype(np.float64)
        assert np.allclose(pred * sd + mu, expected * sd + mu, atol=ATOL, rtol=RTOL)
        assert np.allclose(batched[i, :n] * sd + mu, expected * sd + mu, atol=ATOL, rtol=RTOL)
        k = min(8, n)
        assert np.array_equal(A.recurrent_predict(weights, 'lstm', saved['x'][i, :k]), pred[:k])
        maximum = max(maximum, float(np.max(np.abs(pred - expected)) * sd))
        rows += n
    C.save(C.OUT / 'smoke_independent.json', {'all_pass': True, 'sources': sources,
        'inputs': inputs, 'smoke_sha256': C.sha(C.OUT / 'smoke.json'), 'training_updates': 0,
        'TEST_accessed': False, 'real_TRAIN_rows': rows, 'real_TRAIN_segments': 2, 'max_abs_eV': maximum,
        'atol_eV': ATOL, 'rtol': RTOL, 'equations': 'Frozen B116 scalar gates and independent B117 batched gates'})
    print('B117 nontraining TRAIN prefix qualification passed')

def check_summary(reported, history, params, sd):
    best = int(np.argmin([h['seven_rule_mean_mse'] for h in history]))
    cumulative, curves = 0, []
    for h in history:
        normalized = h['train_normalized_sse'] / h['train_count']
        physical = normalized * sd * sd
        cumulative += h['updates']
        curves.append({'epoch': h['epoch'], 'train_mse_eV2': physical,
            'train_normalized_mse': normalized, 'train_count': h['train_count'],
            'val_mse_eV2': h['seven_rule_mean_mse'],
            'val_minus_online_train_mse_eV2': h['seven_rule_mean_mse'] - physical,
            'val_by_rule': {r: h['val'][r]['mse'] for r in RULES},
            'lr_used': h['lr_used'], 'lr_next': h['lr_next'], 'updates': h['updates'],
            'cumulative_updates': cumulative, 'elapsed_seconds': h['elapsed_seconds']})
    expected = {'completed_epochs': len(history), 'best_epoch': best + 1,
        'best_val_mse_eV2': history[best]['seven_rule_mean_mse'],
        'best_val_rmse_eV': float(np.sqrt(history[best]['seven_rule_mean_mse'])),
        'best_epoch_online_train_mse_eV2': curves[best]['train_mse_eV2'],
        'best_epoch_descriptive_gap_eV2': curves[best]['val_minus_online_train_mse_eV2'],
        'best_val_by_rule': curves[best]['val_by_rule'],
        'last_val_mse_eV2': history[-1]['seven_rule_mean_mse'], 'completed_updates': cumulative,
        'total_updates': params['total_updates'], 'discarded_partial_updates': params['discarded_partial_updates'],
        'actual_seconds': params['actual_seconds'], 'stop_reason': params['stop_reason'], 'curve': curves}
    A.compare_nested(reported, expected)
    return expected


def main():
    start = time.time()
    sources = C.source_guard()
    inputs = C.verify_inputs()
    complete = read(C.OUT / 'fit_complete.json')
    assert complete['all_pass'] and complete['production_fits'] == 1 and not complete['TEST_accessed']
    assert complete['sources'] == sources
    assert not (C.OUT / 'test').exists() and not (C.OUT / 'selection.json').exists()
    assert sorted(p.name for p in (C.OUT / 'fits').iterdir()) == [C.FIT.name]
    A.checked_files(C.OUT, complete['files'])
    params = read(C.FIT / 'params.json')
    assert complete['params_sha256'] == C.sha(C.FIT / 'params.json')
    assert params['sources'] == sources and params['inputs'] == inputs
    assert params['env'] == C.ENV_PINS
    assert params['runtime']['torch'] and params['runtime']['cuda'] and params['runtime']['device']
    assert (params['min_epochs'], params['max_epochs'], params['budget_seconds']) == (50, 200, 21600)
    A.checked_files(C.FIT, params['files'])
    qualified = read(C.OUT / 'smoke_independent.json')
    assert qualified['all_pass'] and qualified['sources'] == sources and not qualified['TEST_accessed']
    assert qualified['training_updates'] == 0 and qualified['smoke_sha256'] == C.sha(C.OUT / 'smoke.json')
    prepared = D.load_prepared(UNIT)
    normalization = A.verify_normalization(prepared)
    data_checks = {}
    for split in ('train', 'val'):
        for rule in RULES:
            data_checks[split + '/' + rule] = A.verify_segments(UNIT, rule, split, prepared[split][rule])
    history = read(C.FIT / 'history.json')
    tracking = verify_history(history, params, prepared)
    prefix = verify_prefix(history)
    producer_prefix = read(C.FIT / 'prefix_replay.json')
    assert producer_prefix['all_pass'] and producer_prefix['epochs'] == 17
    assert producer_prefix['original_history_sha256'] == C.sha(BASE / 'history.json')
    converged = tracking[-1]['converged']
    assert params['converged'] == complete['converged'] == converged
    assert params['status'] == complete['status'] == ('CONVERGED' if converged else 'SAFETY_CAPPED_NOT_CONVERGED')
    if converged:
        assert params['stop_reason'] == 'registered_statistical_stop'
    else:
        assert params['stop_reason'] in ('max_epochs_safety', 'wall_safety', 'wall_safety_incomplete_epoch', 'wall_safety_incomplete_validation', 'wall_safety_after_complete_validation')
        assert len(history) == 200 or params['actual_seconds'] >= 21600
    recovery = verify_checkpoint(history, params)
    weights = npz(C.FIT / 'best.npz')
    old_history, old_params = read(BASE / 'history.json'), read(BASE / 'params.json')
    original_sources = read(C.ROOT / 'data/.b116/workflow/started.json')['sources']
    assert old_params['sources'] == original_sources
    assert old_params['history_sha256'] == C.sha(BASE / 'history.json')
    assert old_params['best_sha256'] == C.sha(BASE / 'best.npz')
    staged = read(C.OUT / 'readout_complete.json')
    assert staged['all_pass'] and not staged['published']
    A.checked_files(C.OUT, staged['files'])
    report = read(C.OUT / 'readout.json')
    assert report['sources'] == sources and report['inputs'] == inputs and report['fit_count'] == 1 and not report['TEST_accessed']
    assert report['unit'] == UNIT and report['architecture'] == 'lstm' and report['seed'] == SEED
    assert report['converged'] == converged
    assert report['primary_validation_design'] is None
    assert report['validation_designs'] == {u: ('RUN: TRAIN/VAL optimization pilot only' if u == UNIT else 'NOT RUN') for u in D.UNITS}
    assert report['fit_complete_sha256'] == C.sha(C.OUT / 'fit_complete.json')
    sd = prepared['identity']['stats']['ti_sd']
    baseline = check_summary(report['baseline'], old_history, old_params, sd)
    focused = check_summary(report['focused'], history, params, sd)
    rules, validation = {}, {}
    for rule in RULES:
        rel = 'val/' + UNIT + '_' + rule + '.npz'
        arr = npz(C.OUT / rel)
        old = npz(C.ROOT / 'data/.b116' / rel)
        ref = D.common_row_support(UNIT, rule, 'val')
        for name in ('idx', 'truth', 'cluster', 'discharge'):
            assert np.array_equal(old[name], ref[name])
        validation[rule] = verify_all_val(rule, arr, weights, prepared)
        new_mse = validation[rule]['saved_prediction_MSE']
        old_mse = float(np.mean((np.asarray(old['Seq_lstm_s1701'], dtype=np.float64) - old['truth']) ** 2))
        b, f = baseline['best_val_by_rule'][rule], focused['best_val_by_rule'][rule]
        assert np.isclose(new_mse, f, rtol=1e-8, atol=.01)
        assert np.isclose(old_mse, b, rtol=1e-8, atol=.01)
        rules[rule] = {'baseline_mse_eV2': b, 'focused_mse_eV2': f,
                       'change_eV2': f - b, 'relative_reduction': 1 - f / b}
        A.compare_nested(report['validation_export_check'][rule], {
            'focused_export_mse_eV2': new_mse, 'baseline_export_mse_eV2': old_mse,
            'focused_export_sha256': C.sha(C.OUT / rel),
            'baseline_export_sha256': C.sha(C.ROOT / 'data/.b116' / rel)})
    A.compare_nested(report['comparison'], {'best_val_change_eV2': focused['best_val_mse_eV2'] - baseline['best_val_mse_eV2'],
        'best_val_relative_reduction': 1 - focused['best_val_mse_eV2'] / baseline['best_val_mse_eV2'],
        'best_val_by_rule': rules, 'completed_epoch_change': len(history) - len(old_history)})
    relation = 'same epoch' if focused['best_epoch'] == baseline['best_epoch'] else ('later' if focused['best_epoch'] > baseline['best_epoch'] else 'earlier')
    assert report['comparison']['best_epoch_relation'] == relation
    old_weights = npz(BASE / 'best.npz')
    identical = set(old_weights) == set(weights) and all(np.array_equal(weights[k], old_weights[k]) for k in weights)
    assert report['comparison']['selected_weights_identical_to_baseline'] == identical
    recovery_report = read(C.FIT / 'recovery.json')
    assert report['recovery_checkpoint'] == {**recovery_report, 'automatic_restart_authorized': False}
    for rel, expected in report['provenance'].items():
        assert C.sha(C.ROOT / rel) == expected
    assert C.source_guard() == sources
    audit = {'all_pass': True, 'sources': sources, 'inputs': inputs,
        'readout_sha256': C.sha(C.OUT / 'readout.json'), 'staged_readout': staged['files'],
        'status': params['status'], 'converged': converged,
        'classification': 'registered operational plateau' if converged else 'safety capped; optimization adequacy not established',
        'TEST_accessed': False, 'production_fits': 1, 'normalization_sha256': normalization,
        'data_checks': data_checks, 'original_prefix': prefix, 'recovery_checkpoint': recovery,
        'independent_stopping': tracking, 'full_VAL_replay': validation,
        'replay_scope': 'ALL common-support VAL rows, ALL seven rules, complete segment context; initialized real TRAIN prefixes qualified before fitting',
        'limitations': 'One fixed unit and seed; other validation designs unrun; repeated VAL use; no structural impossibility or TEST generalization claim',
        'seconds': time.time() - start}
    C.save(C.OUT / 'audit.json', audit)
    print('B117 independent TRAIN/VAL audit passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    (smoke if args.smoke else main)()
