"""Independent fixed-checkpoint B118 arithmetic and full TEST recurrent replay.

No fitting or model selection. The frozen B117 independent functional LSTM
verifies every common TEST row with full segment context. Original B116 raw
input/mask checks are reused. Bootstrap sums use independent bincount arithmetic.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import json
import sys
import time
import zlib
from pathlib import Path
import numpy as np
import common118 as C
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'b117_lstm_focus'))
import audit117 as FROZEN
A, D = FROZEN.A, FROZEN.D

UNIT = 'V1_r2_k0'
RULES = ('D10', 'D50', 'D100', 'D300', 'D650', 'D1000', 'S', 'K10')
MODELS = ('LSTM_focused', 'LSTM_short', 'GP_fast_boost')
PAIRS = (('LSTM_focused', 'GP_fast_boost'), ('LSTM_focused', 'LSTM_short'))
ATOL, RTOL = .05, 1e-4
OLD = C.ROOT / 'data/.b116'
FOCUSED = C.ROOT / 'data/.b117_focused'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def statistics(cells, key, models=MODELS, pairs=PAIRS, groupings=('cluster', 'discharge')):
    """Independent paired20k global-group resampling, with exact shared draws."""
    arrays = list(cells.values())
    losses = [np.column_stack([(np.asarray(a[n], dtype=np.float64) - a['truth']) ** 2 for n in models]) for a in arrays]
    points = np.mean([loss.mean(0) for loss in losses], 0)
    result = {'cells': list(cells), 'cell_count': len(cells), 'models': {}, 'pairs': {}, 'bootstrap': {}}
    assert all(len(loss) and np.isfinite(loss).all() for loss in losses)
    for j, name in enumerate(models):
        result['models'][name] = {'MSE': float(points[j]), 'RMSE': float(np.sqrt(points[j])),
            'bias': float(np.mean([np.mean(a[name] - a['truth']) for a in arrays]))}
    for a, b in pairs:
        result['pairs'][a + '_over_' + b] = {'point': float(1 - points[models.index(a)] / points[models.index(b)])}
    for grouping in groupings:
        labels = np.unique(np.concatenate([a[grouping] for a in arrays]))
        ng = len(labels)
        assert ng >= 2
        counts, sums = [], []
        for a, loss in zip(arrays, losses):
            inverse = np.searchsorted(labels, a[grouping])
            counts.append(np.bincount(inverse, minlength=ng))
            sums.append(np.column_stack([np.bincount(inverse, weights=loss[:, j], minlength=ng) for j in range(len(models))]))
        seed = [20261128, zlib.crc32((key + '/' + grouping).encode())]
        rng = np.random.default_rng(seed)
        draws, accepted, generated, rejected = [], 0, 0, 0
        while accepted < 20000:
            n = min(500, 20000 - accepted)
            multiplicities = rng.multinomial(ng, np.full(ng, 1 / ng), size=n)
            support = np.column_stack([multiplicities @ count for count in counts])
            valid = (support > 0).all(1)
            generated += n
            rejected += int((~valid).sum())
            assert generated < 2000000
            if valid.any():
                macro = sum((multiplicities[valid] @ total) / support[valid, i, None] for i, total in enumerate(sums)) / len(arrays)
                draws.append(macro)
                accepted += len(macro)
        draws = np.concatenate(draws)
        assert draws.shape == (20000, len(models)) and np.isfinite(draws).all()
        quantiles = np.quantile(draws, [.025, .975], axis=0)
        for j, name in enumerate(models):
            result['models'][name][grouping + '_MSE95'] = quantiles[:, j].tolist()
            result['models'][name][grouping + '_RMSE95'] = np.sqrt(quantiles[:, j]).tolist()
        for a, b in pairs:
            q = np.quantile(1 - draws[:, models.index(a)] / draws[:, models.index(b)], [.025, .975])
            result['pairs'][a + '_over_' + b][grouping] = {'ci95': q.tolist(), 'excludes_zero': bool(q[0] > 0 or q[1] < 0)}
        result['bootstrap'][grouping] = {'groups': ng, 'accepted': accepted, 'generated': generated,
            'zero_support_redraws': rejected, 'seed': seed}
    return result


def stratum_masks(x, cuts):
    x64 = np.asarray(x, dtype=np.float64)
    features = {'age': np.expm1(x64[:, 17]),
        'fastchange': np.sqrt(np.mean((x64[:, :13] - x64[:, 22:]) ** 2, 1)),
        'fastlevel': np.sqrt(np.mean(x64[:, :13] ** 2, 1))}
    masks = {}
    for name, value in features.items():
        lo, hi = cuts[name]
        eligible = x[:, 18] >= .5 if name == 'age' else np.ones(len(x), dtype=bool)
        masks[name + '/low'] = eligible & (value <= lo)
        masks[name + '/middle'] = eligible & (value > lo) & (value <= hi)
        masks[name + '/high'] = eligible & (value > hi)
    masks['history/absent'], masks['history/present'] = x[:, 18] < .5, x[:, 18] >= .5
    masks['dt/normal'], masks['dt/large'] = np.expm1(x[:, 15]) <= .015, np.expm1(x[:, 15]) > .015
    return masks


def check_strata(arr, x, cuts, reported, key):
    masks = stratum_masks(x, cuts)
    assert set(masks) == set(reported)
    for label, mask in masks.items():
        item = reported[label]
        n, ng = int(mask.sum()), len(np.unique(arr['cluster'][mask]))
        assert item['rows'] == n and item['clusters'] == ng
        assert item['discharges'] == len(np.unique(arr['discharge'][mask]))
        assert item['interval_available'] == (ng >= 8)
        if not n:
            assert not item['models'] and not item['pairs']
            continue
        for name in MODELS:
            error = np.asarray(arr[name][mask], dtype=np.float64) - arr['truth'][mask]
            A.compare_nested(item['models'][name], {'MSE': float(np.mean(error ** 2)),
                'RMSE': float(np.sqrt(np.mean(error ** 2))), 'bias': float(np.mean(error))})
        for a, b in PAIRS:
            mse_a, mse_b = item['models'][a]['MSE'], item['models'][b]['MSE']
            A.compare_nested(item['pairs'][a + '_over_' + b], {'point': 1 - mse_a / mse_b})
        if ng >= 8:
            sub = {k: np.asarray(v)[mask] for k, v in arr.items() if np.asarray(v).ndim == 1 and len(v) == len(mask)}
            expected = statistics({key: sub}, key + '/' + label, groupings=('cluster',))
            for field in ('models', 'pairs', 'bootstrap'):
                A.compare_nested(item[field], expected[field], key + '/' + label + '/' + field)


def replay_all(rule, segments, arr, weights, prepared):
    ref = D.common_row_support(UNIT, rule, 'test')
    stats = prepared['identity']['stats']
    ids, predictions = [], []
    for batch in D.batches(segments, 'cpu', batch_size=16):
        pred = FROZEN.functional_batch(weights, batch['x'].numpy())
        anonymous = D.anonymous_indices(batch['gi'].numpy())
        keep = batch['mask'].numpy() & np.isin(anonymous, ref['idx'])
        ids.extend(anonymous[keep].tolist())
        predictions.extend((pred[keep] * stats['ti_sd'] + stats['ti_mu']).tolist())
    ids, predictions = np.asarray(ids), np.asarray(predictions)
    order = np.argsort(ids)
    assert np.array_equal(ids[order], np.sort(ref['idx'])), 'Missing/duplicate full TEST support'
    independent = predictions[order][np.searchsorted(ids[order], ref['idx'])]
    saved = arr['LSTM_focused']
    assert np.allclose(independent, saved, atol=ATOL, rtol=RTOL), rule + '/full functional replay'
    point = float(np.mean((independent - ref['truth']) ** 2))
    saved_point = float(np.mean((saved - ref['truth']) ** 2))
    bound = float(np.mean(np.abs(independent - saved) * (np.abs(independent - ref['truth']) + np.abs(saved - ref['truth']))))
    assert abs(point - saved_point) <= bound + 1e-7
    return {'all_pass': True, 'rows': len(independent), 'complete_segments': len(segments),
        'max_abs_eV': float(np.max(np.abs(independent - saved))), 'independent_MSE': point,
        'saved_prediction_MSE': saved_point, 'MSE_absolute_difference_bound': bound,
        'atol_eV': ATOL, 'rtol': RTOL}

def independent_coordinates(idx, discharge):
    _, codes = D._row_mapping()
    inverse = np.argsort(codes)
    assert np.array_equal(codes[inverse], np.arange(len(codes)))
    grid, rows = D.raw_sources()
    gi = np.asarray(rows['gi'])[inverse[np.asarray(idx, dtype=np.int64)]]
    assert np.array_equal(D.anonymous_indices(gi), idx) and len(np.unique(gi)) == len(idx)
    times = np.asarray(grid['time'][gi], dtype=np.float64)
    raw_segments = grid['seg'][gi]
    ordinal = np.empty(len(idx), dtype=np.int64)
    for group in np.unique(discharge):
        take = np.flatnonzero(discharge == group)
        ordered = sorted((float(np.min(times[take][raw_segments[take] == seg])), int(seg)) for seg in np.unique(raw_segments[take]))
        mapping = {raw: number for number, (_, raw) in enumerate(ordered)}
        ordinal[take] = [mapping[int(raw)] for raw in raw_segments[take]]
    return gi, times, ordinal


def verify_trace(arr, reported):
    labels, counts = np.unique(arr['discharge'], return_counts=True)
    pairs = list(zip(labels.tolist(), counts.tolist()))
    group, count = sorted(pairs, key=lambda pair: (abs(pair[1] - 150), pair[0]))[0]
    take = np.flatnonzero(arr['discharge'] == group)
    # Segment ordinals are chronological; physical time orders points within each segment.
    order = take[np.lexsort((arr['time'][take], arr['segment'][take]))]
    assert reported['rule'] == 'D10' and reported['anonymous_discharge'] == group and reported['rows'] == count
    expected_counts = [{'anonymous_discharge': code, 'rows': n} for code, n in pairs]
    assert reported['support_counts'] == expected_counts
    assert np.array_equal(reported['anonymous_idx_ordered'], arr['idx'][order])
    assert np.array_equal(reported['time_seconds_ordered'], arr['time'][order])
    assert np.array_equal(reported['segment_ordered'], arr['segment'][order])
    for segment in np.unique(arr['segment'][order]):
        times = arr['time'][order][arr['segment'][order] == segment]
        assert np.all(np.diff(times) > 0)
    return {'all_pass': True, 'anonymous_discharge': group, 'rows': count,
        'selection': 'Support count closest to150; anonymous code breaks ties; no residual-based choice',
        'segments': len(np.unique(arr['segment'][order])), 'support_counts': expected_counts}


def main():
    started = time.time()
    sources, inputs = C.source_guard(), C.verify_inputs()
    launch = read(C.OUT / 'workflow/started.json')
    assert launch['sources'] == sources and launch['inputs'] == inputs and launch['optimization_steps'] == 0
    assert launch['GPU_end'] == launch['started'] + 7200 and launch['CPU_end'] == launch['started'] + 14400
    begin = read(C.OUT / 'predict_started.json')
    complete = read(C.OUT / 'predictions_complete.json')
    assert begin['sources'] == complete['sources'] == sources
    assert begin['inputs'] == complete['inputs'] == inputs
    assert begin['optimization_steps'] == complete['optimization_steps'] == 0
    assert complete['all_pass'] and complete['scenarios'] == len(complete['files']) == 8
    assert complete['model_state_unchanged']
    assert launch['started'] <= begin['started'] <= complete['completed'] <= launch['GPU_end']
    assert not (C.OUT / 'fits').exists() and not (C.OUT / 'selection.json').exists()
    A.checked_files(C.OUT, complete['files'])
    expected_files = {'test/' + UNIT + '_' + rule + '.npz' for rule in RULES}
    assert set(complete['files']) == expected_files
    actual_files = {p.relative_to(C.OUT).as_posix() for p in (C.OUT / 'test').iterdir() if p.is_file()}
    assert actual_files == expected_files
    # Independently trace the fixed focused checkpoint to its audited VAL-only selection.
    prior_audit, prior_report = read(FOCUSED / 'audit.json'), read(FOCUSED / 'readout.json')
    prior_complete = read(FOCUSED / 'fit_complete.json')
    prior_params = read(C.FIT / 'params.json')
    assert prior_audit['all_pass'] and prior_audit['converged'] and not prior_audit['TEST_accessed']
    assert C.sha(FOCUSED / 'readout.json') == prior_audit['readout_sha256']
    assert C.sha(FOCUSED / 'fit_complete.json') == prior_report['fit_complete_sha256']
    assert C.sha(C.FIT / 'params.json') == prior_complete['params_sha256']
    assert prior_complete['production_fits'] == 1 and not prior_complete['TEST_accessed']
    A.checked_files(FOCUSED, prior_complete['files'])
    assert prior_params['status'] == 'CONVERGED' and prior_params['seed'] == 1701 and prior_params['unit'] == UNIT
    best_sha, latest_sha = C.sha(C.FIT / 'best.npz'), C.sha(C.FIT / 'latest.pt')
    assert best_sha == prior_audit['recovery_checkpoint']['best_sha256'] == inputs['B117_best_sha256']
    assert latest_sha == prior_audit['recovery_checkpoint']['latest_sha256'] == inputs['B117_latest_sha256']
    assert complete['checkpoint_before'] == complete['checkpoint_after'] == {'best': best_sha, 'latest': latest_sha}
    # Independently bind old single-seed predictions and frozen TRAIN cuts to the old audit.
    old_audit, old_report = read(OLD / 'audit.json'), read(OLD / 'readout.json')
    old_complete = read(OLD / 'predictions_complete.json')
    assert old_audit['all_pass'] and C.sha(OLD / 'readout.json') == old_audit['readout_sha256']
    assert C.sha(OLD / 'predictions_complete.json') == old_report['predictions_complete_sha256']
    assert C.sha(OLD / 'fit_complete.json') == old_audit['fit_complete_sha256'] == old_complete['fit_complete_sha256']
    selection_sha = C.sha(OLD / 'selection.json')
    assert selection_sha == old_audit['selection_sha256'] == old_complete['selection_sha256'] == inputs['B116_selection_sha256']
    cuts = read(OLD / 'selection.json')['input_cuts'][UNIT]
    report = read(C.OUT / 'readout.json')
    staged = read(C.OUT / 'readout_complete.json')
    assert staged['all_pass'] and not staged['published']
    A.checked_files(C.OUT, staged['files'])
    assert report['sources'] == sources and report['inputs'] == inputs
    assert report['new_fits'] == report['model_choices'] == 0 and report['unit'] == UNIT
    assert report['primary_validation_design'] is None
    assert report['validation_designs'] == {u: ('EVALUATED: one fixed single fit' if u == UNIT else 'NOT RUN') for u in D.UNITS}
    assert report['input_cuts'] == cuts and report['input_cuts_source_sha256'] == selection_sha
    short_params = read(OLD / 'fits' / UNIT / 'lstm/s1701/params.json')
    assert report['fixed_checkpoint'] == {'architecture': 'lstm', 'seed': 1701,
        'focused_best_epoch': prior_params['best_epoch'], 'short_best_epoch': short_params['best_epoch'],
        'focused_best_sha256': best_sha}
    assert report['predictions_complete_sha256'] == staged['predictions_complete_sha256'] == C.sha(C.OUT / 'predictions_complete.json')
    assert complete['completed'] <= report['created_unix'] <= launch['CPU_end']
    prepared = D.load_prepared(UNIT)
    normalization = A.verify_normalization(prepared)
    weights = C.npz(C.FIT / 'best.npz')
    arrays, data_checks, replays = {}, {}, {}
    identities, discharge_cluster = {}, {}
    assert set(report['cells']) == set(report['failure_strata']) == {UNIT + '/' + rule for rule in RULES}
    for rule in RULES:
        key, rel = UNIT + '/' + rule, 'test/' + UNIT + '_' + rule + '.npz'
        arr = C.npz(C.OUT / rel)
        assert set(arr) == {'idx', 'truth', 'cluster', 'discharge', 'time', 'segment', *MODELS}
        assert all(np.asarray(v).shape == arr['truth'].shape for v in arr.values())
        assert all(np.isfinite(arr[name]).all() for name in MODELS)
        assert C.sha(OLD / rel) == old_complete['files'][rel] == inputs['baseline_test_files'][rel]
        old = C.npz(OLD / rel)
        reference = D.common_row_support(UNIT, rule, 'test')
        for name in ('idx', 'truth', 'cluster', 'discharge', 'GP_fast_boost'):
            assert np.array_equal(arr[name], reference[name]) and np.array_equal(old[name], reference[name])
        assert np.array_equal(arr['LSTM_short'], old['Seq_lstm_s1701'])
        gi, times, segments_ordinal = independent_coordinates(arr['idx'], arr['discharge'])
        assert np.array_equal(arr['time'], times) and np.array_equal(arr['segment'], segments_ordinal)
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
        replays[rule] = replay_all(rule, segments, arr, weights, prepared)
        expected = statistics({key: arr}, key)
        expected.update(rows=len(arr['truth']), clusters=len(np.unique(arr['cluster'])), discharges=len(np.unique(arr['discharge'])))
        A.compare_nested(report['cells'][key], expected, key)
        x = D.aligned_inputs(UNIT, rule, 'test')
        assert np.array_equal(x['idx'], arr['idx'])
        check_strata(arr, x['x'], cuts, report['failure_strata'][key], key)
        arrays[key] = arr
    scopes = {'overall_8rules': arrays, 'overall_D10': {UNIT + '/D10': arrays[UNIT + '/D10']}}
    assert set(report['averages']) == set(scopes)
    for name, cells in scopes.items():
        A.compare_nested(report['averages'][name], statistics(cells, 'average/' + name), name)
    statements = {
        'R1': report['averages']['overall_8rules']['pairs']['LSTM_focused_over_GP_fast_boost']['cluster']['ci95'][0] > 0,
        'R2': report['averages']['overall_D10']['pairs']['LSTM_focused_over_GP_fast_boost']['cluster']['ci95'][0] > 0,
        'R3': report['averages']['overall_8rules']['pairs']['LSTM_focused_over_LSTM_short']['cluster']['ci95'][0] > 0}
    assert report['statements'] == statements
    A.compare_nested(report['global_group_contract'], {'overlapping_row_identities_consistent': True, 'unique_rows': len(identities)})
    trace = verify_trace(arrays[UNIT + '/D10'], report['trace'])
    assert C.sha(C.FIT / 'best.npz') == best_sha and C.sha(C.FIT / 'latest.pt') == latest_sha
    assert C.verify_inputs() == inputs and C.source_guard() == sources
    audit = {'all_pass': True, 'sources': sources, 'inputs': inputs,
        'readout_sha256': C.sha(C.OUT / 'readout.json'), 'staged_readout': staged['files'],
        'optimization_steps': 0, 'model_choices': 0, 'unit': UNIT, 'rules': list(RULES),
        'workflow_started_sha256': C.sha(C.OUT / 'workflow/started.json'),
        'prediction_started_sha256': C.sha(C.OUT / 'predict_started.json'),
        'fixed_checkpoint': {'best_sha256': best_sha, 'latest_sha256': latest_sha,
            'prior_audit_sha256': C.sha(FOCUSED / 'audit.json'), 'prior_fit_complete_sha256': C.sha(FOCUSED / 'fit_complete.json')},
        'normalization_sha256': normalization, 'data_checks': data_checks, 'full_TEST_replay': replays,
        'replay_scope': 'ALL common TEST rows in ALL eight rules, with complete segment context; frozen independent functional LSTM',
        'bootstrap_scope': 'All cell/macro/eligible-stratum points and paired20k intervals independently replayed; synchronized global multiplicities',
        'strata_scope': 'Already frozen B116 TRAIN-only cuts, all cells/strata, no error-defined bins',
        'trace': trace, 'global_unique_rows': len(identities), 'global_discharges': len(discharge_cluster),
        'statements': statements, 'atol_eV': ATOL, 'rtol': RTOL, 'seconds': time.time() - started}
    C.save(C.OUT / 'audit.json', audit)
    print('B118 independent fixed-checkpoint evaluation audit passed')


if __name__ == '__main__':
    main()
