"""Registered anchor-output LSTM readout after the independent VAL gate."""
import time
import zlib
import numpy as np
import common119 as C
import data116 as D
from select116 import strata_masks

FOCUSED, SHORT, GP = 'Anchor_LSTM', 'Direct_LSTM', 'GP_fast_boost'
MODELS = (FOCUSED, SHORT, GP)
PAIRS = ((FOCUSED, GP), (FOCUSED, SHORT))
NBOOT = 20000


def average_statistics(cells, key, models=MODELS, pairs=PAIRS, groupings=('cluster', 'discharge')):
    """Macro MSE before the ratio; synchronized union-group draws across cells.

    Arithmetic follows frozen B116, with the prospectively fixed B119 seed.
    Draws with zero support in ANY included cell are rejected as whole draws.
    """
    arrays = list(cells.values())
    assert arrays
    errors = [np.stack([np.asarray(a[n], dtype=np.float64) - a['truth'] for n in models], axis=1)
              for a in arrays]
    squared = [e ** 2 for e in errors]
    assert all(len(s) and np.isfinite(s).all() for s in squared)
    point = np.mean([s.mean(axis=0) for s in squared], axis=0)
    assert np.all(point > 0)
    result = {'cells': list(cells), 'cell_count': len(cells),
              'models': {n: {'MSE': float(point[i]), 'RMSE': float(np.sqrt(point[i])),
                              'bias': float(np.mean([e[:, i].mean() for e in errors]))}
                         for i, n in enumerate(models)},
              'pairs': {a + '_over_' + b: {'point': float(1 - point[models.index(a)] / point[models.index(b)])}
                        for a, b in pairs}, 'bootstrap': {}}
    for grouping in groupings:
        labels = np.unique(np.concatenate([a[grouping] for a in arrays]))
        ng = len(labels)
        assert ng >= 2
        sums, counts = [], []
        for a, s in zip(arrays, squared):
            inv = np.searchsorted(labels, a[grouping])
            total = np.zeros((ng, len(models)), dtype=np.float64)
            count = np.zeros(ng, dtype=np.int64)
            np.add.at(total, inv, s)
            np.add.at(count, inv, 1)
            sums.append(total)
            counts.append(count)
        seed = [20261128, zlib.crc32(('B119/' + key + '/' + grouping).encode())]
        rng = np.random.default_rng(seed)
        draws, accepted, rejected, generated = [], 0, 0, 0
        while accepted < NBOOT:
            size = min(500, NBOOT - accepted)
            weight = rng.multinomial(ng, np.full(ng, 1 / ng), size=size)
            support = np.stack([weight @ count for count in counts], axis=1)
            valid = np.all(support > 0, axis=1)
            rejected += int((~valid).sum())
            generated += size
            assert generated < 100 * NBOOT, 'Insufficient shared bootstrap support'
            if not valid.any():
                continue
            w = weight[valid]
            macro = np.zeros((int(valid.sum()), len(models)))
            for i, total in enumerate(sums):
                macro += (w @ total) / support[valid, i, None]
            draws.append(macro / len(arrays))
            accepted += int(valid.sum())
        draws = np.concatenate(draws)
        assert draws.shape == (NBOOT, len(models)) and np.isfinite(draws).all()
        ci = np.percentile(draws, [2.5, 97.5], axis=0)
        for i, n in enumerate(models):
            result['models'][n][grouping + '_MSE95'] = ci[:, i].tolist()
            result['models'][n][grouping + '_RMSE95'] = np.sqrt(ci[:, i]).tolist()
        for a, b in pairs:
            skill = 1 - draws[:, models.index(a)] / draws[:, models.index(b)]
            q = np.percentile(skill, [2.5, 97.5])
            result['pairs'][a + '_over_' + b][grouping] = {
                'ci95': q.tolist(), 'excludes_zero': bool(q[0] > 0 or q[1] < 0)}
        result['bootstrap'][grouping] = {'groups': ng, 'accepted': accepted, 'generated': generated,
                                          'zero_support_redraws': rejected, 'seed': seed}
    return result


def statistics(arr, key):
    result = average_statistics({key: arr}, key)
    result.update(rows=len(arr['truth']), clusters=len(np.unique(arr['cluster'])),
                  discharges=len(np.unique(arr['discharge'])))
    return result


def failure_statistics(arr, x, cuts, key):
    output = {}
    for label, mask in strata_masks(x, cuts).items():
        sub = {k: np.asarray(v)[mask] for k, v in arr.items()}
        n, clusters = int(mask.sum()), len(np.unique(sub['cluster']))
        row = {'rows': n, 'clusters': clusters, 'discharges': len(np.unique(sub['discharge'])),
               'interval_available': clusters >= 8, 'models': {}, 'pairs': {}}
        if n:
            for name in MODELS:
                e = np.asarray(sub[name], dtype=np.float64) - sub['truth']
                row['models'][name] = {'MSE': float(np.mean(e ** 2)),
                                       'RMSE': float(np.sqrt(np.mean(e ** 2))), 'bias': float(np.mean(e))}
            for a, b in PAIRS:
                assert row['models'][b]['MSE'] > 0
                row['pairs'][a + '_over_' + b] = {'point': 1 - row['models'][a]['MSE'] / row['models'][b]['MSE']}
        if clusters >= 8:
            full = average_statistics({key: sub}, key + '/' + label, groupings=('cluster',))
            row.update(models=full['models'], pairs=full['pairs'], bootstrap=full['bootstrap'])
        else:
            row['interval_note'] = 'Descriptive only: fewer than eight clusters, including empty strata.'
        output[label] = row
    return output


def trace_selection(arr):
    path = C.P.OUT / 'readout.json'
    frozen = C.read(path)['trace']
    order_by_id = np.argsort(arr['idx'])
    wanted = np.asarray(frozen['anonymous_idx_ordered'])
    positions = np.searchsorted(arr['idx'][order_by_id], wanted)
    assert (positions < len(order_by_id)).all()
    order = order_by_id[positions]
    assert np.array_equal(arr['idx'][order], wanted)
    assert np.array_equal(np.sort(order), np.flatnonzero(arr['discharge'] == frozen['anonymous_discharge']))
    assert len(order) == frozen['rows']
    assert np.array_equal(arr['time'][order], frozen['time_seconds_ordered'])
    assert np.array_equal(arr['segment'][order], frozen['segment_ordered'])
    return {**frozen, 'selection': 'Reuse the exact frozen B118 D10 discharge and all ordered rows; no new example choice',
            'B118_readout_sha256': C.sha(path), 'original_B118_selection': frozen['selection']}, order


def learning_summary(history, params, sd):
    assert len(history) == params['completed_epochs'] and len(history) > 0
    scores = np.asarray([e['seven_rule_mean_mse'] for e in history])
    best = int(np.argmin(scores))
    assert params['best_epoch'] == best + 1 and np.isclose(params['best_val_mse'], scores[best], rtol=1e-12)
    curve, cumulative = [], 0
    for epoch, e in enumerate(history, 1):
        assert e['epoch'] == epoch and e['train_count'] > 0
        cumulative += e['updates']
        assert set(e['val']) == set(C.TRAIN_RULES)
        assert np.isclose(e['seven_rule_mean_mse'], np.mean([e['val'][r]['mse'] for r in C.TRAIN_RULES]), rtol=1e-12)
        for v in e['val'].values():
            assert v['count'] > 0 and np.isclose(v['mse'], v['sse'] / v['count'], rtol=1e-12)
        train = float(e['train_normalized_sse'] / e['train_count'] * sd ** 2)
        curve.append({'epoch': epoch, 'train_normalized_mse': float(e['train_normalized_sse'] / e['train_count']),
                      'train_mse_eV2': train, 'val_mse_eV2': float(e['seven_rule_mean_mse']),
                      'val_minus_online_train_mse_eV2': float(e['seven_rule_mean_mse'] - train),
                      'train_count': e['train_count'], 'val_by_rule': {r: float(e['val'][r]['mse']) for r in C.TRAIN_RULES},
                      'lr_used': e['lr_used'], 'lr_next': e['lr_next'], 'updates': e['updates'],
                      'cumulative_updates': cumulative, 'elapsed_seconds': e['elapsed_seconds']})
    return {'completed_epochs': len(history), 'best_epoch': best + 1,
            'best_val_mse_eV2': float(scores[best]), 'best_val_rmse_eV': float(np.sqrt(scores[best])),
            'best_val_by_rule': curve[best]['val_by_rule'],
            'best_epoch_online_train_mse_eV2': curve[best]['train_mse_eV2'],
            'best_epoch_descriptive_gap_eV2': curve[best]['val_minus_online_train_mse_eV2'],
            'last_val_mse_eV2': float(scores[-1]), 'completed_updates': cumulative,
            'total_updates': params['total_updates'], 'discarded_partial_updates': params['discarded_partial_updates'],
            'actual_seconds': params['actual_seconds'], 'stop_reason': params['stop_reason'], 'curve': curve}


def render(report):
    lines = ['# B119: visible-CES anchor-output LSTM', '',
             'One registered fresh anchor-output LSTM fit, evaluated after its independent VAL gate; exploratory reused TEST.',
             'The direct-output comparator is the fully trained single B117 LSTM, evaluated and audited in B118.',
             'Both use the same recurrent width and input channels. This tests the registered output/initialization package, not an isolated causal mechanism.',
             'Frozen best epochs: anchor ' + str(report['fixed_checkpoint']['anchor_best_epoch'])
             + ', direct ' + str(report['fixed_checkpoint']['direct_best_epoch']) + '.',
             'All four original validation designs are listed; none is primary. Only the fixed first design was evaluated.', '',
             *[u + ': ' + v + '.' for u, v in report['validation_designs'].items()], '',
             'Macro MSE equally weights all eight scenario MSEs before computing skill = 1 - candidate/comparator.',
             'Macro draws use the same global group multiplicities across overlapping scenario supports.',
             'Pointwise 95% intervals condition on the fixed fits and selected checkpoints; no refitting, selection or repeated-exploration uncertainty is included.',
             'No requirement that all individual scenarios or strata have positive skill. No latency, physical-truth or causal failure-condition claim.', '',
             '| Scope | Model | MSE [95% cluster] | RMSE [95% cluster] | Bias |',
             '|---|---|---|---|---:|']
    for scope, result in {**report['averages'], **report['cells']}.items():
        for name, m in result['models'].items():
            lo, hi = m['cluster_MSE95']
            rlo, rhi = m['cluster_RMSE95']
            lines.append(f"| {scope} | {name} | {m['MSE']:.8g} [{lo:.8g}, {hi:.8g}] | {m['RMSE']:.8g} [{rlo:.8g}, {rhi:.8g}] | {m['bias']:+.8g} |")
    lines += ['', '| Scope | Paired MSE skill | Point [95% cluster] | Excludes zero |', '|---|---|---|---|']
    for scope, result in {**report['averages'], **report['cells']}.items():
        for pair, p in result['pairs'].items():
            lo, hi = p['cluster']['ci95']
            lines.append(f"| {scope} | {pair} | {p['point']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {p['cluster']['excludes_zero']} |")
    lines += ['', 'Separate registered statements: ' + str(report['statements']) + '.', '']
    for statement, scope, comparator in (('R1', 'overall_8rules', GP), ('R2', 'overall_8rules', SHORT),
                                          ('R3', 'D10 fastchange/low', SHORT)):
        result = report['averages'][scope] if statement != 'R3' else report['failure_strata'][report['unit'] + '/D10']['fastchange/low']
        p = result['pairs'].get(FOCUSED + '_over_' + comparator, {})
        if 'cluster' not in p:
            lines.append(f'{statement}: {scope}; descriptive only, fewer than eight clusters; registered increment not established.')
            continue
        lo, hi = p['cluster']['ci95']
        outcome = 'increment established conditional on these fixed fits' if report['statements'][statement] else 'specified increment not established'
        lines.append(f"{statement}: {report['unit']}, {scope}, anchor versus {comparator}: skill {p['point']:+.6f} [{lo:+.6f}, {hi:+.6f}] (95% cluster); {outcome}.")
    lines += ['',
              'Input-conditional strata retain frozen B116 TRAIN cuts, including available-CES-only age bins, history presence and elapsed-step bins.',
              'These are descriptive associations with observed inputs. No residual-based bins or shot selection were made.', '',
              '| Scenario | Stratum | Rows | Clusters | Anchor MSE | Direct MSE | GP+fast MSE | Anchor/GP skill [95% cluster] |',
              '|---|---|---:|---:|---:|---:|---:|---|']
    for key, strata in report['failure_strata'].items():
        for label, row in strata.items():
            if not row['rows']:
                lines.append(f'| {key} | {label} | 0 | 0 | | | | unavailable |')
                continue
            p = row['pairs'][FOCUSED + '_over_' + GP]
            interval = 'descriptive only'
            if row['interval_available']:
                lo, hi = p['cluster']['ci95']
                interval = f'[{lo:+.5f}, {hi:+.5f}]'
            m = row['models']
            lines.append(f"| {key} | {label} | {row['rows']} | {row['clusters']} | {m[FOCUSED]['MSE']:.8g} | {m[SHORT]['MSE']:.8g} | {m[GP]['MSE']:.8g} | {p['point']:+.5f} {interval} |")
    t = report['trace']
    lines += ['', f"Trace: D10 anonymous discharge {t['anonymous_discharge']}, all {t['rows']} supported observations.",
              t['selection'] + '. ' + t['order'] + '.',
              'The figure reuses the exact B118 example with measured CES observations and all three predictions on exactly these rows.',
              'JSON retains discharge intervals, stratum RMSE/bias/counts and all accepted/rejected bootstrap counts.',
              'An adequately optimized matched evaluation on unused shots could overturn an unfavorable result.']
    lines += ['', '## Learning curves', '',
              'All complete TRAIN/VAL epochs are shown for direct and anchor output, with learning rates and seven-rule errors at their own selected checkpoints.',
              'TRAIN physical MSE is normalized SSE/count times frozen TRAIN standard deviation squared. Online TRAIN uses changing rule supports and weights, so the TRAIN/VAL gap is descriptive.',
              'Operational VAL plateau criteria are not mathematical convergence or guaranteed optimality. The anchor-output hypothesis does not establish a causal explanation of failures.', '',
              '| Fit | Complete epochs | Best epoch | Best equal-rule VAL MSE | Best-epoch online TRAIN MSE | Completed updates |',
              '|---|---:|---:|---:|---:|---:|']
    for label, v in report['learning'].items():
        lines.append(f"| {label} | {v['completed_epochs']} | {v['best_epoch']} | {v['best_val_mse_eV2']:.8g} | {v['best_epoch_online_train_mse_eV2']:.8g} | {v['completed_updates']} |")
    return '\n'.join(lines) + '\n'


def figures(report, arrays, trace_order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = C.OUT / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    scopes = [('all eight scenarios', report['averages']['overall_8rules'])]
    scopes += [(r, report['cells'][C.UNIT + '/' + r]) for r in C.RULES]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 11), gridspec_kw={'width_ratios': [1.1, 1]})
    labels = []
    for i, (scope, stat) in enumerate(scopes):
        for j, name in enumerate(MODELS):
            y = i * 3 + j
            m = stat['models'][name]
            lo, hi = m['cluster_MSE95']
            left.hlines(y, lo, hi, color='C' + str(j))
            left.plot(m['MSE'], y, 'o', color='C' + str(j), ms=4)
            labels.append(scope + ' / ' + name)
        for j, (a, b) in enumerate(PAIRS):
            y = i * 3 + j
            p = stat['pairs'][a + '_over_' + b]
            lo, hi = p['cluster']['ci95']
            right.hlines(y, lo, hi, color='C' + str(j + 1))
            right.plot(p['point'], y, 'o', color='C' + str(j + 1), ms=4)
    left.set_yticks(range(len(labels)), labels, fontsize=8)
    left.set_xlabel('Physical MSE (eV²), pointwise 95% cluster interval')
    left.set_title('Fixed single-fit error')
    right.axvline(0, color='.4', lw=1)
    right.set_yticks([i * 3 + j for i in range(len(scopes)) for j in range(2)],
                     ['vs GP+fast', 'vs direct fit'] * len(scopes), fontsize=8)
    right.set_xlabel('1 - anchor MSE / comparator MSE')
    right.set_title('Paired MSE skill; no all-positive requirement')
    for ax in (left, right):
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('B119: U0 anchor-output LSTM, reused TEST, fit-conditional uncertainty')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(folder / 'b119_average.png', dpi=170)
    plt.close(fig)
    arr = arrays[C.UNIT + '/D10']
    x = arr['time'][trace_order]
    segment = arr['segment'][trace_order]
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, color in ((GP, 'C2'), (SHORT, 'C1'), (FOCUSED, 'C0')):
        for i, seg in enumerate(np.unique(segment)):
            keep = segment == seg
            ax.plot(x[keep], arr[name][trace_order][keep], label=name if i == 0 else None,
                    color=color, lw=1.3, marker='.', ms=2)
    ax.scatter(x, arr['truth'][trace_order], label='measured CES', color='black', s=10, zorder=5)
    ax.set_xlabel('Actual time (s); irregular CES observation times; segment gaps are not connected')
    ax.set_ylabel('Ion temperature (eV)')
    t = report['trace']
    ax.set_title(f"D10 anonymous discharge {t['anonymous_discharge']}; all {t['rows']} common rows\nExact frozen B118 example; no new trace choice")
    ax.legend(ncol=4)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / 'b119_trace.png', dpi=170)
    plt.close(fig)


def learning_figure(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for label, color in (('direct_B117', 'C1'), ('anchor_B119', 'C0')):
        stat = report['learning'][label]
        curve = stat['curve']
        epoch = [v['epoch'] for v in curve]
        for ax, key, title in ((axes[0, 0], 'train_mse_eV2', 'Online TRAIN sampled-rule mixture'),
                                (axes[0, 1], 'val_mse_eV2', 'Fixed-support equal-seven-rule VAL'),
                                (axes[1, 0], 'lr_used', 'Learning rate used during each epoch')):
            ax.plot(epoch, [v[key] for v in curve], label=label, color=color)
            ax.set_xlabel('Complete epoch')
            ax.set_title(title, fontsize=10)
        chosen = curve[stat['best_epoch'] - 1]
        axes[0, 1].plot(chosen['epoch'], chosen['val_mse_eV2'], 'o', color=color)
    axes[0, 0].set_ylabel('MSE (eV²), optimization in flight')
    axes[0, 1].set_ylabel('MSE (eV²)')
    axes[1, 0].set_ylabel('Learning rate')
    axes[1, 0].set_yscale('log')
    for ax in axes.flat[:3]:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    x = np.arange(len(C.TRAIN_RULES))
    for offset, label, color in ((-.2, 'direct_B117', 'C1'), (.2, 'anchor_B119', 'C0')):
        axes[1, 1].bar(x + offset, [report['learning'][label]['best_val_by_rule'][r] for r in C.TRAIN_RULES],
                       width=.4, color=color, label=label)
    axes[1, 1].set_xticks(x, C.TRAIN_RULES, rotation=30)
    axes[1, 1].set_title('Each fit at its own best equal-rule VAL checkpoint', fontsize=10)
    axes[1, 1].set_ylabel('VAL MSE (eV²)')
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('B119: registered anchor-output package versus trained direct LSTM\nTRAIN/VAL diagnostics; no matched-prefix loss requirement')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(C.OUT / 'figures/b119_learning_curves.png', dpi=170)
    plt.close(fig)


def main():
    sources = C.source_guard()
    inputs = C.verify_inputs()
    fit_complete = C.read(C.OUT / 'fit_complete.json')
    params = C.read(C.FIT / 'params.json')
    assert fit_complete['all_pass'] and fit_complete['converged'] and params['converged']
    assert fit_complete['sources'] == params['sources'] == sources
    assert fit_complete['params_sha256'] == C.sha(C.FIT / 'params.json')
    for rel, expected in fit_complete['files'].items():
        path = (C.OUT / rel).resolve()
        assert path.is_relative_to(C.OUT.resolve()) and C.sha(path) == expected
    val_audit_path = C.OUT / 'val_audit.json'
    val_audit = C.read(val_audit_path)
    assert val_audit['all_pass'] and val_audit['sources'] == sources
    assert val_audit['fit_complete_sha256'] == C.sha(C.OUT / 'fit_complete.json')
    complete = C.read(C.OUT / 'predictions_complete.json')
    assert complete['all_pass'] and complete['sources'] == sources and complete['inputs'] == inputs
    assert complete['scenarios'] == 8 and complete['optimization_steps'] == 0
    assert complete['model_state_unchanged'] and complete['checkpoint_before'] == complete['checkpoint_after']
    assert complete['val_audit_sha256'] == C.sha(val_audit_path)
    assert complete['fit_complete_sha256'] == C.sha(C.OUT / 'fit_complete.json')
    for rel, expected in complete['files'].items():
        path = (C.OUT / rel).resolve()
        assert path.is_relative_to(C.OUT.resolve()) and C.sha(path) == expected
    selection_path = C.ROOT / 'data/.b116/selection.json'
    cuts = C.read(selection_path)['input_cuts'][C.UNIT]
    arrays, cells, strata, identities = {}, {}, {}, {}
    for rule in C.RULES:
        key = C.UNIT + '/' + rule
        path = C.OUT / 'test' / f'{C.UNIT}_{rule}.npz'
        assert path.relative_to(C.OUT).as_posix() in complete['files']
        arr = C.npz(path)
        assert set(arr) == set(('idx', 'truth', 'cluster', 'discharge', 'time', 'segment', *MODELS))
        ref = D.common_row_support(C.UNIT, rule, 'test')
        for name in ('idx', 'truth', 'cluster', 'discharge', GP):
            assert np.array_equal(arr[name], ref[name]), (key, name)
        old = C.npz(C.P.OUT / 'test' / f'{C.UNIT}_{rule}.npz')
        assert np.array_equal(arr[SHORT], old['LSTM_focused'])
        coordinates = C.test_coordinates(arr['idx'], arr['discharge'])
        for name in ('time', 'segment'):
            assert np.array_equal(arr[name], coordinates[name])
        assert all(np.asarray(v).shape == arr['truth'].shape for v in arr.values())
        for idx, cluster, discharge in zip(arr['idx'].tolist(), arr['cluster'].tolist(), arr['discharge'].tolist()):
            assert idx not in identities or identities[idx] == (cluster, discharge)
            identities[idx] = (cluster, discharge)
        arrays[key] = arr
        cells[key] = statistics(arr, key)
        x = D.aligned_inputs(C.UNIT, rule, 'test')
        assert np.array_equal(x['idx'], arr['idx'])
        strata[key] = failure_statistics(arr, x['x'], cuts, key)
    averages = {'overall_8rules': average_statistics(arrays, 'average/overall_8rules'),
                'overall_D10': average_statistics({C.UNIT + '/D10': arrays[C.UNIT + '/D10']}, 'average/overall_D10')}
    trace, order = trace_selection(arrays[C.UNIT + '/D10'])
    direct_params = C.read(C.P.FIT / 'params.json')
    history = C.read(C.FIT / 'history.json')
    direct_report = C.read(C.P.P.OUT / 'readout.json')
    learning = {'direct_B117': direct_report['focused'],
                'anchor_B119': learning_summary(history, params,
                                                float(D.load_prepared(C.UNIT)['identity']['stats']['ti_sd']))}
    low_fast = strata[C.UNIT + '/D10']['fastchange/low']
    low_pair = low_fast['pairs'].get(FOCUSED + '_over_' + SHORT, {})
    report = {'batch': 'B119', 'created_unix': time.time(), 'sources': sources, 'inputs': inputs,
              'new_fits': 1, 'architecture_searches': 0, 'unit': C.UNIT,
              'fixed_checkpoint': {'architecture': C.ARCH, 'seed': C.SEED,
                                   'anchor_best_epoch': params['best_epoch'],
                                   'direct_best_epoch': direct_params['best_epoch'],
                                   'anchor_best_sha256': C.sha(C.FIT / 'best.npz')},
              'validation_designs': {u: ('EVALUATED: one fixed single fit' if u == C.UNIT else 'NOT RUN') for u in D.UNITS},
              'primary_validation_design': None,
              'interval_scope': 'fixed fits and selected checkpoints; exploratory reused TEST; no simultaneous coverage',
              'averaging': 'equal eight-scenario physical MSE before ratio; shared global group multiplicities',
              'input_cuts': cuts, 'input_cuts_source_sha256': C.sha(selection_path),
              'global_group_contract': {'overlapping_row_identities_consistent': True, 'unique_rows': len(identities)},
              'cells': cells, 'averages': averages, 'failure_strata': strata, 'trace': trace,
              'learning': learning,
              'learning_sources': {'anchor_history_sha256': C.sha(C.FIT / 'history.json'),
                                   'anchor_params_sha256': C.sha(C.FIT / 'params.json'),
                                   'direct_B117_readout_sha256': C.sha(C.P.P.OUT / 'readout.json')},
              'val_audit_sha256': C.sha(val_audit_path),
              'fit_complete_sha256': C.sha(C.OUT / 'fit_complete.json'),
              'predictions_complete_sha256': C.sha(C.OUT / 'predictions_complete.json'),
              'statements': {'R1': averages['overall_8rules']['pairs'][FOCUSED + '_over_' + GP]['cluster']['ci95'][0] > 0,
                             'R2': averages['overall_8rules']['pairs'][FOCUSED + '_over_' + SHORT]['cluster']['ci95'][0] > 0,
                             'R3': bool(low_fast['interval_available'] and low_pair.get('cluster', {}).get('ci95', [-np.inf])[0] > 0)},
              'statement_interval_available': {'R1': True, 'R2': True, 'R3': low_fast['interval_available']},
              'statement_definitions': {'R1': 'all-eight macro anchor/GP+fast cluster lower95 > 0',
                                        'R2': 'all-eight macro anchor/direct LSTM cluster lower95 > 0',
                                        'R3': 'D10 frozen fastchange/low anchor/direct LSTM cluster lower95 > 0'}}
    C.save(C.OUT / 'readout.json', report)
    (C.OUT / 'readout.md').write_text(render(report), encoding='utf8')
    figures(report, arrays, order)
    learning_figure(report)
    paths = ('readout.json', 'readout.md', 'figures/b119_average.png', 'figures/b119_trace.png',
             'figures/b119_learning_curves.png')
    C.save(C.OUT / 'readout_complete.json', {'all_pass': True, 'published': False,
           'predictions_complete_sha256': report['predictions_complete_sha256'],
           'files': {p: C.sha(C.OUT / p) for p in paths}})
    print('B119 registered anchor-output readout staged; publication awaits independent audit')


if __name__ == '__main__':
    main()
