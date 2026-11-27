"""TRAIN/VAL learning-curve analysis of one fixed LSTM; never accesses TEST."""
import time
import numpy as np
import common117 as C
import data116 as D


def summarize(history, params, sd):
    """Pure arithmetic over logged optimization and validation statistics."""
    assert len(history) == params['completed_epochs'] and len(history) > 0
    assert [v['epoch'] for v in history] == list(range(1, len(history) + 1))
    score = np.array([v['seven_rule_mean_mse'] for v in history], dtype=np.float64)
    assert np.isfinite(score).all() and np.all(score > 0)
    best_index = int(np.argmin(score))
    assert params['best_epoch'] == best_index + 1
    assert np.isclose(params['best_val_mse'], score[best_index], rtol=1e-12, atol=1e-8)
    curve = []
    cumulative_updates = 0
    for entry in history:
        assert entry['train_count'] > 0
        train_mse = float(entry['train_normalized_sse'] / entry['train_count'] * sd ** 2)
        assert np.isfinite(train_mse) and train_mse >= 0
        assert set(entry['val']) == set(C.TRAIN_RULES)
        for rule, val in entry['val'].items():
            assert val['count'] > 0 and np.isclose(val['mse'], val['sse'] / val['count'], rtol=1e-12)
        assert np.isclose(entry['seven_rule_mean_mse'],
                          np.mean([entry['val'][r]['mse'] for r in C.TRAIN_RULES]), rtol=1e-12)
        cumulative_updates += int(entry['updates'])
        curve.append({'epoch': entry['epoch'], 'train_mse_eV2': train_mse,
                      'train_normalized_mse': float(entry['train_normalized_sse'] / entry['train_count']),
                      'train_count': entry['train_count'],
                      'val_mse_eV2': float(entry['seven_rule_mean_mse']),
                      'val_minus_online_train_mse_eV2': float(entry['seven_rule_mean_mse'] - train_mse),
                      'val_by_rule': {r: float(entry['val'][r]['mse']) for r in C.TRAIN_RULES},
                      'lr_used': entry['lr_used'], 'lr_next': entry['lr_next'],
                      'updates': entry['updates'], 'cumulative_updates': cumulative_updates,
                      'elapsed_seconds': entry['elapsed_seconds']})
    best = curve[best_index]
    return {'completed_epochs': len(curve), 'best_epoch': best_index + 1,
            'best_val_mse_eV2': float(score[best_index]),
            'best_val_rmse_eV': float(np.sqrt(score[best_index])),
            'best_epoch_online_train_mse_eV2': best['train_mse_eV2'],
            'best_epoch_descriptive_gap_eV2': best['val_minus_online_train_mse_eV2'],
            'best_val_by_rule': best['val_by_rule'], 'last_val_mse_eV2': float(score[-1]),
            'completed_updates': cumulative_updates,
            'total_updates': params['total_updates'],
            'discarded_partial_updates': params['discarded_partial_updates'],
            'actual_seconds': params['actual_seconds'], 'stop_reason': params['stop_reason'],
            'curve': curve}


def build_report(old_history, old_params, history, params, sd):
    baseline = summarize(old_history, old_params, sd)
    focused = summarize(history, params, sd)
    converged = bool(params['converged'])
    rules = {}
    for rule in C.TRAIN_RULES:
        old, new = baseline['best_val_by_rule'][rule], focused['best_val_by_rule'][rule]
        rules[rule] = {'baseline_mse_eV2': old, 'focused_mse_eV2': new,
                       'change_eV2': new - old, 'relative_reduction': 1 - new / old}
    old = baseline['best_val_mse_eV2']
    new = focused['best_val_mse_eV2']
    return {'batch': 'B117', 'unit': C.UNIT, 'architecture': 'lstm', 'seed': C.SEED,
            'TEST_accessed': False, 'fit_count': 1, 'converged': converged,
            'validation_designs': {u: ('RUN: TRAIN/VAL optimization pilot only' if u == C.UNIT else 'NOT RUN')
                                   for u in C.B.UNITS},
            'primary_validation_design': None,
            'training_status': 'registered convergence criterion met' if converged else 'NOT CONVERGED: safety boundary reached',
            'baseline': baseline, 'focused': focused,
            'comparison': {'best_val_change_eV2': new - old,
                           'best_val_relative_reduction': 1 - new / old,
                           'best_val_by_rule': rules,
                           'completed_epoch_change': focused['completed_epochs'] - baseline['completed_epochs'],
                           'best_epoch_relation': ('same epoch' if focused['best_epoch'] == baseline['best_epoch']
                                                   else 'later' if focused['best_epoch'] > baseline['best_epoch'] else 'earlier')},
            'interpretation': [
                'One fixed unit/architecture/seed pilot, freshly initialized; no TEST evaluation or GP comparison.',
                'The checkpoint and convergence decision use VAL repeatedly; these are optimization diagnostics.',
                'Online TRAIN loss uses changing sampled-rule supports and changing parameters within each epoch.',
                'TRAIN MSE is normalized SSE/count multiplied by the frozen TRAIN Ti standard deviation squared.',
                'VAL is an equal-seven-rule mean on fixed common VAL supports; the TRAIN/VAL gap is descriptive.',
                'The registered stop is an operational VAL plateau criterion, not mathematical convergence or guaranteed optimality.',
                'No confidence interval or independent-sample generalization conclusion is produced.',
                'A safety stop is not successful convergence, even when the best VAL error decreases.']}


def render(report):
    lines = ['# B117: one LSTM training diagnostic', '', report['training_status'], '',
             'Unit: ' + report['unit'] + '; architecture: lstm; seed: ' + str(report['seed']) + '.', '',
             'All four original validation designs are accounted for below; none is primary.', '',
             *[u + ': ' + status + '.' for u, status in report['validation_designs'].items()], '',
             '| Quantity | Original B116 fit | Focused fit |', '|---|---:|---:|']
    for name, key in [('Completed epochs', 'completed_epochs'), ('Best epoch', 'best_epoch'),
                      ('Completed updates', 'completed_updates'), ('Total updates including partial epoch', 'total_updates'),
                      ('Partial updates outside complete history', 'discarded_partial_updates'),
                      ('Elapsed seconds', 'actual_seconds'), ('Best equal-rule VAL MSE (eV²)', 'best_val_mse_eV2'),
                      ('Best VAL RMSE (eV)', 'best_val_rmse_eV'),
                      ('Online TRAIN MSE at best VAL epoch (eV²)', 'best_epoch_online_train_mse_eV2'),
                      ('Descriptive VAL minus online TRAIN gap (eV²)', 'best_epoch_descriptive_gap_eV2'),
                      ('Final complete-epoch VAL MSE (eV²)', 'last_val_mse_eV2'), ('Stop reason', 'stop_reason')]:
        values = [report[n][key] for n in ('baseline', 'focused')]
        rendered = [format(v, '.8g') if isinstance(v, float) else str(v) for v in values]
        lines.append('| ' + name + ' | ' + ' | '.join(rendered) + ' |')
    c = report['comparison']
    lines += ['', f"Best equal-rule VAL MSE change: {c['best_val_change_eV2']:+.8g} eV²; relative reduction: {c['best_val_relative_reduction']:+.6%}.", '',
              'Selected best epoch relative to original fit: ' + c['best_epoch_relation'] + '.', '',
              'Selected weight arrays exactly identical to the original checkpoint: '
              + str(c.get('selected_weights_identical_to_baseline', 'not yet checked')) + '.', '',
              '| VAL rule | Original best-checkpoint MSE | Focused best-checkpoint MSE | Change (eV²) | Relative reduction |',
              '|---|---:|---:|---:|---:|']
    for rule, row in c['best_val_by_rule'].items():
        lines.append(f"| {rule} | {row['baseline_mse_eV2']:.8g} | {row['focused_mse_eV2']:.8g} | {row['change_eV2']:+.8g} | {row['relative_reduction']:+.6%} |")
    lines += ['', *report['interpretation'], '',
              'The latest validated recovery checkpoint retains optimizer, scheduler, RNG and next-epoch state; it is distinct from the selected best weights.',
              'Recovery hashes and the last validated epoch are recorded in the JSON. No automatic restart is authorized.',
              'The figure shows all logged complete epochs. The vertical marker indicates the original fit epoch boundary.',
              'Full curves, all epoch learning rates and cumulative updates are retained in the JSON.']
    return '\n'.join(lines) + '\n'


def figure(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for label, color in (('baseline', 'C1'), ('focused', 'C0')):
        result = report[label]
        curve = result['curve']
        epoch = [v['epoch'] for v in curve]
        for ax, key, title in ((axes[0, 0], 'train_mse_eV2', 'Online TRAIN loss (sampled rule mixture)'),
                                (axes[0, 1], 'val_mse_eV2', 'Fixed-support equal-seven-rule VAL loss'),
                                (axes[1, 0], 'lr_used', 'Learning rate used for each complete epoch')):
            ax.plot(epoch, [v[key] for v in curve], label=label, color=color, lw=1.5)
            ax.axvline(report['baseline']['completed_epochs'], ls=':', color='.4', lw=1)
            ax.set_xlabel('Complete epoch')
            ax.set_title(title, fontsize=10)
        b = curve[result['best_epoch'] - 1]
        axes[0, 1].plot(b['epoch'], b['val_mse_eV2'], 'o', color=color, ms=5)
    axes[0, 0].set_ylabel('MSE (eV²), training updates in flight')
    axes[0, 1].set_ylabel('MSE (eV²)')
    axes[1, 0].set_ylabel('Learning rate')
    axes[1, 0].set_yscale('log')
    for ax in axes.flat[:3]:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    x = np.arange(len(C.TRAIN_RULES))
    for shift, label, color in ((-.2, 'baseline', 'C1'), (.2, 'focused', 'C0')):
        axes[1, 1].bar(x + shift, [report[label]['best_val_by_rule'][r] for r in C.TRAIN_RULES],
                       width=.4, label=label, color=color)
    axes[1, 1].set_xticks(x, C.TRAIN_RULES, rotation=30)
    axes[1, 1].set_ylabel('VAL MSE (eV²)')
    axes[1, 1].set_title('Each fit at its own best equal-rule VAL checkpoint', fontsize=10)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('One fixed LSTM: TRAIN/VAL diagnostic only\n' + report['training_status'], fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    path = C.OUT / 'figures/b117_learning_curves.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    sources = C.source_guard()
    inputs = C.verify_inputs()
    complete = C.read(C.OUT / 'fit_complete.json')
    assert complete['all_pass'] and complete['sources'] == sources and complete['TEST_accessed'] is False
    params = C.read(C.FIT / 'params.json')
    history = C.read(C.FIT / 'history.json')
    assert C.sha(C.FIT / 'params.json') == complete['params_sha256']
    assert params['sources'] == sources
    assert params['inputs'] == inputs
    baseline = C.ROOT / 'data/.b116/fits' / C.UNIT / 'lstm' / ('s' + str(C.SEED))
    old_params = C.read(baseline / 'params.json')
    old_history = C.read(baseline / 'history.json')
    for rel, expected in params['files'].items():
        path = (C.FIT / rel).resolve()
        assert path.is_relative_to(C.FIT.resolve()) and C.sha(path) == expected
    assert C.sha(baseline / 'history.json') == old_params['history_sha256']
    assert C.sha(baseline / 'best.npz') == old_params['best_sha256']
    assert (params['unit'], params['architecture'], params['seed']) == (C.UNIT, 'lstm', C.SEED)
    # Loading the frozen prepared TRAIN/VAL cache never constructs a TEST segment.
    prepared = D.load_prepared(C.UNIT)
    report = build_report(old_history, old_params, history, params,
                          float(prepared['identity']['stats']['ti_sd']))
    original_weights, focused_weights = C.npz(baseline / 'best.npz'), C.npz(C.FIT / 'best.npz')
    report['comparison']['selected_weights_identical_to_baseline'] = (
        original_weights.keys() == focused_weights.keys()
        and all(np.array_equal(original_weights[k], focused_weights[k]) for k in original_weights))
    recovery = C.read(C.FIT / 'recovery.json')
    assert recovery['validated'] and recovery['epoch'] == params['completed_epochs']
    assert recovery['sources'] == sources
    for name, key in (('latest.pt', 'latest_sha256'), ('best.npz', 'best_sha256'),
                       ('history.json', 'history_sha256')):
        assert C.sha(C.FIT / name) == recovery[key]
    report['recovery_checkpoint'] = {**recovery, 'automatic_restart_authorized': False}
    pointcheck = {}
    for rule in C.TRAIN_RULES:
        rel = f'val/{C.UNIT}_{rule}.npz'
        assert C.sha(C.OUT / rel) == complete['files'][rel]
        new_val = C.npz(C.OUT / rel)
        old_path = C.ROOT / 'data/.b116' / rel
        old_val = C.npz(old_path)
        ref = D.common_row_support(C.UNIT, rule, split='val')
        for key in ('idx', 'truth', 'cluster', 'discharge'):
            assert np.array_equal(ref[key], new_val[key])
            assert np.array_equal(ref[key], old_val[key])
        new_mse = float(np.mean((np.asarray(new_val['prediction'], dtype=np.float64) - ref['truth']) ** 2))
        old_mse = float(np.mean((np.asarray(old_val['Seq_lstm_s1701'], dtype=np.float64) - ref['truth']) ** 2))
        assert np.isclose(new_mse, report['focused']['best_val_by_rule'][rule], rtol=1e-8, atol=.01)
        assert np.isclose(old_mse, report['baseline']['best_val_by_rule'][rule], rtol=1e-8, atol=.01)
        pointcheck[rule] = {'focused_export_mse_eV2': new_mse, 'baseline_export_mse_eV2': old_mse,
                             'focused_export_sha256': C.sha(C.OUT / rel),
                             'baseline_export_sha256': C.sha(old_path)}
    report.update(sources=sources, inputs=inputs, created_unix=time.time(), validation_export_check=pointcheck,
                  fit_complete_sha256=C.sha(C.OUT / 'fit_complete.json'),
                  provenance={str((folder / f).relative_to(C.ROOT).as_posix()): C.sha(folder / f)
                              for folder in (C.FIT, baseline) for f in ('params.json', 'history.json', 'best.npz')})
    C.save(C.OUT / 'readout.json', report)
    (C.OUT / 'readout.md').write_text(render(report), encoding='utf8')
    figure(report)
    files = ('readout.json', 'readout.md', 'figures/b117_learning_curves.png')
    C.save(C.OUT / 'readout_complete.json', {'all_pass': True, 'published': False,
           'fit_complete_sha256': report['fit_complete_sha256'],
           'files': {rel: C.sha(C.OUT / rel) for rel in files}})
    print('B117 TRAIN/VAL diagnostic staged; no TEST access')


if __name__ == '__main__':
    main()
