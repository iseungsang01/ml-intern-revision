"""Fixed B117 checkpoint evaluation only; no fitting, selection or source edits."""
import time
import zlib
import numpy as np
import common118 as C
import data116 as D
from select116 import strata_masks

FOCUSED, SHORT, GP = 'LSTM_focused', 'LSTM_short', 'GP_fast_boost'
MODELS = (FOCUSED, SHORT, GP)
PAIRS = ((FOCUSED, GP), (FOCUSED, SHORT))
NBOOT = 20000


def average_statistics(cells, key, models=MODELS, pairs=PAIRS, groupings=('cluster', 'discharge')):
    """Macro MSE before the ratio; synchronized union-group draws across cells.

    Arithmetic follows frozen B116, with the prospectively fixed B118 seed.
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
        seed = [20261128, zlib.crc32((key + '/' + grouping).encode())]
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
    labels, counts = np.unique(arr['discharge'], return_counts=True)
    choice = min(zip(labels.tolist(), counts.tolist()), key=lambda x: (abs(x[1] - 150), x[0]))
    selected = np.flatnonzero(arr['discharge'] == choice[0])
    order = selected[np.lexsort((arr['time'][selected], arr['segment'][selected]))]
    for segment in np.unique(arr['segment'][order]):
        times = arr['time'][order][arr['segment'][order] == segment]
        assert np.isfinite(times).all() and np.all(np.diff(times) >= 0)
    return {'rule': 'D10', 'anonymous_discharge': choice[0], 'rows': len(order),
            'selection': 'discharge support count closest to 150 D10 rows; lowest anonymous code breaks ties',
            'support_counts': [{'anonymous_discharge': d, 'rows': n} for d, n in zip(labels.tolist(), counts.tolist())],
            'order': 'segments ordered by their earliest physical time, then increasing actual time within segment; lines never connect segments',
            'anonymous_idx_ordered': arr['idx'][order].tolist(),
            'time_seconds_ordered': arr['time'][order].tolist(),
            'segment_ordered': arr['segment'][order].tolist()}, order


def render(report):
    lines = ['# B118: fixed focused LSTM evaluation', '',
             'Exploratory reused-TEST evaluation of one frozen B117 checkpoint; no new fit or model choice.',
             'The original short fit comparator is the same single B116 LSTM seed, not an ensemble.',
             'Frozen best epochs: focused ' + str(report['fixed_checkpoint']['focused_best_epoch'])
             + ', short fit ' + str(report['fixed_checkpoint']['short_best_epoch']) + '.',
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
    for statement, scope, comparator in (('R1', 'overall_8rules', GP), ('R2', 'overall_D10', GP),
                                          ('R3', 'overall_8rules', SHORT)):
        p = report['averages'][scope]['pairs'][FOCUSED + '_over_' + comparator]
        lo, hi = p['cluster']['ci95']
        outcome = 'increment established conditional on these fixed fits' if report['statements'][statement] else 'specified increment not established'
        lines.append(f"{statement}: {report['unit']}, {scope}, focused versus {comparator}: skill {p['point']:+.6f} [{lo:+.6f}, {hi:+.6f}] (95% cluster); {outcome}.")
    lines += ['',
              'Input-conditional strata retain frozen B116 TRAIN cuts, including available-CES-only age bins, history presence and elapsed-step bins.',
              'These are descriptive associations with observed inputs. No residual-based bins or shot selection were made.', '',
              '| Scenario | Stratum | Rows | Clusters | Focused MSE | Short MSE | GP+fast MSE | Focused/GP skill [95% cluster] |',
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
              'The figure plots the measured CES observations and all three predictions on exactly these rows.',
              'JSON retains discharge intervals, stratum RMSE/bias/counts and all accepted/rejected bootstrap counts.',
              'An adequately optimized matched evaluation on unused shots could overturn an unfavorable result.']
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
                     ['vs GP+fast', 'vs short fit'] * len(scopes), fontsize=8)
    right.set_xlabel('1 - focused MSE / comparator MSE')
    right.set_title('Paired MSE skill; no all-positive requirement')
    for ax in (left, right):
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('B118: fixed U0 LSTM, reused TEST, fit-conditional uncertainty')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(folder / 'b118_average.png', dpi=170)
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
    ax.set_title(f"D10 anonymous discharge {t['anonymous_discharge']}; all {t['rows']} common rows\nSupport-only choice: count closest to 150, anonymous-code tie break")
    ax.legend(ncol=4)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / 'b118_trace.png', dpi=170)
    plt.close(fig)


def main():
    sources = C.source_guard()
    inputs = C.verify_inputs()
    complete = C.read(C.OUT / 'predictions_complete.json')
    assert complete['all_pass'] and complete['sources'] == sources and complete['inputs'] == inputs
    assert complete['scenarios'] == 8 and complete['optimization_steps'] == 0
    assert complete['model_state_unchanged'] and complete['checkpoint_before'] == complete['checkpoint_after']
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
        old = C.npz(C.ROOT / 'data/.b116/test' / f'{C.UNIT}_{rule}.npz')
        assert np.array_equal(arr[SHORT], old['Seq_lstm_s1701'])
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
    focused_params = C.read(C.FIT / 'params.json')
    short_params = C.read(C.ROOT / 'data/.b116/fits' / C.UNIT / 'lstm' / ('s' + str(C.SEED)) / 'params.json')
    report = {'batch': 'B118', 'created_unix': time.time(), 'sources': sources, 'inputs': inputs,
              'new_fits': 0, 'model_choices': 0, 'unit': C.UNIT,
              'fixed_checkpoint': {'architecture': C.ARCH, 'seed': C.SEED,
                                   'focused_best_epoch': focused_params['best_epoch'],
                                   'short_best_epoch': short_params['best_epoch'],
                                   'focused_best_sha256': C.sha(C.FIT / 'best.npz')},
              'validation_designs': {u: ('EVALUATED: one fixed single fit' if u == C.UNIT else 'NOT RUN') for u in D.UNITS},
              'primary_validation_design': None,
              'interval_scope': 'fixed fits and selected checkpoints; exploratory reused TEST; no simultaneous coverage',
              'averaging': 'equal eight-scenario physical MSE before ratio; shared global group multiplicities',
              'input_cuts': cuts, 'input_cuts_source_sha256': C.sha(selection_path),
              'global_group_contract': {'overlapping_row_identities_consistent': True, 'unique_rows': len(identities)},
              'cells': cells, 'averages': averages, 'failure_strata': strata, 'trace': trace,
              'predictions_complete_sha256': C.sha(C.OUT / 'predictions_complete.json'),
              'statements': {'R1': averages['overall_8rules']['pairs'][FOCUSED + '_over_' + GP]['cluster']['ci95'][0] > 0,
                             'R2': averages['overall_D10']['pairs'][FOCUSED + '_over_' + GP]['cluster']['ci95'][0] > 0,
                             'R3': averages['overall_8rules']['pairs'][FOCUSED + '_over_' + SHORT]['cluster']['ci95'][0] > 0},
              'statement_definitions': {'R1': 'all-eight macro focused/GP+fast cluster lower95 > 0',
                                        'R2': 'D10 focused/GP+fast cluster lower95 > 0',
                                        'R3': 'all-eight macro focused/short LSTM cluster lower95 > 0'}}
    C.save(C.OUT / 'readout.json', report)
    (C.OUT / 'readout.md').write_text(render(report), encoding='utf8')
    figures(report, arrays, order)
    paths = ('readout.json', 'readout.md', 'figures/b118_average.png', 'figures/b118_trace.png')
    C.save(C.OUT / 'readout_complete.json', {'all_pass': True, 'published': False,
           'predictions_complete_sha256': report['predictions_complete_sha256'],
           'files': {p: C.sha(C.OUT / p) for p in paths}})
    print('B118 fixed-fit readout staged; publication awaits independent audit')


if __name__ == '__main__':
    main()
