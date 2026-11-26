"""Fixed-fit exploratory temporal comparisons; synchronized global-group intervals."""
import time
import zlib
import numpy as np
import common116 as C
import data116 as D
from select116 import ARCHES, features, strata_masks

ENSEMBLES = tuple(f'Seq_{a}_ensemble' for a in ARCHES)
SINGLES = tuple(f'Seq_{a}_s{s}' for a in ARCHES for s in C.SEEDS)
SELECTED = 'Seq_selected_ensemble'
GP = 'GP_fast_boost'
MODELS = ENSEMBLES + SINGLES + (SELECTED, GP)
PAIRS = tuple((a, GP) for a in ENSEMBLES + SINGLES + (SELECTED,)) + ((SELECTED, 'Seq_lstm_ensemble'),)
NBOOT = 20000


def average_statistics(cells, key, models=MODELS, pairs=PAIRS, groupings=('cluster', 'discharge')):
    """Equal cell MSE means; same global cluster weights in all overlapping cells.

    Every invalid zero-support draw is rejected for the entire scope, explicitly
    counted, and replaced. No cell or scenario is silently omitted from a draw.
    """
    assert cells
    arrays = list(cells.values())
    errors = [np.stack([np.asarray(a[n], dtype=np.float64) - a['truth'] for n in models], axis=1)
              for a in arrays]
    squared = [e ** 2 for e in errors]
    assert all(np.isfinite(s).all() and len(s) for s in squared)
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
        rng = np.random.default_rng([20261126, zlib.crc32((key + '/' + grouping).encode())])
        draws, accepted, rejected, generated = [], 0, 0, 0
        while accepted < NBOOT:
            size = min(500, NBOOT - accepted)
            weight = rng.multinomial(ng, np.full(ng, 1 / ng), size=size)
            support = np.stack([weight @ count for count in counts], axis=1)
            valid = np.all(support > 0, axis=1)
            rejected += int((~valid).sum())
            generated += size
            assert generated < 100 * NBOOT, 'Bootstrap scope has insufficient shared support'
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
        result['bootstrap'][grouping] = {'groups': ng, 'accepted': accepted,
                                          'generated': generated, 'zero_support_redraws': rejected,
                                          'seed': [20261126, zlib.crc32((key + '/' + grouping).encode())]}
    return result


def statistics(arr, key):
    result = average_statistics({key: arr}, key)
    result.update(rows=len(arr['truth']), clusters=len(np.unique(arr['cluster'])),
                  discharges=len(np.unique(arr['discharge'])))
    return result


def failure_statistics(arr, x, cuts, key):
    output = {}
    for label, mask in strata_masks(x, cuts).items():
        n = int(mask.sum())
        sub = {k: np.asarray(v)[mask] for k, v in arr.items() if np.asarray(v).ndim == 1 and len(v) == len(mask)}
        clusters = len(np.unique(sub['cluster']))
        row = {'rows': n, 'clusters': clusters, 'discharges': len(np.unique(sub['discharge'])),
               'interval_available': clusters >= 8,
               'models': {}, 'pairs': {}}
        if n:
            for name in (SELECTED, GP):
                e = sub[name] - sub['truth']
                row['models'][name] = {'MSE': float(np.mean(e ** 2)),
                                       'RMSE': float(np.sqrt(np.mean(e ** 2))), 'bias': float(np.mean(e))}
            row['pairs'][SELECTED + '_over_' + GP] = {'point': 1 - row['models'][SELECTED]['MSE'] / row['models'][GP]['MSE']}
        if clusters >= 8:
            full = average_statistics({key: sub}, key + '/' + label, (SELECTED, GP),
                                      ((SELECTED, GP),), ('cluster',))
            row.update(models=full['models'], pairs=full['pairs'], bootstrap=full['bootstrap'])
        else:
            row['interval_note'] = 'Descriptive only: fewer than eight clusters (including empty strata).'
        output[label] = row
    return output


def render(report):
    lines = ['# B116 temporal architecture comparison', '',
             'Exploratory reused-TEST comparison; all four validation designs, none primary.',
             'MSE is in physical target units squared. Skill = 1 - candidate MSE / GP+fast MSE.',
             'Average criteria use equal cell MSE means; synchronized global group draws preserve overlap.',
             'Pointwise 95% intervals condition on trained fits and frozen VAL selection; they omit refitting and selection uncertainty.',
             'No requirement that every interval be positive. No architecture-only causal or measured 10 ms latency claim.', '',
             '| Scope | Model | MSE [95% cluster] | RMSE | Skill vs GP+fast [95% cluster] |',
             '|---|---|---|---|---|']
    for scope, cell in {**report['averages'], **report['cells']}.items():
        for n in ENSEMBLES + (SELECTED, GP):
            m = cell['models'][n]
            lo, hi = m['cluster_MSE95']
            contrast = cell['pairs'].get(n + '_over_' + GP)
            skill = ''
            if contrast:
                a, b = contrast['cluster']['ci95']
                skill = f"{contrast['point']:+.4f} [{a:+.4f}, {b:+.4f}]"
            lines.append(f"| {scope} | {n} | {m['MSE']:.6g} [{lo:.6g}, {hi:.6g}] | {m['RMSE']:.6g} | {skill} |")
    lines += ['', 'Registered average statements: ' + str(report['statements']), '',
              'Input strata are frozen TRAIN terciles. Age cuts/bins use available CES only; absent CES is separate. Cuts never use errors.',
              'Sparse strata remain in the JSON with counts; fewer than eight clusters receives no interval.', '',
              '| Unit/rule | Input stratum | Rows | Clusters | Selected MSE | GP+fast MSE | Skill [95% cluster] |',
              '|---|---|---:|---:|---:|---:|---|']
    for key, strata in report['failure_strata'].items():
        for label, row in strata.items():
            if not row['rows']:
                lines.append(f"| {key} | {label} | 0 | 0 | | | unavailable |")
                continue
            p = row['pairs'][SELECTED + '_over_' + GP]
            interval = 'descriptive'
            if row['interval_available']:
                lo, hi = p['cluster']['ci95']
                interval = f'[{lo:+.4f}, {hi:+.4f}]'
            lines.append(f"| {key} | {label} | {row['rows']} | {row['clusters']} | {row['models'][SELECTED]['MSE']:.6g} | {row['models'][GP]['MSE']:.6g} | {p['point']:+.4f} {interval} |")
    lines += ['', 'Seed results, discharge intervals, per-stratum bias and all bootstrap redraw counts are in JSON.',
              'A matched evaluation on unused shots could overturn an unfavorable average or conditional result.']
    return '\n'.join(lines) + '\n'


def figures(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = C.OUT / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    rows = [(scope, n, data['models'][n]) for scope, data in report['averages'].items()
            for n in ENSEMBLES + (SELECTED, GP)]
    fig, ax = plt.subplots(figsize=(11, 12))
    for i, (scope, n, m) in enumerate(rows):
        lo, hi = m['cluster_MSE95']
        ax.hlines(i, lo, hi, color='C0' if n == SELECTED else '.65')
        ax.plot(m['MSE'], i, 'o', ms=3, color='C0' if n == SELECTED else '.3')
    ax.set_yticks(range(len(rows)), [s + ' / ' + n.replace('Seq_', '').replace('_ensemble', '') for s, n, _ in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Equal-cell physical MSE; pointwise 95% global-cluster interval')
    ax.set_title('B116 average error (all four designs shown)')
    fig.tight_layout()
    fig.savefig(folder / 'b116_average.png', dpi=170)
    plt.close(fig)
    labels = list(next(iter(report['failure_strata'].values())))
    keys = [u + '/' + r for u in C.UNITS for r in ('D10', 'S')]
    matrix = np.full((len(labels), len(keys)), np.nan)
    for j, key in enumerate(keys):
        for i, label in enumerate(labels):
            p = report['failure_strata'][key][label]['pairs'].get(SELECTED + '_over_' + GP)
            if p:
                matrix[i, j] = p['point']
    fig, ax = plt.subplots(figsize=(11, 7))
    limit = max(.1, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    for i in range(len(labels)):
        for j in range(len(keys)):
            row = report['failure_strata'][keys[j]][labels[i]]
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:+.2f}\nn={row['rows']}" + ('' if row['interval_available'] else '*'),
                        ha='center', va='center', fontsize=7)
    ax.set_xticks(range(len(keys)), keys, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title('Input-conditional selected/GP+fast skill: D10 and S\n* fewer than eight clusters; descriptive only')
    fig.colorbar(im, ax=ax, label='1 - selected MSE / GP+fast MSE')
    fig.tight_layout()
    fig.savefig(folder / 'b116_failure.png', dpi=170)
    plt.close(fig)


def main():
    sources = C.source_guard()
    C.verify_inputs()
    fit = C.read(C.OUT / 'fit_complete.json')
    complete = C.read(C.OUT / 'predictions_complete.json')
    selection = C.read(C.OUT / 'selection.json')
    assert fit['all_pass'] and len(fit['fits']) == 36 and complete['all_pass']
    assert fit['sources'] == complete['sources'] == selection['sources'] == sources
    assert complete['fit_complete_sha256'] == C.sha(C.OUT / 'fit_complete.json')
    for receipt in fit['fits']:
        assert receipt['normal'] and C.sha(C.OUT / receipt['path']) == receipt['sha256']
    for rel, digest in {**fit['files'], **complete['files']}.items():
        assert C.sha(C.OUT / rel) == digest
    assert complete['selection_sha256'] == C.sha(C.OUT / 'selection.json')
    assert selection['fit_complete_sha256'] == C.sha(C.OUT / 'fit_complete.json')
    for rel, digest in selection['val_files'].items():
        assert C.sha(C.OUT / rel) == digest
    report = {'batch': 'B116', 'created_unix': time.time(), 'sources': sources,
              'selection_sha256': C.sha(C.OUT / 'selection.json'),
              'predictions_complete_sha256': C.sha(C.OUT / 'predictions_complete.json'),
              'selection': selection['by_unit'], 'input_cuts': selection['input_cuts'],
              'interval_scope': 'fixed fits and frozen selection; exploratory reused TEST',
              'global_cluster_contract': D.check_cluster_contract(),
              'cells': {}, 'averages': {}, 'failure_strata': {}}
    arrays, hashes = {}, {}
    for unit in C.UNITS:
        for rule in C.RULES:
            key = unit + '/' + rule
            path = C.OUT / 'test' / f'{unit}_{rule}.npz'
            arr = C.npz(path)
            ref = D.common_row_support(unit, rule, 'test')
            for name, value in ref.items():
                assert np.array_equal(arr[name], value), (key, name)
            for arch in ARCHES:
                ensemble = np.mean([arr[f'Seq_{arch}_s{s}'] for s in C.SEEDS], axis=0)
                assert np.allclose(ensemble, arr[f'Seq_{arch}_ensemble'], rtol=0, atol=1e-12)
            arr[SELECTED] = arr['Seq_' + selection['by_unit'][unit]['architecture'] + '_ensemble']
            arrays[key] = arr
            hashes[path.relative_to(C.OUT).as_posix()] = C.sha(path)
            report['cells'][key] = statistics(arr, key)
            inputs = D.aligned_inputs(unit, rule, 'test')
            assert np.array_equal(inputs['idx'], arr['idx'])
            report['failure_strata'][key] = failure_statistics(arr, inputs['x'], selection['input_cuts'][unit], key)
    scopes = {'overall_8rules': arrays,
              'overall_D10': {k: a for k, a in arrays.items() if k.endswith('/D10')}}
    scopes.update({u + '/8rules': {k: a for k, a in arrays.items() if k.startswith(u + '/')} for u in C.UNITS})
    for scope, cells in scopes.items():
        report['averages'][scope] = average_statistics(cells, 'average/' + scope)
    report['statements'] = {
        'R1': report['averages']['overall_D10']['pairs'][SELECTED + '_over_' + GP]['cluster']['ci95'][0] > 0,
        'R2': report['averages']['overall_8rules']['pairs'][SELECTED + '_over_' + GP]['cluster']['ci95'][0] > 0,
        'R3': report['averages']['overall_8rules']['pairs'][SELECTED + '_over_Seq_lstm_ensemble']['cluster']['ci95'][0] > 0}
    report['statement_definitions'] = {'R1': 'overall four-design D10 selected/GP+fast lower95 > 0',
                                      'R2': 'overall 32-cell macro selected/GP+fast lower95 > 0',
                                      'R3': 'overall 32-cell macro selected/matched-LSTM lower95 > 0'}
    report['test_files'] = hashes
    C.save(C.OUT / 'readout.json', report)
    (C.OUT / 'readout.md').write_text(render(report), encoding='utf8')
    figures(report)
    outputs = ['readout.json', 'readout.md', 'figures/b116_average.png', 'figures/b116_failure.png']
    C.save(C.OUT / 'readout_complete.json', {'all_pass': True, 'cells': len(arrays),
           'selection_sha256': report['selection_sha256'],
           'files': {p: C.sha(C.OUT / p) for p in outputs}, 'published': False})
    print('B116 readout staged for independent audit; not published')


if __name__ == '__main__':
    main()
