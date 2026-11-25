"""Independent selected-model replay, source checks and paired B114 readout."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import sys
import time
import json
from pathlib import Path
import numpy as np
import common114 as C
from selection114 import ah, checked_files, compute_selection
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'b113_catboost'))
import audit113 as A

FAMILIES = ('cat', 'neural')
BASELINES = ('GP_fast_boost', 'GP_fair_mixture', 'HGB_direct', 'HGB_residual',
             'GP_only', 'B113_Cat_direct', 'B113_Cat_residual', 'B113_Cat_matched')
ENSEMBLES = ('Cat_selected_ensemble', 'NN_selected_ensemble')
SINGLES = tuple(prefix + '_selected_s' + str(s) for prefix in ('Cat', 'NN') for s in C.SEEDS)
MODELS = ENSEMBLES + SINGLES + BASELINES
PAIRS = ((ENSEMBLES[0], ENSEMBLES[1]),) + tuple((n, b) for n in ENSEMBLES + SINGLES for b in BASELINES)


def neural_predict(state, architecture, x, gp=None):
    """Separate CPU functional equations; no producer model/import is used."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    params = {k[6:]: torch.as_tensor(v, dtype=torch.float32)
              for k, v in state.items() if k.startswith('model.')}
    def linear(h, prefix):
        return F.linear(h, params[prefix + '.weight'], params[prefix + '.bias'])
    result = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            h = torch.as_tensor(((x[start:start + 4096] - state['x_mu']) /
                                 state['x_sd']).astype(np.float32))
            if architecture == 'resnet128':
                h = F.gelu(linear(h, 'input'), approximate='none')
                for block in range(3):
                    r = linear(F.gelu(linear(h, f'blocks.{block}.0'), approximate='none'),
                               f'blocks.{block}.2')
                    h = (h + r) / np.sqrt(2)
                h = linear(h, 'output')
            else:
                assert architecture == 'ple128'
                knots = params['knots']
                widths = torch.maximum(knots[:, 1:] - knots[:, :-1], torch.tensor(1e-6))
                h = torch.clamp((h[:, :, None] - knots[:, :-1]) / widths, 0, 1).flatten(1)
                h = F.gelu(linear(h, 'layers.0'), approximate='none')
                h = F.gelu(linear(h, 'layers.2'), approximate='none')
                h = linear(h, 'layers.4')
            result.append(h[:, 0].numpy().astype(np.float64))
    pred = np.concatenate(result) * float(state['y_sd']) + float(state['y_mu'])
    if gp is not None:
        pred += gp
    assert np.isfinite(pred).all()
    return pred


def training_contract(tag):
    coverage, pieces = {}, []
    for rule in C.TRAIN_RULES:
        z = C.npz(C.PRIOR / f'{tag}_{rule}.npz')
        fit = z['split'] == 0
        assert fit.any() and np.isfinite(z['x']).all()
        coverage[rule] = {'rows': int(fit.sum()), **{
            name + '_sha256': ah(z[name][fit]) for name in ('idx', 'x', 'y', 'gp')}}
        pieces.append(tuple(z[name][fit] for name in ('x', 'y', 'gp')))
    x, y, gp = [np.concatenate([p[j] for p in pieces]) for j in range(3)]
    weights = np.concatenate([np.full(len(p[1]), len(y) / (7 * len(p[1]))) for p in pieces])
    mu = np.average(x, axis=0, weights=weights)
    sd = np.maximum(np.sqrt(np.average((x - mu) ** 2, axis=0, weights=weights)), 1e-8)
    return x, y, gp, weights, mu, sd, coverage


def verify_receipt(receipt, context, form):
    x, y, gp, weights, mu, sd, coverage = context
    target = y if form == 'direct' else y - gp
    expected = dict(x=x, y=y, gp=gp, weights=weights, target=target)
    assert receipt['TRAIN_coverage'] == coverage and receipt['TRAIN_rows'] == len(y)
    assert receipt['VAL_fit_rows'] == receipt['TEST_fit_rows'] == 0
    assert receipt['pooled_hashes'] == {k: ah(v) for k, v in expected.items()}


def expected_knots(context):
    x, _, _, weights, mu, sd, _ = context
    rows = []
    for j in range(x.shape[1]):
        ix = np.argsort(x[:, j], kind='stable')
        ww, vals = weights[ix], x[ix, j]
        cdf = (np.cumsum(ww) - ww / 2) / ww.sum()
        knots = np.interp(np.linspace(0, 1, 17), cdf, vals, left=vals[0], right=vals[-1])
        rows.append((knots - mu[j]) / sd[j])
    return np.asarray(rows, dtype=np.float32)


def verify_neural_state(state, receipt, history, context, form, architecture, knots):
    x, y, gp, weights, mu, sd, coverage = context
    target = y if form == 'direct' else y - gp
    ym = float(np.average(target, weights=weights))
    ys = float(max(np.sqrt(np.average((target - ym) ** 2, weights=weights)), 1.0))
    assert np.array_equal(state['x_mu'], mu) and np.array_equal(state['x_sd'], sd)
    assert float(state['y_mu']) == ym and float(state['y_sd']) == ys
    assert bool(state['residual']) == (form == 'residual')
    assert history and receipt['epochs'] == len(history) <= 80
    assert receipt['best_epoch'] == int(np.argmin([h['objective'] for h in history]))
    assert receipt['best_VAL_MSE'] == min(h['objective'] for h in history)
    best = float('inf')
    for epoch, h in enumerate(history):
        assert h['epoch'] == epoch and set(h['VAL_rule_MSE']) == set(C.TRAIN_RULES)
        score = float(np.mean(list(h['VAL_rule_MSE'].values())))
        assert score == h['VAL_MSE'] == h['objective']
        assert h['improved'] == (score < best)
        best = min(best, score)
    if architecture == 'ple128':
        assert np.array_equal(knots, state['model.knots'])


def render(report):
    lines = ['# B114: overnight VAL-selected scalar estimators', '',
             'Exploratory reused-TEST comparison. Family choices use only each outer unit\'s own VAL.',
             'No TEST score chose a configuration, learning form, seed or checkpoint.',
             'Fixed-fit pointwise 95% intervals omit refitting and selection uncertainty.',
             'Seeds are retained separately; these are fixed-budget package/configuration comparisons.', '',
             '| Unit | CatBoost choice/form/VAL ratio | Neural choice/form/VAL ratio |',
             '|---|---|---|']
    for tag, c in report['selection']['cat']['by_unit'].items():
        n = report['selection']['neural']['by_unit'][tag]
        lines.append(f"| {tag} | {c['config']}/{c['form']}/{c['score']:.6f} | {n['architecture']}/{n['form']}/{n['score']:.6f} |")
    lines += ['',
             '| Unit / rule | Model | Rows | MSE [95% shot-cluster] | RMSE |',
             '|---|---|---:|---|---:|']
    for key, cell in report['cells'].items():
        for name, model in cell['models'].items():
            low, high = model['cluster_MSE95']
            lines.append(f"| {key} | {name} | {cell['rows']} | {model['MSE']:.3f} [{low:.3f}, {high:.3f}] | {model['RMSE']:.3f} |")
    lines += ['', '| Unit / rule | Paired MSE skill | Point [95% shot-cluster] | Excludes zero |',
              '|---|---|---|---|']
    for key, cell in report['cells'].items():
        for pair, value in cell['pairs'].items():
            if '_ensemble_over_' in pair:
                low, high = value['cluster']['ci95']
                lines.append(f"| {key} | {pair} | {value['point']:+.6f} [{low:+.6f}, {high:+.6f}] | {value['cluster']['excludes_zero']} |")
    lines += ['', 'Registered statements: ' + json.dumps(report['statements']), '',
              'Positive findings require confirmation on unused observations with matched inputs and budgets.',
              'A matched adequate-budget comparison on unused observations could overturn unfavorable findings.',
              'No physical identification, architecture-class impossibility or 10ms runtime claim follows.']
    return '\n'.join(lines) + '\n'


def main():
    sources = C.source_guard()
    manifest = C.verify_inputs()
    selection = C.read(C.OUT / 'selection.json')
    recomputed = compute_selection()
    assert {k: v for k, v in selection.items() if k != 'selected_at'} == recomputed
    selected_at = selection['selected_at']
    b113 = C.ROOT / 'data/.b113'
    b112 = C.ROOT / 'data/colab_b112'
    report113 = C.read(C.ROOT / 'docs/paper/b113_comparison.json')
    assert report113['independent_audit']['all_pass'] and C.read(b113 / 'audit.json')['all_pass']
    assert C.read(b112 / 'independent_audit.json')['all_pass']
    prior112 = C.read(C.ROOT / 'docs/paper/b112_comparison.json')
    assert C.sha(b112 / 'b112_results.zip') == prior112['independent_audit']['archive_sha256']
    b112hashes = C.read(b112 / 'retrieved/result_hashes.json')
    b113complete = C.read(b113 / 'complete.json')
    assert C.sha(b113 / 'complete.json') == report113['independent_audit']['complete_sha256']
    completions = {}
    for family in FAMILIES:
        root = C.OUT / family
        fit = C.read(root / 'fit_complete.json')
        checked_files(root, fit['receipts'])
        complete = C.read(root / 'prediction_complete.json')
        assert complete['all_pass'] and complete['sources'] == sources
        assert complete['selection'] == selection[family]
        assert complete['selection_sha256'] == C.sha(C.OUT / 'selection.json')
        assert complete['cells'] == len(complete['files']) == 32
        checked_files(root, complete['files'])
        startpath = root / 'estimates/started.json' if family == 'cat' else root / 'predict_started.json'
        assert C.read(startpath)['started'] >= selected_at
        completions[family] = complete
    A.MODELS, A.PAIRS = MODELS, PAIRS
    cells, replay = {}, {}
    for tag, unit in manifest['units'].items():
        context = training_contract(tag)
        knots = expected_knots(context)
        # Check every fitted candidate's claimed training support, including unselected models.
        for family in FAMILIES:
            fit = C.read(C.OUT / family / 'fit_complete.json')
            receipts = [(C.OUT / family / rel) for rel in fit['receipts'] if f'models/{tag}/' in rel]
            assert len(receipts) == (24 if family == 'cat' else 12)
            for path in receipts:
                receipt = C.read(path)
                verify_receipt(receipt, context, receipt['form'])
                if family == 'cat':
                    assert receipt['params']['task_type'] == 'GPU'
                    assert receipt['tree_count'] == receipt['params']['iterations']
                else:
                    assert receipt['params']['AMP'] is False and receipt['params']['TF32'] is False
                    stem = path.name.removesuffix('_complete.json')
                    weightpath = path.with_name(stem + '.npz')
                    historypath = path.with_name(stem + '_history.json')
                    assert C.sha(weightpath) == receipt['weights_sha256']
                    assert C.sha(historypath) == receipt['history_sha256']
                    verify_neural_state(C.npz(weightpath), receipt, C.read(historypath), context,
                                        receipt['form'], receipt['params']['architecture'], knots)
        cat_choice = selection['cat']['by_unit'][tag]
        nn_choice = selection['neural']['by_unit'][tag]
        cfg, cf = cat_choice['config'], cat_choice['form']
        arch, nf = nn_choice['architecture'], nn_choice['form']
        trees, states = {}, {}
        for seed in C.SEEDS:
            catbase = C.OUT / 'cat/models' / tag / cfg
            receipt = C.read(catbase / f'{cf}_s{seed}_complete.json')
            for ext in ('cbm', 'json'):
                assert C.sha(catbase / f'{cf}_s{seed}.{ext}') == receipt['model_sha256'][ext]
            trees[seed] = C.read(catbase / f'{cf}_s{seed}.json')
            assert len(trees[seed]['oblivious_trees']) == receipt['tree_count']
            nnbase = C.OUT / 'neural/models' / tag / arch
            receipt = C.read(nnbase / f'{nf}_s{seed}_complete.json')
            weightpath = nnbase / f'{nf}_s{seed}.npz'
            historypath = nnbase / f'{nf}_s{seed}_history.json'
            assert C.sha(weightpath) == receipt['weights_sha256']
            assert C.sha(historypath) == receipt['history_sha256']
            states[seed] = C.npz(weightpath)
            verify_neural_state(states[seed], receipt, C.read(historypath), context, nf, arch, knots)
        val_replay = {str(seed): [] for seed in C.SEEDS}
        for rule in C.TRAIN_RULES:
            original = C.npz(C.PRIOR / f'{tag}_{rule}.npz')
            valid = original['split'] == 1
            valcat = C.npz(C.OUT / 'cat/val' / f'{tag}_{rule}.npz')
            valnn = C.npz(C.OUT / 'neural/val' / f'{tag}_{rule}.npz')
            for seed in C.SEEDS:
                pred = A.portable_predict(trees[seed], original['x'][valid])
                if cf == 'residual':
                    pred += original['gp'][valid]
                assert np.allclose(pred, valcat[f'Cat_{cfg}_{cf}_s{seed}'], atol=1e-5, rtol=1e-7)
                pred = neural_predict(states[seed], arch, original['x'][valid],
                                      original['gp'][valid] if nf == 'residual' else None)
                assert np.allclose(pred, valnn[f'NN_{arch}_{nf}_s{seed}'], atol=0.01, rtol=1e-5)
                val_replay[str(seed)].append(float(np.mean((pred - original['y'][valid]) ** 2)))
        for seed in C.SEEDS:
            receipt = C.read(C.OUT / 'neural/models' / tag / arch / f'{nf}_s{seed}_complete.json')
            assert np.isclose(np.mean(val_replay[str(seed)]), receipt['best_VAL_MSE'], rtol=1e-4, atol=1e-3)
        for rule in C.RULES:
            original = C.npz(C.PRIOR / f'{tag}_{rule}.npz')
            test = original['split'] == 2
            if rule == 'K10':
                assert test.all()
            arrays = {f: C.npz(C.OUT / f / 'estimates' / f'{tag}_{rule}.npz') for f in FAMILIES}
            prior_rel = f'estimates/{tag}_{rule}.npz'
            assert C.sha(b112 / 'retrieved' / prior_rel) == b112hashes[prior_rel]
            assert C.sha(b113 / prior_rel) == b113complete['estimates'][prior_rel]
            old112, old113 = C.npz(b112 / 'retrieved' / prior_rel), C.npz(b113 / prior_rel)
            arr = {}
            for key, source in [('idx', 'idx'), ('truth', 'y'), ('cluster', 'cluster'), ('discharge', 'discharge')]:
                expected = original[source][test]
                assert np.array_equal(old112[key], expected) and np.array_equal(old113[key], expected)
                for family in FAMILIES:
                    assert np.array_equal(arrays[family][key], expected)
                arr[key] = expected
            for name in BASELINES[:5]:
                arr[name] = old112[name]
            for form in ('direct', 'residual'):
                arr['B113_Cat_' + form] = old113['CatBoost_' + form + '_ensemble']
            arr['B113_Cat_matched'] = arr['B113_Cat_' + cf]
            errors = {}
            for seed in C.SEEDS:
                name = f'Cat_selected_s{seed}'
                arr[name] = arrays['cat'][name]
                pred = A.portable_predict(trees[seed], original['x'][test])
                if cf == 'residual':
                    pred += original['gp'][test]
                errors[name] = float(np.max(np.abs(pred - arr[name])))
                assert np.allclose(pred, arr[name], atol=1e-5, rtol=1e-7), name
                name = f'NN_selected_s{seed}'
                arr[name] = arrays['neural'][name]
                pred = neural_predict(states[seed], arch, original['x'][test],
                                      original['gp'][test] if nf == 'residual' else None)
                errors[name] = float(np.max(np.abs(pred - arr[name])))
                assert np.allclose(pred, arr[name], atol=0.01, rtol=1e-5), name
            for family, prefix in [('cat', 'Cat'), ('neural', 'NN')]:
                name = prefix + '_selected_ensemble'
                arr[name] = np.mean([arr[f'{prefix}_selected_s{s}'] for s in C.SEEDS], axis=0)
                assert np.array_equal(arr[name], arrays[family][name])
            key = unit['name'] + '/' + rule + '/F'
            cells[key] = A.statistics(arr, key)
            replay[key] = errors
            print('B114 independent replay/readout ' + key, flush=True)
    assert len(cells) == 32
    fresh = [cell for key, cell in cells.items() if '/D10/' in key]
    assert len(fresh) == 4
    pairs = {'R1': 'Cat_selected_ensemble_over_B113_Cat_matched',
             'R2': 'Cat_selected_ensemble_over_GP_fast_boost',
             'R3': 'NN_selected_ensemble_over_GP_fast_boost'}
    audit = {'all_pass': True, 'completed': time.time(), 'sources': sources,
             'cells': len(cells), 'fits': 144, 'bootstrap_draws': 20000,
             'portable_replay_max_abs_ev': replay,
             'selection_sha256': C.sha(C.OUT / 'selection.json'),
             'selection_rebuilt_from_VAL_only': True,
             'tree_tolerance': {'atol_ev': 1e-5, 'rtol': 1e-7},
             'neural_tolerance': {'atol_ev': 0.01, 'rtol': 1e-5},
             'local_fits_in_audit': 0}
    report = {'registration': 'PREREGISTRATION_B114.md', 'exploratory': True,
              'selection': selection, 'cells': cells, 'independent_audit': audit,
              'statements': {name: all(c['pairs'][pair]['cluster']['ci95'][0] > 0 for c in fresh)
                             for name, pair in pairs.items()},
              'limits': ['Reused TEST, selection-conditional fixed-fit pointwise intervals.',
                         'No refitting or selection uncertainty included in bootstrap intervals.',
                         'All seeds reported separately; arithmetic ensembles are fixed.',
                         'Fixed-budget package/configuration comparisons; no runtime10ms claim.']}
    C.save(C.OUT / 'audit.json', audit)
    C.save(C.ROOT / 'docs/paper/b114_comparison.json', report)
    (C.ROOT / 'docs/paper/b114_comparison.md').write_text(render(report), encoding='utf8')


if __name__ == '__main__':
    main()
