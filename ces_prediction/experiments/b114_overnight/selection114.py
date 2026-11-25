"""Independent within-outer-unit VAL-only selection; never scores TEST."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import common114 as C

CAT_CONFIGS = ('depth4', 'depth8', 'depth6_long4000', 'depth6_l2_10')
FORMS = ('direct', 'residual')


def ah(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def checked_files(root, files):
    for rel, digest in files.items():
        path = (root / rel).resolve()
        assert path.is_relative_to(root.resolve()) and C.sha(path) == digest, str(path)


def neural_architectures():
    # Literal declaration is finalized with the independent model schema.
    return ('resnet128', 'ple128')


def compute_selection():
    sources = C.source_guard()
    manifest = C.verify_inputs()
    barrier = C.read(C.OUT / 'all_fits_complete.json')
    assert barrier['all_pass']
    families = {'cat': (CAT_CONFIGS, 'config', 'Cat', 96),
                'neural': (neural_architectures(), 'architecture', 'NN', 48)}
    result = {'sources': sources, 'criterion': 'within_each_outer_unit_mean_7rules_VAL_MSE_ratio_to_GP_only',
              'tie_break': 'lexicographic candidate identifier then form',
              'TEST_used_for_selection': False, 'all_candidates': {}, 'VAL_files': {}}
    for family, (configs, field, prefix, expected) in families.items():
        root = C.OUT / family
        fit = C.read(root / 'fit_complete.json')
        assert fit['all_pass'] and fit['fits'] == expected and fit['sources'] == sources
        assert fit.get('TEST_predictions', 0) == 0
        assert len(fit['receipts']) == expected
        checked_files(root, fit['receipts'])
        checked_files(root, fit['val_files'])
        result[family] = {'by_unit': {}}
        result['all_candidates'][family] = {}
        for tag, unit in manifest['units'].items():
            scores = {(config, form): {} for config in configs for form in FORMS}
            for rule in C.TRAIN_RULES:
                original = C.npz(C.PRIOR / f'{tag}_{rule}.npz')
                val = original['split'] == 1
                assert val.any() and rule != 'K10'
                path = root / 'val' / f'{tag}_{rule}.npz'
                arr = C.npz(path)
                assert C.sha(path) == fit['val_files'][path.relative_to(root).as_posix()]
                for key, old in [('idx', 'idx'), ('truth', 'y'), ('gp', 'gp'),
                                 ('cluster', 'cluster'), ('discharge', 'discharge')]:
                    assert np.array_equal(arr[key], original[old][val]), family + '/' + key
                if 'GP_only' in arr:
                    assert np.array_equal(arr['GP_only'], arr['gp'])
                denom = float(np.mean((arr['truth'] - arr['gp']) ** 2))
                assert np.isfinite(denom) and denom > 0
                key = unit['name'] + '/' + rule + '/F'
                result['VAL_files'][path.relative_to(C.OUT).as_posix()] = {
                    'sha256': C.sha(path), 'rows': int(val.sum()),
                    'idx_sha256': ah(arr['idx']), 'truth_sha256': ah(arr['truth']),
                    'gp_sha256': ah(arr['gp'])}
                for config in configs:
                    for form in FORMS:
                        names = [f'{prefix}_{config}_{form}_s{s}' for s in C.SEEDS]
                        for name in names:
                            assert arr[name].shape == arr['truth'].shape and np.isfinite(arr[name]).all()
                        ens = np.mean([arr[n] for n in names], axis=0)
                        assert np.array_equal(ens, arr[f'{prefix}_{config}_{form}_ensemble'])
                        mse = float(np.mean((ens - arr['truth']) ** 2))
                        scores[config, form][key] = {'MSE': mse, 'GP_MSE': denom, 'ratio': mse / denom}
            candidates = []
            for (config, form), cells in scores.items():
                assert len(cells) == 7
                candidates.append({field: config, 'form': form,
                                   'score': float(np.mean([c['ratio'] for c in cells.values()])),
                                   'cells': cells})
            winner = min(candidates, key=lambda c: (c['score'], c[field], c['form']))
            result[family]['by_unit'][tag] = {field: winner[field], 'form': winner['form'], 'score': winner['score']}
            result['all_candidates'][family][tag] = candidates
    result['overall_family_by_unit'] = {
        tag: min(('cat', 'neural'), key=lambda f: (result[f]['by_unit'][tag]['score'], f))
        for tag in manifest['units']}
    return result


def main():
    assert not (C.OUT / 'selection.json').exists(), 'Preserve frozen selection'
    for family in ('cat', 'neural'):
        assert not (C.OUT / family / 'prediction_complete.json').exists(), 'TEST already predicted'
    result = compute_selection()
    result['selected_at'] = time.time()
    C.save(C.OUT / 'selection.json', result)
    print('B114 VAL-only family choices frozen; no TEST score read')


if __name__ == '__main__':
    main()
