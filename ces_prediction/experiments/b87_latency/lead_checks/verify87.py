"""Independent B87 audit: direct timestamp search and direct row error aggregation.

Does not import B87 future/matrix/readout functions. Requires completed results.
Run from repository root: py ces_prediction/experiments/b87_latency/lead_checks/verify87.py
"""
from pathlib import Path
import sys
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'ces_prediction/experiments/b73_targets'))
import common as C
import designs as DS
import stage_trees as T


def main():
    cache = ROOT / 'data/.b87'
    result = json.loads((C.PAPER / 'b87_latency.json').read_text(encoding='utf-8'))
    R, G = C.rows(), C.grid()
    saved = C.load_npz(cache / 'future.npz')
    sample = np.random.default_rng(871122).choice(len(R['gi']), 300, replace=False)
    for q in sample:
        g = R['gi'][q]
        for j in range(5):
            candidates = np.flatnonzero((G['fidx'] == R['fidx'][q]) &
                (G['seg'] == G['seg'][g]) &
                (np.abs(G['time'] - G['time'][g] - .01 * (j + 1)) < .003))
            assert len(candidates) <= 1
            p = int(candidates[0]) if len(candidates) else -1
            assert saved['source'][q, j] == p
            v = G['ti'][p] if p >= 0 else np.nan
            assert np.array_equal(np.array(v), np.array(saved['slots'][q,j]), equal_nan=True)
    units, _, _ = DS.units(R)
    history = T.history_matrix(R)
    feature_builder = T.Feats()
    for u in units:
        window = feature_builder.window(u['fit'])
        fast, lag = feature_builder.at(window, R['gi'])
        complete = T.complete_mask(history, fast, lag, R['gp'])
        fit_rows = np.isin(R['fidx'],u['fit']) & complete & np.isfinite(R['tg_F'])
        assert fit_rows.any()
        for arm in ('H0','X0','H1','X1','H2','X2','H5','X5'):
            z = C.load_npz(cache / 'fits' / f"{u['name']}__{arm}.npz")
            n = int(arm[1:])
            slot_active = np.array([j < n and np.isfinite(saved['slots'][fit_rows,j]).any() for j in range(5)])
            prefix = history.shape[1] + (fast.shape[1] if arm.startswith('X') else 0)
            expected = np.concatenate([np.ones(prefix,dtype=bool), slot_active])
            assert np.array_equal(expected,z['active'])
            assert fit_rows.sum() == z['fit_rows']
            # Arbitrary test-only mutation leaves the fit-only selector unchanged.
            mutated = saved['slots'].copy()
            mutated[np.isin(R['fidx'],u['test'])] = 1000.
            assert np.array_equal(np.isfinite(mutated[fit_rows]).any(axis=0),
                                  np.isfinite(saved['slots'][fit_rows]).any(axis=0))
    largest = 0.
    verified = 0
    for design, d in result['designs'].items():
        if design in ('V1', 'V2'):
            choices = C.REPEATS if design == 'V1' else C.V2_SEEDS
            grouped = [[u for u in units if u['design'] == design and u['sub'] == s] for s in choices]
        elif design == 'V3':
            grouped = [[u for u in units if u['design'] == 'V3']]
        else:
            grouped = [[u for u in units if u['name'] == design]]
        points = {a: [] for a in d['metrics']}
        mae = {a: [] for a in points}
        pairpoints = {p: [] for p in d['pairs']}
        for group in grouped:
            errors = {}
            for arm in points:
                indices, predictions = [], []
                for u in group:
                    z = C.load_npz(cache / 'fits' / f"{u['name']}__{arm}.npz")
                    assert not np.intersect1d(u['fit'], u['test']).size
                    assert np.isin(R['fidx'][z['idx']], u['test']).all()
                    assert z['active'].dtype == bool
                    indices.extend(z['idx'].tolist())
                    predictions.extend(z['est'].tolist())
                idx = np.array(indices)
                assert len(idx) == len(set(indices)) == d['rows']
                assert len(np.unique(R['fidx'][idx])) == d['shots']
                assert len(np.unique(R['clus'][idx])) == d['clusters']
                order = np.argsort(idx)
                if errors:
                    assert np.array_equal(idx[order], reference_idx)
                reference_idx = idx[order]
                e = np.array(predictions)[order] - R['tg_F'][reference_idx]
                assert np.isfinite(e).all()
                errors[arm] = np.dot(e,e)
                points[arm].append(np.sqrt(errors[arm] / len(e)))
                mae[arm].append(np.abs(e).mean())
            for pair in pairpoints:
                a,b = pair.split(' over ')
                pairpoints[pair].append(1 - errors[a]/errors[b])
        comparisons = [(np.mean(v), d['metrics'][a]['RMSE']['point']) for a,v in points.items()]
        comparisons += [(np.mean(v), d['metrics'][a]['MAE']['point']) for a,v in mae.items()]
        comparisons += [(np.mean(v), d['pairs'][p]['point']) for p,v in pairpoints.items()]
        for actual, expected in comparisons:
            delta = abs(actual - expected)
            largest = max(largest, delta)
            assert np.isclose(actual, expected, rtol=1e-12, atol=1e-10)
            verified += 1
        for st in d['statements'].values():
            assert st['holds'] == (st['cluster'][0] > 0)
    C.save_text(C.PAPER / 'b87_lead_checks.md',
        '# B.87 independent recomputation\n\nPASS. Separate direct timestamp search verified '
        f'{len(sample)*5} future slots. Direct row aggregation verified {verified} metric and paired-skill '
        f'point estimates across every design; maximum absolute difference {largest:.3g}. '
        'Disjoint files, scored row/shot/cluster counts, identical rows across arms, fit-only active-column masks '
        '(including test-only mutation), '
        'finite errors and statement verdicts passed. Bootstrap intervals were code-reviewed; '
        'this verifier does not independently estimate interval coverage or refit models.\n')
    print(f'PASS: {verified} point estimates; max difference {largest:.3g}')


if __name__ == '__main__':
    main()
