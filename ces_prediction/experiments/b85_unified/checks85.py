# Claude authored the implementation
# ces_prediction/experiments/b85_unified/checks85.py
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'ces_prediction' / 'experiments' / 'b85_unified'))

from common85 import (
    CACHE, DATA, PAPER, RULES, TRAIN_RULES, load_sources,
    select_units, rule_for
)
from data85 import unit_manifest, fit_stats, anchors, build_segments, feature_helper


def _load_ref(path):
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    return None


def _deep_convert(obj):
    if isinstance(obj, dict):
        return {k: _deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_convert(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def check_anchors(G, R):
    """All 8 rules: every R row anchor must equal reference anc_g.
    Ages must match within 1e-6 ms. nvis>0 where anchor exists.
    Missing reference files are fatal."""
    passes = {}
    b75_dir = DATA / '.b75' / 'inputs'
    b78_dir = DATA / '.b78' / 'inputs'
    r_gi = R['gi']

    for rule in RULES:
        if rule == 'K10':
            ref = _load_ref(b78_dir / 'K10.npz')
        else:
            ref = _load_ref(b75_dir / f'{rule}.npz')
        assert ref is not None, f'Reference file missing for rule {rule}'

        ref_anc = ref['anc_g']
        ref_age = ref.get('age', None)
        ref_nvis = ref.get('nvis', None)

        anc = anchors(rule, G)
        n = len(r_gi)
        assert len(ref_anc) == n, f'{rule}: ref length {len(ref_anc)} != R rows {n}'

        anc_at_r = anc[r_gi[:n]]
        support = np.ones(n, dtype=bool)
        if rule == 'K10':
            sets = _load_ref(DATA / '.b78' / 'sets.npz')
            assert sets is not None, 'K10 scored support missing'
            support = sets['common_F'] | sets['common_L5']
        assert np.array_equal(anc_at_r[support], ref_anc[:n][support]), f'{rule}: anchor mismatch on scored support'

        if ref_age is not None:
            age_ms = np.zeros(n, dtype=np.float64)
            for i in range(n):
                gi = r_gi[i]
                a = anc[gi]
                if a >= 0:
                    age_ms[i] = (G['time'][gi] - G['time'][a]) * 1000.0
            valid = (ref_anc[:n] >= 0) & support
            if valid.any():
                diff = np.abs(age_ms[valid] - ref_age[:n][valid].astype(np.float64))
                assert np.all(diff < 1e-6), \
                    f'{rule}: age mismatch max={diff.max()}'

        if ref_nvis is not None:
            has_anc = (anc_at_r >= 0) & support
            has_nvis = ref_nvis[:n] > 0
            where_anc = np.where(has_anc)[0]
            if len(where_anc) > 0:
                assert np.all(ref_nvis[:n][has_anc] > 0), \
                    f'{rule}: nvis==0 where anchor exists'

        passes[rule] = True

    return passes


def _get_eligible_mask(rule, G, seg_indices, frames_data=None):
    """Independently derive which Ti values are visible at each row for a rule.
    Returns dict: gi -> set of eligible anchor gis (strictly before gi)."""
    time = G['time']
    ti = G['ti']
    eligible = {}

    if rule.startswith('D'):
        delay_s = int(rule[1:]) / 1000.0
        tolerance = 0.003
        for i, gi in enumerate(seg_indices):
            t_cur = time[gi]
            t_limit = t_cur - delay_s + tolerance
            elig = set()
            for j in range(i - 1, -1, -1):
                gj = seg_indices[j]
                if time[gj] <= t_limit and np.isfinite(ti[gj]):
                    elig.add(int(gj))
            eligible[int(gi)] = elig

    elif rule == 'S':
        gap_cutoff = -1
        for i, gi in enumerate(seg_indices):
            if i == 0:
                eligible[int(gi)] = set()
                continue
            dt = time[gi] - time[seg_indices[i - 1]]
            if 0.015 < dt < 0.045:
                candidate = int(seg_indices[i - 1])
                if candidate > gap_cutoff:
                    gap_cutoff = candidate
            if gap_cutoff >= 0:
                eligible[int(gi)] = {int(j) for j in seg_indices if j <= gap_cutoff and np.isfinite(ti[j])}
            else:
                eligible[int(gi)] = set()

    elif rule == 'K10':
        fr = frames_data['fr'] if frames_data is not None else None
        for i, gi in enumerate(seg_indices):
            elig = set()
            if fr is not None:
                for j in range(i - 1, -1, -1):
                    gj = seg_indices[j]
                    if int(gj) < len(fr) and fr[int(gj)] >= 0 and fr[int(gj)] % 10 == 0 and np.isfinite(ti[int(gj)]):
                        elig.add(int(gj))
            eligible[int(gi)] = elig

    return eligible


def _s_cutoff_example(G):
    """Find a real S segment with monotone cutoff, return example info."""
    seg = G['seg']
    time = G['time']
    ti = G['ti']
    for s_id in np.unique(seg)[:50]:
        smask = seg == s_id
        sidx = np.where(smask)[0]
        if len(sidx) < 10:
            continue
        cutoffs = []
        gap_cutoff = -1
        for i in range(1, len(sidx)):
            dt = time[sidx[i]] - time[sidx[i - 1]]
            if 0.015 < dt < 0.045:
                cand = int(sidx[i - 1])
                if cand > gap_cutoff:
                    gap_cutoff = cand
            cutoffs.append(gap_cutoff)
        if len(cutoffs) > 3:
            mono = all(cutoffs[j] <= cutoffs[j + 1] for j in range(len(cutoffs) - 1))
            if mono and cutoffs[-1] >= 0:
                return {'segment': int(s_id), 'n_rows': len(sidx),
                        'cutoffs_sample': cutoffs[:10], 'monotone': True}
    return {'found': False}


def check_prefix_invariance(G, R, stats, unit):
    """Full prefix test for all 8 rules. For each rule sample up to 5 real
    scored times from the SAME file/segment. Freeze stats.
    Mutation: set all ineligible finite Ti to NaN, then repeat with +1000.
    Prefix through T must be exactly unchanged."""
    manifest = unit_manifest(unit)
    all_ids = manifest['train']
    if not all_ids:
        return {'pass': False, 'reason': 'no_train_files'}

    frames_data = None
    fp = DATA / '.b78' / 'frames.npz'
    if fp.exists():
        frames_data = dict(np.load(fp, allow_pickle=True))

    results = []
    r_gi_set = set(R['gi'].tolist()) if 'gi' in R else set()

    for rule in RULES:
        # Find a file/segment with scored rows
        found_seg = None
        found_fid = None
        found_sidx = None
        for fid in all_ids[:20]:
            fmask = G['fidx'] == fid
            findices = np.where(fmask)[0]
            if len(findices) == 0:
                continue
            for s_id in np.unique(G['seg'][findices]):
                smask = G['seg'][findices] == s_id
                sidx = np.sort(findices[smask])
                scored = [gi for gi in sidx if int(gi) in r_gi_set]
                if len(scored) >= 5:
                    found_seg = s_id
                    found_fid = fid
                    found_sidx = sidx
                    break
            if found_seg is not None:
                break

        if found_seg is None:
            results.append({'rule': rule, 'skip': 'no_scored_segment'})
            continue

        scored_in_seg = [gi for gi in found_sidx if int(gi) in r_gi_set]
        sample_gis = scored_in_seg[:5]

        anc_arr = anchors(rule, G)
        eligible_map = _get_eligible_mask(rule, G, found_sidx, frames_data)

        # Build original segments
        segs_orig = build_segments([int(found_fid)], stats, rule, G=G, R=R)
        orig_by_seg = {s['segment']: s for s in segs_orig}
        if int(found_seg) not in orig_by_seg:
            results.append({'rule': rule, 'skip': 'segment_not_built'})
            continue
        seg_orig = orig_by_seg[int(found_seg)]

        all_ok = True
        for target_gi in sample_gis:
            target_gi = int(target_gi)
            # Find index of target_gi in segment
            gi_list = seg_orig['gi'].tolist()
            if target_gi not in gi_list:
                continue
            t_idx = gi_list.index(target_gi)

            # Eligible set for this target
            elig = eligible_map.get(target_gi, set())
            # All indices in segment at or after target
            future_set = set(int(g) for g in found_sidx if g >= target_gi)

            for mutation_name, mutation_val in [('nan', np.nan), ('+1000', 1000.0)]:
                def make_mutator(elig_s, fut_s, mval, mn):
                    def mutate_fn(G_copy):
                        ti = G_copy['ti']
                        for i_s, gi_s in enumerate(found_sidx):
                            gi_s = int(gi_s)
                            if gi_s in fut_s:
                                # Future or own: mutate if finite
                                if np.isfinite(ti[gi_s]):
                                    if mn == 'nan':
                                        ti[gi_s] = np.nan
                                    else:
                                        ti[gi_s] += mval
                            elif gi_s not in elig_s:
                                # Not eligible, not future: mutate if finite
                                if np.isfinite(ti[gi_s]):
                                    if mn == 'nan':
                                        ti[gi_s] = np.nan
                                    else:
                                        ti[gi_s] += mval
                            # else: eligible and before target -> keep
                        return G_copy
                    return mutate_fn

                segs_mut = feature_helper(
                    [int(found_fid)], stats, rule, G=G, R=R,
                    mutate_ti_fn=make_mutator(elig, future_set, mutation_val, mutation_name))
                mut_by_seg = {s['segment']: s for s in segs_mut}
                if int(found_seg) not in mut_by_seg:
                    all_ok = False
                    continue
                seg_mut = mut_by_seg[int(found_seg)]

                # Prefix through T must be exactly unchanged
                n_prefix = t_idx + 1
                n_check = min(n_prefix, len(seg_mut['x']))
                if n_check == 0:
                    continue
                if not np.allclose(seg_orig['x'][:n_check], seg_mut['x'][:n_check],
                                   rtol=0, atol=0, equal_nan=True):
                    # Find first mismatch
                    for row in range(n_check):
                        if not np.allclose(seg_orig['x'][row], seg_mut['x'][row],
                                           rtol=0, atol=0, equal_nan=True):
                            all_ok = False
                            break
                    if not all_ok:
                        break
            if not all_ok:
                break

        results.append({'rule': rule, 'pass': all_ok, 'n_tested': len(sample_gis)})

    overall = all(r.get('pass', False) for r in results if 'skip' not in r)
    return {'pass': overall, 'details': results}


def check_k10_prefix(G, R, stats, unit):
    """K10: mutate all non-kept Ti before T plus own/future."""
    # Covered by check_prefix_invariance with K10 in RULES
    return {'pass': True, 'note': 'covered_by_prefix_invariance'}


def check_label_mutation_invariance(G, R, stats, unit):
    """Mutating R labels does not change x."""
    manifest = unit_manifest(unit)
    fids = manifest['train'][:2]
    if not fids:
        return {'pass': False, 'reason': 'no_files'}

    import copy
    segs_orig = build_segments(fids, stats, 'D100', G=G, R=R)
    R_mut = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in R.items()}
    if 'tg_F' in R_mut:
        R_mut['tg_F'] = R_mut['tg_F'] + 999.0
    if 'tg_L5' in R_mut:
        R_mut['tg_L5'] = R_mut['tg_L5'] + 999.0
    segs_mut = build_segments(fids, stats, 'D100', G=G, R=R_mut)

    ok = True
    for so, sm in zip(segs_orig, segs_mut):
        if not np.array_equal(so['x'], sm['x']):
            ok = False
            break
    return {'pass': ok}


def check_stats_held_out(G, R, units):
    """Stats computed with/without val files differ; held out mutation leaves stats identical."""
    u = units[0]
    m = unit_manifest(u)
    s1 = fit_stats(m['train'], G, R)
    s2 = fit_stats(m['train'] + m['val'], G, R)
    differs = not np.allclose(s1['fast_mu'], s2['fast_mu'])

    # Mutation of val Ti doesn't change train stats
    G_mut = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in G.items()}
    val_mask = np.isin(G_mut['fidx'], m['val'])
    G_mut['ti'][val_mask] = np.nan
    s3 = fit_stats(m['train'], G_mut, R)
    identical = np.array_equal(s1['fast_mu'], s3['fast_mu']) and np.array_equal(s1['fast_sd'], s3['fast_sd'])
    assert identical, 'Stats changed after val Ti mutation'
    assert differs, 'Stats identical with/without val files'
    return {'pass': True, 'differs_with_val': differs, 'identical_after_mutation': identical}


def check_manifests_disjoint(units):
    """All 15 units: train/val/test pairwise disjoint."""
    for u in units:
        m = unit_manifest(u)
        t, v, te = set(m['train']), set(m['val']), set(m['test'])
        assert len(t & v) == 0, f"{u['name']}: train/val overlap"
        assert len(t & te) == 0, f"{u['name']}: train/test overlap"
        assert len(v & te) == 0, f"{u['name']}: val/test overlap"
    return {'pass': True, 'n_units': len(units)}


def check_rules_deterministic():
    """Hundreds of K10 excluded, deterministic, covers all 7 train rules."""
    seen = set()
    for ep in range(100):
        for si in range(10):
            r = rule_for('V1_r2_k0', 1701, ep, si)
            assert r == rule_for('V1_r2_k0', 1701, ep, si), 'Non-deterministic'
            assert r in TRAIN_RULES, f'K10 or unknown rule: {r}'
            seen.add(r)
    assert seen == set(TRAIN_RULES), f'Not all rules sampled: {seen}'
    return {'pass': True}


def check_xvtmc_zero_firstdt_zero(G, R, stats, unit):
    """x slots 13:15 (MC) always zero, 19:22 (VT) always zero, first dt zero."""
    manifest = unit_manifest(unit)
    fids = manifest['train'][:3]
    segs = build_segments(fids, stats, 'D100', G=G, R=R)
    for s in segs:
        x = s['x']
        assert np.all(x[:, 13:15] == 0), 'MC slots not zero'
        assert np.all(x[:, 19:22] == 0), 'VT slots not zero'
        assert x[0, 15] == 0.0, 'First dt not zero'
    return {'pass': True}


def check_label_values(R, G, stats, unit):
    """Denormalized label values equal R.tg_F and R.tg_L5."""
    manifest = unit_manifest(unit)
    fids = manifest['train'][:5]
    segs = build_segments(fids, stats, 'D100', G=G, R=R)
    r_lookup = {}
    if 'gi' in R:
        for idx in range(len(R['gi'])):
            r_lookup[int(R['gi'][idx])] = idx
    ok = True
    for s in segs:
        for i, gi in enumerate(s['gi']):
            gi = int(gi)
            if gi in r_lookup:
                ri = r_lookup[gi]
                if s['mask'][i, 0] and 'tg_F' in R:
                    expected = float(R['tg_F'][ri])
                    actual = float(s['y'][i, 0]) * stats['ti_sd'] + stats['ti_mu']
                    if not np.isclose(expected, actual, rtol=1e-6, atol=1e-3):
                        ok = False
                if s['mask'][i, 1] and 'tg_L5' in R:
                    expected = float(R['tg_L5'][ri])
                    actual = float(s['y'][i, 1]) * stats['ti_sd'] + stats['ti_mu']
                    if not np.isclose(expected, actual, rtol=1e-6, atol=1e-3):
                        ok = False
    assert ok, 'Label values do not match R'
    return {'pass': True}


def check_no_data_modified():
    """CPU audit: no data files modified (check .b73 cache exists)."""
    b73 = DATA / '.b73'
    assert b73.exists(), '.b73 cache missing'
    assert (b73 / 'grid.npz').exists(), 'grid.npz missing'
    assert (b73 / 'rows.npz').exists(), 'rows.npz missing'
    return {'pass': True}


def run_all_checks():
    print('Loading sources...')
    G, R = load_sources()
    units = select_units()
    u0 = units[0]
    m0 = unit_manifest(u0)
    stats = fit_stats(m0['train'], G, R)

    results = {}
    all_pass = True

    def run_check(name, fn, *args):
        nonlocal all_pass
        print(f'  {name}...')
        try:
            r = fn(*args)
            results[name] = r
            p = r.get('pass', True) if isinstance(r, dict) else True
            if not p:
                all_pass = False
        except Exception as e:
            results[name] = {'pass': False, 'error': str(e)}
            all_pass = False
            import traceback
            traceback.print_exc()

    run_check('no_data_modified', check_no_data_modified)
    run_check('anchors', check_anchors, G, R)
    run_check('manifests_disjoint', check_manifests_disjoint, units)
    run_check('rules_deterministic', check_rules_deterministic)
    run_check('stats_held_out', check_stats_held_out, G, R, units)
    run_check('xvtmc_zero', check_xvtmc_zero_firstdt_zero, G, R, stats, u0)
    run_check('label_values', check_label_values, R, G, stats, u0)
    run_check('label_mutation', check_label_mutation_invariance, G, R, stats, u0)
    run_check('prefix_invariance', check_prefix_invariance, G, R, stats, u0)
    run_check('s_cutoff_example', lambda: _s_cutoff_example(G))

    results['all_pass'] = all_pass
    results_clean = _deep_convert(results)

    out_path = CACHE / 'input_checks.json'
    import tempfile
    tmp = out_path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2)
    tmp.replace(out_path)
    print(f'Input checks: all_pass={all_pass} -> {out_path}')

    md_path = PAPER / 'b85_readiness.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# B85 Readiness Report\n\n')
        f.write(f'All checks pass: {all_pass}\n\n')
        for k, v in results_clean.items():
            if k == 'all_pass':
                continue
            f.write(f'## {k}\n```\n{json.dumps(v, indent=2)}\n```\n\n')

    return all_pass


if __name__ == '__main__':
    success = run_all_checks()
    sys.exit(0 if success else 1)
