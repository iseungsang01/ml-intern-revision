"""Registered B115 diagnostics of frozen B114 predictions; no fitting/bootstrap.

Sources and all original 95% intervals remain unchanged. Diagnostics describe
cluster support/leverage and seed spread; they do not identify a unique cause
of interval width. Trace selection uses support counts only, never errors.
"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import json
import time
from pathlib import Path
import numpy as np
import common115 as C

RULES = ('D10', 'D50', 'D100', 'D300', 'D650', 'D1000', 'S', 'K10')
SEEDS = (1701, 1702, 1703)
PRIOR = C.ROOT / 'data/.b114'
PAPER = C.ROOT / 'docs/paper'


def bind(path, hashes):
    path = Path(path)
    hashes[path.relative_to(C.ROOT).as_posix()] = C.sha(path)


def checked_estimates(tag, rule, family, selection, hashes):
    root = PRIOR / family
    receipt_path = root / 'prediction_complete.json'
    receipt = C.read(receipt_path)
    assert receipt['all_pass'] and receipt['cells'] == 32
    assert receipt['selection'] == selection[family]
    assert receipt['selection_sha256'] == C.sha(PRIOR / 'selection.json')
    path = root / 'estimates' / (tag + '_' + rule + '.npz')
    rel = path.relative_to(root).as_posix()
    assert C.sha(path) == receipt['files'][rel]
    arr = C.npz(path)
    prefix = 'Cat' if family == 'cat' else 'NN'
    candidate = prefix + '_selected_ensemble'
    assert np.array_equal(arr[candidate], np.mean([arr[prefix + '_selected_s' + str(s)] for s in SEEDS], axis=0))
    assert all(len(arr[k]) == len(arr['truth']) for k in ('idx', 'cluster', 'discharge', candidate, 'GP_fast_boost'))
    assert np.unique(arr['idx']).size == len(arr['idx'])
    assert all(np.isfinite(arr[k]).all() for k in ('truth', candidate, 'GP_fast_boost'))
    bind(receipt_path, hashes)
    bind(path, hashes)
    return arr, candidate, prefix


def summarize(arr, candidate, prefix, original):
    truth = arr['truth'].astype(np.float64)
    candidate_sq = (arr[candidate].astype(np.float64) - truth) ** 2
    gp_sq = (arr['GP_fast_boost'].astype(np.float64) - truth) ** 2
    n = len(truth)
    assert n > 0 and gp_sq.mean() > 0
    mse_c, mse_g = float(candidate_sq.mean()), float(gp_sq.mean())
    point = float(1 - mse_c / mse_g)
    assert abs(point - original['pairs'][candidate + '_over_GP_fast_boost']['point']) <= 1e-12
    assert np.isclose(mse_c, original['models'][candidate]['MSE'], atol=1e-7, rtol=0)
    assert np.isclose(mse_g, original['models']['GP_fast_boost']['MSE'], atol=1e-7, rtol=0)
    assert n == original['rows']
    groups = {}
    for grouping in ('cluster', 'discharge'):
        labels, inv = np.unique(arr[grouping], return_inverse=True)
        counts = np.bincount(inv)
        sums_c = np.bincount(inv, weights=candidate_sq)
        sums_g = np.bincount(inv, weights=gp_sq)
        gain = (sums_g - sums_c) / n
        assert np.isclose(gain.sum(), mse_g - mse_c, atol=1e-7, rtol=1e-12)
        assert len(labels) >= 2 and (n - counts > 0).all()
        remain_g = gp_sq.sum() - sums_g
        remain_c = candidate_sq.sum() - sums_c
        assert (remain_g > 0).all()
        leave_one_out = 1 - remain_c / remain_g
        groups[grouping] = {
            'groups': int(len(labels)), 'group_size_min': int(counts.min()),
            'group_size_median': float(np.median(counts)), 'group_size_max': int(counts.max()),
            'MSE_gain_contribution_min_ev2': float(gain.min()),
            'MSE_gain_contribution_max_ev2': float(gain.max()),
            'leave_one_group_out_skill_min': float(leave_one_out.min()),
            'leave_one_group_out_skill_max': float(leave_one_out.max()),
            'details': [{'anonymous_group': int(label), 'rows': int(count),
                         'MSE_gain_contribution_ev2': float(value),
                         'leave_one_group_out_skill': float(loco)}
                        for label, count, value, loco in zip(labels, counts, gain, leave_one_out)]}
        expected = original['clusters' if grouping == 'cluster' else 'discharges']
        assert len(labels) == expected
    seeds = {str(s): float(1 - np.mean((arr[prefix + '_selected_s' + str(s)] - truth) ** 2) / mse_g)
             for s in SEEDS}
    pair = original['pairs'][candidate + '_over_GP_fast_boost']
    return {'rows': n, 'candidate': candidate, 'MSE_candidate_ev2': mse_c,
            'MSE_GP_fast_ev2': mse_g, 'skill': point,
            'original_cluster_ci95': pair['cluster']['ci95'],
            'original_discharge_ci95': pair['discharge']['ci95'],
            'seed_skill_points': seeds, 'seed_skill_min': min(seeds.values()),
            'seed_skill_max': max(seeds.values()), 'groups': groups}


def figures(cells, units, arrays, row_inverse, hashes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    names = list(units.values())
    rows = [cells[name + '/S/F'] for name in names]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.15, 1]})
    for i, (name, item) in enumerate(zip(names, rows)):
        p = item['skill'] * 100
        for interval, offset, color, label in [
                ('original_cluster_ci95', -0.10, '#135a91', 'shot 군집 95% 구간'),
                ('original_discharge_ci95', 0.10, '#777777', '방전 95% 구간')]:
            lo, hi = np.asarray(item[interval]) * 100
            yy = i + offset
            ax.hlines(yy, lo, hi, color=color, lw=2)
            ax.vlines([lo, hi], yy - .04, yy + .04, color=color, lw=1.2)
            ax.plot(p, yy, 'o', color=color, ms=6, label=label if i == 0 else None)
        lo, hi = np.asarray(item['original_cluster_ci95']) * 100
        ax.annotate(f'{p:+.2f}% [{lo:+.2f}, {hi:+.2f}]', (p, i), xytext=(0, -25),
                    textcoords='offset points', ha='center', fontsize=10)
        details = item['groups']['cluster']['details']
        gain = np.array([d['MSE_gain_contribution_ev2'] for d in details])
        counts = np.array([d['rows'] for d in details])
        order = np.argsort(gain, kind='stable')
        yy = i + np.linspace(-0.15, 0.15, len(order))
        size = 18 + 100 * counts[order] / counts.max()
        bx.scatter(gain[order], yy, s=size, c=np.where(gain[order] >= 0, '#2878b5', '#c64742'),
                   alpha=.70, edgecolors='white', linewidths=.4)
    labels = [f"{name}\n{item['rows']:,}행 · {item['groups']['cluster']['groups']}군집 · {item['groups']['discharge']['groups']}방전"
              for name, item in zip(names, rows)]
    ax.set_yticks(range(4), labels, fontsize=10)
    ax.set_ylim(3.55, -.55)
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_xlabel('GP+fast 대비 MSE 감소율 (%)\n양수: 후보 오차 감소 / 음수: 후보 오차 증가')
    ax.set_title('A. S 조건: 점추정과 기존 95% 구간', pad=14)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, fontsize=9)
    bx.set_yticks(range(4), names)
    bx.set_ylim(3.55, -.55)
    bx.axvline(0, color='black', lw=1, ls='--')
    bx.set_xlabel('군집별 전체 MSE 감소 기여 (eV²)\n각 점의 합 = MSE(GP+fast) − MSE(후보)')
    bx.set_title('B. 군집별 기여: 각 점은 군집 하나\n점 크기는 행 수, 가로 위치는 부호 있는 기여', pad=14)
    for axis in (ax, bx):
        axis.grid(axis='x', alpha=.18)
    fig.suptitle('행 수만으로 불확실성의 원인을 단정하지 않고, 군집 간 변동을 직접 확인', fontsize=15, y=.98)
    fig.text(.5, .015, 'B114의 고정 VAL 선택 · 원래 구간 그대로 · 새 bootstrap/적합 없음 · S: 최근 notch 직전 CES까지 가용, 분석시간 0 가정',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .93])
    path = PAPER / 'b115_evidence.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    bind(path, hashes)

    # This sample is chosen by support only: most S rows, anonymous-ID tie break.
    tag = next(iter(units)); name = units[tag]
    arr, candidate = arrays[tag]
    labels, counts = np.unique(arr['discharge'], return_counts=True)
    selected = int(min(zip(-counts, labels))[1])
    mask = arr['discharge'] == selected
    assert mask.sum() >= 2
    idx = arr['idx'][mask].astype(np.int64)
    source_rows = row_inverse[idx]
    order = np.argsort(source_rows, kind='stable')
    assert len(np.unique(source_rows)) == len(source_rows)
    xx = source_rows[order] - source_rows.min()
    y = arr['truth'][mask][order]
    gp = arr['GP_fast_boost'][mask][order]
    pred = arr[candidate][mask][order]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    axes[0].plot(xx, y, 'o', color='#222222', ms=3, alpha=.65, label='CES 보고값 F')
    axes[0].plot(xx, gp, color='#1765a0', lw=1.2, alpha=.9, label='GP+fast')
    axes[0].plot(xx, pred, color='#d87816', lw=1.2, alpha=.9, label='VAL 선택 후보')
    axes[0].set_ylabel('보고값 / 추정값 (eV)')
    axes[0].legend(ncol=3, loc='upper right')
    axes[0].set_title(f'{name} / S: 지원 행 수가 가장 많은 익명 방전 {selected}\nTEST 오차로 선택하지 않은 예시 · 평활화 없음')
    axes[1].plot(xx, gp - y, color='#1765a0', lw=1, label='GP+fast − 보고값')
    axes[1].plot(xx, pred - y, color='#d87816', lw=1, label='후보 − 보고값')
    axes[1].axhline(0, color='black', lw=.7)
    axes[1].set_ylabel('오차 (eV)')
    axes[1].set_xlabel('방전의 원본 행 위치 (시간순; 실제 시간 간격을 뜻하지 않음)')
    for axis in axes:
        axis.grid(alpha=.15)
    fig.tight_layout()
    path = PAPER / 'b115_trace.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    bind(path, hashes)
    return {'unit': name, 'rule': 'S', 'anonymous_discharge': selected,
            'rows': int(mask.sum()), 'selection': 'maximum support rows; lowest anonymous ID tie-break',
            'order': 'B111 private row_inverse[idx], increasing original source-row order',
            'x_axis': 'source row offset, not elapsed time', 'error_based_selection': False}


def main():
    sources = C.source_guard()
    manifest = C.verify_inputs()
    destination = C.OUT / 'diagnostics'
    destination.mkdir(parents=True, exist_ok=True)
    assert not (destination / 'complete.json').exists(), 'Preserve completed diagnostics'
    hashes = {}
    report_path = PAPER / 'b114_comparison.json'
    original = C.read(report_path)
    selection_path = PRIOR / 'selection.json'
    selection = C.read(selection_path)
    audit_path = PRIOR / 'audit.json'
    assert original['independent_audit'] == C.read(audit_path)
    assert original['independent_audit']['all_pass'] and original['selection'] == selection
    assert original['independent_audit']['selection_sha256'] == C.sha(selection_path)
    for rel, digest in original['independent_audit']['sources'].items():
        assert C.sha(C.ROOT / rel) == digest
    for path in (report_path, selection_path, audit_path):
        bind(path, hashes)
    cells, arrays = {}, {}
    units = {tag: unit['name'] for tag, unit in manifest['units'].items()}
    assert len(units) == 4
    for tag, unit in units.items():
        family = selection['overall_family_by_unit'][tag]
        assert family == min(('cat', 'neural'), key=lambda f: (selection[f]['by_unit'][tag]['score'], f))
        for rule in RULES:
            arr, name, prefix = checked_estimates(tag, rule, family, selection, hashes)
            source = C.npz(C.ROOT / 'data/colab_b112/bundle' / (tag + '_' + rule + '.npz'))
            test = source['split'] == 2
            for k, old in [('idx', 'idx'), ('truth', 'y'), ('cluster', 'cluster'), ('discharge', 'discharge')]:
                assert np.array_equal(arr[k], source[old][test])
            key = unit + '/' + rule + '/F'
            assert key in original['cells']
            cells[key] = summarize(arr, name, prefix, original['cells'][key])
            cells[key]['selected_family'] = family
            cells[key]['selected_configuration'] = selection[family]['by_unit'][tag]
            if rule == 'S':
                arrays[tag] = arr, name
    assert len(cells) == 32
    private_path = C.ROOT / 'data/colab_b112/private_map.json'
    mapping_path = C.ROOT / 'data/colab_b111/private_map.json'
    private = C.read(private_path)
    assert private['mapping_source_sha256'] == C.sha(mapping_path)
    assert manifest['mapping_source_sha256'] == C.sha(mapping_path)
    mapping = C.read(mapping_path)
    row_inverse = np.asarray(mapping['row_inverse'], dtype=np.int64)
    row_codes = np.asarray(mapping['row_codes'], dtype=np.int64)
    assert np.array_equal(row_codes[row_inverse], np.arange(len(row_inverse)))
    for path in (private_path, mapping_path):
        bind(path, hashes)
    trace = figures(cells, units, arrays, row_inverse, hashes)
    result = {'registration': 'PREREGISTRATION_B115.md', 'sources': sources,
              'source_hashes': hashes, 'cells': cells, 'trace': trace,
              'new_fits': 0, 'new_predictions': 0, 'new_bootstrap_intervals': 0,
              'intervals': 'unaltered B114 pointwise 95% shot-cluster and discharge intervals',
              'limits': ['Cluster count, support imbalance, cluster effect heterogeneity and initialization variation are distinct.',
                         'Leave-one-group-out ranges are descriptive sensitivity ranges, not confidence intervals.',
                         'Seed ranges are initialization dispersion, not independent data or a replacement interval.',
                         'No unique cause of interval width or inevitable benefit of additional fitting is identified.',
                         'Reused TEST exploratory diagnostics; no error-based trace/cluster selection.']}
    C.save(PAPER / 'b115_uncertainty.json', result)
    lines = ['# B115: 기존 구간과 군집별 오차의 직접 확인', '',
             '기존 B114의 각 설계 내부 VAL 선택을 고정하고 같은 예측을 읽었다. 원래 95% 구간을 바꾸지 않았고 새 bootstrap·적합·추론은 없다.', '',
             '단순히 행 수가 많거나 구간 하한이 0에 가깝다는 이유만으로 표본 수 부족이 유일한 원인이라 단정할 수 없다. 행 수, 군집 수, 군집 기여의 불균형, 군집 사이 차이, 초기화 변동을 나누어 보여준다.', '',
             '그림: b115_evidence.png (S의 네 설계 forest/군집 기여), b115_trace.png (지원 행 수 기준으로 고른 실제 방전 예시).', '',
             '| 단위/규칙/F | 행 수 | 군집 수 | 군집 행 수 최소–최대 | MSE 감소율 [원래95%군집구간] | 군집 하나 제외 범위 | 초기화별 점추정 범위 |',
             '|---|---:|---:|---|---|---|---|']
    for key, item in cells.items():
        group = item['groups']['cluster'];lo, hi = item['original_cluster_ci95']
        lines.append(f"| {key} | {item['rows']} | {group['groups']} | {group['group_size_min']}–{group['group_size_max']} | {100*item['skill']:+.3f}% [{100*lo:+.3f}, {100*hi:+.3f}] | [{100*group['leave_one_group_out_skill_min']:+.3f}, {100*group['leave_one_group_out_skill_max']:+.3f}] | [{100*item['seed_skill_min']:+.3f}, {100*item['seed_skill_max']:+.3f}] |")
    lines += ['', '군집 하나 제외 범위와 초기화 범위는 신뢰구간이 아니다. 모델 개선으로 군집별 큰 오차와 초기화 변동을 줄일 수 있는지는 별도 학습·검증의 질문이다. 독립 평가 군집 수가 늘어난 것으로 취급하지 않는다.', '',
              '원천 해시는 JSON source_hashes에 보존했다. 실제 보고값 trace의 가로축은 시간순 원본 행 위치이며 ms 축이 아니다. 진짜 온도·물리적 진실로 해석하지 않는다.', '',
              '재생성: py ces_prediction/experiments/b115_budget/diagnose115.py (등록·소스 커밋 뒤 일회 실행)', '']
    (PAPER / 'b115_uncertainty.md').write_text('\n'.join(lines), encoding='utf8')
    C.save(destination / 'complete.json', {'all_pass': True, 'cells': 32, 'completed': time.time(),
           'sources': sources, 'report_sha256': C.sha(PAPER / 'b115_uncertainty.json'),
           'outputs': {p.name: C.sha(p) for p in (PAPER/'b115_uncertainty.md', PAPER/'b115_evidence.png', PAPER/'b115_trace.png')},
           'new_fits': 0, 'new_bootstrap_intervals': 0})
    print('B115 diagnostics complete: original intervals unchanged, no fits or new bootstrap')


if __name__ == '__main__':
    main()
