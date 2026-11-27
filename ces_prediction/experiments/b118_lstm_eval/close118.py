"""Publish independently audited fixed-checkpoint evaluation; no numerical fitting."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import common118 as C

PY = Path(sys.executable).with_name('python.exe')
OUTPUTS = {'readout.json': 'docs/paper/b118_comparison.json',
           'readout.md': 'docs/paper/b118_comparison.md',
           'figures/b118_trace.png': 'docs/paper/b118_trace.png',
           'figures/b118_average.png': 'docs/paper/b118_average.png'}


def write(path, value):
    path.write_text(value, encoding='utf8', newline='\n')


def main():
    sources, inputs = C.source_guard(), C.verify_inputs()
    audit, report = (C.read(C.OUT / p) for p in ('audit.json', 'readout.json'))
    staged = C.read(C.OUT / 'readout_complete.json')
    assert audit['all_pass'] and staged['all_pass'] and staged['published'] is False
    assert audit['sources'] == report['sources'] == sources
    assert audit['readout_sha256'] == C.sha(C.OUT / 'readout.json')
    assert set(OUTPUTS) <= set(audit['staged_readout'])
    for rel, expected in audit['staged_readout'].items():
        path = (C.OUT / rel).resolve()
        assert path.is_relative_to(C.OUT.resolve()) and C.sha(path) == expected
    for rel in OUTPUTS:
        assert staged['files'][rel] == audit['staged_readout'][rel] == C.sha(C.OUT / rel)
    W = C.OUT / 'workflow'
    W.mkdir(exist_ok=True)
    with (W / 'closeout_started.json').open('x', encoding='utf8') as stream:
        json.dump({'started': time.time(), 'audit_sha256': C.sha(C.OUT / 'audit.json')}, stream)
    for source, target in OUTPUTS.items():
        shutil.copyfile(C.OUT / source, C.ROOT / target)
        assert C.sha(C.ROOT / target) == audit['staged_readout'][source]
    audit_target = 'docs/paper/b118_audit.json'
    shutil.copyfile(C.OUT / 'audit.json', C.ROOT / audit_target)
    assert C.sha(C.ROOT / audit_target) == C.sha(C.OUT / 'audit.json')
    text = ('B118 COMPLETE / independently audited. Fixed B117 LSTM checkpoint evaluated '
            'without additional fitting; exploratory reused TEST comparison. No further fits queued.')
    paragraph = (
        '충분 학습 진단에서 이미 선택한 단일 LSTM 체크포인트를 고정한 뒤 평가 자료에 전방 계산만 수행했다. '
        '같은 지지 행에서 기존 짧은 학습 모델 및 GP와 빠른 진단을 결합한 기준과 비교했다. '
        '조건별 결과와 동일 가중 평균의 점추정, 구간 및 영 제외 여부는 독립 감사한 비교 기록을 따른다. '
        '구간은 고정된 학습과 선택에 조건부이며 재학습 변동을 포함하지 않는다. '
        '평가 자료를 재사용한 탐색 결과이고 단일 검증 설계와 초기화에 한정된다. '
        '그림의 실패 구간 특성은 입력으로 고정한 구분에 따른 기술이며 새로운 구조나 선택에 사용하지 않았다. '
        '다른 검증 설계와 독립 관측에 대한 충분 학습 비교가 해석을 바꿀 수 있다.\n')
    p = C.ROOT / 'PROJECT_KNOWLEDGE.md'
    s = p.read_text(encoding='utf8')
    assert '### 4.86 ' not in s and '## 5. ' in s
    at = s.index('## 5. ')
    write(p, s[:at] + '### 4.86 고정한 충분 학습 LSTM 평가 (B.118; `docs/paper/b118_comparison.md`)\n\n'
          + paragraph + '\n' + s[at:])
    p = C.ROOT / 'THESIS_RESULTS.md'
    write(p, p.read_text(encoding='utf8') + '\n\n## B118 fixed LSTM evaluation\n\n' + text + '\n\n' + paragraph)
    p = C.ROOT / 'HANDOVER.md'
    s = p.read_text(encoding='utf8')
    at = s.index('\n') + 1
    write(p, s[:at] + '\n**' + text + '**\n\n' + paragraph + s[at:])
    for rel in ('README.md', 'ces_prediction/experiments/README.md'):
        p = C.ROOT / rel
        s = p.read_text(encoding='utf8')
        assert 'B118 PREPARED' in s, 'Missing prepared marker: ' + rel
        write(p, s.replace('B118 PREPARED', 'B118 COMPLETE / independently audited'))
    write(C.ROOT / 'docs/paper/b118_status.md', '# B118 complete\n\n' + text + '\n\n' + paragraph)
    p = C.ROOT / 'ces_prediction/check_records.py'
    s = p.read_text(encoding='utf8')
    if '"docs/paper/b118_comparison.md"' not in s:
        assert 'SOURCES = (' in s
        write(p, s.replace('SOURCES = (', 'SOURCES = (\n    "docs/paper/b118_comparison.md",', 1))
    subprocess.run([str(PY), 'ces_prediction/check_records.py'], cwd=C.ROOT, check=True)
    subprocess.run([str(PY), '-m', 'pytest', '-q'], cwd=C.ROOT, check=True)
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=C.ROOT, text=True).strip() == 'main'
    files = ['README.md', 'HANDOVER.md', 'THESIS_RESULTS.md', 'PROJECT_KNOWLEDGE.md',
        'ces_prediction/check_records.py', 'ces_prediction/experiments/README.md',
        'docs/paper/b118_status.md', audit_target, *OUTPUTS.values()]
    subprocess.run(['git', 'diff', '--check', '--', *files], cwd=C.ROOT, check=True)
    subprocess.run(['git', 'add', '--', *files], cwd=C.ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'Publish independently audited B118 fixed-model evaluation', '--', *files], cwd=C.ROOT, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C.ROOT, text=True).strip()
    subprocess.run(['git', 'push', 'origin', 'main'], cwd=C.ROOT, check=True)
    C.save(W / 'closeout.json', {'all_pass': True, 'commit': commit, 'completed': time.time(),
        'inputs': inputs, 'audit_sha256': C.sha(C.OUT / 'audit.json'),
        'published': {rel: C.sha(C.ROOT / rel) for rel in (*OUTPUTS.values(), audit_target)}})


if __name__ == '__main__':
    main()
