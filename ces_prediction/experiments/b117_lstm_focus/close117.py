"""Publish the independently audited single-fit TRAIN/VAL diagnostic."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import common117 as C

PY = Path(sys.executable).with_name('python.exe')
OUTPUTS = {'readout.json': 'docs/paper/b117_training.json',
           'readout.md': 'docs/paper/b117_training.md',
           'figures/b117_learning_curves.png': 'docs/paper/b117_learning_curves.png'}


def write(path, value):
    path.write_text(value, encoding='utf8', newline='\n')


def main():
    sources = C.source_guard()
    C.verify_inputs()
    audit = C.read(C.OUT / 'audit.json')
    staged = C.read(C.OUT / 'readout_complete.json')
    report = C.read(C.OUT / 'readout.json')
    assert audit['all_pass'] and staged['all_pass'] and staged['published'] is False
    assert audit['sources'] == report['sources'] == sources
    assert report['TEST_accessed'] is False
    assert audit['readout_sha256'] == C.sha(C.OUT / 'readout.json')
    assert set(OUTPUTS) <= set(audit['staged_readout'])
    for rel, expected in audit['staged_readout'].items():
        path = (C.OUT / rel).resolve()
        assert path.is_relative_to(C.OUT.resolve()) and C.sha(path) == expected
    for rel in OUTPUTS:
        assert staged['files'][rel] == audit['staged_readout'][rel] == C.sha(C.OUT / rel)
    workflow = C.OUT / 'workflow'
    workflow.mkdir(exist_ok=True)
    with (workflow / 'closeout_started.json').open('x', encoding='utf8') as stream:
        json.dump({'started': time.time(), 'audit_sha256': C.sha(C.OUT / 'audit.json')}, stream)
    for source, target in OUTPUTS.items():
        shutil.copyfile(C.OUT / source, C.ROOT / target)
        assert C.sha(C.ROOT / target) == audit['staged_readout'][source]
    audit_target = 'docs/paper/b117_audit.json'
    shutil.copyfile(C.OUT / 'audit.json', C.ROOT / audit_target)
    assert C.sha(C.ROOT / audit_target) == C.sha(C.OUT / 'audit.json')
    status = 'CONVERGENCE CRITERION MET' if report['converged'] else 'NOT CONVERGED / safety boundary reached'
    text = ('B117 COMPLETE / independently audited. Single fixed LSTM TRAIN/VAL diagnostic; '
            + status + '. No TEST evaluation, new GP comparison or further fit queued.')
    paragraph = (
        '동일한 단일 LSTM을 처음부터 학습하여 기존 시간 제한 학습과 전체 학습 곡선, 검증 오차, '
        '학습률 및 갱신 횟수를 비교했다. 등록한 수렴 기준 충족 여부와 안전 한도 종료를 구분했다. '
        '학습 손실은 고정된 학습 표준편차로 물리 단위로 변환했다. 학습 구간은 규칙 혼합과 갱신 중 '
        '모수에 의존하고 검증은 고정 지지의 규칙별 평균이므로 두 손실의 차이는 기술적 진단이다. '
        '검증 자료를 반복 사용한 단일 학습이며 독립 표본 일반화, 신뢰구간 또는 TEST 성능을 '
        '주장하지 않는다. 수치와 수렴 판정은 독립 감사된 '
        '`docs/paper/b117_training.md`와 JSON 및 학습 곡선에서 확인한다.\n')
    p = C.ROOT / 'PROJECT_KNOWLEDGE.md'
    s = p.read_text(encoding='utf8')
    assert '### 4.85 ' not in s and '## 5. ' in s
    at = s.index('## 5. ')
    write(p, s[:at] + '### 4.85 단일 LSTM 학습 충분성 진단 (B.117; `docs/paper/b117_training.md`)\n\n'
          + paragraph + '\n' + s[at:])
    p = C.ROOT / 'THESIS_RESULTS.md'
    write(p, p.read_text(encoding='utf8') + '\n\n## B117 single LSTM training diagnostic\n\n' + text + '\n\n' + paragraph)
    p = C.ROOT / 'HANDOVER.md'
    s = p.read_text(encoding='utf8')
    at = s.index('\n') + 1
    write(p, s[:at] + '\n**' + text + '**\n\n' + paragraph + s[at:])
    for rel in ('README.md', 'ces_prediction/experiments/README.md'):
        p = C.ROOT / rel
        s = p.read_text(encoding='utf8')
        assert 'B117 PREPARED' in s, 'Missing prepared wiring marker: ' + rel
        write(p, s.replace('B117 PREPARED', 'B117 COMPLETE / independently audited'))
    write(C.ROOT / 'docs/paper/b117_status.md', '# B117 complete\n\n' + text + '\n\n' + paragraph)
    p = C.ROOT / 'ces_prediction/check_records.py'
    s = p.read_text(encoding='utf8')
    if '"docs/paper/b117_training.md"' not in s:
        assert 'SOURCES = (' in s
        write(p, s.replace('SOURCES = (', 'SOURCES = (\n    "docs/paper/b117_training.md",', 1))
    subprocess.run([str(PY), 'ces_prediction/check_records.py'], cwd=C.ROOT, check=True)
    subprocess.run([str(PY), '-m', 'pytest', '-q'], cwd=C.ROOT, check=True)
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=C.ROOT, text=True).strip() == 'main'
    files = ['README.md', 'HANDOVER.md', 'THESIS_RESULTS.md', 'PROJECT_KNOWLEDGE.md',
             'ces_prediction/check_records.py', 'ces_prediction/experiments/README.md',
             'docs/paper/b117_status.md', audit_target, *OUTPUTS.values()]
    subprocess.run(['git', 'diff', '--check', '--', *files], cwd=C.ROOT, check=True)
    subprocess.run(['git', 'add', '--', *files], cwd=C.ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'Publish audited B117 single LSTM training diagnostic',
                    '--', *files], cwd=C.ROOT, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C.ROOT, text=True).strip()
    subprocess.run(['git', 'push', 'origin', 'main'], cwd=C.ROOT, check=True)
    C.save(workflow / 'closeout.json', {'all_pass': True, 'commit': commit, 'completed': time.time(),
           'audit_sha256': C.sha(C.OUT / 'audit.json'),
           'published': {target: C.sha(C.ROOT / target) for target in (*OUTPUTS.values(), audit_target)}})


if __name__ == '__main__':
    main()
