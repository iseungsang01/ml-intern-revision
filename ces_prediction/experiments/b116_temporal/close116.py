"""Publish only independently audited staged B116 outputs; no fitting or scoring."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import common116 as C

PY = Path(sys.executable).with_name('python.exe')
OUTPUTS = {'readout.json': 'docs/paper/b116_comparison.json',
           'readout.md': 'docs/paper/b116_comparison.md',
           'figures/b116_average.png': 'docs/paper/b116_average.png',
           'figures/b116_failure.png': 'docs/paper/b116_failure.png'}


def write(path, text):
    path.write_text(text, encoding='utf8', newline='\n')


def main():
    sources = C.source_guard()
    C.verify_inputs()
    audit = C.read(C.OUT / 'audit.json')
    staged = C.read(C.OUT / 'readout_complete.json')
    assert audit['all_pass'] and staged['all_pass'] and staged['published'] is False
    assert audit['sources'] == sources
    assert audit['selection_sha256'] == C.sha(C.OUT / 'selection.json')
    assert audit['readout_sha256'] == C.sha(C.OUT / 'readout.json')
    assert set(OUTPUTS) <= set(audit['staged_readout'])
    for rel, expected in audit['staged_readout'].items():
        path = (C.OUT / rel).resolve()
        assert path.is_relative_to(C.OUT.resolve()) and C.sha(path) == expected
    for rel in OUTPUTS:
        assert staged['files'][rel] == audit['staged_readout'][rel] == C.sha(C.OUT / rel)
    with (C.OUT / 'workflow/closeout_started.json').open('x', encoding='utf8') as stream:
        json.dump({'started': time.time(), 'audit_sha256': C.sha(C.OUT / 'audit.json')}, stream)
    for source, target in OUTPUTS.items():
        shutil.copyfile(C.OUT / source, C.ROOT / target)
        assert C.sha(C.ROOT / target) == audit['staged_readout'][source]
    audit_public = C.ROOT / 'docs/paper/b116_audit.json'
    shutil.copyfile(C.OUT / 'audit.json', audit_public)
    assert C.sha(audit_public) == C.sha(C.OUT / 'audit.json')
    paragraph = (
        '동일한 실제 시계열 입력으로 순환 구조와 관측 도착 시 갱신 구조를 비교했다. '
        '구조 선택에는 검증 구간만 사용했으며, 비교 기준은 모든 구간의 개별 우세가 아니라 '
        '검증 설계와 가용성 조건을 동일 가중한 평균 오차다. '
        '현재 관측이 있는 목표만 평가하고 원래의 군집과 방전 식별자를 유지했다. '
        '관측 나이와 빠른 진단 변화량 등 입력에서 정한 구간의 실패 특성도 보고했다. '
        '점추정, 구간 및 영 제외 여부는 독립 검증을 마친 비교 기록에서 확인한다. '
        '재사용한 평가 자료에 대한 탐색 결과이며 학습을 다시 수행할 때의 불확실성을 포함하지 않는다.\n')
    text = 'B116 COMPLETE / independently audited. Temporal architecture comparison and input-conditioned failure analysis published. No additional fits queued.'
    p = C.ROOT / 'PROJECT_KNOWLEDGE.md'
    s = p.read_text(encoding='utf8')
    assert '### 4.84 ' not in s
    marker = '## 5. '
    assert marker in s
    at = s.index(marker)
    write(p, s[:at] + '### 4.84 실제 시계열 구조와 평균 오차 비교 (B.116; `docs/paper/b116_comparison.md`)\n\n'
          + paragraph + '\n' + s[at:])
    p = C.ROOT / 'THESIS_RESULTS.md'
    write(p, p.read_text(encoding='utf8') + '\n\n## B116 temporal architecture comparison\n\n' + text + '\n\n' + paragraph)
    p = C.ROOT / 'HANDOVER.md'
    s = p.read_text(encoding='utf8')
    at = s.index('\n') + 1
    write(p, s[:at] + '\n**' + text + '**\n\n' + paragraph + s[at:])
    for rel in ('README.md', 'ces_prediction/experiments/README.md'):
        p = C.ROOT / rel
        s = p.read_text(encoding='utf8')
        # Parent prepares this exact marker before source freeze.
        assert 'B116 PREPARED' in s, 'Missing prepared wiring marker: ' + rel
        write(p, s.replace('B116 PREPARED', 'B116 COMPLETE / independently audited'))
    write(C.ROOT / 'docs/paper/b116_status.md', '# B116 complete\n\n' + text + '\n\n' + paragraph)
    p = C.ROOT / 'ces_prediction/check_records.py'
    s = p.read_text(encoding='utf8')
    if '"docs/paper/b116_comparison.md"' not in s:
        assert 'SOURCES = (' in s
        write(p, s.replace('SOURCES = (', 'SOURCES = (\n    "docs/paper/b116_comparison.md",', 1))
    subprocess.run([str(PY), 'ces_prediction/check_records.py'], cwd=C.ROOT, check=True)
    subprocess.run([str(PY), '-m', 'pytest', '-q'], cwd=C.ROOT, check=True)
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=C.ROOT, text=True).strip() == 'main'
    files = ['README.md', 'HANDOVER.md', 'THESIS_RESULTS.md', 'PROJECT_KNOWLEDGE.md',
             'ces_prediction/check_records.py', 'ces_prediction/experiments/README.md',
             'docs/paper/b116_status.md', 'docs/paper/b116_audit.json', *OUTPUTS.values()]
    subprocess.run(['git', 'diff', '--check', '--', *files], cwd=C.ROOT, check=True)
    subprocess.run(['git', 'add', '--', *files], cwd=C.ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m',
                    'Publish independently audited B116 temporal comparison', '--', *files], cwd=C.ROOT, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C.ROOT, text=True).strip()
    subprocess.run(['git', 'push', 'origin', 'main'], cwd=C.ROOT, check=True)
    C.save(C.OUT / 'workflow/closeout.json', {'all_pass': True, 'commit': commit,
        'completed': time.time(), 'audit_sha256': C.sha(C.OUT / 'audit.json'),
        'published': {target: C.sha(C.ROOT / target) for target in
                      (*OUTPUTS.values(), 'docs/paper/b116_audit.json')}})


if __name__ == '__main__':
    main()
