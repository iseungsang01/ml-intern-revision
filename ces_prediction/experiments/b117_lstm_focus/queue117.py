"""One-start hidden B117 workflow; one production fit, no TEST prediction."""
import json
import os
import sys
import time
import traceback
from pathlib import Path
import common117 as C
import queue116 as Q

W = C.OUT / 'workflow'
PY = Path(sys.executable).with_name('python.exe')


def status(state, **extra):
    C.save(W / 'status.json', {'state': state, 'updated': time.time(), **extra})
    if extra.get('phase') != 'closeout':
        (C.ROOT / 'docs/paper/b117_status.md').write_text(
            '# B117 one-model learning study\n\n' + state + '\n\n'
            + json.dumps(extra, indent=2) + '\n\nLive: data/.b117_focused/progress.json and workflow/status.json.\n', encoding='utf8')


def main():
    W.mkdir(parents=True, exist_ok=True)
    sources = C.source_guard()
    start = time.time()
    with (W / 'started.json').open('x', encoding='utf8') as stream:
        json.dump({'started': start, 'pid': os.getpid(), 'sources': sources,
            'GPU_end': start + 8 * 3600, 'CPU_end': start + 9 * 3600,
            'production_fits': 1, 'TEST_accessed': False}, stream)
    C.verify_inputs()
    Q.W, Q.status = W, status
    Q.wait_idle(start + 8 * 3600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CES_')}
    env.update(C.ENV_PINS)
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    Q.stage('qualify', [C.HERE / 'train117.py', '--qualify'], env, start + 8 * 3600)
    Q.stage('smoke_audit', [C.HERE / 'audit117.py', '--smoke'], env, start + 8 * 3600)
    receipt = C.read(C.OUT / 'smoke_independent.json')
    assert receipt['all_pass'] and receipt['sources'] == sources
    assert receipt['smoke_sha256'] == C.sha(C.OUT / 'smoke.json')
    Q.stage('fit', [C.HERE / 'train117.py'], env, start + 8 * 3600)
    # An incomplete safety-boundary result is still analyzed and labeled honestly;
    # convergence is not fabricated from a zero process exit code.
    for name, file in (('analyze', 'analyze117.py'), ('audit', 'audit117.py'), ('closeout', 'close117.py')):
        Q.stage(name, [C.HERE / file], env, start + 9 * 3600)
    C.save(W / 'status.json', {'state': 'complete', 'updated': time.time(),
        'converged': C.read(C.OUT / 'fit_complete.json')['converged'],
        'commit': C.read(W / 'closeout.json')['commit']})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        W.mkdir(parents=True, exist_ok=True)
        current = C.read(W / 'status.json') if (W / 'status.json').exists() else {}
        status('failed', phase=current.get('phase'), error=repr(error),
               traceback=traceback.format_exc(), no_retry=True)
        raise
