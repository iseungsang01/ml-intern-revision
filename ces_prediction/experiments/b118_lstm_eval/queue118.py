"""Exclusive hidden forward-only workflow; no optimizer or additional training."""
import json
import os
import time
import traceback
import common118 as C
import queue116 as Q

W = C.OUT / 'workflow'


def status(state, **extra):
    C.save(W / 'status.json', {'state': state, 'updated': time.time(), **extra})
    if extra.get('phase') != 'closeout':
        (C.ROOT / 'docs/paper/b118_status.md').write_text('# B118 fixed-model evaluation\n\n'
            + state + '\n\n' + json.dumps(extra, indent=2)
            + '\n\nLive: data/.b118/workflow/status.json. No additional fits.\n', encoding='utf8')


def main():
    W.mkdir(parents=True, exist_ok=True)
    sources, inputs = C.source_guard(), C.verify_inputs()
    start = time.time()
    with (W / 'started.json').open('x', encoding='utf8') as stream:
        json.dump({'started': start, 'pid': os.getpid(), 'sources': sources, 'inputs': inputs,
            'GPU_end': start + 2 * 3600, 'CPU_end': start + 4 * 3600, 'optimization_steps': 0}, stream)
    Q.W, Q.status = W, status
    Q.wait_idle(start + 2 * 3600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CES_')}
    env.update(C.ENV_PINS)
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    Q.stage('predict', [C.HERE / 'predict118.py'], env, start + 2 * 3600)
    for label, file in (('readout', 'readout118.py'), ('audit', 'audit118.py'), ('closeout', 'close118.py')):
        Q.stage(label, [C.HERE / file], env, start + 4 * 3600)
    C.save(W / 'status.json', {'state': 'complete', 'updated': time.time(),
        'optimization_steps': 0, 'commit': C.read(W / 'closeout.json')['commit']})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        W.mkdir(parents=True, exist_ok=True)
        current = C.read(W / 'status.json') if (W / 'status.json').exists() else {}
        status('failed', phase=current.get('phase'), error=repr(error),
               traceback=traceback.format_exc(), no_retry=True)
        raise
