"""One anchored-LSTM fit; independent VAL gate must pass before any TEST inference."""
import json
import os
import time
import traceback
import common119 as C
import queue116 as Q

W = C.OUT / 'workflow'


def status(state, **extra):
    C.save(W / 'status.json', {'state': state, 'updated': time.time(), **extra})
    if extra.get('phase') != 'closeout':
        (C.ROOT / 'docs/paper/b119_status.md').write_text('# B119 single anchored LSTM\n\n'
            + state + '\n\n' + json.dumps(extra, indent=2)
            + '\n\nLive: data/.b119/workflow/status.json and progress.json. No extra fit or automatic retry.\n', encoding='utf8')


def main():
    W.mkdir(parents=True, exist_ok=True)
    sources, inputs = C.source_guard(), C.verify_inputs()
    start = time.time()
    with (W / 'started.json').open('x', encoding='utf8') as stream:
        json.dump({'started': start, 'pid': os.getpid(), 'sources': sources, 'inputs': inputs,
            'GPU_end': start + 8 * 3600, 'CPU_end': start + 10 * 3600,
            'production_fits': 1}, stream)
    Q.W, Q.status = W, status
    Q.wait_idle(start + 8 * 3600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CES_')}
    env.update(C.ENV_PINS)
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    Q.stage('qualify', [C.HERE / 'train119.py', '--qualify'], env, start + 8 * 3600)
    Q.stage('smoke_audit', [C.HERE / 'audit119.py', '--smoke'], env, start + 8 * 3600)
    smoke = C.read(C.OUT / 'smoke_independent.json')
    assert smoke['all_pass'] and smoke['sources'] == sources
    assert smoke['smoke_sha256'] == C.sha(C.OUT / 'smoke.json')
    Q.stage('fit', [C.HERE / 'train119.py'], env, start + 8 * 3600)
    Q.stage('val_audit', [C.HERE / 'audit119.py', '--val'], env, start + 8 * 3600)
    gate, fit = (C.read(C.OUT / p) for p in ('val_audit.json', 'fit_complete.json'))
    assert gate['all_pass'] and gate['sources'] == fit['sources'] == sources
    assert gate['fit_complete_sha256'] == C.sha(C.OUT / 'fit_complete.json')
    if not fit['converged']:
        assert not (C.OUT / 'predict_started.json').exists() and not (C.OUT / 'test').exists()
        status('safety_capped_not_converged', phase='stopped_after_VAL', TEST_accessed=False,
               additional_fits=0, checkpoint_preserved=True)
        return
    Q.stage('predict', [C.HERE / 'predict119.py'], env, start + 8 * 3600)
    for label, file in (('readout', 'readout119.py'), ('audit', 'audit119.py'), ('closeout', 'close119.py')):
        Q.stage(label, [C.HERE / file], env, start + 10 * 3600)
    C.save(W / 'status.json', {'state': 'complete', 'updated': time.time(),
        'production_fits': 1, 'commit': C.read(W / 'closeout.json')['commit']})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        W.mkdir(parents=True, exist_ok=True)
        current = C.read(W / 'status.json') if (W / 'status.json').exists() else {}
        status('failed', phase=current.get('phase'), error=repr(error),
               traceback=traceback.format_exc(), no_retry=True)
        raise
