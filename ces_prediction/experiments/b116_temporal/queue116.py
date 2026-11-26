"""Single-start hidden B116 queue, fixed wall-clock boundaries, no retries."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import common116 as C

W = C.OUT / 'workflow'
PY = Path(sys.executable).with_name('python.exe')
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def status(state, **extra):
    C.save(W / 'status.json', {'state': state, 'updated': time.time(), **extra})
    if extra.get('phase') == 'closeout':
        return
    (C.ROOT / 'docs/paper/b116_status.md').write_text(
        '# B116 status\n\n' + state + '\n\n' + json.dumps(extra, indent=2)
        + '\n\nLive record: data/.b116/workflow/status.json. No retries or duplicate launch.\n',
        encoding='utf8')


def stage(label, args, env, deadline):
    assert time.time() < deadline, 'Registered boundary reached before ' + label
    with (W / (label + '.stdout.log')).open('xb') as out, (W / (label + '.stderr.log')).open('xb') as err:
        child = subprocess.Popen([str(PY), '-u', *map(str, args)], cwd=C.ROOT,
            env=env, stdout=out, stderr=err, creationflags=FLAGS)
        while child.poll() is None:
            status('running', phase=label, child_pid=child.pid, deadline=deadline)
            if time.time() >= deadline:
                child.kill()
                child.wait()
                raise RuntimeError('Owned child exceeded registered boundary: ' + label)
            time.sleep(10)
    C.save(W / (label + '_exit.json'), {'returncode': child.returncode, 'completed': time.time()})
    assert child.returncode == 0, label + ' failed; partials preserved, no retry'


def wait_idle(deadline):
    while time.time() < deadline:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True,
            creationflags=FLAGS)
        values = [[int(v.strip()) for v in row.split(',')] for row in result.stdout.strip().splitlines()]
        assert values, 'GPU inventory missing'
        C.save(W / 'gpu_wait.json', {'updated': time.time(), 'gpus': values, 'GPU_end': deadline})
        # Every device must be idle; never terminate another process or reset GPU.
        if all(used < 1024 and util < 10 for used, util in values):
            return
        status('waiting_for_gpu', phase='wait', deadline=deadline, gpus=values)
        time.sleep(min(30, max(0, deadline - time.time())))
    raise RuntimeError('GPU remained occupied until original registered boundary')


def main():
    W.mkdir(parents=True, exist_ok=True)
    sources = C.source_guard()
    start = time.time()
    gpu_end, cpu_end = start + 12 * 3600, start + 14 * 3600
    with (W / 'started.json').open('x', encoding='utf8') as stream:
        json.dump({'pid': os.getpid(), 'started': start, 'sources': sources,
                   'GPU_end': gpu_end, 'CPU_end': cpu_end}, stream)
    C.verify_inputs()
    import data116 as D
    C.save(W / 'cluster_contract.json', D.check_cluster_contract())
    env = {k: v for k, v in os.environ.items() if not k.startswith('CES_')}
    env.update(C.ENV_PINS)
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    wait_idle(gpu_end)
    stage('smoke', [C.HERE / 'train116.py', '--smoke'], env, gpu_end)
    stage('smoke_audit', [C.HERE / 'audit116.py', '--smoke'], env, gpu_end)
    qualified = C.read(C.OUT / 'smoke_independent.json')
    assert qualified['all_pass'] and qualified['sources'] == sources
    assert qualified['smoke_sha256'] == C.sha(C.OUT / 'smoke.json')
    stage('fit', [C.HERE / 'train116.py', '--fit'], env, gpu_end)
    stage('select', [C.HERE / 'select116.py'], env, gpu_end)
    stage('predict', [C.HERE / 'train116.py', '--predict'], env, gpu_end)
    stage('readout', [C.HERE / 'readout116.py'], env, cpu_end)
    stage('audit', [C.HERE / 'audit116.py'], env, cpu_end)
    assert C.read(C.OUT / 'audit.json')['all_pass']
    stage('closeout', [C.HERE / 'close116.py'], env, cpu_end)
    C.save(W / 'status.json', {'state': 'complete', 'updated': time.time(),
        'independent_audit': True, 'commit': C.read(W / 'closeout.json')['commit']})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        W.mkdir(parents=True, exist_ok=True)
        # Never overwrite a finished publication status during closeout failure.
        current = C.read(W / 'status.json') if (W / 'status.json').exists() else {}
        status('failed', phase=current.get('phase'), error=repr(error),
               traceback=traceback.format_exc(), no_retry=True)
        raise
