"""One-start hidden queue; process-aware GPU gate, no restart or TEST prediction."""
import json
import os
import subprocess
import sys
import time
import traceback
import common120 as C

W = C.OUT/'workflow'
PY = str(__import__('pathlib').Path(sys.executable).with_name('python.exe'))


def status(state, **extra):
    C.save(W/'status.json', {'state':state, 'updated':time.time(), **extra})
    (C.ROOT/'docs/paper/b120_status.md').write_text('# B120 TCN status\n\n'+state+'\n\n'
        +json.dumps(extra, indent=2)+'\n', encoding='utf8')


def gpu_owners():
    import psutil
    owners = []
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            if p.pid == os.getpid(): continue
            args = p.info['cmdline'] or []
            if any('ces_prediction' in a and 'experiments' in a and a.endswith('.py') for a in args):
                owners.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return owners


def stage(label, name, limit, *args):
    env = {k:v for k,v in os.environ.items() if not k.startswith('CES_')}
    env.update(C.ENV)
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (W/(label+'.stdout.log')).open('xb') as out, (W/(label+'.stderr.log')).open('xb') as err:
        child = subprocess.Popen([PY,'-u',str(C.HERE/(name+'120.py')),*args], cwd=C.ROOT,
            env=env, stdout=out, stderr=err, creationflags=C.FLAGS)
        while child.poll() is None:
            status('running', phase=label, child_pid=child.pid)
            if time.time() >= limit:
                child.kill(); child.wait()
                raise RuntimeError('Owned child exceeded registered stage safety limit')
            time.sleep(10)
    C.save(W/(label+'_exit.json'), {'returncode':child.returncode, 'updated':time.time()})
    assert child.returncode == 0, name+' failed; no retry'


def main():
    W.mkdir(parents=True, exist_ok=True)
    sources = C.sources()
    with (W/'started.json').open('x', encoding='utf8') as f:
        json.dump({'started':time.time(), 'pid':os.getpid(), 'sources':sources,
                   'max_wait_hours':12, 'production_fits':1, 'TEST_accessed':False}, f)
    C.prepared()
    end = time.time()+12*3600
    idle_count = 0
    while time.time() < end:
        owners = gpu_owners()
        r = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, creationflags=C.FLAGS)
        usage = [[int(v.strip()) for v in row.split(',')] for row in r.strip().splitlines()]
        assert usage
        idle_count = idle_count+1 if not owners and all(m<1024 and u<10 for m,u in usage) else 0
        status('waiting_for_gpu', owners=owners, gpus=usage, idle_checks=idle_count)
        if idle_count >= 3: break
        time.sleep(20)
    else:
        raise RuntimeError('GPU wait limit reached; no fit started')
    stage('qualify', 'train', time.time()+600, '--qualify')
    stage('fit', 'train', time.time()+C.FIT_SECONDS+1800)
    stage('audit', 'audit', time.time()+3600)
    assert C.read(C.OUT/'audit.json')['all_pass']
    status('complete', converged=C.read(C.OUT/'audit.json')['converged'], TEST_accessed=False)
    files = ['docs/paper/b120_status.md','docs/paper/b120_training.md','docs/paper/b120_training.json',
             'docs/paper/b120_learning_curves.png']
    for cmd in (['git','diff','--check','--',*files], ['git','add','--',*files],
                ['git','commit','--only','-m','Publish audited B120 focused TCN training diagnostic','--',*files],
                ['git','push','origin','HEAD']):
        subprocess.run(cmd, cwd=C.ROOT, check=True, creationflags=C.FLAGS)
    C.save(W/'closeout.json', {'all_pass':True, 'completed':time.time()})


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        W.mkdir(parents=True, exist_ok=True)
        if not (W/'started.json').exists() or C.read(W/'started.json')['pid'] == os.getpid():
            status('failed', error=repr(e), traceback=traceback.format_exc(), no_retry=True)
        raise
