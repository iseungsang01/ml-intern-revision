"""B120 operational adapter: every descendant subprocess is hidden on Windows."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WRAPPER = Path(__file__).resolve()
OriginalPopen = subprocess.Popen


class HiddenPopen(OriginalPopen):
    def __init__(self, args, *pos, **kwargs):
        kwargs['creationflags'] = kwargs.get('creationflags', 0) | getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        if isinstance(args, (list, tuple)):
            args = list(args)
            # Install this same policy before each frozen scientific child's imports.
            for i, value in enumerate(args):
                path = Path(str(value))
                if path.name in ('train120.py', 'audit120.py') and path.parent.name == 'b120_tcn_focus':
                    args[i:i+1] = [str(WRAPPER), '--child', str(path)]
                    break
        super().__init__(args, *pos, **kwargs)


subprocess.Popen = HiddenPopen


def guard():
    import hashlib
    paths = [WRAPPER, HERE.parent/'PREREGISTRATION_B120_ADDENDUM1.md']
    hashes = {}
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        expected = subprocess.check_output(['git','show','HEAD:'+rel], cwd=ROOT)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == hashlib.sha256(expected).hexdigest()
        hashes[rel] = h
    return hashes


def main():
    import json
    import time
    sys.path.insert(0, str(HERE.parent/'b120_tcn_focus'))
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        import runpy
        target = Path(sys.argv[2]).resolve()
        assert target.parent == HERE.parent/'b120_tcn_focus'
        assert target.name in ('train120.py','audit120.py')
        guard()
        sys.argv = [str(target), *sys.argv[3:]]
        runpy.run_path(str(target), run_name='__main__')
        return
    import common120 as C
    import queue120 as Q
    op = guard()
    Q.W = C.OUT/'workflow/background1'
    Q.W.mkdir(parents=True, exist_ok=True)
    original = C.read(C.OUT/'workflow/started.json')
    assert not C.FIT.exists(), 'No automatic scientific retry'
    with (Q.W/'started.json').open('x', encoding='utf8') as f:
        json.dump({'pid':os.getpid(), 'original_started':original['started'], 'started':time.time(),
                   'operational_sources':op, 'scientific_sources':C.sources(),
                   'prior_optimizer_steps':0, 'all_descendants_hidden':True}, f)
    C.prepared()
    end = original['started']+12*3600
    idle = 0
    while time.time() < end:
        owners = Q.gpu_owners()
        raw = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
        usage = [[int(v.strip()) for v in row.split(',')] for row in raw.strip().splitlines()]
        assert usage
        idle = idle+1 if not owners and all(m<1024 and u<10 for m,u in usage) else 0
        Q.status('waiting_for_gpu', owners=owners, gpus=usage, idle_checks=idle, all_descendants_hidden=True)
        if idle >= 3: break
        time.sleep(20)
    else:
        raise RuntimeError('Original GPU wait limit reached')
    Q.stage('qualify','train',time.time()+600,'--qualify')
    Q.stage('fit','train',time.time()+C.FIT_SECONDS+1800)
    Q.stage('audit','audit',time.time()+3600)
    assert C.read(C.OUT/'audit.json')['all_pass'] and guard() == op
    C.save(Q.W/'operational_audit.json', {'all_pass':True, 'operational_sources':op,
           'original_wait_clock_preserved':True, 'scientific_sources_unchanged':True,
           'original_started_sha256':C.sha(C.OUT/'workflow/started.json')})
    Q.status('complete', converged=C.read(C.OUT/'audit.json')['converged'], TEST_accessed=False)
    files = ['docs/paper/b120_status.md','docs/paper/b120_training.md','docs/paper/b120_training.json',
             'docs/paper/b120_learning_curves.png']
    for cmd in (['git','diff','--check','--',*files],['git','add','--',*files],
                ['git','commit','--only','-m','Publish audited B120 focused TCN training diagnostic','--',*files],
                ['git','push','origin','HEAD']):
        subprocess.run(cmd, cwd=ROOT, check=True)
    C.save(Q.W/'closeout.json', {'all_pass':True, 'completed':time.time()})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if '--child' not in sys.argv:
            import traceback
            sys.path.insert(0, str(HERE.parent/'b120_tcn_focus'))
            import queue120 as Q
            import common120 as C
            Q.W = C.OUT/'workflow/background1'
            Q.W.mkdir(parents=True, exist_ok=True)
            started = Q.W/'started.json'
            if not started.exists() or C.read(started)['pid'] == os.getpid():
                Q.status('failed', error=repr(error), traceback=traceback.format_exc(), no_retry=True)
        raise
