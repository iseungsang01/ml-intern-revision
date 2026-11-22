"""Serial registered follow-ups: clean B85 -> B89 runtime -> B88 -> B90.

No scheduler changes existing training. Failed phases stop; restarting is an
explicit operator action and each child validates its own reusable artifacts.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EXPERIMENTS=Path(__file__).resolve().parent
sys.path.insert(0,str(EXPERIMENTS/'b88_anchor'))
import common88 as M
from adapter88 import dependency_ready,input_ready,acquire_file_lock,release_file_lock
CACHE=ROOT/'data/.queued_suite_20261122'
PHASES=(('B89','b89_runtime/run89.py'),('B88','b88_anchor/queue88.py'),('B90','b90_methods/queue90.py'))


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf8');tmp.replace(path)


def status(state,**details):
    save(CACHE/'status.json',{'state':state,'pid':os.getpid(),'updated_local':time.strftime('%Y-%m-%d %H:%M:%S %z'),'phases':[p[0] for p in PHASES],**details})


def preflight():
    M.assert_frozen();input_ready()
    for _,script in PHASES:assert (EXPERIMENTS/script).is_file(),script
    sys.path.insert(0,str(EXPERIMENTS/'b90_methods'))
    import adapter90,common90
    adapter90.input_ready()
    for name in ('verify90.py','verify_outputs90.py'):assert (common90.HERE/'lead_checks'/name).is_file(),name
    for path in (M.CACHE/'lead_checks.json',M.CACHE/'lead_readout_checks.json',common90.CACHE/'lead_checks.json',common90.CACHE/'lead_bootstrap_checks.json'):
        report=json.loads(path.read_text(encoding='utf8'));assert report.get('passes'),str(path)
        if path==common90.CACHE/'lead_checks.json':assert report['execution_sources']==common90.source_inventory(),'B90 independent audit stale'
    scout=json.loads((ROOT/'data/.b89/preflight.json').read_text(encoding='utf8'))
    assert scout.get('passes'),'B89 CPU/source preflight missing'
    sys.path.insert(0,str(EXPERIMENTS/'b89_runtime'))
    import run89
    binding=run89.binding()
    assert scout['binding_hash']==binding['hash'],'B89 preflight source/input/software binding stale'
    assert run89.clean(scout['result'],scout['process_exit'],binding),'B89 preflight lacks clean child exit'
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--status',action='store_true');p.add_argument('--plan',action='store_true');a=p.parse_args()
    if a.status:
        print((CACHE/'status.json').read_text(encoding='utf8') if (CACHE/'status.json').exists() else 'Not launched');return
    if a.plan:
        print(json.dumps({'dependency':'B85 complete with90 clean fits','phases':PHASES,'B88_fits':90,'B90_screen_fits':210,'B90_expansion_fits':120,'selection':'validation only; exactly two methods','concurrent_gpu_children':1}));return
    preflight();lock=CACHE/'suite.lock';assert acquire_file_lock(lock),'Suite already active';phase='B85_dependency'
    try:
        while True:
            try:ready=dependency_ready()
            except Exception as e:status('blocked_dependency',dependency='B85',error=str(e));raise
            if ready:break
            status('waiting_dependency',dependency='B85');time.sleep(30)
        for phase,script in PHASES:
            preflight();status('running',phase=phase)
            with (CACHE/(phase+'.log')).open('a',encoding='utf8') as log:
                result=subprocess.run([sys.executable,'-u',str(EXPERIMENTS/script)],cwd=ROOT,env={**os.environ,'PYTHONUNBUFFERED':'1'},stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(phase+' failed, returncode='+str(result.returncode)+'; no automatic retry')
            save(CACHE/(phase+'_exit.json'),{'phase':phase,'returncode':0,'finished_local':time.strftime('%Y-%m-%d %H:%M:%S %z')})
        status('complete')
    except Exception as e:
        if phase!='B85_dependency':status('failed',phase=phase,error=str(e))
        raise
    finally:release_file_lock(lock)


if __name__=='__main__':main()
