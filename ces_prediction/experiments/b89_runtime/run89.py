"""B89 supervisor: CPU preflight now; bounded fresh CUDA children after B85.

No backend is activated for another experiment. No GPU libraries are imported
by --plan or by the dependency wait loop before B85 reports completion.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CACHE=ROOT/'data/.b89'
PAPER=ROOT/'docs/paper'
UNIT='V1_r2_k0'
CANDIDATES={
    'native':{'cudnn':False,'env':{}},
    'cudnn_default':{'cudnn':True,'env':{}},
    'cudnn_v7':{'cudnn':True,'env':{'TORCH_CUDNN_V8_API_DISABLED':'1'}},
    'cudnn_no_cache':{'cudnn':True,'env':{'TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT':'-1'}},
}
CONTROL_ENV=('TORCH_CUDNN_V8_API_DISABLED','TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT',
             'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf8');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def binding():
    """Pure file reads; pin both inherited and scout execution/data contracts."""
    pins=json.loads((ROOT/'ces_prediction/experiments/b88_anchor/frozen_b85_sources.json').read_text(encoding='utf8'))
    for rel,want in pins.items():
        path=(ROOT/rel).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(),rel
        assert sha(path)==want,'Frozen inherited source changed: '+rel
    files=[HERE/'run89.py',HERE/'probe89.py',
           ROOT/'ces_prediction/experiments/b88_anchor/frozen_b85_sources.json']
    files += [ROOT/'ces_prediction/experiments/b88_anchor'/name for name in
              ('common88.py','adapter88.py','data88.py','model88.py')]
    folder=ROOT/'data/.b85/runs'/f'{UNIT}__1701__X'
    files += [folder/name for name in ('manifest.json','stats.json','config.json','complete.json','process_exit.json')]
    files += [ROOT/'data/.b73'/name for name in ('grid.npz','rows.npz')]
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    payload={'sources':hashes,'inherited_sources':pins,'unit':UNIT,'seed':1701,'rule':'D10',
             'prefixes':2,'max_rows':128,'warmup':3,'steps':10,'repetitions':3,
             'candidates':CANDIDATES,'weight_source':'seed1701 initialization, cloned CPU to CUDA',
             'execution_mode':{'wrapper_training':False,'rnn_training':True,'rnn_dropout':0.0},
             'software':{'python':sys.version,'torch':importlib.metadata.version('torch'),
                         'numpy':importlib.metadata.version('numpy')}}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return {'hash':digest,**payload}


def child(candidate,repeat,bind,cpu=False):
    folder=CACHE/(f"cpu_preflight/{bind['hash'][:16]}" if cpu else f'{candidate}/r{repeat}')
    folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'result.json';exit_path=folder/'process_exit.json'
    if exit_path.exists():
        result=json.loads(result_path.read_text()) if result_path.exists() else None
        att=json.loads(exit_path.read_text())
        expected_sha=sha(result_path) if result_path.exists() else None
        if att.get('binding_hash')==bind['hash'] and att.get('artifact_sha256')==expected_sha:
            return result,att
        raise RuntimeError('Stale scout artifact; preserve and inspect before a new registered execution')
    if result_path.exists():raise RuntimeError('Unattested probe artifact; preserve and inspect before execution')
    env=os.environ.copy()
    for key in CONTROL_ENV:env.pop(key,None)
    env.update(CANDIDATES[candidate]['env'])
    env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if cpu:env['CUDA_VISIBLE_DEVICES']='-1'
    cmd=[sys.executable,'-X','faulthandler','-u',str(HERE/'probe89.py'),
         '--candidate',candidate,'--repeat',str(repeat),'--binding-hash',bind['hash'],
         '--output',str(result_path)]
    if cpu:cmd.append('--cpu-check')
    started=time.time();timeout=False
    with (folder/'process.log').open('w',encoding='utf8') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:code=proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            timeout=True;proc.kill();code=proc.wait() # ONLY this scout's own child.
    result=json.loads(result_path.read_text()) if result_path.exists() else None
    att={'returncode':code,'timed_out':timeout,'candidate':candidate,'repeat':repeat,
         'binding_hash':bind['hash'],'child_pid':proc.pid,'elapsed_seconds':time.time()-started,
         'input_hash':result.get('input_hash') if result else None,
         'artifact_sha256':sha(result_path) if result_path.exists() else None}
    save(exit_path,att)
    return result,att


def clean(result,att,bind,expected_input=None):
    return bool(result and att['returncode']==0 and not att['timed_out'] and
                att['binding_hash']==bind['hash'] and result.get('binding_hash')==bind['hash'] and
                att.get('input_hash')==result.get('input_hash') and result.get('input_hash') and
                (expected_input is None or result['input_hash']==expected_input) and
                result.get('checks',{}).get('all_pass'))


def preflight():
    bind=binding();result,att=child('native',0,bind,cpu=True)
    passed=clean(result,att,bind)
    report={'passes':passed,'binding_hash':bind['hash'],'sourcehashes':bind['sources'],
            'inherited_sources':bind['inherited_sources'],'cpu_only':True,
            'result':result,'process_exit':att}
    save(CACHE/'runtime_manifest.json',bind);save(CACHE/'preflight.json',report)
    save(CACHE/'checks89.json',report)
    assert passed,'B89 CPU preflight failed; see ignored process log'
    print('PASS: B89 real-input CPU preflight; no GPU benchmark executed',flush=True)
    return bind


def dependency_ready():
    path=ROOT/'data/.b85/queue_status.json'
    if not path.is_file():return False
    state=json.loads(path.read_text())['state']
    if state in ('failed','blocked_dependency'):raise RuntimeError('B85 dependency failed')
    if state!='complete':return False
    sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b88_anchor'))
    from adapter88 import dependency_ready as ready
    return ready()


def status(state,**details):
    report={'state':state,'pid':os.getpid(),'updated':time.time(),**details}
    save(CACHE/'status.json',report)
    PAPER.mkdir(parents=True,exist_ok=True)
    text='# B.89 runtime scout status\n\n'+state+'\n\n'+json.dumps(details,indent=2)+'\n'
    (PAPER/'b89_runtime_status.md').write_text(text,encoding='utf8')


def wait_gpu():
    while True:
        assert dependency_ready(),'B85 dependency is not verified complete'
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if int(free.splitlines()[0].strip())>=2500:return
        status('waiting_gpu',minimum_free_MiB=2500);time.sleep(10)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',action='store_true');ap.add_argument('--check',action='store_true')
    args=ap.parse_args()
    if args.plan:
        print(json.dumps({'candidates':CANDIDATES,'children':12,'timeout_seconds_each':180,
                          'dependency':'all90 B85 clean fits and queue complete','recommendation_only':True},indent=2));return
    if args.check:preflight();return
    bind=binding();check=json.loads((CACHE/'preflight.json').read_text(encoding='utf8'))
    assert check.get('passes') and check.get('binding_hash')==bind['hash'],'Preflight missing/stale'
    CACHE.mkdir(parents=True,exist_ok=True)
    lock=CACHE/'runtime.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        while not dependency_ready():status('waiting_dependency');time.sleep(30)
        assert binding()['hash']==bind['hash'],'Sources/inputs changed while waiting'
        outcomes={}
        for candidate in CANDIDATES:
            reps=[]
            for repeat in range(3):
                wait_gpu();status('benchmarking',candidate=candidate,repeat=repeat)
                result,att=child(candidate,repeat,bind)
                assert binding()['hash']==bind['hash'],'Sources/inputs changed during scout'
                reps.append({'result':result,'process_exit':att,'eligible':clean(result,att,bind,check['result']['input_hash'])})
            eligible=all(r['eligible'] for r in reps)
            times=[r['result']['timing']['median_step_seconds'] for r in reps] if eligible else []
            outcomes[candidate]={'eligible':eligible,'repetitions':reps,
                                 'process_step_medians_seconds':times,
                                 'median_seconds':statistics.median(times) if times else None}
            save(CACHE/'partial_results.json',outcomes)
            if candidate=='native' and not eligible:raise RuntimeError('Native baseline failed; suite must stop')
        native=outcomes['native']['median_seconds']
        eligible=[c for c,r in outcomes.items() if r['eligible']]
        fastest=min(eligible,key=lambda c:outcomes[c]['median_seconds'])
        ratio=native/outcomes[fastest]['median_seconds']
        recommendation=fastest if ratio>1.5 else 'native'
        result={'binding_hash':bind['hash'],'candidates':outcomes,'recommendation':recommendation,
                'measured_speed_ratio':ratio,'automatic_activation':False,
                'statements':{'R1':outcomes['native']['eligible'],
                              'R2':any(outcomes[c]['eligible'] for c in outcomes if c!='native'),
                              'R3':recommendation!='native'},
                'limitation':'Three process timings, no confidence interval; prefix/eval runtime probe, not training speed or accuracy.'}
        save(CACHE/'recommendation.json',result);status('complete',**result)
    except Exception as exc:
        status('failed',error=str(exc));raise
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
