"""One registered real four-worker tie-confirmation cohort; no TEST scores."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CACHE=ROOT/'data/.b95'
sys.path.insert(0,str(HERE.parent/'b93_gpu'))
import profile93 as P
from atomic93 import save_json as save

def main():
    pins={}
    for p in (Path(__file__).resolve(),HERE.parent/'PREREGISTRATION_B95_ADDENDUM1.md'):
        rel=p.relative_to(ROOT).as_posix();subprocess.run(['git','cat-file','-e','HEAD:'+rel],cwd=ROOT,check=True);subprocess.run(['git','diff','--quiet','HEAD','--',rel],cwd=ROOT,check=True);pins[rel]=P.sha(p)
    base=json.loads((CACHE/'decision95.json').read_text());assert base['sources']==P.sources() and base['input_binding']==P.prepared_binding()
    assert not(CACHE/'tie95.json').exists();children=[];began=time.monotonic();records=[]
    try:
        for slot in range(4):
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);assert free>=2500
            out=CACHE/'tie'/f'worker{slot}.json';out.parent.mkdir(parents=True,exist_ok=True);log=out.with_suffix('.log').open('w',encoding='utf8')
            cmd=[sys.executable,'-X','faulthandler','-u',str(P.HERE/'worker93.py'),'--path','cuda_dense_native','--rep',str(300+slot),'--output',str(out),'--allow-existing-cpu']
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1'});children.append((p,out,log,time.monotonic()))
        while any(p.poll() is None for p,_,_,_ in children):
            for p,_,_,started in children:
                if p.poll() is None and(time.monotonic()-started>210 or time.monotonic()-began>240):p.kill()
            time.sleep(.2)
        wall=time.monotonic()-began
        for p,out,log,_ in children:
            rc=p.wait();log.close();att={'returncode':rc,'pid':p.pid,'driver_sources':pins,'result_sha256':P.sha(out) if out.exists() else None};save(out.with_suffix('.exit.json'),att)
            records.append({'result':out.relative_to(ROOT).as_posix(),'exit':out.with_suffix('.exit.json').relative_to(ROOT).as_posix(),'result_sha256':att['result_sha256'],'exit_sha256':P.sha(out.with_suffix('.exit.json')),'clean':rc==0 and out.exists()})
        report={'workers':4,'wall_seconds':wall,'throughput':4/wall,'clean':all(r['clean'] for r in records),'records':records,'driver_sources':pins,'base_decision_sha256':P.sha(CACHE/'decision95.json'),'test_accessed':False}
        save(CACHE/'tie95.json',report);print(json.dumps({k:report[k] for k in ('workers','wall_seconds','throughput','clean')}),flush=True)
    finally:
        for p,_,log,_ in children:
            if p.poll() is None:p.kill();p.wait()
            if not log.closed:log.close()

if __name__=='__main__':main()
