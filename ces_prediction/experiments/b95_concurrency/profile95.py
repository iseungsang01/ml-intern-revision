"""Actual full-epoch cohorts; no predictive readout or production mutation."""
import ctypes,json,math,os,statistics,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CACHE=ROOT/'data/.b95'
sys.path.insert(0,str(HERE.parent/'b93_gpu'))
import profile93 as P
from atomic93 import save_json as save

def guard():
    paths=[HERE/'profile95.py',HERE.parent/'PREREGISTRATION_B95.md']
    for p in paths:
        rel=p.relative_to(ROOT).as_posix()
        subprocess.run(['git','cat-file','-e','HEAD:'+rel],cwd=ROOT,check=True)
        subprocess.run(['git','diff','--quiet','HEAD','--',rel],cwd=ROOT,check=True)
    return {p.relative_to(ROOT).as_posix():P.sha(p) for p in paths}

def gpu():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    used,total,free,util=map(int,raw.splitlines()[0].split(','));return {'used_mib':used,'total_mib':total,'free_mib':free,'util_percent':util}

def host_available():
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','page_total','page_free','virtual_total','virtual_free','extended')]
    z=Status();z.length=ctypes.sizeof(z);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(z));return z.available

def choose(cohorts):
    clean=[c for c in cohorts if c['clean']]
    ref=[c['throughput'] for c in clean if c['workers']==2]
    if len(ref)!=2:return 2
    eligible={2:max(ref)}
    for n in (3,4,6):
        vals=[c['throughput'] for c in clean if c['workers']==n]
        if len(vals)==2 and min(vals)>=1.10*max(ref):eligible[n]=min(vals)
    best=max(eligible,key=eligible.get)
    return min(n for n,v in eligible.items() if v>=eligible[best]/1.05)

def main():
    pins=guard();sources=P.registered();binding=P.prepared_binding()
    prior=json.loads((ROOT/'data/.b93/runtime_decision93.json').read_text());assert prior['sources']==sources
    assert not (CACHE/'decision95.json').exists(),'Decision is immutable'
    CACHE.mkdir(parents=True,exist_ok=True);start=time.monotonic();cohorts=[];software=None
    def cohort(n):
        nonlocal software
        if 900-(time.monotonic()-start)<30:return None
        assert guard()==pins and P.sources()==sources and P.prepared_binding()==binding
        index=len(cohorts);children=[];samples=[];begin=time.monotonic();records=[]
        print(json.dumps({'starting_workers':n,'cohort':index}),flush=True)
        try:
            for slot in range(n):
                assert gpu()['free_mib']>=1800,'VRAM admission failed; no fallback'
                out=CACHE/f'cohort{index}'/f'worker{slot}.json';out.parent.mkdir(parents=True,exist_ok=True)
                log=out.with_suffix('.log').open('w',encoding='utf8')
                cmd=[sys.executable,'-X','faulthandler','-u',str(P.HERE/'worker93.py'),'--path','cuda_dense_native','--rep',str(100+index*10+slot),'--output',str(out),'--allow-existing-cpu']
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1'})
                children.append((child,out,log,time.monotonic()))
            while any(p.poll() is None for p,_,_,_ in children):
                samples.append(gpu())
                for p,_,_,launched in children:
                    if p.poll() is None and (time.monotonic()-launched>300 or time.monotonic()-start>900):p.kill()
                time.sleep(.5)
            wall=time.monotonic()-begin
            for p,out,log,launched in children:
                rc=p.wait();log.close();att={'returncode':rc,'pid':p.pid,'sources':sources,'driver_sources':pins,'result_sha256':P.sha(out) if out.exists() else None}
                save(out.with_suffix('.exit.json'),att)
                good=False
                if rc==0 and out.exists():
                    z=json.loads(out.read_text());assert z['all_pass'] and z['sources']==sources and z['input_binding']==binding
                    assert z['config']==P.PATHS['cuda_dense_native'] and z['actual_batch_binding']==prior['actual_batch_binding']
                    if software is None:software=z['software']
                    assert software==z['software'];good=True
                records.append({'result':out.relative_to(ROOT).as_posix(),'exit':out.with_suffix('.exit.json').relative_to(ROOT).as_posix(),'result_sha256':P.sha(out) if out.exists() else None,'exit_sha256':P.sha(out.with_suffix('.exit.json')),'clean':good})
            c={'workers':n,'wall_seconds':wall,'throughput':n/wall,'clean':len(records)==n and all(r['clean'] for r in records),'records':records,'peak_gpu_mib':max((s['used_mib'] for s in samples),default=0),'gpu_total_mib':samples[0]['total_mib'] if samples else gpu()['total_mib'],'samples':samples}
            cohorts.append(c);save(CACHE/'progress95.json',{'cohorts':cohorts,'elapsed':time.monotonic()-start});print(json.dumps({k:c[k] for k in ('workers','wall_seconds','throughput','clean','peak_gpu_mib')}),flush=True)
            return c
        finally:
            for p,_,log,_ in children:
                if p.poll() is None:p.kill();p.wait()
                if not log.closed:log.close()
    for n in (2,3,4):cohort(n)
    clean=[c for c in cohorts if c['clean']];refs=[c for c in clean if c['workers']==2]
    if refs:
        best=max(clean,key=lambda c:c['throughput'])
        if best['workers']==4 and best['throughput']>=1.10*refs[0]['throughput'] and host_available()>=8*1024**3 and best['peak_gpu_mib']*1.5<best['gpu_total_mib']*.8:cohort(6)
        clean=[c for c in cohorts if c['clean']];best=max(clean,key=lambda c:c['throughput'])
        if best['workers']>2 and best['throughput']>=1.10*refs[0]['throughput']:
            cohort(best['workers']);cohort(2)
    d={'selected_workers':choose(cohorts),'cohorts':cohorts,'sources':sources,'driver_sources':pins,'input_binding':binding,'elapsed_seconds':time.monotonic()-start,'test_accessed':False,'scope':'Full first epoch plus seven validation conditions; fresh process wall throughput, no predictive scores.'}
    assert guard()==pins;save(CACHE/'decision95.json',d);print(json.dumps({'selected_workers':d['selected_workers'],'elapsed':d['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
