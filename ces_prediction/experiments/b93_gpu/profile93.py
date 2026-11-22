"""Fresh-process full-epoch runtime qualification; no model selection or test readout."""
import argparse,ctypes,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=ROOT/'data';CACHE=DATA/'.b93'
PATHS={'cpu8_dense':{'device':'cpu','dense':True,'cudnn':False,'threads':8,'mkldnn':True},
       'cuda_packed_native':{'device':'cuda','dense':False,'cudnn':False,'threads':2,'mkldnn':False},
       'cuda_dense_native':{'device':'cuda','dense':True,'cudnn':False,'threads':2,'mkldnn':False}}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,z):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(z,sort_keys=True,indent=2),encoding='utf8');tmp.replace(path)
def sources():
    pinpath=ROOT/'ces_prediction/experiments/b92_reduced/frozen_sources92.json';pins=json.loads(pinpath.read_text(encoding='utf8'))
    for rel,h in pins.items():
        p=(ROOT/rel).resolve();assert p.is_relative_to(ROOT) and sha(p)==h,'Frozen source changed: '+rel
    paths=list((ROOT/'ces_prediction/experiments/b92_reduced').glob('*.py'))+[pinpath,HERE/'profile93.py',HERE/'worker93.py',ROOT/'ces_prediction/experiments/PREREGISTRATION_B93.md']
    return {**pins,**{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
def registered():
    reg='ces_prediction/experiments/PREREGISTRATION_B93.md'
    subprocess.run(['git','cat-file','-e','HEAD:'+reg],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--quiet','HEAD','--',reg],cwd=ROOT,check=True)
    return sources()
def prepared_binding():
    rd=DATA/'.b92/cpu_r2/pilot/V1_r2_k0__1701__C';config=json.loads((rd/'config.json').read_text())
    key=config['prepared_key'];path=DATA/'.b92/cpu_r2/prepared'/f'V1_r2_k0__{key}.pt';meta=path.with_suffix('.json');m=json.loads(meta.read_text())
    assert m['key']==key and sha(path)==m['sha256']==config['prepared_artifact_sha256']
    return {'unit':'V1_r2_k0','seed':1701,'code':'C','prepared_key':key,'prepared_path':path.relative_to(ROOT).as_posix(),'prepared_sha256':sha(path),'metadata_sha256':sha(meta),'config_path':(rd/'config.json').relative_to(ROOT).as_posix(),'config_sha256':sha(rd/'config.json')}
def alive(pid):
    if pid<=0:return True
    if sys.platform!='win32':return Path('/proc',str(pid)).exists()
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.OpenProcess.restype=ctypes.c_void_p;k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)];k.CloseHandle.argtypes=[ctypes.c_void_p]
    h=k.OpenProcess(0x1000,False,int(pid))
    if not h:return ctypes.get_last_error()!=87
    try:
        z=ctypes.c_ulong();return not k.GetExitCodeProcess(h,ctypes.byref(z)) or z.value==259
    finally:k.CloseHandle(h)
def gpu_ready(allow_existing_cpu=False):
    prior=DATA/'.b92/cpu_r2/launch_pid.txt'
    if prior.exists() and not allow_existing_cpu:assert not alive(int(prior.read_text())),'Prior CPU queue is still alive'
    rec=json.loads((DATA/'.b91/author_stop.json').read_text())
    assert rec['state']=='superseded_by_author_revision'
    for role in ('master','trainer','runner','queue'):assert not alive(rec[role]['ProcessId']),'Prior training process alive'
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    assert int(free.splitlines()[0])>=2500,'Insufficient free VRAM; unrelated holders unchanged'
def contention_snapshot():
    state=DATA/'.b92/cpu_r2/queue_status.json'
    result={'time':time.time(),'existing_cpu_queue_state':json.loads(state.read_text()) if state.exists() else None}
    if sys.platform=='win32':
        cmd="Get-Process python* -ErrorAction SilentlyContinue | Select-Object Id,CPU,WorkingSet64,StartTime | ConvertTo-Json -Compress"
        p=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True)
        result['python_resources']=p.stdout.strip();result['resource_query_returncode']=p.returncode
    result['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    return result
def verify_decision():
    d=json.loads((CACHE/'runtime_decision93.json').read_text())
    assert d['sources']==sources() and d['input_binding']==prepared_binding(),'Stale full-epoch runtime decision'
    assert d['selected_path'] and d['selected_config']['device']=='cuda' and not d['selected_config']['cudnn']
    rows=[r for r in d['records'] if r['path']==d['selected_path'] and (r.get('phase','serial')=='serial' or d.get('parallel_workers')==2)]
    assert len(rows)==(5 if d.get('parallel_workers')==2 else 3)
    for r in rows:
        path=CACHE/'probes'/f"{r['path']}__{r['rep']}.json";exitpath=path.with_suffix('.exit.json')
        assert r['state']=='clean' and r['returncode']==0 and sha(path)==r['result_sha256'] and sha(exitpath)==r['exit_sha256']
        att=json.loads(exitpath.read_text());assert att['returncode']==0 and att['result_sha256']==sha(path) and att['sources']==d['sources']
    return d
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',action='store_true');ap.add_argument('--gpu-only',action='store_true');ap.add_argument('--parallel-test2',action='store_true');ap.add_argument('--allow-existing-cpu',action='store_true');a=ap.parse_args();pins=registered()
    assert a.gpu_only,'Current B93 protocol is GPU-only; no fresh CPU benchmark is authorized'
    assert not (a.allow_existing_cpu or a.parallel_test2) or a.gpu_only,'Parallel qualification requires GPU-only mode; no competing CPU benchmark'
    paths={k:v for k,v in PATHS.items() if not a.gpu_only or v['device']=='cuda'};budget=900 if a.gpu_only else 1200
    if a.plan:print(json.dumps({'paths':paths,'CPU_children':0 if a.gpu_only else 1,'GPU_children_each':3,'parallel_test2':a.parallel_test2,'allow_existing_cpu':a.allow_existing_cpu,'GPU_timeout':300,'overall_budget':budget,'no_training_started':True}));return
    assert not (CACHE/'runtime_decision93.json').exists(),'Immutable B93 decision exists'
    binding=prepared_binding();CACHE.mkdir(parents=True,exist_ok=True);records=[];started=time.monotonic();software={};membership=None;initial_contention=contention_snapshot()
    def command(name,rep,out):
        cmd=[sys.executable,'-X','faulthandler','-u',str(HERE/'worker93.py'),'--path',name,'--rep',str(rep),'--output',str(out)]
        if a.allow_existing_cpu:cmd.append('--allow-existing-cpu')
        return cmd
    def finish(name,rep,out,code,phase,contention):
        nonlocal membership
        att={'path':name,'rep':rep,'phase':phase,'returncode':code,'timeout':code is None,'result_sha256':sha(out) if out.exists() else None,'sources':pins,'supervisor_pid':os.getpid(),'time':time.time()};save(out.with_suffix('.exit.json'),att)
        r={'path':name,'rep':rep,'phase':phase,'returncode':code,'state':'failed' if code is not None else 'timeout','exit_sha256':sha(out.with_suffix('.exit.json')),'contention_at_start':contention}
        if code==0 and out.is_file():
            z=json.loads(out.read_text());assert z['sources']==pins and z['input_binding']==binding and z['config']==PATHS[name] and z['all_pass']
            if membership is None:membership=z['actual_batch_binding']
            assert membership==z['actual_batch_binding'],'Actual full epoch membership/order/masks differ'
            if name in software:assert software[name]==z['software']
            else:software[name]=z['software']
            r.update(state='clean',result=z,result_sha256=sha(out))
        records.append(r);save(CACHE/'progress93.json',{'records':records,'elapsed_seconds':time.monotonic()-started})
        print(name+' '+str(rep)+' '+r['state'],flush=True)
        return r
    for name,cfg in paths.items():
        for rep in range(1 if cfg['device']=='cpu' else 3):
            remaining=budget-(time.monotonic()-started)
            if remaining<1:records.append({'path':name,'rep':rep,'state':'not_tested_budget'});continue
            assert sources()==pins and prepared_binding()==binding
            if cfg['device']=='cuda':gpu_ready(a.allow_existing_cpu)
            out=CACHE/'probes'/f'{name}__{rep}.json';out.parent.mkdir(parents=True,exist_ok=True);limit=min(480 if cfg['device']=='cpu' else 300,remaining)
            print('Starting '+name+' repetition '+str(rep),flush=True)
            contention=contention_snapshot()
            with out.with_suffix('.log').open('w',encoding='utf8') as log:
                try:code=subprocess.run(command(name,rep,out),cwd=ROOT,env={**os.environ,'PYTHONUNBUFFERED':'1'},stdout=log,stderr=subprocess.STDOUT,timeout=limit).returncode
                except subprocess.TimeoutExpired:code=None
            finish(name,rep,out,code,'serial',contention)
    cpu=[r for r in records if r['path']=='cpu8_dense'];eligible={}
    if a.gpu_only or (len(cpu)==1 and cpu[0]['state']=='clean'):
        for name,cfg in PATHS.items():
            rr=[r for r in records if r['path']==name and r.get('phase','serial')=='serial']
            if cfg['device']=='cuda' and len(rr)==3 and all(r['state']=='clean' for r in rr):eligible[name]=statistics.median(r['result']['timing']['epoch_seconds'] for r in rr)
    fastest=min(eligible,key=eligible.get) if eligible else None
    if fastest and 'cuda_packed_native' in eligible and eligible['cuda_packed_native']<=eligible[fastest]*1.05:fastest='cuda_packed_native'
    factor=cpu[0]['result']['timing']['epoch_seconds']/eligible[fastest] if fastest and not a.gpu_only else None
    selected=fastest if a.gpu_only else fastest if factor is not None and factor>=1.25 else None
    parallel_workers=1;parallel_wall=None;parallel_gain=None
    if selected and a.parallel_test2 and budget-(time.monotonic()-started)>5:
        children=[];parallel_start=time.monotonic();contention=contention_snapshot()
        try:
            for rep in (30,31):
                gpu_ready(a.allow_existing_cpu);out=CACHE/'probes'/f'{selected}__{rep}.json';log=out.with_suffix('.log').open('w',encoding='utf8')
                process=subprocess.Popen(command(selected,rep,out),cwd=ROOT,env={**os.environ,'PYTHONUNBUFFERED':'1'},stdout=log,stderr=subprocess.STDOUT)
                children.append((rep,process,out,log,time.monotonic()))
            states={}
            while len(states)<len(children):
                for rep,process,out,log,launch in children:
                    if rep in states:continue
                    code=process.poll()
                    if code is not None:states[rep]=code
                    elif time.monotonic()-launch>=300 or time.monotonic()-started>=budget:
                        process.kill();process.wait();states[rep]=None
                if len(states)<len(children):time.sleep(.1)
            parallel_wall=time.monotonic()-parallel_start;rr=[]
            for rep,process,out,log,launch in children:
                log.close();rr.append(finish(selected,rep,out,states[rep],'parallel',contention))
            if len(rr)==2 and all(r['state']=='clean' for r in rr):
                parallel_gain=2*eligible[selected]/parallel_wall
                if parallel_gain>=1.1:parallel_workers=2
        finally:
            for rep,process,out,log,launch in children:
                if process.poll() is None:process.kill();process.wait()
                if not log.closed:log.close()
    d={'sources':pins,'input_binding':binding,'actual_batch_binding':membership,'records':records,'eligible_epoch_seconds':eligible,'selected_path':selected,'selected_config':PATHS[selected] if selected else None,'speed_factor_vs_cpu':factor,'gpu_only':a.gpu_only,'allow_existing_cpu':a.allow_existing_cpu,'initial_contention':initial_contention,'parallel_workers':parallel_workers,'parallel_two_jobs_wall_seconds':parallel_wall,'parallel_aggregate_gain':parallel_gain,'elapsed_seconds':time.monotonic()-started,'status':'qualified_for_independent_audit' if selected else 'no_qualified_acceleration','scope':'First complete actual training epoch plus all seven full validation conditions; dropout .1; existing CPU contention recorded; no CPU-vs-GPU speed claim in GPU-only mode; no convergence or scientific test evidence.'}
    assert sources()==pins and prepared_binding()==binding;save(CACHE/'runtime_decision93.json',d);save(CACHE/'runtime_report93.json',d)
    print(json.dumps({k:d[k] for k in ('selected_path','speed_factor_vs_cpu','eligible_epoch_seconds','elapsed_seconds','status')}),flush=True)
if __name__=='__main__':main()
