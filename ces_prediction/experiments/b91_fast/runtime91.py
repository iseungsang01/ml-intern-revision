"""Bounded fresh-process B91 runtime comparison, never a training controller."""
import argparse,ctypes,hashlib,json,os,platform,statistics,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=ROOT/'data';CACHE=DATA/'.b91';PAPER=ROOT/'docs/paper'
PATHS={
 'cuda_packed_native':{'device':'cuda','dense':False,'cudnn':False,'threads':2,'mkldnn':False},
 'cuda_dense_native':{'device':'cuda','dense':True,'cudnn':False,'threads':2,'mkldnn':False},
 'cuda_dense_cudnn':{'device':'cuda','dense':True,'cudnn':True,'threads':2,'mkldnn':False},
 'cuda_packed_cudnn':{'device':'cuda','dense':False,'cudnn':True,'threads':2,'mkldnn':False},
 'cpu_packed_native2':{'device':'cpu','dense':False,'cudnn':False,'threads':2,'mkldnn':False},
 'cpu_dense_mkldnn2':{'device':'cpu','dense':True,'cudnn':False,'threads':2,'mkldnn':True},
 'cpu_dense_mkldnn8':{'device':'cpu','dense':True,'cudnn':False,'threads':8,'mkldnn':True}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf8');tmp.replace(path)
def sources():
    pins=json.loads((ROOT/'ces_prediction/experiments/b88_anchor/frozen_b85_sources.json').read_text(encoding='utf8'))
    for rel,digest in pins.items():
        p=(ROOT/rel).resolve();assert p.is_relative_to(ROOT) and sha(p)==digest,'Frozen B85 source changed: '+rel
    for rel in ('ces_prediction/experiments/PREREGISTRATION_B91.md','ces_prediction/experiments/b88_anchor/frozen_b85_sources.json'):
        pins[rel]=sha(ROOT/rel)
    for name in ('runtime91.py','probe91.py','validation91.py'):pins[(HERE/name).relative_to(ROOT).as_posix()]=sha(HERE/name)
    return pins
def assert_registered():
    reg='ces_prediction/experiments/PREREGISTRATION_B91.md'
    subprocess.run(['git','ls-files','--error-unmatch',reg],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--quiet','HEAD','--',reg],cwd=ROOT,check=True)
    return sources()
def alive(pid):
    if int(pid)<=0:return True
    if sys.platform!='win32':return Path('/proc',str(pid)).exists()
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.OpenProcess.restype=ctypes.c_void_p
    k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)];k.CloseHandle.argtypes=[ctypes.c_void_p]
    h=k.OpenProcess(0x1000,False,int(pid))
    if not h:return ctypes.get_last_error()!=87
    try:
        code=ctypes.c_ulong();return not k.GetExitCodeProcess(h,ctypes.byref(code)) or code.value==259
    finally:k.CloseHandle(h)
def gpu_ready():
    p=CACHE/'author_stop.json';rec=json.loads(p.read_text(encoding='utf8'))
    assert rec['state']=='superseded_by_author_revision' and rec['runner_exited'] and rec['queue_exited']
    assert rec['trainer_exit_code']!=259
    for role in ('master','trainer','runner','queue'):assert not alive(rec[role]['ProcessId']),'Old owned process still alive: '+role
    target=(ROOT/rec['preserved_checkpoint']).resolve();assert target.is_relative_to(CACHE) and sha(target)==rec['checkpoint_sha256']
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    assert int(free.splitlines()[0].strip())>=2500,'Less than 2500 MiB free; other GPU holders unchanged'
    return sha(p)
def dense_class():
    import torch
    from model85 import B85Model
    class DenseB85(B85Model):
        def forward(self,x,lengths=None):
            out,_=self.lstm_ti(self._mask_input(x));out=self.norm_ti(out)
            return torch.cat((self.head_F(out),self.head_L5(out)),dim=-1)
    return DenseB85
def baseline_paths():
    sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b85_unified'))
def real_inputs():
    baseline_paths();import data85 as D
    from common85 import load_sources
    G,R=load_sources();rd=DATA/'.b85/runs/V1_r2_k0__1701__X'
    manifest=json.loads((rd/'manifest.json').read_text());stats=json.loads((rd/'stats.json').read_text())
    assert not set(manifest['train'])&set(manifest['val']) and not set(manifest['train'])&set(manifest['test'])
    allsegs=sorted(D.build_segments(manifest['train'],stats,'D10',G,R),key=lambda s:(s['fidx'],s['segment']))
    selected=[s for s in allsegs if s['mask'][:512].any()][:16];assert len(selected)==16
    prefixes=[]
    for s in selected:prefixes.append({k:(v[:512].copy() if k in ('x','y','mask','gi') else v) for k,v in s.items()})
    if len(set(len(s['x']) for s in prefixes))==1:
        s=prefixes[-1]
        for key in ('x','y','mask','gi'):s[key]=s[key][:-1]
        assert s['mask'].any(),'Unequal final actual prefix has no supervision'
    assert len(set(len(s['x']) for s in prefixes))>1
    binding={'manifest':sha(rd/'manifest.json'),'stats':sha(rd/'stats.json'),'grid':sha(DATA/'.b73/grid.npz'),'rows':sha(DATA/'.b73/rows.npz')}
    import numpy as np
    digest=hashlib.sha256()
    for s in prefixes:
        for key in ('x','y','mask','gi'):digest.update(np.ascontiguousarray(s[key]).tobytes())
    binding['batch_arrays']=digest.hexdigest();binding['selection']=[{'fidx':s['fidx'],'segment':s['segment'],'length':len(s['x'])} for s in prefixes]
    return prefixes,stats,manifest,G,R,binding
def verify_decision():
    record=json.loads((CACHE/'runtime_decision.json').read_text(encoding='utf8'))
    assert record['sources']==sources() and record['native_clean'] and record['selected_path'],'Runtime decision stale or unqualified'
    selected=record['selected_path'];rows=[r for r in record['records'] if r['path']==selected]
    assert len(rows)==3 and all(r['state']=='clean' and r['returncode']==0 for r in rows)
    for r in rows:
        output=CACHE/'probes'/f"{selected}__{r['rep']}.json"
        assert sha(output)==r['output_hash'],'Runtime child artifact changed'
        attestation=output.with_suffix('.exit.json');att=json.loads(attestation.read_text(encoding='utf8'))
        assert sha(attestation)==r['exit_hash'] and att['returncode']==0 and att['result_sha256']==sha(output) and att['sources']==sources(),'Missing or stale supervising exit attestation'
    _,_,_,_,_,binding=real_inputs();assert binding==record['input_binding'],'Runtime training inputs changed'
    import torch
    assert torch.__version__==record['torch_version'],'Runtime torch version changed'
    return record
def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu-only',action='store_true');p.add_argument('--python312',type=Path);p.add_argument('--plan',action='store_true');a=p.parse_args()
    pins=assert_registered();CACHE.mkdir(parents=True,exist_ok=True)
    decision=CACHE/'runtime_decision.json';assert not decision.exists(),'Immutable decision already exists'
    selected={k:v for k,v in PATHS.items() if not a.cpu_only or v['device']=='cpu'}
    if a.python312:
        assert a.python312.is_file()
        for dense in (False,True):selected['py312_cuda_'+('dense' if dense else 'packed')+'_cudnn']={'device':'cuda','dense':dense,'cudnn':True,'threads':2,'mkldnn':False,'python':str(a.python312)}
    if a.plan:print(json.dumps({'paths':selected,'repetitions':3,'child_timeout_seconds':120,'overall_seconds':900,'no_probe_started':True}));return
    started=time.monotonic();records=[];binding=None;torch_version=None;software_by_path={}
    for name,cfg in selected.items():
        for rep in range(3):
            remaining=900-(time.monotonic()-started)
            if remaining<1:
                records.append({'path':name,'rep':rep,'state':'not_tested_budget'});continue
            assert sources()==pins,'Source changed during probes'
            if cfg['device']=='cuda':gpu_ready()
            out=CACHE/'probes'/f'{name}__{rep}.json';log=out.with_suffix('.log');out.parent.mkdir(parents=True,exist_ok=True)
            cmd=[cfg.get('python',sys.executable),'-X','faulthandler','-u',str(HERE/'probe91.py'),'--path',name,'--rep',str(rep),'--output',str(out)]
            env={**os.environ,'PYTHONUNBUFFERED':'1'}
            rec={'path':name,'rep':rep,'config':cfg,'timeout_seconds':min(120,remaining),'sources':pins}
            with log.open('w',encoding='utf8') as stream:
                try:code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=min(120,remaining)).returncode
                except subprocess.TimeoutExpired:code=None;rec['state']='timeout'
            rec['returncode']=code
            attestation=out.with_suffix('.exit.json')
            save(attestation,{'path':name,'rep':rep,'returncode':code,'result_sha256':sha(out) if out.exists() else None,'sources':pins,'supervisor_pid':os.getpid(),'observed_at':time.time(),'timeout':code is None})
            rec['exit_hash']=sha(attestation)
            if code==0 and out.is_file():
                result=json.loads(out.read_text(encoding='utf8'));assert result['sources']==pins and result['all_pass']
                assert result['config']=={k:v for k,v in cfg.items() if k!='python'},'Child device/backend configuration differs'
                if name in software_by_path:assert result['software']==software_by_path[name],'Software differs between repetitions'
                else:software_by_path[name]=result['software']
                if binding is None:binding=result['input_binding'];torch_version=result['software']['torch']
                assert result['input_binding']==binding,'Different real inputs across runtime candidates'
                if result['software']['torch']!=torch_version:rec['state']='ineligible_torch_version'
                else:rec.update(state='clean',result=result,output_hash=sha(out))
            else:rec.setdefault('state','failed')
            records.append(rec);save(CACHE/'runtime_progress.json',{'records':records,'elapsed_seconds':time.monotonic()-started})
    eligible={}
    for name in selected:
        rr=[r for r in records if r['path']==name]
        if len(rr)==3 and all(r['state']=='clean' for r in rr):
            for r in rr:
                output=CACHE/'probes'/f"{name}__{r['rep']}.json";att=output.with_suffix('.exit.json')
                assert sha(output)==r['output_hash'] and sha(att)==r['exit_hash'] and json.loads(att.read_text())['returncode']==0
                assert r['result']['input_binding']==binding and r['result']['sources']==sources()
            eligible[name]=statistics.median(r['result']['training']['median_batch_seconds'] for r in rr)
    native='cuda_packed_native';chosen=None;factor=None
    if native in eligible:
        fastest=min(eligible,key=eligible.get)
        if eligible[native]<=eligible[fastest]*1.05:fastest=native
        factor=eligible[native]/eligible[fastest];chosen=fastest if factor>=1.25 else native
    result={'sources':pins,'records':records,'eligible_median_batch_seconds':eligible,'selected_path':chosen,'selected_config':selected[chosen] if chosen else None,'speed_factor':factor,'native_clean':native in eligible,'input_binding':binding,'torch_version':torch_version,
            'elapsed_seconds':time.monotonic()-started,'cpu_only':a.cpu_only,'status':'ready_for_independent_audit' if chosen else 'no_qualified_native_reference',
            'limitations':'Bounded real-prefix batch timing only; not measured full-epoch speedup; three point repetitions, no confidence interval.'}
    if not a.cpu_only:save(decision,result)
    save(CACHE/'runtime_report.json',result)
    text='# B.91 runtime probes\n\n'+result['limitations']+'\n\n'+json.dumps({k:v for k,v in result.items() if k not in ('records','sources')},indent=2)+'\n'
    for suffix in ('runtime','checks','readiness'):(PAPER/f'b91_{suffix}.md').write_text(text,encoding='utf8')
    print(json.dumps({'selected_path':chosen,'status':result['status'],'eligible':eligible}))
if __name__=='__main__':main()
