"""Independent cohort arithmetic, evidence and global/local dispatch checks."""
import hashlib,json,math,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CACHE=ROOT/'data/.b95'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def compute_qualification():
    d=read(CACHE/'decision95.json');assert d['test_accessed'] is False
    for rel,h in d['driver_sources'].items():assert sha(ROOT/rel)==h
    sys.path.insert(0,str(HERE.parent/'b93_gpu'));import profile93 as P
    assert d['sources']==P.sources() and d['input_binding']==P.prepared_binding()
    old=read(ROOT/'data/.b93/runtime_decision93.json');software=None
    for c in d['cohorts']:
        assert c['workers'] in (2,3,4,6) and math.isfinite(c['wall_seconds']) and c['wall_seconds']>0
        assert math.isclose(c['throughput'],c['workers']/c['wall_seconds'],rel_tol=1e-12)
        clean=[]
        for r in c['records']:
            path=(ROOT/r['result']).resolve();ex=(ROOT/r['exit']).resolve();assert path.is_relative_to(CACHE) and ex.is_relative_to(CACHE)
            assert sha(ex)==r['exit_sha256'];att=read(ex)
            assert att['sources']==d['sources'] and att['driver_sources']==d['driver_sources']
            good=att['returncode']==0 and path.exists()
            if path.exists():assert sha(path)==r['result_sha256']==att['result_sha256']
            if good:
                z=read(path);assert z['all_pass'] and z['sources']==d['sources'] and z['input_binding']==d['input_binding']
                assert z['actual_batch_binding']==old['actual_batch_binding']
                assert z['config']==dict(device='cuda',dense=True,cudnn=False,threads=2,mkldnn=False)
                assert c['wall_seconds']>=z['timing']['total_prepare_epoch_seconds']
                if software is None:software=z['software']
                assert software==z['software']
            assert r['clean']==good;clean.append(good)
        assert c['clean']==(len(clean)==c['workers'] and all(clean))
        assert c['peak_gpu_mib']==max((s['used_mib'] for s in c['samples']),default=0)
    refs=[c['throughput'] for c in d['cohorts'] if c['clean'] and c['workers']==2];eligible={2:max(refs) if refs else 0}
    if len(refs)==2:
        for n in (3,4,6):
            values=[c['throughput'] for c in d['cohorts'] if c['clean'] and c['workers']==n]
            if len(values)==2 and min(values)>=max(refs)*1.10:eligible[n]=min(values)
    highest=max(eligible.values());selected=min(n for n,v in eligible.items() if v>=highest/1.05)
    assert selected==d['selected_workers']
    tie=read(CACHE/'tie95.json');assert tie['base_decision_sha256']==sha(CACHE/'decision95.json') and tie['test_accessed'] is False
    for rel,h in tie['driver_sources'].items():assert sha(ROOT/rel)==h
    assert tie['workers']==4 and 0<tie['wall_seconds']<=245 and math.isclose(tie['throughput'],4/tie['wall_seconds'],rel_tol=1e-12)
    good=[]
    for r in tie['records']:
        out=(ROOT/r['result']).resolve();ex=(ROOT/r['exit']).resolve();assert out.is_relative_to(CACHE) and ex.is_relative_to(CACHE)
        assert sha(ex)==r['exit_sha256'];att=read(ex);assert att['driver_sources']==tie['driver_sources']
        ok=att['returncode']==0 and out.exists()
        if out.exists():assert sha(out)==r['result_sha256']==att['result_sha256']
        if ok:
            z=read(out);assert z['all_pass'] and z['sources']==d['sources'] and z['input_binding']==d['input_binding'] and z['actual_batch_binding']==old['actual_batch_binding']
            assert z['config']==dict(device='cuda',dense=True,cudnn=False,threads=2,mkldnn=False) and z['software']==software
            assert tie['wall_seconds']>=z['timing']['total_prepare_epoch_seconds']
        assert r['clean']==ok;good.append(ok)
    assert tie['clean']==(len(good)==4 and all(good))
    four=[c['throughput'] for c in d['cohorts'] if c['workers']==4 and c['clean']]
    if tie['clean']:four.append(tie['throughput'])
    if len(refs)==2 and len(four)==2 and min(four)>=1.10*max(refs):eligible[4]=min(four)
    highest=max(eligible.values());selected=min(n for n,v in eligible.items() if v>=highest/1.05)
    return {'all_pass':True,'selected_workers':selected,'base_decision_sha256':sha(CACHE/'decision95.json'),'tie_sha256':sha(CACHE/'tie95.json'),'elapsed_seconds':d['elapsed_seconds'],'tie_wall_seconds':tie['wall_seconds'],'eligible_throughput':{str(k):v for k,v in eligible.items()},'test_accessed':False}

def qualification(publish=False):
    result=compute_qualification();path=CACHE/'production_policy95.json'
    if publish and not path.exists():
        sys.path.insert(0,str(HERE.parent/'b93_gpu'));from atomic93 import save_json
        save_json(path,result)
    assert read(path)==result,'Published policy differs from independently reconstructed evidence'
    return {**result,'decision_sha256':sha(path)}

def dispatch_config(path,sources,decision):
    c=read(path)
    if 'dispatch_sources95' not in c:return # Historical, immutable B94 fits.
    assert c['dispatch_sources95']==sources and c['dispatch_policy_sha256']==decision['decision_sha256']
    assert c['dispatch_registration']=='PREREGISTRATION_B94_ADDENDUM3.md'
    n=c['global_gpu_worker_limit'];i=c['global_worker_slot'];g=i//2
    assert n in (1,decision['selected_workers']) and 0<=i<n
    assert c['local_worker_group']==g and c['worker_slot']==i%2 and c['scheduled_concurrency']==min(2,n-2*g)
    assert c['scheduled_concurrency_scope']=='local group; global capacity recorded separately'
    ex=path.parent/'process_exit.json'
    if ex.exists():
        e=read(ex);assert e['global_worker_slot']==i and e['worker_slot']==i%2 and e['global_gpu_worker_limit']==n

if __name__=='__main__':
    d=qualification(publish=True);sys.path.insert(0,str(HERE.parent/'b93_gpu'));from atomic93 import save_json
    save_json(CACHE/'audit95.json',d);print(json.dumps(d))
