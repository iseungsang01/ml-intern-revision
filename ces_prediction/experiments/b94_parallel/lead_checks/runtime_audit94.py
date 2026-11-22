"""Independent B93 evidence/threshold audit for B94. Never trains or loads TEST."""
import hashlib,json,math,statistics,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
DATA=ROOT/'data';BASE=ROOT/'ces_prediction/experiments/b92_reduced'


def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(z):return hashlib.sha256(json.dumps(z,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def safe(rel):
    p=(ROOT/rel).resolve();assert not Path(rel).is_absolute() and p.is_relative_to(ROOT),rel
    return p


def committed(p):
    rel=Path(p).relative_to(ROOT).as_posix()
    subprocess.run(['git','cat-file','-e','HEAD:'+rel],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--quiet','HEAD','--',rel],cwd=ROOT,check=True)


def source_closure():
    frozen=BASE/'frozen_sources92.json';legacy=load(frozen)
    closure={**legacy,frozen.relative_to(ROOT).as_posix():sha(frozen)}
    closure.update({p.relative_to(ROOT).as_posix():sha(p) for p in BASE.glob('*.py')})
    for name in ('profile93.py','worker93.py'):
        p=BASE.parent/'b93_gpu'/name;closure[p.relative_to(ROOT).as_posix()]=sha(p)
    reg=BASE.parent/'PREREGISTRATION_B93.md';closure[reg.relative_to(ROOT).as_posix()]=sha(reg)
    for rel,h in legacy.items():assert sha(safe(rel))==h,'Inherited numeric source changed: '+rel
    return closure


def qualification():
    path=DATA/'.b93/runtime_decision93.json';d=load(path)
    independent=load(DATA/'.b93/lead_audit93.json')
    assert independent['all_pass'] and independent['decision_sha256']==sha(path),'Current independent B93 audit required'
    assert d['gpu_only'] is True and d['status']=='qualified_for_independent_audit'
    assert d['sources']==source_closure(),'B93 source closure mismatch'
    binding=d['input_binding'];assert binding['unit']=='V1_r2_k0' and binding['seed']==1701 and binding['code']=='C'
    pp=safe(binding['prepared_path']);cp=safe(binding['config_path'])
    assert pp.is_relative_to(DATA/'.b92/cpu_r2/prepared') and cp.is_relative_to(DATA/'.b92/cpu_r2')
    meta=load(pp.with_suffix('.json'));config=load(cp)
    assert sha(pp)==binding['prepared_sha256']==meta['sha256']==config['prepared_artifact_sha256']
    assert sha(pp.with_suffix('.json'))==binding['metadata_sha256'] and sha(cp)==binding['config_sha256']
    assert binding['prepared_key']==meta['key']==config['prepared_key']
    configs={name:{'device':'cuda','dense':dense,'cudnn':False,'threads':2,'mkldnn':False}
             for name,dense in [('cuda_packed_native',False),('cuda_dense_native',True)]}
    eligible={};evidence={};records=d['records'];members=d['actual_batch_binding']
    assert members['seed']==1701 and members['epoch']==0 and members['shuffle_seed']==1701
    assert len(members['validation_batches'])==len(members['validation_supervised_counts'])==7
    assert all(x>0 for x in members['train_supervised_counts'])
    def verify(row):
        assert row['path'] in configs and row['state']=='clean' and row['returncode']==0
        p=DATA/'.b93/probes'/f"{row['path']}__{row['rep']}.json";ep=p.with_suffix('.exit.json')
        z,att=load(p),load(ep)
        assert sha(p)==row['result_sha256'] and sha(ep)==row['exit_sha256']
        assert z==row['result'] and z['all_pass'] and z['sources']==d['sources']
        assert z['input_binding']==binding and z['actual_batch_binding']==members and z['config']==configs[row['path']]
        assert att['path']==row['path'] and att['rep']==row['rep'] and att['phase']==row.get('phase','serial')
        assert att['returncode']==0 and not att['timeout'] and att['result_sha256']==sha(p) and att['sources']==d['sources']
        c=z['checks'];assert c['actual_dropout']==.1 and c['validation_rule_count']==7
        assert c['all_training_losses_finite'] and c['all_validation_sums_finite'] and c['all_parameters_finite']
        assert not c['scientific_loss_values_exported'] and not c['test_inputs_read']
        assert c['training_updates']==members['train_batches']
        sw=z['software'];assert not sw['cudnn_enabled'] and not sw['mkldnn_enabled'] and sw['threads']==2
        assert sw['python']==sys.version
        t=z['timing'];assert all(math.isfinite(t[k]) and t[k]>0 for k in ('epoch_seconds','training_seconds','seven_rule_validation_seconds'))
        assert math.isclose(t['epoch_seconds'],t['training_seconds']+t['seven_rule_validation_seconds'],rel_tol=.005)
        return z
    for name in configs:
        rows=[r for r in records if r['path']==name and r.get('phase','serial')=='serial']
        assert len({r['rep'] for r in rows})==len(rows),'Duplicate B93 repetitions'
        if len(rows)==3 and {r['rep'] for r in rows}=={0,1,2} and all(r['state']=='clean' and r['returncode']==0 for r in rows):
            zz=[verify(r) for r in rows];assert all(z['software']==zz[0]['software'] for z in zz)
            eligible[name]=statistics.median(z['timing']['epoch_seconds'] for z in zz)
            evidence[name]=rows
    assert eligible and eligible==d['eligible_epoch_seconds'],'B93 median arithmetic mismatch'
    selected=min(eligible,key=eligible.get)
    if 'cuda_packed_native' in eligible and eligible['cuda_packed_native']<=eligible[selected]*1.05:selected='cuda_packed_native'
    assert selected==d['selected_path'] and configs[selected]==d['selected_config']
    pr=[r for r in records if r['path']==selected and r.get('phase')=='parallel'];workers=1;gain=None
    if len(pr)==2 and {r['rep'] for r in pr}=={30,31} and all(r['state']=='clean' and r['returncode']==0 for r in pr):
        zz=[verify(r) for r in pr];wall=d['parallel_two_jobs_wall_seconds']
        assert math.isfinite(wall) and wall>=max(z['timing']['epoch_seconds'] for z in zz)>0
        gain=2*eligible[selected]/wall
        assert math.isclose(gain,d['parallel_aggregate_gain'],rel_tol=1e-12)
        if gain>=1.10:workers=2
    assert workers==d['parallel_workers'] and workers in (1,2)
    return {'decision_sha256':sha(path),'selected_path':selected,'config':configs[selected],
            'workers':workers,'serial_medians':eligible,'parallel_gain':gain,'input_binding':binding,
            'actual_batch_binding':members,'source_count':len(d['sources']),
            'independent_b93_audit_sha256':sha(DATA/'.b93/lead_audit93.json')}


def actual_input_binding(q):
    """Reconstruct full real TRAIN/VAL batches and initial weights on CPU only."""
    import os,random
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import numpy as np
    import torch
    sys.path.insert(0,str(BASE))
    import common92 as M
    from model92 import create_model
    p=safe(q['input_binding']['prepared_path'])
    z=torch.load(p,map_location='cpu',weights_only=False)
    assert z['key']==q['input_binding']['prepared_key'] and z['identity']['sources']==M.sources()
    for name,h in z['identity']['raw'].items():assert sha(DATA/'.b73'/name)==h
    manifest=z['identity']['manifest'];tr,va,te=(set(manifest[k]) for k in ('train','val','test'))
    assert tr and va and te and not(tr&va or tr&te or va&te)
    keys=sorted((s['fidx'],s['segment']) for s in z['train'][M.TRAIN_RULES[0]])
    assert keys==z['keys'] and len(keys)==len(set(keys))
    lookup={r:{(s['fidx'],s['segment']):s for s in z['train'][r]} for r in M.TRAIN_RULES}
    train=[lookup[M.B85.rule_for('V1_r2_k0',1701,0,seg)][fid,seg] for fid,seg in keys]
    order=list(range(len(train)));random.Random(1701).shuffle(order);train=[train[i] for i in order]
    h=hashlib.sha256();counts=[0,0]
    def bind(segments,allowed):
        totals=[0,0];n_batches=0
        for start in range(0,len(segments),16):
            ss=segments[start:start+16];lengths=np.array([len(s['x']) for s in ss],dtype=np.int64);L=int(lengths.max())
            x=np.zeros((len(ss),L,35),np.float32);y=np.zeros((len(ss),L,2),np.float32);mask=np.zeros((len(ss),L,2),bool)
            for j,s in enumerate(ss):
                assert s['fidx'] in allowed and s['fidx'] not in te
                n=len(s['x']);x[j,:n]=s['x'];y[j,:n]=s['y'];mask[j,:n]=s['mask']
            for a in (x,y,mask,lengths):h.update(a.tobytes())
            c=mask.sum((0,1));totals=[totals[k]+int(c[k]) for k in range(2)];n_batches+=1
        return n_batches,totals
    n_train,counts=bind(train,tr);vb={};vc={}
    for r in M.TRAIN_RULES:
        h.update(r.encode());vb[r],vc[r]=bind(z['val'][r],va)
    random.seed(1701);np.random.seed(1701);torch.manual_seed(1701)
    model=create_model('C',{**q['config'],'device':'cpu'},1701)
    ih=hashlib.sha256()
    for name,value in model.state_dict().items():ih.update(name.encode());ih.update(value.detach().contiguous().numpy().tobytes())
    actual={'sha256':h.hexdigest(),'train_batches':n_train,'train_segments':len(train),
            'train_rows':sum(len(s['x']) for s in train),'train_supervised_counts':counts,
            'validation_batches':vb,'validation_supervised_counts':vc,'initial_weights_sha256':ih.hexdigest(),
            'seed':1701,'epoch':0,'shuffle_seed':1701}
    assert actual==q['actual_batch_binding'],'Independent real batch/initial weights binding mismatch'
    return {'train_batches':n_train,'validation_rule_count':len(vb),'test_constructed':False,'batch_digest':h.hexdigest()}
