"""B104 namespace and immutable source/input provenance; no numerical fitting."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CACHE=ROOT/'data/.b104';BASE=ROOT/'data/.b94'
sys.path.insert(0,str(HERE.parent/'b94_parallel'))
import common94 as OLD
from atomic93 import save_json
UNIT='V3_q3';SEEDS=(1701,1702,1703);CODES=('TX','TH');SHORT=1200;LONG=3600
LABEL={'TX':'FAST','TH':'HISTORY'};JOBS=[(s,c) for c in ('TH','TX') for s in SEEDS]
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,x):return save_json(Path(p),x)
def sources():
    files=[*HERE.glob('*.py'),HERE.parent/'PREREGISTRATION_B104.md']
    out={}
    for p in files:OLD.committed(p);out[p.relative_to(ROOT).as_posix()]=sha(p)
    return out
def prov():return load(CACHE/'provenance.json')
def guard():
    z=prov();assert sources()==z['sources'] and OLD.guard()==z['original_frozen'],'Source drift'
    assert sha(ROOT/'data/.b96/policy96.json')==z['four_policy_sha256']
    return {**z['original_frozen'],**z['sources']}
def runtime():
    import runtime94
    z=runtime94.decision();assert z==prov()['runtime'];return z
def remaining():return max(0,load(CACHE/'suite_clock.json')['deadline']-time.time())
def run_dir(stage,seed,code,observe=True):
    assert stage in ('smoke','confirm') and seed in SEEDS and code in CODES
    return CACHE/stage/f'{seed}__{LABEL[code]}__obs{int(observe)}'
def env(stage,code,seed,observe=True):
    import train92
    return {**train92.ENV_PINS,'CES_SEQ_EPOCHS':'2' if stage=='smoke' else '40','CES_SEQ_DEVICE':'cuda',
        'CES_SEQ_SEED':str(seed),'CES_B104_CODE':code,'CES_B104_OUTPUT_DIR':str(CACHE),
        'CES_B104_OBSERVE':str(int(observe)),'CES_B104_SECONDS':str(120 if stage=='smoke' else LONG),
        'CES_B104_SHORT_SECONDS':str(SHORT)}
def load_prepared():
    import torch
    z=prov();p=ROOT/z['prepared_path'];assert sha(p)==z['prepared_sha256']
    data=torch.load(p,map_location='cpu',weights_only=False)
    assert data['key']==z['prepared_key'] and digest(data['identity']['manifest'])==z['manifest_hash'] and digest(data['identity']['stats'])==z['stats_hash']
    assert data['identity']['raw']==z['raw'],'Cached TRAIN/VAL differs from current raw grid/rows'
    return {**data,'artifact_sha256':z['prepared_sha256']}
def prepare():
    CACHE.mkdir(exist_ok=True);assert not (CACHE/'provenance.json').exists(),'Inspect existing preparation, never overwrite'
    sys.path.insert(0,str(ROOT/'ces_prediction/research/posteval_claude'));import evidence as E
    assert E.gate() is not None,'Original evaluation not audited'
    import runtime94
    rt=runtime94.decision();assert rt['device']=='cuda' and not rt['cudnn']
    sys.path.insert(0,str(HERE.parent/'b96_four'));import common96
    _,four=common96.policy();assert four['selected_workers']==4
    cfg=load(BASE/'confirm'/f'{UNIT}__1701__TX'/'config.json');p=BASE/'prepared'/f"{UNIT}__{cfg['prepared_key']}.pt"
    assert sha(p)==cfg['prepared_artifact_sha256']
    old={}
    for seed,code in JOBS:
        rd=BASE/'confirm'/f'{UNIT}__{seed}__{code}';c=load(rd/'config.json')
        assert load(rd/'process_exit.json')['returncode']==0 and c['prepared_artifact_sha256']==sha(p)
        assert c['stats_hash']==cfg['stats_hash'] and c['manifest_hash']==cfg['manifest_hash']
        old[f'{seed}__{code}']={n:sha(rd/n) for n in ('config.json','metrics.json','complete.json','process_exit.json')}
    z={'sources':sources(),'original_frozen':OLD.guard(),'runtime':rt,'prepared_path':p.relative_to(ROOT).as_posix(),
       'prepared_sha256':sha(p),'prepared_key':cfg['prepared_key'],'manifest_hash':cfg['manifest_hash'],'stats_hash':cfg['stats_hash'],
       'raw':{n:sha(ROOT/'data/.b73'/n) for n in ('grid.npz','rows.npz')},
       'sets75_sha256':sha(ROOT/'data/.b75/sets.npz'),'four_policy_sha256':sha(ROOT/'data/.b96/policy96.json'),'original_runs':old,'old_post_sha256':sha(BASE/'lead_checks/post.json')}
    save(CACHE/'provenance.json',z);load_prepared();print(json.dumps({'prepared':True,'unit':UNIT,'fits':6}),flush=True)
def verify_run(stage,seed,code,observe=True):
    import numpy as np,torch
    torch.set_num_threads(1);guard();rd=run_dir(stage,seed,code,observe)
    c=load(rd/'config.json');h=c.pop('config_hash');assert digest(c)==h
    assert c['experiment']=='B104' and c['sources']==sources() and c['frozen']==guard()
    assert (c['stage'],c['unit'],c['seed'],c['code'])==(stage,UNIT,seed,code)
    assert c['env']==env(stage,code,seed,observe) and c['runtime']==prov()['runtime']
    assert (c['epochs'],c['seconds'])==((2,120) if stage=='smoke' else (40,LONG))
    assert (c['batch'],c['lr'],c['weight_decay'],c['clip'],c['dropout'],c['early_stop'],c['scheduler_patience'],c['factor'])==(16,.001,.0001,1.,.1,6,2,.5)
    assert c['prepared_artifact_sha256']==prov()['prepared_sha256'] and c['manifest_hash']==prov()['manifest_hash'] and c['stats_hash']==prov()['stats_hash']
    assert digest(load(rd/'manifest.json'))==c['manifest_hash'] and digest(load(rd/'stats.json'))==c['stats_hash']
    m=load(rd/'metrics.json');done=load(rd/'complete.json');exit=load(rd/'process_exit.json');obs=load(rd/'observer.json')
    assert exit['returncode']==0 and not exit['timeout'] and exit['config_hash']==h==done['config_hash']==obs['config_hash']
    assert m['history'] and len(m['history'])==m['completed_epochs']<=c['epochs']
    for i,e in enumerate(m['history']):
        assert e['epoch']==i and set(e['validation'])=={'D10','D50','D100','D300','D650','D1000','S'}
        values=[]
        for v in e['validation'].values():
            sums=np.asarray(v['squared_sums']);counts=np.asarray(v['counts']);valid=counts>0;assert valid.any() and np.isfinite(sums).all()
            mse=float((sums[valid]/counts[valid]).mean());assert np.isclose(mse,v['mse'],rtol=1e-10,atol=1e-12);values.append(mse)
        assert np.isclose(np.mean(values),e['val_loss'],rtol=1e-10,atol=1e-12)
    assert done['best_val_loss']==min(e['val_loss'] for e in m['history'])
    assert sha(rd/'best.pt')==done['best_sha256'] and sha(rd/'latest.pt')==done['latest_sha256']
    assert sha(rd/'short_best.pt')==obs['short_sha256'] and obs['rng_unchanged'] and obs['overhead_seconds']<=6
    assert done['best_val_loss']<=obs['short_val_loss']
    if observe:
        assert len(obs['epochs'])==len(m['history'])
        eligible=[]
        for e,o in zip(m['history'],obs['epochs']):
            assert e['epoch']==o['epoch'] and e['val_loss']==o['val_loss'] and o['rng_unchanged']
            assert o['eligible']==bool(max(o['max_train_start'],o['max_val_start'])<SHORT)
            if o['eligible']:eligible.append(e)
        assert eligible and obs['short_val_loss']==min(e['val_loss'] for e in eligible)
    for name in ('best.pt','short_best.pt'):
        w=torch.load(rd/name,map_location='cpu',weights_only=True);assert w and all(torch.isfinite(v).all() for v in w.values())
    latest=torch.load(rd/'latest.pt',map_location='cpu',weights_only=False)
    assert latest['validated'] and latest['config_hash']==h and latest['updates']==m['validated_updates'] and latest['epoch']==len(m['history'])-1
    eligible=[e for e,o in zip(m['history'],obs['epochs']) if o['eligible']] if observe else m['history']
    return {'config_hash':h,'best_val_loss':done['best_val_loss'],'short_val_loss':obs['short_val_loss'],'epochs':len(m['history']),'stop':m['stop'],'short_epoch':obs['short_epoch'],
        'short_eligible_epochs':len(eligible),'short_validated_updates_at_budget':eligible[-1]['updates'],'short_selected_updates':m['history'][obs['short_epoch']]['updates'],
        'long_validated_updates':m['validated_updates'],'wall_seconds':m['wall_seconds'],'prep_seconds':m['prep_seconds'],
        'observer_overhead_seconds':obs['overhead_seconds'],'boundary_flag_count':sum(o.get('boundary_flag',False) for o in obs['epochs'])}
