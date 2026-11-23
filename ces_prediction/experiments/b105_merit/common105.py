"""B105 isolated execution/provenance; existing numerical sources stay frozen."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=ROOT/'data';CACHE=DATA/'.b105';PAPER=ROOT/'docs/paper'
sys.path.insert(0,str(HERE.parent/'b94_parallel'))
import common94 as OLD
import common92 as BASEM
from atomic93 import save_json
UNITS=('V1_r2_k0','V2_s1_k0','V3_q3','V4a');SEED=1701
TRAIN_RULES=('D10','D50','D100','D300','D650','D1000','S');RULES=TRAIN_RULES+('K10',)
sha=OLD.sha;digest=OLD.digest

def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,z):return save_json(Path(p),z)
def sources():
    files=[*HERE.glob('*.py'),HERE.parent/'PREREGISTRATION_B105.md',ROOT/'ces_prediction/research/review_b105.py']
    for p in files:OLD.committed(p)
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}
def prov():return load(CACHE/'provenance.json')
def guard():
    z=prov();assert sources()==z['sources'] and OLD.guard()==z['frozen'],'Source drift'
    return {**z['frozen'],**z['sources']}
def inputs_guard():
    for rel,h in prov()['inputs'].items():assert sha(ROOT/rel)==h,'Input artifact drift: '+rel
def runtime():
    import runtime94
    r=runtime94.decision();assert r==prov()['runtime'];return r
def remaining():return load(CACHE/'suite_clock.json')['deadline']-time.time()
def rd(unit,stage='confirm'):return CACHE/stage/unit
def env(unit,stage='confirm'):
    import train92
    return {**train92.ENV_PINS,'CES_SEQ_EPOCHS':'1' if stage=='smoke' else '40','CES_SEQ_DEVICE':'cuda',
            'CES_SEQ_SEED':str(SEED),'CES_B105_OUTPUT_DIR':str(CACHE),'CES_B105_UNIT':unit,
            'CES_B105_CHANNELS':'35','CES_B105_SECONDS':'120' if stage=='smoke' else '3600'}
def prepared(unit):
    import torch
    z=prov()['prepared'][unit];p=ROOT/z['path'];assert sha(p)==z['sha256']
    a=torch.load(p,map_location='cpu',weights_only=False)
    assert a['key']==z['key'] and digest(a['identity']['manifest'])==z['manifest_hash'] and digest(a['identity']['stats'])==z['stats_hash']
    assert a['identity']['raw']==prov()['raw']
    return {**a,'artifact_sha256':z['sha256']}
def prepare():
    CACHE.mkdir(exist_ok=True);assert not (CACHE/'provenance.json').exists(),'Existing provenance; inspect'
    import runtime94
    assert load(DATA/'.b104/lead_completion.json')['all_pass']
    paths={}
    for unit in UNITS:
        old=DATA/'.b94/confirm'/f'{unit}__1701__TX';c=load(old/'config.json');p=DATA/'.b94/prepared'/f"{unit}__{c['prepared_key']}.pt"
        assert sha(p)==c['prepared_artifact_sha256'] and load(old/'process_exit.json')['returncode']==0
        paths[unit]={'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'key':c['prepared_key'],'manifest_hash':c['manifest_hash'],'stats_hash':c['stats_hash'],
                     'original_best_sha256':sha(old/'best.pt'),'original_config_sha256':sha(old/'config.json')}
    files=[DATA/'.b73/grid.npz',DATA/'.b73/rows.npz',DATA/'.b75/sets.npz',DATA/'.b78/sets.npz',PAPER/'b94_methods.json']
    files += [DATA/('.b78' if r=='K10' else '.b75')/'inputs'/f'{r}.npz' for r in RULES]
    z={'sources':sources(),'frozen':OLD.guard(),'runtime':runtime94.decision(),'prepared':paths,
       'raw':{p.name:sha(p) for p in files[:2]},'inputs':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    save(CACHE/'provenance.json',z)
    for u in UNITS:prepared(u)
    print(json.dumps({'prepared':True,'units':len(paths),'scientific_fits':0}))
def verify_run(unit,stage='confirm'):
    import math,torch,numpy as np
    guard();p=rd(unit,stage);c=load(p/'config.json');h=c.pop('config_hash');assert digest(c)==h
    assert c['experiment']=='B105' and c['sources']==sources() and c['frozen']==guard()
    assert c['unit']==unit and c['seed']==SEED and c['code']=='A35' and c['env']==env(unit,stage)
    assert (c['epochs'],c['seconds'])==((1,120) if stage=='smoke' else (40,3600))
    assert c['prepared_artifact_sha256']==prov()['prepared'][unit]['sha256']
    for name,key in [('manifest.json','manifest_hash'),('stats.json','stats_hash')]:assert digest(load(p/name))==c[key]
    m=load(p/'metrics.json');d=load(p/'complete.json');x=load(p/'process_exit.json')
    assert x['returncode']==0 and not x['timeout'] and x['config_hash']==d['config_hash']==h
    assert m['history'] and len(m['history'])==m['completed_epochs']<=c['epochs']
    for i,e in enumerate(m['history']):
        assert e['epoch']==i and set(e['validation'])==set(TRAIN_RULES)
        v=[]
        for a in e['validation'].values():
            sums=np.asarray(a['squared_sums']);n=np.asarray(a['counts']);assert (n>0).all() and np.isfinite(sums).all()
            v.append(float(np.mean(sums/n)));assert math.isclose(v[-1],a['mse'],rel_tol=1e-9,abs_tol=1e-11)
        assert math.isclose(float(np.mean(v)),e['val_loss'],rel_tol=1e-9,abs_tol=1e-11)
    assert d['best_val_loss']==min(e['val_loss'] for e in m['history'])
    assert sha(p/'best.pt')==d['best_sha256'] and sha(p/'latest.pt')==d['latest_sha256']
    w=torch.load(p/'best.pt',map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for v in w.values())
    assert w['stem.weight'].shape==(128,35)
    latest=torch.load(p/'latest.pt',map_location='cpu',weights_only=False)
    assert latest['validated'] and latest['config_hash']==h and latest['epoch']==len(m['history'])-1 and latest['updates']==m['validated_updates']
    return {'epochs':len(m['history']),'best_val_loss':d['best_val_loss'],'stop':m['stop'],'wall_seconds':m['wall_seconds'],'config_hash':h,'best_sha256':d['best_sha256']}
