"""Fresh F-only LSTM + out-of-file correction, sharing B105 outer partitions."""
import json,os,sys,time,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=ROOT/'data';CACHE=DATA/'.b106';PAPER=ROOT/'docs/paper'
sys.path.insert(0,str(HERE.parent/'b105_merit'))
import common105 as PREV
import common92 as M
from atomic93 import save_json
UNITS=PREV.UNITS;SEED=1701;RULES=PREV.RULES;TRAIN_RULES=PREV.TRAIN_RULES
CASES=tuple(u+'__outer' for u in UNITS)+tuple(u+'__inner'+str(k) for k in range(4) for u in UNITS)
sha=PREV.sha;digest=PREV.digest;load=PREV.load
def save(p,z):return save_json(Path(p),z)
def sources():
    paths=[*HERE.glob('*.py'),HERE.parent/'PREREGISTRATION_B106.md']
    for p in paths:PREV.OLD.committed(p)
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
def prov():return load(CACHE/'provenance.json')
def guard():
    p=prov();assert sources()==p['sources'] and PREV.guard()==p['inherited_frozen']
    return {**p['inherited_frozen'],**p['sources']}
def runtime():
    r=PREV.runtime();assert r==prov()['runtime'];return r
def remaining():return load(CACHE/'suite_clock.json')['deadline']-time.time()
def rd(case,stage='confirm'):return CACHE/stage/case
def env(case,stage='confirm'):
    import train92
    return {**train92.ENV_PINS,'CES_SEQ_EPOCHS':'1' if stage=='smoke' else '80','CES_SEQ_DEVICE':'cuda','CES_SEQ_SEED':str(SEED),
            'CES_B106_CASE':case,'CES_B106_OUTPUT_DIR':str(CACHE),'CES_B106_TARGET':'F','CES_B106_TRAIN_RULE':'D10','CES_B106_SECONDS':'120' if stage=='smoke' else '3600'}
def inputs_guard():
    PREV.inputs_guard()
    assert sha(PAPER/'b105_merit.json')==prov()['prior_report_sha256']
    for rel,h in prov()['reference_sha256'].items():assert sha(ROOT/rel)==h,'Reference changed: '+rel
def prepared(case):
    import torch
    meta=load(CACHE/'prepared'/f'{case}.json');p=CACHE/'prepared'/f'{case}.pt';assert sha(p)==meta['sha256']
    z=torch.load(p,map_location='cpu',weights_only=False);assert digest(z['identity'])==meta['key']==z['key']
    return {**z,'artifact_sha256':meta['sha256']}
def prepare():
    import numpy as np,torch
    import data88 as D
    CACHE.mkdir(exist_ok=True);assert not (CACHE/'provenance.json').exists(),'Prior preparation exists; inspect'
    assert load(DATA/'.b105/audit.json')['all_pass'];PREV.guard();PREV.inputs_guard()
    G,R=M.load_sources();plans={}
    for u in UNITS:
        original=PREV.prepared(u);man=original['identity']['manifest'];files=sorted(man['train']);rng=np.random.default_rng([20261106,zlib.crc32(u.encode())])
        if u.startswith('V2'):
            groups={}
            for fid in files:
                cc=np.unique(R['clus'][R['fidx']==fid]);assert len(cc)<=1
                g=int(cc[0]) if len(cc) else 1000000+fid;groups.setdefault(g,[]).append(fid)
            bins=[[] for _ in range(4)]
            for g in rng.permutation(sorted(groups)):min(bins,key=len).extend(groups[int(g)])
        else:bins=[sorted(map(int,x)) for x in np.array_split(rng.permutation(files),4)]
        assert set(sum(bins,[]))==set(files) and sum(map(len,bins))==len(files) and all(bins)
        for role in ['outer']+['inner'+str(k) for k in range(4)]:
            case=u+'__'+role;held=[] if role=='outer' else sorted(bins[int(role[-1])]);train=files if role=='outer' else sorted(set(files)-set(held))
            manifest={'name':case,'parent_unit':u,'train':train,'val':man['val'],'test':man['test'],'held':held}
            assert not set(train)&set(held) and not (set(train)|set(held))&set(man['val']+man['test'])
            stats=original['identity']['stats'] if role=='outer' else D.fit_stats(train,G,R)
            if role=='outer':train_segments=original['train']['D10'];val_segments=original['val']['D10']
            else:train_segments=D.build_segments(train,stats,'D10',G,R);val_segments=D.build_segments(man['val'],stats,'D10',G,R)
            def F_only(ss):
                out=[]
                for s in ss:
                    y=s['y'].copy();mask=s['mask'].copy();y[:,1]=0;mask[:,1]=False;out.append({**s,'y':y,'mask':mask})
                return out
            identity={'manifest':manifest,'stats':stats,'raw':PREV.prov()['raw'],'sources':sources(),'target':'F','rule':'D10','OOF':'file exclusion, not prospective inner validation'}
            key=digest(identity);z={'key':key,'identity':identity,'train':{'D10':F_only(train_segments)},'val':{'D10':F_only(val_segments)}}
            z['keys']=sorted((s['fidx'],s['segment']) for s in z['train']['D10'])
            p=CACHE/'prepared'/f'{case}.pt';p.parent.mkdir(exist_ok=True);torch.save(z,p);save(p.with_suffix('.json'),{'key':key,'sha256':sha(p)})
            plans[case]={'manifest_hash':digest(manifest),'stats_hash':digest(stats),'artifact_sha256':sha(p),'held_count':len(held)}
            print('Prepared '+case,flush=True)
    refs=[]
    for u in UNITS:
        for rule in RULES:
            refs += [DATA/('.b78' if rule=='K10' else '.b75')/'fits'/f'{u}__{name}__{rule}__F.npz' for name in ('T0','TX')]
            refs += [DATA/'.b105/predictions'/f'{u}__{name}__test__{rule}.npz' for name in ('TCN22','TCN35')]
            refs += [DATA/'.b105/trees'/u/f'{rule}__F__test__mixture.npz']
            if rule!='K10':refs += [DATA/'.b105/trees'/u/f'{rule}__F__test__specialized.npz']
    save(CACHE/'provenance.json',{'sources':sources(),'inherited_frozen':PREV.guard(),'runtime':PREV.runtime(),'plans':plans,'prior_report_sha256':sha(PAPER/'b105_merit.json'),
          'reference_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in refs}})
def verify_run(case,stage='confirm'):
    import numpy as np,torch
    guard();p=rd(case,stage);c=load(p/'config.json');h=c.pop('config_hash');assert digest(c)==h and c['sources']==sources() and c['frozen']==guard()
    assert c['experiment']=='B106' and c['stage']==stage and c['unit']==case and c['seed']==SEED and c['code']=='LSTM_F'
    assert c['case']==case and c['target']=='F' and c['rule']=='D10' and c['env']==env(case,stage)
    assert (c['epochs'],c['seconds'])==((1,120) if stage=='smoke' else (80,3600))
    assert c['prepared_artifact_sha256']==prov()['plans'][case]['artifact_sha256']
    assert digest(load(p/'manifest.json'))==prov()['plans'][case]['manifest_hash'] and digest(load(p/'stats.json'))==prov()['plans'][case]['stats_hash']
    d=load(p/'complete.json');x=load(p/'process_exit.json');m=load(p/'metrics.json');assert x['returncode']==0 and not x['timeout'] and x['config_hash']==d['config_hash']==h
    assert m['history'] and len(m['history'])==m['completed_epochs']<=c['epochs']
    assert d['best_val_loss']==min(e['val_loss'] for e in m['history'])
    for i,e in enumerate(m['history']):
        assert set(e['validation'])=={'D10'}
        v=e['validation']['D10'];assert e['epoch']==i and v['counts'][0]>0 and v['counts'][1]==0
        assert np.isclose(v['squared_sums'][0]/v['counts'][0],e['val_loss'],rtol=1e-10)
    assert sha(p/'best.pt')==d['best_sha256'] and sha(p/'latest.pt')==d['latest_sha256']
    w=torch.load(p/'best.pt',map_location='cpu',weights_only=True);assert w and all(torch.isfinite(t).all() for t in w.values())
    latest=torch.load(p/'latest.pt',map_location='cpu',weights_only=False)
    assert latest['validated'] and latest['config_hash']==h and latest['epoch']==len(m['history'])-1 and latest['updates']==m['validated_updates']
    return {'epochs':len(m['history']),'best_val_loss':d['best_val_loss'],'stop':m['stop'],'best_sha256':d['best_sha256'],'wall_seconds':m['wall_seconds']}
