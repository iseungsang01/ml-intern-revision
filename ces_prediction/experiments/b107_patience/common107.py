"""Separate continuation namespace; the running B106 source stays immutable."""
import os,sys,time,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=ROOT/'data';CACHE=DATA/'.b107';PAPER=ROOT/'docs/paper'
sys.path.insert(0,str(HERE.parent/'b106_lstm'))
import common106 as OLD
from atomic93 import save_json
sha=OLD.sha;digest=OLD.digest;load=OLD.load;M=OLD.M
CASES=OLD.CASES;UNITS=OLD.UNITS;RULES=OLD.RULES;TRAIN_RULES=OLD.TRAIN_RULES;SEED=OLD.SEED
EPOCH_CAP=400;CHILD_SECONDS=86400;SUITE_SECONDS=259200
def save(p,z):return save_json(Path(p),z)
def sources():
    pp=[*HERE.glob('*.py'),HERE.parent/'PREREGISTRATION_B107.md',HERE.parent/'PREREGISTRATION_B107_ADDENDUM1.md',HERE.parent/'b99_recovery/monitor99.py']
    for p in pp:OLD.PREV.OLD.committed(p)
    return {p.relative_to(ROOT).as_posix():sha(p) for p in pp}
def prov():return load(CACHE/'provenance.json')
def guard():
    z=prov();assert z['sources']==sources() and z['inherited']==OLD.guard()
    assert sha(DATA/'.b106/provenance.json')==z['original_provenance_sha256']
    return {**z['inherited'],**z['sources']}
def inputs_guard():OLD.inputs_guard()
def runtime():guard();return OLD.runtime()
def prepared(case):return OLD.prepared(case)
def rd(case,stage='confirm'):
    p=CACHE/'selected.json'
    if p.exists() and stage=='confirm':return ROOT/load(p)[case]['path']
    return CACHE/stage/case
def new_rd(case):return CACHE/'confirm'/case
def env(case):
    import train92
    return {**train92.ENV_PINS,'CES_SEQ_EPOCHS':str(EPOCH_CAP),'CES_SEQ_DEVICE':'cuda','CES_SEQ_SEED':str(SEED),
            'CES_B107_CASE':case,'CES_B107_OUTPUT_DIR':str(CACHE),'CES_B107_SECONDS':str(CHILD_SECONDS),'CES_B107_TARGET':'F','CES_B107_RULE':'D10'}
def resume_seed(case):return SEED+zlib.crc32(case.encode())
def prepare():
    CACHE.mkdir(exist_ok=True);assert not (CACHE/'provenance.json').exists()
    OLD.guard();OLD.inputs_guard();s=sources();mem=load(DATA/'.b106/memory_checks.json')
    assert mem['all_pass'] and mem['workers']==4 and mem['sources']==OLD.sources()
    z={'sources':s,'inherited':OLD.guard(),'original_provenance_sha256':sha(DATA/'.b106/provenance.json'),
       'reference_sha256':OLD.prov()['reference_sha256'],'memory_reserve_MiB':mem['per_worker_MiB'],'runtime':OLD.runtime()}
    save(CACHE/'provenance.json',z)
    print('B107 registered source/data/runtime preparation passed',flush=True)
def verify_new(case):
    import numpy as np,torch
    guard();p=new_rd(case);cfg=load(p/'config.json');h=cfg.pop('config_hash')
    assert digest(cfg)==h and cfg['sources']==sources() and cfg['frozen']==guard() and cfg['env']==env(case)
    assert cfg['case']==case and cfg['seed']==SEED and cfg['epochs']==EPOCH_CAP and cfg['seconds']==CHILD_SECONDS
    prep=prepared(case)
    assert cfg['prepared_sha256']==prep['artifact_sha256'] and cfg['prepared_key']==prep['key']
    for name in ('manifest','stats'):assert digest(load(p/(name+'.json')))==digest(prep['identity'][name])
    if cfg['parent']:
        for rel,expected in cfg['parent']['hashes'].items():assert sha(ROOT/rel)==expected
    m=load(p/'metrics.json');d=load(p/'complete.json');x=load(p/'process_exit.json');hh=m['history']
    assert x['returncode']==0 and not x['timeout'] and x['config_hash']==d['config_hash']==h
    assert hh and len(hh)==m['completed_epochs']<=EPOCH_CAP and len(hh)>cfg['start_epoch']
    best=float('inf');ni=0
    for i,e in enumerate(hh):
        assert i==e['epoch'] and set(e['validation'])=={'D10'}
        v=e['validation']['D10'];assert v['counts'][0]>0 and v['counts'][1]==0
        assert np.isclose(v['squared_sums'][0]/v['counts'][0],e['val_loss'],rtol=1e-10)
        improved=e['val_loss']<best;assert improved==e['improved']
        if improved:best=e['val_loss'];ni=0
        else:ni+=1
    assert best==d['best_val_loss'] and ni==m['trailing_no_improvement']
    assert sha(p/'best.pt')==d['best_sha256'] and sha(p/'latest.pt')==d['latest_sha256']
    w=torch.load(p/'best.pt',map_location='cpu',weights_only=True);assert all(torch.isfinite(t).all() for t in w.values())
    latest=torch.load(p/'latest.pt',map_location='cpu',weights_only=False)
    assert latest['validated'] and latest['config_hash']==h and latest['epoch']==len(hh)-1 and latest['updates']==m['validated_updates']
    assert all(k in latest for k in ('rng_cpu','rng_cuda','rng_numpy','rng_python'))
    return {'epochs':len(hh),'new_epochs':len(hh)-cfg['start_epoch'],'best_val_loss':best,'stop':m['stop'],
            'patience_met':m['stop']=='patience' and ni>=6,'best_sha256':d['best_sha256'],'wall_seconds':m['wall_seconds'],'parent':cfg['parent']}
def verify_run(case,stage='confirm'):
    selection=load(CACHE/'selected.json')[case]
    if selection['kind']=='original':
        result=OLD.verify_run(case);assert result['stop']=='converged'
        assert selection['path']==OLD.rd(case).relative_to(ROOT).as_posix() and selection['best_sha256']==result['best_sha256']
        history=load(OLD.rd(case)/'metrics.json')['history'];trailing=0
        for e in reversed(history):
            if e['improved']:break
            trailing+=1
        assert trailing>=6
        return {**result,'new_epochs':0,'patience_met':True,'parent':None}
    assert selection['path']==new_rd(case).relative_to(ROOT).as_posix()
    r=verify_new(case);assert r['patience_met'],'Emergency guard hit: no final adequate-training comparison'
    assert r['best_sha256']==selection['best_sha256']
    return r
