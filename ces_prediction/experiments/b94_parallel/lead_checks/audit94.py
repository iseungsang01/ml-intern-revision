"""Independent B94 gates. No fit, no TEST construction in pre/prepared/pilot.

Only post may read TEST predictions, after every confirmation/exit gate passes.
"""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
from runtime_audit94 import ROOT,DATA,BASE,load,sha,digest,safe,committed,source_closure,qualification,actual_input_binding

HERE=Path(__file__).resolve().parent;TOP=HERE.parent;CACHE=DATA/'.b94'
RULES=('D10','D50','D100','D300','D650','D1000','S')
PILOT_UNITS=('V1_r2_k0','V2_s1_k0','V3_q3','V4a')


def audit_sources():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in HERE.glob('*.py')}


def source_identity():
    reg=TOP.parent/'PREREGISTRATION_B94.md';pin=TOP/'frozen_sources94.json'
    for p in [reg,pin,*TOP.glob('*.py'),*HERE.glob('*.py')]:committed(p)
    frozen=load(pin);required=source_closure()
    # B94 freezes B93 runtime methods but need not repeat the B93 registration
    # inside its inherited manifest: the qualified decision separately binds it.
    for rel,h in required.items():
        if rel.endswith('/PREREGISTRATION_B93.md'):continue
        assert frozen.get(rel)==h,'Missing original numeric-source closure: '+rel
    owned=[*TOP.glob('*.py'),*HERE.glob('*.py')]
    extra=[TOP.parent/'b93_gpu'/n for n in ('durable93.py','status93.py')]
    for path in owned+extra:
        assert frozen.get(path.relative_to(ROOT).as_posix())==sha(path),'Execution/audit module not frozen: '+str(path)
    for rel,h in frozen.items():assert sha(safe(rel))==h,'B94 frozen source mismatch: '+rel
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in TOP.glob('*.py')}
    assert sources and all(n in {Path(p).name for p in sources} for n in ('common94.py','runtime94.py','adapter94.py','bootstrap94.py','scheduler94.py'))
    clock=load(DATA/'.b92/suite_clock.json')
    # The registered CPU revision continued the original clock while binding
    # updated B92 execution sources; its provenance must never reset the cap.
    continued=load(DATA/'.b92/cpu_r2/suite_clock.json')
    assert continued['continued_from']=='data/.b92/suite_clock.json'
    assert continued['previous_clock_sha256']==sha(DATA/'.b92/suite_clock.json')
    assert float(continued['started'])==float(clock['started']),'CPU continuation reset original suite clock'
    assert continued['sources']=={p.name:sha(p) for p in BASE.glob('*.py')},'Continued B92 source identity changed'
    expected={'started':float(clock['started']),'deadline':float(clock['started'])+86400,
              'original_clock_sha256':sha(DATA/'.b92/suite_clock.json'),'sources':sources}
    if (CACHE/'suite_clock.json').is_file():assert load(CACHE/'suite_clock.json')==expected,'B94 reset or rebound the suite clock'
    assert expected['deadline']>expected['started']
    return sources,frozen,expected


def pre():
    sources,frozen,clock=source_identity();q=qualification()
    assert time.time()<clock['deadline'],'Original suite deadline already elapsed'
    binding=actual_input_binding(q)
    return {'passed':True,'phase':'pre','sources':sources,'frozen':frozen,
            'audit_source_hashes':audit_sources(),'clock':clock,'runtime':q,
            'real_input_recomputation':binding,'test_accessed':False}


def prior():
    sources,frozen,clock=source_identity();z=load(CACHE/'lead_checks/pre.json')
    assert z['passed'] and z['sources']==sources and z['frozen']==frozen
    assert z['audit_source_hashes']==audit_sources(),'Independent audit source changed after pre gate'
    assert z['clock']==clock and z['runtime']['decision_sha256']==sha(DATA/'.b93/runtime_decision93.json')
    assert z['runtime']['independent_b93_audit_sha256']==sha(DATA/'.b93/lead_audit93.json')
    return z


def prepared():
    p=prior();paths=sorted((CACHE/'prepared').glob('*.pt'));assert paths,'No B94 prepared inputs'
    old_path=CACHE/'lead_checks/prepared.json'
    old=load(old_path) if old_path.exists() else {}
    old_entries=old.get('units',{}) if old.get('audit_source_hashes')==audit_sources() and old.get('sources')==p['sources'] else {}
    import numpy as np
    import torch
    result={}
    for path in paths:
        assert path.resolve().is_relative_to(CACHE/'prepared')
        meta=load(path.with_suffix('.json'));file_sha=sha(path)
        assert meta['sha256']==file_sha
        saved=old_entries.get(path.name)
        if saved and saved['artifact_sha256']==file_sha and saved['metadata_sha256']==sha(path.with_suffix('.json')):
            result[path.name]=saved;continue
        z=torch.load(path,map_location='cpu',weights_only=False);identity=z['identity']
        assert meta['key']==z['key']==digest(identity) and path.name.endswith('__'+z['key']+'.pt')
        assert identity['sources']==p['sources'] and identity['frozen']==p['frozen'] and identity['width']==35
        for name,h in identity['raw'].items():assert sha(DATA/'.b73'/name)==h
        manifest=identity['manifest'];tr,va,te=(set(manifest[k]) for k in ('train','val','test'))
        assert tr and va and te and not(tr&va or tr&te or va&te)
        assert set(identity['stats']['train_files'])==tr
        assert identity['stats']['ti_sd']>0 and math.isfinite(identity['stats']['ti_mu'])
        assert set(z['train'])==set(z['val'])==set(RULES)
        reference={};counts={};rows={}
        for part,allowed in [('train',tr),('val',va)]:
            reference[part]={};rows[part]={};counts[part]={}
            for rule in RULES:
                keys=[];total=np.zeros(2,dtype=np.int64);nrows=0
                for s in z[part][rule]:
                    x,y,mask,gi=(s[k] for k in ('x','y','mask','gi'));n=len(x)
                    key=(s['fidx'],s['segment']);keys.append(key)
                    assert s['fidx'] in allowed and s['fidx'] not in te
                    assert x.shape==(n,35) and y.shape==mask.shape==(n,2) and gi.shape==(n,)
                    assert n and np.isfinite(x).all() and np.isfinite(y).all() and np.diff(gi).min(initial=1)>0
                    assert mask.dtype==bool and np.all(y[~mask]==0)
                    assert np.all(x[:,13:15]==0) and np.all(x[:,19:22]==0)
                    label_digest=hashlib.sha256(y.tobytes()+mask.tobytes()+gi.tobytes()).hexdigest()
                    if rule==RULES[0]:reference[part][key]=label_digest
                    else:assert reference[part].get(key)==label_digest,'Availability rule changed labels or membership'
                    total+=mask.sum(0);nrows+=n
                assert len(keys)==len(set(keys)) and sorted(keys)==sorted(reference[part])
                counts[part][rule]=total.tolist();rows[part][rule]=nrows
        assert z['keys']==sorted(reference['train']) and len(z['keys'])==len(set(z['keys']))
        result[path.name]={'key':z['key'],'artifact_sha256':file_sha,'metadata_sha256':sha(path.with_suffix('.json')),
                           'manifest_hash':digest(manifest),'stats_hash':digest(identity['stats']),
                           'supervised_counts':counts,'rows':rows,'test_constructed':False}
        del z
    return {'passed':True,'phase':'prepared','sources':p['sources'],'audit_source_hashes':audit_sources(),
            'units':result,'test_accessed':False,'normalization':'Frozen audited builder and raw/manifest/statistic hashes; no TEST construction or new statistical fitting.'}


def verify_run(rd,job,p=None):
    p=prior() if p is None else p
    stage,unit,seed,code=job;rd=Path(rd)
    assert rd.resolve()==(CACHE/stage/f'{unit}__{seed}__{code}').resolve(),'Run escaped B94 namespace'
    c=load(rd/'config.json');h=c.pop('config_hash');assert digest(c)==h
    assert c['sources']==p['sources'] and c['frozen']==p['frozen']
    assert c['experiment']=='B94' and c['registration']=='PREREGISTRATION_B94.md'
    assert (c['stage'],c['unit'],c['seed'],c['code'])==job
    assert c['training_loop_sha256']==sha(BASE/'train92.py')
    r=c['runtime'];assert 'production_override_sha256' not in r
    assert all(r[k]==v for k,v in p['runtime']['config'].items())
    assert r['b93_decision_sha256']==p['runtime']['decision_sha256'] and r['selected_path']==p['runtime']['selected_path']
    assert r['parallel_workers']==p['runtime']['workers']
    assert c['scheduled_concurrency'] in (1,2) and c['scheduled_concurrency']<=r['parallel_workers']
    assert 0<=c['worker_slot']<c['scheduled_concurrency']
    epochs,seconds=(1,120) if stage.startswith('smoke') else (12,600) if stage=='pilot' else (40,1200)
    assert (c['epochs'],c['seconds'],c['batch'],c['dropout'],c['early_stop'])==(epochs,seconds,16,.1,6)
    assert (c['lr'],c['weight_decay'],c['clip'],c['scheduler_patience'],c['factor'])==(.001,.0001,1.,2,.5)
    env=c['env'];assert env['CES_SEQ_DEVICE']=='cuda' and env['CES_SEQ_BATCH']=='16'
    assert env['CES_SEQ_EPOCHS']==str(epochs) and env['CES_DROP_STUCK_TARGETS']=='1' and env['CES_TI_SPIKE_CUT_EV']=='3000'
    assert env['CES_PER_SHOT_NORM']==env['CES_SEQ_NORM_EXPANDING']=='0'
    assert env['CES_B94_CODE']==code and Path(env['CES_B94_OUTPUT_DIR']).resolve()==CACHE.resolve()
    assert not any(k.startswith('CES_B92_') for k in env)
    prepared_path=CACHE/'prepared'/f"{unit}__{c['prepared_key']}.pt"
    assert sha(prepared_path)==c['prepared_artifact_sha256']
    prep_audit=load(CACHE/'lead_checks/prepared.json');entry=prep_audit['units'][prepared_path.name]
    assert prep_audit['passed'] and prep_audit['sources']==p['sources'] and prep_audit['audit_source_hashes']==audit_sources()
    assert entry['artifact_sha256']==c['prepared_artifact_sha256']
    assert digest(load(rd/'manifest.json'))==c['manifest_hash']==entry['manifest_hash']
    assert digest(load(rd/'stats.json'))==c['stats_hash']==entry['stats_hash']
    done,exit,metrics=(load(rd/f) for f in ('complete.json','process_exit.json','metrics.json'))
    assert exit['returncode']==0 and not exit['timeout'] and exit['config_hash']==done['config_hash']==h
    assert exit['experiment']=='B94' and exit['worker_slot']==c['worker_slot']
    assert metrics['completed_epochs']==len(metrics['history']) and 0<len(metrics['history'])<=epochs
    assert math.isfinite(done['best_val_loss']) and done['best_val_loss']>=0
    for i,e in enumerate(metrics['history']):
        assert e['epoch']==i and set(e['validation'])==set(RULES)
        losses=[]
        for rule,z in e['validation'].items():
            assert len(z['squared_sums'])==len(z['counts'])==2 and all(n>0 for n in z['counts'])
            assert all(math.isfinite(v) and v>=0 for v in z['squared_sums'])
            v=sum(s/n for s,n in zip(z['squared_sums'],z['counts']))/2
            assert math.isclose(v,z['mse'],rel_tol=1e-10,abs_tol=1e-12);losses.append(v)
            if not stage.startswith('smoke'):assert z['counts']==entry['supervised_counts']['val'][rule]
        assert math.isclose(sum(losses)/7,e['val_loss'],rel_tol=1e-10,abs_tol=1e-12)
    assert done['best_val_loss']==min(e['val_loss'] for e in metrics['history'])
    assert metrics['validated_updates']==metrics['history'][-1]['updates']
    assert metrics['discarded_incomplete_epoch_updates']==metrics['updates']-metrics['validated_updates']>=0
    assert sha(rd/'best.pt')==done['best_sha256'] and sha(rd/'latest.pt')==done['latest_sha256']
    import torch
    best=torch.load(rd/'best.pt',map_location='cpu',weights_only=True)
    latest=torch.load(rd/'latest.pt',map_location='cpu',weights_only=False)
    assert best and all(torch.isfinite(v).all() for v in best.values())
    assert latest['validated'] and latest['config_hash']==h and latest['epoch']==metrics['history'][-1]['epoch']
    assert latest['updates']==metrics['validated_updates']
    return {'config_hash':h,'best_val_loss':done['best_val_loss'],
            'files':{n:sha(rd/n) for n in ('config.json','complete.json','process_exit.json','metrics.json','best.pt','latest.pt')}}


def pilot(allow_test_artifacts=False):
    p=prior();selection=load(CACHE/'selection.json');values={};inputs={}
    assert selection['sources']==p['sources'] and selection['test_accessed'] is False
    for unit in PILOT_UNITS:
        for code in ('C','A','T'):
            z=verify_run(CACHE/'pilot'/f'{unit}__1701__{code}',('pilot',unit,1701,code),p)
            values[unit,code]=z['best_val_loss'];inputs[unit+'|'+code]={'best_val_loss':z['best_val_loss'],'files':z['files']}
    assert inputs==selection['inputs'],'Pilot decision does not bind exact validated artifacts'
    candidates={}
    for code in ('A','T'):
        assert all(values[u,'C']>0 for u in PILOT_UNITS)
        ratios={u:values[u,code]/values[u,'C'] for u in PILOT_UNITS};mean=sum(ratios.values())/4
        candidates[code]={'ratios':ratios,'mean':mean,'eligible':mean<.98 and ratios['V3_q3']<=1.05 and ratios['V4a']<=1.05}
        actual=selection['candidates'][code]
        assert actual['ratios']==ratios and math.isclose(actual['mean'],mean,rel_tol=1e-14) and actual['eligible']==candidates[code]['eligible']
    eligible=[c for c in ('A','T') if candidates[c]['eligible']];chosen=None
    if len(eligible)==1:chosen=eligible[0]
    elif len(eligible)==2:chosen='A' if abs(candidates['A']['mean']-candidates['T']['mean'])<=.005 else min(eligible,key=lambda c:candidates[c]['mean'])
    assert chosen==selection['chosen']
    if not allow_test_artifacts:
        assert not (CACHE/'predictions').exists() and not (CACHE/'readout.json').exists(),'TEST artifacts existed before the confirmation gate'
    return {'passed':True,'phase':'pilot','sources':p['sources'],'audit_source_hashes':audit_sources(),
            'selection_sha256':sha(CACHE/'selection.json'),'chosen':chosen,'pilot_fits':12,'test_accessed':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=('pre','prepared','pilot','post'));ap.add_argument('--output',type=Path);args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    try:
        if args.phase=='post':
            from output_audit94 import post
            result=post()
        else:result={'pre':pre,'prepared':prepared,'pilot':pilot}[args.phase]()
        result['audited_at']=time.time()
        if args.output:
            path=args.output.resolve();assert path.is_relative_to(CACHE/'lead_checks')
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,sort_keys=True,indent=2),encoding='utf8');tmp.replace(path)
        print(json.dumps({'passed':True,'phase':args.phase,'test_accessed':result.get('test_accessed',False)},sort_keys=True))
    except BaseException as exc:
        print(json.dumps({'passed':False,'phase':args.phase,'error':type(exc).__name__+': '+str(exc)},sort_keys=True));raise


if __name__=='__main__':main()
