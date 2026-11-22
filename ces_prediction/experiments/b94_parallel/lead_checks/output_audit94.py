"""Independent post-fit arithmetic and evidence-scope audit; no training.

TEST labels/predictions are opened only after all registered fits and the
immutable validation-only selection have passed independent checks.
"""
import math,sys,zlib
import numpy as np
from audit94 import BASE,CACHE,DATA,ROOT,RULES,PILOT_UNITS,prior,pilot,verify_run,audit_sources
from runtime_audit94 import load,sha,digest

UNITS=tuple([f'V1_r2_k{k}' for k in range(5)]+[f'V2_s1_k{k}' for k in range(5)]+[f'V3_q{k}' for k in range(1,5)]+['V4a'])

def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def close(a,b):
    assert np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=False),(a,b)

def skill(name,errors,control,labels,record):
    """Recompute registered paired-init then shot/cluster bootstrap independently."""
    num,den=errors**2,control**2
    values=1-num.sum(1)/den.sum(1)
    close(record['point'],values.mean());close(record['per_init'],values)
    for form,lab in labels.items():
        _,inv=np.unique(lab,return_inverse=True);n=int(inv.max())+1
        totals=[np.stack([np.bincount(inv,num[i],minlength=n),np.bincount(inv,den[i],minlength=n)],axis=1) for i in range(len(num))]
        rng=np.random.default_rng([20261192,zlib.crc32((name+'/'+form).encode())]);picks=rng.integers(0,len(num),20000);out=np.empty(20000)
        for i in range(len(num)):
            positions=np.flatnonzero(picks==i)
            for start in range(0,len(positions),500):
                pos=positions[start:start+500];weights=rng.multinomial(n,np.full(n,1/n),len(pos));s=weights@totals[i]
                with np.errstate(divide='ignore',invalid='ignore'):out[pos]=1-s[:,0]/s[:,1]
        finite=out[np.isfinite(out)];assert len(finite)
        close(record[form],np.quantile(finite,[.025,.975]))
        assert record.get(form+'_undefined_draws',0)==20000-len(finite)
    assert record['excludes_zero']==bool(record['cluster'][0]>0 or record['cluster'][1]<0)

def post():
    p=prior();selected=pilot(allow_test_artifacts=True);chosen=selected['chosen'];assert chosen in ('A','T')
    original_gate=load(CACHE/'lead_checks/pilot.json')
    assert original_gate['passed'] and original_gate['chosen']==chosen and original_gate['selection_sha256']==sha(CACHE/'selection.json')
    assert original_gate['audit_source_hashes']==audit_sources() and original_gate['test_accessed'] is False
    codes=('A','C') if chosen=='A' else ('TX','TH')
    jobs=[('confirm',u,1701,c) for u in UNITS for c in codes]+[('confirm',u,s,c) for u in PILOT_UNITS for s in (1702,1703) for c in codes]
    fit_records={}
    for job in jobs:
        stage,u,seed,c=job;rd=CACHE/stage/f'{u}__{seed}__{c}';fit_records[u,seed,c]=verify_run(rd,job,p)
    for stage,cc in [('smoke',('C','A','T')),('smoke_confirm',codes)]:
        for c in cc:verify_run(CACHE/stage/f'{PILOT_UNITS[0]}__1701__{c}',(stage,PILOT_UNITS[0],1701,c),p)
    checks=load(CACHE/'prediction_checks.json')
    assert checks['all_pass'] and checks['labels_not_inputs'] and checks['sources']==p['sources'] and checks['selection_hash']==sha(CACHE/'selection.json')
    # All preconditions above are TRAIN/VAL and completion metadata only.
    R=npz(DATA/'.b73/rows.npz');n=len(R['gi']);assert len(np.unique(R['gi']))==n
    sets75=npz(DATA/'.b75/sets.npz');sets78=npz(DATA/'.b78/sets.npz')
    report=load(CACHE/'readout.json')
    assert report==load(CACHE/'reports/b94_methods.json')
    assert report['registration']=='PREREGISTRATION_B94.md' and report['selection']==load(CACHE/'selection.json')
    assert report['bootstrap_draws']==20000 and report['bootstrap_seed']==20261192
    assert set(report['scopes'])=={'FULL15_one_seed','FOUR_units_three_seeds'}
    predictions={};support_by_unit={};prediction_hashes={}
    for _,u,s,c in jobs:
        for rule in (*RULES,'K10'):
            path=CACHE/'predictions'/f'{u}__{s}__{c}__{rule}.npz';z=npz(path);idx=z['idx'];est=z['est']
            assert idx.ndim==1 and len(idx) and idx.dtype.kind in 'iu' and np.all(np.diff(idx)>0)
            assert idx.min()>=0 and idx.max()<n and est.shape==(len(idx),2) and np.isfinite(est).all()
            assert str(z['config_hash'])==fit_records[u,s,c]['config_hash'] and str(z['sources_hash'])==digest(p['sources'])
            if u in support_by_unit:assert np.array_equal(support_by_unit[u],idx)
            else:support_by_unit[u]=idx
            manifest=load(CACHE/'confirm'/f'{u}__{s}__{c}'/'manifest.json')
            assert set(R['fidx'][idx]).issubset(set(manifest['test']))
            predictions[u,s,c,rule]=z;prediction_hashes[path.name]=sha(path)
    checked=0
    for scope,units,seeds in [('FULL15_one_seed',UNITS,[1701]),('FOUR_units_three_seeds',tuple(u for u in UNITS if u in PILOT_UNITS),[1701,1702,1703])]:
        sr=report['scopes'][scope];assert sr['seeds']==seeds and sr['units']==list(units) and set(sr['designs'])=={'V1','V2','V3','V4a'}
        for design in ('V1','V2','V3','V4a'):
            group=[u for u in units if u.startswith(design)];rec=sr['designs'][design]
            assert set(rec['cells'])=={r+'/'+t for r in (*RULES,'K10') for t in ('F','L5')}|{'D10/F_full'}
            for rule in (*RULES,'K10'):
                pooled={c:np.full((len(seeds),n,2),np.nan) for c in codes};support=None
                for c in codes:
                    for j,s in enumerate(seeds):
                        seen=np.zeros(n,bool)
                        for u in group:
                            z=predictions[u,s,c,rule];idx=z['idx'];assert not seen[idx].any();seen[idx]=True;pooled[c][j,idx]=z['est']
                        if support is None:support=seen
                        else:assert np.array_equal(support,seen)
                for head,target in enumerate(('F','L5')):
                    keys=[rule+'/'+target]+(['D10/F_full'] if (rule,target)==('D10','F') else [])
                    for key in keys:
                        mask=np.ones(n,bool) if key.endswith('_full') else (sets78 if rule=='K10' else sets75)[('S_' if rule=='S' else 'common_')+target]
                        truth=R['tg_'+target];idx=np.flatnonzero(support&mask&np.isfinite(truth));cell=rec['cells'][key]
                        labels={'cluster':R['clus'][idx],'discharge':R['fidx'][idx]}
                        assert cell['rows']==len(idx) and cell['shots']==len(np.unique(labels['discharge'])) and cell['clusters']==len(np.unique(labels['cluster']))
                        assert cell['excluded_from_prediction_support']==int(support.sum()-len(idx))
                        errs={c:pooled[c][:,idx,head]-truth[idx] for c in codes}
                        for c,e in errs.items():
                            met=cell['metrics'][c]
                            for metric,v in [('RMSE',np.sqrt((e**2).mean(1))),('MAE',np.abs(e).mean(1)),('medianAE',np.median(np.abs(e),axis=1))]:
                                close(met[metric]['point'],v.mean());close(met[metric]['per_init'],v)
                        skill(scope+'/'+design+'/'+key+'/pair',errs[codes[0]],errs[codes[1]],labels,cell['skill']);checked+=1
            for sid,key,threshold in [('S1','D10/F_full',-.05),('S2','D650/L5',0),('S3','S/L5',0),('S4','K10/L5',0)]:
                st=rec['statements'][sid];expected={**rec['cells'][key]['skill'],'cell':key,'threshold':threshold,'holds':bool(rec['cells'][key]['skill']['cluster'][0]>threshold),'expectation':'uncertain'}
                assert st==expected
    assert set(report['latency'])==set(codes)
    for z in report['latency'].values():
        assert z['threads']==1 and 0<z['samples']<=1000 and all(math.isfinite(z[k]) and z[k]>=0 for k in ('median_ms','p99_ms','preprocessing_ms_per_row'))
        assert z['p99_ms']>=z['median_ms']
    expected_budgets={str((CACHE/stage/f'{u}__{s}__{c}').relative_to(ROOT)):load(CACHE/stage/f'{u}__{s}__{c}'/'metrics.json') for stage,u,s,c in [('pilot',u,1701,c) for u in PILOT_UNITS for c in ('C','A','T')]+jobs}
    assert report['fit_budgets']==expected_budgets
    return {'passed':True,'phase':'post','sources':p['sources'],'audit_source_hashes':audit_sources(),'test_accessed':True,
            'selection_sha256':sha(CACHE/'selection.json'),'confirmation_fits':len(jobs),'cells_independently_recomputed':checked,
            'prediction_hashes':prediction_hashes,'readout_sha256':sha(CACHE/'readout.json'),
            'limits':['Main-pair point metrics and full registered skill bootstrap independently recomputed; supplementary-tree and median-error interval algorithms remain frozen-source audited.','Latency timings and child watchdog wall capture are source-audited, not repeated.']}
