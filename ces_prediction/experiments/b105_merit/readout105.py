"""Paired report metrics with pointwise conditional cluster/discharge intervals."""
import json,zlib
import numpy as np
import common105 as C
import common92 as M
from evaluate105 import npz,aligned
DRAW=20000

def statistics(err,labels,name):
    # Columns: candidate, reference; common rows, fixed one initialization.
    e2=err**2;out={'point':float(1-e2[:,0].mean()/e2[:,1].mean()),'MSE':e2.mean(0).tolist(),'RMSE':np.sqrt(e2.mean(0)).tolist()}
    for form,lab in labels.items():
        _,inv=np.unique(lab,return_inverse=True);n=inv.max()+1
        sums=np.column_stack([np.bincount(inv,weights=e2[:,j],minlength=n) for j in range(2)])
        counts=np.bincount(inv,minlength=n);rng=np.random.default_rng([20261105,zlib.crc32((name+'/'+form).encode())]);values=[];ms=[]
        for start in range(0,DRAW,500):
            w=rng.multinomial(n,np.full(n,1/n),min(500,DRAW-start));ss=w@sums;cc=w@counts
            q=ss/cc[:,None];ms.append(q);values.append(1-ss[:,0]/ss[:,1])
        v=np.concatenate(values);a=np.concatenate(ms);valid=np.isfinite(v)
        assert valid.all(),'Undefined bootstrap draw; stop, no redraw'
        ci=np.quantile(v,[.025,.975]);out[form]={'ci95':ci.tolist(),'excludes_zero':bool(ci[0]>0 or ci[1]<0),
            'MSE_ci95':np.quantile(a,[.025,.975],axis=0).T.tolist(),'RMSE_ci95':np.sqrt(np.quantile(a,[.025,.975],axis=0)).T.tolist(),'undefined':int((~valid).sum())}
    return out

def specs():
    records=[];passed=C.load(C.CACHE/'gate.json')['all_pass']
    for unit in C.UNITS:
        for split in (('val','test') if passed else ('val',)):
            for rule in C.TRAIN_RULES if split=='val' else C.RULES:
                for h,target in enumerate(('F','L5')):
                    if split=='val':
                        base=C.CACHE/'trees'/unit/f'{rule}__{target}__val.npz';refs=[('fair_specialized',base,None),('fair_mixture',C.CACHE/'trees'/unit/f'{rule}__{target}__val__mixture.npz',None)]
                    else:
                        base=C.CACHE/'trees'/unit/f'{rule}__{target}__test__mixture.npz';refs=[('fair_mixture',base,None)]
                        if rule!='K10':refs.append(('fair_specialized',C.CACHE/'trees'/unit/f'{rule}__{target}__test__specialized.npz',None))
                        refs.append(('incumbent_outerfit',C.DATA/('.b78' if rule=='K10' else '.b75')/'fits'/f'{unit}__TX__{rule}__{target}.npz',None))
                        refs.append(('TCN22',C.CACHE/'predictions'/f'{unit}__TCN22__test__{rule}.npz',h))
                        # Non-learning row-local baselines use committed cached H and G below.
                        for field in ('H','G'):refs.append((field,C.DATA/('.b78' if rule=='K10' else '.b75')/'inputs'/f'{rule}.npz',field))
                    idx=npz(base)['idx']
                    for label,ref,col in refs:
                        records.append({'name':f'{unit}/{split}/{rule}/{target}/TCN35_over_{label}','unit':unit,'split':split,'rule':rule,'target':target,
                            'candidate':str((C.CACHE/'predictions'/f'{unit}__TCN35__{split}__{rule}.npz').relative_to(C.ROOT)),
                            'candidate_head':h,'reference':str(ref.relative_to(C.ROOT)),'reference_head':col,'idx_source':str(base.relative_to(C.ROOT))})
        # Existing predictions only: every available B84 mixture comparison, not selected favorable endpoints.
        for rule in C.RULES:
            ref=C.DATA/'.b84/fits'/f'{unit}__TXall__{rule}.npz'
            if ref.exists():records.append({'name':f'{unit}/existing_test/{rule}/L5/TCN22_over_B84_mixture','unit':unit,'split':'existing_test','rule':rule,'target':'L5',
                'candidate':str((C.DATA/'.b94/predictions'/f'{unit}__1701__TX__{rule}.npz').relative_to(C.ROOT)),'candidate_head':1,
                'reference':str(ref.relative_to(C.ROOT)),'reference_head':None,'idx_source':str(ref.relative_to(C.ROOT))})
    return records

def arrays(spec,R):
    idx=npz(C.ROOT/spec['idx_source'])['idx'];truth=R['tg_'+spec['target']][idx]
    c=aligned(C.ROOT/spec['candidate'],idx,spec['candidate_head'])
    col=spec['reference_head']
    r=npz(C.ROOT/spec['reference'])[col][idx] if isinstance(col,str) else aligned(C.ROOT/spec['reference'],idx,col)
    assert len(idx) and np.isfinite(truth).all() and np.isfinite(c).all() and np.isfinite(r).all()
    return idx,np.column_stack([c.astype(np.float64)-truth,r.astype(np.float64)-truth])

def main():
    C.guard();C.inputs_guard();assert C.load(C.CACHE/'fit_checks.json')['all_pass'];G,R=M.load_sources();gate=C.load(C.CACHE/'gate.json');result={};sources={}
    allspec=specs();C.save(C.CACHE/'pair_specs.json',allspec)
    for s in allspec:
        idx,err=arrays(s,R);lab={'cluster':R['clus'][idx],'discharge':R['fidx'][idx]}
        for k in ('candidate','reference','idx_source'):sources[s[k]]=C.sha(C.ROOT/s[k])
        result[s['name']]={'rows':len(idx),'clusters':len(np.unique(lab['cluster'])),'discharges':len(np.unique(lab['discharge'])),
                           'statistics':statistics(err,lab,s['name'])}
    incumbent=[v['statistics']['cluster']['ci95'][0] for k,v in result.items() if '/test/' in k and k.endswith('TCN35_over_incumbent_outerfit')]
    assert len(incumbent)==(64 if gate['all_pass'] else 0)
    statements={'S1':{'holds':gate['all_pass'],'quantity':'all-unit VAL resource gate, not inferential noninferiority'},
                'S2':{'holds':bool(min(incumbent)>0) if incumbent else None,'status':'evaluated' if incumbent else 'not_evaluated_after_failed_gate','quantity':'all64 incumbent-relative TEST interval lower endpoints >0'},
                'S3':{'holds':bool(min(incumbent)>-.05) if incumbent else None,'status':'evaluated' if incumbent else 'not_evaluated_after_failed_gate','quantity':'all64 incumbent-relative TEST interval lower endpoints >-0.05; no measured operational benefit'}}
    report={'registration':'PREREGISTRATION_B105.md','exploratory':True,'seeds':[C.SEED],'units':list(C.UNITS),'gate':gate,
        'candidate_TEST_evaluated':gate['all_pass'],'draws':DRAW,'seed':20261105,'sources':C.sources(),'prediction_sources':sources,'cells':result,
        'fits':C.load(C.CACHE/'fit_checks.json')['runs'],'pair_specs_sha256':C.sha(C.CACHE/'pair_specs.json'),'statements':statements,
        'legacy_LSTM_correction':{'available':False,'reason':'D1 artifacts exist but original fitted-stage training manifests were not independently bound to this new per-unit deployment contract; no old scripts/obsolete results restored'},
        'limits':['One fixed unit per design and one initialization; no design-wide or initialization-robust claim.',
                  'VAL checkpoint-selection and allocation optimism; passing gate is not statistical noninferiority.',
                  'New candidate TEST only conditional on all-unit VAL gate; existing TEST-informed scope.',
                  'New fair trees use internal TRAIN only and no automatic early stopping; incumbent outer-fit trees differ.',
                  'Fixed-fit pointwise intervals, no refit/selection/init/multiplicity uncertainty. No individual discovery claims.',
                  'F and L5 are different sensor-derived report targets, not independent physical truth.',
                  '35-channel3600s vs old22-channel1200s differs in input/parameter and time budget, not pure architecture effect.',
                  'B84 mixture available on L5/V1,V3,V4 only; missing V2/F not imputed.']}
    C.save(C.CACHE/'readout.json',report)
    lines=['# B.105 estimator merit screening','',*report['limits'],'','Candidate TEST evaluated: '+str(report['candidate_TEST_evaluated']),'',
           '| Unit | VAL candidate / fair specialized MSE | Allocation pass |','|---|---|---|']
    for u,z in gate['units'].items():lines.append(f"| {u} | {z['candidate_to_specialized_ratio']:.6f} | {z['pass']} |")
    lines+=['','Skill = 1 - MSE(candidate) / MSE(reference); point [95% cluster interval], all paired rows.','',
            '| Comparison | Rows | Skill [95%] | Excludes zero |','|---|---|---|---|']
    for name,c in result.items():
        s=c['statistics'];ci=s['cluster']['ci95'];lines.append(f"| {name} | {c['rows']} | {s['point']:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}] | {s['cluster']['excludes_zero']} |")
    lines+=['','Legacy LSTM + correction: unavailable under this batch\'s verified manifest contract; not replaced by GP/tree comparison.',
            'A new independently specified adequately trained common-information comparison on unused observations would overturn a negative interpretation.']
    (C.CACHE/'readout.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('B105 paired readout cells '+str(len(result)),flush=True)
if __name__=='__main__':main()
