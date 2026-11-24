"""MSE-first common-row comparison, sharing bootstrap draws across every model."""
import zlib
import numpy as np
import common106 as C
from predict106 import npz

def align(path,idx,head=None):
    z=npz(path);j=np.searchsorted(z['idx'],idx);assert (j<len(z['idx'])).all() and np.array_equal(z['idx'][j],idx)
    p=z['est'][j];return p if head is None else p[:,head]
def cell_arrays(unit,rule,R):
    m=C.load(C.rd(unit+'__outer')/'manifest.json');s=npz(C.DATA/('.b78' if rule=='K10' else '.b75')/'sets.npz')
    keep=s[('S_' if rule=='S' else 'common_')+'F']&np.isin(R['fidx'],m['test']);idx=np.flatnonzero(keep)
    z=npz(C.CACHE/'estimates'/f'{unit}__{rule}.npz');j=np.searchsorted(z['idx'],idx);assert np.array_equal(z['idx'][j],idx)
    models={k:z[k][j] for k in ('LSTM','LSTM_frozen_D10','LSTM_mixed_post')}
    root=C.DATA/('.b78' if rule=='K10' else '.b75')/'fits'
    for label,name in [('GP_history_boost','T0'),('GP_fast_boost','TX')]:models[label]=align(root/f'{unit}__{name}__{rule}__F.npz',idx)
    for label in ('TCN22','TCN35'):models[label]=align(C.DATA/'.b105/predictions'/f'{unit}__{label}__test__{rule}.npz',idx,0)
    root=C.DATA/'.b105/trees'/unit;models['GP_fair_mixture']=align(root/f'{rule}__F__test__mixture.npz',idx)
    if rule!='K10':models['GP_fair_specialized']=align(root/f'{rule}__F__test__specialized.npz',idx)
    raw=npz(C.DATA/('.b78' if rule=='K10' else '.b75')/'inputs'/f'{rule}.npz');models['GP_only']=raw['G'][idx];models['hold']=raw['H'][idx]
    matrix=np.column_stack(list(models.values())).astype(np.float64);assert len(idx) and np.isfinite(matrix).all() and np.isfinite(R['tg_F'][idx]).all()
    return idx,list(models),matrix

def compute(err,labels,name,names):
    square=err**2;means=square.mean(0);result={k:{'MSE':float(means[j]),'RMSE':float(np.sqrt(means[j]))} for j,k in enumerate(names)}
    pairs=[(a,b) for a in ('LSTM','LSTM_frozen_D10','LSTM_mixed_post') for b in ('GP_fast_boost','TCN35','GP_only')]
    pairs += [('LSTM_frozen_D10','LSTM'),('LSTM_mixed_post','LSTM'),('GP_fast_boost','GP_history_boost')]
    pi=[(names.index(a),names.index(b)) for a,b in pairs];contrasts={a+'_over_'+b:{'point':float(1-means[i]/means[j])} for (a,b),(i,j) in zip(pairs,pi)}
    for form,lab in labels.items():
        _,inv=np.unique(lab,return_inverse=True);n=inv.max()+1;sums=np.column_stack([np.bincount(inv,weights=square[:,k],minlength=n) for k in range(len(names))]);count=np.bincount(inv,minlength=n)
        rng=np.random.default_rng([20261106,zlib.crc32((name+'/'+form).encode())]);draws=[]
        for start in range(0,20000,500):
            w=rng.multinomial(n,np.full(n,1/n),500);draws.append((w@sums)/(w@count)[:,None])
        draws=np.concatenate(draws);assert np.isfinite(draws).all()
        ci=np.quantile(draws,[.025,.975],axis=0)
        for j,k in enumerate(names):result[k][form+'_MSE95']=ci[:,j].tolist();result[k][form+'_RMSE95']=np.sqrt(ci[:,j]).tolist()
        for (a,b),(i,j) in zip(pairs,pi):
            v=1-draws[:,i]/draws[:,j];assert np.isfinite(v).all();ci=np.quantile(v,[.025,.975]);contrasts[a+'_over_'+b][form]={'ci95':ci.tolist(),'excludes_zero':bool(ci[0]>0 or ci[1]<0)}
    return {'models':result,'pairs':contrasts}

def main():
    C.guard();C.inputs_guard();G,R=C.M.load_sources();cells={};hashes={}
    for u in C.UNITS:
        for rule in C.RULES:
            idx,names,matrix=cell_arrays(u,rule,R);labels={'cluster':R['clus'][idx],'discharge':R['fidx'][idx]};key=u+'/'+rule+'/F'
            cells[key]={'rows':len(idx),'clusters':len(np.unique(labels['cluster'])),'discharges':len(np.unique(labels['discharge'])),**compute(matrix-R['tg_F'][idx,None],labels,key,names)}
            p=C.CACHE/'estimates'/f'{u}__{rule}.npz';hashes[p.relative_to(C.ROOT).as_posix()]=C.sha(p)
    report={'registration':'PREREGISTRATION_B106.md','target':'current_report_F','exploratory':True,'units':list(C.UNITS),'seeds':[C.SEED],'sources':C.sources(),
        'draws':20000,'bootstrap_seed':20261106,'cells':cells,'estimate_sha256':hashes,'reference_sha256':C.prov()['reference_sha256'],
        'fits':C.load(C.CACHE/'fit_checks.json')['fits'],'limits':['One initialization and one unit per design; no independent campaign confirmation.',
        'LSTM Ti architecture retained, newD1/outer manifests/F-only/internal-TRAIN OOF protocol; not old checkpoint reproduction.',
        'frozen_D10 is a D10-trained correction; mixed_post is a declared extension; first-stage LSTM remains D10-trained.',
        'Cross-fitting is file-exclusion calibration within historical TRAIN, not prospective temporal inner evaluation.',
        'TCN has different F/L5/regime objective and budget; deployment comparison, not pure architecture attribution.',
        'Pointwise fixed-fit95% intervals omit refit/init/model-selection uncertainty; no individual discovery claims.']}
    fresh=[cells[u+'/D10/F']['pairs']['LSTM_frozen_D10_over_GP_fast_boost']['cluster']['ci95'][0] for u in C.UNITS]
    allmix=[c['pairs']['LSTM_mixed_post_over_GP_fast_boost']['cluster']['ci95'][0] for c in cells.values()]
    fast=[cells[u+'/D10/F']['pairs']['GP_fast_boost_over_GP_history_boost']['cluster']['ci95'][0] for u in C.UNITS]
    report['statements']={'S1_frozen_LSTM_post_over_GP_all_fresh':bool(min(fresh)>0),'S2_mixed_post_over_GP_all_scenarios':bool(min(allmix)>0),'S3_GP_fast_increment_all_fresh':bool(min(fast)>0)}
    C.save(C.CACHE/'readout.json',report)
    lines=['# B106: LSTM + correction versus GP/boosting and TCN','',*report['limits'],'','MSE in eV^2, point [95% shot-cluster interval]; lower is better. All models in a cell use identical F rows.','',
           '| Unit / scenario | Model | Rows | MSE [95%] | RMSE |','|---|---|---|---|---|']
    for key,c in cells.items():
        for name,v in c['models'].items():lines.append(f"| {key} | {name} | {c['rows']} | {v['MSE']:.3f} [{v['cluster_MSE95'][0]:.3f}, {v['cluster_MSE95'][1]:.3f}] | {v['RMSE']:.3f} |")
    lines+=['','## Paired skill','', 'Skill=1-MSE(candidate)/MSE(reference); positive means lower candidate MSE.','',
            '| Unit / scenario | Comparison | Point [95% cluster] | Excludes zero |','|---|---|---|---|']
    for key,c in cells.items():
        for name,v in c['pairs'].items():ci=v['cluster']['ci95'];lines.append(f"| {key} | {name} | {v['point']:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}] | {v['cluster']['excludes_zero']} |")
    lines+=['','A negative conclusion would change after a separately specified adequate-budget comparison on unused data; no automatic further architecture search.']
    (C.CACHE/'readout.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('B106 all32 common-row MSE cells produced',flush=True)
if __name__=='__main__':main()
