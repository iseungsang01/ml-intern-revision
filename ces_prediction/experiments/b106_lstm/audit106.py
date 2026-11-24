"""Independent raw-source/correction/MSE/paired interval audit; no producer helpers."""
import zlib,sys
import numpy as np,joblib
import common106 as C
sys.path.insert(0,str(C.HERE.parent/'b75_availability'));sys.path.insert(0,str(C.HERE.parent/'b78_sparse'))
import stage_fits as SF
import common78
def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def take(path,idx,head=None):
    a=read(path);pos={int(i):j for j,i in enumerate(a['idx'])};j=np.array([pos[int(i)] for i in idx]);v=a['est'][j];return v if head is None else v[:,head]
def main():
    C.guard();C.inputs_guard();G,R=C.M.load_sources();report=C.load(C.CACHE/'readout.json');assert report['sources']==C.sources()
    for rel,h in report['estimate_sha256'].items():assert C.sha(C.ROOT/rel)==h
    for rel,h in report['reference_sha256'].items():assert C.sha(C.ROOT/rel)==h
    assert len(report['fits'])==20
    for case in C.CASES:C.verify_run(case)
    ctx=SF.Ctx()
    for u in C.UNITS:
        m=C.load(C.rd(u+'__outer')/'manifest.json');f8,_,_=ctx.unit_arrays({'fit':m['train']});done=C.load(C.CACHE/'correction'/u/'complete.json')
        for rel,h in done['OOF_sources'].items():
            p=C.ROOT/rel;assert C.sha(p)==h;case=p.name.split('__held__')[0];man=C.load(C.rd(case)/'manifest.json');z=read(p)
            assert not set(man['held'])&set(man['train']+man['val']+man['test']) and set(man['held'])<=set(m['train'])
            assert np.isin(R['fidx'][z['idx']],man['held']).all() and str(z['model_sha256'])==C.load(C.rd(case)/'complete.json')['best_sha256']
        models={name[:-7]:joblib.load(C.CACHE/'correction'/u/name) for name,h in done['models'].items() if C.sha(C.CACHE/'correction'/u/name)==h};assert len(models)==2
        for rule in C.RULES:
            key=u+'/'+rule+'/F';cell=report['cells'][key];sets=read(C.DATA/('.b78' if rule=='K10' else '.b75')/'sets.npz');mask=sets[('S_' if rule=='S' else 'common_')+'F']
            idx=np.flatnonzero(mask&np.isin(R['fidx'],m['test']));assert len(idx)==cell['rows']
            raw=read(C.CACHE/'estimates'/f'{u}__{rule}.npz');j=np.searchsorted(raw['idx'],idx);assert np.array_equal(raw['idx'][j],idx)
            predpath=C.CACHE/'predictions'/f'{u}__outer__test__{rule}.npz';pred=read(predpath)
            assert str(pred['model_sha256'])==C.load(C.rd(u+'__outer')/'complete.json')['best_sha256'] and str(pred['source_hash'])==C.digest(C.sources())
            base=take(predpath,idx);assert np.array_equal(base,raw['LSTM'][j])
            I=common78.load_inputs(rule) if rule=='K10' else SF.K.load_inputs(rule);h=I['H'][idx];gp=I['G'][idx];d1=base-h;d2=base-gp
            X=np.concatenate((f8[idx],R['tstep'][idx,None],I['age'][idx,None],h[:,None],d1[:,None],d2[:,None],np.abs(d1[:,None]),np.abs(d2[:,None])),axis=1)
            ok=np.isfinite(X).all(1)
            for label,mod in models.items():
                expected=base.copy();expected[ok]+=mod.predict(X[ok]);assert np.allclose(expected,raw['LSTM_'+label][j],rtol=1e-12,atol=1e-9)
            values={name:raw[name][j] for name in ('LSTM','LSTM_frozen_D10','LSTM_mixed_post')}
            root=C.DATA/('.b78' if rule=='K10' else '.b75')/'fits'
            for label,learner in [('GP_history_boost','T0'),('GP_fast_boost','TX')]:values[label]=take(root/f'{u}__{learner}__{rule}__F.npz',idx)
            for label in ('TCN22','TCN35'):values[label]=take(C.DATA/'.b105/predictions'/f'{u}__{label}__test__{rule}.npz',idx,0)
            values['GP_fair_mixture']=take(C.DATA/'.b105/trees'/u/f'{rule}__F__test__mixture.npz',idx)
            if rule!='K10':values['GP_fair_specialized']=take(C.DATA/'.b105/trees'/u/f'{rule}__F__test__specialized.npz',idx)
            values['GP_only']=gp;values['hold']=h;names=list(cell['models']);assert set(names)==set(values)
            err=np.stack([values[n]-R['tg_F'][idx] for n in names],axis=1);sq=err**2;means=sq.sum(0)/len(idx)
            for i,n in enumerate(names):assert np.isclose(means[i],cell['models'][n]['MSE'],rtol=1e-12,atol=1e-8)
            for form,field in (('cluster','clus'),('discharge','fidx')):
                groups,inv=np.unique(R[field][idx],return_inverse=True);n=len(groups);sums=np.zeros((n,len(names)));counts=np.zeros(n)
                np.add.at(sums,inv,sq);np.add.at(counts,inv,1);rng=np.random.default_rng([20261106,zlib.crc32((key+'/'+form).encode())]);allm=[]
                for _ in range(40):
                    w=rng.multinomial(n,np.repeat(1/n,n),500);ss=np.einsum('bg,gc->bc',w,sums);nn=np.einsum('bg,g->b',w,counts);allm.append(ss/nn[:,None])
                dm=np.concatenate(allm);ci=np.percentile(dm,[2.5,97.5],axis=0)
                for i,name in enumerate(names):assert np.allclose(ci[:,i],cell['models'][name][form+'_MSE95'],rtol=1e-9,atol=1e-7)
                for pair,p in cell['pairs'].items():
                    a,b=pair.split('_over_');ia,ib=names.index(a),names.index(b);v=1-dm[:,ia]/dm[:,ib];q=np.percentile(v,[2.5,97.5]);assert np.isfinite(v).all()
                    assert np.allclose(q,p[form]['ci95'],rtol=1e-9,atol=1e-10) and bool(q[0]>0 or q[1]<0)==p[form]['excludes_zero']
                    assert np.isclose(1-means[ia]/means[ib],p['point'],rtol=1e-12,atol=1e-12)
    cells=report['cells'];s=report['statements']
    assert s['S1_frozen_LSTM_post_over_GP_all_fresh']==all(cells[u+'/D10/F']['pairs']['LSTM_frozen_D10_over_GP_fast_boost']['cluster']['ci95'][0]>0 for u in C.UNITS)
    assert s['S2_mixed_post_over_GP_all_scenarios']==all(c['pairs']['LSTM_mixed_post_over_GP_fast_boost']['cluster']['ci95'][0]>0 for c in cells.values())
    assert s['S3_GP_fast_increment_all_fresh']==all(cells[u+'/D10/F']['pairs']['GP_fast_boost_over_GP_history_boost']['cluster']['ci95'][0]>0 for u in C.UNITS)
    C.save(C.CACHE/'audit.json',{'all_pass':True,'MSE_cells':32,'draws':20000,'OOF_exclusion_checked':True,'post_predictions_rebuilt':True,'independent_MSE_intervals_pairs':True,'report_sha256':C.sha(C.CACHE/'readout.json'),'sources':C.sources()})
    print('B106 independent raw prediction/correction/MSE audit passed',flush=True)
if __name__=='__main__':main()
