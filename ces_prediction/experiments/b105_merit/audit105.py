"""Independent raw-prediction/gate/bootstrap audit; never calls producer statistics."""
import zlib
import numpy as np
import common105 as C
import common92 as M

def main():
    C.guard();C.inputs_guard();G,R=M.load_sources();report=C.load(C.CACHE/'readout.json');specs=C.load(C.CACHE/'pair_specs.json')
    assert C.sha(C.CACHE/'pair_specs.json')==report['pair_specs_sha256'] and report['sources']==C.sources()
    for rel,h in report['prediction_sources'].items():assert C.sha(C.ROOT/rel)==h
    gates={u:{'candidate':[],'reference':[]} for u in C.UNITS}
    scales={u:C.prepared(u)['identity']['stats']['ti_sd'] for u in C.UNITS}
    manifests={u:C.load(C.rd(u)/'manifest.json') for u in C.UNITS}
    masks={}
    for group,path in (('regular',C.DATA/'.b75/sets.npz'),('K10',C.DATA/'.b78/sets.npz')):
        with np.load(path,allow_pickle=False) as a:masks[group]={k:a[k] for k in a.files if k.startswith(('common_','S_'))}
    def load(path):
        with np.load(C.ROOT/path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    def take(path,idx,col):
        a=load(path)
        if isinstance(col,str):return a[col][idx]
        positions={int(i):j for j,i in enumerate(a['idx'])};jj=np.array([positions[int(i)] for i in idx]);v=a['est'][jj]
        return v if col is None else v[:,col]
    for spec in specs:
        for k in ('candidate','reference','idx_source'):
            assert spec[k] in report['prediction_sources']
            assert (C.ROOT/spec[k]).resolve().is_relative_to(C.ROOT.resolve())
        cell=report['cells'][spec['name']];idx=load(spec['idx_source'])['idx'];y=R['tg_'+spec['target']][idx].astype(np.float64)
        assert idx.ndim==1 and len(idx)>0 and (np.diff(idx)>0).all()
        split='val' if spec['split']=='val' else 'test';m=manifests[spec['unit']]
        mask=masks['K10' if spec['rule']=='K10' else 'regular'][('S_' if spec['rule']=='S' else 'common_')+spec['target']]
        expected=np.flatnonzero(np.isin(R['fidx'],m[split])&mask)
        assert np.array_equal(idx,expected),'Common support or manifest mismatch'
        pred=load(spec['candidate']);assert (np.diff(pred['idx'])>0).all()
        if spec['split']!='existing_test':
            assert str(pred['model_sha256'])==C.load(C.rd(spec['unit'])/'complete.json')['best_sha256']
            assert str(pred['sources_sha256'])==C.digest(C.sources())
        else:
            cfgpath=C.DATA/'.b94/confirm'/f"{spec['unit']}__1701__TX/config.json";cfg=C.load(cfgpath)
            assert C.sha(cfgpath)==C.prov()['prepared'][spec['unit']]['original_config_sha256']
            assert str(pred['config_hash'])==cfg['config_hash']
        candidate=take(spec['candidate'],idx,spec['candidate_head']).astype(np.float64)
        ref=take(spec['reference'],idx,spec['reference_head']).astype(np.float64)
        square=np.stack([(candidate-y)**2,(ref-y)**2],axis=1);point=1-square[:,0].sum()/square[:,1].sum();target=cell['statistics']
        assert np.isclose(point,target['point'],rtol=1e-12,atol=1e-12) and cell['rows']==len(idx)
        assert np.allclose(square.mean(0),target['MSE'],rtol=1e-12,atol=1e-8)
        for form,field in (('cluster','clus'),('discharge','fidx')):
            labels=R[field][idx];groups,inv=np.unique(labels,return_inverse=True);n=len(groups)
            sums=np.zeros((n,2));np.add.at(sums,inv,square);cnt=np.zeros(n);np.add.at(cnt,inv,1)
            rng=np.random.default_rng([20261105,zlib.crc32((spec['name']+'/'+form).encode())]);values=[];mse=[]
            for start in range(0,20000,500):
                weight=rng.multinomial(n,np.repeat(1/n,n),min(500,20000-start))
                numerator=np.einsum('bg,gc->bc',weight,sums);denominator=np.einsum('bg,g->b',weight,cnt)
                values.extend((1-numerator[:,0]/numerator[:,1]).tolist());mse.extend((numerator/denominator[:,None]).tolist())
            ci=np.percentile(values,[2.5,97.5]);v=target[form]
            assert np.allclose(ci,v['ci95'],rtol=1e-9,atol=1e-10)
            assert np.allclose(np.percentile(mse,[2.5,97.5],axis=0).T,v['MSE_ci95'],rtol=1e-9,atol=1e-7)
            assert bool(ci[0]>0 or ci[1]<0)==v['excludes_zero'] and v['undefined']==0 and np.isfinite(values).all()
        if spec['split']=='val' and spec['name'].endswith('over_fair_specialized'):
            sd=scales[spec['unit']];gates[spec['unit']]['candidate'].append(float(square[:,0].mean()/sd**2));gates[spec['unit']]['reference'].append(float(square[:,1].mean()/sd**2))
    for u,values in gates.items():
        assert len(values['candidate'])==len(values['reference'])==14
        ratio=np.mean(values['candidate'])/np.mean(values['reference']);g=report['gate']['units'][u]
        assert np.isclose(ratio,g['candidate_to_specialized_ratio'],rtol=1e-12,atol=1e-12) and bool(ratio<=1.05)==g['pass']
    assert report['candidate_TEST_evaluated']==all(v['pass'] for v in report['gate']['units'].values())
    lows=[v['statistics']['cluster']['ci95'][0] for k,v in report['cells'].items() if '/test/' in k and k.endswith('TCN35_over_incumbent_outerfit')]
    assert len(lows)==(64 if report['candidate_TEST_evaluated'] else 0)
    assert report['statements']['S1']['holds']==report['candidate_TEST_evaluated']
    for label,threshold in (('S2',0),('S3',-.05)):
        assert report['statements'][label]['holds']==(bool(min(lows)>threshold) if lows else None)
    if not report['candidate_TEST_evaluated']:
        assert not list((C.CACHE/'predictions').glob('*__test__*.npz')),'Candidate TEST forbidden after failed gate'
    C.save(C.CACHE/'audit.json',{'all_pass':True,'pairs':len(specs),'draws':20000,'independent_points_intervals_gate':True,'sources':C.sources(),'report_sha256':C.sha(C.CACHE/'readout.json')})
    print('B105 independent raw-prediction/interval/gate audit passed',flush=True)
if __name__=='__main__':main()
