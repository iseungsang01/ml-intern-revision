"""Independent raw-prediction/paired-bootstrap verification; never calls producer statistics."""
import zlib
import numpy as np
import common104 as C

def run():
    import common92 as M
    from pipeline104 import all_ready
    all_ready();report=C.load(C.CACHE/'readout.json');checks=C.load(C.CACHE/'prediction_checks.json')
    assert checks['all_pass'] and checks['labels_not_inputs'] and checks['sources']==C.sources()
    assert report['sources']==C.sources() and report['draws']==20000 and report['seed']==20261104 and report['chunk']==500
    assert report['provenance_sha256']==C.sha(C.CACHE/'provenance.json')
    for n,h in C.prov()['raw'].items():assert C.sha(C.ROOT/'data/.b73'/n)==h
    _,R=M.load_sources()
    with np.load(C.ROOT/'data/.b75/sets.npz',allow_pickle=False) as z:masks={k:z[k] for k in ('common_F','common_L5')}
    assert C.sha(C.ROOT/'data/.b75/sets.npz')==C.prov()['sets75_sha256']
    data={};references=[];hashes={}
    for rule in ('D10','D50'):
        values=[];reference=None
        for seed in C.SEEDS:
            rr=[]
            for budget,code in [('SHORT','TX'),('SHORT','TH'),('LONG','TX'),('LONG','TH')]:
                p=C.CACHE/'predictions'/f'{seed}__{C.LABEL[code]}__{budget}__{rule}.npz';hashes[p.name]=C.sha(p)
                cfg=C.load(C.run_dir('confirm',seed,code)/'config.json');manifest=C.load(C.run_dir('confirm',seed,code)/'manifest.json')
                with np.load(p,allow_pickle=False) as z:idx=z['idx'].copy();est=z['est'].astype(np.float64);config=str(z['config_hash']);source=str(z['source_hash'])
                assert config==cfg['config_hash'] and source==C.digest(C.sources())
                assert idx.ndim==1 and len(idx) and idx.min()>=0 and idx.max()<len(R['gi']) and np.all(np.diff(idx)>0)
                assert est.shape==(len(idx),2) and np.isfinite(est).all() and set(R['fidx'][idx]).issubset(set(manifest['test']))
                if reference is None:reference=idx
                else:assert np.array_equal(reference,idx)
                rr.append(est)
            values.append(rr)
        references.append(reference)
        values=np.asarray(values)
        for head,target in enumerate(('F','L5')):
            ii=reference[masks['common_'+target][reference]&np.isfinite(R['tg_'+target][reference])]
            positions=np.searchsorted(reference,ii);e=values[:,:,positions,head]-R['tg_'+target][ii]
            key=rule+'/'+target;cell=report['cells'][key]
            assert (cell['rows'],cell['clusters'],cell['discharges'])==(len(ii),len(np.unique(R['clus'][ii])),len(np.unique(R['fidx'][ii])))
            data[key]={'loss':e*e,'cluster':R['clus'][ii],'discharge':R['fidx'][ii]}
    assert hashes==report['prediction_hashes'] and len(hashes)==24
    names=('short_fast','long_fast','fast_budget','history_budget','delta')
    def quantities(t):
        with np.errstate(divide='ignore',invalid='ignore'):
            a=1-t[...,0]/t[...,1];b=1-t[...,2]/t[...,3]
            return np.stack((a,b,1-t[...,2]/t[...,0],1-t[...,3]/t[...,1],b-a),axis=-1)
    for key,d in data.items():
        v=quantities(d['loss'].sum(axis=2))
        for j,name in enumerate(names):
            got=report['cells'][key]['statistics'][name]
            assert np.allclose(got['per_init'],v[:,j],rtol=1e-10,atol=1e-10) and np.isclose(got['point'],v[:,j].mean(),rtol=1e-10,atol=1e-10)
    for form,column in [('cluster','clus'),('discharge','fidx')]:
        support=np.unique(np.concatenate(references));labels=np.unique(R[column][support]);n=len(labels);totals={};fixed={};mixture={}
        for key,d in data.items():
            inv=np.searchsorted(labels,d[form]);a=np.zeros((n,3,4))
            for i in range(3):
                for v in range(4):np.add.at(a[:,i,v],inv,d['loss'][i,v])
            totals[key]=a;fixed[key]=np.empty((20000,5));mixture[key]=np.empty((20000,5))
        rng=np.random.default_rng([20261104,zlib.crc32(form.encode())]);pick=np.random.default_rng([20261104,42,zlib.crc32(form.encode())]).integers(0,3,20000)
        for start in range(0,20000,500):
            end=min(start+500,20000);w=rng.multinomial(n,np.ones(n)/n,size=end-start)
            for key in data:
                sums=np.tensordot(w,totals[key],axes=([1],[0]));v=quantities(sums)
                fixed[key][start:end]=v.mean(axis=1);mixture[key][start:end]=v[np.arange(end-start),pick[start:end]]
        for key in data:
            for j,name in enumerate(names):
                record=report['cells'][key]['statistics'][name][form]
                for kind,values in [('fixed',fixed[key][:,j]),('mixture',mixture[key][:,j])]:
                    finite=values[np.isfinite(values)];assert len(finite)
                    ci=np.quantile(finite,[.025,.975]);assert np.allclose(record[kind],ci,rtol=1e-10,atol=1e-10)
                    assert record[kind+'_undefined']==20000-len(finite)
                    assert record[kind+'_excludes_zero']==bool(ci[0]>0 or ci[1]<0)
    C.save(C.CACHE/'audit.json',{'all_pass':True,'report_sha256':C.sha(C.CACHE/'readout.json'),'prediction_hashes':hashes,
        'sources':C.sources(),'independent_cells':4,'independent_quantities_per_cell':5,'draws':20000,'time':__import__('time').time()})
    print('B104 independent audit passed',flush=True)

if __name__=='__main__':run()
