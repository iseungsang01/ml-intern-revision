"""Independent raw-grid support, source and feature-layout reconstruction."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch

def verify(root,info,path):
    # Does not import data98/model98 or call their selection/filter helpers.
    hp=info['metadata']['hashes'];prepared=next(root/p for p in hp if p.endswith('.pt'))
    for p,h in hp.items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h
    z=torch.load(prepared,map_location='cpu',weights_only=False);stats=z['identity']['stats'];m=z['identity']['manifest']
    assert not (set(m['train'])|set(m['val']))&set(m['test'])
    with np.load(root/'data/.b73/grid.npz',allow_pickle=True) as g:
        tm=g['time'];ti=g['ti'];sg=g['seg'];fi=g['fidx'];fast=g['fast'];shots=g['shot']
    with np.load(root/'data/.b79/beam_cols.npz') as b:power=np.where(np.isfinite(b['P']),b['P'],0).sum(1)
    bad=set(json.loads((root/'data/.b79/checks/validity.json').read_text())['invalid_files'])
    order=np.argsort(shots,kind='stable');clus=np.zeros(len(shots),dtype=int);clus[order]=np.r_[0,np.cumsum(np.diff(shots[order])>3)]
    with np.load(path) as p:
        gi=p['gi'];source=p['source_gi'];rules=p['mapping'][:,2];hold=p['pred'][:,3];target=p['y'];clusters=p['cluster'];pred=p['pred']
    actual={(int(row),int(rule)):(int(src),float(value),float(y),int(c),int(k)) for k,(row,rule,src,value,y,c) in enumerate(zip(gi,rules,source,hold,target,clusters))}
    assert len(actual)==len(gi),'Duplicate query'
    wanted=set();sampled=0;maxerr=0.
    for row in z['val']['D10']:
        fid=row['fidx'];ids=row['gi'];n=len(ids)
        assert fid in m['val'] and fid not in m['test']
        if int(shots[fid]) in bad:continue
        localtime=tm[ids];localti=ti[ids];observed=np.flatnonzero(np.isfinite(localti))
        normalized=((fast[ids].astype(np.float32)-np.asarray(stats['fast_mu']))/np.asarray(stats['fast_sd'])).astype(np.float32)
        assert normalized.shape==(n,13) and np.array_equal(normalized,row['x'][:,:13])
        u=np.column_stack([np.ones(n),normalized[:,9:].mean(1,dtype=np.float64),normalized[:,:9].mean(1,dtype=np.float64),power[ids]])
        if not np.isfinite(u).all():continue
        source_rows={};lastgap=-1
        for i in range(n):
            if i and .015<float(localtime[i]-localtime[i-1])<.045:lastgap=i-1
            if not row['mask'][i,0]:continue
            prior=observed[observed<i];chosen=[]
            for d in [.010,.050,.100,.300,.650,1.000]:
                candidates=[int(j) for j in prior if localtime[j]<=localtime[i]-d+.003]
                chosen.append(max(candidates,default=-1))
            chosen.append(max((int(j) for j in prior if j<=lastgap),default=-1))
            if min(chosen)<0:continue
            source_rows[i]=chosen
            for r,s in enumerate(chosen):
                key=(int(ids[i]),r);wanted.add(key);record=actual[key]
                assert record[:4]==(int(ids[s]),float(localti[s]),float(localti[i]),int(clus[fid]))
                assert sg[ids[s]]==sg[ids[i]] and fi[ids[s]]==fi[ids[i]] and ids[s]<ids[i]
        if not source_rows or sampled>=2:continue
        sampled+=1
        # Direct scalar state evolution on independently reconstructed u.
        for kind,column in [('M',0),('H',1)]:
            model=info['models'][kind];beta=np.asarray(model['beta']);tau=model['tau']
            force=u@beta if kind=='M' else np.full(n,beta[0])
            for i,sources in source_rows.items():
                for r,s in enumerate(sources):
                    value=float(localti[s])
                    for j in range(s+1,i+1):
                        decay=np.exp(-float(localtime[j]-localtime[j-1])/tau)
                        value=decay*value+(1-decay)*force[j-1]
                    expected=pred[actual[(int(ids[i]),r)][4],column]
                    maxerr=max(maxerr,abs(value-expected))
    assert wanted==set(actual) and maxerr<=.001
    return {'all_pass':True,'queries':len(wanted),'raw_support_reconstructed':True,'raw_feature_layout_verified':True,'direct_raw_max_abs_eV':float(maxerr),'direct_segments':sampled}
