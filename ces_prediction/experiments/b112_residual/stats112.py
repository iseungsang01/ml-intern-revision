"""Fixed paired B112 MSE/skill readout, without model selection."""
import zlib
import numpy as np
from common112 import MODELS,PAIRS
def compute(arr,key):
    names=[n for n in MODELS if n in arr]
    err=np.column_stack([arr[n] for n in names]).astype(np.float64)-arr['truth'][:,None]
    sq=err**2;means=sq.mean(0)
    models={n:{'MSE':float(means[j]),'RMSE':float(np.sqrt(means[j]))} for j,n in enumerate(names)}
    pairs={a+'_over_'+b:{'point':float(1-means[names.index(a)]/means[names.index(b)])} for a,b in PAIRS}
    for form in ('cluster','discharge'):
        _,inv=np.unique(arr[form],return_inverse=True);ng=int(inv.max()+1);assert ng>=2
        count=np.bincount(inv);sums=np.column_stack([np.bincount(inv,weights=sq[:,j],minlength=ng) for j in range(len(names))])
        rng=np.random.default_rng([20261125,zlib.crc32((key+'/'+form).encode())]);draws=[]
        for _ in range(40):
            w=rng.multinomial(ng,np.full(ng,1/ng),500);draws.append((w@sums)/(w@count)[:,None])
        mm=np.concatenate(draws);assert np.isfinite(mm).all() and (mm>0).all();ci=np.quantile(mm,[.025,.975],axis=0)
        for j,n in enumerate(names):models[n][form+'_MSE95']=ci[:,j].tolist();models[n][form+'_RMSE95']=np.sqrt(ci[:,j]).tolist()
        for a,b in PAIRS:
            v=1-mm[:,names.index(a)]/mm[:,names.index(b)];q=np.quantile(v,[.025,.975])
            pairs[a+'_over_'+b][form]={'ci95':q.tolist(),'excludes_zero':bool(q[0]>0 or q[1]<0)}
    return {'rows':len(arr['idx']),'clusters':int(len(np.unique(arr['cluster']))),'discharges':int(len(np.unique(arr['discharge']))),'models':models,'pairs':pairs}
