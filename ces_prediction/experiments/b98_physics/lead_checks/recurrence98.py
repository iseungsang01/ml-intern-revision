"""Independent direct scalar recurrence and real-trace mutation checks."""
import copy
import numpy as np

def direct(segment,model):
    t=segment['t'];u=segment['u'];a=segment['anchor'];rows=np.flatnonzero(segment['mask'])
    out=np.empty((len(rows),7));beta=np.asarray(model['beta']);tau=model['tau']
    forcing=u@beta if model['kind']=='M' else np.full(len(t),beta[0])
    for r,row in enumerate(rows):
        for k in range(7):
            start=int(a[row,k]);state=float(segment['ti'][start])
            for j in range(start+1,row+1):
                alpha=np.exp(-(t[j]-t[j-1])/tau)
                state=alpha*state+(1-alpha)*forcing[j-1]
            out[r,k]=state
    return out.ravel()

def check(segments,models,predict):
    selected=segments[:2];errors={};guards=[]
    for kind in ('M','H'):
        model=models[kind];delta=[]
        for s in selected:
            actual,_,_=predict([s],model);expected=direct(s,model)
            err=float(np.max(np.abs(actual-expected)));assert err<=.001,err;delta.append(err)
            indices=np.flatnonzero(s['mask']);cut=int(indices[len(indices)//2])
            prefix={k:(v[:cut+1].copy() if isinstance(v,np.ndarray) else v) for k,v in s.items()}
            pp,_,_=predict([prefix],model)
            assert np.allclose(pp,actual[:len(pp)],rtol=0,atol=.001)
            future=copy.deepcopy(s);future['u'][cut:,1:]+=17
            fp,_,_=predict([future],model)
            assert np.allclose(fp[:len(pp)],pp,rtol=0,atol=.001)
            hidden=copy.deepcopy(s);hidden['ti'][cut:]+=500
            hp,_,_=predict([hidden],model)
            assert np.allclose(hp[:len(pp)],pp,rtol=0,atol=.001)
            labels=copy.deepcopy(s);labels['target'][labels['mask']]+=500
            lp,_,_=predict([labels],model)
            assert np.array_equal(lp,actual)
            guards.append({'kind':kind,'prefix':True,'current_and_future_forcing_excluded':True,'current_and_future_CES_excluded':True,'label_isolation':True})
        errors[kind]=max(delta)
    # Static model is direct current/anchor arithmetic, not a relaxation ODE.
    for s in selected:
        expected=[]
        for row in np.flatnonzero(s['mask']):
            for src in s['anchor'][row]:expected.append(s['ti'][src]+np.dot(models['L']['beta'],s['u'][row,1:]-s['u'][src,1:]))
        actual,_,_=predict([s],models['L']);assert np.allclose(actual,expected,rtol=0,atol=.001)
        indices=np.flatnonzero(s['mask']);cut=int(indices[len(indices)//2])
        future=copy.deepcopy(s);future['u'][cut+1:,1:]+=17;future['ti'][cut:]+=500
        changed,_,_=predict([future],models['L']);prefix_rows=int(np.sum(indices<=cut))*7
        assert np.allclose(changed[:prefix_rows],actual[:prefix_rows],rtol=0,atol=.001)
        labels=copy.deepcopy(s);labels['target'][labels['mask']]+=500
        isolated,_,_=predict([labels],models['L']);assert np.array_equal(isolated,actual)
        guards.append({'kind':'L','future_fast_and_current_future_CES_excluded':True,'label_isolation':True})
    return {'all_pass':True,'direct_recurrence_max_abs_eV':errors,'real_segments':len(selected),'guards':guards}
