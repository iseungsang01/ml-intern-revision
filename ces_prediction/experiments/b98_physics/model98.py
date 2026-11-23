"""Five-parameter forced first-order report predictor, not a Ti transport law."""
import numpy as np
from scipy.optimize import minimize_scalar

def filter_basis(t,u,tau):
    """Exact left-ZOH basis; bounded-exponent blocks handle irregular intervals."""
    assert tau>0 and np.all(np.diff(t)>0) and np.isfinite(u).all()
    out=np.zeros_like(u,dtype=np.float64);a=1
    while a<len(t):
        b=max(a+1,min(len(t),int(np.searchsorted(t,t[a-1]+20*tau,side='right'))))
        rho=np.exp(-(t[a:b]-t[a-1])/tau)
        increment=-np.expm1(-np.diff(t[a-1:b])/tau)
        out[a:b]=rho[:,None]*(out[a-1]+np.cumsum((increment/rho)[:,None]*u[a-1:b-1],axis=0))
        a=b
    return out

def matrix(segments,kind,tau=None):
    xx=[];base=[];yy=[];meta=[]
    for k,s in enumerate(segments):
        rows=np.repeat(np.flatnonzero(s['mask']),7);src=s['anchor'][s['mask']].ravel()
        if not len(rows):continue
        assert np.all((0<=src)&(src<rows))
        history=s['ti'][src]
        if kind=='L':
            x=s['u'][rows,1:]-s['u'][src,1:];b=history
        else:
            rho=np.exp(-(s['t'][rows]-s['t'][src])/tau);b=rho*history
            if kind=='H':x=(1-rho)[:,None]
            else:
                z=filter_basis(s['t'],s['u'],tau)
                x=z[rows]-rho[:,None]*z[src]
                x[:,0]=1-rho
        xx.append(x);base.append(b);yy.append(s['target'][rows])
        meta.append(np.column_stack([np.full(len(rows),k),rows,np.tile(np.arange(7),int(len(rows)/7))]))
    assert xx,'No common labeled source rows'
    return np.vstack(xx),np.concatenate(base),np.concatenate(yy),np.vstack(meta)

def fit(segments,kind):
    profile=[]
    def objective(logtau):
        tau=float(np.exp(logtau)) if kind!='L' else None
        x,b,y,_=matrix(segments,kind,tau)
        beta,_,rank,sv=np.linalg.lstsq(x,y-b,rcond=None)
        loss=float(np.mean((b+x@beta-y)**2))
        assert np.isfinite(loss) and np.isfinite(beta).all()
        profile.append({'tau':tau,'beta':beta.tolist(),'train_mse':loss,'rank':int(rank),
                        'condition':float(sv[0]/sv[-1]) if sv[-1]>0 else None})
        return loss
    solver=None
    if kind=='L':objective(0)
    else:
        grid=np.linspace(np.log(.005),np.log(2.),9)
        losses=[objective(a) for a in grid];j=int(np.argmin(losses))
        result=minimize_scalar(objective,bounds=(grid[max(0,j-1)],grid[min(8,j+1)]),method='bounded',options={'maxiter':12,'xatol':.001})
        solver={'success':bool(result.success),'message':str(result.message),'nfev':int(result.nfev)}
    best=min(profile,key=lambda r:r['train_mse'])
    return {'kind':kind,**best,'profile':profile,'solver':solver,'nominal_parameters':{'M':5,'H':2,'L':3}[kind]}

def predict(segments,model):
    x,b,y,meta=matrix(segments,model['kind'],model['tau'])
    return b+x@np.asarray(model['beta']),y,meta
