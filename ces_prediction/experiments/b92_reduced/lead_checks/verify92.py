"""Parent independent real-input models, vector loss, batching and validator."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import sys,json,copy
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import common92 as M


def main():
    pins=M.guard();sources=M.sources()
    import torch
    import data88 as D,data85 as D85,train85 as T85
    from model92 import create_model
    from prepare92 import batches
    from train92 import vector_loss,validate
    torch.set_num_threads(1)
    runtime={'device':'cpu','dense':True,'cudnn':False,'threads':1,'mkldnn':False}
    G,R=M.load_sources();u=M.select_units()[0];manifest=D.unit_manifest(u);stats=D.fit_stats(manifest['train'],G,R)
    segs=D.build_segments(manifest['train'][:12],stats,'D10',G,R)
    real=[s for s in segs if len(s['x'])>=160 and s['mask'][:64].any()][:2];assert len(real)==2
    small=[{k:(v[:n] if k in ('x','y','mask','gi') else v) for k,v in s.items()} for s,n in zip(real,(48,64))]
    batch=batches(small,'cpu',2,shuffle_seed=92)[0]
    old=T85.make_batches([{**s,'x':s['x'][:,:22]} for s in small],2,shuffle_seed=92)[0]
    assert torch.equal(old['x'],batch['x'][...,:22]) and torch.equal(old['y'],batch['y']) and torch.equal(old['mask'],batch['mask'])
    a=create_model('A',runtime,1701).double().eval();c=create_model('C',runtime,1701).double().eval()
    assert a.param_count()==c.param_count() and all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
    x=batch['x'].double();y=batch['y'].double();mask=batch['mask'];lengths=batch['lengths']
    changed=x.clone();changed[...,22:]+=1
    with torch.no_grad():
        assert torch.equal(c(x,lengths),c(changed,lengths))
        assert (x[...,18]>0).any() and not torch.equal(a(x,lengths),a(changed,lengths))
    for code in ('A','C','TX','TH'):
        dense=create_model(code,runtime,1701).double().eval()
        packed=create_model(code,{**runtime,'dense':False},1701).double().eval();packed.load_state_dict(dense.state_dict())
        pd=dense(x,lengths);pp=packed(x,lengths);valid=torch.arange(x.shape[1])[None,:]<lengths[:,None]
        assert torch.allclose(pd[valid],pp[valid],atol=1e-10,rtol=1e-10),code
        mu=torch.tensor(stats['ti_mu'],dtype=torch.float64);sd=torch.tensor(stats['ti_sd'],dtype=torch.float64)
        v=vector_loss(pd,y,mask,mu,sd);reference=T85.compute_loss(pd,y,mask,mu,sd,'cpu')
        gv=torch.autograd.grad(v,pd,retain_graph=True)[0];gr=torch.autograd.grad(reference,pd,retain_graph=True)[0]
        assert torch.allclose(v,reference,atol=1e-10,rtol=1e-10) and torch.allclose(gv,gr,atol=1e-10,rtol=1e-10)
        # Independent NumPy head means with actual reported labels only.
        p=pd.detach().numpy();yy=y.numpy();mm=mask.numpy();parts=[]
        for h in range(2):
            keep=mm[:,:,h]
            if keep.any():parts.append(np.mean((p[:,:,h][keep]-yy[:,:,h][keep])**2)+.1*np.maximum(-stats['ti_mu']/stats['ti_sd']-p[:,:,h][keep],0).mean())
        assert np.isclose(v.item(),np.mean(parts),atol=1e-10,rtol=1e-10)
        if code=='TH':
            mut=x.clone();mut[...,:15]+=1000
            assert torch.equal(dense(x,lengths),dense(mut,lengths))
        state=None;stream=[]
        with torch.no_grad():
            for row in x[0]:
                out,state=dense.forward_step(row[None,None],state);stream.append(out)
            assert torch.allclose(pd[:1,:int(lengths[0])],torch.stack(stream,1)[:,:int(lengths[0])],atol=1e-10,rtol=1e-9)
    val={}
    for rule in M.TRAIN_RULES:
        original=D.build_segments(manifest['val'][:1],stats,rule,G,R)
        original=[s for s in original if s['mask'][:96].any()][:2];assert original
        val[rule]=[{k:(v[:96] if k in ('x','y','mask','gi') else v) for k,v in s.items()} for s in original]
    model=create_model('A',runtime,1701).eval();got,_=validate(model,{r:batches(ss,'cpu',1) for r,ss in val.items()})
    expected=[]
    with torch.no_grad():
        for rule,ss in val.items():
            sq=np.zeros(2);ct=np.zeros(2)
            for s in ss:
                p=model(torch.from_numpy(s['x']).unsqueeze(0)).numpy()[0];e=p.astype(np.float64)-s['y']
                sq+=np.where(s['mask'],e*e,0).sum(0);ct+=s['mask'].sum(0)
            present=ct>0;expected.append(float(np.mean(sq[present]/ct[present])))
    assert np.isclose(got,np.mean(expected),atol=1e-6,rtol=1e-5)
    # New cache/selection guards must stop premature scoring; no TEST data needed.
    import predict92
    try:predict92.guarded_ready()
    except (FileNotFoundError,AssertionError):pass
    else:raise AssertionError('Premature TEST entry point was not blocked')
    assert pins==M.guard() and sources==M.sources()
    report={'passes':True,'sources':sources,'frozen':pins,'real_train_val_only':True,'checks':['original22/batching','paired A/C','A positive and C anchor invariance','dense packed valid-prefix','original and NumPy loss/gradient','T-H fast invariance','stream/reset','NumPy seven-rule selector','premature TEST guard']}
    M.save(M.CACHE/'lead_checks.json',report);M.C.save_text(M.PAPER/'b92_lead_checks.md','# B.92 independent model/loss/objective audit\n\n'+json.dumps(report,indent=2)+'\n')
    print('PASS: B92 independent real-input loss/gradient and selector')


if __name__=='__main__':main()
