"""Parent's independent real-segment model equations, causal and loss audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import common90 as M
import data85 as D
from adapter90 import activate


def main():
    M.assert_registered();M.assert_frozen();sources=M.source_inventory()
    import torch
    from torch.nn import functional as F
    from models90 import create_model
    torch.set_num_threads(1)
    g,r=M.load_sources();unit=M.select_units()[0];manifest=D.unit_manifest(unit);stats=D.fit_stats(manifest['train'],g,r)
    real=D.build_segments(manifest['test'],stats,'D10',g,r)
    s=next(s for s in real if len(s['x'])>=160 and s['mask'][:64].all(1).any())
    x=torch.from_numpy(s['x'][:64]).unsqueeze(0);y=torch.from_numpy(s['y'][:64]).unsqueeze(0);mask=torch.from_numpy(s['mask'][:64]).unsqueeze(0)
    results={}
    original_cache=M.B85.CACHE
    for method in M.METHODS:
        t=activate(method);mx=t.create_model('X',1701).eval();mh=t.create_model('H',1701).eval()
        assert t.CACHE==M.CACHE and M.B85.CACHE==original_cache==M.DATA/'.b85'
        assert all(torch.equal(v,mh.state_dict()[k]) for k,v in mx.state_dict().items())
        with torch.no_grad():
            out=mx(x);state=None;step=[]
            for row in x[0]:
                p,state=mx.forward_step(row[None],state);step.append(p)
            assert torch.allclose(out,torch.stack(step,1),atol=1e-5,rtol=0),method
            changed=x.clone();changed[...,:15]+=1
            assert torch.equal(mh(x),mh(changed)),method
            assert not torch.equal(mx(x),mx(changed)),method
            future=x.clone();future[:,32:,:13]+=1000
            assert torch.allclose(mx(x)[:,:32],mx(future)[:,:32],atol=1e-5,rtol=0),method
            # Independently expand the one-step GRU/observer equations.
            row=x[:,10];dt=torch.expm1(row[:,15:16])
            first,_=mx.forward_step(row,None)
            if method=='G':
                h=row
                for cell in mx.cells:
                    gates=F.linear(h,cell.weight_ih,cell.bias_ih)
                    hidden=F.linear(torch.zeros((1,160)),cell.weight_hh,cell.bias_hh)
                    ir,iz,inn=gates.chunk(3,1);hr,hz,hn=hidden.chunk(3,1)
                    z=torch.sigmoid(iz+hz);new=torch.tanh(inn+torch.sigmoid(ir+hr)*hn)
                    h=(1-z)*new
                assert torch.allclose(first,mx.heads(h),atol=1e-6,rtol=1e-6)
                # Nonzero real-prefix state verifies the decay, not just reset.
                state=None
                for rr in x[0,:10]:_,state=mx.forward_step(rr[None],state)
                h=row
                for cell,rate,previous in zip(mx.cells,mx.rates,state):
                    decayed=previous*torch.exp(-F.softplus(rate(row[:,[15,17,18]]))*dt)
                    ir,iz,inn=F.linear(h,cell.weight_ih,cell.bias_ih).chunk(3,1)
                    hr,hz,hn=F.linear(decayed,cell.weight_hh,cell.bias_hh).chunk(3,1)
                    z=torch.sigmoid(iz+hz);new=torch.tanh(inn+torch.sigmoid(ir+hr)*hn)
                    h=(1-z)*new+z*decayed
                got,_=mx.forward_step(row,state)
                assert torch.allclose(got,mx.heads(h),atol=1e-6,rtol=1e-6)
            if method in ('R','C'):
                q=torch.log1p(torch.exp(mx.process))+1e-4
                a=torch.tanh(mx.transition) if method=='R' else torch.exp(-(torch.log1p(torch.exp(mx.transition))+1e-4)*dt)
                pv=a*a+(q if method=='R' else q*(1-a*a))
                prior=(1-(a.abs() if method=='R' else a))*torch.tanh(mx.drive(row))
                noise=torch.log1p(torch.exp(mx.obs_noise(row[:,15:22])))+1e-4
                mean=(noise*prior+pv*torch.tanh(mx.observation(row)))/(pv+noise)
                assert torch.allclose(first,mx.heads(mean),atol=1e-6,rtol=1e-6)
            if method=='T':
                long=torch.from_numpy(s['x'][:160]).unsqueeze(0);remote=long.clone();remote[:,:33,:13]+=1000
                assert torch.allclose(mx(long)[:,-1],mx(remote)[:,-1],atol=1e-5,rtol=0)
        # Loss uses actual labels; scalar NumPy equations do not call trainer helpers.
        p=mx(x);loss=t.compute_loss(p,y,mask,torch.tensor(500.),torch.tensor(100.),'cpu')
        raw=p.detach().numpy();target=y.numpy();valid=mask.numpy();parts=[];penalties=[]
        lv=mx.latest_logvar.detach().numpy() if method=='N' else None
        for h in range(2):
            k=valid[:,:,h]
            if not k.any():continue
            e=raw[:,:,h][k]-target[:,:,h][k]
            if method=='U':v=np.where(np.abs(e)<=1,.5*e**2,np.abs(e)-.5)
            elif method=='V':v=2.5*np.log1p(e**2/4)
            elif method=='N':v=.5*(lv[:,:,h][k]+e**2/np.exp(lv[:,:,h][k]))
            else:v=e**2
            parts.append(float(v.mean()));penalties.append(float(np.maximum(-5-raw[:,:,h][k],0).mean()))
        assert np.isclose(loss.item(),np.mean(parts)+.1*np.mean(penalties),rtol=2e-6,atol=1e-6),method
        mx.zero_grad(set_to_none=True);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mx.parameters())
        if method=='N':assert all(p.grad is not None and torch.isfinite(p.grad).all() for name,p in mx.named_parameters() if name.startswith('logvar_'))
        results[method]={'parameters':mx.param_count(),'paired':True,'causal_streaming':True,'independent_equations':method in 'GRC','numpy_masked_loss':True}
    assert M.source_inventory()==sources
    report={'passes':True,'execution_sources':sources,'scope':'real held-out inputs and labels; untrained equations and gradients only, no performance inference','TCN_remote_past_rows':160,'methods':results}
    M.save(M.CACHE/'lead_checks.json',report);M.C.save_text(M.PAPER/'b90_lead_checks.md','# B.90 independent model and loss audit\n\n'+json.dumps(report,indent=2)+'\n')
    print('PASS: B90 independent real-model equations and NumPy losses')


if __name__=='__main__':main()
