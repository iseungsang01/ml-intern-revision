"""Real-segment model contract checks; caller must verify committed source pins.

This module loads no observations and starts no training. Mutation checks operate
on copies of actual supplied inputs. Zero arrays are used solely as batch padding.
"""
import argparse,io,json
import numpy as np
import torch
from models90 import METHODS,ARCHITECTURES,create_model

def _predict(model,x,lengths=None):
    model.eval()
    with torch.no_grad():return model(x,lengths)

def audit_models(real_segments,methods=METHODS,device='cpu',*,guard):
    if not callable(guard):raise RuntimeError('Registered/frozen-source guard is required')
    guard() # Must precede reading supplied real observations or model construction.
    assert len(real_segments)>=2,'Two real supervised segment prefixes required'
    segs=real_segments[:2]
    for s in segs:
        assert all(k in s for k in ('x','y','mask','gi','fidx','segment'))
        assert s['x'].ndim==2 and s['x'].shape[1]==22 and len(s['x'])
        assert len(s['gi'])==len(s['x']) and np.all(np.diff(s['gi'])>0)
        assert np.isfinite(s['x']).all() and s['mask'].dtype==bool
    lengths=torch.tensor([len(s['x']) for s in segs],dtype=torch.long)
    padded=np.zeros((2,int(lengths.max()),22),dtype=np.float32)
    for i,s in enumerate(segs):padded[i,:len(s['x'])]=s['x']
    x=torch.from_numpy(padded).to(device);result={}
    for method in methods:
        mx=create_model(method,'X',1701).to(device);mh=create_model(method,'H',1701).to(device)
        assert mx.param_count()==mh.param_count()
        assert all(torch.equal(v,mh.state_dict()[k]) for k,v in mx.state_dict().items())
        method_result={'parameters':mx.param_count(),'paired_initial_state':True,'variants':{}}
        for var,model in (('X',mx),('H',mh)):
            full=_predict(model,x,lengths);assert full.shape==(*x.shape[:2],2) and torch.isfinite(full).all()
            for i,s in enumerate(segs):
                n=len(s['x']);one=x[i:i+1,:n];separate=_predict(model,one,torch.tensor([n]))
                assert torch.allclose(full[i:i+1,:n],separate,atol=1e-5,rtol=0),'Padded batch/reset mismatch'
            n=int(lengths[0]);one=x[:1,:n];state=None;stream=[];stream_var=[]
            with torch.no_grad():
                for row in one[0]:
                    pred,state=model.forward_step(row.reshape(1,1,22),state);assert pred.shape==(1,2)
                    stream.append(pred)
                    if method=='N':
                        assert model.latest_logvar.shape==pred.shape;stream_var.append(model.latest_logvar.clone())
            step=torch.stack(stream,1)
            assert torch.allclose(full[:1,:n],step,atol=1e-5,rtol=0),'Streaming mismatch'
            if method=='N':
                _predict(model,one);assert torch.allclose(model.latest_logvar,torch.stack(stream_var,1),atol=1e-5,rtol=0)
            with torch.no_grad():first,_=model.forward_step(one[:,0],None)
            assert torch.allclose(first,full[:1,0],atol=1e-5,rtol=0)
            changed=one.clone();changed[...,:15]+=1
            altered=_predict(model,changed)
            if var=='H':assert torch.equal(_predict(model,one),altered),'H fast invariance failed'
            else:assert not torch.allclose(_predict(model,one),altered,atol=1e-7,rtol=0),'X fast positive control failed'
            cutoff=max(1,n//2);future=one.clone();future[:,cutoff:,:13]+=1000
            assert torch.allclose(_predict(model,one)[:,:cutoff],_predict(model,future)[:,:cutoff],atol=1e-5,rtol=0),'Future input leakage'
            rf_checked=False
            if method=='T' and n>model.receptive_field_rows:
                remote=one.clone();remote[:,:n-model.receptive_field_rows,:13]+=1000
                assert torch.allclose(_predict(model,one)[:,-1],_predict(model,remote)[:,-1],atol=1e-5,rtol=0),'TCN receptive field mismatch'
                rf_checked=True
            buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
            restored=create_model(method,var,0).to(device);restored.load_state_dict(torch.load(buffer,map_location=device,weights_only=True))
            assert torch.allclose(_predict(model,one),_predict(restored,one),atol=1e-6,rtol=0),'Save/load mismatch'
            # A backward diagnostic with actual finite targets and original masks.
            target=torch.from_numpy(np.asarray(segs[0]['y'],dtype=np.float32)).unsqueeze(0).to(device)
            mask=torch.from_numpy(np.asarray(segs[0]['mask'],dtype=bool)).unsqueeze(0).to(device)
            assert mask.any(),'A real supervised prefix is required for backward checks'
            model.train();model.zero_grad(set_to_none=True);pred=model(one)
            assert torch.isfinite(target[mask]).all()
            if method=='N':
                lv=model.latest_logvar;assert lv.shape==pred.shape and lv.requires_grad
                assert ((lv>=-8)&(lv<=8)).all();lv.retain_grad()
                loss=.5*(lv[mask]+(pred[mask]-target[mask]).square()*torch.exp(-lv[mask])).mean()
            else:loss=(pred[mask]-target[mask]).square().mean()
            assert torch.isfinite(loss);loss.backward()
            for p in model.parameters():
                if p.grad is not None:assert torch.isfinite(p.grad).all()
            if method=='N':
                assert lv.grad is not None and torch.isfinite(lv.grad).all() and lv.grad.abs().sum()>0
                assert any(p.grad is not None and p.grad.abs().sum()>0 for name,p in model.named_parameters() if name.startswith('logvar_'))
            model_result={'all_pass':True,'stream_full':True,'reset_padded_batch':True,'future_invariance':True,'fast_intervention':'invariant' if var=='H' else 'responsive','save_load':True,'real_target_backward':True}
            if method=='T':model_result.update(receptive_field_rows=127,remote_history_mutation_checked=rf_checked,note='A supplied prefix longer than127 rows is needed for the remote-history intervention' if not rf_checked else '')
            if method=='N':model_result['variance_gradient']=True
            method_result['variants'][var]=model_result
        method_result['all_pass']=True;result[method]=method_result
    return {'all_pass':True,'scope':'Untrained real-input model plumbing; no scientific performance scores','methods':result}

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');args=p.parse_args()
    if not args.plan:p.error('This module only exposes audit_models to the registered caller; use --plan for metadata')
    print(json.dumps({'methods':METHODS,'architectures':ARCHITECTURES,'audit':'Requires caller guard for committed registration and immutable source pins; no data loaded'}))
if __name__=='__main__':main()
