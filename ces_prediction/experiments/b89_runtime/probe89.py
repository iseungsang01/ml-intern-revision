"""One bounded real-input B89 CPU preflight or fresh-process CUDA probe."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--repeat',type=int,required=True)
    ap.add_argument('--binding-hash',required=True);ap.add_argument('--output',required=True);ap.add_argument('--cpu-check',action='store_true')
    a=ap.parse_args()
    from run89 import CANDIDATES,CONTROL_ENV,ROOT,CACHE,UNIT,binding,save
    assert a.candidate in CANDIDATES
    for key in CONTROL_ENV:os.environ.pop(key,None)
    os.environ.update(CANDIDATES[a.candidate]['env'])
    if a.cpu_check:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    output=Path(a.output).resolve();assert output.is_relative_to(CACHE.resolve())
    bind=binding();assert bind['hash']==a.binding_hash,'Probe source/input binding mismatch'
    # All candidate environment controls above are set before torch import.
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b85_unified'))
    import common85 as M,data85 as D,train85 as T
    from model85 import create_model
    assert M.select_units()[0]['name']==UNIT
    folder=ROOT/'data/.b85/runs'/f'{UNIT}__1701__X'
    stats=json.loads((folder/'stats.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
    assert set(manifest['train']).isdisjoint(manifest['val']) and set(manifest['train']).isdisjoint(manifest['test'])
    segments=[]
    for fid in sorted(manifest['train']):
        for segment in D.build_segments([fid],stats,'D10'):
            n=min(128,len(segment['x']))
            if n and segment['mask'][:n].any():
                segments.append({k:(v[:n].copy() if hasattr(v,'shape') else v) for k,v in segment.items()})
                if len(segments)==2:break
        if len(segments)==2:break
    assert len(segments)==2,'Two supervised real prefixes required'
    batch=T.make_batches(segments,2)[0]
    x,y,mask,lengths=(batch[k] for k in ('x','y','mask','lengths'))
    assert x.shape[0]==2 and x.shape[1]<=128 and x.shape[2]==22 and mask.any()
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    h=hashlib.sha256()
    for key in ('x','y','mask','lengths'):h.update(batch[key].numpy().tobytes())
    input_hash=h.hexdigest()
    selected=[{'fidx':int(s['fidx']),'segment':int(s['segment']),'rows':len(s['x']),
               'first_gi':int(s['gi'][0]),'last_gi':int(s['gi'][-1])} for s in segments]
    torch.manual_seed(1701)
    cpu=create_model('X',1701).cpu().eval()
    mu=torch.tensor(float(stats['ti_mu']));sd=torch.tensor(float(stats['ti_sd']))
    cpu.zero_grad(set_to_none=True);reference=cpu(x,lengths)
    reference_loss=T.compute_loss(reference,y,mask,mu,sd,'cpu');reference_loss.backward()
    reference=reference.detach().clone()
    reference_grads={name:None if p.grad is None else p.grad.detach().clone() for name,p in cpu.named_parameters()}
    assert torch.isfinite(reference).all() and all(g is None or torch.isfinite(g).all() for g in reference_grads.values())
    cpu.zero_grad(set_to_none=True)
    checks={'real_training_prefixes':True,'finite_cpu_forward_and_gradients':True}
    # Prefix/streaming/reset/save-load checks follow frozen B85 guards, on bounded real prefixes.
    def guards(model,bx,by,bm,device):
        with torch.no_grad():
            full=model(bx,lengths);state=None;steps=[];n=int(lengths[0])
            for i in range(n):
                step,state=model.forward_step(bx[:1,i:i+1],state);steps.append(step)
            assert (full[:1,:n]-torch.stack(steps,dim=1)).abs().max().item()<1e-5,'Streaming mismatch'
            for i in range(2):
                separate=model(bx[i:i+1],lengths[i:i+1]);n=int(lengths[i])
                assert (full[i,:n]-separate[0,:n]).abs().max().item()<1e-5,'State reset mismatch'
            clone=copy.deepcopy(model)
            path=output.parent/'weights_roundtrip.pt';path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(model.state_dict(),path)
            clone.load_state_dict(torch.load(path,map_location=device,weights_only=True));path.unlink()
            assert torch.allclose(full,clone(bx,lengths),atol=1e-6),'Save/load mismatch'
        return {'streaming':True,'state_reset':True,'save_load':True}
    timing=None;gpu=None;errors={}
    if a.cpu_check:
        assert not torch.cuda.is_available(),'CPU preflight must not see CUDA'
        checks.update(guards(cpu,x,y,mask,'cpu'))
    else:
        assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
        # Imported frozen model/trainer pin cuDNN off. Override only inside THIS scout.
        torch.backends.cudnn.enabled=CANDIDATES[a.candidate]['cudnn']
        device=torch.device('cuda')
        model=copy.deepcopy(cpu).to(device).eval()
        bx,by,bm=x.to(device),y.to(device),mask.to(device)
        dmu,dsd=mu.to(device),sd.to(device)
        model.zero_grad(set_to_none=True);actual=model(bx,lengths)
        loss=T.compute_loss(actual,by,bm,dmu,dsd,device);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(actual).all() and torch.isfinite(loss)
        forward=actual.detach().cpu();errors['forward_max_abs']=float((forward-reference).abs().max())
        assert torch.allclose(forward,reference,atol=1e-5,rtol=1e-4),'CPU/CUDA forward mismatch'
        grad_max=0.
        for name,p in model.named_parameters():
            ref=reference_grads[name]
            assert (p.grad is None)==(ref is None),'Gradient None pattern: '+name
            if ref is not None:
                actual_grad=p.grad.detach().cpu();assert torch.isfinite(actual_grad).all()
                grad_max=max(grad_max,float((actual_grad-ref).abs().max()))
                assert torch.allclose(actual_grad,ref,atol=1e-4,rtol=1e-3),'CPU/CUDA gradient mismatch: '+name
        errors['gradient_max_abs']=grad_max;checks['cpu_cuda_forward_gradient_equivalence']=True
        checks.update(guards(model,bx,by,bm,device))
        torch.cuda.reset_peak_memory_stats();times=[]
        for step in range(13):
            torch.cuda.synchronize();started=time.perf_counter()
            model.zero_grad(set_to_none=True)
            pred=model(bx,lengths);objective=T.compute_loss(pred,by,bm,dmu,dsd,device);objective.backward()
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            assert torch.isfinite(objective)
            if step>=3:times.append(elapsed)
        assert len(times)==10 and all(t>0 for t in times)
        timing={'step_seconds':times,'median_step_seconds':float(np.median(times)),
                'min_step_seconds':min(times),'max_step_seconds':max(times),'warmup_steps':3,'timed_steps':10,
                'scope':'synchronized zero_grad+forward+frozen loss+backward; no optimizer updates'}
        gpu={'name':torch.cuda.get_device_name(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'cudnn_version':torch.backends.cudnn.version()}
    checks['all_pass']=True
    result={'candidate':a.candidate,'repeat':a.repeat,'binding_hash':bind['hash'],'input_hash':input_hash,
            'prefixes':selected,'input_shape':list(x.shape),'cpu_only':a.cpu_check,'seed':1701,
            'weight_source':bind['weight_source'],'checks':checks,'timing':timing,'gpu':gpu,
            'numeric_differences':errors,'candidate_environment':CANDIDATES[a.candidate]['env'],
            'torch':torch.__version__,'cuda':torch.version.cuda,'allow_tf32':False,
            'cudnn_enabled':False if a.cpu_check else CANDIDATES[a.candidate]['cudnn']}
    save(output,result)
    print('Probe artifact written; eligibility still requires normal parent-observed exit zero',flush=True)


if __name__=='__main__':main()
