"""One normal-exit real-input runtime probe. Supervisor alone attests exit status."""
import argparse,copy,hashlib,json,os,platform,sys,time
from pathlib import Path
import runtime91 as M

def main():
    p=argparse.ArgumentParser();p.add_argument('--path',required=True);p.add_argument('--rep',required=True,type=int);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    cfg=M.PATHS.get(a.path)
    if cfg is None:
        assert a.path in ('py312_cuda_dense_cudnn','py312_cuda_packed_cudnn')
        cfg={'device':'cuda','dense':'_dense_' in a.path,'cudnn':True,'threads':2,'mkldnn':False}
        assert sys.version_info[:2]==(3,12)
    pins=M.assert_registered()
    stop_hash=M.gpu_ready() if cfg['device']=='cuda' else None
    if cfg['device']=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    M.baseline_paths()
    import numpy as np
    import torch
    from model85 import B85Model
    import train85 as T
    # Input loading and CPU reference never initialize CUDA.
    started=time.perf_counter();segments,stats,manifest,G,R,binding=M.real_inputs();load_seconds=time.perf_counter()-started
    torch.set_num_threads(2);torch.backends.mkldnn.enabled=False;torch.backends.cudnn.enabled=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.manual_seed(1701);base=B85Model('X');initial=copy.deepcopy(base.state_dict())
    b=T.make_batches(segments,16)[0]
    for s in segments:assert s['mask'].any()
    valid=torch.arange(b['x'].shape[1])[None,:]<b['lengths'][:,None]
    assert not b['mask'][~valid].any()
    def model(dense,device):
        result=(M.dense_class() if dense else B85Model)('X');result.load_state_dict(initial);result.to(device)
        result.eval();result.lstm_ti.dropout=0.;result.lstm_ti.train(True)
        return result
    def sync(device):
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    def one_step(net,batch,opt,device,timed=False):
        batch={k:(v.to(device) if k in ('x','y','mask') else v) for k,v in batch.items()}
        opt.zero_grad(set_to_none=True);sync(device);start=time.perf_counter()
        pred=net(batch['x'],batch['lengths']);loss=T.compute_loss(pred,batch['y'],batch['mask'],torch.tensor(stats['ti_mu'],device=device),torch.tensor(stats['ti_sd'],device=device),device)
        sync(device);f=time.perf_counter()-start;start=time.perf_counter();loss.backward();sync(device);back=time.perf_counter()-start
        gradients={k:None if q.grad is None else q.grad.detach().cpu().clone() for k,q in net.named_parameters()} if not timed else None
        start=time.perf_counter();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();sync(device);update=time.perf_counter()-start
        assert torch.isfinite(loss) and torch.isfinite(norm)
        return pred.detach().cpu() if not timed else None,gradients,{'forward_loss_seconds':f,'backward_seconds':back,'clip_optimizer_seconds':update}
    ref=model(False,'cpu');ro=torch.optim.AdamW(ref.parameters(),lr=.001,weight_decay=.0001)
    refout,refgrad,_=one_step(ref,b,ro,'cpu');reference_state={k:v.detach().clone() for k,v in ref.state_dict().items()}
    refopt=copy.deepcopy(ro.state_dict())
    torch.set_num_threads(cfg['threads']);torch.backends.mkldnn.enabled=cfg['mkldnn'];torch.backends.cudnn.enabled=cfg['cudnn']
    device=torch.device(cfg['device'])
    if cfg['device']=='cuda':
        M.gpu_ready();assert torch.cuda.is_available()
    assert torch.get_num_threads()==cfg['threads'] and torch.backends.mkldnn.enabled==cfg['mkldnn'] and torch.backends.cudnn.enabled==cfg['cudnn']
    candidate=model(cfg['dense'],device);batch={k:(v.to(device) if k in ('x','y','mask') else v) for k,v in b.items()}
    opt=torch.optim.AdamW(candidate.parameters(),lr=.001,weight_decay=.0001)
    out,grad,_=one_step(candidate,batch,opt,device)
    assert torch.allclose(out[valid],refout[valid],atol=1e-5,rtol=1e-4),'Valid output mismatch'
    maxgrad=0.;maxparam=0.
    for name,param in candidate.named_parameters():
        left,right=grad[name],refgrad[name];assert (left is None)==(right is None),'Gradient None pattern changed'
        if left is not None:
            assert torch.isfinite(left).all() and torch.allclose(left,right,atol=1e-4,rtol=1e-3),'Gradient mismatch: '+name
            maxgrad=max(maxgrad,float((left-right).abs().max()))
        got=param.detach().cpu();expected=reference_state[name]
        assert torch.allclose(got,expected,atol=1e-4,rtol=1e-3),'AdamW parameter mismatch: '+name
        maxparam=max(maxparam,float((got-expected).abs().max()))
    co=opt.state_dict()
    for key,state in refopt['state'].items():
        for name,value in state.items():
            got=co['state'][key][name]
            if torch.is_tensor(value):assert torch.allclose(got.cpu(),value,atol=1e-4,rtol=1e-3),'Optimizer state mismatch'
    # Reset to cloned initial weights for independent stream, padding and persistence tests.
    candidate=model(cfg['dense'],device)
    def masked_gradients(x):
        candidate.zero_grad(set_to_none=True);pred=candidate(x,batch['lengths'])
        loss=T.compute_loss(pred,batch['y'],batch['mask'],torch.tensor(stats['ti_mu'],device=device),torch.tensor(stats['ti_sd'],device=device),device)
        loss.backward()
        return {k:None if q.grad is None else q.grad.detach().clone() for k,q in candidate.named_parameters()}
    unmodified_grad=masked_gradients(batch['x'])
    altered_padding=batch['x'].clone();altered_padding[~valid.to(device)]+=1000
    altered_grad=masked_gradients(altered_padding)
    for name,left in unmodified_grad.items():
        right=altered_grad[name];assert (left is None)==(right is None)
        if left is not None:assert torch.allclose(left,right,atol=1e-4,rtol=1e-3),'Padding changes masked gradients: '+name
    candidate.zero_grad(set_to_none=True)
    with torch.no_grad():
        full=candidate(batch['x'],batch['lengths'])
        mutated=batch['x'].clone();mutated[~valid.to(device)]+=1000
        assert torch.allclose(candidate(mutated,batch['lengths'])[valid.to(device)],full[valid.to(device)],atol=1e-5,rtol=1e-4),'Padding changes valid prefixes'
        for i in (0,len(segments)-1):
            n=int(b['lengths'][i]);xx=batch['x'][i:i+1,:n];whole=candidate(xx,torch.tensor([n]));state=None;stream=[]
            for row in xx[0]:
                y,state=candidate.forward_step(row.reshape(1,1,22),state);stream.append(y)
            assert torch.allclose(torch.stack(stream,1),whole,atol=1e-5,rtol=1e-4),'Streaming mismatch'
            assert torch.allclose(whole,full[i:i+1,:n],atol=1e-5,rtol=1e-4),'Segment reset mismatch'
        cut=max(1,int(b['lengths'].min())//2);changed=batch['x'].clone();changed[:,cut:]+=1000
        assert torch.allclose(candidate(changed,batch['lengths'])[:,:cut],full[:,:cut],atol=1e-5,rtol=1e-4),'Future input changes prefix'
    checkpoint=a.output.with_suffix('.weights.pt');checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save(candidate.state_dict(),checkpoint)
    restored=model(cfg['dense'],device);restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    with torch.no_grad():assert torch.allclose(restored(batch['x'],batch['lengths']),full,atol=1e-6,rtol=1e-5)
    candidate=model(cfg['dense'],device);opt=torch.optim.AdamW(candidate.parameters(),lr=.001,weight_decay=.0001);timings=[]
    wall=time.perf_counter()
    for step in range(7):
        start=time.perf_counter();step_batch=T.make_batches(segments,16)[0];collated=time.perf_counter()-start
        _,_,timing=one_step(candidate,step_batch,opt,device,True);sync(device);timing['batch_seconds']=time.perf_counter()-start;timing['collation_seconds']=collated
        if step>=2:timings.append(timing)
    train_wall=time.perf_counter()-wall
    from validation91 import audit_validation
    validation=audit_validation(candidate,T,stats,manifest,G,R,device)
    assert pins==M.sources(),'Sources changed during probe'
    report={'all_pass':True,'path':a.path,'rep':a.rep,'config':cfg,'sources':pins,'input_binding':binding,'author_stop_sha256':stop_hash,
            'software':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'mkldnn_enabled':torch.backends.mkldnn.enabled,'cudnn_enabled':torch.backends.cudnn.enabled,'threads':torch.get_num_threads(),'gpu_name':torch.cuda.get_device_name(0) if cfg['device']=='cuda' else None,'platform':platform.platform()},
            'equivalence':{'outputs':True,'gradients':True,'gradient_none_pattern':True,'AdamW_parameters_and_state':True,'clipping_norm1':True,'padding_gradient_invariance':True,'padding_future_prefix_reset_stream_saveload':True,'stream_scope':'first and last real training prefixes','max_abs_gradient_difference':maxgrad,'max_abs_parameter_difference':maxparam,'dropout':0.,'RNN_training_flag':True,'reference':'CPU packed native threads2 FP32'},
            'training':{'warmup_steps':2,'measured_steps':5,'steps':timings,'median_batch_seconds':float(np.median([v['batch_seconds'] for v in timings])),'seven_step_wall_seconds':train_wall,'load_prepare_seconds':load_seconds,'timing_includes':'per-batch real collation and host/device transfer, forward, full frozen masked loss, backward, zero_grad, clipping1, AdamW and explicit synchronization; one-time source preparation and validation separately reported'},'validation':validation}
    M.save(a.output,report)
    sync(device)
    print(json.dumps({'all_pass':True,'path':a.path,'median_batch_seconds':report['training']['median_batch_seconds']}),flush=True)
    # Ordinary interpreter teardown is part of eligibility, observed by the parent.
if __name__=='__main__':main()
