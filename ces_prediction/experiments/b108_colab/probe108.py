"""One fixed real-workload timing child; never a scientific checkpoint."""
import argparse,json,os,time,platform
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np,torch
from portable108 import TemperatureLSTM,loss108
def wait_for(p):
    deadline=time.monotonic()+590
    while not p.exists():
        assert time.monotonic()<deadline,'Barrier timeout';time.sleep(.02)
def main(a):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=a.backend=='cudnn';torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();torch.manual_seed(1701)
    model=TemperatureLSTM().to('cuda');model.load_state_dict(torch.load(Path(a.root)/(a.unit+'.pt'),map_location='cpu',weights_only=True))
    with np.load(Path(a.root)/(a.unit+'.npz'),allow_pickle=False) as f:data={k:f[k] for k in f.files}
    def batch(prefix):return tuple(torch.from_numpy(data[prefix+'_'+k]).to('cuda') for k in ('x','y','mask'))
    model.eval()
    with torch.no_grad():
        ref=model(torch.from_numpy(data['ref_x']).to('cuda')).cpu().numpy()
        assert np.allclose(ref,data['ref_prediction'],atol=1e-4,rtol=1e-4),'Original branch eval disagreement'
    reference_max_error=float(np.max(np.abs(ref-data['ref_prediction'])))
    mu=torch.tensor(float(data['mu']),device='cuda');sd=torch.tensor(float(data['sd']),device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    def step():
        model.train();x,y,mask=batch('train');opt.zero_grad(set_to_none=True);p=model(x);loss=loss108(p,y,mask,mu,sd)
        assert mask[...,0].any(),'Missing real F supervision'
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    out=Path(a.output);out.with_suffix('.ready').write_text('ready');wait_for(Path(a.gate))
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):step()
    torch.cuda.synchronize();out.with_suffix('.warm').write_text('warm');wait_for(Path(a.time_gate))
    latencies=[];begin=time.perf_counter()
    for _ in range(5):
        torch.cuda.synchronize();t=time.perf_counter();step();torch.cuda.synchronize();latencies.append(time.perf_counter()-t)
    end=time.perf_counter();model.eval();torch.cuda.synchronize();t=time.perf_counter()
    with torch.no_grad():
        x,y,m=batch('val');v=loss108(model(x),y,m,mu,sd);assert torch.isfinite(v)
    torch.cuda.synchronize();val_seconds=time.perf_counter()-t
    model.train();x,y,m=batch('long');opt.zero_grad(set_to_none=True);v=loss108(model(x),y,m,mu,sd);assert torch.isfinite(v);v.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());torch.cuda.synchronize()
    result={'passed':True,'unit':a.unit,'backend':a.backend,'timed_steps':5,'timed_begin':begin,'timed_end':end,'latency_seconds':latencies,
            'median_step_seconds':float(np.median(latencies)),'val_seconds':val_seconds,'peak_reserved_MiB':torch.cuda.max_memory_reserved()/1024**2,
            'train_shape':list(data['train_x'].shape),'long_shape':list(data['long_x'].shape),'reference_max_abs_error':reference_max_error,
            'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'cudnn_enabled':torch.backends.cudnn.enabled,
            'tf32':False,'dtype':'float32','dropout':.1,'threads':2,'cpu_count':os.cpu_count(),'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0)}
    out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','unit','backend','output','gate','time-gate'):p.add_argument('--'+n,required=True)
    main(p.parse_args())
