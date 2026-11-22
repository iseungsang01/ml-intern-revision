"""Real-data CUDA teardown probe; not a training or scientific result.

Run from repo root with py -X faulthandler. Modes isolate initialization,
real-segment forward/backward and deterministic explicit cleanup.
"""
import argparse
import gc
import os
import sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('import','cuda','linear','forward','backward'),default='backward')
    ap.add_argument('--cleanup',action='store_true')
    ap.add_argument('--threads-before',action='store_true')
    ap.add_argument('--disable-cudnn',action='store_true')
    args=ap.parse_args()
    if args.threads_before:
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            os.environ[key]='2'
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.disable_cudnn:torch.backends.cudnn.enabled=False
    print('torch imported',flush=True)
    if args.mode=='import':return
    assert torch.cuda.is_available()
    torch.cuda.init()
    print('cuda initialized',flush=True)
    if args.mode=='cuda':
        torch.cuda.synchronize()
        return
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import common85 as M
    import data85 as D
    from model85 import create_model
    # The production model pins cuDNN off. This diagnostic deliberately selects
    # its own backend AFTER that import so the original failure stays reproducible.
    torch.backends.cudnn.enabled=not args.disable_cudnn
    G,R=M.load_sources()
    manifest=D.unit_manifest(M.select_units()[0])
    stats=D.fit_stats(manifest['train'][:12],G,R)
    segs=D.build_segments(manifest['train'][:1],stats,'D10',G,R)
    seg=next(s for s in segs if s['mask'].any())
    x=torch.from_numpy(seg['x']).unsqueeze(0).cuda()
    y=torch.from_numpy(seg['y']).unsqueeze(0).cuda()
    mask=torch.from_numpy(seg['mask']).unsqueeze(0).cuda()
    model=create_model('X',1701)
    if args.mode=='linear':
        weight=model.lstm_ti.weight_ih_l0[:160].detach().cuda()
        pred=torch.nn.functional.linear(x,weight)
    else:
        model=model.cuda()
        pred=model(x)
    assert torch.isfinite(pred).all()
    print('real segment forward complete',flush=True)
    if args.mode=='backward':
        loss=((pred-y)[mask]**2).mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        print('real segment backward complete',flush=True)
    torch.cuda.synchronize()
    if args.cleanup:
        model.zero_grad(set_to_none=True)
        del pred,x,y,mask,model
        if args.mode=='linear':del weight
        if args.mode=='backward':del loss
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        print('explicit cleanup complete',flush=True)
    print('probe returning normally',flush=True)

if __name__=='__main__':
    if '--worker' in sys.argv:
        sys.argv.remove('--worker')
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(main).result()
        print('worker joined; interpreter exiting normally',flush=True)
    else:main()
