"""Seven-rule cached-prefix validator with unchanged equal-rule/head aggregation."""
import time

def build_validation(manifest,stats,G,R):
    import data85 as D
    from common85 import TRAIN_RULES
    out={}
    for rule in TRAIN_RULES:
        segs=sorted(D.build_segments(manifest['val'],stats,rule,G,R),key=lambda s:(s['fidx'],s['segment']))
        selected=[s for s in segs if s['mask'][:512].any()][:16]
        assert selected,'No real validation supervision for '+rule
        out[rule]=[{k:(v[:512].copy() if k in ('x','y','mask','gi') else v) for k,v in s.items()} for s in selected]
    return out

def cached_batches(by_rule,trainer,device):
    started=time.perf_counter();result={}
    if str(device).startswith('cuda'):
        import torch
        required=sum(s[k].nbytes for ss in by_rule.values() for s in ss for k in ('x','y','mask'))
        free,_=torch.cuda.mem_get_info()
        assert free>required+256*1024**2,'Insufficient free memory for validation cache'
    for rule,segs in by_rule.items():
        result[rule]=[{k:(v.to(device) if k in ('x','y','mask') else v) for k,v in b.items()} for b in trainer.make_batches(segs,16)]
    if str(device).startswith('cuda'):
        import torch;torch.cuda.synchronize()
    return result,time.perf_counter()-started

def cached_validate(model,batches,device):
    import torch
    model.eval();sums=[];counts=[]
    with torch.no_grad():
        for rule,bs in batches.items():
            sq=torch.zeros(2,device=device,dtype=torch.float64);ct=torch.zeros(2,device=device,dtype=torch.int64)
            for b in bs:
                pred=model(b['x'],b['lengths']);mask=b['mask']
                # Match the inherited FP32 per-head reduction, then accumulate in double.
                sq+=torch.stack([((pred[:,:,h][mask[:,:,h]]-b['y'][:,:,h][mask[:,:,h]])**2).sum() for h in range(2)]).to(torch.float64)
                ct+=mask.sum(dim=(0,1))
            sums.append(sq);counts.append(ct)
        allvalues=torch.stack((torch.stack(sums),torch.stack(counts).to(torch.float64))).cpu().numpy()
    rulemeans=[]
    for sq,ct in zip(allvalues[0],allvalues[1]):
        valid=ct>0
        if valid.any():rulemeans.append(float((sq[valid]/ct[valid]).mean()))
    return sum(rulemeans)/len(rulemeans),{'squared_sums':allvalues[0].tolist(),'counts':allvalues[1].tolist(),'rules':list(batches)}

def audit_validation(model,trainer,stats,manifest,G,R,device):
    import numpy as np
    import torch
    started=time.perf_counter();by_rule=build_validation(manifest,stats,G,R);prepare=time.perf_counter()-started
    batches,transfer=cached_batches(by_rule,trainer,device)
    start=time.perf_counter();original=trainer.validate(model,by_rule,torch.tensor(stats['ti_mu'],device=device),torch.tensor(stats['ti_sd'],device=device),device)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    old_seconds=time.perf_counter()-start
    start=time.perf_counter();new,details=cached_validate(model,batches,device)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    new_seconds=time.perf_counter()-start
    assert np.isclose(original,new,atol=1e-6,rtol=1e-5),'Cached validation changes selector'
    return {'all_pass':True,'original_objective':original,'cached_objective':new,'prepare_seconds':prepare,'cache_transfer_seconds':transfer,
            'original_seconds':old_seconds,'cached_seconds':new_seconds,'cached_end_to_end_seconds':prepare+transfer+new_seconds,
            'rows_per_rule':{r:sum(len(s['x']) for s in ss) for r,ss in by_rule.items()},**details,
            'scope':'Actual validation prefixes only, all seven rules and original labels/masks; not full-epoch timing.'}
