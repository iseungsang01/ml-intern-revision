"""Independent causal dense/packed gradients and seven-rule objective audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import sys,json,copy
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import runtime91 as M


def main():
    pins=M.assert_registered();M.baseline_paths()
    import torch
    import common85 as B,data85 as D,train85 as T
    from model85 import B85Model
    from validation91 import cached_batches,cached_validate
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    g,r=B.load_sources();u=B.select_units()[0];manifest=D.unit_manifest(u);stats=D.fit_stats(manifest['train'],g,r)
    candidates=D.build_segments(manifest['train'][:12],stats,'D10',g,r)
    real=[s for s in candidates if len(s['x'])>=48 and s['mask'][:48].any()][:2];assert len(real)==2
    ss=[{k:(v[:n] if k in ('x','y','mask','gi') else v) for k,v in s.items()} for s,n in zip(real,(31,48))]
    batch=T.make_batches(ss,2)[0];x=batch['x'].double();y=batch['y'].double();mask=batch['mask'];lens=batch['lengths']
    torch.manual_seed(911);packed=B85Model('X').double();dense=M.dense_class()('X').double();dense.load_state_dict(packed.state_dict())
    for model in (packed,dense):model.eval();model.lstm_ti.dropout=0
    valid=torch.arange(48)[None,:]<lens[:,None]
    a=packed(x,lens);b=dense(x,lens);assert torch.allclose(a[valid],b[valid],atol=1e-10,rtol=1e-10)
    mu=torch.tensor(stats['ti_mu'],dtype=torch.float64);sd=torch.tensor(stats['ti_sd'],dtype=torch.float64)
    la=T.compute_loss(a,y,mask,mu,sd,'cpu');lb=T.compute_loss(b,y,mask,mu,sd,'cpu');assert torch.allclose(la,lb,atol=1e-10,rtol=1e-10)
    la.backward();lb.backward()
    for (_,p),(_,q) in zip(packed.named_parameters(),dense.named_parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:assert torch.allclose(p.grad,q.grad,atol=1e-10,rtol=1e-8)
    mutated=x.clone();mutated[~valid]+=1000
    assert torch.allclose(dense(mutated,lens)[valid],b[valid],atol=1e-10,rtol=1e-10)
    model=M.dense_class()('X').float().eval();by_rule={}
    for rule in B.TRAIN_RULES:
        source=D.build_segments(manifest['val'][:1],stats,rule,g,r)
        source=next(s for s in source if s['mask'][:64].any())
        by_rule[rule]=[{k:(v[:64] if k in ('x','y','mask','gi') else v) for k,v in source.items()}]
    batches,_=cached_batches(by_rule,T,'cpu');got,detail=cached_validate(model,batches,'cpu');independent=[]
    with torch.no_grad():
        for rule,segs in by_rule.items():
            err=[];counts=[]
            for s in segs:
                p=model(torch.from_numpy(s['x']).unsqueeze(0)).numpy()[0];m=s['mask'];e=(p.astype(np.float64)-s['y'])**2
                err.append(np.where(m,e,0).sum(0));counts.append(m.sum(0))
            sq=np.sum(err,0);ct=np.sum(counts,0);has=ct>0;independent.append(float(np.mean(sq[has]/ct[has])))
    assert np.isclose(got,np.mean(independent),atol=1e-6,rtol=1e-5)
    assert pins==M.sources()
    report={'passes':True,'sources':pins,'float64_valid_output_loss_gradient_equivalence':True,'padding_invariance':True,'numpy_seven_rule_objective':True,'scope':'real TRAIN/VAL input only; not fitted accuracy or measured speed'}
    M.save(M.CACHE/'lead_checks.json',report);M.PAPER.joinpath('b91_lead_checks.md').write_text('# B.91 independent arithmetic audit\n\n'+json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('PASS: B91 independent float64 gradients and NumPy seven-rule objective')


if __name__=='__main__':main()
