"""Real-input and loss readiness; no fitted scientific errors or synthetic rows."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import common90 as M
import data85 as D
import checks85 as B
from adapter90 import activate

def guard():M.assert_registered();M.assert_frozen()

def loss_checks(segments):
    import torch
    result={}
    for method in M.METHODS:
        t=activate(method);s=segments[0];batch=t.make_batches([s],1)[0]
        model=t.create_model('X',1701).eval();p=model(batch['x'],batch['lengths']);y,mask=batch['y'],batch['mask']
        mu=torch.tensor(500.);sd=torch.tensor(100.);actual=t.compute_loss(p,y,mask,mu,sd,'cpu');parts=[];pen=[]
        lv=model.latest_logvar if method=='N' else None
        for h in range(2):
            keep=mask[:,:,h]
            if not keep.any():continue
            e=p[:,:,h][keep]-y[:,:,h][keep];a=e.abs()
            if method=='U':v=torch.where(a<=1.,.5*e.square(),a-.5).mean()
            elif method=='V':v=(2.5*torch.log(1.+e.square()/4.)).mean()
            elif method=='N':
                l=lv[:,:,h][keep];v=(.5*l+.5*e.square()/l.exp()).mean()
            else:v=e.square().mean()
            parts.append(v);pen.append(torch.maximum(-mu/sd-p[:,:,h][keep],torch.zeros_like(e)).mean())
        assert parts,'Real segment must have supervised labels'
        expected=sum(parts)/len(parts)+.1*sum(pen)/len(pen)
        assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
        changed=y.clone();changed[~mask]+=1000
        assert torch.allclose(actual,t.compute_loss(p,changed,mask,mu,sd,'cpu'),atol=1e-6,rtol=0)
        actual.backward()
        assert all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters())
        if method=='N':assert any(q.grad is not None and q.grad.abs().sum()>0 for name,q in model.named_parameters() if 'var' in name),'Variance head has no gradient'
        # Inherited validation function remains byte-identical; evaluate it on the real rule batch.
        import adapter88
        baseline=adapter88.baseline_trainer()
        assert t.validate.__code__.co_code==baseline.validate.__code__.co_code
        val=t.validate(model,{'D10':[s]},mu,sd,'cpu')
        with torch.no_grad():
            pp=model(batch['x'],batch['lengths'])
            mse=sum((pp[:,:,h][mask[:,:,h]]-y[:,:,h][mask[:,:,h]]).square().mean().item() for h in range(2) if mask[:,:,h].any())/sum(bool(mask[:,:,h].any()) for h in range(2))
        assert abs(val-mse)<1e-5*max(1.,abs(mse))
        result[method]={'masked_loss_recomputed':True,'masked_targets_ignored':True,'finite_gradients':True,'pure_MSE_validation':True}
    return result

def main():
    guard();sources=M.source_inventory();M.CACHE.mkdir(parents=True,exist_ok=True)
    M.save(M.CACHE/'source_inventory.json',{'execution_sources':sources,'frozen':M.assert_frozen()})
    import torch
    torch.set_num_threads(1)
    G,R=M.load_sources();units=M.select_units();results={}
    print('B90 inherited real-data checks',flush=True)
    results['anchors']=B.check_anchors(G,R)
    results['manifests']=B.check_manifests_disjoint(units)
    results['stats_heldout']=B.check_stats_held_out(G,R,units)
    examples=[]
    for design in ('V1','V2','V3','V4a'):
        print('B90 input design '+design,flush=True)
        u=next(u for u in units if u['name'].startswith(design));manifest=D.unit_manifest(u);stats=D.fit_stats(manifest['train'],G,R)
        results[design+'/prefix']=B.check_prefix_invariance(G,R,stats,u)
        results[design+'/K10']=B.check_k10_prefix(G,R,stats,u)
        results[design+'/labels']=B.check_label_mutation_invariance(G,R,stats,u)
        results[design+'/values']=B.check_label_values(R,G,stats,u)
        for rule in M.RULES:
            segs=D.build_segments(manifest['train'][:1],stats,rule,G,R)
            assert segs and all(s['x'].shape[1]==22 for s in segs)
            if design=='V1' and rule=='D10':
                candidates=D.build_segments(manifest['train'][:12],stats,rule,G,R)
                for i,s in enumerate(sorted(candidates,key=lambda a:len(a['x']),reverse=True)[:2]):
                    limit=160 if i==0 else 113
                    if s['mask'][:limit].any():examples.append({k:(v[:limit] if k in ('x','y','mask','gi') else v) for k,v in s.items()})
            results[design+'/'+rule]={'segments':len(segs),'first22_original':True,'targets_masks_original':True}
    assert len(examples)==2
    from model_checks90 import audit_models
    print('B90 actual-prefix model contracts',flush=True)
    results['models']=audit_models(examples,methods=M.METHODS,device='cpu',guard=guard)
    print('B90 independent loss recomputation',flush=True)
    results['losses']=loss_checks(examples)
    assert sources==M.source_inventory(),'Sources changed during audit'
    M.save(M.CACHE/'input_checks.json',{'all_pass':True,'execution_sources':sources,'frozen_sources':M.assert_frozen(),'results':B._deep_convert(results)})
    text='# B.90 readiness\n\nPASS: real D1 inputs, all four designs/eight rules, inherited anchors/prefix and held-out-statistics checks, actual-segment model contracts and independently recomputed masked loss with unchanged pure-MSE validation. No synthetic observations or scientific test scoring. GPU plumbing smokes remain gated on clean B88 completion.\n'
    M.C.save_text(M.PAPER/'b90_readiness.md',text);M.C.save_text(M.PAPER/'b90_checks.md',text)
    print('B90 real-input/model/loss checks passed')
if __name__=='__main__':main()
