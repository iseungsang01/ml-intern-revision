"""Pre-fit real-input/loss/model readiness; no TEST segments or scientific scores."""
import copy,io,json,time
import numpy as np
import torch
import common92 as M
import data88 as D
import data85 as B
from model92 import create_model
from prepare92 import prepare_unit,batches
def model_contract(model,segs,runtime):
    bb=batches(segs[:2],runtime['device'],batch_size=2);assert bb;b=bb[0];model.eval()
    with torch.no_grad():
        full=model(b['x'],b['lengths']);assert torch.isfinite(full).all()
        n=int(b['lengths'][0]);one=b['x'][:1,:n];state=None;out=[]
        for row in one[0]:
            p,state=model.forward_step(row.reshape(1,1,35),state);out.append(p)
        assert torch.allclose(full[:1,:n],torch.stack(out,1),atol=1e-5,rtol=0),'Full/stream mismatch'
        for i in range(len(segs[:2])):
            n=int(b['lengths'][i]);separate=model(b['x'][i:i+1,:n],b['lengths'][i:i+1])
            assert torch.allclose(full[i:i+1,:n],separate,atol=1e-5,rtol=0),'Padded/reset mismatch'
        changed=b['x'].clone();changed[:,max(1,int(b['lengths'][0])//2):,:13]+=1
        k=max(1,int(b['lengths'][0])//2)
        assert torch.allclose(full[:,:k],model(changed,b['lengths'])[:,:k],atol=1e-5,rtol=0)
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0);other=copy.deepcopy(model)
        other.load_state_dict(torch.load(buffer,map_location=runtime['device'],weights_only=True))
        assert torch.allclose(full,other(b['x'],b['lengths']),atol=1e-6,rtol=0)
    return {'finite':True,'stream_reset_padding':True,'causal_prefix':True,'save_load':True}
def main():
    pins=M.guard();sources=M.sources();M.CACHE.mkdir(parents=True,exist_ok=True)
    M.save(M.CACHE/'audit_inventory.json',{'sources':sources,'frozen':pins,'recorded':time.time()})
    torch.set_num_threads(1);G,R=M.load_sources();cases=[];example=None
    for unit in M.PILOT_UNITS:
        prep=prepare_unit(unit);stats=prep['identity']['stats'];manifest=prep['identity']['manifest']
        assert M.digest(prep['identity'])==prep['key']
        for change in ({'width':22},{'raw':{}},{'stats':{}},{'manifest':{}},{'sources':{}}):assert M.digest({**prep['identity'],**change})!=prep['key']
        for rule in M.TRAIN_RULES:
            old=B.build_segments(manifest['train'][:1],stats,rule,G,R);new=D.build_segments(manifest['train'][:1],stats,rule,G,R)
            for a,b in zip(new,old):
                assert np.array_equal(a['x'][:,:22],b['x'])
                for field in ('y','mask','gi'):assert np.array_equal(a[field],b[field])
                cached=next(s for s in prep['train'][rule] if (s['fidx'],s['segment'])==(a['fidx'],a['segment']))
                for field in ('x','y','mask','gi'):assert np.array_equal(cached[field],a[field])
            s=next(s for s in new if len(s['x'])>4);ii=s['gi'];j=len(ii)//2
            anc=D.anchors(rule,G)[ii];visible=set(anc[:j+1][anc[:j+1]>=0].tolist())
            gm=dict(G);gm['ti']=G['ti'].copy();invisible=ii[:j+1][~np.isin(ii[:j+1],list(visible))];gm['ti'][invisible]=np.nan
            altered=next(z for z in D.build_segments([s['fidx']],stats,rule,gm,R) if z['segment']==s['segment'])
            assert np.array_equal(s['x'][:j+1],altered['x'][:j+1])
            gm=dict(G);gm['fast']=G['fast'].copy();gm['fast'][ii[j+1:]]+=1000
            altered=next(z for z in D.build_segments([s['fidx']],stats,rule,gm,R) if z['segment']==s['segment'])
            assert np.array_equal(s['x'][:j+1],altered['x'][:j+1])
            cases.append(unit+'/'+rule)
        if example is None:example=prep
    # Freeze train-only stats; K10 excluded from preparation and selection.
    assert all('K10' not in example[k] for k in ('train','val'))
    chosen=[s for s in example['val']['D10'] if s['mask'].any()][:2];assert len(chosen)==2
    small=[]
    for s in chosen:
        end=max(32,int(np.flatnonzero(s['mask'].any(1))[0])+1);small.append({**s,**{k:s[k][:end] for k in ('x','y','mask','gi')}})
    runtime={'device':'cpu','dense':False,'cudnn':False,'threads':1,'mkldnn':False};reports={}
    for code in ('A','C','T','TX','TH'):
        reports[code]=model_contract(create_model(code,runtime,1701),small,runtime)
    a=create_model('A',runtime,1701);c=create_model('C',runtime,1701)
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
    from train92 import vector_loss,validate
    from train85 import compute_loss
    batch=batches(small,'cpu',batch_size=2)[0];stats=example['identity']['stats'];mu=torch.tensor(stats['ti_mu']);sd=torch.tensor(stats['ti_sd'])
    # Real model predictions and targets; independent original branching formula.
    a.train();pred=a(batch['x'],batch['lengths']);pred.retain_grad();v=vector_loss(pred,batch['y'],batch['mask'],mu,sd)
    old=compute_loss(pred,batch['y'],batch['mask'],mu,sd,'cpu');assert torch.allclose(v,old,atol=1e-6,rtol=1e-6)
    g1=torch.autograd.grad(v,pred,retain_graph=True)[0];g2=torch.autograd.grad(old,pred)[0];assert torch.allclose(g1,g2,atol=1e-6,rtol=1e-6)
    # Rebatch actual validation segments; exact per-rule/head aggregation remains.
    vv={r:[{**s,**{k:s[k][:max(32,int(np.flatnonzero(s['mask'].any(1))[0])+1)] for k in ('x','y','mask','gi')}} for s in example['val'][r] if s['mask'].any()][:2] for r in M.TRAIN_RULES}
    value1,_=validate(a,{r:batches(ss,'cpu',1) for r,ss in vv.items()});value2,_=validate(a,{r:batches(ss,'cpu',2) for r,ss in vv.items()})
    assert abs(value1-value2)<=1e-6*max(1,abs(value1))
    assert sources==M.sources(),'Source changed during checks'
    M.save(M.CACHE/'input_checks.json',{'all_pass':True,'sources':sources,'cases':cases,'models':reports,'vector_loss_gradient':True,'validation_rebatch':True,'test_accessed':False})
    M.C.save_text(M.PAPER/'b92_readiness.md','# B.92 readiness\n\nSource-bound real TRAIN/VAL input/cache, prefix mutation, model stream/reset/padding/save-load, vectorized loss/gradient and validation rebatching checks passed. No TEST segments or scores were read. Fresh selected-runtime/dropout.1 smoke children are required before production.\n')
    print(json.dumps({'all_pass':True,'cases':len(cases),'test_accessed':False}))
if __name__=='__main__':main()
