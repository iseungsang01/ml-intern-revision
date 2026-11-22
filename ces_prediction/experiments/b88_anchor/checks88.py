"""Real-input CPU readiness audit. No fitted-model errors are calculated."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time
import numpy as np
import torch
import common88 as M
import data88 as D
import data85 as B
from model88 import create_model
from adapter88 import source_inventory

def eligible(G,ii,j,rule,frames):
    prior=ii[:j];prior=prior[np.isfinite(G['ti'][prior])]
    if rule.startswith('D'):
        return prior[G['time'][prior]<=G['time'][ii[j]]-int(rule[1:])/1000+.003]
    if rule=='K10':return prior[(frames[prior]>=0)&(frames[prior]%10==0)]
    gaps=np.flatnonzero((np.diff(G['time'][ii[:j+1]])>.015)&(np.diff(G['time'][ii[:j+1]])<.045))
    return prior[prior<=ii[gaps[-1]]] if len(gaps) else prior[:0]

def main():
    M.CACHE.mkdir(parents=True,exist_ok=True)
    pins=M.assert_frozen()
    # Inventory is recorded before touching actual observations.
    sources=source_inventory()
    M.C.save_json(M.CACHE/'source_inventory.json',{'frozen':pins,'execution_sources':sources,'recorded':time.time()})
    torch.set_num_threads(1)
    G,R=M.load_sources();frames=np.load(M.DATA/'.b78/frames.npz',allow_pickle=True)['fr']
    details=[];model_seg=None
    units=M.select_units()
    for prefix in ('V1','V2','V3','V4a'):
        u=next(u for u in units if u['name'].startswith(prefix))
        manifest=D.unit_manifest(u);stats=D.fit_stats(manifest['train'],G,R)
        for rule in M.RULES:
            new=D.build_segments(manifest['test'][:1],stats,rule,G,R)
            old=B.build_segments(manifest['test'][:1],stats,rule,G,R)
            assert new and len(new)==len(old)
            for a,b in zip(new,old):
                assert np.array_equal(a['x'][:,:22],b['x'])
                for k in ('y','mask','gi'):assert np.array_equal(a[k],b[k])
                assert a['mask'].dtype==b['mask'].dtype
                ii=a['gi'];anc=D.anchors(rule,G)[ii]
                for j in sorted(set([0,len(ii)//2,len(ii)-1])):
                    e=eligible(G,ii,j,rule,frames)
                    assert anc[j]==(e[-1] if len(e) else -1)
            s=max(new,key=lambda x:len(x['gi']));ii=s['gi'];j=len(ii)//2;t=ii[j]
            visible=eligible(G,ii,j,rule,frames)
            invisible=ii[:j+1][~np.isin(ii[:j+1],visible)]
            for remove in (False,True):
                gm=dict(G);gm['ti']=G['ti'].copy()
                gm['ti'][invisible]=np.nan if remove else gm['ti'][invisible]+1000
                altered=D.build_segments([s['fidx']],stats,rule,gm,R)
                altered=next(a for a in altered if a['segment']==s['segment'])
                assert np.array_equal(s['x'][:j+1],altered['x'][:j+1]),'CES prefix violation'
            gm=dict(G);gm['fast']=G['fast'].copy();gm['fast'][ii[j+1:]]+=1000
            altered=next(a for a in D.build_segments([s['fidx']],stats,rule,gm,R) if a['segment']==s['segment'])
            assert np.array_equal(s['x'][:j+1],altered['x'][:j+1]),'Fast prefix violation'
            anc=D.anchors(rule,G)[ii];has=np.flatnonzero(anc>=0)
            if len(has):
                k=int(has[len(has)//2]);source=int(anc[k]);gm=dict(G);gm['fast']=G['fast'].copy();gm['fast'][source]+=1
                changed=next(a for a in D.build_segments([s['fidx']],stats,rule,gm,R) if a['segment']==s['segment'])
                assert not np.array_equal(s['x'][k,22:],changed['x'][k,22:])
                model_seg=s
            details.append({'design':prefix,'rule':rule,'segments':len(new),'rows':sum(len(a['x']) for a in new)})
    for rule in M.RULES:
        folder='.b78' if rule=='K10' else '.b75'
        with np.load(M.DATA/folder/'inputs'/f'{rule}.npz',allow_pickle=True) as z:ref=z['anc_g']
        support=np.ones(len(R['gi']),bool)
        if rule=='K10':
            with np.load(M.DATA/'.b78/sets.npz',allow_pickle=True) as z:support=z['common_F']|z['common_L5']
        assert np.array_equal(D.anchors(rule,G)[R['gi']][support],ref[support])
    assert model_seg is not None
    a=create_model('A',1701).eval();c=create_model('C',1701).eval()
    assert a.param_count()==c.param_count()
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
    # Actual segment prefix, including its true reset row.
    x=torch.from_numpy(model_seg['x'][:min(200,len(model_seg['x']))]).unsqueeze(0)
    with torch.no_grad():
        pa=a(x);pc=c(x);changed=x.clone();changed[...,22:]+=1
        assert torch.equal(pc,c(changed))
        assert not torch.equal(a._fuse_input(x),a._fuse_input(changed))
        nohist=x[...,18]==0
        assert nohist.any() and torch.allclose(pa[nohist],pc[nohist],atol=1e-6,rtol=0)
        state=None;step=[]
        for row in x[0]:
            p,state=a.forward_step(row.reshape(1,35),state);step.append(p)
        assert torch.allclose(pa,torch.stack(step,1),atol=1e-5,rtol=0)
        first,_=a.forward_step(x[:,0],None)
        assert torch.allclose(first,pa[:,0],atol=1e-5,rtol=0)
    assert sources==source_inventory(),'Execution source changed during audit'
    result={'all_pass':True,'scope':'CPU real input/untrained forward only; CUDA smoke deferred','checks':details,'parameters':a.param_count(),'frozen_sources':pins,'execution_sources':sources}
    M.C.save_json(M.CACHE/'input_checks.json',result)
    M.C.save_text(M.PAPER/'b88_readiness.md','# B.88 readiness\n\nCPU real-input audits passed. All four designs and eight rules represented; frozen source inventory recorded before audit. Original 22 inputs/targets/masks, cache anchors, independently derived sampled anchors, whole-prefix mutations, positive anchor mutation, matched initial parameters, C anchor invariance, no-history agreement, streaming and reset checks passed. CUDA smoke and scientific readout remain deferred until B.85 completes.\n')
    print(json.dumps({'all_pass':True,'cases':len(details),'parameters':a.param_count()}))

if __name__=='__main__':main()
