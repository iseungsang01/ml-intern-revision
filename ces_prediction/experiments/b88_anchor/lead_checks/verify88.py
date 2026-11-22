"""Independent real-input B88 audit. No fitted training or scientific errors."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b85_unified'))
sys.path.insert(0,str(HERE))
import common85 as M
import data85 as D
import common88 as N
import data88 as A


def main():
    pins=json.loads((HERE/'frozen_b85_sources.json').read_text())
    for path,expected in pins.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    original_cache=M.CACHE
    G,R=M.load_sources();u=M.select_units()[0];manifest=D.unit_manifest(u)
    stats=D.fit_stats(manifest['train'],G,R)
    chosen=None
    for fid in manifest['test']:
        ids=np.flatnonzero(G['fidx']==fid)
        gaps=ids[1:][(np.diff(G['time'][ids])>.015)&(np.diff(G['time'][ids])<.045)&(np.diff(G['seg'][ids])==0)]
        for first in gaps:
            seg=ids[G['seg'][ids]==G['seg'][first]]
            if len(seg)>30:
                g=int(seg[min(len(seg)-1,np.searchsorted(seg,first)+15)])
                chosen=(int(fid),seg,g);break
        if chosen:break
    assert chosen
    fid,ids,g=chosen;fr=dict(np.load(M.DATA/'.b78/frames.npz',allow_pickle=False))['fr']
    checks=0;last=None;absent=None
    for rule in M.RULES:
        base=D.build_segments([fid],stats,rule,G,R);new=A.build_segments([fid],stats,rule,G,R)
        assert len(base)==len(new)
        anchors=D.anchors(rule,G)
        for b,n in zip(base,new):
            assert n['x'].shape==(len(b['x']),35)
            assert np.array_equal(n['x'][:,:22],b['x'])
            assert np.array_equal(n['gi'],b['gi']) and np.array_equal(n['y'],b['y'])
            assert n['mask'].dtype==bool and np.array_equal(n['mask'],b['mask'])
            ai=anchors[n['gi']];valid=ai>=0;expected=np.zeros((len(ai),13),np.float32)
            assert np.all(ai[valid]<n['gi'][valid])
            assert np.all(G['fidx'][ai[valid]]==fid)
            assert np.all(G['seg'][ai[valid]]==G['seg'][n['gi'][valid]])
            expected[valid]=((G['fast'][ai[valid]].astype(np.float32)-stats['fast_mu'])/stats['fast_sd']).astype(np.float32)
            assert np.array_equal(n['x'][:,22:35],expected),rule
            if rule=='S' and len(n['x']) and n['x'][0,18]==0:
                upto=np.flatnonzero(n['x'][:,18]>0)
                count=min(30,int(upto[0]) if len(upto) else len(n['x']))
                if count:absent=n['x'][:count].copy()
        n=next(s for s in new if g in s['gi']);loc=np.flatnonzero(n['gi']<=g)
        # Direct timestamp eligibility, independent from both builders.
        if rule.startswith('D'):
            eligible=(ids<g)&(G['time'][ids]<=G['time'][g]-int(rule[1:])/1000+.003)
        elif rule=='S':
            gap=ids[1:][(np.diff(G['time'][ids])>.015)&(np.diff(G['time'][ids])<.045)&(ids[1:]<=g)]
            cutoff=int(gap[-1])-1 if len(gap) else -1;eligible=(ids<g)&(ids<=cutoff)
        else:eligible=(ids<g)&(fr[ids]>=0)&(fr[ids]%10==0)
        good=ids[eligible&np.isfinite(G['ti'][ids])];anchor=int(good[-1]) if len(good) else -1
        assert anchors[g]==anchor
        for kind in ('remove','raise','future_fast'):
            changed={**G}
            if kind=='future_fast':
                changed['fast']=G['fast'].copy();changed['fast'][ids[ids>g]]+=1000
            else:
                changed['ti']=G['ti'].copy()
                if kind=='remove':changed['ti'][ids[~eligible]]=np.nan
                else:changed['ti'][ids[~eligible]]+=1000
            alt=next(s for s in A.build_segments([fid],stats,rule,changed,R) if g in s['gi'])
            assert np.array_equal(n['x'][loc],alt['x'][loc]),(rule,kind)
            checks+=1
        if anchor>=0:
            changed={**G,'fast':G['fast'].copy()}
            changed['fast'][anchor,0]+=max(abs(float(G['fast'][anchor,0]))*.1,float(stats['fast_sd'][0]),1.)
            alt=next(s for s in A.build_segments([fid],stats,rule,changed,R) if g in s['gi'])
            at=int(np.flatnonzero(n['gi']==g)[0])
            assert not np.array_equal(n['x'][at,22:],alt['x'][at,22:]),rule
        last=n
    import torch
    from model88 import create_model
    ma=create_model('A',1701).cpu().eval();mc=create_model('C',1701).cpu().eval()
    assert ma.lstm_ti.input_size==39 and ma.lstm_ti.hidden_size==160 and ma.lstm_ti.num_layers==2
    assert sum(p.numel() for p in ma.parameters())==sum(p.numel() for p in mc.parameters())<1000000
    assert all(torch.equal(v,mc.state_dict()[k]) for k,v in ma.state_dict().items())
    valid=np.flatnonzero(last['x'][:,18]>0)
    assert len(valid),'Positive model intervention needs an actual visible anchor'
    end=min(len(last['x']),int(valid[0])+16)
    x=torch.from_numpy(last['x'][:end]).unsqueeze(0);mut=x.clone();mut[:,:,22:]+=1
    with torch.no_grad():
        assert torch.equal(mc(x),mc(mut))
        assert not torch.equal(ma(x),ma(mut))
        assert absent is not None
        blank=torch.from_numpy(absent).unsqueeze(0)
        assert torch.equal(ma(blank),mc(blank))
        for model in (ma,mc):
            full=model(x);state=None;steps=[]
            for t in range(x.shape[1]):
                out,state=model.forward_step(x[:,t],state);steps.append(out)
            assert torch.allclose(full,torch.stack(steps,1),rtol=1e-4,atol=1e-5)
    import train85 as T
    old_batches=T.make_batches
    import adapter88
    adapted=adapter88.activate()
    assert adapted.CACHE==N.CACHE and M.CACHE==original_cache==ROOT/'data/.b85'
    dynamic=adapted.make_batches([last],2,shuffle_seed=7)[0]
    original=old_batches([{**last,'x':last['x'][:,:22]}],2,shuffle_seed=7)[0]
    assert torch.equal(dynamic['x'][:,:,:22],original['x'])
    assert torch.equal(dynamic['y'],original['y']) and torch.equal(dynamic['mask'],original['mask'])
    for path,expected in pins.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    report={'passes':True,'pinned_sources':len(pins),'rules':list(M.RULES),'prefix_mutations':checks,
            'scope':'real input anchor alignment, frozen22/masks/labels, timestamp mutation, paired architecture, C invariance, streaming, isolated adapter; no fitted performance claim'}
    N.CACHE.mkdir(parents=True,exist_ok=True)
    (N.CACHE/'lead_checks.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (N.PAPER/'b88_lead_checks.md').write_text('# B.88 independent input audit\n\nPASS.\n\n'+json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('PASS: B88 independent real-input audit')


if __name__=='__main__':main()
