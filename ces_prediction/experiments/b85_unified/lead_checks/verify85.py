"""Independent real-data input audit, no training or scientific error readout.

Run from repo root with py. This code does not reuse B85 anchor logic.
"""
import sys
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common85 as M
import data85 as D


def main():
    G,R = M.load_sources()
    units = M.select_units()
    assert len(units) == 15 and len({u['name'] for u in units}) == 15
    manifest = D.unit_manifest(units[0])
    for u in units:
        m = D.unit_manifest(u)
        tr,va,te = (set(m[k]) for k in ('train','val','test'))
        assert tr and va and te and not (tr&va or tr&te or va&te)
        assert tr|va == set(u['fit']) and te == set(u['test'])
        assert len(va) == max(1,int(np.ceil(.15*len(u['fit']))))
        if u['name'].startswith(('V1','V2')):
            import zlib
            ids = np.sort(u['fit'])
            p = np.random.default_rng([20261185,zlib.crc32(u['name'].encode())]).permutation(len(ids))
            assert va == set(ids[p[:len(va)]])
        else:
            latest = sorted(u['fit'], key=lambda fid:G['shot'][fid])[-len(va):]
            assert va == set(latest)
    stats = D.fit_stats(manifest['train'],G,R)
    fit = np.isin(G['fidx'],manifest['train'])
    assert set(stats['train_files']) == set(manifest['train'])
    assert np.isclose(stats['ti_mu'],np.nanmean(G['ti'][fit]),rtol=2e-6)
    assert np.isclose(stats['ti_sd'],np.nanstd(G['ti'][fit]),rtol=2e-6)
    assert np.allclose(stats['fast_mu'],np.mean(G['fast'][fit].astype(float),axis=0),rtol=2e-4,atol=1e-5)
    assert np.allclose(stats['fast_sd'],np.std(G['fast'][fit].astype(float),axis=0).clip(1e-6),rtol=2e-4,atol=1e-5)
    refs={}
    for rule in M.RULES:
        batch = '.b78' if rule=='K10' else '.b75'
        ref=dict(np.load(M.DATA/batch/'inputs'/f'{rule}.npz',allow_pickle=False))
        a=D.anchors(rule,G)
        support=np.ones(len(R['gi']),dtype=bool)
        if rule=='K10':
            sets=dict(np.load(M.DATA/'.b78/sets.npz',allow_pickle=False))
            support=sets['common_F']|sets['common_L5']
        assert np.array_equal(a[R['gi'][support]],ref['anc_g'][support]),rule
        refs[rule] = int(support.sum())
    fr=dict(np.load(M.DATA/'.b78/frames.npz',allow_pickle=False))['fr']
    # One nontrivial real segment with a known gap and sufficient context.
    chosen=None
    for fid in manifest['test']:
        rows=np.flatnonzero(G['fidx']==fid)
        gaps=rows[1:][(np.diff(G['time'][rows])>.015)&(np.diff(G['time'][rows])<.045)&(np.diff(G['seg'][rows])==0)]
        for g in gaps:
            segrows=rows[G['seg'][rows]==G['seg'][g]]
            if len(segrows)>30:
                chosen=(fid,segrows,int(segrows[min(len(segrows)-1,np.searchsorted(segrows,g)+15)]))
                break
        if chosen:break
    assert chosen,'No suitable real segment for prefix audit'
    fid,idx,g=chosen
    prefix = idx[idx<=g]
    row_of={int(gi):j for j,gi in enumerate(R['gi'])}
    checks=0
    for rule in M.RULES:
        original=D.build_segments([fid],stats,rule,G=G,R=R)
        b=next(s for s in original if g in s['gi'])
        loc=np.flatnonzero(b['gi']<=g)
        assert np.array_equal(b['gi'][loc],prefix)
        for k,gi in enumerate(b['gi']):
            row=row_of.get(int(gi))
            for h,name in enumerate(('F','L5')):
                value=np.nan if row is None else R['tg_'+name][row]
                assert bool(b['mask'][k,h]) == bool(np.isfinite(value))
                if np.isfinite(value):
                    assert np.isclose(b['y'][k,h],(value-stats['ti_mu'])/stats['ti_sd'],rtol=1e-5,atol=1e-6)
        # Direct visibility definition, independent from D.anchors.
        if rule.startswith('D'):
            eligible=(G['time'][idx]<=G['time'][g]-int(rule[1:])/1000+.003)&(idx<g)
        elif rule=='S':
            pastgap=idx[1:][(np.diff(G['time'][idx])>.015)&(np.diff(G['time'][idx])<.045)&(idx[1:]<=g)]
            cutoff=int(pastgap[-1])-1 if len(pastgap) else -1
            eligible=(idx<=cutoff)&(idx<g)
        else:
            eligible=(fr[idx]>=0)&(fr[idx]%10==0)&(idx<g)
        valid=idx[eligible & np.isfinite(G['ti'][idx])]
        expected=valid[-1] if len(valid) else -1
        assert D.anchors(rule,G)[g] == expected
        hidden=idx[~eligible]
        for operation in ('remove','raise'):
            changed={**G,'ti':G['ti'].copy()}
            if operation=='remove':changed['ti'][hidden]=np.nan
            else:changed['ti'][hidden]+=1000
            after=D.build_segments([fid],stats,rule,G=changed,R=R)
            z=next(s for s in after if g in s['gi'])
            assert np.array_equal(b['x'][loc],z['x'][loc]),(rule,operation)
            checks+=1
        changed={**G,'fast':G['fast'].copy()}
        changed['fast'][idx[idx>g]]+=1000
        after=D.build_segments([fid],stats,rule,G=changed,R=R)
        z=next(s for s in after if g in s['gi'])
        assert np.array_equal(b['x'][loc],z['x'][loc]),rule
        assert np.all(b['x'][:,13:15]==0) and np.all(b['x'][:,19:]==0)
        steps=np.r_[0,np.diff(G['time'][b['gi']])]
        assert np.allclose(b['x'][:,15],np.log1p(steps),atol=1e-7)
        checks+=1
    import torch
    from model85 import create_model
    model=create_model('X',1701).cpu().eval()
    twin=create_model('H',1701).cpu().eval()
    assert all(torch.equal(value,twin.state_dict()[key]) for key,value in model.state_dict().items())
    x=torch.from_numpy(b['x'][:30]).unsqueeze(0)
    with torch.no_grad():
        full=model(x)
        state=None
        steps=[]
        for t in range(x.shape[1]):
            out,state=model.forward_step(x[:,t],state)
            steps.append(out)
        assert torch.allclose(full,torch.stack(steps,dim=1),rtol=1e-4,atol=1e-5)
        # A separate evaluation has no retained state from a previous sequence.
        assert torch.equal(model(x),full)
        changed_fast=x.clone()
        changed_fast[:,:,:15]+=1000
        assert torch.equal(twin(x),twin(changed_fast))
    report={'passes':True,'units':len(units),'anchor_rows_per_rule':refs,'prefix_mutations':checks,
            'scope':'Real-data anchors, full prefix visibility, future fast inputs, split seeds and training-only statistics. No trained performance claim.'}
    (M.CACHE/'lead_input_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (M.PAPER/'b85_lead_checks.md').write_text('# B.85 independent input audit\n\nPASS.\n\n'+json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS: independent B85 input audit')


if __name__=='__main__':main()
