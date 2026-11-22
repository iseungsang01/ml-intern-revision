"""Fit all B87 units, check them, then compute the registered readout. Resumable."""
import argparse
import json
import common87 as M
import numpy as np

C,T = M.C,M.T

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fit-only',action='store_true')
    args=ap.parse_args()
    R,G=C.rows(),C.grid()
    units,_,_=M.DS.units(R)
    H=T.history_matrix(R)
    slots,src=M.future(G,R['gi'])
    C.save_npz(M.CACHE/'future.npz',slots=slots,source=src)
    # These checks use no errors or skills.
    assert all(not np.intersect1d(u['fit'],u['test']).size for u in units)
    for j in range(5):
        q=np.flatnonzero(src[:,j]>=0)
        p=src[q,j]
        assert np.all(G['fidx'][p]==R['fidx'][q])
        assert np.all(G['seg'][p]==G['seg'][R['gi'][q]])
        assert np.all(np.abs(G['time'][p]-G['time'][R['gi'][q]]-.01*(j+1))<.003)
        assert np.all(p>R['gi'][q])
    sample=M.rng('mutations').choice(len(R['gi']),100,replace=False)
    for q in sample:
        gc={**G,'ti':G['ti'].copy()}
        gc['ti'][R['gi'][q]]+=1000
        changed,_=M.future(gc,R['gi'][q:q+1])
        assert np.array_equal(changed,slots[q:q+1],equal_nan=True)
        for j in (0,1,2,5):
            gc['ti']=G['ti'].copy()
            beyond=(G['time']>G['time'][R['gi'][q]]+.01*j+.003)&(G['fidx']==R['fidx'][q])
            gc['ti'][beyond]+=1000
            changed,_=M.future(gc,R['gi'][q:q+1])
            assert np.array_equal(changed[:,:j],slots[q:q+1,:j],equal_nan=True)
    C.save_json(M.CACHE/'input_checks.json',{'passes':True,'mutation_rows':len(sample)})
    fe=T.Feats()
    for u in units:
        todo=[a for a in M.ARMS if not (M.CACHE/'fits'/f"{u['name']}__{a}.npz").exists()]
        if not todo:
            continue
        wf=fe.window(u['fit'])
        fast,fastl=fe.at(wf,R['gi'])
        ok=T.complete_mask(H,fast,fastl,R['gp'])
        fit=np.isin(R['fidx'],u['fit'])&ok&np.isfinite(R['tg_F'])
        idx=np.flatnonzero(np.isin(R['fidx'],u['test']))
        for arm in todo:
            X=M.matrix(H,fast,slots,arm)
            if arm[0]=='X':
                assert np.array_equal(X[:,H.shape[1]:H.shape[1]+8],fast,equal_nan=True)
            active=np.isfinite(X[fit]).any(axis=0)
            model=T.gbr().fit(X[fit][:,active],(R['tg_F']-R['gp'])[fit])
            est=R['gp'][idx].copy()
            est[ok[idx]]+=model.predict(X[idx[ok[idx]]][:,active])
            assert np.isfinite(est).all()
            C.save_npz(M.CACHE/'fits'/f"{u['name']}__{arm}.npz",idx=idx,est=est,complete=ok[idx],fit_rows=np.int64(fit.sum()),active=active)
            C.log(f"B87 {u['name']} {arm} saved")
    for u in units:
        idx=np.flatnonzero(np.isin(R['fidx'],u['test']))
        for arm in M.ARMS:
            z=C.load_npz(M.CACHE/'fits'/f"{u['name']}__{arm}.npz")
            assert np.array_equal(idx,z['idx']) and np.isfinite(z['est']).all()
    C.save_json(M.CACHE/'checks.json',{'passes':True,'units':len(units),'arms':list(M.ARMS),'sampled_mutations':100})
    C.save_text(C.PAPER/'b87_checks.md','# B.87 pre-readout checks\n\nPASS: disjoint fit/test files; same-file/segment future slots; target and beyond-horizon mutations; unchanged present-fast windows; finite predictions; identical row indices in every arm. Checked before computing errors.\n')
    if not args.fit_only:
        import readout87
        readout87.main()

if __name__=='__main__':
    main()
