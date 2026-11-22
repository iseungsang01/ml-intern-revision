"""B87 registered metrics, after every registered pre-score check passes."""
import json
import common87 as M
import numpy as np
C=M.C
PAIRS=tuple(dict.fromkeys([('X0','H0')]+[('X0',f'H{j}') for j in (1,2,5)]+[(f'X{j}','X0') for j in (1,2,5)]+[(f'X{j}',f'H{j}') for j in (0,1,2,5)]))

def cell(name, values, invs, metric):
    """Each value entry is nrows x 2 numerator/denominator additive contributions."""
    def stat(v):
        ratio=v[...,0]/v[...,1]
        return np.sqrt(ratio) if metric=='RMSE' else (1-ratio if metric=='skill' else ratio)
    result={'point':float(np.mean([stat(v.sum(axis=0)) for v in values]))}
    for form,inv in invs.items():
        nu=int(inv.max())+1
        totals=[np.column_stack([np.bincount(inv,v[:,k],nu) for k in range(2)]) for v in values]
        rng=M.rng(name+'/'+form)
        picks=rng.integers(0,len(values),20000)
        draws=np.empty(20000)
        for s in range(len(values)):
            pos=np.flatnonzero(picks==s)
            for start in range(0,len(pos),500):
                q=pos[start:start+500]
                w=rng.multinomial(nu,np.full(nu,1/nu),size=len(q))
                draws[q]=stat(w@totals[s])
        assert np.isfinite(draws).all()
        result[form]=np.quantile(draws,[.025,.975]).tolist()
    result['excludes_zero']=bool(result['cluster'][0]>0 or result['cluster'][1]<0)
    return result

def main():
    assert json.loads((M.CACHE/'checks.json').read_text())['passes']
    R=C.rows()
    units,_,_=M.DS.units(R)
    validate_selectors(R,units)
    out={'registration':'PREREGISTRATION_B87.md','bootstrap_draws':20000,'seed':20261187,
         'interpretation':'Exploratory used-data timestamp-horizon comparison; not a physical-temperature or deployment validation.','designs':{}}
    md=['# B.87 matched regression at fixed reporting delays','',out['interpretation'],'',
        'Point estimates and 95% shot-cluster intervals below; JSON also contains discharge intervals. Positive skill means reduced MSE on identical rows. Delays refer to timestamps; exposure and processing add physical latency.', '',
        'Intervals condition on the fitted models and do not include refitting or feature-choice uncertainty. This is an exploratory analysis of previously used data, with no multiplicity adjustment or new confirmation set.','']
    for design in ('V1','V2','V3','V4a','V4b'):
        subs=[]
        for ul in M.groups(units,design):
            idx=np.sort(np.concatenate([C.load_npz(M.CACHE/'fits'/f"{u['name']}__H0.npz")['idx'] for u in ul]))
            assert len(np.unique(idx))==len(idx)
            preds={}
            for arm in M.ARMS:
                full=np.full(len(R['y']),np.nan)
                for u in ul:
                    z=C.load_npz(M.CACHE/'fits'/f"{u['name']}__{arm}.npz")
                    full[z['idx']]=z['est']
                preds[arm]=full[idx]
                assert np.isfinite(preds[arm]).all()
            subs.append((idx,preds))
        idx=subs[0][0]
        assert all(np.array_equal(idx,s[0]) for s in subs)
        y=R['tg_F'][idx]
        assert np.isfinite(y).all()
        invs={form:np.unique(R[key][idx],return_inverse=True)[1] for form,key in [('cluster','clus'),('discharge','fidx')]}
        d={'rows':len(idx),'shots':len(np.unique(R['fidx'][idx])),'clusters':len(np.unique(R['clus'][idx])),'metrics':{},'pairs':{}}
        md += [f'## {design}', '', f"Rows {d['rows']}; shots {d['shots']}; clusters {d['clusters']}.",'','| Arm | RMSE (eV) | MAE (eV) |','|---|---|---|']
        errors=[{a:p[a]-y for a in M.ARMS} for _,p in subs]
        for arm in M.ARMS:
            metrics={}
            for metric in ('RMSE','MAE'):
                vv=[np.column_stack([e[arm]**2 if metric=='RMSE' else np.abs(e[arm]),np.ones(len(y))]) for e in errors]
                metrics[metric]=cell(f'{design}/{arm}/{metric}',vv,invs,metric)
            d['metrics'][arm]=metrics
            md.append('| '+arm+' | '+' | '.join(fmt(metrics[m]) for m in ('RMSE','MAE'))+' |')
        md += ['','| Pair | MSE skill | Excludes zero |','|---|---|---|']
        for a,b in PAIRS:
            name=f'{a} over {b}'
            v=cell(f'{design}/{name}',[np.column_stack([e[a]**2,e[b]**2]) for e in errors],invs,'skill')
            d['pairs'][name]=v
            md.append(f"| {name} | {fmt(v,4)} | {v['excludes_zero']} |")
        d['statements']={}
        for letter,name,expected in [('A','X0 over H1','uncertain'),('B','X0 over H5','false'),('C','X5 over X0','true'),('D','X5 over H5','uncertain')]:
            v=d['pairs'][name]
            d['statements'][letter]={**v,'pair':name,'holds':v['cluster'][0]>0,'expectation':expected}
        out['designs'][design]=d
        md += ['', '| Statement | Pair | MSE skill (95% cluster interval) | Excludes zero | Verdict (lower bound > 0) | Expectation |',
               '|---|---|---|---|---|---|']
        for k,v in d['statements'].items():
            verdict='holds' if v['holds'] else 'does not hold'
            md.append(f"| {k} | {v['pair']} | {fmt(v,4)} | {v['excludes_zero']} | {verdict} | {v['expectation']} |")
        md.append('')
        C.log(f'B87 readout {design} complete')
    C.save_json(C.PAPER/'b87_latency.json',out)
    C.save_text(C.PAPER/'b87_latency.md','\n'.join(md)+'\n')

def fmt(v,digits=2):
    return f"{v['point']:.{digits}f} [{v['cluster'][0]:.{digits}f}, {v['cluster'][1]:.{digits}f}]"

def validate_selectors(R,units):
    """No scores: recompute selectors and mutate only the held-out feature rows."""
    H=M.T.history_matrix(R)
    slots=C.load_npz(M.CACHE/'future.npz')['slots']
    fe=M.T.Feats()
    for u in units:
        fast,fastl=fe.at(fe.window(u['fit']),R['gi'])
        ok=M.T.complete_mask(H,fast,fastl,R['gp'])
        fit=np.isin(R['fidx'],u['fit'])&ok&np.isfinite(R['tg_F'])
        test=np.isin(R['fidx'],u['test'])
        for arm in M.ARMS:
            X=M.matrix(H,fast,slots,arm)
            active=np.isfinite(X[fit]).any(axis=0)
            stored=C.load_npz(M.CACHE/'fits'/f"{u['name']}__{arm}.npz")['active']
            assert np.array_equal(active,stored)
            assert np.isnan(X[fit][:,~active]).all()
            X[test]=1000.0
            assert np.array_equal(active,np.isfinite(X[fit]).any(axis=0))
    C.save_json(M.CACHE/'selector_checks.json',{'passes':True,'units':len(units),'test_rows_mutated':True})
    p=C.PAPER/'b87_checks.md'
    text=p.read_text(encoding='utf-8')
    note='\nPASS: fit-only all-missing column selectors independently reconstructed for all unit/arm caches; replacing every test feature row by 1000 leaves each selector unchanged. This compatibility check precedes the numerical readout.\n'
    if note not in text:
        C.save_text(p,text+note)

if __name__=='__main__':
    main()
