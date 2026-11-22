"""Check B85 bootstrap using existing REAL B87 predictions, no B85 scores.

Independent explicitly repeated-row computation validates initialization draws,
cluster multiplicities and quantiles. No artificial observations are generated.
"""
import sys,json,zlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common85 as M
import readout85 as B


def main():
    _,R=M.load_sources()
    idx=np.sort(np.random.default_rng(8501).choice(len(R['gi']),2048,replace=False))
    predictions={a:[] for a in ('X0','H0')}
    for repeat in (2,3,4):
        for arm in predictions:
            full=np.full(len(R['gi']),np.nan)
            for fold in range(5):
                with np.load(M.DATA/'.b87/fits'/f'V1_r{repeat}_k{fold}__{arm}.npz') as z:
                    full[z['idx']]=z['est']
            assert np.isfinite(full[idx]).all()
            predictions[arm].append(full[idx])
    y=R['tg_F'][idx]
    num=(np.asarray(predictions['X0'])-y)**2
    den=(np.asarray(predictions['H0'])-y)**2
    errors=np.abs(np.asarray(predictions['X0'])-y)
    labels={'cluster':R['clus'][idx],'discharge':R['fidx'][idx]}
    name='lead-real-B87-bootstrap'
    result=B.bootstrap(name,num,den,labels,draws=257)
    med=B.median_bootstrap(name,errors,labels,draws=257)
    point=np.mean([1-n.sum()/d.sum() for n,d in zip(num,den)])
    assert np.isclose(result['point'],point,atol=1e-13)
    for form,lab in labels.items():
        _,inv=np.unique(lab,return_inverse=True);nu=inv.max()+1
        rng=np.random.default_rng([20261185,zlib.crc32((name+'/'+form).encode())])
        picks=rng.integers(0,3,257);skills=np.empty(257);medians=np.empty(257)
        for init in range(3):
            positions=np.flatnonzero(picks==init)
            weights=rng.multinomial(nu,np.full(nu,1/nu),len(positions))
            for draw,w in zip(positions,weights):
                # Explicit repeated rows, independent from per-unit sums/WQ.
                rows=np.repeat(np.arange(len(idx)),w[inv])
                skills[draw]=1-num[init,rows].sum()/den[init,rows].sum()
                medians[draw]=np.median(errors[init,rows])
        assert np.allclose(result[form],np.quantile(skills,[.025,.975]),atol=1e-12)
        assert np.allclose(med[form],np.quantile(medians,[.025,.975]),atol=1e-10)
    # The scientific entry point must reject the incomplete B85 batch before scores.
    try:B.guarded_ready(M.select_units())
    except AssertionError:blocked=True
    else:blocked=False
    assert blocked,'Expected incomplete B85 batch to block readout'
    report={'passes':True,'source':'existing real B87 predictions only; no B85 performance readout',
            'rows':len(idx),'initialization_streams':3,'audit_draws':257,
            'checks':['skill point','initialization-first draws','cluster/discharge multiplicities',
                      'skill interval quantiles','weighted median interval quantiles','incomplete B85 guard']}
    (M.CACHE/'lead_readout_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (M.PAPER/'b85_readout_lead_checks.md').write_text('# B.85 readout algorithm audit\n\nPASS.\n\n'+json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS: B85 bootstrap audited against explicit resampled real B87 rows')


if __name__=='__main__':main()
