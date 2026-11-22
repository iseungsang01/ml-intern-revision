"""Independent repeated-row audit of B90 one- and three-seed bootstrap."""
import sys,json,zlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common90 as M
import readout90 as R90


def main():
    M.assert_registered();M.assert_frozen();_,r=M.load_sources()
    idx=np.sort(np.random.default_rng(9001).choice(len(r['gi']),2048,replace=False));p={a:[] for a in ('X0','H0')}
    for repeat in (2,3,4):
        for arm in p:
            full=np.full(len(r['gi']),np.nan)
            for fold in range(5):
                with np.load(M.DATA/'.b87/fits'/f'V1_r{repeat}_k{fold}__{arm}.npz') as z:full[z['idx']]=z['est']
            assert np.isfinite(full[idx]).all();p[arm].append(full[idx])
    truth=r['tg_F'][idx];num=(np.asarray(p['X0'])-truth)**2;den=(np.asarray(p['H0'])-truth)**2;ae=np.abs(np.asarray(p['X0'])-truth)
    labels={'cluster':r['clus'][idx],'discharge':r['fidx'][idx]};b=R90.bootstrap_module()
    for count in (1,3):
        name='parent-real-B87-B90-'+str(count);got=b.bootstrap(name,num[:count],den[:count],labels,draws=257);med=b.median_bootstrap(name,ae[:count],labels,draws=257)
        assert np.isclose(got['point'],np.mean(1-num[:count].sum(1)/den[:count].sum(1)),atol=1e-13)
        for form,lab in labels.items():
            _,inv=np.unique(lab,return_inverse=True);nu=inv.max()+1;rng=np.random.default_rng([20261190,zlib.crc32((name+'/'+form).encode())]);picks=rng.integers(0,count,257)
            skill=np.empty(257);median=np.empty(257)
            for i in range(count):
                positions=np.flatnonzero(picks==i);weights=rng.multinomial(nu,np.full(nu,1/nu),len(positions))
                for draw,w in zip(positions,weights):
                    rows=np.repeat(np.arange(len(idx)),w[inv]);skill[draw]=1-num[i,rows].sum()/den[i,rows].sum();median[draw]=np.median(ae[i,rows])
            assert np.allclose(got[form],np.quantile(skill,[.025,.975]),atol=1e-12)
            assert np.allclose(med[form],np.quantile(median,[.025,.975]),atol=1e-10)
    try:R90.main()
    except (AssertionError,FileNotFoundError):pass
    else:raise AssertionError('Readout must reject incomplete fits before scores')
    report={'passes':True,'source':'Existing real B87 predictions only; no B90 scientific score','rows':len(idx),'draws':257,'initialization_counts':[1,3],'seed':20261190,'premature_readout_rejected':True}
    M.save(M.CACHE/'lead_bootstrap_checks.json',report);M.C.save_text(M.PAPER/'b90_bootstrap_lead_checks.md','# B.90 independent bootstrap audit\n\n'+json.dumps(report,indent=2)+'\n')
    print('PASS: B90 one/three-seed bootstrap against repeated real rows')


if __name__=='__main__':main()
