"""Independent one/three-seed bootstrap quantiles using existing real B87 rows."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import sys,zlib,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common92 as M
import readout92 as R92


def main():
    M.guard();_,r=M.load_sources();idx=np.sort(np.random.default_rng(9201).choice(len(r['gi']),2048,replace=False));p={a:[] for a in ('X0','H0')}
    for repeat in (2,3,4):
        for arm in p:
            out=np.full(len(r['gi']),np.nan)
            for fold in range(5):
                with np.load(M.DATA/'.b87/fits'/f'V1_r{repeat}_k{fold}__{arm}.npz') as z:out[z['idx']]=z['est']
            assert np.isfinite(out[idx]).all();p[arm].append(out[idx])
    y=r['tg_F'][idx];num=(np.asarray(p['X0'])-y)**2;den=(np.asarray(p['H0'])-y)**2;ae=np.abs(np.asarray(p['X0'])-y)
    labels={'cluster':r['clus'][idx],'discharge':r['fidx'][idx]};b=R92.B
    for count in (1,3):
        name='independent-B92-real-B87-'+str(count);got=b.bootstrap(name,num[:count],den[:count],labels,draws=257);med=b.median_bootstrap(name,ae[:count],labels,draws=257)
        assert np.isclose(got['point'],np.mean(1-num[:count].sum(1)/den[:count].sum(1)),atol=1e-13)
        for form,lab in labels.items():
            _,inv=np.unique(lab,return_inverse=True);nu=inv.max()+1;rng=np.random.default_rng([20261192,zlib.crc32((name+'/'+form).encode())]);pick=rng.integers(0,count,257);skill=np.empty(257);median=np.empty(257)
            for i in range(count):
                pos=np.flatnonzero(pick==i);weights=rng.multinomial(nu,np.full(nu,1/nu),len(pos))
                for draw,w in zip(pos,weights):
                    rows=np.repeat(np.arange(len(idx)),w[inv]);skill[draw]=1-num[i,rows].sum()/den[i,rows].sum();median[draw]=np.median(ae[i,rows])
            assert np.allclose(got[form],np.quantile(skill,[.025,.975]),atol=1e-12)
            assert np.allclose(med[form],np.quantile(median,[.025,.975]),atol=1e-10)
    report={'passes':True,'real_source':'B87 only, no B92 accuracy scores','rows':len(idx),'draws':257,'seeds':[1,3],'bootstrap_seed':20261192}
    M.save(M.CACHE/'lead_bootstrap_checks.json',report);M.C.save_text(M.PAPER/'b92_bootstrap_lead_checks.md','# B.92 independent bootstrap audit\n\n'+json.dumps(report,indent=2)+'\n')
    print('PASS: B92 repeated-real-row bootstrap, one and three seeds')


if __name__=='__main__':main()
