"""Independent repeated-real-row bootstrap audit, then optional completed-fit audit."""
import argparse,json,sys,zlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import common88 as M
import readout88 as B


def completed():
    B.ready()
    _,r=M.load_sources();n=len(r['gi']);record=json.loads((M.CACHE/'readout.json').read_text())
    checked=0
    for design,d in record['designs'].items():
        units=[u for u in M.select_units() if u['name'].startswith(design)]
        for key,cell in d['cells'].items():
            rule,target=key.split('/');head=0 if target.startswith('F') else 1
            y=r['tg_F' if head==0 else 'tg_L5']
            if target=='F_full':mask=np.isfinite(y)
            else:
                with np.load(M.DATA/('.b78' if rule=='K10' else '.b75')/'sets.npz') as z:
                    mask=z[('S_' if rule=='S' else 'common_')+target]&np.isfinite(y)
            preds={}
            for var in ('A','C','B85X'):
                p=np.full((3,n),np.nan)
                for i,seed in enumerate(M.INITS):
                    for u in units:
                        folder=M.DATA/'.b85/predictions' if var=='B85X' else M.CACHE/'predictions'
                        v='X' if var=='B85X' else var
                        with np.load(folder/f"{u['name']}__{seed}__{v}__{rule}.npz") as z:
                            idx=z['idx'];assert np.isnan(p[i,idx]).all();p[i,idx]=z['est'][:,head]
                preds[var]=p
            for title,ref in (('A over C','C'),('A over B85X','B85X')):
                keep=mask&np.isfinite(preds['A']).all(0)&np.isfinite(preds[ref]).all(0)
                num=((preds['A'][:,keep]-y[keep])**2).sum(1)
                den=((preds[ref][:,keep]-y[keep])**2).sum(1)
                independent=float(np.mean(1-num/den))
                assert np.isclose(independent,cell['pairs'][title]['point'],rtol=1e-10,atol=1e-12)
                checked+=1
    report={'passes':True,'paired_cell_points':checked,'scope':'independent NPZ identity pooling and squared-error points; fixed-model intervals audited separately'}
    (M.CACHE/'output_lead_checks.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (M.PAPER/'b88_output_lead_checks.md').write_text('# B.88 independent output audit\n\n'+json.dumps(report,indent=2)+'\n',encoding='utf8')


def bootstrap():
    _,r=M.load_sources();idx=np.sort(np.random.default_rng(8801).choice(len(r['gi']),2048,replace=False))
    p={a:[] for a in ('X0','H0')}
    for repeat in (2,3,4):
        for arm in p:
            out=np.full(len(r['gi']),np.nan)
            for fold in range(5):
                with np.load(M.DATA/'.b87/fits'/f'V1_r{repeat}_k{fold}__{arm}.npz') as z:out[z['idx']]=z['est']
            assert np.isfinite(out[idx]).all();p[arm].append(out[idx])
    y=r['tg_F'][idx];num=(np.asarray(p['X0'])-y)**2;den=(np.asarray(p['H0'])-y)**2
    ae=np.abs(np.asarray(p['X0'])-y);labels={'cluster':r['clus'][idx],'discharge':r['fidx'][idx]};name='independent-real-B87-B88'
    got=B.B.bootstrap(name,num,den,labels,draws=257);med=B.B.median_bootstrap(name,ae,labels,draws=257)
    assert np.isclose(got['point'],np.mean(1-num.sum(1)/den.sum(1)),atol=1e-13)
    for form,lab in labels.items():
        _,inv=np.unique(lab,return_inverse=True);nu=inv.max()+1
        rng=np.random.default_rng([20261188,zlib.crc32((name+'/'+form).encode())]);pick=rng.integers(0,3,257)
        skill=np.empty(257);median=np.empty(257)
        for i in range(3):
            pos=np.flatnonzero(pick==i);weights=rng.multinomial(nu,np.full(nu,1/nu),len(pos))
            for draw,w in zip(pos,weights):
                rows=np.repeat(np.arange(len(idx)),w[inv]);skill[draw]=1-num[i,rows].sum()/den[i,rows].sum();median[draw]=np.median(ae[i,rows])
        assert np.allclose(got[form],np.quantile(skill,[.025,.975]),atol=1e-12)
        assert np.allclose(med[form],np.quantile(median,[.025,.975]),atol=1e-10)
    try:B.ready()
    except (AssertionError,FileNotFoundError):pass
    else:raise AssertionError('Incomplete fits must reject readout')
    report={'passes':True,'real_source':'B87 cached predictions only','rows':len(idx),'draws':257,'seed':20261188,'checks':['points','cluster/discharge repeated-row intervals','median intervals','premature score guard']}
    (M.CACHE/'lead_readout_checks.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (M.PAPER/'b88_readout_lead_checks.md').write_text('# B.88 independent bootstrap audit\n\n'+json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completed',action='store_true');args=parser.parse_args()
    completed() if args.completed else bootstrap()
    print('PASS: B88 independent readout audit')
