"""Independent prediction replay, source verification and paired cluster readout."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import data98 as D
import numpy as np
import model98 as M
import input98

def main():
    z=json.loads((D.CACHE/'fit_complete.json').read_text());assert z['all_checks']
    for p,h in z['sources'].items():assert D.sha(D.ROOT/p)==h
    out={'batch':'B98','scope':z['scope'],'outer_test_evaluated':False,'target':'F','units':{},'sources':z['sources'],'fit_seconds':z['seconds']}
    for number,unit in enumerate(D.UNITS):
        info=json.loads((D.CACHE/(unit+'.json')).read_text());path=D.CACHE/(unit+'.npz')
        assert D.sha(path)==info['prediction_sha256']
        raw_checks=input98.verify(D.ROOT,info,path)
        groups,meta=D.load(unit);assert meta['hashes']==info['metadata']['hashes']
        with np.load(path) as a:pred=a['pred'];y=a['y'];mapping=a['mapping'];clus=a['cluster'];shot=a['shot']
        for k,kind in enumerate(('M','H','L')):
            replay,labels,indices=M.predict(groups['val'],info['models'][kind])
            assert np.array_equal(replay,pred[:,k]) and np.array_equal(labels,y) and np.array_equal(indices,mapping)
        assert np.isfinite(pred).all() and np.isfinite(y).all()
        labels,ci=np.unique(clus,return_inverse=True);n=len(labels)
        assert n>=2,'Insufficient shot clusters for registered interval'
        counts=np.zeros((7,n));sums=np.zeros((7,n,4));metrics={}
        for rule in range(7):
            keep=mapping[:,2]==rule;err=pred[keep]-y[keep,None]
            for c in range(n):
                mask=ci[keep]==c;counts[rule,c]=mask.sum();sums[rule,c]=(err[mask]**2).sum(axis=0)
            mse=np.mean(err**2,axis=0);mae=np.mean(np.abs(err),axis=0)
            assert np.allclose(mse,sums[rule].sum(0)/counts[rule].sum(),rtol=1e-12,atol=1e-6)
            metrics[D.RULES[rule]]={'rows':int(keep.sum()),'mse_eV2':dict(zip(['M','H','L','hold'],mse.tolist())),'mae_eV':dict(zip(['M','H','L','hold'],mae.tolist()))}
        weights=np.random.default_rng(20261198).multinomial(n,np.full(n,1/n),size=20000)
        denom=weights@counts.T;assert np.all(denom>0)
        risks=np.einsum('bc,rcm->brm',weights,sums)/denom[:,:,None]
        for r,rule in enumerate(D.RULES):
            base=np.array(list(metrics[rule]['mse_eV2'].values()))
            for k,name in ((3,'hold'),(1,'H'),(2,'L')):
                assert np.all(risks[:,r,k]>0)
                samples=1-risks[:,r,0]/risks[:,r,k];lo,hi=np.quantile(samples,[.025,.975])
                metrics[rule]['skill_M_over_'+name]={'point':float(1-base[0]/base[k]),'lo':float(lo),'hi':float(hi),'excludes_zero':bool(lo>0 or hi<0)}
        out['units'][unit]={'metrics':metrics,'clusters':n,'shots':int(len(np.unique(shot))),
                             'train_val_cluster_overlap':meta['train_val_cluster_overlap'],
                             'seconds':info['seconds'],'tau':{k:info['models'][k]['tau'] for k in ['M','H']},
                             'parameters':{k:{p:info['models'][k][p] for p in ['tau','beta','rank','condition','solver']} for k in ['M','H','L']},
                             'checks':info['checks'],'prediction_sha256':info['prediction_sha256'],'drops':meta['drops']}
        out['units'][unit]['independent_raw_checks']=raw_checks
        print(json.dumps({'unit':unit,'per_rule_metrics_checked':len(metrics),'clusters':n}),flush=True)
    out['independent_readout_pass']=True;out['bootstrap']={'draws':20000,'seed':20261198,'empty_draws':0,'unit':'original shot cluster','uncertainty':'Pointwise, unadjusted fixed-fit internal validation; no fitting/selection uncertainty; TRAIN/VAL cluster overlap disclosed'}
    (D.ROOT/'docs/paper/b98_physics.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    (D.CACHE/'readout_complete.json').write_text(json.dumps({'all_pass':True,'result_sha256':D.sha(D.ROOT/'docs/paper/b98_physics.json')},indent=2),encoding='utf8')

if __name__=='__main__':main()
