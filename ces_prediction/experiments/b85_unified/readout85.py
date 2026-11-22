"""Claude-origin B85 readout; local corrections to pooling/bootstrap/data contracts."""
import hashlib,json,platform,sys,time,zlib
from pathlib import Path
import numpy as np
import common85 as M


def rng_for(name):
    return np.random.default_rng([20261185,zlib.crc32(name.encode())])


def bootstrap(name,numerators,denominators,row_units,kind='skill',draws=20000):
    """Arrays (init,row); sample paired init first, then unit multiplicities."""
    num,den=np.asarray(numerators),np.asarray(denominators)
    assert num.shape==den.shape and num.ndim==2 and num.shape[1]
    def stat(a,b):
        with np.errstate(divide='ignore',invalid='ignore'): v=a/b
        return 1-v if kind=='skill' else np.sqrt(v) if kind=='RMSE' else v
    pts=stat(num.sum(1),den.sum(1))
    out={'point':float(pts.mean()),'per_init':pts.tolist()}
    for form,labels in row_units.items():
        _,inv=np.unique(labels,return_inverse=True);nu=inv.max()+1
        totals=[np.column_stack([np.bincount(inv,num[i],nu),np.bincount(inv,den[i],nu)]) for i in range(len(num))]
        rng=rng_for(name+'/'+form);picks=rng.integers(0,len(num),draws);vals=np.empty(draws)
        for i in range(len(num)):
            positions=np.flatnonzero(picks==i)
            for start in range(0,len(positions),500):
                pos=positions[start:start+500];w=rng.multinomial(nu,np.full(nu,1/nu),len(pos));s=w@totals[i]
                vals[pos]=stat(s[:,0],s[:,1])
        finite=vals[np.isfinite(vals)];assert len(finite)
        out[form]=np.quantile(finite,[.025,.975]).tolist()
        if len(finite)!=draws:out[form+'_undefined_draws']=int(draws-len(finite))
    out['excludes_zero']=bool(out['cluster'][0]>0 or out['cluster'][1]<0)
    return out


def median_bootstrap(name,errors,row_units,draws=20000):
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'b82_localize'))
    from boot82 import WQ
    pts=np.median(errors,axis=1);out={'point':float(pts.mean()),'per_init':pts.tolist()}
    for form,labels in row_units.items():
        _,inv=np.unique(labels,return_inverse=True);nu=inv.max()+1
        rng=rng_for(name+'/'+form);picks=rng.integers(0,len(errors),draws);vals=np.empty(draws)
        for i in range(len(errors)):
            positions=np.flatnonzero(picks==i);q=WQ(errors[i],inv,nu)
            for start in range(0,len(positions),500):
                pos=positions[start:start+500];w=rng.multinomial(nu,np.full(nu,1/nu),len(pos))
                vals[pos]=q.draws(w,(.5,))[:,0]
        out[form]=np.quantile(vals,[.025,.975]).tolist()
    return out


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def baseline_path(unit,learner,rule,target):
    if learner=='TXfresh':return M.DATA/'.b73/trees'/f'{unit}__TX__F.npz'
    if learner=='TXall':return M.DATA/'.b84/fits'/f'{unit}__TXall__{rule}.npz'
    return M.DATA/('.b78' if rule=='K10' else '.b75')/'fits'/f'{unit}__{learner}__{rule}__{target}.npz'


def guarded_ready(units):
    from train85 import verify_complete,get_run_dir
    for u in units:
        for init in M.INITS:
            for var in M.VARIANTS:assert verify_complete(get_run_dir(u['name'],init,var)),'Registered fit missing'
    for name in ('input_checks.json','prediction_checks.json'):
        assert json.loads((M.CACHE/name).read_text(encoding='utf8')).get('all_pass'),name
    for u in units:
        expected=None
        for init in M.INITS:
            for var in M.VARIANTS:
                config=json.loads((get_run_dir(u['name'],init,var)/'config.json').read_text())
                for rule in M.RULES:
                    p=M.CACHE/'predictions'/f"{u['name']}__{init}__{var}__{rule}.npz"
                    z=npz(p);assert z['est'].shape==(len(z['idx']),2) and np.isfinite(z['est']).all()
                    assert str(z['config_hash'])==config['config_hash'],'Stale prediction config'
                    if expected is None:expected=z['idx']
                    else:assert np.array_equal(expected,z['idx']),'Prediction row support mismatch'


def baseline_inventory(units):
    inventory={}
    for u in units:
        for rule in M.RULES:
            for target in ('F','L5'):
                learners=['T0','TX']+(['TXall'] if target=='L5' else [])+(['TXfresh'] if (rule,target)==('D10','F') else [])
                for learner in learners:
                    p=baseline_path(u['name'],learner,rule,target);key=f"{u['name']}|{rule}|{target}|{learner}"
                    assert learner=='TXall' or p.is_file(),'Required baseline missing: '+str(p)
                    e={'path':p.relative_to(M.ROOT).as_posix(),'available':p.is_file()}
                    if p.is_file():e.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(npz(p)['idx']))
                    else:e['note']='optional baseline unavailable; no imputation'
                    inventory[key]=e
    M.C.save_json(M.CACHE/'baseline_inventory.json',inventory)
    return inventory


def pooled_predictions(units,rule,nrows):
    out={v:np.full((len(M.INITS),nrows,2),np.nan) for v in M.VARIANTS};support=None
    for i,init in enumerate(M.INITS):
        for var in M.VARIANTS:
            seen=np.zeros(nrows,bool)
            for u in units:
                z=npz(M.CACHE/'predictions'/f"{u['name']}__{init}__{var}__{rule}.npz");idx=z['idx']
                assert not seen[idx].any() and z['est'].shape==(len(idx),2) and np.isfinite(z['est']).all()
                out[var][i,idx]=z['est'];seen[idx]=True
            if support is None:support=seen
            else:assert np.array_equal(support,seen),'Unequal prediction support'
    return out,support


def pooled_baseline(units,learner,rule,target,nrows):
    out=np.full(nrows,np.nan);seen=np.zeros(nrows,bool)
    for u in units:
        p=baseline_path(u['name'],learner,rule,target)
        if not p.is_file():return None
        z=npz(p);idx=z['idx'];assert z['est'].shape==(len(idx),) and not seen[idx].any()
        out[idx]=z['est'];seen[idx]=True
    return np.broadcast_to(out,(len(M.INITS),nrows))


def metrics(name,est,truth,units):
    error=est-truth;ones=np.ones_like(error)
    out={m:bootstrap(name+'/'+m,num,ones,units,m) for m,num in [('RMSE',error**2),('MAE',np.abs(error))]}
    out['medianAE']=median_bootstrap(name+'/medianAE',np.abs(error),units)
    eligible=truth>100
    out['share_within_10pct']=bootstrap(name+'/share10',((np.abs(error)<=.1*truth)&eligible).astype(float),np.broadcast_to(eligible,error.shape).astype(float),units,'ratio') if eligible.any() else {'point':None,'note':'no targets above 100 eV'}
    out['share_outside_0_3000']=bootstrap(name+'/outside',((est<0)|(est>3000)).astype(float),ones,units,'ratio')
    out.update(prediction_min_per_init=est.min(1).tolist(),prediction_max_per_init=est.max(1).tolist(),share10_denominator=int(eligible.sum()))
    return out


def evaluate_cell(name,R,pred,support,mask,baselines,head):
    truth=R['tg_'+('F' if head==0 else 'L5')];base=support&mask&np.isfinite(truth);idx=np.flatnonzero(base);assert len(idx)
    labels={'cluster':R['clus'][idx],'discharge':R['fidx'][idx]}
    cell={'rows':len(idx),'shots':len(np.unique(labels['discharge'])),'clusters':len(np.unique(labels['cluster'])),'excluded_from_predictions':int(support.sum()-len(idx)),'metrics':{},'pairs':{},'baseline_notes':{}}
    for v in M.VARIANTS:cell['metrics'][v]=metrics(name+'/'+v,pred[v][:,idx,head],truth[idx],labels)
    cell['pairs']['X over H']=bootstrap(name+'/X over H',(pred['X'][:,idx,head]-truth[idx])**2,(pred['H'][:,idx,head]-truth[idx])**2,labels)
    for learner,array in baselines.items():
        if array is None:cell['baseline_notes'][learner]='unavailable';continue
        pi=np.flatnonzero(base&np.isfinite(array).all(0));assert len(pi)
        lab={'cluster':R['clus'][pi],'discharge':R['fidx'][pi]}
        cell['metrics'][learner]=metrics(name+'/'+learner,array[:,pi],truth[pi],lab)
        pair=bootstrap(name+'/X over '+learner,(pred['X'][:,pi,head]-truth[pi])**2,(array[:,pi]-truth[pi])**2,lab)
        pair.update(rows=len(pi),shots=len(np.unique(lab['discharge'])),clusters=len(np.unique(lab['cluster'])),excluded_from_cell=int(len(idx)-len(pi)))
        cell['pairs']['X over '+learner]=pair
    return cell


def descriptive_ages(R,G,pred,support,rule):
    import data85 as D
    anc=D.anchors(rule,G);gi=R['gi'];has=anc[gi]>=0;age=np.full(len(gi),np.nan)
    age[has]=(G['time'][gi[has]]-G['time'][anc[gi[has]]])*1000
    bins=[('no_history',~has)]+[(f'{a}-{b}ms',has&(age>=a)&(age<b)) for a,b in zip((0,50,100,300,1000),(50,100,300,1000,np.inf))]
    out={}
    for head,target in enumerate(('F','L5')):
        truth=R['tg_'+target];out[target]={}
        for label,band in bins:
            m=support&band&np.isfinite(truth)
            if m.any():
                vals=np.sqrt(np.mean((pred['X'][:,m,head]-truth[m])**2,axis=1))
                out[target][label]={'rows':int(m.sum()),'RMSE':float(vals.mean()),'per_init':vals.tolist()}
    return out


def cpu_latency(unit):
    import torch,data85 as D
    from model85 import create_model
    torch.set_num_threads(1);folder=M.CACHE/'runs'/f"{unit['name']}__{M.INITS[0]}__X"
    stats=json.loads((folder/'stats.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
    start=time.perf_counter();segs=D.build_segments(manifest['test'],stats,'D10');pre=1000*(time.perf_counter()-start)/sum(len(s['x']) for s in segs)
    model=create_model('X',M.INITS[0]);model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True));model.eval();times=[]
    with torch.no_grad():
        for seg in segs:
            state=None
            for j,row in enumerate(seg['x']):
                x=torch.from_numpy(row.copy()).reshape(1,1,22);t=time.perf_counter();_,state=model.forward_step(x,state);elapsed=1000*(time.perf_counter()-t)
                if j>=20:times.append(elapsed)
                if len(times)>=2000:break
            if len(times)>=2000:break
    assert times
    return {'median_ms':float(np.median(times)),'p99_ms':float(np.quantile(times,.99)),'preprocessing_amortized_ms_per_row':pre,'samples':len(times),'threads':1,'platform':platform.platform(),'note':'shared machine; acquisition/transfer latency excluded; preprocessing amortized separately'}


def fmt(v):
    return 'undefined' if v.get('point') is None else f"{v['point']:.5f} [{v['cluster'][0]:.5f}, {v['cluster'][1]:.5f}]"


def main():
    units=M.select_units();guarded_ready(units);baseline_inventory(units);G,R=M.load_sources();n=len(R['gi'])
    sets75=npz(M.DATA/'.b75/sets.npz');sets78=npz(M.DATA/'.b78/sets.npz')
    groups={d:[u for u in units if u['name'].startswith(d)] for d in ('V1','V2','V3','V4a')}
    result={'registration':'PREREGISTRATION_B85.md','initializations':list(M.INITS),'bootstrap_draws':20000,'designs':{},'limitations':['Exploratory previously used data; F/L5 sensor-derived labels, not independent physical truth.','Intervals condition on fixed fitted models, three initialization seeds; no multiplicity correction.','Network excludes early-stop validation files; comparator fit sets are larger.','One constant rule per segment; arbitrary mid-segment switching not tested.']}
    md=['# B.85 unified availability estimator','',*result['limitations'],'','Point and 95% shot-cluster intervals below; JSON includes discharge intervals, each initialization, all support/exclusion counts. No ensemble is formed.','']
    for design,group in groups.items():
        rec={'cells':{},'descriptive_age_bins_full_test_rows':{}}
        for rule in M.RULES:
            pred,support=pooled_predictions(group,rule,n);rec['descriptive_age_bins_full_test_rows'][rule]=descriptive_ages(R,G,pred,support,rule)
            for head,target in enumerate(('F','L5')):
                mask=(sets78 if rule=='K10' else sets75)[('S_' if rule=='S' else 'common_')+target]
                baselines={b:pooled_baseline(group,b,rule,target,n) for b in ('T0','TX')}
                if target=='L5':baselines['TXall']=pooled_baseline(group,'TXall',rule,target,n)
                key=rule+'/'+target;rec['cells'][key]=evaluate_cell(design+'/'+key,R,pred,support,mask,baselines,head)
            if rule=='D10':rec['cells']['D10/F_full']=evaluate_cell(design+'/D10/F_full',R,pred,support,np.ones(n,bool),{'TXfresh':pooled_baseline(group,'TXfresh',rule,'F',n)},0)
        specs=[('U1','D10/F_full','X over TXfresh',-.05),('U2','D650/L5','X over TX',-.05),('U3','S/L5','X over TX',-.05),('U4','K10/L5','X over T0',0),('U5','D650/L5','X over H',0)]
        rec['statements']={};md+=['## '+design,'','| Statement | MSE skill (95% cluster interval) | Excludes zero | Holds | Expected |','|---|---|---|---|---|']
        for sid,cell,pair,threshold in specs:
            v=rec['cells'][cell]['pairs'][pair];st={**v,'cell':cell,'pair':pair,'threshold':threshold,'holds':bool(v['cluster'][0]>threshold),'expectation':'holds' if sid=='U5' else 'uncertain'};rec['statements'][sid]=st
            md.append(f"| {sid} | {fmt(v)} | {v['excludes_zero']} | {st['holds']} | {st['expectation']} |")
        for key,cell in rec['cells'].items():
            md+=['',f"### {key}: {cell['rows']} rows, {cell['shots']} shots, {cell['clusters']} clusters",'','| Estimator | RMSE (eV) | MAE (eV) | Median AE (eV) |','|---|---|---|---|']
            for learner,met in cell['metrics'].items():md.append('| '+learner+' | '+' | '.join(fmt(met[m]) for m in ('RMSE','MAE','medianAE'))+' |')
            md+=['','| Pair | MSE skill | Excludes zero |','|---|---|---|']
            for pair,v in cell['pairs'].items():md.append(f"| {pair} | {fmt(v)} | {v['excludes_zero']} |")
            for learner,note in cell['baseline_notes'].items():md.append(learner+': '+note)
        result['designs'][design]=rec
    result['propose_replacement']=all(result['designs'][d]['statements'][s]['holds'] for d in groups for s in ('U1','U2','U3','U4')) and all(result['designs'][d]['statements']['U5']['holds'] for d in ('V3','V4a'))
    result['cpu_latency']=cpu_latency(units[0]);md+=['','Replacement criterion met: '+str(result['propose_replacement']),'Failure is not equivalence or intrinsic impossibility; revised mixtures require new registration.']
    M.C.save_json(M.PAPER/'b85_unified.json',result);M.C.save_text(M.PAPER/'b85_unified.md','\n'.join(md).rstrip()+'\n')


if __name__=='__main__':main()
