"""Screen and promoted results kept separate; no ranking by test errors."""
import importlib.util,json,platform,time,zlib
import numpy as np
import common90 as M
import data85 as D
from adapter90 import activate,guarded_ready,dependency_ready
from predict90 import npz,path,validate_predictions,prediction_hash

def bootstrap_module():
    spec=importlib.util.spec_from_file_location('_b90_pure_readout85',M.HERE.parent/'b85_unified/readout85.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    b.rng_for=lambda name:np.random.default_rng([20261190,zlib.crc32(name.encode())])
    return b

def baseline_ready():
    from adapter88 import baseline_trainer
    t=baseline_trainer()
    for u in M.select_units():
        for seed in M.INITS:
            rd=t.get_run_dir(u['name'],seed,'X');assert t.verify_complete(rd);cfg=json.loads((rd/'config.json').read_text())
            for rule in M.RULES:
                z=npz(M.DATA/'.b85/predictions'/f"{u['name']}__{seed}__X__{rule}.npz")
                assert z['est'].shape==(len(z['idx']),2) and np.isfinite(z['est']).all()
                assert str(z['config_hash'])==cfg['config_hash']

def pooled(method,units,seeds,rule,n):
    result={v:np.full((len(seeds),n,2),np.nan) for v in ('X','H','B85X')};support=None
    for i,seed in enumerate(seeds):
        for var in result:
            seen=np.zeros(n,bool)
            for u in units:
                p=M.DATA/'.b85/predictions'/f"{u['name']}__{seed}__X__{rule}.npz" if var=='B85X' else path(method,u['name'],seed,var,rule)
                z=npz(p);idx=z['idx'];assert ((idx>=0)&(idx<n)).all() and not seen[idx].any()
                result[var][i,idx]=z['est'];seen[idx]=True
            if var!='B85X':
                if support is None:support=seen
                else:assert np.array_equal(support,seen)
    return result,support

def evaluate(b,name,R,pred,support,mask,head):
    truth=R['tg_'+('F' if head==0 else 'L5')];idx=np.flatnonzero(support&mask&np.isfinite(truth));assert len(idx)
    lab={'cluster':R['clus'][idx],'discharge':R['fidx'][idx]}
    cell={'rows':len(idx),'shots':len(np.unique(lab['discharge'])),'clusters':len(np.unique(lab['cluster'])),'excluded_from_predictions':int(support.sum()-len(idx)),'metrics':{},'pairs':{}}
    for v in ('X','H'):cell['metrics'][v]=b.metrics(name+'/'+v,pred[v][:,idx,head],truth[idx],lab)
    cell['pairs']['X over H']=b.bootstrap(name+'/X over H',(pred['X'][:,idx,head]-truth[idx])**2,(pred['H'][:,idx,head]-truth[idx])**2,lab)
    pi=idx[np.isfinite(pred['B85X'][:,idx,head]).all(0)];assert len(pi)
    bl={'cluster':R['clus'][pi],'discharge':R['fidx'][pi]}
    cell['metrics']['B85X']=b.metrics(name+'/B85X',pred['B85X'][:,pi,head],truth[pi],bl)
    pair=b.bootstrap(name+'/X over B85X',(pred['X'][:,pi,head]-truth[pi])**2,(pred['B85X'][:,pi,head]-truth[pi])**2,bl)
    pair.update(rows=len(pi),shots=len(np.unique(bl['discharge'])),clusters=len(np.unique(bl['cluster'])),excluded_from_cell=len(idx)-len(pi));cell['pairs']['X over B85X']=pair
    return cell

def latency(method,unit):
    import torch
    from models90 import create_model
    torch.set_num_threads(1);rd=activate(method).get_run_dir(unit['name'],1701,'X')
    stats=json.loads((rd/'stats.json').read_text());manifest=json.loads((rd/'manifest.json').read_text())
    started=time.perf_counter();segs=D.build_segments(manifest['test'],stats,'D10');pre=1000*(time.perf_counter()-started)/sum(len(s['x']) for s in segs)
    model=create_model(method=method,variant='X',init_seed=1701);model.load_state_dict(torch.load(rd/'best.pt',map_location='cpu',weights_only=True));model.eval();elapsed=[]
    with torch.no_grad():
        for s in segs:
            state=None
            for j,row in enumerate(s['x']):
                x=torch.from_numpy(row.copy()).reshape(1,1,22);start=time.perf_counter();_,state=model.forward_step(x,state);ms=1000*(time.perf_counter()-start)
                if j>=20:elapsed.append(ms)
                if len(elapsed)>=2000:break
            if len(elapsed)>=2000:break
    assert elapsed
    return {'median_ms':float(np.median(elapsed)),'p99_ms':float(np.quantile(elapsed,.99)),'samples':len(elapsed),'threads':1,'preprocessing_amortized_ms_per_row':pre,'platform':platform.platform(),'note':'held-out CPU trace, acquisition/transfer excluded; shared machine; preprocessing separate'}

def fmt(v):return 'undefined' if v.get('point') is None else f"{v['point']:.5f} [{v['cluster'][0]:.5f}, {v['cluster'][1]:.5f}]"

def main():
    M.assert_registered();M.assert_frozen();assert dependency_ready();guarded_ready();validate_predictions();baseline_ready()
    pc=json.loads((M.CACHE/'prediction_checks.json').read_text());assert pc['all_pass'] and pc['source_hash']==prediction_hash()
    # No scientific errors above this point.
    b=bootstrap_module();G,R=M.load_sources();units=M.select_units();n=len(R['gi']);sets75=npz(M.DATA/'.b75/sets.npz');sets78=npz(M.DATA/'.b78/sets.npz')
    promoted=M.promotion()['promoted'];latencies={m:latency(m,units[0]) for m in M.METHODS}
    groups={design:[u for u in units if u['name'].startswith(design)] for design in ('V1','V2','V3','V4a')}
    specs=[('M1','D10/F_full','X over B85X',-.05),('M2','D650/L5','X over H',0),('M3','S/L5','X over H',0),('M4','K10/L5','X over H',0)]
    from models90 import create_model,ARCHITECTURES
    for stage,methods,seeds in [('screen',M.METHODS,(1701,)),('methods',promoted,M.INITS)]:
        result={'registration':'PREREGISTRATION_B90.md','stage':stage,'initializations':list(seeds),'bootstrap_seed':20261190,'bootstrap_draws':20000,'promotion':M.promotion(),'methods':{},'limitations':['Used-data exploratory comparison; no independent physical truth or new test set.','Intervals omit model-selection, refit and multiplicity uncertainty; no ensemble.','One-seed screen and three-seed promoted analyses are separate.','No calibrated uncertainty, physical-state identification or automatic manuscript replacement.']}
        md=[f'# B.90 {stage}','',*result['limitations'],'','Point and 95% shot-cluster intervals; discharge and per-seed values are in JSON.','']
        for method in methods:
            count=create_model(method=method,variant='X',init_seed=1701).param_count()
            rec={'architecture':ARCHITECTURES[method],'parameters':count,'cpu_latency':latencies[method],'designs':{}}
            md+=['## Method '+method,'',f'Parameters: {count}; CPU p99 {latencies[method]["p99_ms"]:.4f} ms.','']
            for design,group in groups.items():
                dr={'cells':{},'statements':{}}
                for rule in M.RULES:
                    pred,support=pooled(method,group,seeds,rule,n)
                    for h,target in enumerate(('F','L5')):
                        mask=(sets78 if rule=='K10' else sets75)[('S_' if rule=='S' else 'common_')+target]
                        dr['cells'][rule+'/'+target]=evaluate(b,f'{stage}/{method}/{design}/{rule}/{target}',R,pred,support,mask,h)
                    if rule=='D10':dr['cells']['D10/F_full']=evaluate(b,f'{stage}/{method}/{design}/D10/F_full',R,pred,support,np.ones(n,bool),0)
                md+=['### '+design,'','| Statement | Skill, 95% interval | Excludes zero | Holds | Expected |','|---|---|---|---|---|']
                for sid,key,pair,threshold in specs:
                    v=dr['cells'][key]['pairs'][pair];st={**v,'cell':key,'pair':pair,'threshold':threshold,'expectation':'uncertain','holds':bool(v['cluster'][0]>threshold),'registered_for_promoted':stage=='methods'};dr['statements'][sid]=st
                    md.append(f"| {sid} | {fmt(v)} | {v['excludes_zero']} | {st['holds']} | uncertain |")
                for key,cell in dr['cells'].items():
                    md+=['',f"#### {key}: {cell['rows']} rows, {cell['shots']} shots, {cell['clusters']} clusters",'','| Estimator | RMSE | MAE | Median AE | Within 10% (>100 eV) |','|---|---|---|---|---|']
                    for v,met in cell['metrics'].items():md.append('| '+v+' | '+' | '.join(fmt(met[k]) for k in ('RMSE','MAE','medianAE','share_within_10pct'))+' |')
                    md+=['','| Pair | MSE skill | Excludes zero |','|---|---|---|']
                    for pair,v in cell['pairs'].items():md.append(f"| {pair} | {fmt(v)} | {v['excludes_zero']} |")
                rec['designs'][design]=dr
                print('Readout '+stage+'/'+method+'/'+design,flush=True)
            rec['candidate_for_discussion']=stage=='methods' and all(rec['designs'][d]['statements'][s]['holds'] for d in groups for s in ('M1','M2','M4')) and rec['cpu_latency']['p99_ms']<=10
            rec['S_claim_supported']=stage=='methods' and all(rec['designs'][d]['statements']['M3']['holds'] for d in groups)
            result['methods'][method]=rec
        result['overturning_measurement']='An untouched campaign with acquisition/arrival timestamps and matched fast/history comparisons.'
        M.save(M.CACHE/f'b90_{stage}.json',result);M.save(M.PAPER/f'b90_{stage}.json',result);M.C.save_text(M.PAPER/f'b90_{stage}.md','\n'.join(md)+'\n')
if __name__=='__main__':main()
