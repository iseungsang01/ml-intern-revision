"""Two separate registered evidence scopes; no automatic model adoption."""
import json,platform,time,zlib
import numpy as np
import torch
import common92 as M
import readout85 as B
import data88 as D
from predict92 import guarded_ready,validate_predictions,npz
from model92 import create_model
B.rng_for=lambda name:np.random.default_rng([20261192,zlib.crc32(name.encode())])
def pooled(units,seeds,codes,rule,n):
    out={c:np.full((len(seeds),n,2),np.nan) for c in codes};support=None
    for j,s in enumerate(seeds):
        for code in codes:
            seen=np.zeros(n,bool)
            for u in units:
                z=npz(M.CACHE/'predictions'/f"{u['name']}__{s}__{code}__{rule}.npz");idx=z['idx'];assert not seen[idx].any()
                out[code][j,idx]=z['est'];seen[idx]=True
            if support is None:support=seen
            else:assert np.array_equal(support,seen)
    return out,support
def trees(units,seeds,rule,target,n,full=False):
    out={};notes={}
    for learner in (('TXfresh',) if full else ('T0','TX')):
        values=np.full(n,np.nan);seen=np.zeros(n,bool);missing=[]
        for u in units:
            p=B.baseline_path(u['name'],learner,rule,target)
            if not p.exists():missing.append(u['name']);continue
            z=npz(p);idx=z['idx'];assert not seen[idx].any() and z['est'].shape==(len(idx),)
            values[idx]=z['est'];seen[idx]=True
        out[learner]=np.broadcast_to(values,(len(seeds),n));notes[learner]={'missing_units':missing,'available_rows':int(seen.sum()),'different_model_budget':True}
    return out,notes
def cell(name,R,pred,support,mask,head,codes,baselines,notes):
    truth=R['tg_'+('F' if head==0 else 'L5')];idx=np.flatnonzero(support&mask&np.isfinite(truth));assert len(idx)
    labels={'cluster':R['clus'][idx],'discharge':R['fidx'][idx]}
    result={'rows':len(idx),'shots':len(np.unique(labels['discharge'])),'clusters':len(np.unique(labels['cluster'])),'excluded_from_prediction_support':int(support.sum()-len(idx)),'metrics':{}}
    for c in codes:result['metrics'][c]=B.metrics(name+'/'+c,pred[c][:,idx,head],truth[idx],labels)
    result['skill']=B.bootstrap(name+'/pair',(pred[codes[0]][:,idx,head]-truth[idx])**2,(pred[codes[1]][:,idx,head]-truth[idx])**2,labels)
    result['supplementary_tree_notes']=notes;result['supplementary_tree_pairs']={}
    base=support&mask&np.isfinite(truth)
    for learner,array in baselines.items():
        jj=np.flatnonzero(base&np.isfinite(array).all(0))
        if not len(jj):continue
        labels={'cluster':R['clus'][jj],'discharge':R['fidx'][jj]}
        result['metrics'][learner]=B.metrics(name+'/'+learner,array[:,jj],truth[jj],labels)
        pair=B.bootstrap(name+'/first over '+learner,(pred[codes[0]][:,jj,head]-truth[jj])**2,(array[:,jj]-truth[jj])**2,labels)
        pair.update(rows=len(jj),shots=len(np.unique(labels['discharge'])),clusters=len(np.unique(labels['cluster'])),excluded_from_cell=len(idx)-len(jj));result['supplementary_tree_pairs'][learner]=pair
    return result
def latency(unit,code,runtime):
    stats=json.loads((M.run_dir('confirm',unit['name'],1701,code)/'stats.json').read_text());manifest=json.loads((M.run_dir('confirm',unit['name'],1701,code)/'manifest.json').read_text())
    start=time.perf_counter();segs=D.build_segments(manifest['test'],stats,'D10');prep=1000*(time.perf_counter()-start)/sum(len(s['x']) for s in segs)
    rt={**runtime,'device':'cpu','threads':1,'cudnn':False};model=create_model(code,rt,1701);model.load_state_dict(torch.load(M.run_dir('confirm',unit['name'],1701,code)/'best.pt',map_location='cpu',weights_only=True));model.eval();times=[]
    with torch.no_grad():
        for seg in segs:
            state=None
            for j,row in enumerate(seg['x']):
                x=torch.from_numpy(row.copy()).reshape(1,1,35);start=time.perf_counter();_,state=model.forward_step(x,state);elapsed=1000*(time.perf_counter()-start)
                if j>=20:times.append(elapsed)
                if len(times)>=1000:break
            if len(times)>=1000:break
    assert times
    return {'median_ms':float(np.median(times)),'p99_ms':float(np.quantile(times,.99)),'samples':len(times),'threads':1,'preprocessing_ms_per_row':prep,'platform':platform.platform(),'note':'Actual held-out trace; excludes acquisition/transfer; preprocessing separate'}
def fmt(z):return f"{z['point']:.5f} [{z['cluster'][0]:.5f}, {z['cluster'][1]:.5f}]"
def main():
    selected,runtime,jobs=guarded_ready();checks=json.loads((M.CACHE/'prediction_checks.json').read_text());assert checks['all_pass'] and checks['sources']==M.sources() and checks['selection_hash']==M.sha(M.CACHE/'selection.json');validate_predictions()
    G,R=M.load_sources();n=len(R['gi']);codes=M.pair(selected['chosen']);sets75=npz(M.DATA/'.b75/sets.npz');sets78=npz(M.DATA/'.b78/sets.npz');allunits=M.select_units()
    result={'registration':'PREREGISTRATION_B92.md','selection':selected,'bootstrap_draws':20000,'bootstrap_seed':20261192,'scopes':{},'limitations':['Used-data exploration; selection/checkpoint reuse of validation; no refit, model-selection or multiplicity uncertainty.','F/L5 are sensor-derived labels, not independent temperature truth.','FULL15 uses one seed; FOUR-unit subset uses three seeds only on the fixed units. No pooled missing seeds or global adoption decision.'],'overturning_measurement':'Untouched campaign with acquisition/arrival timestamps and matched comparisons at a validated adequate training budget.'}
    md=['# B.92 budgeted selected pair','',*result['limitations'],'','Point and95% cluster intervals; discharge intervals and per-initialization details in JSON.','']
    for scope,units,seeds in [('FULL15_one_seed',allunits,[1701]),('FOUR_units_three_seeds',[u for u in allunits if u['name'] in M.PILOT_UNITS],[1701,1702,1703])]:
        sr={'seeds':seeds,'units':[u['name'] for u in units],'designs':{}}
        for design in ('V1','V2','V3','V4a'):
            group=[u for u in units if u['name'].startswith(design)];rec={'cells':{},'statements':{}}
            for rule in M.RULES:
                pred,support=pooled(group,seeds,codes,rule,n)
                for h,target in enumerate(('F','L5')):
                    mask=(sets78 if rule=='K10' else sets75)[('S_' if rule=='S' else 'common_')+target];key=rule+'/'+target;baselines,notes=trees(group,seeds,rule,target,n);rec['cells'][key]=cell(scope+'/'+design+'/'+key,R,pred,support,mask,h,codes,baselines,notes)
                if rule=='D10':
                    baselines,notes=trees(group,seeds,rule,'F',n,full=True);rec['cells']['D10/F_full']=cell(scope+'/'+design+'/D10/F_full',R,pred,support,np.ones(n,bool),0,codes,baselines,notes)
            md+=['## '+scope+' / '+design,'','| Statement | Skill (95% interval) | Excludes zero | Holds |','|---|---|---|---|']
            for sid,key,threshold in [('S1','D10/F_full',-.05),('S2','D650/L5',0),('S3','S/L5',0),('S4','K10/L5',0)]:
                z=rec['cells'][key]['skill'];rec['statements'][sid]={**z,'cell':key,'threshold':threshold,'holds':bool(z['cluster'][0]>threshold),'expectation':'uncertain'};md.append(f"| {sid} | {fmt(z)} | {z['excludes_zero']} | {rec['statements'][sid]['holds']} |")
            for key,cc in rec['cells'].items():
                md+=['',f"### {key}: {cc['rows']} rows, {cc['shots']} shots, {cc['clusters']} clusters",'','| Estimator | RMSE | MAE | MedianAE |','|---|---|---|---|']
                for c,v in cc['metrics'].items():md.append('| '+c+' | '+' | '.join(fmt(v[k]) for k in ('RMSE','MAE','medianAE'))+' |')
            sr['designs'][design]=rec
        result['scopes'][scope]=sr
    result['latency']={c:latency(allunits[0],c,runtime) for c in codes}
    result['fit_budgets']={str(M.run_dir(*job).relative_to(M.ROOT)):json.loads((M.run_dir(*job)/'metrics.json').read_text()) for job in M.pilot_jobs()+jobs}
    md+=['','No automatic adoption. Limited-budget failure does not establish equivalence or intrinsic inferiority.',result['overturning_measurement']]
    M.save(M.CACHE/'readout.json',result);M.save(M.PAPER/'b92_methods.json',result);M.C.save_text(M.PAPER/'b92_methods.md','\n'.join(md)+'\n')
if __name__=='__main__':main()
