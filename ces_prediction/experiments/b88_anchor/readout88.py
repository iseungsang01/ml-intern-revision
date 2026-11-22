"""Guarded B88 readout; frozen B85 pure metrics with registered RNG seed."""
import json,platform,time,zlib
import numpy as np
import common88 as M
import readout85 as B
from adapter88 import activate,guarded_ready,dependency_ready,baseline_trainer
from predict88 import npz,validate_predictions,prediction_hash
B.rng_for=lambda name:np.random.default_rng([20261188,zlib.crc32(name.encode())])

def ready():
    M.assert_frozen();assert dependency_ready();guarded_ready()
    checks=json.loads((M.CACHE/'prediction_checks.json').read_text(encoding='utf8'))
    assert checks.get('all_pass') and checks['source_hash']==prediction_hash()
    validate_predictions()
    # Validate supplementary comparator prediction/config identity before any score.
    t=baseline_trainer()
    for u in M.select_units():
        for seed in M.INITS:
            rd=t.get_run_dir(u['name'],seed,'X');assert t.verify_complete(rd)
            cfg=json.loads((rd/'config.json').read_text(encoding='utf8'))
            for rule in M.RULES:
                z=npz(M.DATA/'.b85/predictions'/f"{u['name']}__{seed}__X__{rule}.npz")
                assert z['idx'].ndim==1 and len(z['idx']) and np.all(np.diff(z['idx'])>0)
                assert z['est'].shape==(len(z['idx']),2) and np.isfinite(z['est']).all()
                assert str(z['config_hash'])==cfg['config_hash']
    return True

def pooled_predictions(units,rule,nrows):
    out={v:np.full((len(M.INITS),nrows,2),np.nan) for v in (*M.VARIANTS,'B85X')};support=None;bsupport=None
    for i,seed in enumerate(M.INITS):
        for var in out:
            seen=np.zeros(nrows,bool)
            for u in units:
                folder=M.DATA/'.b85/predictions' if var=='B85X' else M.CACHE/'predictions';v='X' if var=='B85X' else var
                z=npz(folder/f"{u['name']}__{seed}__{v}__{rule}.npz");idx=z['idx']
                assert ((idx>=0)&(idx<nrows)).all() and not seen[idx].any(),'Overlapping fold support'
                out[var][i,idx]=z['est'];seen[idx]=True
            if var=='B85X':
                if bsupport is None:bsupport=seen
                else:assert np.array_equal(bsupport,seen)
            elif support is None:support=seen
            else:assert np.array_equal(support,seen)
    return out,support,bsupport

def evaluate_cell(name,R,pred,support,mask,head):
    truth=R['tg_'+('F' if head==0 else 'L5')];keep=support&mask&np.isfinite(truth);idx=np.flatnonzero(keep);assert len(idx)
    labels={'cluster':R['clus'][idx],'discharge':R['fidx'][idx]}
    cell={'rows':len(idx),'shots':len(np.unique(labels['discharge'])),'clusters':len(np.unique(labels['cluster'])),'excluded_from_predictions':int(support.sum()-len(idx)),'metrics':{},'pairs':{}}
    for var in M.VARIANTS:cell['metrics'][var]=B.metrics(name+'/'+var,pred[var][:,idx,head],truth[idx],labels)
    cell['pairs']['A over C']=B.bootstrap(name+'/A over C',(pred['A'][:,idx,head]-truth[idx])**2,(pred['C'][:,idx,head]-truth[idx])**2,labels)
    common=keep&np.isfinite(pred['B85X'][:,:,head]).all(0);bi=np.flatnonzero(common);assert len(bi)
    lab={'cluster':R['clus'][bi],'discharge':R['fidx'][bi]}
    cell['metrics']['B85X']=B.metrics(name+'/B85X',pred['B85X'][:,bi,head],truth[bi],lab)
    pair=B.bootstrap(name+'/A over B85X',(pred['A'][:,bi,head]-truth[bi])**2,(pred['B85X'][:,bi,head]-truth[bi])**2,lab)
    pair.update(rows=len(bi),shots=len(np.unique(lab['discharge'])),clusters=len(np.unique(lab['cluster'])),excluded_from_cell=len(idx)-len(bi));cell['pairs']['A over B85X']=pair
    return cell

def cpu_latency(unit):
    import torch,data88 as D
    from model88 import create_model
    torch.set_num_threads(1);rd=activate().get_run_dir(unit['name'],M.INITS[0],'A')
    stats=json.loads((rd/'stats.json').read_text());manifest=json.loads((rd/'manifest.json').read_text())
    started=time.perf_counter();segs=D.build_segments(manifest['test'],stats,'D10');pre=1000*(time.perf_counter()-started)/sum(len(s['x']) for s in segs)
    model=create_model('A',M.INITS[0]);model.load_state_dict(torch.load(rd/'best.pt',map_location='cpu',weights_only=True));model.eval();elapsed=[]
    with torch.no_grad():
        for s in segs:
            state=None
            for j,row in enumerate(s['x']):
                x=torch.from_numpy(row.copy()).reshape(1,1,35);start=time.perf_counter();_,state=model.forward_step(x,state);ms=1000*(time.perf_counter()-start)
                if j>=20:elapsed.append(ms)
                if len(elapsed)>=2000:break
            if len(elapsed)>=2000:break
    assert elapsed
    return {'median_ms':float(np.median(elapsed)),'p99_ms':float(np.quantile(elapsed,.99)),'samples':len(elapsed),'threads':1,'preprocessing_amortized_ms_per_row':pre,'platform':platform.platform(),'note':'actual held-out trace; acquisition/transfer excluded; preprocessing separate; shared machine'}

def fmt(v):return 'undefined' if v.get('point') is None else f"{v['point']:.5f} [{v['cluster'][0]:.5f}, {v['cluster'][1]:.5f}]"
def main():
    ready() # No scientific error is calculated before this complete guard.
    G,R=M.load_sources();n=len(R['gi']);units=M.select_units();sets75=npz(M.DATA/'.b75/sets.npz');sets78=npz(M.DATA/'.b78/sets.npz')
    groups={d:[u for u in units if u['name'].startswith(d)] for d in ('V1','V2','V3','V4a')}
    result={'registration':'PREREGISTRATION_B88.md','initializations':M.INITS,'bootstrap_draws':20000,'bootstrap_seed':20261188,'designs':{},'limitations':['Used-data exploration; F/L5 are sensor-derived labels, not independent physical truth.','Intervals condition on fixed models and three initialization seeds; no refit uncertainty or multiplicity correction.','Small cluster counts and constant per-segment availability; arbitrary switching untested.'],'overturning_measurement':'Untouched campaign with acquisition/arrival timestamps and an increment against matched C.'}
    md=['# B.88 explicit measurement-time anchor','',*result['limitations'],'','Point and 95% cluster intervals; discharge intervals and per-initialization metrics are in JSON. No ensemble.','']
    specs=[('A1','D10/F_full','A over C',-.05,'holds'),('A2','D650/L5','A over C',0,'uncertain'),('A3','S/L5','A over C',0,'uncertain'),('A4','K10/L5','A over C',0,'uncertain'),('A5','D10/F_full','A over B85X',-.05,'uncertain')]
    for design,group in groups.items():
        rec={'cells':{},'statements':{}}
        for rule in M.RULES:
            pred,support,bsupport=pooled_predictions(group,rule,n)
            for h,target in enumerate(('F','L5')):
                mask=(sets78 if rule=='K10' else sets75)[('S_' if rule=='S' else 'common_')+target]
                rec['cells'][rule+'/'+target]=evaluate_cell(design+'/'+rule+'/'+target,R,pred,support,mask,h)
            if rule=='D10':rec['cells']['D10/F_full']=evaluate_cell(design+'/D10/F_full',R,pred,support,np.ones(n,bool),0)
        md+=['## '+design,'','| Statement | MSE skill (95% interval) | Excludes zero | Holds | Expected |','|---|---|---|---|---|']
        for sid,key,pair,threshold,expected in specs:
            v=rec['cells'][key]['pairs'][pair];st={**v,'cell':key,'pair':pair,'threshold':threshold,'expectation':expected,'holds':bool(v['cluster'][0]>threshold)};rec['statements'][sid]=st
            md.append(f"| {sid} | {fmt(v)} | {v['excludes_zero']} | {st['holds']} | {expected} |")
        for key,cell in rec['cells'].items():
            md+=['',f"### {key}: {cell['rows']} rows, {cell['shots']} shots, {cell['clusters']} clusters",'','| Estimator | RMSE | MAE | Median AE |','|---|---|---|---|']
            for v,met in cell['metrics'].items():md.append('| '+v+' | '+' | '.join(fmt(met[k]) for k in ('RMSE','MAE','medianAE'))+' |')
            md+=['','| Pair | MSE skill | Excludes zero |','|---|---|---|']
            for pair,v in cell['pairs'].items():md.append(f"| {pair} | {fmt(v)} | {v['excludes_zero']} |")
        result['designs'][design]=rec
    result['cpu_latency']=cpu_latency(units[0])
    result['propose_replacement']=all(result['designs'][d]['statements'][s]['holds'] for d in groups for s in ('A1','A2','A4','A5')) and result['cpu_latency']['p99_ms']<=10
    result['S_claim_supported']=all(result['designs'][d]['statements']['A3']['holds'] for d in groups)
    md+=['','Candidate criterion met: '+str(result['propose_replacement']),'S statement supported in all designs: '+str(result['S_claim_supported']),'No automatic manuscript change. Failure is not equivalence or intrinsic impossibility.',result['overturning_measurement']]
    M.C.save_json(M.CACHE/'readout.json',result);M.C.save_json(M.PAPER/'b88_anchor.json',result);M.C.save_text(M.PAPER/'b88_anchor.md','\n'.join(md)+'\n')
if __name__=='__main__':main()
