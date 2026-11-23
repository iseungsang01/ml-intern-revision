"""Read existing TRAIN/VAL trajectories only; bounded redacted Claude design review."""
import json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CACHE=ROOT/'data/.b104'
sys.path.insert(0,str(HERE.parent/'b85_unified'));import claude_review85 as API

def call(stage,model,prompt,max_tokens=8000,extra=None):
    folder=CACHE/'claude';folder.mkdir(parents=True,exist_ok=True)
    assert len(list(folder.glob('*.request.json')))<8,'Bounded8 calls for104 design/implementation/post review'
    req=folder/(stage+'.request.json');assert not req.exists(),'No blind paid attempt replay'
    payload={'model':model,'max_tokens':max_tokens,'thinking':{'type':'adaptive'},'output_config':{'effort':'low'},'messages':[{'role':'user','content':prompt}],**(extra or {})}
    req.write_text(json.dumps(payload,indent=2),encoding='utf8');start=time.time()
    try:r=API.request('/messages',API.credential(),payload)
    except Exception as e:
        (folder/(stage+'.error.json')).write_text(json.dumps({'error':str(e),'seconds':time.time()-start}),encoding='utf8');raise
    (folder/(stage+'.response.json')).write_text(json.dumps({'response':r,'seconds':time.time()-start},indent=2),encoding='utf8')
    return r

def scout():
    CACHE.mkdir(exist_ok=True);records=[]
    for seed in (1701,1702,1703):
        for code in ('TX','TH'):
            folder=ROOT/'data/.b94/confirm'/f'V3_q3__{seed}__{code}'
            z=json.loads((folder/'metrics.json').read_text());c=json.loads((folder/'config.json').read_text());end=json.loads((folder/'process_exit.json').read_text());assert end['returncode']==0
            history=z['history'];best=min(range(len(history)),key=lambda i:history[i]['val_loss'])
            records.append({'init':seed,'model':'FAST' if code=='TX' else 'HISTORY','epochs':len(history),'stop':z['stop'],'seconds':z['wall_seconds'],'best_epoch':best,
                'val_curve':[{'epoch':h['epoch'],'mse':h['val_loss'],'train_objective':h['train_loss'],'lr':h['lr'],'updates':h['updates']} for h in history],
                'budget':c['seconds'],'epoch_cap':c['epochs'],'prepared_hash':c['prepared_artifact_sha256']})
    (CACHE/'trainval_scout.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    prompt='''Review a minimal, TEST-informed exploratory follow-up; at most700 words, concrete corrections. No journal acceptance/novelty claims. Preserve pedestal-top report nowcasting. Prior audited fixed-budget TCN comparison motivates checking training budget only; do not select new architectures/units/seeds from TEST. No TEST performance values are sent. Anonymized TRAIN/VAL aggregate curves below, no raw observations/identity.
Proposal: fixed temporal unit already specified by prior plan,3 existing initializations, FAST/HISTORY matched TCN pair =6 fresh fits. Same original frozen numerical train loop, train-only statistics, exact cached TRAIN/VAL inputs, seven availability-rule validation each epoch, AdamW/dropout/loss/scheduler/earlystop6 unchanged. Original40-epoch cap retained, only per-fit wall budget1200->3600 seconds. Four global GPU workers as previously qualified, no other jobs stopped. New independent suite cap3h; old completed study clock untouched.
Important improvement: instrument same fresh trajectory to save best fully validated checkpoint completed by1200s as SHORT baseline; extended best by3600s/40epochs/earlystop is LONG. Do not claim exact continuation of old checkpoints (no full RNG). Validation observer calls original unchanged validator; when it returns, record elapsed=now-(deadline-config_seconds), save a prefix best only if fully validated elapsed<=1200. Measure snapshot I/O, preserve timers. Stops before1200 may make identical SHORT/LONG, report honestly. Concurrency affects throughput; report epochs/updates/budget stops, not convergence or architectural superiority.
Evaluate only after all6 fits pass independent checks. Minimal endpoints D10 and D50, both F current CES report and L5 future-five-report median. These D-rules are history availability delays, not forecast horizons. For each rule/head, same rows and paired3inits: skill_fast at SHORT/LONG =mean_init(1-MSE_FAST/MSE_HISTORY); primary budget effect per model=mean_init(1-MSE_LONG/MSE_SHORT); and change in fast skill LONG-SHORT with paired-init then shared cluster/discharge resampling of all4 errors,20k draws. Existing20min B94 trajectories are contextual checks only, not the matched primary baseline. No support intersection across validation designs. This new study is explicitly restricted to the fixed temporal unit, not evidence across all four designs. All four previous designs remain reported in existing paper/results.
Controls: real fresh smoke both models, source/data/software binding, explicit CES env, old artifacts read-only, no current/future target inputs, prefix state snapshots no RNG mutation, global4 VRAM guard, no blind numerical retry; external interruption may archive and allow one fresh retry within registered total cap only; deterministic errors alert/stop. Local15s process/status observer, phase/error and5min progress notifications with real delivery check. No automatic paid API replay. Preregister/commit before scientific runs; independent readout and source audit before post-eval Claude planning. Advise exact tolerances/control tests, sensible negative/positive outcome rules and whether3600seconds is worth running based solely on curves. Future independent confirmation required.\n'''+json.dumps(records)
    r=call('design','claude-fable-5-2',prompt)
    assert r.get('stop_reason')=='end_turn','Incomplete review; inspect before launch'
    text='\n'.join(b['text'] for b in r['content'] if b['type']=='text')
    (CACHE/'design_review.txt').write_text(text,encoding='utf8')
    print(json.dumps({'review_complete':True,'records':len(records),'summary':[{'model':r['model'],'epochs':r['epochs'],'best_epoch':r['best_epoch'],'stop':r['stop']} for r in records]}))

if __name__=='__main__':scout()
