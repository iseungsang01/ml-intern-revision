"""Bounded six-fit dispatch, fixed held-out readout and audited report publication."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import common104 as C
from status93 import process_identity,snapshot
OWNED={}
def event(kind,**details):
    path=C.CACHE/'dispatch_events.json';rows=C.load(path) if path.exists() else [];rows.append({'event':kind,'time':time.time(),**details});C.save(path,rows)

def status(phase,**kw):
    C.save(C.CACHE/'queue_status.json',{'pid':os.getpid(),'birth':process_identity(os.getpid())['birth'],'state':phase,'updated':time.time(),**kw})
def cmd(stage,seed,code,obs):return [sys.executable,'-u',str(C.HERE/'train104.py'),'--stage',stage,'--seed',str(seed),'--code',code,'--observe',str(int(obs))]
def start(stage,seed,code,obs):
    assert C.remaining()>(4920 if stage=='confirm' else 420),'Insufficient remaining suite time before launch'
    rd=C.run_dir(stage,seed,code,obs)
    assert not (rd/'config.json').exists(),'Incomplete old trajectory; explicit external recovery required'
    rd.mkdir(parents=True,exist_ok=True);log=(rd/'process.log').open('ab')
    env={k:v for k,v in os.environ.items() if not k.startswith('CES_')};env.update(C.env(stage,code,seed,obs));env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
    p=subprocess.Popen(cmd(stage,seed,code,obs),cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    z={'pid':p.pid,'birth':process_identity(p.pid)['birth'],'started':time.time(),'stage':stage,'seed':seed,'code':code,'observe':obs}
    e={'process':p,'log':log,'rd':rd,'meta':z,'hard':420 if stage=='smoke' else 4020};OWNED[p.pid]=e
    C.save(rd/'child_start.json',z);event('start',**z,concurrent_owned=len(OWNED));return e
def finish(entry,timeout=False):
    p=entry['process'];rc=p.wait();OWNED.pop(p.pid,None);entry['log'].close();rd=entry['rd'];cfg=C.load(rd/'config.json') if (rd/'config.json').exists() else {}
    C.save(rd/'process_exit.json',{'returncode':rc,'timeout':timeout,'pid':p.pid,'birth':entry['meta']['birth'],'elapsed':time.time()-entry['meta']['started'],'config_hash':cfg.get('config_hash')})
    event('exit',pid=p.pid,returncode=rc,timeout=timeout)
    assert rc==0 and not timeout,'Child failed: '+rd.name
    return C.verify_run(entry['meta']['stage'],entry['meta']['seed'],entry['meta']['code'],entry['meta']['observe'])
def recover(stage,seed,code,obs):
    rd=C.run_dir(stage,seed,code,obs)
    if not (rd/'config.json').exists() and not (rd/'child_start.json').exists():return
    assert stage=='confirm','Smoke failure requires inspection'
    assert not (rd/'complete.json').exists(),'Completion exists without verified normal exit: inspection required'
    start=C.load(rd/'child_start.json');identity=process_identity(start['pid'])
    assert identity['alive'] is False or identity['alive'] is True and identity['birth']!=start['birth'],'Old child alive/identity unknown'
    text=(rd/'process.log').read_text(encoding='utf8',errors='replace')
    assert not any(s in text for s in ('Traceback','Nonfinite','nonfinite','AssertionError')),'Deterministic failure; no retry'
    dest=C.CACHE/'external_attempts'/rd.name;assert not dest.exists(),'One retry already used'
    assert C.remaining()>4920,'Not enough suite time for registered fresh retry plus readout'
    dest.parent.mkdir(exist_ok=True);files={p.relative_to(rd).as_posix():C.sha(p) for p in rd.rglob('*') if p.is_file()}
    assert rd.resolve().is_relative_to(C.CACHE.resolve()) and dest.resolve().is_relative_to(C.CACHE.resolve())
    rd.rename(dest);C.save(dest/'archival_manifest.json',{'files':files,'time':time.time(),'fresh_retry':True})
def checks():
    import numpy as np,torch
    import train92 as T
    from prepare92 import batches
    from model92 import create_model
    rt=C.runtime();results={}
    for code in C.CODES:
        for obs in (False,True):
            rd=C.run_dir('smoke',1701,code,obs)
            if (rd/'process_exit.json').exists():C.verify_run('smoke',1701,code,obs);continue
            e=start('smoke',1701,code,obs)
            try:e['process'].wait(timeout=min(e['hard'],C.remaining()))
            except subprocess.TimeoutExpired:e['process'].kill();finish(e,True)
            finish(e)
        off=C.run_dir('smoke',1701,code,False);on=C.run_dir('smoke',1701,code,True)
        a=C.load(off/'metrics.json')['history'];b=C.load(on/'metrics.json')['history'];assert len(a)==len(b)==2
        av=np.array([e['val_loss'] for e in a]);bv=np.array([e['val_loss'] for e in b]);assert np.allclose(av,bv,rtol=1e-6,atol=1e-7)
        exact={'validation_per_epoch':(av==bv).tolist()}
        wa=torch.load(off/'best.pt',map_location='cpu',weights_only=True);wb=torch.load(on/'best.pt',map_location='cpu',weights_only=True)
        assert set(wa)==set(wb) and all(torch.allclose(wa[k],wb[k],rtol=1e-6,atol=1e-7) for k in wa)
        exact['best_parameters']={k:torch.equal(wa[k],wb[k]) for k in wa}
        la=torch.load(off/'latest.pt',map_location='cpu',weights_only=False)['model'];lb=torch.load(on/'latest.pt',map_location='cpu',weights_only=False)['model']
        assert set(la)==set(lb) and all(torch.allclose(la[k],lb[k],rtol=1e-6,atol=1e-7) for k in la)
        exact['final_parameters']={k:torch.equal(la[k],lb[k]) for k in la};results[C.LABEL[code]]=exact
        prepared=C.load_prepared();val={}
        import common92 as M
        for rule in M.TRAIN_RULES:
            segments=prepared['val'][rule]
            selected=[s for s in segments if s['mask'].any()][:2];rows=[]
            for s in selected:
                hits=np.flatnonzero(s['mask'].any(1));n=min(len(s['x']),max(64,int(hits[0])+1))
                rows.append({**s,**{k:s[k][:n] for k in ('x','y','mask','gi')}})
            val[rule]=[{k:v.to(rt['device']) if k!='lengths' else v for k,v in batch.items()} for batch in batches(rows,'cpu')]
        model=create_model(code,rt,1701).to(rt['device'])
        for name,expected in [('short_best.pt',C.load(on/'observer.json')['short_val_loss']),('best.pt',C.load(on/'complete.json')['best_val_loss'])]:
            model.load_state_dict(torch.load(on/name,map_location=rt['device'],weights_only=True));value,_=T.validate(model,val)
            assert np.isclose(value,expected,rtol=1e-6,atol=1e-7),'Snapshot reload mismatch'
        del prepared,val,model;torch.cuda.empty_cache()
    C.save(C.CACHE/'checks.json',{'all_pass':True,'sources':C.sources(),'observer_tolerance':[1e-6,1e-7],'real_data':True,'exact_equality':results})
def dispatch():
    pending=list(C.JOBS);active={};done=[]
    for seed,code in list(pending):
        rd=C.run_dir('confirm',seed,code)
        if (rd/'process_exit.json').exists() and C.load(rd/'process_exit.json')['returncode']==0:
            C.verify_run('confirm',seed,code);done.append((seed,code));pending.remove((seed,code))
        else:recover('confirm',seed,code,True)
    failure=None
    while pending or active:
        assert C.remaining()>0,'B104 suite deadline reached'
        for key,e in list(active.items()):
            p=e['process'];timeout=time.time()-e['meta']['started']>e['hard']
            if timeout and p.poll() is None:p.kill()
            if p.poll() is not None:
                try:finish(e,timeout);done.append(key)
                except Exception as err:
                    try:
                        assert p.returncode!=0,'Artifact/validation failure is not an external interruption'
                        assert not timeout,'Parent hard timeout is not an external interruption'
                        recover('confirm',*key,True);pending.append(key)
                    except Exception:failure=failure or repr(err)
                del active[key]
        if failure:pending=[]
        while pending and len(active)<4 and not failure:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if free<1200:
                assert active,'Insufficient VRAM; no unrelated process stopped';break
            key=pending.pop(0);active[key]=start('confirm',*key,True)
        status('training',completed=len(done),total=6,active=[e['meta'] for e in active.values()],failure=failure)
        if active:time.sleep(1)
    if failure:raise RuntimeError(failure)
    assert len(done)==6
def all_ready():
    check=C.load(C.CACHE/'checks.json');assert check['all_pass'] and check['sources']==C.sources()
    return {f'{s}__{c}':C.verify_run('confirm',s,c) for s,c in C.JOBS}
def npz(p):
    import numpy as np
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def pred_path(seed,code,budget,rule):return C.CACHE/'predictions'/f'{seed}__{C.LABEL[code]}__{budget}__{rule}.npz'
def predict():
    import numpy as np,torch
    import common92 as M, data88 as D
    from model92 import create_model
    all_ready();G,R=M.load_sources();lookup={int(g):i for i,g in enumerate(R['gi'])};rt=C.runtime()
    folder=C.CACHE/'predictions';folder.mkdir(exist_ok=True)
    for seed,code in C.JOBS:
        rd=C.run_dir('confirm',seed,code);cfg=C.load(rd/'config.json');stats=C.load(rd/'stats.json');manifest=C.load(rd/'manifest.json')
        short=torch.load(rd/'short_best.pt',map_location='cpu',weights_only=True);long=torch.load(rd/'best.pt',map_location='cpu',weights_only=True)
        identical=all(torch.equal(short[k],long[k]) for k in short);cached={}
        for budget,w in [('SHORT',short),('LONG',long)]:
            model=create_model(code,rt,seed).to(rt['device']);model.load_state_dict(w);model.eval()
            for rule in ('D10','D50'):
                if budget=='LONG' and identical:idx,est=cached[rule]
                else:
                    idx=[];est=[]
                    with torch.no_grad():
                        for seg in D.build_segments(manifest['test'],stats,rule,G,R):
                            y=model(torch.from_numpy(seg['x']).unsqueeze(0).to(rt['device']),torch.tensor([len(seg['x'])])).cpu().numpy()[0]*stats['ti_sd']+stats['ti_mu']
                            for i,g in enumerate(seg['gi']):
                                if int(g) in lookup:idx.append(lookup[int(g)]);est.append(y[i])
                    idx=np.asarray(idx,dtype=np.int64);est=np.asarray(est,dtype=np.float32);order=np.argsort(idx);idx,est=idx[order],est[order]
                assert len(idx) and np.all(np.diff(idx)>0) and est.shape==(len(idx),2) and np.isfinite(est).all()
                p=pred_path(seed,code,budget,rule);tmp=p.with_suffix('.tmp.npz');np.savez(tmp,idx=idx,est=est,config_hash=cfg['config_hash'],budget=budget,source_hash=C.digest(C.sources()));tmp.replace(p);cached[rule]=(idx,est)
            altered={**R,'tg_F':R['tg_F']+500,'tg_L5':R['tg_L5']+500}
            a=D.build_segments(manifest['test'][:1],stats,'D10',G,R);b=D.build_segments(manifest['test'][:1],stats,'D10',G,altered)
            assert len(a)==len(b) and all(np.array_equal(x['x'],y['x']) for x,y in zip(a,b))
            del model;torch.cuda.empty_cache()
        print('Predicted '+str(seed)+' '+C.LABEL[code],flush=True)
    C.save(C.CACHE/'prediction_checks.json',{'all_pass':True,'labels_not_inputs':True,'sources':C.sources(),'count':24})
def build_cells():
    import numpy as np,common92 as M
    _,R=M.load_sources();sets=npz(C.ROOT/'data/.b75/sets.npz');assert C.sha(C.ROOT/'data/.b75/sets.npz')==C.prov()['sets75_sha256']
    cells={};details={};references=[];hashes={}
    for rule in ('D10','D50'):
        arrays=[];reference=None
        for seed in C.SEEDS:
            versions=[]
            for budget,code in [('SHORT','TX'),('SHORT','TH'),('LONG','TX'),('LONG','TH')]:
                p=pred_path(seed,code,budget,rule);z=npz(p);hashes[p.name]=C.sha(p)
                cfg=C.load(C.run_dir('confirm',seed,code)/'config.json')
                assert str(z['config_hash'])==cfg['config_hash'] and str(z['source_hash'])==C.digest(C.sources()) and str(z['budget'])==budget
                idx=z['idx'];assert idx.ndim==1 and len(idx) and np.all(np.diff(idx)>0) and idx.min()>=0 and idx.max()<len(R['gi'])
                assert z['est'].shape==(len(idx),2) and np.isfinite(z['est']).all()
                if reference is None:reference=idx
                else:assert np.array_equal(reference,idx)
                versions.append(z['est'])
            arrays.append(versions)
        references.append(reference)
        values=np.asarray(arrays,dtype=np.float64)
        for head,target in enumerate(('F','L5')):
            truth=R['tg_'+target][reference];keep=sets['common_'+target][reference]&np.isfinite(truth);ii=reference[keep];name=rule+'/'+target
            cells[name]={'errors':values[:,:,keep,head]-truth[keep],'cluster':R['clus'][ii],'discharge':R['fidx'][ii]}
            details[name]={'rows':len(ii),'clusters':len(np.unique(R['clus'][ii])),'discharges':len(np.unique(R['fidx'][ii]))}
    assert len(hashes)==24 and len(list((C.CACHE/'predictions').glob('*.npz')))==24
    support=np.unique(np.concatenate(references));universe={f:np.unique(R[k][support]) for f,k in [('cluster','clus'),('discharge','fidx')]}
    return cells,universe,details,hashes
def readout():
    import stats104
    fits=all_ready();cells,universe,details,hashes=build_cells();stats=stats104.compute(cells,universe)
    report={'registration':'PREREGISTRATION_B104.md','exploratory':True,'unit':C.UNIT,'seeds':list(C.SEEDS),'sources':C.sources(),
        'provenance_sha256':C.sha(C.CACHE/'provenance.json'),'draws':20000,'seed':20261104,'chunk':500,'conditional_fixed_inits':True,
        'cells':{k:{**details[k],'statistics':v} for k,v in stats.items()},'fits':fits,'prediction_hashes':hashes,
        'limits':['One TEST-informed temporal unit only; all four previous validation designs remain in B94, not new evidence here.','Three inits fixed for primary cluster intervals; init-mixture intervals are supplementary.','F/L5 report labels, not independent physical truth.','SHORT depends on this actual concurrency/load; no convergence or architecture-superiority claim.']}
    C.save(C.CACHE/'readout.json',report)
    lines=['# B.104 matched-trajectory budget sensitivity','',*report['limits'],'','Point [95% conditional cluster interval]; no changed-significance inference.','',
        '| Cell | Quantity | Point | Per-init | Form | Fixed95% | Mixture95% | Undefined fixed/mix | 0 excluded fixed/mix |','|---|---|---|---|---|---|---|---|---|']
    for k,c in report['cells'].items():
        for q,v in c['statistics'].items():
            for form in ('cluster','discharge'):
                ci=v[form];lo,hi=ci['fixed'];ml,mh=ci['mixture']
                lines.append(f"| {k} | {q} | {v['point']:.6f} | {v['per_init']} | {form} | [{lo:.6f}, {hi:.6f}] | [{ml:.6f}, {mh:.6f}] | {ci['fixed_undefined']}/{ci['mixture_undefined']} | {ci['fixed_excludes_zero']}/{ci['mixture_excludes_zero']} |")
    (C.CACHE/'readout.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
def postreview():
    import scout104 as S
    report=C.load(C.CACHE/'readout.json');assert C.load(C.CACHE/'audit.json')['all_pass']
    evidence={k:report[k] for k in ('exploratory','cells','limits')}
    evidence.update({'scope':'One fixed previously proposed temporal unit','initialization_count':3,'fit_summaries':[{'model':C.LABEL[k.split('__')[1]],**{n:v[n] for n in ('epochs','stop','short_epoch','best_val_loss','short_val_loss')}} for k,v in report['fits'].items()]})
    assert C.load(C.CACHE/'audit.json')['report_sha256']==C.sha(C.CACHE/'readout.json')
    for stage,model in [('result_review','claude-fable-5-2'),('result_challenge','claude-opus-5-5')]:
        prompt='Review audited exploratory matched-trajectory training-budget results in Korean, at most700 words. No new training or novelty/acceptance/physics claims. Fixed-init intervals conditional on3initializations; report all4cells. Delta is LONG fast skill minus SHORT fast skill, no significance-change argument. Same-unit source curves chose scope after TEST, no independent confirmation. Give supported conclusion, falsifier, at most2 bounded next proposals.\n'+json.dumps(evidence)
        if stage=='result_challenge':prompt+='\nPrior review:\n'+(C.CACHE/'result_review.txt').read_text(encoding='utf8')
        rp=C.CACHE/'claude'/(stage+'.response.json');qp=C.CACHE/'claude'/(stage+'.request.json')
        if rp.exists():
            assert C.load(qp)['messages'][0]['content']==prompt;r=C.load(rp)['response']
        else:r=S.call(stage,model,prompt)
        assert r.get('stop_reason')=='end_turn','Incomplete post review'
        (C.CACHE/(stage+'.txt')).write_text('\n'.join(b['text'] for b in r['content'] if b['type']=='text'),encoding='utf8')
    text='# B.104 Claude post-evaluation review\n\n'+(C.CACHE/'result_review.txt').read_text(encoding='utf8')+'\n\n## Adversarial review\n\n'+(C.CACHE/'result_challenge.txt').read_text(encoding='utf8')
    (C.ROOT/'docs/paper/b104_claude.md').write_text(text,encoding='utf8')
def run():
    status('starting');C.guard();status('smoke');checks();status('training');dispatch();all_ready()
    status('prediction');predict();status('readout');readout();status('audit')
    import audit104;audit104.run()
    assert C.load(C.CACHE/'audit.json')['report_sha256']==C.sha(C.CACHE/'readout.json')
    for source,dest in [('readout.json','b104_budget.json'),('readout.md','b104_budget.md')]:
        shutil.copyfile(C.CACHE/source,C.ROOT/'docs/paper'/dest);assert C.sha(C.CACHE/source)==C.sha(C.ROOT/'docs/paper'/dest)
    status('post_review');postreview();status('complete')
def register():
    import durable93 as D
    C.guard();assert not (C.CACHE/'suite_clock.json').exists(),'Existing clock: inspect, never reset'
    now=time.time();deadline=now+10800;C.save(C.CACHE/'suite_clock.json',{'started':now,'deadline':deadline,'registration':'PREREGISTRATION_B104.md'})
    name='B93_GPU_B104_budget';watch='B93_Research_B104';pythonw=Path(sys.executable).with_name('pythonw.exe');q=D.quote_ps
    ui={k:os.environ[k] for k in ('CVMUX','CVMUX_SESSION_ID','CVMUX_SOCKET_PATH','CVMUX_SOCKET_PASSWORD') if k in os.environ};assert ui.get('CVMUX_SOCKET_PASSWORD')
    C.save(C.CACHE/'watch_config.json',{'started':now,'deadline':deadline,'ui':ui,'notify_bin':shutil.which('cvmux.cmd'),'task_name':name})
    args=['-u',str(C.HERE.parent/'b99_recovery/monitor99.py'),'supervise','--queue',str(C.HERE/'pipeline104.py'),'--cache',str(C.CACHE),'--deadline',str(deadline),'--queue-arg','run']
    for task,arguments in [(name,args),(watch,['-u',str(C.HERE/'watch104.py')])]:
        script=f"""$ErrorActionPreference='Stop'
if(Get-ScheduledTask -TaskName '{task}' -ErrorAction SilentlyContinue){{throw 'Task already registered'}}
$a=New-ScheduledTaskAction -Execute {q(pythonw)} -Argument {q(subprocess.list2cmdline(arguments))} -WorkingDirectory {q(C.ROOT)}
$s=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds 11000) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$p=New-ScheduledTaskPrincipal -UserId ($env:USERDOMAIN+'\\'+$env:USERNAME) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName '{task}' -Action $a -Settings $s -Principal $p | Out-Null
"""
        r=D.powershell(script);assert r.returncode==0,r.stderr
    C.save(C.CACHE/'durable_registration.json',{'task_name':name,'registered':True,'command':[str(pythonw),*args],'deadline':deadline,'restart_on_failure':0})
    r=D.powershell("Start-ScheduledTask -TaskName '"+watch+"'");assert r.returncode==0
    for _ in range(30):
        time.sleep(1);p=C.CACHE/'watch_status.json'
        if p.exists() and C.load(p).get('notify_returncode')==0:break
    else:raise RuntimeError('No delivered monitor notification; training not launched')
    D.start(name,C.CACHE);print(json.dumps({'started_task':name,'watch_task':watch,'deadline':deadline}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','register','run','checks','predict','readout','postreview','status'));a=p.parse_args()
    try:
        if a.action=='prepare':C.prepare()
        elif a.action=='postreview':postreview()
        elif a.action=='status':print(json.dumps(snapshot(C.CACHE)))
        else:globals()[a.action]()
    except BaseException as e:
        for owned in list(OWNED.values()):
            proc=owned['process'];identity=process_identity(proc.pid)
            if proc.poll() is None and identity['alive'] is True and identity['birth']==owned['meta']['birth']:
                proc.kill()
                try:finish(owned,True)
                except Exception:pass
        if a.action not in ('prepare','status','register'):status('failed',error=repr(e))
        raise
