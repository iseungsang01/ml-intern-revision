"""Registered-order 1/2-process allocation, separate RNG/exit evidence per fit."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from threading import Event,Thread
import common94 as G
import runtime94
from adapter94 import activate
_AUDITED_PREPARED=set()

def status(state,**extra):G.save(G.CACHE/'queue_status.json',{'pid':os.getpid(),'state':state,'updated':time.time(),**extra})
def command(action,*args):return [sys.executable,'-X','faulthandler','-u',str(G.HERE/'bootstrap94.py'),action,*map(str,args)]
def subprocess_phase(name,cmd):
    assert G.remaining()>0,'Original 24h deadline reached';status(name)
    with (G.CACHE/f'{name}.log').open('a',encoding='utf8') as log:
        child=subprocess.Popen(cmd,cwd=G.ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'})
        try:rc=child.wait(timeout=G.remaining())
        except subprocess.TimeoutExpired:
            child.kill();child.wait();raise RuntimeError(name+' exceeded original suite deadline')
    assert rc==0,name+' failed; no retries'
def audit(phase):
    target=G.HERE/'lead_checks/audit94.py';assert target.is_file(),'Independent audit is mandatory'
    subprocess_phase('audit_'+phase,[sys.executable,'-u',str(target),'--phase',phase,'--output',str(G.CACHE/'lead_checks'/f'{phase}.json')])
def jobs_for(stage,M):
    if stage=='smoke':return [('smoke',M.PILOT_UNITS[0],1701,c) for c in ('C','A','T')]
    if stage=='pilot':return M.pilot_jobs()
    selected=__import__('select92').lock_selection();assert selected['chosen'],'No validation advancement'
    if stage=='smoke_confirm':return [('smoke_confirm',M.PILOT_UNITS[0],1701,c) for c in M.pair(selected['chosen'])]
    return M.confirmation_jobs(selected['chosen'])

def dispatch(stage,runtime,T):
    import common92 as M
    from prepare92 import prepare_unit
    jobs=jobs_for(stage,M);limit=1 if stage.startswith('smoke') else runtime['parallel_workers']
    pending=list(enumerate(jobs));active={};failure=None;completed=[];failure_signal=Event()
    def wait_child(proc,hard,outcome,started):
        expired=False
        try:rc=proc.wait(timeout=hard)
        except subprocess.TimeoutExpired:proc.kill();rc=proc.wait();expired=True
        outcome.update(returncode=rc,timeout=expired,elapsed=time.time()-started,done=True)
        if rc or expired:failure_signal.set()
    def finish(slot,entry):
        proc,log,rd,started,hard,job,index,outcome=entry;rc=outcome['returncode'];log.close()
        cfg=json.loads((rd/'config.json').read_text()) if (rd/'config.json').exists() else {}
        G.save(rd/'process_exit.json',{'returncode':rc,'config_hash':cfg.get('config_hash'),'timeout':outcome['timeout'],'elapsed':outcome['elapsed'],'pid':proc.pid,'worker_slot':slot,'dispatch_index':index,'experiment':'B94'})
        assert rc==0 and not outcome['timeout'],'Failed scientific child: '+rd.name
        assert T.verify_complete(rd);completed.append(index)
    while pending or active:
        for slot,entry in list(active.items()):
            proc,_,_,started,hard,_,_,outcome=entry
            if outcome.get('done'):
                try:finish(slot,entry)
                except Exception as e:failure=failure or repr(e)
                del active[slot]
        if failure or failure_signal.is_set():pending=[] # In-flight owned children still finish normally.
        if pending and len(active)<limit:
            slot=next(s for s in range(limit) if s not in active);index,job=pending[0];stage_,unit,seed,code=job;rd=M.run_dir(*job)
            try:
                G.guard();assert G.remaining()>0,'Original suite deadline reached'
                if (rd/'complete.json').exists():
                    assert T.verify_complete(rd);completed.append(index);pending.pop(0);continue
                assert not (rd/'config.json').exists() and not (rd/'process_exit.json').exists(),'Prior incomplete/failed trajectory; no retry'
                # Unit lock in the adapter prevents partially published shared arrays.
                prep_start=time.perf_counter();prepared=prepare_unit(unit)
                prepared_identity=(prepared['key'],prepared['artifact_sha256'])
                G.save(rd/'runner_preparation.json',{'seconds':time.perf_counter()-prep_start,'cache_key':prepared['key'],'artifact_sha256':prepared['artifact_sha256'],'sources':G.sources()});del prepared
                if prepared_identity not in _AUDITED_PREPARED:
                    audit('prepared');_AUDITED_PREPARED.add(prepared_identity)
                free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if int(free.splitlines()[0])<2500:
                    assert active,'Insufficient free VRAM; no runtime fallback'
                    time.sleep(1);continue
                budget=120 if stage.startswith('smoke') else 600 if stage=='pilot' else 1200
                hard=min(budget if stage.startswith('smoke') else budget+120,G.remaining());assert hard>0
                assert not failure_signal.is_set(),'Another child failed; cease new dispatch'
                env={k:v for k,v in os.environ.items() if not k.startswith('CES_')};env.update(T.effective_env(stage,code,runtime));env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',B94_WORKER_SLOT=str(slot),B94_CONCURRENCY=str(limit))
                log=(rd/'process.log').open('a',encoding='utf8')
                proc=subprocess.Popen(command('train','--stage',stage,'--unit',unit,'--seed',seed,'--code',code),cwd=G.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                started=time.time();outcome={}
                active[slot]=(proc,log,rd,started,hard,job,index,outcome);pending.pop(0)
                Thread(target=wait_child,args=(proc,hard,outcome,started),daemon=True).start()
            except Exception as e:failure=failure or repr(e);pending=[]
        status(stage,qualified_workers=runtime['parallel_workers'],scheduled_concurrency=limit,active=[{'slot':s,'pid':e[0].pid,'run':e[2].name,'dispatch_index':e[6]} for s,e in active.items()],completed=len(completed),total=len(jobs),failure=failure)
        if active:time.sleep(1)
    if failure:raise RuntimeError(failure)
    assert len(completed)==len(jobs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');a=p.parse_args()
    G.guard();T=activate();import common92 as M
    if a.plan:
        print(json.dumps({'pilot':M.pilot_jobs(),'confirmation_max':46,'deadline':G.clock()['deadline'],'runtime_decision_required':True}));return
    from adapter88 import acquire_file_lock,release_file_lock
    lock=G.CACHE/'queue.lock';assert acquire_file_lock(lock),'A B94 queue already exists'
    try:
        runtime=runtime94.decision();G.CACHE.mkdir(parents=True,exist_ok=True)
        clock=G.clock();cp=G.CACHE/'suite_clock.json'
        if cp.exists():assert json.loads(cp.read_text())==clock,'Suite clock/source identity changed'
        else:G.save(cp,clock)
        assert G.remaining()>0
        audit('pre')
        try:T.checks_ready()
        except (AssertionError,FileNotFoundError,KeyError,json.JSONDecodeError):subprocess_phase('readiness',command('checks'))
        for stage in ('smoke','pilot'):dispatch(stage,runtime,T)
        selected=__import__('select92').lock_selection();audit('pilot')
        if selected['chosen'] is None:status('stopped_no_advancement');return
        for stage in ('smoke_confirm','confirm'):dispatch(stage,runtime,T)
        subprocess_phase('prediction',command('predict'));subprocess_phase('readout',command('readout'));audit('post')
        # Only independently audited new B94 reports may reach the paper folder.
        dest=G.ROOT/'docs/paper'
        for path in (G.CACHE/'reports').glob('b94_*'):
            assert path.suffix in ('.json','.md');shutil.copyfile(path,dest/path.name)
        status('complete')
    except BaseException as e:status('failed',error=repr(e));raise
    finally:release_file_lock(lock)
if __name__=='__main__':main()
