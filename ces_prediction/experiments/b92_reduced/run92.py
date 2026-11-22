"""Supervise real child exits; a timeout or partial child cannot count complete."""
import argparse,json,os,subprocess,sys,time
import common92 as M
from train92 import ENV_PINS,effective_env,verify_complete
from runtime92 import decision
from adapter88 import acquire_file_lock,release_file_lock
def suite_remaining():
    p=M.CACHE/'suite_clock.json';z=json.loads(p.read_text());assert z['sources']==M.sources()
    return max(0,86400-(time.time()-z['started']))
def start_clock():
    p=M.CACHE/'suite_clock.json'
    if p.exists():assert json.loads(p.read_text())['sources']==M.sources()
    else:M.save(p,{'started':time.time(),'sources':M.sources()})
def run_job(job):
    stage,u,seed,code=job;M.guard();runtime=decision();rd=M.run_dir(*job)
    if (rd/'complete.json').exists():assert verify_complete(rd);return
    assert not (rd/'process_exit.json').exists(),'Previous failed process: no blind retry'
    # Raw unit/rule preparation is outside the child soft+120 timeout, while
    # remaining inside the suite's global 24h clock. Child only reloads this cache.
    from prepare92 import prepare_unit
    prep_started=time.perf_counter();prepared=prepare_unit(u);prep_elapsed=time.perf_counter()-prep_started
    M.save(rd/'runner_preparation.json',{'seconds':prep_elapsed,'cache_key':prepared['key'],'sources':M.sources()})
    del prepared
    if runtime['device']=='cuda':
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        assert int(free.splitlines()[0])>=2500,'Insufficient free memory; no fallback'
    remaining=suite_remaining();assert remaining>0,'24h suite wall cap reached'
    budget=120 if stage.startswith('smoke') else 600 if stage=='pilot' else 1200
    timeout=min(budget if stage.startswith('smoke') else budget+120,remaining);rd.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('CES_')};env.update(effective_env(stage,code,runtime));env['PYTHONUNBUFFERED']='1'
    if runtime['device']=='cpu':env['CUDA_VISIBLE_DEVICES']='-1'
    command=[sys.executable,'-X','faulthandler','-u',str(M.HERE/'train92.py'),'--stage',stage,'--unit',u,'--seed',str(seed),'--code',code]
    started=time.time();timed_out=False
    with (rd/'process.log').open('a',encoding='utf8') as log:
        child=subprocess.Popen(command,cwd=M.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill();rc=child.wait();timed_out=True # only this owned bounded child
    cfg=json.loads((rd/'config.json').read_text()) if (rd/'config.json').exists() else {}
    M.save(rd/'process_exit.json',{'returncode':rc,'config_hash':cfg.get('config_hash'),'timeout':timed_out,'elapsed':time.time()-started,'pid':child.pid})
    assert rc==0 and not timed_out,'Training child failed or exceeded hard budget'
    assert verify_complete(rd)
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('smoke','smoke_confirm','pilot','confirm'),required=True);p.add_argument('--plan',action='store_true');a=p.parse_args()
    M.guard()
    if a.stage=='smoke':jobs=[('smoke',M.PILOT_UNITS[0],1701,c) for c in ('C','A','T')]
    elif a.stage=='smoke_confirm':jobs=[('smoke_confirm',M.PILOT_UNITS[0],1701,c) for c in M.pair(M.selection()['chosen'])]
    elif a.stage=='pilot':jobs=M.pilot_jobs()
    else:
        from select92 import lock_selection
        z=lock_selection();assert z['chosen'],'No candidate advanced';jobs=M.confirmation_jobs(z['chosen'])
    if a.plan:print(json.dumps(jobs));return
    lock=M.CACHE/'batch.lock';assert acquire_file_lock(lock)
    try:
        for job in jobs:run_job(job)
    finally:release_file_lock(lock)
if __name__=='__main__':main()
