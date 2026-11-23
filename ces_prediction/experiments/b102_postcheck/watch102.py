"""Local phase/error notifications, bounded evaluation-only interruption recovery."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
import finish102 as F
sys.path.insert(0,str(F.HERE.parent/'b93_gpu'))
import durable93 as D
from status93 import snapshot,process_identity

def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,PermissionError,json.JSONDecodeError):return {}

def main():
    z=F.prepared();config=json.loads((F.CACHE/'watch_config.json').read_text());deadline=z['deadline']+7200
    # Task Scheduler lacks the originating local UI session environment.
    allowed={'CVMUX','CVMUX_SESSION_ID','CVMUX_SOCKET_PATH','CVMUX_SOCKET_PASSWORD'}
    assert set(config.get('cvmux_env',{}))<=allowed
    os.environ.update(config.get('cvmux_env',{}))  # Local ignored config; never log/send.
    budget=F.CACHE/'restart_budget.json'
    restarts=int(json.loads(budget.read_text())['used']) if budget.exists() else 0
    assert restarts in (0,1)
    if restarts==0:assert not (F.CACHE/'automatic_restart_prior').exists()
    last=None;last_note=0;fresh_after=float(config['launched_at']);stale_since=None
    required_logs={'durable.stderr.log','durable.stdout.log','readout.log','audit_post.log','artifact_validation.log','../.b99/monitor99.stderr.log'}
    assert set(config['log_offsets'])==required_logs
    def notify(message):
        nonlocal last_note
        record={'time':time.time(),'message':message,'restarts':restarts}
        F.G.save(F.CACHE/'watch_status.json',record)
        # cvmux is a local UI channel, never an outside message service.
        r=D.powershell('& '+D.quote_ps(config['notify_bin'])+' notify '+D.quote_ps(message))
        record['notify_returncode']=r.returncode;F.G.save(F.CACHE/'watch_status.json',record)
        last_note=time.time()
    while time.time()<deadline:
        state=snapshot(F.G.CACHE);claude=read(F.ROOT/'data/.claude_posteval_20261123/status.json')
        effective=state['effective_state'];phase=state.get('queue',{}).get('state')
        pause=any(p.stat().st_mtime>config['launched_at'] for p in (F.ROOT/'data/.author_pause').glob('pause_*.json'))
        if pause:notify('Research paused by author; automatic recovery disabled');return
        if min(state.get('supervisor',{}).get('updated',0),state.get('queue',{}).get('updated',0))<=fresh_after:
            if stale_since is None:stale_since=time.time();notify('Waiting for fresh evaluation status; no automatic restart on stale metadata')
            if time.time()-stale_since>=60:notify('Research STOPPED: no fresh evaluation status within60s');return
            time.sleep(15);continue
        stale_since=None;signature=(effective,phase,claude.get('state'))
        if signature!=last or time.time()-last_note>=300:
            notify('Research: evaluation='+str(effective)+' / '+str(phase)+'; Claude='+str(claude.get('state')))
            last=signature
        if effective=='interrupted' and restarts==0 and time.time()<z['deadline']:
            F.prepared()
            assert not state.get('supervisor_alive') and not state.get('queue_process_alive')
            error=str(state.get('queue',{}).get('error',''))
            supervisor=state.get('supervisor',{})
            if supervisor.get('error') or supervisor.get('returncode') not in (None,0):error+=' supervisor diagnostic'
            for name in required_logs:
                offset=config['log_offsets'][name]
                p=F.G.CACHE/name
                assert p.exists() and p.stat().st_size>=offset,'Missing/truncated diagnostic log; refuse restart'
                with p.open('rb') as f:f.seek(offset);tail=f.read()
                if b'Traceback' in tail or b'AssertionError' in tail or b'KeyError' in tail:error+=' new traceback'
            if error:notify('Evaluation interruption has diagnostic error; automatic restart refused: '+error[:180]);return
            old=F.CACHE/'automatic_restart_prior';old.mkdir(exist_ok=False)
            for name in ('supervisor_status.json','queue_status.json','durable.lock','queue.lock'):
                p=F.G.CACHE/name
                if p.exists():
                    if name.endswith('.lock'):assert process_identity(int(p.read_text().strip()))['alive'] is False
                    shutil.copy2(p,old/name)
                    if name.endswith('.lock'):p.unlink()
            restarts+=1;F.G.save(F.CACHE/'restart_budget.json',{'used':restarts,'time':time.time()})
            notify('Restarting interrupted evaluation-only postchecks once; no training or API replay')
            fresh_after=time.time()
            r=D.powershell("$ErrorActionPreference='Stop'; Start-ScheduledTask -TaskName B93_GPU_B94_postcheck102")
            assert r.returncode==0
            for _ in range(30):
                time.sleep(1);new=snapshot(F.G.CACHE)
                if new.get('supervisor_alive') and new.get('queue_process_alive') and min(new['supervisor']['updated'],new['queue']['updated'])>fresh_after:break
            else:notify('Automatic evaluation restart has no live queue; stopped');return
            # Never launch/replay Claude from automatic recovery; alert on its failure.
        elif effective in ('failed','deadline_exceeded','interrupted'):
            notify('Research STOPPED: '+str(state.get('queue',{}).get('error',effective))[:220]+'; preserved data/.b102 evidence');return
        elif claude.get('state')=='failed':
            notify('Claude STOPPED: '+str(claude.get('error'))[:220]+'; no paid request replay');return
        elif effective=='complete' and claude.get('state')=='draft_ready_for_lead_review':
            notify('Evaluation audit and Claude research draft completed; ready for lead review');return
        time.sleep(15)
    notify('Research status notifier reached its time limit')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        F.G.save(F.CACHE/'watch_error.json',{'time':time.time(),'error':repr(e)})
        cfg=read(F.CACHE/'watch_config.json')
        if cfg.get('notify_bin'):D.powershell('& '+D.quote_ps(cfg['notify_bin'])+' notify '+D.quote_ps('Research notifier failed: '+repr(e)[:180]))
        raise
