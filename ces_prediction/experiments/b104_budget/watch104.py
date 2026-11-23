"""Separate local notifications and one guarded external-queue recovery."""
import json,os,shutil,sys,time
import common104 as C
import durable93 as D
from status93 import snapshot,process_identity

def main():
    cfg=C.load(C.CACHE/'watch_config.json');allowed={'CVMUX','CVMUX_SESSION_ID','CVMUX_SOCKET_PATH','CVMUX_SOCKET_PASSWORD'}
    assert set(cfg['ui'])<=allowed;os.environ.update(cfg['ui']);assert cfg['notify_bin']
    deadline=cfg['deadline'];last=None;last_note=0;grace=time.time();budget=C.CACHE/'restart_budget.json'
    used=C.load(budget)['used'] if budget.exists() else 0;assert used in (0,1)
    def note(message):
        nonlocal last_note
        z={'time':time.time(),'message':message,'restarts':used};C.save(C.CACHE/'watch_status.json',z)
        r=D.powershell('& '+D.quote_ps(cfg['notify_bin'])+' notify '+D.quote_ps(message));z['notify_returncode']=r.returncode
        C.save(C.CACHE/'watch_status.json',z)
        if r.returncode==0:last_note=time.time()
    note('B104 monitor ready; GPU study waits for source and real-data smoke gates')
    while time.time()<deadline+120:
        if any(p.stat().st_mtime>cfg['started'] for p in (C.ROOT/'data/.author_pause').glob('pause_*.json')):
            note('B104 author pause detected; no automatic restart');return
        s=snapshot(C.CACHE);q=s.get('queue') or {};sup=s.get('supervisor') or {};phase=q.get('state','not_started');state=s['effective_state']
        if min(q.get('updated',0),sup.get('updated',0))<=cfg['started']:
            if time.time()-grace>60:note('B104 no fresh study status within60s; inspect launch');return
            time.sleep(5);continue
        signature=(state,phase,str(q.get('failure') or ''))
        if signature!=last or time.time()-last_note>=300:
            active=[]
            for job in q.get('active',[]):
                name=C.LABEL[job['code']]+str(job['seed']);path=C.run_dir(job['stage'],job['seed'],job['code'],job['observe'])/'process.log'
                try:
                    lines=path.read_text(encoding='utf8',errors='replace').splitlines()
                    row=next((json.loads(x) for x in reversed(lines) if x.startswith('{') and '"epoch"' in x),{})
                    if row:name+=':e'+str(row['epoch']+1)
                except (OSError,ValueError):pass
                active.append(name)
            note('B104 '+state+'/'+phase+'; completed='+str(q.get('completed','-'))+'/6; '+','.join(active)+('; child failure, no new dispatch: '+str(q['failure'])[:120] if q.get('failure') else ''));last=signature
        if state in ('failed','interrupted','deadline_exceeded'):
            error=str(q.get('error') or q.get('failure') or sup.get('error') or '')
            live=any((lambda p:p['alive'] is None or p['alive'] is True and p['birth']==j['birth'])(process_identity(j['pid'])) for j in q.get('active',[]))
            diagnostics=False
            for name in ('durable.stderr.log',):
                p=C.CACHE/name
                if not p.exists():diagnostics=True
                else:
                    text=p.read_text(encoding='utf8',errors='replace');diagnostics|=any(v in text for v in ('Traceback','AssertionError','Nonfinite','nonfinite'))
            if used==0 and not error and not live and not diagnostics and phase in ('training','starting') and C.remaining()>4020:
                C.guard();dest=C.CACHE/'queue_external_attempt';assert not dest.exists();dest.mkdir()
                for name in ('queue_status.json','supervisor_status.json','durable.lock'):
                    p=C.CACHE/name
                    if p.exists():
                        if name.endswith('.lock'):assert process_identity(int(p.read_text().strip()))['alive'] is False
                        shutil.copy2(p,dest/name)
                        if name.endswith('.lock'):p.unlink()
                used=1;C.save(budget,{'used':used,'time':time.time()});cfg['started']=time.time();grace=time.time()
                note('B104 restarting externally interrupted queue once; registered incomplete attempts preserved; no API replay')
                r=D.powershell('Start-ScheduledTask -TaskName '+D.quote_ps(cfg['task_name']));assert r.returncode==0
                time.sleep(5);continue
            note('B104 STOPPED: '+(error or state)[:180]+'; evidence data/.b104 preserved');return
        if state=='complete':note('B104 complete: fit/readout audit and Claude post review finished');return
        time.sleep(15)
    note('B104 reached suite deadline; inspect final status')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        C.save(C.CACHE/'watch_error.json',{'time':time.time(),'error':repr(e)})
        try:
            c=C.load(C.CACHE/'watch_config.json');os.environ.update(c['ui'])
            D.powershell('& '+D.quote_ps(c['notify_bin'])+' notify '+D.quote_ps('B104 notifier failed: '+repr(e)[:160]))
        except Exception:pass
        raise
