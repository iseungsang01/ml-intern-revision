"""Independent local observer for continuation handoff and epoch progress."""
import json,os,time
import common107 as C
import durable93 as D
from status93 import snapshot
from ui105 import unprotect
def main():
    cfg=C.load(C.CACHE/'watch_config.json');os.environ.update(unprotect(cfg['ui_protected']));last=None;last_at=0
    def note(message):
        r=D.powershell('& '+D.quote_ps(cfg['notify_bin'])+' notify '+D.quote_ps(message))
        C.save(C.CACHE/'watch_status.json',{'updated':time.time(),'message':message,'notify_returncode':r.returncode})
    note('B107 monitor ready: validation-patience LSTM continuation; original GPU children preserved')
    while time.time()<cfg['deadline']+120:
        s=snapshot(C.CACHE);q=s.get('queue') or {};state=s['effective_state'];phase=q.get('state','waiting');signature=(state,phase,str(q.get('failure') or q.get('error') or ''))
        if signature!=last or time.time()-last_at>=300:
            active=[]
            for j in q.get('active',[]):
                epoch=''
                try:
                    rows=[json.loads(t) for t in (C.ROOT/j['log']).read_text(encoding='utf8',errors='replace').splitlines() if t.startswith('{')]
                    if rows:epoch=':e'+str(rows[-1]['epoch']+1)
                except (OSError,ValueError):pass
                active.append(j['case']+epoch)
            note('B107 '+state+'/'+phase+' patience_done='+str(q.get('completed','-'))+'/20 '+','.join(active));last=signature;last_at=time.time()
        if state in ('complete','failed','interrupted','deadline_exceeded'):
            note('B107 '+state+': '+str(q.get('error') or q.get('summary') or 'see data/.b107')[:180]);return
        time.sleep(15)
if __name__=='__main__':
    try:main()
    except BaseException as e:C.save(C.CACHE/'watch_error.json',{'updated':time.time(),'error':repr(e)});raise
