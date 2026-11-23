"""Independent local progress notifications, without automatic scientific retries."""
import os,time
import common105 as C
import durable93 as D
from status93 import snapshot

def main():
    from ui105 import unprotect
    cfg=C.load(C.CACHE/'watch_config.json');ui=unprotect(cfg['ui_protected']);assert set(ui)<=set(('CVMUX','CVMUX_SESSION_ID','CVMUX_SOCKET_PATH','CVMUX_SOCKET_PASSWORD'))
    os.environ.update(ui);last=None;last_at=0
    def note(msg):
        r=D.powershell('& '+D.quote_ps(cfg['notify_bin'])+' notify '+D.quote_ps(msg))
        C.save(C.CACHE/'watch_status.json',{'updated':time.time(),'message':msg,'notify_returncode':r.returncode})
    note('B105 merit study monitor ready: fair CPU references then bounded GPU screening')
    while time.time()<cfg['deadline']+120:
        s=snapshot(C.CACHE);q=s.get('queue') or {};phase=q.get('state','waiting');state=s['effective_state']
        sig=(state,phase,str(q.get('error','')))
        if sig!=last or time.time()-last_at>=300:
            active=[]
            for j in q.get('active',[]):
                p=C.CACHE/j['relative_log'];epoch=''
                try:
                    for line in reversed(p.read_text(encoding='utf8',errors='replace').splitlines()):
                        if line.startswith('{') and '"epoch"' in line:
                            import json
                            epoch=':e'+str(json.loads(line)['epoch']+1);break
                except (OSError,ValueError):pass
                active.append(j['unit']+epoch)
            note('B105 '+state+'/'+phase+' done='+str(q.get('completed','-'))+' '+','.join(active));last=sig;last_at=time.time()
        if state in ('complete','failed','interrupted','deadline_exceeded'):
            note('B105 '+state+': '+str(q.get('error') or q.get('summary') or 'see data/.b105 and docs/paper/b105_status.md')[:200]);return
        time.sleep(15)
    note('B105 study deadline reached; no automatic retry')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        C.save(C.CACHE/'watch_error.json',{'error':repr(e),'updated':time.time()});raise
