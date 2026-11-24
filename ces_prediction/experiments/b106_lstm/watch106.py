"""Independent15s local health observer; phase/failure and5min progress messages."""
import os,time,json
import common106 as C
import durable93 as D
from status93 import snapshot
from ui105 import unprotect
def main():
    cfg=C.load(C.CACHE/'watch_config.json');os.environ.update(unprotect(cfg['ui_protected']));last=None;last_at=0
    def note(msg):
        r=D.powershell('& '+D.quote_ps(cfg['notify_bin'])+' notify '+D.quote_ps(msg));C.save(C.CACHE/'watch_status.json',{'updated':time.time(),'message':msg,'notify_returncode':r.returncode})
    note('B106 monitor ready: original LSTM temperature branch + OOF correction, no Claude API')
    while time.time()<cfg['deadline']+120:
        s=snapshot(C.CACHE);q=s.get('queue') or {};state=s['effective_state'];phase=q.get('state','waiting');signature=(state,phase,str(q.get('error','')))
        if signature!=last or time.time()-last_at>=300:
            active=[]
            for j in q.get('active',[]):
                p=C.CACHE/j['relative_log'];epoch=''
                try:
                    rows=[json.loads(t) for t in p.read_text(encoding='utf8',errors='replace').splitlines() if t.startswith('{')]
                    if rows:epoch=':e'+str(rows[-1].get('epoch',-1)+1)
                except (OSError,ValueError):pass
                active.append(j['unit']+epoch)
            note('B106 '+state+'/'+phase+' completed='+str(q.get('completed','-'))+'/20 '+','.join(active));last=signature;last_at=time.time()
        if state in ('complete','failed','interrupted','deadline_exceeded'):
            note('B106 '+state+': '+str(q.get('error') or q.get('summary') or 'see data/.b106')[:200]);return
        time.sleep(15)
if __name__=='__main__':
    try:main()
    except BaseException as e:C.save(C.CACHE/'watch_error.json',{'error':repr(e),'updated':time.time()});raise
