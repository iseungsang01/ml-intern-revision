"""Observer/retriever/independent-audit only; never launches or retries fits."""
import contextlib,json,os,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CON=ROOT/'data/.colab_connection'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b93_gpu'));sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b105_merit'))
import durable93 as D
from ui105 import unprotect
from colab_b111_operator import poll,STATE

def main():
    config=json.loads((ROOT/'data/.b106/watch_config.json').read_text());os.environ.update(unprotect(config['ui_protected']));os.environ['CUDA_VISIBLE_DEVICES']='-1'
    shared={'pid':os.getpid(),'state':'observing','started':time.time()};stop=threading.Event()
    def heartbeat():
        while not stop.wait(3):
            part=ROOT/'data/colab_b111/b111_results.zip.part';D.save(CON/'b111_watch.json',{**shared,'updated':time.time(),'downloaded_bytes':part.stat().st_size if part.exists() else None})
    threading.Thread(target=heartbeat,daemon=True).start()
    def note(msg):
        r=D.powershell('& '+D.quote_ps(config['notify_bin'])+' notify '+D.quote_ps(msg));D.save(CON/'b111_notification.json',{'time':time.time(),'message':msg,'returncode':r.returncode})
    last=None;last_at=0;errors=0
    try:
        note('B111 observer ready: OOF inference, eight correction fits, all-scenario evaluation on Colab')
        while time.time()-shared['started']<14400:
            try:
                poll();p=json.loads((CON/'b111_progress.json').read_text());errors=0
            except Exception as error:
                errors+=1;shared.update(state='observer_error',error=repr(error));note('B111 observer issue: '+repr(error)[:150])
                if errors>=3:raise
                time.sleep(15);continue
            shared.update(state=p.get('state','unknown'),completed=p.get('completed',p.get('completed_cells')),total=p.get('total',p.get('total_cells')))
            phase=(shared['state'],shared['completed'],p.get('failure'))
            if phase!=last or time.time()-last_at>=300:
                note('B111 Colab '+str(shared['state'])+' '+str(shared['completed'])+'/'+str(shared['total']));last=phase;last_at=time.time()
            if p.get('failure'):raise RuntimeError('Scientific failure; no restart: '+str(p['failure']))
            if json.loads(STATE.read_text()).get('downloaded'):
                shared['state']='independent_audit';r=subprocess.run([sys.executable,str(ROOT/'ces_prediction/experiments/b111_full_compare/audit111.py')],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
                assert r.returncode==0,'Independent audit failed; preserve artifacts, no automatic retry'
                shared['state']='complete';note('B111 complete: eight OOF correction models trained and all32 scenario cells independently audited.');return
            time.sleep(30)
        raise RuntimeError('Observer lifetime reached; no scientific restart')
    finally:stop.set();D.save(CON/'b111_watch.json',{**shared,'updated':time.time(),'observer_exited':True})

if __name__=='__main__':
    with (CON/'b111_watch.stdout.log').open('a',encoding='utf8',buffering=1) as out,(CON/'b111_watch.stderr.log').open('a',encoding='utf8',buffering=1) as err:
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):main()
