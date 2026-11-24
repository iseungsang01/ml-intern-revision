"""Local observer/retriever only; never launches or retries scientific fits."""
import contextlib,json,os,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CON=ROOT/'data/.colab_connection';OUT=ROOT/'data/colab_b109'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b93_gpu'));sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b105_merit'))
import durable93 as D
from ui105 import unprotect
from colab_b109_operator import poll,STATE
def save(p,z):t=p.with_suffix('.tmp');t.write_text(json.dumps(z,indent=2),encoding='utf8');t.replace(p)
def main():
    config=json.loads((ROOT/'data/.b106/watch_config.json').read_text());os.environ.update(unprotect(config['ui_protected']));os.environ['CUDA_VISIBLE_DEVICES']='-1'
    shared={'pid':os.getpid(),'state':'waiting','started':time.time()};stop=threading.Event()
    def heartbeat():
        while not stop.wait(3):
            part=OUT/'b109_colab_results.zip.part';save(CON/'b109_watch.json',{**shared,'updated':time.time(),'downloaded_bytes':part.stat().st_size if part.exists() else None})
    threading.Thread(target=heartbeat,daemon=True).start()
    def note(msg):
        r=D.powershell('& '+D.quote_ps(config['notify_bin'])+' notify '+D.quote_ps(msg));save(CON/'b109_notification.json',{'time':time.time(),'message':msg,'returncode':r.returncode})
    note('B109 Colab observer ready; local GPU training remains paused');last=None;last_at=0;errors=0
    try:
        while time.time()-shared['started']<14400:
            if not STATE.exists() or 'poll_cell' not in json.loads(STATE.read_text()):time.sleep(5);continue
            try:
                poll();errors=0;p=json.loads((CON/'b109_progress.json').read_text());shared.update(state=p.get('state','unknown'),completed=p.get('completed'),total=p.get('total'))
                phase=(p.get('state'),p.get('completed'),p.get('failure'))
                if phase!=last or time.time()-last_at>=300:
                    active=','.join(e['tag']+':e'+str((e.get('progress') or {}).get('epoch','-')) for e in p.get('active',[]));note('B109 T4 '+str(p.get('state'))+' completed='+str(p.get('completed','-'))+'/9 '+active);last=phase;last_at=time.time()
                if json.loads(STATE.read_text()).get('downloaded'):
                    shared['state']='independent_audit';r=subprocess.run([sys.executable,str(ROOT/'ces_prediction/research/audit_colab_b109.py')],env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'},cwd=ROOT)
                    assert r.returncode==0,'Retrieved-state audit failed';shared['state']='complete';note('B109 complete:9 T4 paths audited,11 retained;20 selected checkpoints available. No TEST evaluation yet.');return
                if p.get('failure'):raise RuntimeError('Remote scientific failure; no retry: '+str(p['failure']))
            except Exception as error:
                errors+=1;shared.update(state='observer_error',error=repr(error));note('B109 observer issue: '+repr(error)[:160])
                if errors>=3:raise
            time.sleep(30)
        raise RuntimeError('Observer lifetime reached; scientific job not relaunched')
    finally:stop.set();save(CON/'b109_watch.json',{**shared,'updated':time.time(),'observer_exited':True})
if __name__=='__main__':
    CON.mkdir(parents=True,exist_ok=True)
    with (CON/'b109_watch.stdout.log').open('a',encoding='utf8',buffering=1) as out,(CON/'b109_watch.stderr.log').open('a',encoding='utf8',buffering=1) as err:
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):main()
