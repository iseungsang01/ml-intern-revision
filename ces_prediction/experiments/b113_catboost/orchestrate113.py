"""Hidden one-shot local GPU workflow: registered smoke -> fits -> independent audit."""
import os,sys,json,time,subprocess,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
import run113 as R
OUT=ROOT/'data/.b113_workflow'
def save(z):
 R.save(OUT/'status.json',dict(z,updated=time.time()))
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'started.json').open('x') as f:json.dump({'started':time.time(),'pid':os.getpid(),'sources':R.source_guard()},f)
 deadline=time.time()+4*3600
 env={k:v for k,v in os.environ.items() if not k.startswith('CES_')};env.update(R.ENV_PINS)
 env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 # Read-only idle gate; never stop another process.
 smi=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 (OUT/'gpu_before.txt').write_text(smi.stdout)
 used,util=[int(v.strip()) for v in smi.stdout.strip().splitlines()[0].split(',')]
 assert used<1024 and util<10,'GPU occupied; no fit launched'
 for label,args in [('smoke',[str(HERE/'run113.py'),'--smoke']),('fits',[str(HERE/'run113.py'),'--run']),('audit',[str(HERE/'audit113.py')])]:
  save({'state':'running','phase':label,'deadline':deadline})
  with (OUT/(label+'.stdout.log')).open('xb') as out,(OUT/(label+'.stderr.log')).open('xb') as err:
   proc=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
   while proc.poll() is None:
    save({'state':'running','phase':label,'child_pid':proc.pid,'deadline':deadline})
    if time.time()>deadline:
     proc.kill();proc.wait();raise RuntimeError('Owned child exceeded registered deadline; incomplete')
    time.sleep(5)
  R.save(OUT/(label+'_exit.json'),{'returncode':proc.returncode,'ended':time.time()})
  assert proc.returncode==0,label+' failed; preserve partials, no retry'
  if label=='smoke':
   import numpy as np
   from catboost import CatBoostRegressor
   from audit113 import portable_predict
   _,x,*_=R.B.load_unit('U0')
   m=CatBoostRegressor();m.load_model(str(R.OUT/'smoke/direct_s1701.cbm'))
   pred=m.predict(x[:512],thread_count=1)
   independent=portable_predict(R.C.read(R.OUT/'smoke/direct_s1701.json'),x[:512])
   assert np.allclose(pred,independent,rtol=1e-7,atol=1e-5),'Smoke portable tree mismatch'
   R.save(OUT/'smoke_portable_check.json',{'all_pass':True,'max_abs':float(np.max(np.abs(pred-independent)))})
 audit=R.C.read(ROOT/'data/.b113/audit.json');assert audit['all_pass']
 save({'state':'complete','independent_audit':True})
 (ROOT/'docs/paper/b113_status.md').write_text('# B113 complete\n\nAll registered fits and independent audit completed. Results: b113_comparison.json/md. Fixed-budget exploratory reused-TEST comparison; no automatic estimator adoption.\n',encoding='utf8')
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  OUT.mkdir(parents=True,exist_ok=True)
  save({'state':'failed','error':repr(exc),'traceback':traceback.format_exc(),'no_retry':True})
  (ROOT/'docs/paper/b113_status.md').write_text('# B113 incomplete\n\nWorkflow failed; inspect data/.b113_workflow/status.json and phase logs. No automatic retry; partials preserved.\n',encoding='utf8')
  raise
