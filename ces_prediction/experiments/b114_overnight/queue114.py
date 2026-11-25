"""Single hidden B114 workflow. No retries, no background GPU concurrency."""
import os,sys,time,json,subprocess,traceback
from datetime import datetime,timezone
from pathlib import Path
import common114 as C
W=C.OUT/'workflow'
GPU_END=datetime(2026,11,25,23,0,tzinfo=timezone.utc).timestamp()
CPU_END=datetime(2026,11,26,0,0,tzinfo=timezone.utc).timestamp()
def status(state,**extra):
 C.save(W/'status.json',{'state':state,'updated':time.time(),**extra})
 if extra.get('phase')=='closeout':return
 text='# B114 overnight status\n\nState: '+state+'\n\n'+json.dumps(extra,indent=2)+'\n\nPlan: b114_overnight_plan.md; live receipts: data/.b114/workflow/status.json. Do not launch a duplicate or edit frozen scientific sources.\n'
 (C.ROOT/'docs/paper/b114_status.md').write_text(text,encoding='utf8')
def idle():
 result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 (W/'gpu_before.txt').write_text(result.stdout)
 used,util=map(lambda x:int(x.strip()),result.stdout.strip().splitlines()[0].split(','))
 assert used<1024 and util<10,'GPU occupied; no external process is stopped'
def stage(label,args,env,cpu=False):
 deadline=CPU_END if cpu else GPU_END
 assert time.time()<deadline,label+' missed registered boundary'
 with (W/(label+'.stdout.log')).open('xb') as out,(W/(label+'.stderr.log')).open('xb') as err:
  child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=C.ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  while child.poll() is None:
   status('running',phase=label,child_pid=child.pid,deadline=deadline)
   if time.time()>=deadline:
    child.kill();child.wait();C.save(W/(label+'_exit.json'),{'returncode':child.returncode,'deadline_exceeded':True});raise RuntimeError('Registered boundary: '+label)
   time.sleep(15)
 C.save(W/(label+'_exit.json'),{'returncode':child.returncode,'completed':time.time()})
 assert child.returncode==0,label+' failed; partials preserved, no automatic retry'
def smoke_replay():
 import numpy as np
 from catboost import CatBoostRegressor
 from audit114 import neural_predict,A
 from models114 import ScalarNet
 import torch
 _,x,*_=C.load_unit('U0');x=x[:512]
 base=C.OUT/'cat/smoke';m=CatBoostRegressor();m.load_model(str(base/'direct_s1701.cbm'))
 pred=m.predict(x,thread_count=1);replay=A.portable_predict(C.read(base/'direct_s1701.json'),x)
 assert np.allclose(pred,replay,atol=1e-5,rtol=1e-7)
 errors={'cat':float(np.max(np.abs(pred-replay)))}
 for arch in ('resnet128','ple128'):
  z=C.npz(C.OUT/'neural/smoke'/arch/'direct_s1701.npz')
  model=ScalarNet(len(z['x_mu']),arch,z.get('model.knots'))
  model.load_state_dict({k[6:]:torch.as_tensor(v) for k,v in z.items() if k.startswith('model.')});model.eval()
  with torch.no_grad():pred=model(torch.as_tensor(((x-z['x_mu'])/z['x_sd']).astype(np.float32))).numpy().astype(np.float64)*float(z['y_sd'])+float(z['y_mu'])
  replay=neural_predict(z,arch,x)
  assert np.allclose(pred,replay,atol=.01,rtol=1e-5)
  errors[arch]=float(np.max(np.abs(pred-replay)))
 C.save(W/'smoke_independent.json',{'all_pass':True,'max_abs_ev':errors,'real_TRAIN_rows':512})
def main():
 W.mkdir(parents=True,exist_ok=True)
 sources=C.source_guard();C.verify_inputs()
 with (W/'started.json').open('x') as f:json.dump({'pid':os.getpid(),'started':time.time(),'sources':sources,'GPU_end':GPU_END,'CPU_end':CPU_END},f)
 idle()
 env={k:v for k,v in os.environ.items() if not k.startswith('CES_')};env.update(C.ENV_PINS)
 env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 for family in ('cat','neural'):stage(family+'_smoke',[C.HERE/(family+'114.py'),'--smoke'],env)
 smoke_replay()
 for family in ('cat','neural'):stage(family+'_fit',[C.HERE/(family+'114.py'),'--fit'],env)
 cat=C.read(C.OUT/'cat/fit_complete.json');nn=C.read(C.OUT/'neural/fit_complete.json')
 assert cat['all_pass'] and nn['all_pass'] and cat['fits']==96 and nn['fits']==48
 assert cat['sources']==nn['sources']==sources
 C.save(C.OUT/'all_fits_complete.json',{'all_pass':True,'fits':144,'sources':sources,'completed':time.time(),'cat_receipt_sha256':C.sha(C.OUT/'cat/fit_complete.json'),'neural_receipt_sha256':C.sha(C.OUT/'neural/fit_complete.json')})
 stage('selection',[C.HERE/'selection114.py'],env)
 for family in ('cat','neural'):stage(family+'_predict',[C.HERE/(family+'114.py'),'--predict'],env)
 stage('independent_audit',[C.HERE/'audit114.py'],env,cpu=True)
 assert C.read(C.OUT/'audit.json')['all_pass']
 stage('closeout',[C.HERE/'close114.py'],env,cpu=True)
 C.save(W/'status.json',{'state':'complete','fits':144,'independent_audit':True,'final_commit':C.read(W/'closeout.json').get('commit'),'updated':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  W.mkdir(parents=True,exist_ok=True);status('failed',error=repr(exc),traceback=traceback.format_exc(),no_retry=True)
  raise
