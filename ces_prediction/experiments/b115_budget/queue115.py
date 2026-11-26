"""Hidden sequential B115 diagnostics, GPU schedule experiment and audit."""
import os,sys,time,json,subprocess,traceback
from pathlib import Path
import common115 as C
W=C.OUT/'workflow';PY=Path(sys.executable).with_name('python.exe')
def status(state,**extra):
 C.save(W/'status.json',{'state':state,'updated':time.time(),**extra})
 if extra.get('phase')=='closeout':return
 (C.ROOT/'docs/paper/b115_status.md').write_text('# B115 status\n\n'+state+'\n\n'+json.dumps(extra,indent=2)+'\n\nPlan: b115_plan.md. Live: data/.b115/workflow/status.json. No duplicate or frozen-source edits.\n',encoding='utf8')
def stage(label,args,env,deadline):
 assert time.time()<deadline,'Registered boundary reached'
 with (W/(label+'.stdout.log')).open('xb') as out,(W/(label+'.stderr.log')).open('xb') as err:
  p=subprocess.Popen([str(PY),'-u',*map(str,args)],cwd=C.ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  while p.poll() is None:
   status('running',phase=label,child_pid=p.pid,deadline=deadline)
   if time.time()>=deadline:
    p.kill();p.wait();raise RuntimeError('Owned child exceeded registered boundary: '+label)
   time.sleep(10)
 C.save(W/(label+'_exit.json'),{'returncode':p.returncode,'completed':time.time()})
 assert p.returncode==0,label+' failed; preserve partials, no retry'
def smoke_check():
 import numpy as np,torch
 sys.path.insert(0,str(C.HERE.parent/'b114_overnight'))
 from models114 import ScalarNet
 from audit114 import neural_predict
 _,x,*_=C.load_unit('U0');x=x[:512];errors={}
 for arch in ('resnet128','ple128'):
  z=C.npz(C.OUT/'smoke'/arch/'direct_s1701.npz');m=ScalarNet(len(z['x_mu']),arch,z.get('model.knots'))
  m.load_state_dict({k[6:]:torch.as_tensor(v) for k,v in z.items() if k.startswith('model.')});m.eval()
  with torch.no_grad():p=m(torch.as_tensor(((x-z['x_mu'])/z['x_sd']).astype(np.float32))).numpy().astype(np.float64)*float(z['y_sd'])+float(z['y_mu'])
  q=neural_predict(z,arch,x);assert np.allclose(p,q,atol=.01,rtol=1e-5)
  errors[arch]=float(np.max(np.abs(p-q)))
 C.save(W/'smoke_independent.json',{'all_pass':True,'max_abs_ev':errors})
def main():
 W.mkdir(parents=True,exist_ok=True);sources=C.source_guard();C.verify_inputs();start=time.time();gpu_end=start+7200;cpu_end=gpu_end+1800
 with (W/'started.json').open('x') as f:json.dump({'pid':os.getpid(),'started':start,'sources':sources,'GPU_end':gpu_end,'CPU_end':cpu_end},f)
 env={k:v for k,v in os.environ.items() if not k.startswith('CES_')};env.update(C.ENV_PINS)
 env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 stage('diagnostics',[C.HERE/'diagnose115.py'],env,gpu_end)
 smi=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 (W/'gpu_before.txt').write_text(smi.stdout);used,util=[int(v.strip()) for v in smi.stdout.strip().splitlines()[0].split(',')]
 assert used<1024 and util<10,'GPU occupied; no other process stopped'
 stage('smoke',[C.HERE/'train115.py','--smoke'],env,gpu_end);smoke_check()
 stage('fit',[C.HERE/'train115.py','--fit'],env,gpu_end)
 stage('predict',[C.HERE/'train115.py','--predict'],env,gpu_end)
 stage('audit',[C.HERE/'audit115.py'],env,cpu_end);assert C.read(C.OUT/'audit.json')['all_pass']
 stage('closeout',[C.HERE/'close115.py'],env,cpu_end)
 C.save(W/'status.json',{'state':'complete','fits':12,'independent_audit':True,'updated':time.time(),'commit':C.read(W/'closeout.json')['commit']})
if __name__=='__main__':
 try:main()
 except BaseException as e:
  W.mkdir(parents=True,exist_ok=True);status('failed',error=repr(e),traceback=traceback.format_exc(),no_retry=True);raise
