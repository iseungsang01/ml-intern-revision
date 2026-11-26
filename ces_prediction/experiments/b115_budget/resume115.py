"""Wait for idle GPU before the first scientific B115 fit; never retry fits."""
import os,time,json,subprocess,sys,traceback
from pathlib import Path
import common115 as C
import queue115 as Q
W=Q.W

def main():
 sources=C.source_guard();C.verify_inputs()
 assert C.read(C.OUT/'diagnostics/complete.json')['all_pass']
 original=C.read(W/'started.json');deadline=original['GPU_end'];cpu_end=original['CPU_end']
 failed=C.read(W/'status.json');assert failed['state']=='failed' and 'GPU occupied' in failed['error']
 assert not (C.OUT/'smoke').exists() and not (C.OUT/'fit_started.json').exists() and not (C.OUT/'prediction_complete.json').exists()
 for rel,h in original['sources'].items():assert C.sha(C.ROOT/rel)==h
 with (W/'resource_resume_started.json').open('x') as f:json.dump({'started':time.time(),'pid':os.getpid(),'sources':sources,'original_GPU_end':deadline},f)
 C.save(W/'initial_gpu_guard_failure.json',failed)
 env={k:v for k,v in os.environ.items() if not k.startswith('CES_')};env.update(C.ENV_PINS)
 env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 while time.time()<deadline:
  r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  used,util=[int(v.strip()) for v in r.stdout.strip().splitlines()[0].split(',')]
  apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True)
  C.save(W/'resource_probe.json',{'time':time.time(),'memory_used_MiB':used,'utilization_percent':util,'processes':apps.stdout,'returncode':apps.returncode})
  if used<1024 and util<10:break
  Q.status('waiting_for_gpu',memory_used_MiB=used,utilization_percent=util,deadline=deadline,scientific_fits_started=0)
  time.sleep(min(30,max(.1,deadline-time.time())))
 else:
  Q.status('blocked',reason='GPU never became idle before original boundary',scientific_fits_started=0);return
 assert time.time()<deadline
 Q.stage('smoke',[C.HERE/'train115.py','--smoke'],env,deadline);Q.smoke_check()
 Q.stage('fit',[C.HERE/'train115.py','--fit'],env,deadline)
 Q.stage('predict',[C.HERE/'train115.py','--predict'],env,deadline)
 Q.stage('audit',[C.HERE/'audit115.py'],env,cpu_end);assert C.read(C.OUT/'audit.json')['all_pass']
 Q.stage('closeout',[C.HERE/'close115.py'],env,cpu_end)
 C.save(W/'status.json',{'state':'complete','fits':12,'independent_audit':True,'resource_resume_only':True,'updated':time.time(),'commit':C.read(W/'closeout.json')['commit']})
if __name__=='__main__':
 try:main()
 except BaseException as e:Q.status('failed',error=repr(e),traceback=traceback.format_exc(),no_retry=True);raise
