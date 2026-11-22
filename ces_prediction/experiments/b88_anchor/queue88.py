"""Wait for the frozen B85 dependency, then run B88 phases without retries."""
import os,subprocess,sys,time
import common88 as M
from adapter88 import dependency_ready,acquire_file_lock,release_file_lock
from run88 import save
def status(state,**details):save(M.CACHE/'queue_status.json',{'state':state,'pid':os.getpid(),'updated':time.time(),**details})
def main():
    M.assert_frozen();M.CACHE.mkdir(parents=True,exist_ok=True)
    lock=M.CACHE/'queue.lock';assert acquire_file_lock(lock),'B88 queue already active'
    phase='dependency'
    try:
        while True:
            try:ready=dependency_ready()
            except Exception as e:
                status('blocked_dependency',error=str(e));raise
            if ready:break
            status('waiting_dependency');time.sleep(30)
        for phase,filename,args in [('smoke','run88.py',['--smoke']),('registered_batch','run88.py',[]),('predict','predict88.py',[]),('readout','readout88.py',[]),('independent_output_audit','lead_checks/verify_readout88.py',['--completed'])]:
            M.assert_frozen();status(phase)
            result=subprocess.run([sys.executable,'-u',str(M.HERE/filename),*args],cwd=M.ROOT,env={**os.environ,'PYTHONUNBUFFERED':'1'})
            if result.returncode:
                status('failed',phase=phase,returncode=result.returncode);raise RuntimeError(phase+' failed; no blind retry')
        status('complete')
    except Exception:
        if phase!='dependency':status('failed',phase=phase)
        raise
    finally:release_file_lock(lock)
if __name__=='__main__':main()
