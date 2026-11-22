"""One serial queue after clean B88 completion; never blindly retry failures."""
import os,subprocess,sys,time
import common90 as M
from adapter90 import dependency_ready,acquire_file_lock,release_file_lock
def status(state,**details):
    rec={'state':state,'pid':os.getpid(),'updated':time.time(),**details};M.save(M.CACHE/'queue_status.json',rec)
    M.C.save_text(M.PAPER/'b90_status.md','# B.90 queue status\n\n'+state+'\n\n'+str(details)+'\n')
def main():
    M.assert_registered();M.assert_frozen();lock=M.CACHE/'queue.lock';assert acquire_file_lock(lock),'B90 queue already active';phase='dependency'
    try:
        while True:
            try:ready=dependency_ready()
            except Exception as e:status('blocked_dependency',error=str(e));raise
            if ready:break
            status('waiting_dependency');time.sleep(30)
        phases=[('checks','checks90.py',[]),('smoke','run90.py',['--stage','smoke']),('screen','run90.py',['--stage','screen']),
                ('promote','run90.py',['--stage','promote']),('promoted','run90.py',['--stage','promoted']),('predict','predict90.py',[]),('readout','readout90.py',[]),
                ('independent_output_audit','lead_checks/verify_outputs90.py',[])]
        for phase,file,args in phases:
            M.assert_frozen();assert dependency_ready();status(phase)
            code=subprocess.run([sys.executable,'-u',str(M.HERE/file),*args],cwd=M.ROOT,env={**os.environ,'PYTHONUNBUFFERED':'1'}).returncode
            if code:raise RuntimeError(f'{phase} failed with code {code}; no blind retry')
        status('complete')
    except Exception as e:
        if phase!='dependency':status('failed',phase=phase,error=str(e))
        raise
    finally:release_file_lock(lock)
if __name__=='__main__':main()
