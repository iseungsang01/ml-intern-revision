"""One bounded portfolio queue after B91 qualification; no implicit retries."""
import json,os,subprocess,sys,time
import common92 as M
from adapter88 import acquire_file_lock,release_file_lock
from runtime92 import decision
from run92 import start_clock,suite_remaining
def status(state,**extra):M.save(M.CACHE/'queue_status.json',{'state':state,'pid':os.getpid(),'updated':time.time(),**extra})
def phase(name,args):
    assert suite_remaining()>0,'24h suite cap'
    status(name)
    cmd=[sys.executable,'-u',str(M.HERE/args[0]),*args[1:]]
    proc=subprocess.Popen(cmd,cwd=M.ROOT)
    try:rc=proc.wait(timeout=suite_remaining())
    except subprocess.TimeoutExpired:
        # Runner owns a training child, so terminating its tree must preserve
        # clean-exit rejection. taskkill applies only this exact owned subtree.
        subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        proc.wait();raise RuntimeError('24h suite wall cap reached')
    assert rc==0,name+' failed; no blind retry'
def main():
    M.guard();lock=M.CACHE/'queue.lock';assert acquire_file_lock(lock)
    try:
        while not (M.DATA/'.b91/runtime_decision.json').exists():
            status('waiting_runtime');time.sleep(15)
        try:decision()
        except Exception as e:status('blocked_runtime',error=str(e));raise
        start_clock()
        from train92 import checks_ready
        try:checks_ready()
        except (AssertionError,FileNotFoundError,KeyError,json.JSONDecodeError):phase('readiness',['checks92.py'])
        phase('smoke',['run92.py','--stage','smoke'])
        phase('pilot',['run92.py','--stage','pilot'])
        from select92 import lock_selection
        selected=lock_selection()
        if selected['chosen'] is None:
            status('stopped_no_advancement');M.C.save_text(M.PAPER/'b92_status.md','# B.92 status\n\nPilot completed; no candidate passed the validation-only allocation gate. No test inference or scientific test score was computed.\n');return
        phase('selected_pair_smoke',['run92.py','--stage','smoke_confirm'])
        phase('confirmation',['run92.py','--stage','confirm'])
        phase('prediction',['predict92.py']);phase('readout',['readout92.py'])
        phase('independent_output_audit',['lead_checks/verify_outputs92.py']);status('complete')
    except Exception as e:
        if not (M.CACHE/'queue_status.json').exists() or json.loads((M.CACHE/'queue_status.json').read_text())['state']!='blocked_runtime':status('failed',error=str(e))
        raise
    finally:release_file_lock(lock)
if __name__=='__main__':main()
