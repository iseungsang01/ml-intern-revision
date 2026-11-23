"""Existing local notifier, bound only to the publication batch/task."""
import json,sys,time
import publish103 as F
sys.path.insert(0,str(F.HERE.parent/'b102_postcheck'))
import watch102 as W
W.F=F
original=W.D.powershell
def powershell(script):
    return original(script.replace('B93_GPU_B94_postcheck102','B93_GPU_B94_publish103').replace('data/.b102 evidence','data/.b103 evidence'))
W.D.powershell=powershell
if __name__=='__main__':
    try:W.main()
    except BaseException as e:
        F.G.save(F.CACHE/'watch_error.json',{'time':time.time(),'error':repr(e)})
        c=W.read(F.CACHE/'watch_config.json')
        if c.get('notify_bin'):powershell('& '+W.D.quote_ps(c['notify_bin'])+' notify '+W.D.quote_ps('Research notifier failed: '+repr(e)[:180]))
        raise
