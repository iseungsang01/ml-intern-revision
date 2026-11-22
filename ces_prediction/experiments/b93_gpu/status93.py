"""Read-only process-identity and heartbeat status; a stale phase is not progress."""
import argparse,ctypes,json,os,time
from pathlib import Path

def process_identity(pid):
    if not pid or int(pid)<=0:return {'alive':False,'birth':None}
    if os.name!='nt':return {'alive':Path('/proc',str(pid)).exists(),'birth':None}
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.OpenProcess.restype=ctypes.c_void_p
    k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    k.GetProcessTimes.argtypes=[ctypes.c_void_p,*([ctypes.POINTER(ctypes.c_ulonglong)]*4)];k.CloseHandle.argtypes=[ctypes.c_void_p]
    h=k.OpenProcess(0x1000,False,int(pid))
    if not h:return {'alive':None if ctypes.get_last_error()!=87 else False,'birth':None}
    try:
        code=ctypes.c_ulong();ok=k.GetExitCodeProcess(h,ctypes.byref(code))
        times=[ctypes.c_ulonglong() for _ in range(4)]
        got=k.GetProcessTimes(h,*[ctypes.byref(t) for t in times])
        return {'alive':bool(code.value==259) if ok else None,'birth':str(times[0].value) if got else None}
    finally:k.CloseHandle(h)

def read(path):
    try:return json.loads(Path(path).read_text(encoding='utf8'))
    except FileNotFoundError:return None

def snapshot(cache,stale_seconds=20):
    cache=Path(cache);sup=read(cache/'supervisor_status.json');queue=read(cache/'queue_status.json');now=time.time()
    result={'observed_at':now,'cache':str(cache),'supervisor':sup,'queue':queue}
    if sup:
        identity=process_identity(sup.get('pid'));same=identity['alive'] is True and (sup.get('birth') is None or identity['birth']==sup['birth'])
        age=max(0,now-sup['updated']);terminal=sup['state'] in ('complete','failed','deadline_exceeded','interrupted')
        result.update(supervisor_alive=same,heartbeat_age_seconds=age,heartbeat_stale=age>stale_seconds)
        result['effective_state']=sup['state'] if terminal else 'interrupted' if not same else 'stale_heartbeat' if age>stale_seconds else sup['state']
        if sup.get('queue_pid'):
            child=process_identity(sup['queue_pid']);result['queue_process_alive']=child['alive'] is True and (sup.get('queue_birth') is None or child['birth']==sup['queue_birth'])
    elif queue:
        identity=process_identity(queue.get('pid'));terminal=queue['state'] in ('complete','failed','stopped_no_advancement','blocked_runtime','blocked_dependency')
        result.update(queue_process_alive=identity['alive'],effective_state=queue['state'] if terminal or identity['alive'] is True else 'interrupted',note='No durable heartbeat; phase timestamp alone is not liveness evidence')
    else:result['effective_state']='not_started'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--stale-seconds',type=float,default=20);a=p.parse_args();print(json.dumps(snapshot(a.cache,a.stale_seconds),indent=2))
