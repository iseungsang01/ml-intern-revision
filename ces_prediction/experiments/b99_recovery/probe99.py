"""Real Windows sharing locks and harmless timer process; no observations."""
import ctypes,json,subprocess,sys,time
import monitor99 as M
from pathlib import Path

def main():
    root=M.CACHE/'metadata_probe';root.mkdir(parents=True,exist_ok=True)
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_void_p]
    k.CreateFileW.restype=ctypes.c_void_p;k.CloseHandle.argtypes=[ctypes.c_void_p]
    def lock(p):
        h=k.CreateFileW(str(p.resolve()),0x80000000,0,None,3,0,None)
        assert h not in (None,ctypes.c_void_p(-1).value),ctypes.get_last_error()
        return h
    read=root/'queue_status.json';write=root/'supervisor_status.json'
    for p in (read,write):p.write_text(json.dumps({'timestamp':time.time()}),encoding='utf8')
    proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(14)'],creationflags=subprocess.CREATE_NO_WINDOW)
    checks={}
    h=lock(read)
    try:
        try:read.read_text();raise AssertionError('Real sharing lock did not deny original read')
        except PermissionError:checks['original_read_permission_failure_reproduced']=True
        M.MonitorPath.read_budget=.2
        started=time.monotonic();r=json.loads(M.MonitorPath(read).read_text())
        checks['bounded_read_degrades']=r['state']=='metadata_unavailable' and time.monotonic()-started<1
        checks['child_alive_during_read_loss']=proc.poll() is None
    finally:k.CloseHandle(h);M.MonitorPath.read_budget=10
    checks['read_recovers_after_release']='timestamp' in json.loads(M.MonitorPath(read).read_text())
    before=write.read_bytes();h=lock(write)
    try:
        M.save(write,{'timestamp':time.time(),'purpose':'timestamp-only plumbing'})
        checks['child_alive_after_write_loss']=proc.poll() is None
    finally:k.CloseHandle(h)
    checks['held_target_preserved']=write.read_bytes()==before
    checks['write_loss_reported']=json.loads((M.CACHE/'monitor_fallback_status.json').read_text())['kind']=='heartbeat_write_unavailable'
    M.save(write,{'timestamp':time.time(),'released':True})
    checks['write_recovers_after_release']=json.loads(write.read_text())['released']
    checks['harmless_child_normal_exit']=proc.wait(timeout=20)==0
    assert all(checks.values()),checks
    result={'all_pass':True,'checks':checks,'purpose':'Real OS metadata locks; no plasma observations or model training'}
    (M.CACHE/'probe99.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))

if __name__=='__main__':main()
