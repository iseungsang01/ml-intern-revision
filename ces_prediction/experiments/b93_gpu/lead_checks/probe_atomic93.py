"""Actual Windows sharing-lock plumbing; timestamps only, no observations/models."""
import ctypes,json,sys,threading,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from atomic93 import save_json


def main():
    assert sys.platform=='win32'
    root=HERE.parents[3];directory=root/'data/.b93/atomic_plumbing'
    directory.mkdir(parents=True,exist_ok=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_void_p]
    kernel.CreateFileW.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    def lock(path):
        handle=kernel.CreateFileW(str(path),0x80000000,3,None,3,0,None)
        assert handle not in (None,ctypes.c_void_p(-1).value),ctypes.get_last_error()
        return handle
    path=directory/'timestamp.json';save_json(path,{'recorded_at':time.time()})
    handle=lock(path)
    def release():
        time.sleep(.4);assert kernel.CloseHandle(handle)
    thread=threading.Thread(target=release);thread.start()
    stamp=time.time();started=time.monotonic()
    save_json(path,{'recorded_at':stamp})
    release_elapsed=time.monotonic()-started;thread.join()
    assert .35<=release_elapsed<5 and json.loads(path.read_text())['recorded_at']==stamp
    preserved=path.read_bytes();handle=lock(path);started=time.monotonic();failed=False
    try:
        save_json(path,{'recorded_at':time.time()})
    except PermissionError:failed=True
    finally:kernel.CloseHandle(handle)
    failure_elapsed=time.monotonic()-started
    assert failed and 9.5<=failure_elapsed<15 and path.read_bytes()==preserved
    report={'all_pass':True,'purpose':'Actual Windows metadata replacement under real non-delete-sharing handles; no observations or models',
            'released_lock_elapsed_seconds':release_elapsed,'held_lock_failed_after_seconds':failure_elapsed,
            'target_preserved_on_failure':True,'checked_at':time.time()}
    save_json(directory/'checks.json',report);print(json.dumps(report))


if __name__=='__main__':main()
