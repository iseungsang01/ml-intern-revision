"""Quiesce controllers, preserve normal trainer exits, then start four slots."""
import ctypes,json,subprocess,sys,time
from pathlib import Path
import common96 as C
G=C.G
k=ctypes.WinDLL('kernel32',use_last_error=True);nt=ctypes.WinDLL('ntdll')
k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.OpenProcess.restype=ctypes.c_void_p
k.CloseHandle.argtypes=[ctypes.c_void_p];k.GetProcessTimes.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
k.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint]
for f in (nt.NtSuspendProcess,nt.NtResumeProcess):f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_long

def code(h):
    v=ctypes.c_ulong();assert k.GetExitCodeProcess(h,ctypes.byref(v));return v.value

def children(pid):
    script=f"Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    out=subprocess.check_output(['powershell','-NoProfile','-Command',script],text=True).strip()
    if not out:return []
    z=json.loads(out);items=z if isinstance(z,list) else [z]
    # Source/checkpoint verification may briefly own a git subprocess; only
    # a numerical trainer prevents the safe training boundary.
    return [p for p in items if 'bootstrap95.py' in (p.get('CommandLine') or '').lower() and ' train ' in (p.get('CommandLine') or '').lower()]

def main():
    sources,_=C.policy();plan=json.loads((C.CACHE/'transition_plan.json').read_text());handles={};parent_resumed=False;retired=False
    try:
        for pid,birth in plan['births'].items():
            pid=int(pid);h=k.OpenProcess(0x0800|0x1000|0x100000|1,False,pid);assert h,('Process missing',pid)
            times=[ctypes.c_ulonglong() for _ in range(4)];assert k.GetProcessTimes(h,*[ctypes.byref(t) for t in times]);assert str(times[0].value)==birth
            handles[pid]=h
        qp,sp=plan['old_queue'],plan['old_supervisor'];active=plan['active']
        print('Waiting for existing trainers to terminate normally',flush=True)
        while any(code(handles[a['pid']])==259 for a in active):
            assert time.time()<G.clock()['deadline'],'Original deadline reached'
            for a in active:
                created=(int(plan['births'][str(a['pid'])])-116444736000000000)/1e7
                hard=720 if plan['stage']=='pilot' else 1320
                assert code(handles[a['pid']])!=259 or time.time()-created<hard,'Original child hard deadline reached; restore controller'
            time.sleep(.2)
        assert all(code(handles[a['pid']])==0 for a in active),'Existing trainer failed; restore original controller'
        dirs=[G.CACHE/plan['stage']/a['run'] for a in active]
        assert all((rd/'complete.json').exists() for rd in dirs)
        assert nt.NtResumeProcess(handles[qp])==0;parent_resumed=True
        began=time.monotonic()
        while not all((rd/'process_exit.json').exists() for rd in dirs):
            assert time.monotonic()-began<60,'Attestation boundary unavailable'
            time.sleep(.02)
        assert nt.NtSuspendProcess(handles[qp])==0;parent_resumed=False
        assert not children(qp),'Another trainer raced the boundary; leave original queue running'
        from adapter94 import activate
        T=activate()
        for rd in dirs:
            assert T.verify_complete(rd)
            c=json.loads((rd/'config.json').read_text());C.OLD.dispatch_config(rd/'config.json',C.OLD.sources(),C.OLD.qualification())
        assert C.sources()==sources
        # All numerical children have already exited normally. Retire controllers only.
        assert k.TerminateProcess(handles[qp],0) and k.TerminateProcess(handles[sp],0);retired=True
        archive=(C.CACHE/'transition_archive').resolve();archive.mkdir(parents=True,exist_ok=True)
        for name in ('queue.lock','durable.lock'):
            src=(G.CACHE/name).resolve();dst=(archive/name).resolve()
            assert src.is_relative_to(G.CACHE.resolve()) and dst.is_relative_to(archive)
            if src.exists():assert not dst.exists();src.rename(dst)
        subprocess.run(['powershell','-NoProfile','-Command',f"Disable-ScheduledTask -TaskName '{plan['old_task']}' | Out-Null"],check=True)
        sys.path.insert(0,str(G.HERE.parent/'b93_gpu'));import durable93
        G.save(C.CACHE/'transition_complete.json',{'normal_exit_children':[a['pid'] for a in active],'sources':sources,'recorded_at':time.time(),'old_controllers_retired':True})
        print(json.dumps(durable93.start(plan['new_task'],G.CACHE)),flush=True)
    finally:
        if not retired:
            if 'qp' in locals() and not parent_resumed:nt.NtResumeProcess(handles[qp])
            if 'sp' in locals():nt.NtResumeProcess(handles[sp])
        for h in handles.values():k.CloseHandle(h)

if __name__=='__main__':main()
