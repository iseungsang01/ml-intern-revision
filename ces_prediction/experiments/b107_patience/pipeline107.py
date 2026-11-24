"""Process-bound handoff, four workers and patience-qualified final readout."""
import argparse,ctypes,json,os,shutil,subprocess,sys,time
from pathlib import Path
import common107 as C
from status93 import process_identity,snapshot
import durable93 as D

def status(state,**more):C.save(C.CACHE/'queue_status.json',{'state':state,'pid':os.getpid(),'birth':process_identity(os.getpid())['birth'],'updated':time.time(),**more})
class Observed:
    def __init__(self,pid,birth,access=0x100000|0x1000):
        self.pid=int(pid);self.birth=birth;self.k=ctypes.WinDLL('kernel32',use_last_error=True)
        self.k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];self.k.OpenProcess.restype=ctypes.c_void_p
        self.k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        self.k.CloseHandle.argtypes=[ctypes.c_void_p]
        self.handle=self.k.OpenProcess(access,False,self.pid);assert self.handle,'Cannot open owned process '+str(pid)
        assert process_identity(pid)['birth']==birth,'Process birth changed'
    def poll(self):
        x=ctypes.c_ulong();assert self.k.GetExitCodeProcess(self.handle,ctypes.byref(x));return None if x.value==259 else x.value
    def close(self):
        if self.handle:self.k.CloseHandle(self.handle);self.handle=None

def handoff():
    C.guard();assert not (C.CACHE/'handoff.json').exists(),'Handoff already attempted; inspect'
    q=C.load(C.DATA/'.b106/queue_status.json');assert q['state']=='training','Expected original training group'
    owner=Observed(q['pid'],q['birth'],0x100000|0x1000|0x800|1);assert owner.poll() is None
    nt=ctypes.WinDLL('ntdll');nt.NtSuspendProcess.argtypes=[ctypes.c_void_p];nt.NtSuspendProcess.restype=ctypes.c_long
    nt.NtResumeProcess.argtypes=[ctypes.c_void_p];nt.NtResumeProcess.restype=ctypes.c_long
    owner.k.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint];suspended=False;killed=False;active=[]
    try:
        assert nt.NtSuspendProcess(owner.handle)==0;suspended=True
        for case in C.CASES:
            p=C.OLD.rd(case)
            if not (p/'child_start.json').exists() or (p/'process_exit.json').exists():continue
            m=C.load(p/'child_start.json');o=Observed(m['pid'],m['birth'])
            active.append({'observer':o,'path':p,'case':case,'origin':'B106','started':m['started'],'pid':m['pid'],'birth':m['birth'],'log':(p/'process.log').relative_to(C.ROOT).as_posix()})
        script=f"@(Get-CimInstance Win32_Process -Filter 'ParentProcessId={owner.pid}' | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
        r=D.powershell(script);assert r.returncode==0,r.stderr;children=json.loads(r.stdout) if r.stdout.strip() else []
        if isinstance(children,int):children=[children]
        assert set(children)<={e['pid'] for e in active},'Unaccounted child of original queue'
        record={'time':time.time(),'original_queue':{'pid':owner.pid,'birth':owner.birth},'original_phase':q,
                'adopted':[{k:e[k] for k in ('case','origin','started','pid','birth','log')} for e in active],
                'reason':'Author-authorized removal of one-hour scientific stopping cap; GPU children left running unchanged.'}
        C.save(C.CACHE/'handoff_intent.json',record)
        assert owner.k.TerminateProcess(owner.handle,0xB107);killed=True
        for _ in range(100):
            if owner.poll() is not None:break
            time.sleep(.05)
        assert owner.poll() is not None
        C.save(C.CACHE/'handoff.json',{**record,'original_queue_exit':owner.poll(),'gpu_children_terminated':0})
        C.save(C.DATA/'.b106/handoff_to_b107.json',{'time':time.time(),'replacement':str(C.CACHE),'reason':record['reason'],'original_queue_exit':owner.poll(),'not_a_scientific_fit_failure':True})
        (C.PAPER/'b106_status.md').write_text('# B106 orchestration superseded by B107\n\nAuthor requested continuation to validation patience. Original GPU children are adopted without stopping; new control/evaluation is under b107_status.md and data/.b107. Original code/configs/checkpoints remain preserved. Original supervisor queue exit is an intentional orchestration handoff, not a fit failure. No final B106 merit result is published.\n',encoding='utf8')
        D.powershell("Stop-ScheduledTask -TaskName 'B106_LSTM_watch' -ErrorAction SilentlyContinue")
        return active
    except BaseException:
        for e in active:e['observer'].close()
        raise
    finally:
        if suspended and not killed:nt.NtResumeProcess(owner.handle)
        owner.close()

def spawn(case,mode):
    p=C.new_rd(case);p.mkdir(parents=True,exist_ok=True);assert not (p/'child_start.json').exists()
    log=(p/'process.log').open('ab');env={k:v for k,v in os.environ.items() if not k.startswith('CES_')}
    env.update(C.env(case));env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    proc=subprocess.Popen([sys.executable,'-u',str(C.HERE/'train107.py'),'--case',case,'--mode',mode],cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    e={'proc':proc,'stream':log,'path':p,'case':case,'mode':mode,'origin':'B107','started':time.time(),'pid':proc.pid,'birth':process_identity(proc.pid)['birth'],'log':(p/'process.log').relative_to(C.ROOT).as_posix()}
    C.save(p/'child_start.json',{k:e[k] for k in ('case','mode','origin','started','pid','birth','log')});return e

def group(active):
    selected={};pending=[];failure=None
    def select_original(case):
        r=C.OLD.verify_run(case)
        if r['stop']=='converged':selected[case]={'kind':'original','path':C.OLD.rd(case).relative_to(C.ROOT).as_posix(),'best_sha256':r['best_sha256']}
        else:
            assert r['stop'] in ('budget_limited','epoch_cap');pending.append((case,'resume'))
    adopted={e['case'] for e in active}
    for case in C.CASES:
        if case in adopted:continue
        if (C.OLD.rd(case)/'process_exit.json').exists():select_original(case)
        else:assert not (C.OLD.rd(case)/'child_start.json').exists();pending.append((case,'fresh'))
    while active or pending:
        assert time.time()<C.load(C.CACHE/'suite_clock.json')['deadline'],'Emergency suite guard: incomplete'
        for e in list(active):
            timeout=e['origin']=='B107' and time.time()-e['started']>C.CHILD_SECONDS+600
            rc=e['observer'].poll() if e['origin']=='B106' else e['proc'].poll()
            if timeout and rc is None:e['proc'].kill();rc=e['proc'].wait()
            if rc is None:continue
            try:
                if e['origin']=='B107':e['stream'].close()
                cfg=C.load(e['path']/'config.json') if (e['path']/'config.json').exists() else {}
                C.save(e['path']/'process_exit.json',{'pid':e['pid'],'birth':e['birth'],'returncode':rc,'timeout':timeout,'config_hash':cfg.get('config_hash'),'elapsed':time.time()-e['started'],
                       'observer':'B107 adopted Windows process handle' if e['origin']=='B106' else 'B107 parent Popen.wait'})
                assert rc==0 and not timeout,'Training child failed: '+e['case']
                if e['origin']=='B106':select_original(e['case'])
                else:
                    r=C.verify_new(e['case']);assert r['patience_met'],'Emergency guard hit; no adequate-training conclusion'
                    selected[e['case']]={'kind':'continued' if e['mode']=='resume' else 'fresh','path':e['path'].relative_to(C.ROOT).as_posix(),'best_sha256':r['best_sha256']}
            except Exception as err:failure=failure or repr(err)
            finally:
                if e['origin']=='B106':e['observer'].close()
                active.remove(e)
        if failure:pending=[]
        pending.sort(key=lambda x:(0 if x[1]=='resume' and x[0].endswith('__outer') else 1,C.CASES.index(x[0])))
        while pending and len(active)<4 and not failure:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if free<C.prov()['memory_reserve_MiB']:break
            case,mode=pending.pop(0);active.append(spawn(case,mode))
        status('training',completed=len(selected),total=20,pending=len(pending),active=[{k:e[k] for k in ('case','origin','started','pid','birth','log')} for e in active],failure=failure)
        if active or pending:time.sleep(1)
    if failure:raise RuntimeError(failure)
    assert set(selected)==set(C.CASES);C.save(C.CACHE/'selected.json',selected)

def step(action,gpu=False,timeout=7200):
    p=C.CACHE/'steps';p.mkdir(exist_ok=True);env={k:v for k,v in os.environ.items() if not k.startswith('CES_')};pins=C.env('all_cases');pins['CES_SEQ_DEVICE']='cuda' if gpu else 'cpu'
    env.update(pins);env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    C.save(p/(action+'.env.json'),{'env':pins,'sources':C.sources()})
    script='checks107.py' if action=='readiness' else 'analysis107.py';args=[] if action=='readiness' else [action]
    with (p/(action+'.log')).open('ab') as log:
        child=subprocess.Popen([sys.executable,'-u',str(C.HERE/script),*args],cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:child.kill();child.wait();raise RuntimeError('Step timeout '+action)
        assert rc==0,'Failed step '+action

def run():
    status('readiness');step('readiness',timeout=1800)
    status('handoff');active=handoff();group(active)
    for action,gpu in [('fits',False),('held',True),('correction_fit',False),('test',True),('correction_test',False),('readout',False),('audit',False)]:
        status(action);step(action,gpu)
    a=C.load(C.CACHE/'audit.json');assert a['all_pass'] and a['report_sha256']==C.sha(C.CACHE/'readout.json')
    for src,dst in [('readout.json','b107_lstm.json'),('readout.md','b107_lstm.md')]:shutil.copyfile(C.CACHE/src,C.PAPER/dst)
    (C.PAPER/'b107_status.md').write_text('# B107 complete\n\nAll twenty selected paths met validation patience. Optimizer-state continuation, OOF corrections, eight-scenario MSE and independent audit completed. Results b107_lstm.json/md. No additional fits or Claude/API calls queued.\n',encoding='utf8')
    status('complete',summary='Patience-qualified LSTM+post comparison independently audited')

def register():
    from ui105 import unprotect
    C.guard();assert not (C.CACHE/'suite_clock.json').exists();now=time.time();deadline=now+C.SUITE_SECONDS
    C.save(C.CACHE/'suite_clock.json',{'started':now,'deadline':deadline,'registration':'PREREGISTRATION_B107.md','emergency_only':True})
    old=C.load(C.DATA/'.b106/watch_config.json');assert unprotect(old['ui_protected'])
    C.save(C.CACHE/'watch_config.json',{'deadline':deadline,'ui_protected':old['ui_protected'],'notify_bin':old['notify_bin']})
    pythonw=Path(sys.executable).with_name('pythonw.exe');name='B107_patience_watch';q=D.quote_ps
    train_args=['-u',str(C.HERE.parent/'b99_recovery/monitor99.py'),'supervise','--queue',str(C.HERE/'pipeline107.py'),'--cache',str(C.CACHE),'--deadline',str(deadline),'--queue-arg','run']
    for task,arguments in [('B93_B107_patience',train_args),(name,['-u',str(C.HERE/'watch107.py')])]:
        args=subprocess.list2cmdline(arguments)
        script=f"""$ErrorActionPreference='Stop'
if(Get-ScheduledTask -TaskName '{task}' -ErrorAction SilentlyContinue){{throw 'Existing task'}}
$a=New-ScheduledTaskAction -Execute {q(pythonw)} -Argument {q(args)} -WorkingDirectory {q(C.ROOT)}
$s=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds {C.SUITE_SECONDS+300}) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$p=New-ScheduledTaskPrincipal -UserId ($env:USERDOMAIN+'\\'+$env:USERNAME) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName '{task}' -Action $a -Settings $s -Principal $p | Out-Null
"""
        rr=D.powershell(script);assert rr.returncode==0,rr.stderr
    C.save(C.CACHE/'durable_registration.json',{'registered':True,'task_name':'B93_B107_patience','command':[str(pythonw),*train_args],'deadline':deadline,'restart_on_failure':0})
    rr=D.powershell("Start-ScheduledTask -TaskName '"+name+"'");assert rr.returncode==0,rr.stderr
    for _ in range(30):
        time.sleep(1);p=C.CACHE/'watch_status.json'
        if p.exists() and C.load(p).get('notify_returncode')==0:break
    else:raise RuntimeError('Notifier unavailable; no handoff launched')
    D.start('B93_B107_patience',C.CACHE);print('B107 continuation controller launched',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','register','run','status'));a=p.parse_args()
    try:
        if a.action=='prepare':C.prepare()
        elif a.action=='status':print(json.dumps(snapshot(C.CACHE)))
        else:globals()[a.action]()
    except BaseException as err:
        if a.action=='run':status('failed',error=repr(err))
        raise
