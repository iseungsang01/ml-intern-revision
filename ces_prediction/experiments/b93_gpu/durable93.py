"""One-shot Windows task/breakaway supervisor; never retry a scientific child."""
import argparse,base64,json,math,os,re,subprocess,sys,time
from pathlib import Path
from status93 import process_identity,snapshot
from atomic93 import save_json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PYTHON=Path(sys.executable).with_name('python.exe') if os.name=='nt' else Path(sys.executable)

def save(path,obj):
    return save_json(path,obj)
def safe_path(value):
    p=(ROOT/value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    assert p.is_relative_to(ROOT),'Path outside workspace';return p
def powershell(script):
    script="$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "+script
    encoded=base64.b64encode(script.encode('utf-16-le')).decode()
    return subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',encoded],encoding='utf-8',errors='replace',capture_output=True)
def quote_ps(s):return "'"+str(s).replace("'","''")+"'"
def deadline_default():
    z=json.loads((ROOT/'data/.b92/suite_clock.json').read_text());return float(z['started'])+86400
def kill_owned_tree(proc):
    if proc.poll() is None:
        if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        else:proc.kill()
        proc.wait(timeout=30)
def supervise(queue,cache,deadline,queue_args):
    queue=safe_path(queue);cache=safe_path(cache);assert queue.is_file()
    assert snapshot(cache).get('queue_process_alive') is not True,'An existing queue is alive; do not launch a duplicate'
    cache.mkdir(parents=True,exist_ok=True);lock=cache/'durable.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Supervisor lock exists; inspect before any explicit resume')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    info={'pid':os.getpid(),'birth':process_identity(os.getpid())['birth'],'started':time.time(),'deadline':deadline,'queue':str(queue),'command':[str(PYTHON),'-u',str(queue),*queue_args]}
    proc=None
    def heartbeat(state,**details):save(cache/'supervisor_status.json',{**info,'state':state,'updated':time.time(),**details})
    try:
        assert time.time()<deadline,'Original 24h suite deadline already passed'
        heartbeat('starting')
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        with (cache/'durable.stdout.log').open('a',encoding='utf8') as out,(cache/'durable.stderr.log').open('a',encoding='utf8') as err:
            proc=subprocess.Popen(info['command'],cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=flags,env={**os.environ,'PYTHONUNBUFFERED':'1'})
            info.update(queue_pid=proc.pid,queue_birth=process_identity(proc.pid)['birth'])
            while proc.poll() is None:
                if time.time()>deadline+120:
                    kill_owned_tree(proc);heartbeat('deadline_exceeded',returncode=proc.returncode);return 1
                phase=None
                try:phase=json.loads((cache/'queue_status.json').read_text()).get('state')
                except (FileNotFoundError,json.JSONDecodeError):pass
                heartbeat('running',phase=phase);time.sleep(5)
        heartbeat('complete' if proc.returncode==0 else 'failed',returncode=proc.returncode)
        return proc.returncode
    except BaseException as e:
        if proc is not None:kill_owned_tree(proc)
        heartbeat('failed',error=repr(e),returncode=proc.returncode if proc else None);raise
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()

def watch(pid,birth,cache,deadline):
    cache=safe_path(cache);info={'pid':os.getpid(),'birth':process_identity(os.getpid())['birth'],'queue_pid':pid,'queue_birth':birth,'watch_only':True,'started':time.time(),'deadline':deadline}
    while True:
        identity=process_identity(pid);same=identity['alive'] is True and identity['birth']==birth
        phase=None
        try:phase=json.loads((cache/'queue_status.json').read_text()).get('state')
        except (FileNotFoundError,json.JSONDecodeError):pass
        state='running' if same else 'complete' if phase in ('complete','stopped_no_advancement') else 'interrupted'
        if same and time.time()>deadline+120:state='deadline_exceeded'
        save(cache/'supervisor_status.json',{**info,'updated':time.time(),'state':state,'phase':phase,'queue_exit_code':None,'note':'Watch only; never launches, kills or retries the observed queue'})
        if state!='running':return 0 if state=='complete' else 1
        time.sleep(5)
def register(name,queue,cache,deadline,queue_args,plumbing=False,watch_pid=None,birth=None):
    assert re.fullmatch(r'B93_[A-Za-z0-9_-]+',name),'Owned task names must start B93_'
    cache=safe_path(cache);cache.mkdir(parents=True,exist_ok=True)
    if watch_pid:action_args=['-u',str(HERE/'durable93.py'),'watch','--pid',str(watch_pid),'--birth',birth,'--cache',str(cache),'--deadline',str(deadline)]
    elif plumbing:action_args=['-u',str(HERE/'durable93.py'),'plumbing','--cache',str(cache)]
    else:action_args=['-u',str(HERE/'durable93.py'),'supervise','--queue',str(safe_path(queue)),'--cache',str(cache),'--deadline',str(deadline),*sum((['--queue-arg',a] for a in queue_args),[])]
    args=subprocess.list2cmdline(action_args);limit=max(1,math.ceil(deadline-time.time()+120))
    scheduler_python=PYTHON.with_name('pythonw.exe');assert scheduler_python.is_file(),'Windowless interpreter required for hidden task'
    script=f"""$ErrorActionPreference='Stop'
$name={quote_ps(name)}
$old=Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
if($old -and $old.State -eq 'Running'){{throw 'Owned task is already running'}}
$action=New-ScheduledTaskAction -Execute {quote_ps(scheduler_python)} -Argument {quote_ps(args)} -WorkingDirectory {quote_ps(ROOT)}
$settings=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds {limit}) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal=New-ScheduledTaskPrincipal -UserId ($env:USERDOMAIN+'\\'+$env:USERNAME) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Settings $settings -Principal $principal -Force | Out-Null
Get-ScheduledTask -TaskName $name | Select-Object TaskName,State | ConvertTo-Json -Compress
"""
    result=powershell(script)
    record={'task_name':name,'method':'task_scheduler','registered':result.returncode==0,'registration_returncode':result.returncode,'registration_output':result.stdout.strip(),'registration_error':result.stderr.strip(),'command':[str(scheduler_python),*action_args],'deadline':deadline,'max_execution_seconds':limit,'restart_on_failure':0,'multiple_instances':'IgnoreNew','plumbing_only':plumbing,'created_at':time.time()}
    save(cache/'durable_registration.json',record);return record
def start(name,cache):
    cache=safe_path(cache);record=json.loads((cache/'durable_registration.json').read_text());assert record['task_name']==name
    assert record['registered'],'Task registration failed; use explicit breakaway only after inspection'
    if 'supervise' in record['command']:
        assert snapshot(cache).get('queue_process_alive') is not True,'Existing queue alive; this task may not duplicate it'
    result=powershell(f"$ErrorActionPreference='Stop'; Start-ScheduledTask -TaskName {quote_ps(name)}")
    if result.returncode:raise RuntimeError(result.stderr)
    return {'started_task':name,'method':'task_scheduler'}
def breakaway(command,cache):
    assert os.name=='nt';cache=safe_path(cache);cache.mkdir(parents=True,exist_ok=True)
    flags=0x01000000|0x00000008|0x00000200 # BREAKAWAY_FROM_JOB, DETACHED_PROCESS, NEW_PROCESS_GROUP
    with (cache/'breakaway.stdout.log').open('a',encoding='utf8') as out,(cache/'breakaway.stderr.log').open('a',encoding='utf8') as err:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=flags,close_fds=True)
    record={'method':'breakaway_detached','pid':proc.pid,'birth':process_identity(proc.pid)['birth'],'command':command,'created_at':time.time(),'restart_on_failure':0}
    save(cache/'breakaway_launch.json',record);return record
def plumbing(cache):
    cache=safe_path(cache);save(cache/'plumbing_timestamp.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'timestamp':time.time(),'purpose':'Actual process/task launch plumbing only; no observations or ML work'})
    return 0
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('register','start','supervise','status','plumbing','breakaway','watch'))
    p.add_argument('--name',default='B93_GPU_r3');p.add_argument('--queue',type=Path);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--deadline',type=float);p.add_argument('--queue-arg',action='append',default=[]);p.add_argument('--plumbing-only',action='store_true')
    p.add_argument('--pid',type=int);p.add_argument('--birth');a=p.parse_args()
    if a.action=='plumbing':return plumbing(a.cache)
    if a.action=='status':print(json.dumps(snapshot(safe_path(a.cache)),indent=2));return 0
    if a.action=='start':print(json.dumps(start(a.name,a.cache)));return 0
    deadline=a.deadline if a.deadline is not None else deadline_default()
    if a.pid and not a.birth:
        identity=process_identity(a.pid);assert identity['alive'] is True and identity['birth'];a.birth=identity['birth']
    if a.action=='watch':return watch(a.pid,a.birth,a.cache,deadline)
    if a.action=='register':print(json.dumps(register(a.name,a.queue,a.cache,deadline,a.queue_arg,a.plumbing_only,a.pid,a.birth)));return 0
    if a.action=='supervise':return supervise(a.queue,a.cache,deadline,a.queue_arg)
    if a.pid:args=['-u',str(HERE/'durable93.py'),'watch','--pid',str(a.pid),'--birth',a.birth,'--cache',str(safe_path(a.cache)),'--deadline',str(deadline)]
    elif a.plumbing_only:args=['-u',str(HERE/'durable93.py'),'plumbing','--cache',str(safe_path(a.cache))]
    else:args=['-u',str(HERE/'durable93.py'),'supervise','--queue',str(safe_path(a.queue)),'--cache',str(safe_path(a.cache)),'--deadline',str(deadline),*sum((['--queue-arg',v] for v in a.queue_arg),[])]
    print(json.dumps(breakaway([sys.executable,*args],a.cache)));return 0
if __name__=='__main__':raise SystemExit(main())
