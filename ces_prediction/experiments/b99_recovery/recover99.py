"""Explicit preserved-attempt recovery; never silently retries scientific jobs."""
import argparse,json,math,subprocess,sys,time
from pathlib import Path
import monitor99 as M
sys.path.insert(0,str(M.HERE.parent/'b96_four'))
import common96 as C
G=C.G
PARTIALS=['V3_q2__1701__TH','V3_q3__1701__TX','V3_q3__1701__TH','V3_q4__1701__TX']

def sources():
    files=[*M.HERE.glob('*.py'),M.HERE/'lead_checks/verify99.py',M.HERE.parent/'PREREGISTRATION_B99.md',M.HERE.parent/'PREREGISTRATION_B94_ADDENDUM4.md']
    for p in files:G.committed(p)
    return {p.relative_to(M.ROOT).as_posix():G.sha(p) for p in files}

def verify_existing():
    s,q=C.policy()
    from adapter94 import activate
    T=activate();completed={}
    for stage in ('smoke','pilot','smoke_confirm','confirm'):
        records=[]
        for p in sorted((G.CACHE/stage).glob('*/complete.json')):
            assert T.verify_complete(p.parent);C.dispatch_config(p.parent/'config.json',s,q)
            records.append({'run':p.parent.name,'complete_sha256':G.sha(p),'exit_sha256':G.sha(p.parent/'process_exit.json')})
        completed[stage]=records
    assert len(completed['pilot'])==12 and len(completed['confirm'])==23
    return completed

def archive():
    M.CACHE.mkdir(parents=True,exist_ok=True)
    assert not (M.CACHE/'archive99.json').exists(),'Archive already recorded; inspect before proceeding'
    assert json.loads((M.CACHE/'probe99.json').read_text())['all_pass']
    state=M.D.snapshot(G.CACHE)
    assert state.get('queue_process_alive') is False and state.get('supervisor_alive') is False
    assert state['supervisor']['state']=='failed'
    assert {a['run'] for a in state['queue']['active']}==set(PARTIALS)
    assert all(M.D.process_identity(a['pid'])['alive'] is not True for a in state['queue']['active'])
    completed=verify_existing();pins=G.guard();ss=sources()
    dest=(G.CACHE/'interrupted_permission99').resolve();dest.mkdir(exist_ok=False)
    files={}
    for name in PARTIALS:
        src=(G.CACHE/'confirm'/name).resolve();dst=(dest/name).resolve()
        assert src.is_relative_to((G.CACHE/'confirm').resolve()) and dst.is_relative_to(dest)
        assert not (src/'complete.json').exists() and not (src/'process_exit.json').exists()
        files[name]={str(p.relative_to(src)):G.sha(p) for p in src.rglob('*') if p.is_file()}
        src.rename(dst)
    control=(dest/'old_control').resolve();control.mkdir()
    for name in ('queue_status.json','supervisor_status.json','queue.lock','durable.lock','durable_registration.json'):
        src=(G.CACHE/name).resolve();dst=(control/name).resolve()
        assert src.is_relative_to(G.CACHE.resolve()) and dst.is_relative_to(control)
        if src.exists():src.rename(dst)
    record={'sources99':ss,'frozen':pins,'completed':completed,'archived':files,
            'deadline':G.clock()['deadline'],'original_status':state,'archived_at':time.time(),
            'scientific_restarts_only':PARTIALS,'new_task':'B93_GPU_B94_recovery99'}
    G.save(M.CACHE/'archive99.json',record);print(json.dumps({'archived_partials':len(files),'verified_confirmation':len(completed['confirm'])}))

def launch():
    z=json.loads((M.CACHE/'archive99.json').read_text());assert z['sources99']==sources()
    check=json.loads((M.CACHE/'lead_checks99.json').read_text());assert check['all_pass'] and check['archive_sha256']==G.sha(M.CACHE/'archive99.json')
    assert time.time()<z['deadline'] and G.guard()==z['frozen']
    assert not (G.CACHE/'queue.lock').exists() and not (G.CACHE/'durable.lock').exists()
    old='B93_GPU_B94_four96';name=z['new_task'];pythonw=Path(sys.executable).with_name('pythonw.exe')
    args=['-u',str(M.HERE/'monitor99.py'),'supervise','--queue',str(M.HERE.parent/'b96_four/production96.py'),'--cache',str(G.CACHE),'--deadline',str(z['deadline'])]
    quote=M.D.quote_ps
    script=f"""$ErrorActionPreference='Stop'
$old=Get-ScheduledTask -TaskName '{old}'
if($old.State -eq 'Running'){{throw 'Old task is running'}}
Disable-ScheduledTask -TaskName '{old}' | Out-Null
$action=New-ScheduledTaskAction -Execute {quote(pythonw)} -Argument {quote(subprocess.list2cmdline(args))} -WorkingDirectory {quote(M.ROOT)}
$settings=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds {math.ceil(z['deadline']-time.time()+120)}) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal=New-ScheduledTaskPrincipal -UserId ($env:USERDOMAIN+'\\'+$env:USERNAME) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName '{name}' -Action $action -Settings $settings -Principal $principal -Force | Out-Null
"""
    r=M.D.powershell(script);assert r.returncode==0,r.stderr
    record={'task_name':name,'registered':True,'command':[str(pythonw),*args],'deadline':z['deadline'],
            'method':'task_scheduler','restart_on_failure':0,'multiple_instances':'IgnoreNew','sources99':sources()}
    G.save(G.CACHE/'durable_registration.json',record)
    print(json.dumps(M.D.start(name,G.CACHE)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['archive','launch']);a=p.parse_args()
    archive() if a.action=='archive' else launch()
