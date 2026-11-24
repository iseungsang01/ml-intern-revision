"""Bounded20-fit LSTM cross-fitting, correction learning and MSE-first comparison."""
import argparse,os,sys,subprocess,shutil,time
from pathlib import Path
import common106 as C
import pipeline105 as P
P.C=C
from status93 import snapshot,process_identity
def group():
    mem=C.load(C.CACHE/'memory_checks.json');workers=mem['workers'];assert mem['all_pass'] and mem['sources']==C.sources()
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);assert free>=workers*mem['per_worker_MiB']
    pending=list(C.CASES);active=[];done=0;failure=None
    while pending or active:
        assert C.remaining()>0,'Suite deadline'
        for e in list(active):
            timeout=time.time()-e['meta']['started']>4200
            if timeout and e['p'].poll() is None:e['p'].kill()
            if e['p'].poll() is not None:
                try:P.finish(e,timeout);done+=1
                except Exception as err:failure=failure or repr(err)
                active.remove(e)
        if failure:pending=[]
        while pending and len(active)<workers and not failure:
            if C.remaining()<7800:failure='Insufficient budget for new fit plus1h readout; active fits may finish';pending=[];break
            case=pending.pop(0);active.append(P.spawn('train106.py',['--case',case],case,'confirm',4200))
        P.status('training',completed=done,total=20,pending=len(pending),active=[e['meta'] for e in active],failure=failure)
        if active:time.sleep(1)
    if failure:raise RuntimeError(failure)
    assert done==20
def step(script,args=(),timeout=1800):
    out=C.CACHE/'steps';out.mkdir(exist_ok=True);name=script.removesuffix('.py')+'_'+'_'.join(args)
    env={k:v for k,v in os.environ.items() if not k.startswith('CES_')};pins=C.env('all_cases')
    gpu=script=='predict106.py' or args==['memory'];pins['CES_SEQ_DEVICE']='cuda' if gpu else 'cpu'
    env.update(pins);env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    C.save(out/(name+'.env.json'),{'env':pins,'hash':C.digest(pins),'sources':C.sources()})
    with (out/(name+'.log')).open('ab') as log:
        p=subprocess.Popen([sys.executable,'-u',str(C.HERE/script),*args],cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:rc=p.wait(timeout=max(1,min(timeout,C.remaining())))
        except subprocess.TimeoutExpired:p.kill();p.wait();raise RuntimeError('Step timeout: '+name)
        assert rc==0,'Failed step '+name+'; inspect data/.b106/steps'
def run():
    P.status('readiness');step('checks106.py',['inputs'],900)
    P.status('smoke');case=C.CASES[0];e=P.spawn('train106.py',['--case',case,'--stage','smoke'],case,'smoke',420)
    try:e['p'].wait(timeout=420)
    except subprocess.TimeoutExpired:e['p'].kill();P.finish(e,True)
    P.finish(e);P.status('full_batch_memory');step('checks106.py',['memory'],900)
    group();P.status('fit_audit');step('checks106.py',['fits'],900)
    P.status('OOF_prediction');step('predict106.py',['held'],5400)
    P.status('correction_fit');step('correction106.py',['fit'],900)
    P.status('TEST_prediction');step('predict106.py',['test'],3600);step('correction106.py',['test'],900)
    P.status('MSE_readout');step('readout106.py',timeout=1200)
    P.status('independent_audit');step('audit106.py',timeout=1800)
    a=C.load(C.CACHE/'audit.json');assert a['all_pass'] and a['report_sha256']==C.sha(C.CACHE/'readout.json')
    for src,dst in [('readout.json','b106_lstm.json'),('readout.md','b106_lstm.md')]:shutil.copyfile(C.CACHE/src,C.PAPER/dst)
    (C.PAPER/'b106_status.md').write_text('# B106 complete\n\nTwenty LSTM fits, OOF correction training and all scenario MSE comparisons passed independent audit. No Claude API calls. Results b106_lstm.json/md. No additional fits queued.\n',encoding='utf8')
    P.status('complete',summary='LSTM+post / GP+boosting / TCN MSE comparison independently audited')
def register():
    import durable93 as D
    from ui105 import protect,unprotect
    C.guard();assert not (C.CACHE/'suite_clock.json').exists(),'Existing clock; inspect'
    now=time.time();deadline=now+28800;C.save(C.CACHE/'suite_clock.json',{'started':now,'deadline':deadline,'registration':'PREREGISTRATION_B106.md'})
    ui={k:os.environ[k] for k in ('CVMUX','CVMUX_SESSION_ID','CVMUX_SOCKET_PATH','CVMUX_SOCKET_PASSWORD') if k in os.environ}
    assert ui.get('CVMUX_SOCKET_PASSWORD');cipher=protect(ui);assert unprotect(cipher)==ui
    name='B106_LSTM_training';watch='B106_LSTM_watch';pythonw=Path(sys.executable).with_name('pythonw.exe');q=D.quote_ps
    C.save(C.CACHE/'watch_config.json',{'started':now,'deadline':deadline,'ui_protected':cipher,'notify_bin':shutil.which('cvmux.cmd')})
    args=['-u',str(C.HERE.parent/'b99_recovery/monitor99.py'),'supervise','--queue',str(C.HERE/'pipeline106.py'),'--cache',str(C.CACHE),'--deadline',str(deadline),'--queue-arg','run']
    for task,arguments in [(name,args),(watch,['-u',str(C.HERE/'watch106.py')])]:
        script=f"""$ErrorActionPreference='Stop'
if(Get-ScheduledTask -TaskName '{task}' -ErrorAction SilentlyContinue){{throw 'Existing task'}}
$a=New-ScheduledTaskAction -Execute {q(pythonw)} -Argument {q(subprocess.list2cmdline(arguments))} -WorkingDirectory {q(C.ROOT)}
$s=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds 29400) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$p=New-ScheduledTaskPrincipal -UserId ($env:USERDOMAIN+'\\'+$env:USERNAME) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName '{task}' -Action $a -Settings $s -Principal $p | Out-Null
"""
        r=D.powershell(script);assert r.returncode==0,r.stderr
    C.save(C.CACHE/'durable_registration.json',{'registered':True,'task_name':name,'command':[str(pythonw),*args],'deadline':deadline,'restart_on_failure':0})
    assert D.powershell("Start-ScheduledTask -TaskName '"+watch+"'").returncode==0
    for _ in range(30):
        time.sleep(1);p=C.CACHE/'watch_status.json'
        if p.exists() and C.load(p).get('notify_returncode')==0:break
    else:raise RuntimeError('No notifier acceptance; training not launched')
    D.start(name,C.CACHE);print('B106 durable20-fit suite launched')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','register','run','status'));a=p.parse_args()
    try:
        if a.action=='prepare':C.prepare()
        elif a.action=='status':print(__import__('json').dumps(snapshot(C.CACHE)))
        else:globals()[a.action]()
    except BaseException as err:
        for e in list(P.OWN.values()):
            identity=process_identity(e['p'].pid)
            if e['p'].poll() is None and identity['alive'] is True and identity['birth']==e['meta']['birth']:
                e['p'].kill()
                try:P.finish(e,True)
                except Exception:pass
        if a.action not in ('prepare','register','status'):P.status('failed',error=repr(err))
        raise
