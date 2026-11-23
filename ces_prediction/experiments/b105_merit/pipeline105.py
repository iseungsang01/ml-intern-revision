"""Committed-source gates, durable execution, bounded fits, audited publication."""
import argparse,os,shutil,subprocess,sys,time
from pathlib import Path
import common105 as C
from status93 import process_identity,snapshot
OWN={}

def status(state,**more):C.save(C.CACHE/'queue_status.json',{'state':state,'pid':os.getpid(),'birth':process_identity(os.getpid())['birth'],'updated':time.time(),**more})
def spawn(script,args,unit,kind,hard):
    path=(C.rd(unit,kind) if kind in ('smoke','confirm') else C.CACHE/kind/unit);path.mkdir(parents=True,exist_ok=True)
    assert not (path/'child_start.json').exists(),'Prior attempt exists; inspect, no blind retry'
    log=(path/'process.log').open('ab');env={k:v for k,v in os.environ.items() if not k.startswith('CES_')}
    env.update(C.env(unit,'smoke' if kind=='smoke' else 'confirm'));env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if kind=='trees':env.update(CES_SEQ_DEVICE='cpu',CUDA_VISIBLE_DEVICES='')
    p=subprocess.Popen([sys.executable,'-u',str(C.HERE/script),*args],cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    meta={'pid':p.pid,'birth':process_identity(p.pid)['birth'],'started':time.time(),'unit':unit,'kind':kind,'relative_log':(path/'process.log').relative_to(C.CACHE).as_posix()}
    C.save(path/'child_start.json',meta);e={'p':p,'path':path,'log':log,'meta':meta,'hard':hard};OWN[p.pid]=e;return e
def finish(e,timeout=False):
    p=e['p'];rc=p.wait();e['log'].close();OWN.pop(p.pid,None);path=e['path'];cfg=C.load(path/'config.json') if (path/'config.json').exists() else {}
    C.save(path/'process_exit.json',{'pid':p.pid,'birth':e['meta']['birth'],'returncode':rc,'timeout':timeout,'config_hash':cfg.get('config_hash'),'elapsed':time.time()-e['meta']['started']})
    assert rc==0 and not timeout,'Child failed: '+str(path.relative_to(C.CACHE))
    if e['meta']['kind'] in ('smoke','confirm'):C.verify_run(e['meta']['unit'],e['meta']['kind'])
def group(kind,units):
    active=[];pending=list(units);done=0;failure=None
    workers=4
    if kind=='confirm':
        mem=C.load(C.CACHE/'memory_checks.json');assert mem['all_pass'] and mem['sources']==C.sources();workers=mem['workers']
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        assert free>=workers*mem['per_worker_reserved_MiB'],'Reserved-slot VRAM admission failed'
    while pending or active:
        assert C.remaining()>0,'Suite deadline reached'
        for e in list(active):
            timeout=time.time()-e['meta']['started']>e['hard']
            if timeout and e['p'].poll() is None:e['p'].kill()
            if e['p'].poll() is not None:
                try:finish(e,timeout);done+=1
                except Exception as err:failure=failure or repr(err)
                active.remove(e)
        if failure:pending=[]
        while pending and len(active)<workers and not failure:
            if kind=='confirm':
                if C.remaining()<=5820:
                    failure='Insufficient remaining time for new fit plus30min readout reserve; active fits may finish';pending=[];break
                u=pending.pop(0);e=spawn('train105.py',['--unit',u],u,'confirm',4020)
            else:
                u=pending.pop(0);e=spawn('trees105.py',['--unit',u],u,'trees',2700)
            active.append(e)
        status('training' if kind=='confirm' else 'CPU_references',completed=done,total=4,active=[e['meta'] for e in active],failure=failure)
        if active:time.sleep(1)
    if failure:raise RuntimeError(failure)
    assert done==len(units)
def step(script,args=(),timeout=900):
    path=C.CACHE/'steps';path.mkdir(exist_ok=True);name=script.removesuffix('.py')+'_'+'_'.join(args)
    with (path/(name+'.log')).open('ab') as log:
        unit=args[args.index('--unit')+1] if '--unit' in args else 'all_units'
        pins=C.env(unit)
        cpu=script!='evaluate105.py' and args!=['memory'];pins['CES_SEQ_DEVICE']='cpu' if cpu else 'cuda'
        env={k:v for k,v in os.environ.items() if not k.startswith('CES_')}
        env.update(pins);env.update(CUDA_VISIBLE_DEVICES='' if cpu else '0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        C.save(path/(name+'.environment.json'),{'env':pins,'env_sha256':C.digest(pins),'sources':C.sources()})
        p=subprocess.Popen([sys.executable,'-u',str(C.HERE/script),*args],cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:rc=p.wait(timeout=max(1,min(timeout,C.remaining())))
        except subprocess.TimeoutExpired:p.kill();p.wait();raise RuntimeError('Timed out step '+name)
        assert rc==0,'Failed step '+name+'; see data/.b105/steps/'
def postreview():
    report=C.load(C.CACHE/'readout.json');assert C.load(C.CACHE/'audit.json')['report_sha256']==C.sha(C.CACHE/'readout.json')
    script=C.ROOT/'ces_prediction/research/review_b105.py'
    p=subprocess.run([sys.executable,str(script),'post'],cwd=C.ROOT,timeout=300,capture_output=True,text=True)
    (C.CACHE/'post_review.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf8');assert p.returncode==0,'Scientific result audited, Claude post-review failed; do not repeat fits'
    review=C.CACHE/'review/post.txt';(C.PAPER/'b105_claude.md').write_text('# B105 completed Fable post-evaluation advice\n\n'+review.read_text(encoding='utf8'),encoding='utf8')
def run():
    status('readiness');C.guard();step('checks105.py',['inputs'],900)
    status('CPU_references');group('trees',C.UNITS)
    status('smoke');e=spawn('train105.py',['--unit',C.UNITS[0],'--stage','smoke'],C.UNITS[0],'smoke',420)
    try:e['p'].wait(timeout=420)
    except subprocess.TimeoutExpired:e['p'].kill();finish(e,True)
    finish(e)
    status('full_batch_memory_probe');step('checks105.py',['memory'],600)
    status('training');group('confirm',C.UNITS);step('checks105.py',['fits'],900)
    status('VAL_predictions');step('evaluate105.py',['val'],900);step('evaluate105.py',['gate'],180)
    gate=C.load(C.CACHE/'gate.json')
    if gate['all_pass']:
        status('conditional_TEST');step('evaluate105.py',['test'],1200)
        for u in C.UNITS:step('trees105.py',['--unit',u,'--part','test'],600)
    status('readout');step('readout105.py',timeout=1200)
    status('independent_audit');step('audit105.py',timeout=1500)
    a=C.load(C.CACHE/'audit.json');assert a['all_pass'] and a['report_sha256']==C.sha(C.CACHE/'readout.json')
    for src,dest in [('readout.json','b105_merit.json'),('readout.md','b105_merit.md')]:
        shutil.copyfile(C.CACHE/src,C.PAPER/dest);assert C.sha(C.CACHE/src)==C.sha(C.PAPER/dest)
    final='# B.105 scientific evaluation complete\n\nAll four fits exited normally and independent audit passed.\nVAL allocation gate: '+str(gate['all_pass'])+'. Candidate TEST evaluated: '+str(gate['all_pass'])+'.\nSources and intervals: b105_merit.json/md. No further scientific fits queued.\n'
    (C.PAPER/'b105_status.md').write_text(final+'\nClaude post review pending; scientific report already audited.\n',encoding='utf8')
    status('post_review');postreview()
    (C.PAPER/'b105_status.md').write_text(final+'\nBounded Fable post review completed: b105_claude.md.\n',encoding='utf8')
    status('complete',summary='Audited merit report complete; VAL gate='+str(gate['all_pass']))
def register():
    import durable93 as D
    C.guard();assert not (C.CACHE/'suite_clock.json').exists(),'Clock already exists; never reset'
    now=time.time();deadline=now+9000;C.save(C.CACHE/'suite_clock.json',{'started':now,'deadline':deadline,'registration':'PREREGISTRATION_B105.md'})
    name='B105_GPU_merit';watch='B105_Merit_watch';pythonw=Path(sys.executable).with_name('pythonw.exe');q=D.quote_ps
    ui={k:os.environ[k] for k in ('CVMUX','CVMUX_SESSION_ID','CVMUX_SOCKET_PATH','CVMUX_SOCKET_PASSWORD') if k in os.environ}
    assert ui.get('CVMUX_SOCKET_PASSWORD') and shutil.which('cvmux.cmd'),'Local notifications must be available'
    from ui105 import protect,unprotect
    protected=protect(ui);assert unprotect(protected)==ui
    C.save(C.CACHE/'watch_config.json',{'started':now,'deadline':deadline,'ui_protected':protected,'notify_bin':shutil.which('cvmux.cmd')})
    args=['-u',str(C.HERE.parent/'b99_recovery/monitor99.py'),'supervise','--queue',str(C.HERE/'pipeline105.py'),'--cache',str(C.CACHE),'--deadline',str(deadline),'--queue-arg','run']
    for task,arguments in [(name,args),(watch,['-u',str(C.HERE/'watch105.py')])]:
        script=f"""$ErrorActionPreference='Stop'
if(Get-ScheduledTask -TaskName '{task}' -ErrorAction SilentlyContinue){{throw 'Task already exists'}}
$a=New-ScheduledTaskAction -Execute {q(pythonw)} -Argument {q(subprocess.list2cmdline(arguments))} -WorkingDirectory {q(C.ROOT)}
$s=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds 9300) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$p=New-ScheduledTaskPrincipal -UserId ($env:USERDOMAIN+'\\'+$env:USERNAME) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName '{task}' -Action $a -Settings $s -Principal $p | Out-Null
"""
        r=D.powershell(script);assert r.returncode==0,r.stderr
    C.save(C.CACHE/'durable_registration.json',{'task_name':name,'registered':True,'deadline':deadline,'restart_on_failure':0,'command':[str(pythonw),*args]})
    r=D.powershell("Start-ScheduledTask -TaskName '"+watch+"'");assert r.returncode==0
    for _ in range(30):
        time.sleep(1);p=C.CACHE/'watch_status.json'
        if p.exists() and C.load(p).get('notify_returncode')==0:break
    else:raise RuntimeError('No accepted notifier delivery; scientific work not launched')
    D.start(name,C.CACHE);print('B105 durable suite and independent monitor launched')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','register','run','status'));a=p.parse_args()
    try:
        if a.action=='prepare':C.prepare()
        elif a.action=='status':print(__import__('json').dumps(snapshot(C.CACHE)))
        else:globals()[a.action]()
    except BaseException as err:
        for e in list(OWN.values()):
            identity=process_identity(e['p'].pid)
            if e['p'].poll() is None and identity['alive'] is True and identity['birth']==e['meta']['birth']:
                e['p'].kill()
                try:finish(e,True)
                except Exception:pass
        if a.action not in ('prepare','status','register'):status('failed',error=repr(err))
        raise
