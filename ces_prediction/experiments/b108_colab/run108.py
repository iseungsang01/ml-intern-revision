"""Fresh-process one/four-worker timing with explicit normal exit records."""
import argparse,hashlib,json,os,platform,subprocess,sys,time,zipfile
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,z):Path(p).write_text(json.dumps(z,indent=2),encoding='utf8')
def main(expect_t4=False):
    root=Path(__file__).resolve().parent;manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert Path(name).name==name and sha(root/name)==h,'Bundle file changed: '+name
    import torch
    assert torch.cuda.is_available(),'Select a GPU runtime first';gpu=torch.cuda.get_device_name(0)
    if expect_t4:assert 'T4' in gpu,'Expected author-provided T4; got '+gpu
    out=root/'results';out.mkdir();groups=[];suite_started=time.monotonic()
    for rep in range(3):
        for workers in (1,4):
            for backend in ('native','cudnn'):
                name=f'{backend}_w{workers}_r{rep}';directory=out/name;directory.mkdir();gate=directory/'go';timed_gate=directory/'timed_go';active=[];started=time.perf_counter()
                if time.monotonic()-suite_started>1800:
                    groups.append({'name':name,'backend':backend,'workers':workers,'rep':rep,'all_pass':False,'unrun':'Suite timing guard'});continue
                for unit in ('A','B','C','D')[:workers]:
                    result=directory/(unit+'.json');log=(directory/(unit+'.log')).open('wb')
                    command=[sys.executable,'-u',str(root/'probe108.py'),'--root',str(root),'--unit',unit,'--backend',backend,'--output',str(result),'--gate',str(gate),'--time-gate',str(timed_gate)]
                    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    active.append((proc,result,log))
                timeout=False
                while any(p.poll() is None for p,_,_ in active):
                    if not gate.exists() and all(r.with_suffix('.ready').exists() or p.poll() is not None for p,r,_ in active):gate.write_text('go')
                    if not timed_gate.exists() and all(r.with_suffix('.warm').exists() or p.poll() is not None for p,r,_ in active):timed_gate.write_text('go')
                    if time.perf_counter()-started>600:
                        timeout=True
                        for p,_,_ in active:
                            if p.poll() is None:p.kill()
                        break
                    time.sleep(.02)
                records=[]
                for p,result,log in active:
                    rc=p.wait();log.close();r=json.loads(result.read_text()) if result.exists() else None
                    record={'unit':result.stem,'returncode':rc,'timeout':timeout,'result':r};save(result.with_suffix('.exit.json'),record);records.append(record)
                good=not timeout and all(r['returncode']==0 and r['result'] and r['result']['passed'] for r in records)
                group={'name':name,'backend':backend,'workers':workers,'rep':rep,'all_pass':bool(good),'launch_to_exit_seconds':time.perf_counter()-started,'children':records}
                if good:
                    rr=[r['result'] for r in records];span=max(r['timed_end'] for r in rr)-min(r['timed_begin'] for r in rr)
                    group['synchronized_steps_per_second']=sum(r['timed_steps'] for r in rr)/span
                groups.append(group);save(out/'partial.json',groups);print(name,'PASS' if good else 'UNQUALIFIED',flush=True)
    summary={'purpose':'Repeated real-batch runtime qualification, not full training/MSE','gpu':gpu,'python':platform.python_version(),'torch':torch.__version__,'cpu_count':os.cpu_count(),
             'bundle_manifest_sha256':sha(root/'manifest.json'),'groups':groups,'configurations':{}}
    import statistics
    for backend in ('native','cudnn'):
        bb=[g for g in groups if g['backend']==backend];qualified=len(bb)==6 and all(g['all_pass'] for g in bb)
        summary['configurations'][backend]={'eligible_all_15_children_clean':qualified}
        for w in (1,4):
            gg=[g for g in bb if g['workers']==w and g['all_pass']]
            summary['configurations'][backend][f'workers{w}']={'clean_repetitions':len(gg),'median_steps_per_second':statistics.median(g['synchronized_steps_per_second'] for g in gg) if len(gg)==3 else None}
    save(out/'summary.json',summary)
    with zipfile.ZipFile(root/'b108_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(root))
    print(json.dumps(summary['configurations'],indent=2));print('Download b108_results.zip; no observations included.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expect-t4',action='store_true');main(p.parse_args().expect_t4)
