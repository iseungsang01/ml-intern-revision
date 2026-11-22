"""Sequential B88 children with runner-observed, hash-bound exit attestations."""
import argparse,json,os,subprocess,sys,time
import common88 as M
from adapter88 import activate,dependency_ready,input_ready,acquire_file_lock,release_file_lock
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf8');tmp.replace(path)
def wait_vram():
    while True:
        assert dependency_ready(),'B85 dependency not complete'
        out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if int(out.splitlines()[0].strip())>=2500:return
        print('Waiting for at least 2500 MiB free GPU memory',flush=True);time.sleep(10)
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--plan',action='store_true');a=p.parse_args()
    M.assert_frozen();t=activate();units=M.select_units()
    jobs=[(u['name'],i,v) for u in units for i in M.INITS for v in M.VARIANTS]
    if a.smoke:jobs=[(units[0]['name'],M.INITS[0],v) for v in M.VARIANTS]
    if a.plan:print(json.dumps({'jobs':jobs,'smoke':a.smoke}));return
    if a.status:
        print(json.dumps({'total':len(jobs),'complete':sum((t.get_run_dir(*j,smoke=a.smoke)/'complete.json').is_file() for j in jobs)}));return
    assert dependency_ready(),'B85 dependency not complete'
    input_ready()
    lock=M.CACHE/'batch.lock';assert acquire_file_lock(lock),'B88 batch already active'
    env={k:v for k,v in os.environ.items() if not k.startswith('CES_')};env.update(t.ENV_PINS);env['PYTHONUNBUFFERED']='1'
    try:
        for unit,seed,var in jobs:
            M.assert_frozen();rd=t.get_run_dir(unit,seed,var,smoke=a.smoke)
            if (rd/'complete.json').exists():
                assert t.verify_complete(rd);continue
            wait_vram();rd.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'-X','faulthandler','-u',str(M.HERE/'train88.py'),'--unit',unit,'--init',str(seed),'--variant',var]
            if a.smoke:cmd.append('--smoke')
            print('Starting '+rd.name,flush=True)
            with (rd/'process.log').open('a',encoding='utf8') as log:
                result=subprocess.run(cmd,cwd=M.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            cfg=json.loads((rd/'config.json').read_text(encoding='utf8')) if (rd/'config.json').exists() else {}
            save(rd/'process_exit.json',{'returncode':result.returncode,'config_hash':cfg.get('config_hash'),'run':rd.name,'timestamp':time.time()})
            if result.returncode:raise RuntimeError('Child failed: '+rd.name)
            assert t.verify_complete(rd)
    finally:release_file_lock(lock)
if __name__=='__main__':main()
