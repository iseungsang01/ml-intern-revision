"""Serial GPU fits and validation-only promotion; no test access here."""
import argparse,json,math,os,subprocess,sys,time
import common90 as M
from adapter90 import activate,dependency_ready,input_ready,acquire_file_lock,release_file_lock

def validation_record():
    from adapter88 import baseline_trainer
    baseline=baseline_trainer();records=[];scores={}
    for method in M.METHODS:
        ratios=[];t=activate(method)
        for u in M.select_units():
            for var in M.VARIANTS:assert t.verify_complete(t.get_run_dir(u['name'],1701,var))
            rd=t.get_run_dir(u['name'],1701,'X');br=baseline.get_run_dir(u['name'],1701,'X');assert baseline.verify_complete(br)
            left=json.loads((rd/'metrics.json').read_text(encoding='utf8'))['best_val_loss']
            right=json.loads((br/'metrics.json').read_text(encoding='utf8'))['best_val_loss']
            assert math.isfinite(left) and left>=0 and math.isfinite(right) and right>0
            assert json.loads((rd/'manifest.json').read_text())==json.loads((br/'manifest.json').read_text()),'Validation manifest differs'
            assert json.loads((rd/'stats.json').read_text())==json.loads((br/'stats.json').read_text()),'Training statistics differ'
            ratios.append(left/right)
            records.append({'method':method,'unit':u['name'],'candidate_val':left,'b85_val':right,'ratio':left/right,
                            'hashes':{str(p.relative_to(M.ROOT)):M.sha(p) for folder in (rd,br) for p in (folder/'metrics.json',folder/'config.json',folder/'manifest.json',folder/'stats.json',folder/'complete.json',folder/'process_exit.json')}})
        scores[method]=sum(ratios)/len(ratios)
    order=sorted(M.METHODS,key=lambda m:(scores[m],m))
    return {'stage':'validation_only','formula':'mean over 15 units(candidate X init1701 best_val_loss / B85 X init1701 best_val_loss)',
            'candidate_order':list(M.METHODS),'ranking':order,'scores':scores,'promoted':order[:2],'validation_inputs':records,'execution_sources':M.source_inventory(),
            'test_predictions_accessed':False,'independent_test_selection':False}

def jobs(stage):
    if stage=='smoke':return [(m,M.select_units()[0]['name'],1701,v) for m in M.METHODS for v in M.VARIANTS]
    if stage=='screen':return M.screen_jobs()
    return [(m,u['name'],s,v) for m in M.promotion()['promoted'] for u in M.select_units() for s in M.INITS[1:] for v in M.VARIANTS]

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('smoke','screen','promote','promoted'),required=True);p.add_argument('--plan',action='store_true');a=p.parse_args()
    M.assert_registered();M.assert_frozen()
    if a.plan:print(json.dumps({'stage':a.stage,'jobs':jobs(a.stage) if a.stage!='promote' else 'validation-only top two'}));return
    assert dependency_ready();input_ready()
    lock=M.CACHE/'batch.lock';assert acquire_file_lock(lock),'B90 batch already active'
    try:
        if a.stage=='promote':
            record=validation_record();path=M.CACHE/'promotion.json'
            if path.exists():assert json.loads(path.read_text())==record,'Locked promotion mismatch'
            else:M.save(path,record)
            return
        for method,unit,seed,var in jobs(a.stage):
            input_ready();assert dependency_ready();t=activate(method);smoke=a.stage=='smoke';rd=t.get_run_dir(unit,seed,var,smoke)
            if (rd/'complete.json').exists():assert t.verify_complete(rd);continue
            while True:
                free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if int(free.splitlines()[0].strip())>=2500:break
                print('Waiting for free GPU memory; other processes unchanged',flush=True);time.sleep(10)
            rd.mkdir(parents=True,exist_ok=True)
            env={k:v for k,v in os.environ.items() if not k.startswith('CES_')};env.update(t.ENV_PINS);env['PYTHONUNBUFFERED']='1'
            cmd=[sys.executable,'-X','faulthandler','-u',str(M.HERE/'train90.py'),'--method',method,'--unit',unit,'--init',str(seed),'--variant',var]
            if smoke:cmd.append('--smoke')
            print('Starting '+str(rd.relative_to(M.CACHE)),flush=True)
            with (rd/'process.log').open('a',encoding='utf8') as log:result=subprocess.run(cmd,cwd=M.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            cfg=json.loads((rd/'config.json').read_text()) if (rd/'config.json').exists() else {}
            M.save(rd/'process_exit.json',{'returncode':result.returncode,'config_hash':cfg.get('config_hash'),'run':rd.name,'method':method,'timestamp':time.time()})
            if result.returncode:raise RuntimeError('Child failed: '+str(rd))
            assert t.verify_complete(rd)
    finally:release_file_lock(lock)
if __name__=='__main__':main()
