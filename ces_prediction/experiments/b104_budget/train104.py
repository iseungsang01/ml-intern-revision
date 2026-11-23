"""Frozen numerical loop with read-only deadline observer; no checkpoint continuation."""
import argparse,hashlib,os,pickle,random,sys,time
import numpy as np,torch
import common104 as C
import train92 as T
import common92 as M

def rng_hash():
    return C.digest({'python':hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        'numpy':hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        'torch':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        'cuda':[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()]})

def main(stage,seed,code,observe):
    assert stage=='smoke' or observe;assert os.environ['CES_SEQ_SEED']==str(seed)
    C.guard();M.CACHE=C.CACHE;M.sources=C.sources;M.guard=C.guard;M.save=C.save
    M.run_dir=lambda st,u,se,co:C.run_dir(st,se,co,observe)
    T.prepare_unit=lambda name:C.load_prepared()
    T.decision=C.runtime;T.checks_ready=lambda:C.guard()
    T.effective_env=lambda st,co,rt:C.env(st,co,seed,observe)
    original_config=T.config
    def config(st,u,se,co,rt,prepared):
        z=original_config(st,u,se,co,rt,prepared);z.pop('config_hash')
        z.update(experiment='B104',registration='PREREGISTRATION_B104.md',epochs=2 if st=='smoke' else 40,
            seconds=120 if st=='smoke' else C.LONG,observer_enabled=observe,global_worker_limit=4,
            loop_sha256=C.sha(C.HERE.parent/'b92_reduced/train92.py'))
        z['config_hash']=C.digest(z);return z
    T.config=config;T.verify_complete=lambda rd:C.verify_run(stage,seed,code,observe)
    records=[];train_entries=[];overhead=0.;best_short=float('inf');short_state=None;short_epoch=None;clones=0;train_started=None
    original_batches=T.batches;original_validate=T.validate
    if observe:
        def batches(*args,**kwargs):
            base=original_batches(*args,**kwargs)
            if kwargs.get('shuffle_seed') is None:return base
            stamps=[];train_entries.append(stamps)
            class Trace:
                def __iter__(self):
                    nonlocal overhead
                    for b in base:
                        start=time.perf_counter();stamps.append(start);overhead+=time.perf_counter()-start;yield b
            return Trace()
        T.batches=batches
        def validate(model,val,deadline=None):
            nonlocal overhead,best_short,short_state,short_epoch,clones,train_started
            assert deadline is not None
            begins=time.perf_counter();before=rng_hash();overhead+=time.perf_counter()-begins
            stamps=[]
            class Trace:
                def __init__(self,source):self.source=source
                def __iter__(self):
                    nonlocal overhead
                    for b in self.source:
                        t=time.perf_counter();stamps.append(t);overhead+=time.perf_counter()-t;yield b
            value=original_validate(model,{k:Trace(v) for k,v in val.items()},deadline)
            t=time.perf_counter();t0=deadline-(120 if stage=='smoke' else C.LONG)
            if train_started is None:train_started=time.time()-(time.perf_counter()-t0)
            if value is not None:
                epoch=len(records);assert len(train_entries)==epoch+1 and train_entries[-1] and stamps
                mt=max(train_entries[-1])-t0;mv=max(stamps)-t0;eligible=max(mt,mv)<C.SHORT
                if eligible and value[0]<best_short:
                    best_short=value[0];short_epoch=epoch;short_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};clones+=1
                same=before==rng_hash();assert same,'Observer changed RNG'
                records.append({'epoch':epoch,'val_loss':value[0],'max_train_start':mt,'max_val_start':mv,
                    'completed_at':time.perf_counter()-t0,'eligible':eligible,'boundary_flag':any(abs(x-t0-C.SHORT)<=.001 for x in (*train_entries[-1],*stamps)),'rng_unchanged':same})
            else:assert before==rng_hash()
            overhead+=time.perf_counter()-t;assert overhead<=6,'Observer overhead limit exceeded'
            return value
        T.validate=validate
    T.train(stage,C.UNIT,seed,code)
    rd=C.run_dir(stage,seed,code,observe);cfg=C.load(rd/'config.json');metrics=C.load(rd/'metrics.json')
    if not observe:
        assert stage=='smoke' and metrics['wall_seconds']<C.SHORT
        short_state=torch.load(rd/'best.pt',map_location='cpu',weights_only=True)
        short_epoch=min(range(len(metrics['history'])),key=lambda i:metrics['history'][i]['val_loss']);best_short=metrics['history'][short_epoch]['val_loss']
    assert short_state is not None and short_epoch is not None
    tmp=rd/'short_best.tmp';torch.save(short_state,tmp);tmp.replace(rd/'short_best.pt')
    C.save(rd/'observer.json',{'config_hash':cfg['config_hash'],'short_epoch':short_epoch,'short_val_loss':best_short,
        'short_sha256':C.sha(rd/'short_best.pt'),'epochs':records,'overhead_seconds':overhead,'clones':clones,'rng_unchanged':True,'train_started_unix':train_started})
    C.guard()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('smoke','confirm'),required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--code',choices=C.CODES,required=True);p.add_argument('--observe',type=int,choices=(0,1),default=1);a=p.parse_args()
    main(a.stage,a.seed,a.code,bool(a.observe))
