"""Bounded fresh trajectories; only fully validated epochs can produce a best."""
import argparse,json,os,random,time
import numpy as np
import torch
import common92 as M
from adapter88 import acquire_file_lock,release_file_lock
from prepare92 import prepare_unit,batches,epoch_segments
from runtime92 import decision,configure,synchronize
from model92 import create_model
ENV_PINS={'CES_DATA_DIR':str(M.DATA),'CES_DROP_STUCK_TARGETS':'1','CES_TI_SPIKE_CUT_EV':'3000','CES_PER_SHOT_NORM':'0','CES_SEQ_NORM_EXPANDING':'0','CES_SEQ_ZERO_MC':'1','CES_SEQ_ZERO_VT_HISTORY':'1','CES_SEQ_ZERO_VT':'1','CES_SEQ_BATCH':'16','CES_LR':'.001','CES_SEQ_TRAIN_CTX':'0','CES_SEQ_EVAL_CTX':'0'}
ENV_PINS.update({'CES_SEQ_LOSS':'mse','CES_SEQ_SELECT':'mse','CES_SEQ_FIXED_EPOCHS':'0'})
def effective_env(stage,code,runtime):
    cap=1 if stage.startswith('smoke') else 12 if stage=='pilot' else 40
    return {**ENV_PINS,'CES_SEQ_EPOCHS':str(cap),'CES_SEQ_DEVICE':runtime['device'],'CES_B92_CODE':code,'CES_B92_OUTPUT_DIR':str(M.CACHE)}
def vector_loss(pred,y,mask,mu,sd):
    m=mask.to(pred.dtype);count=m.sum((0,1));has=(count>0).to(pred.dtype)
    error=torch.where(mask,pred-y,torch.zeros_like(pred))
    mse=error.square().sum((0,1))/count.clamp_min(1)
    neg=(torch.relu(-mu/sd-pred)*m).sum((0,1))/count.clamp_min(1)
    return ((mse+.1*neg)*has).sum()/has.sum().clamp_min(1)
def validate(model,val,deadline=None):
    model.eval();values=[];details={}
    with torch.no_grad():
        for rule,bb in val.items():
            sums=None;counts=None
            for b in bb:
                if deadline is not None and time.perf_counter()>=deadline:return None
                p=model(b['x'],b['lengths']);e=torch.where(b['mask'],p-b['y'],torch.zeros_like(p))
                s=e.square().sum((0,1));n=b['mask'].sum((0,1))
                sums=s if sums is None else sums+s;counts=n if counts is None else counts+n
            assert sums is not None
            ss=sums.detach().cpu().numpy();nn=counts.detach().cpu().numpy();valid=nn>0;assert valid.any()
            assert np.isfinite(ss).all();v=float((ss[valid]/nn[valid]).mean());values.append(v)
            details[rule]={'squared_sums':ss.tolist(),'counts':nn.tolist(),'mse':v}
    return float(np.mean(values)),details
def config(stage,unit,seed,code,runtime,prepared):
    epochs,seconds=(1,120) if stage.startswith('smoke') else (12,600) if stage=='pilot' else (40,1200)
    z={'stage':stage,'unit':unit,'seed':seed,'code':code,'epochs':epochs,'seconds':seconds,'runtime':runtime,'prepared_key':prepared['key'],'sources':M.sources(),'frozen':M.guard(),'batch':16,'lr':.001,'weight_decay':.0001,'clip':1.,'dropout':.1,'early_stop':6,'scheduler_patience':2,'factor':.5,'env':ENV_PINS}
    z['env']=effective_env(stage,code,runtime)
    z['manifest_hash']=M.digest(prepared['identity']['manifest']);z['stats_hash']=M.digest(prepared['identity']['stats']);z['prepared_artifact_sha256']=prepared['artifact_sha256']
    z['config_hash']=M.digest(z);return z
def verify_complete(rd):
    M.guard();rd=M.Path(rd) if hasattr(M,'Path') else rd
    if not (rd/'complete.json').exists():return False
    z=json.loads((rd/'config.json').read_text());h=z.pop('config_hash');assert h==M.digest(z) and z['sources']==M.sources() and z['frozen']==M.guard()
    assert z['runtime']['decision_sha256']==M.sha(M.DATA/'.b91/runtime_decision.json')
    assert M.sha(M.CACHE/'prepared'/f"{z['unit']}__{z['prepared_key']}.pt")==z['prepared_artifact_sha256']
    complete=json.loads((rd/'complete.json').read_text());exit=json.loads((rd/'process_exit.json').read_text())
    assert complete['config_hash']==h and exit['config_hash']==h and exit['returncode']==0
    metrics=json.loads((rd/'metrics.json').read_text());assert metrics['history'] and np.isfinite(complete['best_val_loss'])
    assert M.digest(json.loads((rd/'manifest.json').read_text()))==z['manifest_hash']
    assert M.digest(json.loads((rd/'stats.json').read_text()))==z['stats_hash']
    assert complete['best_val_loss']==min(x['val_loss'] for x in metrics['history'])
    best=torch.load(rd/'best.pt',map_location='cpu',weights_only=True);latest=torch.load(rd/'latest.pt',map_location='cpu',weights_only=False)
    assert best and latest['config_hash']==h and latest['validated'] and latest['epoch']==metrics['history'][-1]['epoch']
    assert M.sha(rd/'best.pt')==complete['best_sha256'] and M.sha(rd/'latest.pt')==complete['latest_sha256']
    return True
def checks_ready():
    z=json.loads((M.CACHE/'input_checks.json').read_text());assert z['all_pass'] and z['sources']==M.sources()
def train(stage,unit,seed,code):
    M.guard();checks_ready();runtime=decision();configure(runtime)
    assert all(os.environ.get(k)==v for k,v in effective_env(stage,code,runtime).items()),'Explicit effective CES environment required; use run92 supervisor'
    rd=M.run_dir(stage,unit,seed,code);assert acquire_file_lock(rd/'.lock'),'Run lock held'
    try:
        if (rd/'complete.json').exists():assert verify_complete(rd);return
        assert not (rd/'config.json').exists(),'Incomplete previous child; no blind retry'
        prep_started=time.perf_counter();prepared=prepare_unit(unit);cfg=config(stage,unit,seed,code,runtime,prepared)
        M.save(rd/'config.json',cfg);M.save(rd/'manifest.json',prepared['identity']['manifest']);M.save(rd/'stats.json',prepared['identity']['stats'])
        if stage.startswith('smoke'):
            def prefixes(rows):
                selected=[s for s in rows if s['mask'].any()][:2];assert selected
                out=[]
                for s in selected:
                    hits=np.flatnonzero(s['mask'].any(1));n=min(len(s['x']),max(64,int(hits[0])+1))
                    out.append({**s,**{k:s[k][:n] for k in ('x','y','mask','gi')}})
                return out
            prepared={**prepared,'train':{r:prefixes(prepared['train'][r]) for r in M.TRAIN_RULES},'val':{r:prefixes(prepared['val'][r]) for r in M.TRAIN_RULES}}
            prepared['keys']=sorted((s['fidx'],s['segment']) for s in prepared['train'][M.TRAIN_RULES[0]])
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        smoke_started=time.perf_counter()
        model=create_model(code,runtime,seed).to(runtime['device']);configure(runtime)
        if stage.startswith('smoke'):
            from checks92 import model_contract
            model_contract(model,prepared['val'][M.TRAIN_RULES[0]],runtime)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2)
        # Cache validation batches in host memory; transfer once if free-memory permits.
        val={r:batches(prepared['val'][r],'cpu') for r in M.TRAIN_RULES}
        memory=sum(b[k].numel()*b[k].element_size() for bb in val.values() for b in bb for k in ('x','y','mask'))
        cached_on_device=False
        if runtime['device']=='cuda':
            free,_=torch.cuda.mem_get_info()
            if memory<min(free*.35,512*1024**2):
                val={r:[{k:v.to('cuda') if k!='lengths' else v for k,v in b.items()} for b in bb] for r,bb in val.items()};cached_on_device=True
        if runtime['device']!='cpu' and not cached_on_device:
            # A bounded host-backed iterable keeps the exact objective and memberships.
            class Transfer:
                def __init__(self,bb):self.bb=bb
                def __iter__(self):
                    for b in self.bb:yield {k:v.to(runtime['device']) if k!='lengths' else v for k,v in b.items()}
            val={r:Transfer(bb) for r,bb in val.items()}
        prep_seconds=time.perf_counter()-prep_started;synchronize(runtime);started=time.perf_counter();deadline=(smoke_started if stage.startswith('smoke') else started)+cfg['seconds']
        history=[];best=float('inf');ni=0;updates=0;validated_updates=0;validation_seconds=0.;stop='epoch_cap'
        stats=prepared['identity']['stats'];mu=torch.tensor(stats['ti_mu'],device=runtime['device']);sd=torch.tensor(stats['ti_sd'],device=runtime['device'])
        for epoch in range(cfg['epochs']):
            model.train();loss_sum=torch.zeros((),device=runtime['device']);nb=0;interrupted=False
            # Host collation preserves inherited exact train-batch membership/order.
            bb=batches(epoch_segments(prepared,unit,seed,epoch),'cpu',shuffle_seed=epoch*10000+seed)
            for host in bb:
                if time.perf_counter()>=deadline:interrupted=True;break
                b={k:v.to(runtime['device']) if k!='lengths' else v for k,v in host.items()}
                optimizer.zero_grad(set_to_none=True);pred=model(b['x'],b['lengths']);loss=vector_loss(pred,b['y'],b['mask'],mu,sd)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                loss_sum+=loss.detach();nb+=1;updates+=1
            if interrupted:stop='budget_limited';break
            vt=time.perf_counter();validated=validate(model,val,deadline);validation_seconds+=time.perf_counter()-vt
            if validated is None:stop='budget_limited';break
            value,detail=validated;assert np.isfinite(value);scheduler.step(value);improved=value<best
            if improved:
                best=value;ni=0;tmp=rd/'best.tmp';torch.save(model.state_dict(),tmp);tmp.replace(rd/'best.pt')
            else:ni+=1
            history.append({'epoch':epoch,'train_loss':float((loss_sum/max(nb,1)).cpu()),'val_loss':value,'validation':detail,'improved':improved,'updates':updates,'lr':optimizer.param_groups[0]['lr']})
            validated_updates=updates
            tmp=rd/'latest.tmp';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'validated':True,'config_hash':cfg['config_hash'],'updates':updates},tmp);tmp.replace(rd/'latest.pt')
            print(json.dumps(history[-1]),flush=True)
            if ni>=6:stop='converged';break
            if time.perf_counter()>=deadline:stop='budget_limited';break
        assert history and (rd/'best.pt').exists(),'No fully validated checkpoint within budget'
        synchronize(runtime)
        metrics={'history':history,'stop':stop,'completed_epochs':len(history),'updates':updates,'validated_updates':validated_updates,'discarded_incomplete_epoch_updates':updates-validated_updates,'wall_seconds':time.perf_counter()-started,'prep_seconds':prep_seconds,'validation_seconds':validation_seconds,'validation_cached_on_device':cached_on_device,'validation_tensor_bytes':memory}
        M.save(rd/'metrics.json',metrics)
        M.save(rd/'complete.json',{'config_hash':cfg['config_hash'],'best_val_loss':best,'stop':stop,'best_sha256':M.sha(rd/'best.pt'),'latest_sha256':M.sha(rd/'latest.pt')})
    finally:release_file_lock(rd/'.lock')
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('smoke','smoke_confirm','pilot','confirm'),required=True);p.add_argument('--unit',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--code',required=True);a=p.parse_args();train(a.stage,a.unit,a.seed,a.code)
if __name__=='__main__':main()
