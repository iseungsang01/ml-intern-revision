"""Same numeric objective and ordering, restoring validated optimizer state."""
import argparse,os,random,shutil,time
import numpy as np,torch
import common107 as C
from model106 import create_model
from prepare92 import batches
from train92 import vector_loss,validate
from runtime92 import configure,synchronize

def train(case,mode):
    C.guard();C.inputs_guard();assert all(os.environ.get(k)==v for k,v in C.env(case).items())
    rt=C.runtime();configure(rt);prep=C.prepared(case);p=C.new_rd(case);assert not (p/'config.json').exists(),'No blind retry'
    parent=None;old_history=[];best=float('inf');ni=0;updates=0;parent_wall=0.;start_epoch=0
    torch.manual_seed(C.SEED);random.seed(C.SEED);np.random.seed(C.SEED)
    model=create_model('LSTM_F',rt,C.SEED).to('cuda');configure(rt)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2)
    if mode=='resume':
        C.OLD.verify_run(case);old=C.OLD.rd(case);m=C.load(old/'metrics.json');assert m['stop'] in ('budget_limited','epoch_cap')
        w=torch.load(old/'latest.pt',map_location='cpu',weights_only=False)
        assert w['validated'];model.load_state_dict(w['model']);opt.load_state_dict(w['optimizer']);sch.load_state_dict(w['scheduler'])
        old_history=m['history'];start_epoch=w['epoch']+1;updates=w['updates'];best=min(e['val_loss'] for e in old_history);parent_wall=m['wall_seconds']
        for e in reversed(old_history):
            if e['improved']:break
            ni+=1
        assert start_epoch==len(old_history) and ni<6
        assert sch.state_dict()==w['scheduler'] and opt.param_groups[0]['lr']==w['optimizer']['param_groups'][0]['lr']
        for k,s in w['optimizer']['state'].items():
            for n,v in s.items():
                if torch.is_tensor(v):assert torch.equal(opt.state_dict()['state'][k][n].cpu(),v.cpu())
        parent={'path':old.relative_to(C.ROOT).as_posix(),'epoch':start_epoch,'validated_updates':updates,
                'hashes':{(old/n).relative_to(C.ROOT).as_posix():C.sha(old/n) for n in ('config.json','metrics.json','best.pt','latest.pt','complete.json','process_exit.json')},
                'rng_note':'Old RNG not saved; declared deterministic resumed dropout stream, not bitwise uninterrupted reproduction.'}
        shutil.copyfile(old/'best.pt',p/'best.pt');torch.manual_seed(C.resume_seed(case));random.seed(C.resume_seed(case));np.random.seed(C.resume_seed(case)%2**32)
    else:assert not (C.OLD.rd(case)/'child_start.json').exists(),'Fresh path was already attempted'
    cfg={'experiment':'B107','case':case,'mode':mode,'seed':C.SEED,'resume_seed':C.resume_seed(case) if parent else None,'start_epoch':start_epoch,
         'epochs':C.EPOCH_CAP,'seconds':C.CHILD_SECONDS,'sources':C.sources(),'frozen':C.guard(),'env':C.env(case),'parent':parent,
         'prepared_key':prep['key'],'prepared_sha256':prep['artifact_sha256'],'runtime':rt,'batch':16,'lr':.001,'weight_decay':.0001,
         'clip':1.,'early_stop':6,'scheduler_patience':2,'scheduler_factor':.5,'rule':'D10','target':'F'}
    cfg['config_hash']=C.digest(cfg);C.save(p/'config.json',cfg)
    for name in ('manifest','stats'):C.save(p/(name+'.json'),prep['identity'][name])
    C.save(p/'resume_checks.json',{'passed':True,'start_epoch':start_epoch,'updates':updates,'trailing_no_improvement':ni,'lr':opt.param_groups[0]['lr'],'optimizer_moments_restored':bool(parent),'parent':parent})
    val=batches(prep['val']['D10'],'cpu');stats=prep['identity']['stats'];mu=torch.tensor(stats['ti_mu'],device='cuda');sd=torch.tensor(stats['ti_sd'],device='cuda')
    class Transfer:
        def __iter__(self):
            for b in val:yield {k:v.to('cuda') if k!='lengths' else v for k,v in b.items()}
    history=list(old_history);synchronize(rt);started=time.perf_counter();deadline=started+C.CHILD_SECONDS;stop='emergency_epoch_cap';validated_updates=updates
    for epoch in range(start_epoch,C.EPOCH_CAP):
        model.train();loss_sum=torch.zeros((),device='cuda');nb=0;interrupted=False
        for host in batches(prep['train']['D10'],'cpu',shuffle_seed=epoch*10000+C.SEED):
            if time.perf_counter()>=deadline:interrupted=True;break
            b={k:v.to('cuda') if k!='lengths' else v for k,v in host.items()};opt.zero_grad(set_to_none=True)
            pred=model(b['x'],b['lengths']);loss=vector_loss(pred,b['y'],b['mask'],mu,sd)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            loss_sum+=loss.detach();nb+=1;updates+=1
        if interrupted:stop='emergency_time_guard';break
        validated=validate(model,{'D10':Transfer()},deadline)
        if validated is None:stop='emergency_time_guard';break
        value,detail=validated;assert np.isfinite(value);sch.step(value);improved=value<best
        if improved:
            best=value;ni=0;tmp=p/'best.tmp';torch.save(model.state_dict(),tmp);tmp.replace(p/'best.pt')
        else:ni+=1
        history.append({'epoch':epoch,'train_loss':float((loss_sum/max(nb,1)).cpu()),'val_loss':value,'validation':detail,'improved':improved,'updates':updates,'lr':opt.param_groups[0]['lr']})
        validated_updates=updates
        state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'epoch':epoch,'validated':True,'config_hash':cfg['config_hash'],'updates':updates,
               'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_numpy':np.random.get_state(),'rng_python':random.getstate()}
        tmp=p/'latest.tmp';torch.save(state,tmp);tmp.replace(p/'latest.pt');print(__import__('json').dumps(history[-1]),flush=True)
        if ni>=6:stop='patience';break
    assert len(history)>start_epoch and (p/'best.pt').exists(),'No additional fully validated epoch'
    synchronize(rt);wall=time.perf_counter()-started
    C.save(p/'metrics.json',{'history':history,'completed_epochs':len(history),'new_epochs':len(history)-start_epoch,'stop':stop,'trailing_no_improvement':ni,
           'updates':updates,'validated_updates':validated_updates,'discarded_incomplete_epoch_updates':updates-validated_updates,'wall_seconds':parent_wall+wall,'extension_wall_seconds':wall})
    C.save(p/'complete.json',{'config_hash':cfg['config_hash'],'best_val_loss':best,'stop':stop,'best_sha256':C.sha(p/'best.pt'),'latest_sha256':C.sha(p/'latest.pt')})
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--case',required=True,choices=C.CASES);a.add_argument('--mode',required=True,choices=('fresh','resume'));z=a.parse_args();train(z.case,z.mode)
