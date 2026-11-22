"""One dropout-.1 actual full training+seven-validation epoch, ordinary exit only."""
import argparse,hashlib,json,os,platform,random,sys,time
from pathlib import Path
import profile93 as P
def main():
    p=argparse.ArgumentParser();p.add_argument('--path',required=True,choices=P.PATHS);p.add_argument('--rep',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-existing-cpu',action='store_true');a=p.parse_args();cfg=P.PATHS[a.path];pins=P.registered();binding=P.prepared_binding()
    if cfg['device']=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    else:P.gpu_ready(a.allow_existing_cpu)
    sys.path.insert(0,str(P.ROOT/'ces_prediction/experiments/b92_reduced'))
    import numpy as np
    import torch
    import common92 as M
    from prepare92 import batches,epoch_segments
    from model92 import create_model
    from runtime92 import configure,synchronize
    from train92 import vector_loss,validate
    start=time.perf_counter();prepared=torch.load(P.ROOT/binding['prepared_path'],map_location='cpu',weights_only=False)
    assert prepared['key']==binding['prepared_key'] and prepared['identity']['sources']==M.sources()
    for name,h in prepared['identity']['raw'].items():assert P.sha(P.DATA/'.b73'/name)==h
    assert len(prepared['keys'])==len(set(prepared['keys'])) and prepared['keys']==sorted(prepared['keys'])
    random.seed(1701);np.random.seed(1701);torch.manual_seed(1701)
    configure(cfg);model=create_model('C',cfg,1701).to(cfg['device']);configure(cfg)
    assert torch.backends.cudnn.enabled is False and torch.get_num_threads()==cfg['threads'] and torch.backends.mkldnn.enabled==cfg['mkldnn']
    if cfg['device']=='cuda':assert next(model.parameters()).is_cuda and torch.cuda.is_available()
    assert model.base.lstm_ti.dropout==.1
    initial=hashlib.sha256()
    for name,v in model.state_dict().items():initial.update(name.encode());initial.update(v.detach().cpu().contiguous().numpy().tobytes())
    train=epoch_segments(prepared,'V1_r2_k0',1701,0)
    train_batches=batches(train,'cpu',shuffle_seed=1701)
    val={rule:batches(prepared['val'][rule],'cpu') for rule in M.TRAIN_RULES}
    digest=hashlib.sha256();train_counts=[0,0];val_counts={}
    def bind(bb):
        for b in bb:
            for name in ('x','y','mask','lengths'):digest.update(b[name].contiguous().numpy().tobytes())
    bind(train_batches)
    for b in train_batches:
        counts=b['mask'].sum((0,1)).tolist();train_counts=[train_counts[h]+counts[h] for h in range(2)]
    for rule,bb in val.items():
        digest.update(rule.encode());bind(bb);val_counts[rule]=[sum(int(b['mask'][:,:,h].sum()) for b in bb) for h in range(2)]
    batch_binding={'sha256':digest.hexdigest(),'train_batches':len(train_batches),'train_segments':len(train),'train_rows':sum(len(s['x']) for s in train),'train_supervised_counts':train_counts,'validation_batches':{r:len(bb) for r,bb in val.items()},'validation_supervised_counts':val_counts,'initial_weights_sha256':initial.hexdigest(),'seed':1701,'epoch':0,'shuffle_seed':1701}
    assert all(n>0 for n in train_counts) and all(all(n>0 for n in v) for v in val_counts.values())
    tensor_bytes=sum(b[k].numel()*b[k].element_size() for bb in val.values() for b in bb for k in ('x','y','mask'))
    cached=True
    if cfg['device']=='cuda':
        free,_=torch.cuda.mem_get_info()
        assert tensor_bytes<min(free*.35,512*1024**2),'Full validation cache does not fit registered device budget'
        val={r:[{k:(v.to('cuda') if k!='lengths' else v) for k,v in b.items()} for b in bb] for r,bb in val.items()}
    synchronize(cfg);prepare_seconds=time.perf_counter()-start
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    stats=prepared['identity']['stats'];mu=torch.tensor(stats['ti_mu'],device=cfg['device']);sd=torch.tensor(stats['ti_sd'],device=cfg['device'])
    model.train();assert model.base.lstm_ti.training and model.base.lstm_ti.dropout==.1
    synchronize(cfg);epoch_start=time.perf_counter();train_start=epoch_start
    # The production loop collates once per epoch; include that actual host work.
    train_batches=batches(epoch_segments(prepared,'V1_r2_k0',1701,0),'cpu',shuffle_seed=1701)
    for i,host in enumerate(train_batches):
        b={k:(v.to(cfg['device']) if k!='lengths' else v) for k,v in host.items()};optimizer.zero_grad(set_to_none=True)
        pred=model(b['x'],b['lengths']);loss=vector_loss(pred,b['y'],b['mask'],mu,sd)
        assert torch.isfinite(loss),'Nonfinite actual training loss'
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if (i+1)%20==0:print(json.dumps({'stage':'train','completed_batches':i+1,'total_batches':len(train_batches)}),flush=True)
    synchronize(cfg);training_seconds=time.perf_counter()-train_start;validation_start=time.perf_counter()
    validation=validate(model,val,None);assert validation is not None
    objective,details=validation;assert np.isfinite(objective) and all(np.isfinite(v['squared_sums']).all() for v in details.values())
    assert {r:[int(v) for v in z['counts']] for r,z in details.items()}==val_counts
    synchronize(cfg);validation_seconds=time.perf_counter()-validation_start;epoch_seconds=time.perf_counter()-epoch_start
    assert all(torch.isfinite(q).all() for q in model.parameters())
    assert P.sources()==pins and P.prepared_binding()==binding,'Source/cache changed during full epoch'
    report={'all_pass':True,'path':a.path,'rep':a.rep,'config':cfg,'sources':pins,'input_binding':binding,'actual_batch_binding':batch_binding,
            'timing':{'prepare_cache_seconds':prepare_seconds,'training_seconds':training_seconds,'seven_rule_validation_seconds':validation_seconds,'epoch_seconds':epoch_seconds,'total_prepare_epoch_seconds':prepare_seconds+epoch_seconds,'warmup':'none; actual first epoch includes backend initialization','validation_cached_on_device':cfg['device']=='cuda','validation_tensor_bytes':tensor_bytes},
            'software':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn_enabled':torch.backends.cudnn.enabled,'mkldnn_enabled':torch.backends.mkldnn.enabled,'threads':torch.get_num_threads(),'gpu_name':torch.cuda.get_device_name(0) if cfg['device']=='cuda' else None,'platform':platform.platform()},
            'checks':{'training_updates':len(train_batches),'all_training_losses_finite':True,'all_validation_sums_finite':True,'all_parameters_finite':True,'actual_dropout':.1,'validation_rule_count':len(details),'scientific_loss_values_exported':False,'test_inputs_read':False}}
    P.save(a.output,report);print(json.dumps({'stage':'complete','path':a.path,'epoch_seconds':epoch_seconds,'train_batches':len(train_batches)}),flush=True)
if __name__=='__main__':main()
