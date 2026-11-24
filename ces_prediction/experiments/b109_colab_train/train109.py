"""Same F/D10 trajectories, fully validated optimizer-state Colab continuation."""
import argparse,hashlib,json,os,random,shutil,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np,torch
from portable108 import TemperatureLSTM,loss108
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,z):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(z,indent=2),encoding='utf8');t.replace(p)
def batches(d,split,epoch=None,device='cuda'):
    lengths=d[split+'_lengths'];starts=np.r_[0,np.cumsum(lengths)];order=list(range(len(lengths)))
    if epoch is not None:random.Random(epoch*10000+1701).shuffle(order)
    for offset in range(0,len(order),16):
        ids=order[offset:offset+16];n=int(max(lengths[i] for i in ids));x=np.zeros((len(ids),n,22),np.float32);y=np.zeros((len(ids),n,2),np.float32);m=np.zeros((len(ids),n,2),bool)
        for j,i in enumerate(ids):
            a,b=int(starts[i]),int(starts[i+1]);k=b-a;x[j,:k]=d[split+'_x'][a:b];y[j,:k]=d[split+'_y'][a:b];m[j,:k]=d[split+'_mask'][a:b]
        yield torch.from_numpy(x).to(device),torch.from_numpy(y).to(device),torch.from_numpy(m).to(device)
def validate(model,d):
    model.eval();s=torch.zeros(2,device='cuda');n=torch.zeros(2,dtype=torch.int64,device='cuda')
    with torch.no_grad():
        for x,y,m in batches(d,'val'):
            e=torch.where(m,model(x)-y,torch.zeros_like(y));s+=e.square().sum((0,1));n+=m.sum((0,1))
    ss=s.cpu().tolist();nn=n.cpu().tolist();assert nn[0]>0 and nn[1]==0 and all(np.isfinite(ss))
    return ss[0]/nn[0],{'D10':{'squared_sums':ss,'counts':nn,'mse':ss[0]/nn[0]}}
def fit(tag):
    manifest=json.loads((ROOT/'manifest.json').read_text());entry=manifest['tasks'][tag]
    for name in ('train109.py','portable108.py',tag+'.npz',entry['weights']):assert sha(ROOT/name)==manifest['files'][name]
    torch.set_num_threads(2);torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.mkldnn.enabled=False
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
    torch.manual_seed(1701);random.seed(1701);np.random.seed(1701)
    p=ROOT/'runs'/tag;p.mkdir(parents=True,exist_ok=True);assert not (p/'config.json').exists(),'No blind retry'
    with np.load(ROOT/(tag+'.npz'),allow_pickle=False) as f:d={k:f[k] for k in f.files}
    assert not d['train_mask'][:,1].any() and not d['val_mask'][:,1].any()
    initial=torch.load(ROOT/entry['weights'],map_location='cpu',weights_only=False)
    model=TemperatureLSTM().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2)
    history=[];best=float('inf');ni=0;updates=0;start_epoch=0
    if entry['mode']=='resume':
        model.load_state_dict(initial['model']);opt.load_state_dict(initial['optimizer']);scheduler.load_state_dict(initial['scheduler']);history=initial['history'];best=min(h['val_loss'] for h in history);start_epoch=initial['epoch']+1;updates=initial['updates']
        for h in reversed(history):
            if h['improved']:break
            ni+=1
        assert ni<6 and len(history)==start_epoch
    else:model.load_state_dict(initial)
    model.eval()
    with torch.no_grad():assert np.allclose(model(torch.from_numpy(d['ref_x']).cuda()).cpu().numpy(),d['ref_prediction'],atol=1e-4,rtol=1e-4)
    readiness={'prefix_equal':True,'mode':entry['mode'],'start_epoch':start_epoch,'lr':opt.param_groups[0]['lr'],'ni':ni}
    if entry['mode']=='resume':
        model.load_state_dict(torch.load(ROOT/entry['best'],map_location='cpu',weights_only=True));value,_=validate(model,d)
        assert abs(value-best)<=1e-4*max(1,abs(best)),'Preserved best/VAL mismatch'
        readiness.update(historical_best=best,restored_best_val=value);model.load_state_dict(initial['model']);shutil.copyfile(ROOT/entry['best'],p/'best.pt')
        torch.set_rng_state(initial['rng_cpu'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in initial['rng_cuda']]);np.random.set_state(initial['rng_numpy']);random.setstate(initial['rng_python'])
    cfg={'registration':'PREREGISTRATION_B109.md','tag':tag,'entry':entry,'mode':entry['mode'],'start_epoch':start_epoch,'seed':1701,'dropout':.1,'batch':16,'dtype':'float32','cudnn':True,'tf32':False,'epoch_guard':400,'seconds_guard':86400,'patience':6,'manifest_sha256':sha(ROOT/'manifest.json'),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'env':{k:v for k,v in os.environ.items() if k.startswith('CES_')}}
    save(p/'config.json',cfg);save(p/'readiness.json',readiness);started=time.perf_counter();stop='emergency_epoch_guard';mu=torch.tensor(float(d['mu']),device='cuda');sd=torch.tensor(float(d['sd']),device='cuda')
    for epoch in range(start_epoch,400):
        model.train();total=torch.zeros((),device='cuda');nb=0;interrupted=False
        for x,y,m in batches(d,'train',epoch):
            if time.perf_counter()-started>86400:interrupted=True;break
            opt.zero_grad(set_to_none=True);loss=loss108(model(x),y,m,mu,sd);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();total+=loss.detach();nb+=1;updates+=1
        if interrupted:stop='emergency_time_guard';break
        value,detail=validate(model,d);assert np.isfinite(value);scheduler.step(value);improved=value<best
        if improved:
            best=value;ni=0;t=p/'best.tmp';torch.save(model.state_dict(),t);t.replace(p/'best.pt')
        else:ni+=1
        history.append({'epoch':epoch,'train_loss':float((total/max(nb,1)).cpu()),'val_loss':value,'validation':detail,'improved':improved,'updates':updates,'lr':opt.param_groups[0]['lr']})
        state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'updates':updates,'history':history,'best_val':best,'ni':ni,'validated':True,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_numpy':np.random.get_state(),'rng_python':random.getstate(),'config_sha256':sha(p/'config.json')}
        t=p/'latest.tmp';torch.save(state,t);t.replace(p/'latest.pt');save(p/'progress.json',{'epoch':epoch+1,'new_epochs':len(history)-start_epoch,'ni':ni,'best_val':best,'updated':time.time()});print(json.dumps(history[-1]),flush=True)
        if ni>=6:stop='patience';break
    assert len(history)>start_epoch and (p/'best.pt').exists()
    save(p/'metrics.json',{'history':history,'stop':stop,'trailing_no_improvement':ni,'completed_epochs':len(history),'new_epochs':len(history)-start_epoch,'updates':updates,'wall_seconds':time.perf_counter()-started})
    save(p/'complete.json',{'best_val':best,'best_sha256':sha(p/'best.pt'),'latest_sha256':sha(p/'latest.pt'),'config_sha256':sha(p/'config.json'),'patience_met':stop=='patience' and ni>=6})
def verify(tag):
    p=ROOT/'runs'/tag;m=json.loads((p/'metrics.json').read_text());c=json.loads((p/'complete.json').read_text());exit=json.loads((p/'process_exit.json').read_text());assert exit['returncode']==0 and not exit['timeout'] and c['patience_met'] and m['stop']=='patience'
    for name,key in [('best.pt','best_sha256'),('latest.pt','latest_sha256'),('config.json','config_sha256')]:assert sha(p/name)==c[key]
    history=m['history'];assert min(e['val_loss'] for e in history)==c['best_val'];ni=0
    for i,e in enumerate(history):
        assert e['epoch']==i;v=e['validation']['D10'];assert v['counts'][0]>0 and v['counts'][1]==0
        assert np.isclose(v['squared_sums'][0]/v['counts'][0],e['val_loss'],rtol=1e-10)
    for e in reversed(history):
        if e['improved']:break
        ni+=1
    assert ni>=6
    return {'epochs':len(history),'new_epochs':m['new_epochs'],'best_val':c['best_val'],'wall_seconds':m['wall_seconds']}
def run():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert sha(ROOT/name)==h
    pending=list(manifest['tasks']);active=[];done={};failure=None;(ROOT/'runs').mkdir(exist_ok=True)
    while pending or active:
        for e in list(active):
            timeout=time.time()-e['started']>87000
            if timeout and e['proc'].poll() is None:e['proc'].kill()
            if e['proc'].poll() is None:continue
            rc=e['proc'].wait();e['log'].close();save(e['path']/'process_exit.json',{'returncode':rc,'timeout':timeout,'elapsed':time.time()-e['started']})
            try:done[e['tag']]=verify(e['tag'])
            except Exception as error:failure=failure or repr(error)
            active.remove(e)
        if failure:pending=[]
        while pending and len(active)<4 and not failure:
            free,_=torch.cuda.mem_get_info()
            if free<1630*1024**2:break
            tag=pending.pop(0);p=ROOT/'runs'/tag;p.mkdir(exist_ok=True);assert not (p/'child_start.json').exists();log=(p/'process.log').open('wb')
            env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'train109.py'),'--case',tag],stdout=log,stderr=subprocess.STDOUT,env=env);started=time.time();save(p/'child_start.json',{'pid':proc.pid,'started':started,'tag':tag})
            active.append({'proc':proc,'log':log,'path':p,'tag':tag,'started':started})
        save(ROOT/'status.json',{'state':'training' if active or pending else 'failed' if failure else 'complete','completed':len(done),'total':len(manifest['tasks']),'pending':len(pending),'active':[{'tag':e['tag'],'pid':e['proc'].pid} for e in active],'failure':failure,'updated':time.time()})
        if active or pending:time.sleep(1)
    if failure:raise RuntimeError(failure)
    save(ROOT/'fit_audit.json',{'all_pass':True,'fits':done,'manifest_sha256':sha(ROOT/'manifest.json')})
    with zipfile.ZipFile(ROOT/'b109_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT/'runs').rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
        for name in ('fit_audit.json','status.json','manifest.json'):z.write(ROOT/name,name)
    print('All nine Colab paths passed normal-exit/patience audit.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case');a=p.parse_args();fit(a.case) if a.case else run()
