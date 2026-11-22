"""Process-local bindings to frozen B85 training, with validation MSE unchanged."""
import importlib.util,json
from pathlib import Path
import common90 as M
import adapter88 as A88
pid_alive,acquire_file_lock,release_file_lock=A88.pid_alive,A88.acquire_file_lock,A88.release_file_lock
_TRAINERS={}

def input_ready():
    M.assert_registered();M.assert_frozen()
    c=json.loads((M.CACHE/'input_checks.json').read_text(encoding='utf8'))
    assert c['all_pass'] and c['execution_sources']==M.source_inventory(),'Stale B90 input checks'
    return True

def dependency_ready():
    M.assert_frozen()
    p=M.DATA/'.b88/queue_status.json'
    if not p.is_file():return False
    state=json.loads(p.read_text(encoding='utf8'))['state']
    if state in ('failed','blocked_dependency'):raise RuntimeError('B88 dependency failed')
    if state!='complete':return False
    A88.guarded_ready()
    return True

def activate(method):
    assert method in M.METHODS;M.assert_frozen()
    if method in _TRAINERS:return _TRAINERS[method]
    from models90 import create_model
    spec=importlib.util.spec_from_file_location('_b90_'+method+'_frozen_train85',M.HERE.parent/'b85_unified/train85.py')
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    t.CACHE=M.CACHE
    current={}
    def factory(variant='X',init_seed=1701):
        model=create_model(method=method,variant=variant,init_seed=init_seed)
        original=model.forward
        def forward(*a,**kw):
            current['model']=model
            return original(*a,**kw)
        model.forward=forward
        return model
    t.create_model=factory
    t.ENV_PINS={**t.ENV_PINS,'CES_B90_METHOD':method,'CES_B90_OUTPUT_DIR':str(M.CACHE)}
    t.get_run_dir=lambda unit,seed,var,smoke=False:M.CACHE/('smoke' if smoke else 'runs')/method/f'{unit}__{seed}__{var}'
    def config_hash(config,unit):
        from models90 import ARCHITECTURES
        config['experiment']='B90';config['method']=method;config['architecture']=ARCHITECTURES[method]
        config['stage']='smoke' if config['smoke'] else 'screen' if config['init']==1701 else 'promoted'
        config['training_loss']={'U':'Huber delta1','V':'Student-t df4 scale1','N':'Gaussian logvariance clamp[-8,8]'}.get(method,'MSE')
        config['validation_loss']='pure MSE';config['negative_penalty']=.1
        clean={k:v for k,v in config.items() if k!='config_hash'}
        return M.hashlib.sha256(json.dumps({'config':clean,'unit':unit,'method':method,'sources':M.source_inventory(),'frozen':M.assert_frozen()},sort_keys=True).encode()).hexdigest()
    t._config_hash=config_hash
    t.acquire_lock=lambda rd:acquire_file_lock(Path(rd)/'.lock')
    t.release_lock=lambda rd:release_file_lock(Path(rd)/'.lock')
    if method in 'UVN':
        def loss(pred,y,mask,mu,sd,device):
            torch=t.torch;parts=[];penalties=[]
            for h in range(2):
                keep=mask[:,:,h]
                if not keep.any():continue
                p=pred[:,:,h][keep];res=p-y[:,:,h][keep]
                if method=='U':v=torch.nn.functional.huber_loss(p,y[:,:,h][keep],delta=1.,reduction='mean')
                elif method=='V':v=(2.5*torch.log1p(res.square()/4.)).mean()
                else:
                    lv=current['model'].latest_logvar[:,:,h][keep]
                    v=(.5*(lv+res.square()*torch.exp(-lv))).mean()
                parts.append(v);penalties.append(torch.relu((0.-mu)/sd-p).mean())
            if not parts:return torch.tensor(0.,device=device,requires_grad=True)
            return sum(parts)/len(parts)+.1*sum(penalties)/len(penalties)
        t.compute_loss=loss
    _TRAINERS[method]=t
    return t

def guarded_ready():
    input_ready();record=M.promotion()
    from run90 import validation_record
    assert validation_record()==record,'Promotion validation inputs changed'
    for m,u,s,v in M.all_jobs():
        t=activate(m);assert t.verify_complete(t.get_run_dir(u,s,v)),'Required B90 fit absent'
    return True
