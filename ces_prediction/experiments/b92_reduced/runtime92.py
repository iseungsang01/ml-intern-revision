"""Consume the immutable B91 qualification without a silent fallback."""
import json,platform,sys
import torch
import common92 as M
def decision():
    M.guard();path=M.DATA/'.b91/runtime_decision.json';z=json.loads(path.read_text(encoding='utf8'))
    import importlib.util
    verifier=M.HERE.parent/'b91_fast/runtime91.py'
    sys.path.insert(0,str(verifier.parent))
    s=importlib.util.spec_from_file_location('_b92_decision91',verifier);v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
    verified=v.verify_decision();assert verified==z
    assert z.get('selected_path') and z.get('native_clean'),'No eligible B91 runtime'
    config=z['selected_config'];assert config['device'] in ('cpu','cuda') and isinstance(config['dense'],bool)
    return {**config,'decision_sha256':M.sha(path),'decision':z}
def configure(z):
    torch.backends.cudnn.enabled=bool(z.get('cudnn',False));torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(int(z.get('threads',1)))
    torch.backends.mkldnn.enabled=bool(z.get('mkldnn',True))
    if z['device']=='cuda':assert torch.cuda.is_available(),'Qualified CUDA unavailable'
def synchronize(z):
    if z['device']=='cuda':torch.cuda.synchronize()
