"""Minimal per-process adapter. Rejected Claude draft is retained in provenance cache."""
import ctypes,hashlib,importlib.util,json,os,random,sys
from pathlib import Path
import numpy as np
import common88 as M
import common85
import data88
from model88 import create_model
_T=None
def source_inventory():
    M.assert_frozen()
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(M.HERE.glob('*.py')) if p.name!='claude_author88.py'}
def input_ready():
    p=M.CACHE/'input_checks.json'
    assert p.is_file(),'Input checks missing'
    checks=json.loads(p.read_text(encoding='utf8'))
    assert checks.get('all_pass') and checks.get('execution_sources')==source_inventory(),'Input checks stale or failed'
    return True
def pid_alive(pid):
    if pid<=0:return True
    if sys.platform!='win32':return Path('/proc',str(pid)).exists()
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.OpenProcess.restype=ctypes.c_void_p
    k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    k.CloseHandle.argtypes=[ctypes.c_void_p]
    h=k.OpenProcess(0x1000,False,pid)
    if not h:return ctypes.get_last_error()!=87 # only invalid PID proves dead
    try:
        code=ctypes.c_ulong()
        return not k.GetExitCodeProcess(h,ctypes.byref(code)) or code.value==259
    finally:k.CloseHandle(h)
def acquire_file_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(2):
        try:
            fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            try:os.write(fd,str(os.getpid()).encode())
            finally:os.close(fd)
            return True
        except FileExistsError:
            try:old=int(path.read_text())
            except (ValueError,OSError):return False
            if pid_alive(old):return False
            if attempt==0:path.unlink(missing_ok=True)
    return False
def release_file_lock(path):
    path=Path(path)
    if path.exists() and path.read_text()==str(os.getpid()):path.unlink()
def make_batches(segments,batch_size,shuffle_seed=None):
    import torch
    if shuffle_seed is not None:
        indices=list(range(len(segments)));random.Random(shuffle_seed).shuffle(indices)
        segments=[segments[i] for i in indices]
    out=[]
    for i in range(0,len(segments),batch_size):
        ss=segments[i:i+batch_size];lengths=[len(s['x']) for s in ss];width=ss[0]['x'].shape[1]
        assert width==35 and all(s['x'].shape[1]==width for s in ss)
        x=np.zeros((len(ss),max(lengths),width),np.float32);y=np.zeros((len(ss),max(lengths),2),np.float32);mask=np.zeros_like(y,dtype=bool)
        for j,s in enumerate(ss):
            n=lengths[j];x[j,:n]=s['x'];y[j,:n]=s['y'];mask[j,:n]=s['mask']
        out.append({'x':torch.tensor(x),'y':torch.tensor(y),'mask':torch.tensor(mask),'lengths':torch.tensor(lengths,dtype=torch.long)})
    return out
def config_hash(config,unit):
    config.pop('config_hash',None)
    config['b88_model']={'input':35,'fused':39,'phi':[13,32,16],'backbone':[160,160],'heads':64,'output':'direct','anchor_embedding':'hasflag' if config['variant']=='A' else 'zero'}
    config['experiment']='B88'
    return hashlib.sha256(json.dumps({'config':config,'unit':unit,'sources':source_inventory(),'frozen':M.assert_frozen()},sort_keys=True).encode()).hexdigest()
def baseline_trainer():
    M.assert_frozen();assert common85.CACHE==M.DATA/'.b85'
    spec=importlib.util.spec_from_file_location('_b88_frozen_dependency_train85',M.HERE.parent/'b85_unified/train85.py')
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    assert t.CACHE==M.DATA/'.b85';return t
def dependency_ready():
    M.assert_frozen();assert common85.CACHE==M.DATA/'.b85'
    p=M.DATA/'.b85/queue_status.json'
    if not p.exists():return False
    state=json.loads(p.read_text(encoding='utf8'))['state']
    if state in ('failed','blocked_dependency'):raise RuntimeError('B85 dependency failed')
    if state!='complete':return False
    t=baseline_trainer()
    for u in M.select_units():
        for seed in M.INITS:
            for var in ('X','H'):
                assert t.verify_complete(t.get_run_dir(u['name'],seed,var)),'B85 clean completion missing'
    return True
def activate():
    global _T
    M.assert_frozen();assert common85.CACHE==M.DATA/'.b85'
    if _T is None:
        # Only this uniquely imported module's local globals are patched.
        spec=importlib.util.spec_from_file_location('_b88_adapted_train85',M.HERE.parent/'b85_unified/train85.py')
        t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
        t.CACHE=M.CACHE;t.build_segments=data88.build_segments;t.create_model=create_model
        t.ENV_PINS={**t.ENV_PINS,'CES_B88_OUTPUT_DIR':str(M.CACHE),'CES_B88_ENCODER':'13,32,16','CES_B88_INPUT_WIDTH':'35'}
        t.make_batches=make_batches;t._config_hash=config_hash
        t.acquire_lock=lambda rd:acquire_file_lock(Path(rd)/'.lock')
        t.release_lock=lambda rd:release_file_lock(Path(rd)/'.lock')
        _T=t
    return _T
def guarded_ready():
    t=activate();input_ready()
    for u in M.select_units():
        for seed in M.INITS:
            for var in M.VARIANTS:assert t.verify_complete(t.get_run_dir(u['name'],seed,var)),'Registered fit missing'
    return True
