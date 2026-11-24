"""Explicit author stop: preserve validated checkpoints, never stop foreign jobs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import ctypes,hashlib,io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b107_patience'))
import common107 as C
from pipeline107 import Observed
from status93 import process_identity
import durable93 as D
CODE=0xE0000107
def sha(b):return hashlib.sha256(b).hexdigest()
def stop(o):
    if o.poll() is None:
        o.k.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        o.k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        assert o.k.TerminateProcess(o.handle,CODE)
        assert o.k.WaitForSingleObject(o.handle,15000)==0
    return o.poll()
def main():
    q=C.load(C.CACHE/'queue_status.json');sup=C.load(C.CACHE/'supervisor_status.json');assert q['state']=='training'
    out=C.CACHE/'author_stop_20261124_1244';out.mkdir(exist_ok=False)
    C.save(out/'queue_before.json',q);C.save(out/'supervisor_before.json',sup)
    owner=Observed(q['pid'],q['birth'],0x100000|0x1000|0x800|1)
    nt=ctypes.WinDLL('ntdll');nt.NtSuspendProcess.argtypes=[ctypes.c_void_p];nt.NtSuspendProcess.restype=ctypes.c_long
    assert nt.NtSuspendProcess(owner.handle)==0
    C.save(out/'stop_intent.json',{'reason':'Author explicitly requested stopping local experiments','time':time.time(),'queue_suspended':True})
    D.powershell("Stop-ScheduledTask -TaskName 'B107_patience_watch' -ErrorAction SilentlyContinue")
    import torch
    assert not torch.cuda.is_available();records=[]
    for e in q['active']:
        assert e['origin']=='B107','Unexpected adopted original child; inspect before stopping'
        p=C.new_rd(e['case']);cfg=C.load(p/'config.json');h=cfg.pop('config_hash');assert C.digest(cfg)==h
        o=Observed(e['pid'],e['birth'],0x100000|0x1000|0x800|1)
        saved=None
        for _ in range(40):
            raw=(p/'latest.pt').read_bytes();w=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
            lines=[json.loads(t) for t in (p/'process.log').read_text(encoding='utf8',errors='replace').splitlines() if t.startswith('{')]
            if lines and lines[-1]['epoch']==w['epoch']:saved=(raw,w,lines);break
            time.sleep(.05)
        assert saved is not None,'No matching fully validated checkpoint/log boundary'
        raw,w,lines=saved;assert w['validated'] and w['config_hash']==h
        assert all(k in w for k in ('model','optimizer','scheduler','rng_cpu','rng_cuda','rng_python','rng_numpy'))
        assert all(torch.isfinite(v).all() for v in w['model'].values())
        if o.poll() is None:assert nt.NtSuspendProcess(o.handle)==0
        dest=out/e['case'];dest.mkdir();(dest/'latest_preserved.pt').write_bytes(raw)
        for name in ('config.json','manifest.json','stats.json','best.pt','process.log','resume_checks.json','metrics.json','complete.json'):
            if (p/name).exists():(dest/name).write_bytes((p/name).read_bytes())
        history=[]
        if cfg['parent']:history=C.load(ROOT/cfg['parent']['path']/'metrics.json')['history']
        history+=lines;assert len(history)==w['epoch']+1 and history[-1]['updates']==w['updates']
        C.save(dest/'validated_history.json',history)
        r={'case':e['case'],'pid':e['pid'],'birth':e['birth'],'validated_epoch':w['epoch']+1,'validated_updates':w['updates'],
           'checkpoint_sha256':sha(raw),'config_hash':h,'preserved':str(dest.relative_to(ROOT)),'reason':'Author requested stop; incomplete epoch updates discarded'}
        C.save(dest/'before_stop.json',r)
        rc=stop(o);o.close();r['returncode']=rc;r['interrupted']=rc!=0;records.append(r)
        if not (p/'process_exit.json').exists():C.save(p/'process_exit.json',{**r,'timeout':False,'author_requested_stop':True})
        print(e['case']+' stopped; validated epoch '+str(r['validated_epoch']),flush=True)
    queue_rc=stop(owner);owner.close()
    identity=process_identity(sup['pid'])
    if identity['alive'] is True and identity['birth']==sup['birth']:
        observer=Observed(sup['pid'],sup['birth'],0x100000|0x1000|1)
        for _ in range(70):
            if observer.poll() is not None:break
            time.sleep(.1)
        if observer.poll() is None:stop(observer)
        observer.close()
    previous=C.load(C.CACHE/'supervisor_status.json')
    result={'state':'interrupted','reason':'author_requested_pause','time':time.time(),'completed_patience_paths':q['completed'],'pending_not_dispatched':q['pending'],
            'stopped_children':records,'queue_exit':queue_rc,'automatic_resume':False,'Colab_started':False}
    C.save(out/'author_stop.json',result);C.save(C.CACHE/'author_stop.json',result)
    C.save(C.CACHE/'queue_status.json',{**q,'state':'interrupted','reason':'author_requested_pause','updated':time.time(),'active':[],'stopped_children':records})
    C.save(C.CACHE/'supervisor_status.json',{**previous,'state':'interrupted','phase':'paused_by_author','updated':time.time(),'reason':'author_requested_pause'})
    print('Local experiments stopped; checkpoints and RNG preserved; no automatic resume.',flush=True)
if __name__=='__main__':main()
