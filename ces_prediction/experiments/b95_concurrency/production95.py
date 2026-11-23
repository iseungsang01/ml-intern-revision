"""Reuse the audited scheduler, expanding independent fixed groups only."""
import json,sys
from pathlib import Path
import common95 as C

def main():
    sources,decision=C.policy();n=decision['selected_workers'];G=C.G
    from adapter94 import activate
    T=activate()
    import scheduler94 as S
    original_env=T.effective_env
    # Each frozen component retains its original two-slot qualification.
    T.effective_env=lambda stage,code,runtime:original_env(stage,code,{**runtime,'parallel_workers':2})
    original_verify=T.verify_complete
    def verify(rd):
        ok=original_verify(rd)
        if ok:C.dispatch_config(Path(rd)/'config.json',sources,decision)
        return ok
    T.verify_complete=verify
    original_save=G.save
    def save(path,obj):
        path=Path(path)
        if path.name=='process_exit.json' and path.parent.joinpath('config.json').exists():
            cfg=json.loads((path.parent/'config.json').read_text())
            if 'dispatch_sources95' in cfg:
                obj={**obj,'global_worker_slot':obj['worker_slot'],'worker_slot':cfg['worker_slot'],'global_gpu_worker_limit':cfg['global_gpu_worker_limit']}
        return original_save(path,obj)
    G.save=save
    S.command=lambda action,*args:[sys.executable,'-X','faulthandler','-u',str(C.HERE/'bootstrap95.py'),action,*map(str,args)]
    original_dispatch=S.dispatch
    S.dispatch=lambda stage,runtime,trainer:original_dispatch(stage,{**runtime,'parallel_workers':n},trainer)
    original_phase=S.subprocess_phase
    def phase(name,cmd):
        if name=='prediction':
            assert C.sources()==sources
            for stage in ('pilot','confirm'):
                for p in (G.CACHE/stage).glob('*/complete.json'):assert verify(p.parent)
        return original_phase(name,cmd)
    S.subprocess_phase=phase
    record={'registration':'PREREGISTRATION_B94_ADDENDUM3.md','global_worker_limit':n,'original_group_limit':2,'dispatch_sources':sources,'qualification':decision,'original_deadline':G.clock()['deadline'],'cpu_queue_resumed':False}
    G.save(G.CACHE/'dispatch_policy95.json',record)
    print(json.dumps({'global_gpu_workers':n,'reusing_completed_fits':True}),flush=True)
    S.main()
    G.save(G.ROOT/'docs/paper/b94_dispatch_policy95.json',record)

if __name__=='__main__':main()
