"""Four global slots; original job order, training, evidence and gates."""
import json,sys
from pathlib import Path
import common96 as C

def main():
    sources,decision=C.policy();G=C.G
    from adapter94 import activate
    T=activate()
    import scheduler94 as S
    original_env=T.effective_env
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
        if path.name=='process_exit.json' and (path.parent/'config.json').exists():
            c=json.loads((path.parent/'config.json').read_text())
            if 'dispatch_sources96' in c:obj={**obj,'global_worker_slot':obj['worker_slot'],'worker_slot':c['worker_slot'],'global_gpu_worker_limit':c['global_gpu_worker_limit']}
        return original_save(path,obj)
    G.save=save
    S.command=lambda action,*args:[sys.executable,'-X','faulthandler','-u',str(C.HERE/'bootstrap96.py'),action,*map(str,args)]
    original_dispatch=S.dispatch
    S.dispatch=lambda stage,runtime,trainer:original_dispatch(stage,{**runtime,'parallel_workers':4},trainer)
    original_phase=S.subprocess_phase
    def phase(name,cmd):
        if name=='prediction':
            assert C.sources()==sources
            for stage in ('pilot','confirm'):
                for p in (G.CACHE/stage).glob('*/complete.json'):assert verify(p.parent)
        return original_phase(name,cmd)
    S.subprocess_phase=phase
    record={'registration':'PREREGISTRATION_B96.md','global_worker_limit':4,'original_group_limit':2,
            'dispatch_sources':sources,'qualification':decision,'original_deadline':G.clock()['deadline'],'cpu_queue_resumed':False}
    G.save(G.CACHE/'dispatch_policy96.json',record)
    print(json.dumps({'global_gpu_workers':4,'reusing_completed_fits':True}),flush=True)
    S.main();G.save(G.ROOT/'docs/paper/b94_dispatch_policy96.json',record)

if __name__=='__main__':main()
