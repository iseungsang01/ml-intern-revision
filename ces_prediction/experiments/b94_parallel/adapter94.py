"""Per-process adapters only. No inherited source or CPU fit is ever modified."""
import json,os,time
from pathlib import Path
import common94 as G
import runtime94
_T=None
def activate():
    global _T
    if _T is not None:return _T
    G.guard()
    import common92 as M
    M.CACHE=G.CACHE;M.HERE=G.HERE;M.PAPER=G.CACHE/'reports';M.guard=G.guard;M.sources=G.sources
    # Preserve every inherited numeric helper; redirect only generated legacy
    # report names into the B94 cache, with explicit B94 registration metadata.
    original_save=M.save
    def save(path,obj):
        path=Path(path)
        if path.parent==M.PAPER:path=path.with_name(path.name.replace('b92_','b94_'))
        if isinstance(obj,dict) and obj.get('registration')=='PREREGISTRATION_B92.md':
            obj={**obj,'registration':'PREREGISTRATION_B94.md','inherited_protocol':'B92, fresh B94 CUDA fits; no CPU weights pooled'}
        return original_save(path,obj)
    M.save=save
    class Reports:
        def __getattr__(self,k):return getattr(M.B85.C,k)
        def save_text(self,path,text):
            path=Path(path)
            if path.parent==M.PAPER:path=path.with_name(path.name.replace('b92_','b94_'))
            return M.B85.C.save_text(path,text.replace('B.92','B.94'))
    M.C=Reports()
    import prepare92 as P
    from adapter88 import acquire_file_lock,release_file_lock
    original_prepare=P.prepare_unit
    def prepare(name):
        assert name in {u['name'] for u in M.select_units()}
        lock=G.CACHE/'prepare_locks'/f'{name}.lock'
        while not acquire_file_lock(lock):
            assert G.remaining()>0,'Original deadline reached waiting for prepared inputs';time.sleep(1)
        try:return original_prepare(name)
        finally:release_file_lock(lock)
    P.prepare_unit=prepare
    import runtime92 as R
    R.decision=runtime94.decision
    import train92 as T
    T.decision=runtime94.decision;T.prepare_unit=prepare
    old_config=T.config
    original_env=T.effective_env
    def environment(stage,code,runtime):
        env={k:v for k,v in original_env(stage,code,runtime).items() if not k.startswith('CES_B92_')}
        return {**env,'CES_B94_CODE':code,'CES_B94_OUTPUT_DIR':str(G.CACHE),'CES_B94_QUALIFIED_WORKERS':str(runtime['parallel_workers'])}
    T.effective_env=environment
    def config(stage,unit,seed,code,runtime,prepared):
        c=old_config(stage,unit,seed,code,runtime,prepared);c.pop('config_hash')
        c.update(experiment='B94',registration='PREREGISTRATION_B94.md',training_loop_sha256=G.sha(G.BASE/'train92.py'),worker_slot=int(os.environ.get('B94_WORKER_SLOT','0')),scheduled_concurrency=int(os.environ.get('B94_CONCURRENCY','1')))
        assert c['scheduled_concurrency']<=runtime['parallel_workers']
        c['config_hash']=G.digest(c);return c
    T.config=config
    def verify(rd):
        G.guard();rd=Path(rd)
        if not (rd/'complete.json').exists():return False
        c=json.loads((rd/'config.json').read_text());h=c.pop('config_hash');assert h==G.digest(c)
        assert c['sources']==G.sources() and c['frozen']==G.guard() and c['experiment']=='B94'
        assert 'production_override_sha256' not in c['runtime'] and c['runtime']['device']=='cuda' and not c['runtime']['cudnn']
        assert c['runtime']['b93_decision_sha256']==G.sha(G.DATA/'.b93/runtime_decision93.json')
        assert c['runtime']==runtime94.decision(),'Completion runtime must equal recomputed qualified runtime'
        qualified=json.loads((G.DATA/'.b93/runtime_decision93.json').read_text())
        assert all(c['runtime'][k]==v for k,v in qualified['selected_config'].items())
        assert c['runtime']['parallel_workers']==qualified['parallel_workers'] and c['runtime']['selected_path']==qualified['selected_path']
        assert c['registration']=='PREREGISTRATION_B94.md' and c['env']==environment(c['stage'],c['code'],c['runtime'])
        assert 0<=c['worker_slot']<c['scheduled_concurrency']<=c['runtime']['parallel_workers']
        assert c['training_loop_sha256']==G.sha(G.BASE/'train92.py')
        assert G.sha(G.CACHE/'prepared'/f"{c['unit']}__{c['prepared_key']}.pt")==c['prepared_artifact_sha256']
        done=json.loads((rd/'complete.json').read_text());exit=json.loads((rd/'process_exit.json').read_text());metrics=json.loads((rd/'metrics.json').read_text())
        assert done['config_hash']==exit['config_hash']==h and exit['returncode']==0 and not exit['timeout']
        assert metrics['history'] and done['best_val_loss']==min(e['val_loss'] for e in metrics['history'])
        import math
        assert all(math.isfinite(e['val_loss']) for e in metrics['history'])
        assert G.digest(json.loads((rd/'manifest.json').read_text()))==c['manifest_hash'] and G.digest(json.loads((rd/'stats.json').read_text()))==c['stats_hash']
        import torch
        best=torch.load(rd/'best.pt',map_location='cpu',weights_only=True);latest=torch.load(rd/'latest.pt',map_location='cpu',weights_only=False)
        assert best and latest['validated'] and latest['config_hash']==h and latest['epoch']==metrics['history'][-1]['epoch']
        assert latest['updates']==metrics['history'][-1]['updates']
        assert all(torch.isfinite(v).all() for v in best.values())
        assert G.sha(rd/'best.pt')==done['best_sha256'] and G.sha(rd/'latest.pt')==done['latest_sha256']
        return True
    T.verify_complete=verify
    _T=T;return T
