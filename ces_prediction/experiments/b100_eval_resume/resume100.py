"""Evaluation-only reboot recovery; original arithmetic and final audit."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CACHE=ROOT/'data/.b100'
sys.path.insert(0,str(HERE.parent/'b96_four'))
import common96 as C
G=C.G
sys.path.insert(0,str(G.HERE))  # Explicit in every fresh inference/queue process.

def sources():
    paths=[HERE/'resume100.py',HERE.parent/'PREREGISTRATION_B100.md',HERE.parent/'PREREGISTRATION_B94_ADDENDUM5.md']
    for p in paths:G.committed(p)
    return {p.relative_to(ROOT).as_posix():G.sha(p) for p in paths}

def prepared():
    z=json.loads((CACHE/'prepared.json').read_text());assert z['sources']==sources()
    assert G.guard()==z['frozen'] and time.time()<z['deadline']
    return z

def prepare():
    CACHE.mkdir(parents=True,exist_ok=True)
    assert not (CACHE/'prepared.json').exists(),'Explicit one-shot recovery only'
    pins=G.guard();ss=sources();s,d=C.policy()
    sys.path.insert(0,str(HERE.parent/'b93_gpu'))
    from status93 import process_identity
    pause=json.loads((ROOT/'data/.author_pause/pause_20261123_182401.json').read_text())
    assert all(process_identity(r['pid'])['alive'] is False for r in pause['suspended'])
    for name in ('queue.lock','durable.lock'):
        lock=G.CACHE/name
        if lock.exists():assert process_identity(int(lock.read_text().strip()))['alive'] is False
    assert time.time()<G.clock()['deadline']
    sys.path.insert(0,str(HERE.parent/'b94_parallel/lead_checks'))
    import audit94 as A
    p=A.prior();pilot=A.pilot(allow_test_artifacts=True)
    from adapter94 import activate
    activate();import predict92 as P
    _,_,jobs=P.guarded_ready();assert len(jobs)==46
    assert len(P.M.RULES)==8
    expected={f'{u}__{seed}__{code}__{rule}.npz' for _,u,seed,code in jobs for rule in P.M.RULES}
    assert all(f.name in expected or f.name.removesuffix('.tmp.npz')+'.npz' in expected for f in (G.CACHE/'predictions').glob('*.npz'))
    fits={}
    for job in jobs:
        stage,u,seed,code=job;rd=G.CACHE/stage/f'{u}__{seed}__{code}'
        fits[rd.name]=A.verify_run(rd,job,p);C.dispatch_config(rd/'config.json',s,d)
    log=(G.CACHE/'prediction.log').read_text(encoding='utf8')
    logged={line.removeprefix('Predicted ').strip() for line in log.splitlines() if line.startswith('Predicted ')}
    reuse={};pending=[];partial=[]
    import numpy as np
    for job in jobs:
        stage,u,seed,code=job;name=f'{u}__{seed}__{code}'
        files=[G.CACHE/'predictions'/f'{name}__{rule}.npz' for rule in P.M.RULES]
        if name in logged and all(f.is_file() for f in files):
            support=None
            for f in files:
                z=P.npz(f);idx=z['idx'];est=z['est']
                assert str(z['config_hash'])==fits[name]['config_hash'] and str(z['sources_hash'])==G.digest(G.sources())
                assert idx.ndim==1 and idx.dtype.kind in 'iu' and len(idx) and np.all(np.diff(idx)>0)
                assert est.shape==(len(idx),2) and np.isfinite(est).all()
                if support is None:support=idx
                else:assert np.array_equal(idx,support)
                reuse[f.name]=G.sha(f)
        else:
            pending.append(list(job));partial.extend(f for f in files if f.exists())
            partial.extend((G.CACHE/'predictions').glob(name+'__*.tmp.npz'))
    # Archive before mutation; hashes retain both interrupted and completed evidence.
    archive=CACHE/'prior';assert not archive.exists();archived={}
    for f in partial:
        target=archive/'predictions'/f.name
        assert f.resolve().is_relative_to(G.CACHE.resolve()) and target.resolve().is_relative_to(archive.resolve())
        archived[str(f.relative_to(ROOT))]=G.sha(f)
    G.save(CACHE/'archive_intent.json',{'sources':ss,'reused':reuse,'pending':pending,'partial_hashes':archived,'automatic_retry':False})
    archive.mkdir(exist_ok=False)
    for f in partial:
        target=archive/'predictions'/f.name;target.parent.mkdir(exist_ok=True);f.rename(target)
    for name in ('queue_status.json','supervisor_status.json','queue.lock','durable.lock','durable_registration.json','prediction.log'):
        f=G.CACHE/name
        if f.exists():
            archived[str(f.relative_to(ROOT))]=G.sha(f);shutil.copy2(f,archive/name)
            if name.endswith('.lock'):f.unlink()
    z={'sources':ss,'frozen':pins,'deadline':G.clock()['deadline'],'fits':fits,'pilot':pilot,
       'reused':reuse,'pending':pending,'archived':archived,'prepared_at':time.time(),
       'training_restarted':False,'prediction_log_sha256':G.sha(archive/'prediction.log')}
    G.save(CACHE/'prepared.json',z)
    print(json.dumps({'confirmation_fits':len(fits),'reused_predictions':len(reuse),'pending_groups':len(pending)}),flush=True)

def predict():
    z=prepared()
    from adapter94 import activate
    activate();import predict92 as P
    original=P.guarded_ready;calls=0
    def ready():
        nonlocal calls
        selection,runtime,jobs=original();calls+=1
        if calls==1:
            expected={tuple(j) for j in jobs};assert all(tuple(j) in expected for j in z['pending'])
            return selection,runtime,[tuple(j) for j in z['pending']]
        assert calls==2
        assert all((G.CACHE/'predictions'/f'{u}__{seed}__{code}__{rule}.npz').is_file() for _,u,seed,code in z['pending'] for rule in P.M.RULES)
        return selection,runtime,jobs
    P.guarded_ready=ready;P.main();assert calls==2
    assert all(G.sha(G.CACHE/'predictions'/n)==h for n,h in z['reused'].items())
    assert len(list((G.CACHE/'predictions').glob('*.npz')))==368
    G.save(CACHE/'prediction_resume.json',{'all_pass':True,'reused_hashes_unchanged':True,'sources':sources()})

def queue():
    z=prepared()
    sys.path.insert(0,str(HERE.parent/'b94_parallel'))
    import scheduler94 as S
    S.subprocess_phase('prediction',[sys.executable,'-u',str(HERE/'resume100.py'),'predict'])
    S.subprocess_phase('readout',[sys.executable,'-u',str(HERE.parent/'b96_four/bootstrap96.py'),'readout'])
    S.audit('post')
    assert sources()==z['sources']
    for path in (G.CACHE/'reports').glob('b94_*'):
        assert path.suffix in ('.json','.md');shutil.copyfile(path,ROOT/'docs/paper'/path.name)
    S.status('complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','predict','queue'));a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='predict':predict()
    else:
        try:queue()
        except BaseException as e:
            G.save(G.CACHE/'queue_status.json',{'pid':os.getpid(),'state':'failed','updated':time.time(),'error':repr(e)})
            raise
