"""Publish two audited files; ignore unrelated auxiliary readiness documentation."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CACHE=ROOT/'data/.b103'
sys.path.insert(0,str(HERE.parent/'b102_postcheck'))
import finish102 as F
G=F.G

def sources():
    paths=[*HERE.glob('*.py'),HERE.parent/'PREREGISTRATION_B103.md',HERE.parent/'PREREGISTRATION_B94_ADDENDUM8.md']
    for p in paths:G.committed(p)
    return {p.relative_to(ROOT).as_posix():G.sha(p) for p in paths}

def prepared():
    z=json.loads((CACHE/'prepared.json').read_text());assert z['sources']==sources()
    assert z['frozen']==G.guard() and time.time()<z['deadline'];return z

def prepare():
    from status93 import snapshot,process_identity
    s=snapshot(G.CACHE);assert not s['supervisor_alive'] and not s['queue_process_alive']
    CACHE.mkdir(exist_ok=True);assert not (CACHE/'prepared.json').exists()
    prior=CACHE/'prior';prior.mkdir(exist_ok=False)
    for name in ('queue_status.json','supervisor_status.json','durable_registration.json','durable.stderr.log'):
        shutil.copy2(G.CACHE/name,prior/name)
    for name in ('durable.lock','queue.lock'):
        p=G.CACHE/name
        if p.exists():
            assert process_identity(int(p.read_text().strip()))['alive'] is False
            shutil.copy2(p,prior/name);p.unlink()
    F.validate()
    post=G.CACHE/'lead_checks/post.json'
    files={name:G.sha(G.CACHE/'reports'/name) for name in ('b94_methods.json','b94_methods.md')}
    G.save(CACHE/'prepared.json',{'sources':sources(),'frozen':G.guard(),'deadline':G.clock()['deadline'],'launch_time':time.time(),'post_sha256':G.sha(post),'files':files})
    print(json.dumps({'artifact_checks_passed':True,'audit_passed':json.loads(post.read_text())['passed']}),flush=True)

def queue():
    z=prepared();F.validate()
    p=G.CACHE/'lead_checks/post.json';assert G.sha(p)==z['post_sha256'];post=json.loads(p.read_text())
    assert post['passed'] and post['phase']=='post' and post['confirmation_fits']==46 and post['cells_independently_recomputed']==136
    assert post['readout_sha256']==G.sha(G.CACHE/'readout.json') and post['selection_sha256']==G.sha(G.CACHE/'selection.json')
    sys.path.insert(0,str(G.HERE/'lead_checks'));import audit94 as A
    assert post['audit_source_hashes']==A.audit_sources() and post['sources']==G.sources()
    assert post['prediction_hashes']=={p.name:G.sha(p) for p in (G.CACHE/'predictions').glob('*.npz')}
    report=json.loads((G.CACHE/'readout.json').read_text())
    assert report==json.loads((G.CACHE/'reports/b94_methods.json').read_text())
    assert prepared()==z
    for name,h in z['files'].items():assert G.sha(G.CACHE/'reports'/name)==h
    for name,h in z['files'].items():
        dest=ROOT/'docs/paper'/name;shutil.copyfile(G.CACHE/'reports'/name,dest);assert G.sha(dest)==h
    G.save(CACHE/'published.json',{'all_pass':True,'post_sha256':z['post_sha256'],'readout_sha256':post['readout_sha256'],'files':z['files'],'time':time.time()})
    import scheduler94 as S
    S.status('complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','queue'));a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        try:queue()
        except BaseException as e:
            G.save(G.CACHE/'queue_status.json',{'pid':os.getpid(),'state':'failed','updated':time.time(),'error':repr(e)});raise
