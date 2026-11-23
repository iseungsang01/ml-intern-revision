"""Repair marginal-metric namespace only; cached paired evidence is immutable."""
import argparse,copy,json,os,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CACHE=ROOT/'data/.b101'
sys.path.insert(0,str(HERE.parent/'b96_four'))
import common96 as C
G=C.G
sys.path.insert(0,str(G.HERE))

def sources():
    paths=[HERE/'repair101.py',HERE.parent/'PREREGISTRATION_B101.md',HERE.parent/'PREREGISTRATION_B94_ADDENDUM6.md']
    for p in paths:G.committed(p)
    return {p.relative_to(ROOT).as_posix():G.sha(p) for p in paths}

def prepared():
    z=json.loads((CACHE/'prepared.json').read_text());assert z['sources']==sources()
    assert z['frozen']==G.guard() and time.time()<z['deadline']
    return z

def prepare():
    CACHE.mkdir(exist_ok=True);assert not (CACHE/'prepared.json').exists()
    from status93 import snapshot,process_identity
    status=snapshot(G.CACHE)
    assert status['effective_state']=='failed' and not status['supervisor_alive'] and not status['queue_process_alive']
    for name in ('queue.lock','durable.lock'):
        p=G.CACHE/name
        if p.exists():assert process_identity(int(p.read_text().strip()))['alive'] is False
    paths=[G.CACHE/'readout.json',G.CACHE/'reports/b94_methods.json',G.CACHE/'reports/b94_methods.md',
           G.CACHE/'audit_post.log',G.CACHE/'queue_status.json',G.CACHE/'supervisor_status.json',G.CACHE/'durable_registration.json']
    assert all(p.is_file() for p in paths)
    old=json.loads(paths[0].read_text());assert old==json.loads(paths[1].read_text())
    collision=sum('TX' in c['supplementary_tree_pairs'] for s in old['scopes'].values() for d in s['designs'].values() for c in d['cells'].values())
    assert collision==128
    preds={p.name:G.sha(p) for p in (G.CACHE/'predictions').glob('*.npz')};assert len(preds)==368
    archive=CACHE/'prior';assert not archive.exists()
    journal={str(p.relative_to(G.CACHE)):G.sha(p) for p in paths}
    z={'sources':sources(),'frozen':G.guard(),'deadline':G.clock()['deadline'],'predictions':preds,'prior_files':journal,'collision_cells':collision,'time':time.time()}
    assert time.time()<z['deadline'];G.save(CACHE/'archive_intent.json',z)
    archive.mkdir()
    for p in paths:
        dest=archive/p.relative_to(G.CACHE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name in ('queue.lock','durable.lock'):
        p=G.CACHE/name
        if p.exists():shutil.copy2(p,archive/name);p.unlink()
    G.save(CACHE/'prepared.json',z);print(json.dumps({'preserved_predictions':len(preds),'repair_cells':collision}),flush=True)

def readout():
    z=prepared();prior=CACHE/'prior/readout.json';assert G.sha(prior)==z['prior_files']['readout.json']
    old=json.loads(prior.read_text());expected=copy.deepcopy(old)
    from adapter94 import activate
    activate();import readout92 as R
    import numpy as np
    seen=set();repaired=[]
    def cell(name,rows,pred,support,mask,head,codes,baselines,notes):
        scope,design,key=name.split('/',2);identity=(scope,design,key);assert identity not in seen;seen.add(identity)
        original=old['scopes'][scope]['designs'][design]['cells'][key];out=copy.deepcopy(original)
        truth=rows['tg_'+('F' if head==0 else 'L5')];idx=np.flatnonzero(support&mask&np.isfinite(truth))
        assert len(idx)==out['rows'] and codes==('TX','TH')
        assert notes==original['supplementary_tree_notes']
        # Determine collision from an actually emitted supplementary comparison.
        for learner in baselines:
            if learner in original['supplementary_tree_pairs']:
                assert 'TREE_'+learner not in out['metrics']
                out['metrics']['TREE_'+learner]=out['metrics'].pop(learner)
        out['supplementary_tree_pairs']={'TREE_'+k:v for k,v in original['supplementary_tree_pairs'].items()}
        out['supplementary_tree_notes']={'TREE_'+k:v for k,v in notes.items()}
        expected_cell=copy.deepcopy(out)
        if 'TX' in original['supplementary_tree_pairs']:
            labels={'cluster':rows['clus'][idx],'discharge':rows['fidx'][idx]}
            out['metrics']['TX']=R.B.metrics(name+'/TX',pred['TX'][:,idx,head],truth[idx],labels)
            # Reconstructed metrics are checked below and by the original post audit.
            expected_cell['metrics']['TX']=copy.deepcopy(out['metrics']['TX']);repaired.append(name)
        expected['scopes'][scope]['designs'][design]['cells'][key]=expected_cell
        print('Repaired '+name,flush=True)
        return out
    R.cell=cell;R.latency=lambda unit,code,runtime:copy.deepcopy(old['latency'][code])
    R.main();actual=json.loads((G.CACHE/'readout.json').read_text())
    assert len(seen)==136 and len(repaired)==z['collision_cells']
    assert actual==expected,'Unexpected change outside repaired metrics/tree namespace'
    assert all(G.sha(G.CACHE/'predictions'/n)==h for n,h in z['predictions'].items())
    G.save(CACHE/'repair_checks.json',{'all_pass':True,'repaired':repaired,'predictions_unchanged':True,'unaffected_content_unchanged':True,'sources':sources(),'readout_sha256':G.sha(G.CACHE/'readout.json')})

def queue():
    z=prepared();import scheduler94 as S
    S.subprocess_phase('readout',[sys.executable,'-u',str(HERE/'repair101.py'),'readout'])
    S.audit('post');assert sources()==z['sources']
    for p in (G.CACHE/'reports').glob('b94_*'):
        assert p.suffix in ('.json','.md');shutil.copyfile(p,ROOT/'docs/paper'/p.name)
    S.status('complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','readout','queue'));a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='readout':readout()
    else:
        try:queue()
        except BaseException as e:
            G.save(G.CACHE/'queue_status.json',{'pid':os.getpid(),'state':'failed','updated':time.time(),'error':repr(e)})
            raise
