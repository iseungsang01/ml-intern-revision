"""Resume artifact checks/audit only; no statistical regeneration."""
import argparse,copy,json,os,shutil,sys,time
from pathlib import Path,PureWindowsPath
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CACHE=ROOT/'data/.b102'
sys.path.insert(0,str(HERE.parent/'b101_readout_repair'))
import repair101 as OLD
G=OLD.G
sys.path.insert(0,str(G.HERE))

def sources():
    paths=[*HERE.glob('*.py'),HERE.parent/'PREREGISTRATION_B102.md',HERE.parent/'PREREGISTRATION_B102_ADDENDUM1.md',HERE.parent/'PREREGISTRATION_B94_ADDENDUM7.md']
    for p in paths:G.committed(p)
    return {p.relative_to(ROOT).as_posix():G.sha(p) for p in paths}

def prepared():
    z=json.loads((CACHE/'prepared.json').read_text());assert z['sources']==sources()
    assert z['frozen']==G.guard() and time.time()<z['deadline'];return z

def prepare():
    CACHE.mkdir(exist_ok=True);assert not (CACHE/'prepared.json').exists()
    from status93 import snapshot,process_identity
    s=snapshot(G.CACHE);assert not s['supervisor_alive'] and not s['queue_process_alive']
    archive=CACHE/'prior';archive.mkdir(exist_ok=False)
    files={}
    for name in ('readout.json','reports/b94_methods.json','reports/b94_methods.md','queue_status.json','supervisor_status.json','durable_registration.json','readout.log','durable.stderr.log'):
        p=G.CACHE/name;dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True)
        files[name]=G.sha(p);shutil.copy2(p,dest)
    for name in ('durable.lock','queue.lock'):
        p=G.CACHE/name
        if p.exists():
            assert process_identity(int(p.read_text().strip()))['alive'] is False
            shutil.copy2(p,archive/name);p.unlink()
    z={'sources':sources(),'frozen':G.guard(),'deadline':G.clock()['deadline'],'files':files,'launch_time':time.time(),'b101_manifest_sha256':G.sha(OLD.CACHE/'prepared.json')}
    G.save(CACHE/'prepared.json',z);print(json.dumps({'artifacts_preserved':True}),flush=True)

def validate():
    z=prepared();assert G.sha(OLD.CACHE/'prepared.json')==z['b101_manifest_sha256']
    prior=OLD.prepared();keys={PureWindowsPath(k).as_posix():v for k,v in prior['prior_files'].items()}
    assert len(keys)==len(prior['prior_files'])
    for k,h in keys.items():assert G.sha(OLD.CACHE/'prior'/k)==h
    for k,h in z['files'].items():
        assert G.sha(CACHE/'prior'/k)==h
        if k.startswith(('readout.json','reports/')):assert G.sha(G.CACHE/k)==h
    old=json.loads((OLD.CACHE/'prior/readout.json').read_text());actual=json.loads((G.CACHE/'readout.json').read_text())
    expected=copy.deepcopy(old);count=0
    for scope,s in old['scopes'].items():
        for design,d in s['designs'].items():
            for name,c in d['cells'].items():
                out=copy.deepcopy(c)
                for learner in c['supplementary_tree_pairs']:out['metrics']['TREE_'+learner]=out['metrics'].pop(learner)
                out['supplementary_tree_pairs']={'TREE_'+k:v for k,v in c['supplementary_tree_pairs'].items()}
                out['supplementary_tree_notes']={'TREE_'+k:v for k,v in c['supplementary_tree_notes'].items()}
                if 'TX' in c['supplementary_tree_pairs']:
                    out['metrics']['TX']=actual['scopes'][scope]['designs'][design]['cells'][name]['metrics']['TX'];count+=1
                expected['scopes'][scope]['designs'][design]['cells'][name]=out
    assert count==128 and actual==expected
    assert actual==json.loads((G.CACHE/'reports/b94_methods.json').read_text())
    assert {p.name:G.sha(p) for p in (G.CACHE/'predictions').glob('*.npz')}==prior['predictions']
    assert OLD.baselines(old)==prior['baselines']
    assert all(G.sha(ROOT/n)==h for n,h in prior['fit_files'].items())
    assert G.sha(G.CACHE/'selection.json')==prior['selection_sha256']
    md=G.CACHE/'reports/b94_methods.md';assert G.sha(md)!=keys['reports/b94_methods.md']
    names={'TX','TH','T0','TXfresh','TREE_TX','TREE_T0','TREE_TXfresh'}
    def nonmetrics(text):return [v for v in text.splitlines() if not(v.startswith('| ') and v.split('|')[1].strip() in names)]
    assert nonmetrics(md.read_text(encoding='utf8'))==nonmetrics((OLD.CACHE/'prior/reports/b94_methods.md').read_text(encoding='utf8'))
    seen=set();scope=design=key=None
    def fmt(v):return f"{v['point']:.5f} [{v['cluster'][0]:.5f}, {v['cluster'][1]:.5f}]"
    for line in md.read_text(encoding='utf8').splitlines():
        if line.startswith('## '):scope,design=line[3:].split(' / ')
        elif line.startswith('### '):key=line[4:].split(': ',1)[0]
        elif line.startswith('| ') and line.split('|')[1].strip() in names:
            learner=line.split('|')[1].strip();identity=(scope,design,key,learner);assert identity not in seen;seen.add(identity)
            values=actual['scopes'][scope]['designs'][design]['cells'][key]['metrics'][learner]
            assert line=='| '+learner+' | '+' | '.join(fmt(values[k]) for k in ('RMSE','MAE','medianAE'))+' |'
    assert seen=={(s,d,k,c) for s,ss in actual['scopes'].items() for d,dd in ss['designs'].items() for k,cc in dd['cells'].items() for c in cc['metrics']}
    lines=(CACHE/'prior/readout.log').read_text(encoding='utf8').splitlines()
    assert len([v for v in lines if v.startswith(('Repaired ','Carried '))])==136
    assert "KeyError: 'reports/b94_methods.md'" in '\n'.join(lines)
    G.save(CACHE/'validation.json',{'all_pass':True,'unaffected_content_preserved':True,'predictions_unchanged':True,'manifest_normalization_only':True,'markdown_metric_rows_checked':len(seen),'original_post_audit_required':True,'sources':sources(),'readout_sha256':G.sha(G.CACHE/'readout.json')})
    print(json.dumps({'validation_passed':True,'no_recomputation':True}),flush=True)

def queue():
    z=prepared();import scheduler94 as S
    S.subprocess_phase('artifact_validation',[sys.executable,'-u',str(HERE/'finish102.py'),'validate'])
    S.audit('post');assert prepared()==z
    v=json.loads((CACHE/'validation.json').read_text());assert v['all_pass'] and v['sources']==sources()
    assert v['readout_sha256']==G.sha(G.CACHE/'readout.json')==z['files']['readout.json']
    outputs=list((G.CACHE/'reports').glob('b94_*'));assert {p.name for p in outputs}=={'b94_methods.json','b94_methods.md'}
    for p in outputs:assert p.suffix in ('.json','.md') and G.sha(p)==z['files']['reports/'+p.name]
    for p in outputs:
        dest=ROOT/'docs/paper'/p.name;shutil.copyfile(p,dest);assert G.sha(dest)==G.sha(p)
    S.status('complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','validate','queue'));a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='validate':validate()
    else:
        try:queue()
        except BaseException as e:
            G.save(G.CACHE/'queue_status.json',{'pid':os.getpid(),'state':'failed','updated':time.time(),'error':repr(e)});raise
