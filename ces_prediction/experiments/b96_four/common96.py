"""Author-selected four-worker policy; historical dispatch remains immutable."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CACHE=ROOT/'data/.b96'
sys.path.insert(0,str(HERE.parent/'b95_concurrency'))
import common95 as OLD
G=OLD.G

def sources():
    out=OLD.sources()
    for p in [*(HERE/n for n in ('common96.py','bootstrap96.py','production96.py','transition96.py')),HERE.parent/'PREREGISTRATION_B96.md']:
        G.committed(p);out[p.relative_to(ROOT).as_posix()]=G.sha(p)
    return out

def policy(publish=False):
    G.guard();s=sources();q=OLD.qualification();d=json.loads((ROOT/'data/.b95/decision95.json').read_text())
    four=[c for c in d['cohorts'] if c['workers']==4 and c['clean']];assert len(four)==1
    assert len(four[0]['records'])==4
    # OLD.qualification independently verifies every recorded result/exit,
    # source/input/software binding and normal completion, including these four.
    record={'registration':'PREREGISTRATION_B96.md','selected_workers':4,
            'basis':'Author-selected balanced capacity; overrides proof-of-10percent-improvement threshold, not an asserted optimum',
            'base_policy95_sha256':q['decision_sha256'],'feasibility_cohort':four[0],
            'dispatch_sources96':s,'original_deadline':G.clock()['deadline'],'test_accessed':False}
    path=CACHE/'policy96.json'
    if publish and not path.exists():G.save(path,record)
    assert json.loads(path.read_text())==record
    return s,{**record,'decision_sha256':G.sha(path)}

def dispatch_config(path,s,decision):
    c=json.loads(Path(path).read_text())
    if 'dispatch_sources96' not in c:
        OLD.dispatch_config(Path(path),OLD.sources(),OLD.qualification());return
    assert c['dispatch_sources96']==s and c['dispatch_policy96_sha256']==decision['decision_sha256']
    assert c['dispatch_registration']=='PREREGISTRATION_B96.md'
    n=c['global_gpu_worker_limit'];i=c['global_worker_slot'];g=i//2
    assert n in (1,4) and 0<=i<n and c['local_worker_group']==g
    assert c['scheduled_concurrency']==min(2,n-2*g) and c['worker_slot']==i%2
    assert c['scheduled_concurrency_scope']=='local group; global capacity recorded separately'
    p=Path(path).parent/'process_exit.json'
    if p.exists():
        e=json.loads(p.read_text());assert e['global_worker_slot']==i and e['worker_slot']==i%2 and e['global_gpu_worker_limit']==n

if __name__=='__main__':
    _,q=policy(publish=True);print(json.dumps({'selected_workers':4,'policy_sha256':q['decision_sha256']}))
