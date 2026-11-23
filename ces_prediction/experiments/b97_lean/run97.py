"""Registered actual TRAIN/VAL trace bridge; no fit, TEST scoring or GPU use."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
CACHE=ROOT/'data/.b97'
PROJECT=ROOT/'formal/visibility'
TOOLS=ROOT/'data/.lean_toolchain/lean-4.36.0-windows/bin'
RULES=['D10','D50','D100','D300','D650','D1000','S','K10']
sys.path.insert(0,str(HERE.parent/'b85_unified'))
import data85 as D

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def inputs():
    config=ROOT/'data/.b94/confirm/V1_r2_k0__1701__TX/config.json'
    c=json.loads(config.read_text())
    prep=ROOT/'data/.b94/prepared'/('V1_r2_k0__'+c['prepared_key']+'.pt')
    assert sha(prep)==c['prepared_artifact_sha256']
    z=torch.load(prep,map_location='cpu',weights_only=False)
    manifest=z['identity']['manifest'];del z
    assert not set(manifest['train'])&set(manifest['val'])
    assert not (set(manifest['train'])|set(manifest['val']))&set(manifest['test'])
    selected={part:min(manifest[part]) for part in ('train','val')}
    raw=ROOT/'data/.b73/grid.npz';frames=ROOT/'data/.b78/frames.npz'
    with np.load(raw,allow_pickle=True) as g:
        keep=np.flatnonzero(np.isin(g['fidx'],list(selected.values())))
        G={k:g[k][keep] for k in ('time','seg','ti','fidx')}
    with np.load(frames,allow_pickle=True) as f:fr=f['fr'][keep]
    assert len(keep)>0 and np.isfinite(G['time']).all()
    assert not set(G['fidx'])&set(manifest['test'])
    return G,fr,selected,{str(p.relative_to(ROOT)):sha(p) for p in (config,prep,raw,frames)},keep

def run():
    reg=HERE.parent/'PREREGISTRATION_B97.md'
    bound_sources=[reg,*PROJECT.glob('*.lean'),*PROJECT.glob('*.toml'),PROJECT/'lean-toolchain',*HERE.glob('*.py'),HERE/'lead_checks/verify97.py',HERE.parent/'b85_unified/data85.py']
    for p in bound_sources:
        rel=p.relative_to(ROOT).as_posix()
        subprocess.run(['git','cat-file','-e','HEAD:'+rel],cwd=ROOT,check=True)
        subprocess.run(['git','diff','--exit-code','HEAD','--',rel],cwd=ROOT,check=True)
    assert not (CACHE/'result.json').exists(),'No silent rerun of completed assurance batch'
    CACHE.mkdir(parents=True,exist_ok=True)
    for p in PROJECT.glob('*.lean'):
        assert not re.search(r'\b(sorry|admit|axiom|native_decide|unsafe|implemented_by|extern)\b',p.read_text(encoding='utf8'))
    env={**os.environ,'PATH':str(TOOLS)+os.pathsep+os.environ['PATH']}
    for name,cmd in [('build',['lake.exe','build']),('axioms',['lake.exe','env','lean','Visibility.lean'])]:
        with (CACHE/(name+'.log')).open('w',encoding='utf8') as log:
            subprocess.run([str(TOOLS/cmd[0]),*cmd[1:]],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    audit=(CACHE/'axioms.log').read_text(encoding='utf8')
    assert 'sorryAx' not in audit and 'error:' not in audit
    assert audit.count('depends on axioms:')+audit.count('does not depend on any axioms')==6
    for block in re.findall(r'depends on axioms:\s*\[([^\]]*)\]',audit):
        assert {a.strip() for a in block.split(',') if a.strip()}<={'propext','Quot.sound','Classical.choice'}
    assert D.RULES==RULES,'Registered rules changed; no implicit fallback'
    G,fr,selected,hashes,original=inputs()
    queries=[]
    # Limit the imported original function to selected TRAIN/VAL data. This does
    # not change its source or search algorithm, nor touch its global base cache.
    with patch.object(D,'load_sources',return_value=(G,{})),patch.object(D,'_load_frames',return_value={'fr':fr}):
        for rule in D.RULES:
            D._ANCHOR_CACHE.clear();anchors=D.anchors(rule,G=G)
            for part,fid in selected.items():
                file_rows=np.flatnonzero(G['fidx']==fid)
                segments=[]
                for seg in dict.fromkeys(G['seg'][file_rows].tolist()):
                    rows=file_rows[G['seg'][file_rows]==seg]
                    if len(rows)>=2:segments.append(rows)
                assert segments,'Missing actual segments'
                for rows in segments[:2]:
                    assert np.all(np.diff(G['time'][rows])>=0)
                    targets=sorted({int(rows[k]) for k in (0,len(rows)//4,len(rows)//2,3*len(rows)//4,len(rows)-1)})
                    for t in targets:
                        prefix=file_rows[file_rows<t]
                        # Encoding covers every earlier selected-grid index,
                        # including rows of the other file (which stay false).
                        allowed=np.zeros(t,dtype=bool)
                        same=prefix[G['seg'][prefix]==G['seg'][t]]
                        threshold=None
                        if rule.startswith('D'):
                            threshold=float(G['time'][t]-int(rule[1:])/1000.+.003)
                            candidates=same[G['time'][same]<=threshold]
                        elif rule=='K10':
                            candidates=same[(fr[same]>=0)&(fr[same]%10==0)]
                        elif rule=='S':
                            current=rows[rows<=t]
                            dt=np.r_[0.,np.diff(G['time'][current])]
                            gaps=current[(dt>.015)&(dt<.045)]
                            cutoff=int(gaps[-1]-1) if len(gaps) else -1
                            threshold=cutoff
                            candidates=same[same<=cutoff]
                        else:raise ValueError(rule)
                        candidates=candidates[np.isfinite(G['ti'][candidates])]
                        allowed[candidates]=True
                        expected=int(candidates[-1]) if len(candidates) else -1
                        assert int(anchors[t])==expected
                        if expected>=0:
                            assert expected<t and G['seg'][expected]==G['seg'][t] and G['fidx'][expected]==G['fidx'][t]
                        queries.append({'id':len(queries),'partition':part,'rule':rule,'target':t,
                                        'expected':expected,'threshold':threshold,
                                        'bits':''.join('1' if b else '0' for b in allowed)})
    text=''.join(f"{q['id']} {q['bits']} {q['expected'] if q['expected']>=0 else 'none'}\n" for q in queries)
    (CACHE/'queries.json').write_text(json.dumps(queries),encoding='utf8')
    exe=PROJECT/'.lake/build/bin/visibilityBridge.exe'
    completed=subprocess.run([str(exe)],input=text,text=True,capture_output=True,check=True,timeout=120)
    (CACHE/'bridge.stdout').write_text(completed.stdout,encoding='utf8')
    assert len(completed.stdout.splitlines())==len(queries)
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [*PROJECT.glob('*.lean'),*PROJECT.glob('*.toml'),PROJECT/'lean-toolchain',Path(__file__),HERE/'toolchain97.py',HERE/'lead_checks/verify97.py',reg,HERE.parent/'b85_unified/data85.py']}
    result={'experiment':'B97','status':'checks_passed','queries':len(queries),'rules':list(D.RULES),
            'partitions':['train','val'],'source_hashes':sources,'input_hashes':hashes,
            'query_sha256':sha(CACHE/'queries.json'),'bridge_output_sha256':sha(CACHE/'bridge.stdout'),
            'axioms_log':audit,'lean_version':subprocess.check_output([str(TOOLS/'lean.exe'),'--version'],text=True).strip(),
            'compiler_sha256':sha(TOOLS/'lean.exe'),'executable_sha256':sha(exe),
            'toolchain_archive_sha256':sha(ROOT/'data/.lean_toolchain/lean-4.36.0-windows.zip'),
            'new_training':False,'test_segments_constructed':False,'full_pipeline_verified':False}
    (CACHE/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ('status','queries','lean_version')}))

if __name__=='__main__':run()
