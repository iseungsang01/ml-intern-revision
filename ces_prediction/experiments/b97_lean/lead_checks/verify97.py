"""Independent per-row reconstruction from real inputs, not runner bit masks."""
import hashlib,json,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import run97 as R

def reload_inputs():
    """Separate source/manifest/filter/frame reconstruction, not R.inputs()."""
    config=R.ROOT/'data/.b94/confirm/V1_r2_k0__1701__TX/config.json'
    cfg=json.loads(config.read_text())
    prep=R.ROOT/'data/.b94/prepared'/('V1_r2_k0__'+cfg['prepared_key']+'.pt')
    assert hashlib.sha256(prep.read_bytes()).hexdigest()==cfg['prepared_artifact_sha256']
    artifact=torch.load(prep,map_location='cpu',weights_only=False)
    m=artifact['identity']['manifest'];del artifact
    selected={p:sorted(m[p])[0] for p in ['train','val']}
    assert all(selected[p] not in m['test'] for p in selected)
    raw=R.ROOT/'data/.b73/grid.npz';frames=R.ROOT/'data/.b78/frames.npz'
    with np.load(raw,allow_pickle=True) as source:
        indices=np.array([i for i,f in enumerate(source['fidx']) if int(f) in selected.values()])
        grid={key:source[key][indices] for key in ['time','seg','ti','fidx']}
    with np.load(frames,allow_pickle=True) as source:fr=source['fr'][indices]
    hashes={p.relative_to(R.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [config,prep,raw,frames]}
    return grid,fr,selected,hashes

def main():
    z=json.loads((R.CACHE/'result.json').read_text())
    for name,h in z['source_hashes'].items():assert R.sha(R.ROOT/name)==h
    assert R.sha(R.TOOLS/'lean.exe')==z['compiler_sha256']
    assert R.sha(R.PROJECT/'.lake/build/bin/visibilityBridge.exe')==z['executable_sha256']
    assert R.sha(R.ROOT/'data/.lean_toolchain/lean-4.36.0-windows.zip')==z['toolchain_archive_sha256']
    G,fr,selected,hashes=reload_inputs();assert hashes==z['input_hashes']
    qpath=R.CACHE/'queries.json';out=R.CACHE/'bridge.stdout'
    assert R.sha(qpath)==z['query_sha256'] and R.sha(out)==z['bridge_output_sha256']
    queries=json.loads(qpath.read_text());answers=out.read_text().splitlines()
    assert len(queries)==len(answers)==z['queries']
    covered=set();outcomes={};original={}
    with patch.object(R.D,'load_sources',return_value=(G,{})),patch.object(R.D,'_load_frames',return_value={'fr':fr}):
        for rule in R.RULES:
            R.D._ANCHOR_CACHE.clear();original[rule]=R.D.anchors(rule,G=G)
    for q,line in zip(queries,answers):
        t=q['target'];rule=q['rule'];fid=G['fidx'][t];seg=G['seg'][t]
        assert int(fid)==selected[q['partition']]
        cutoff=-1
        if rule=='S':
            indices=[j for j in range(t+1) if G['seg'][j]==seg and G['fidx'][j]==fid]
            for a,b in zip(indices,indices[1:]):
                delta=float(G['time'][b]-G['time'][a])
                if .015<delta<.045:cutoff=b-1
        valid=[];bits=[]
        for j in range(t):
            yes=bool(G['fidx'][j]==fid and G['seg'][j]==seg and np.isfinite(G['ti'][j]))
            if rule.startswith('D'):yes=yes and bool(G['time'][j]<=G['time'][t]-int(rule[1:])/1000.+.003)
            elif rule=='S':yes=yes and j<=cutoff
            elif rule=='K10':yes=yes and int(fr[j])>=0 and int(fr[j])%10==0
            else:raise ValueError(rule)
            bits.append('1' if yes else '0')
            if yes:valid.append(j)
        expected=max(valid,default=-1)
        assert q['bits']==''.join(bits) and q['expected']==expected
        assert int(original[rule][t])==expected
        if rule=='S':assert q['threshold']==cutoff
        if rule.startswith('D'):assert q['threshold']==float(G['time'][t]-int(rule[1:])/1000.+.003)
        assert line==f"{q['id']} {expected if expected>=0 else 'none'}"
        covered.add((q['partition'],rule))
        key=q['partition']+'/'+rule
        count=outcomes.setdefault(key,{'some':0,'none':0})
        count['some' if expected>=0 else 'none']+=1
    assert len(covered)==16
    # Verify the registered query set itself, independent of recorded queries.
    wanted=set()
    for part,fid in selected.items():
        rows=np.flatnonzero(G['fidx']==fid)
        segs=[rows[G['seg'][rows]==sg] for sg in dict.fromkeys(G['seg'][rows].tolist())]
        for segrows in [a for a in segs if len(a)>=2][:2]:
            for k in {0,len(segrows)//4,len(segrows)//2,3*len(segrows)//4,len(segrows)-1}:
                for rule in z['rules']:wanted.add((part,rule,int(segrows[k])))
    assert {(q['partition'],q['rule'],q['target']) for q in queries}==wanted
    replay=''.join(f"{q['id']} {q['bits']} {q['expected'] if q['expected']>=0 else 'none'}\n" for q in queries)
    native=subprocess.run([str(R.PROJECT/'.lake/build/bin/visibilityBridge.exe')],input=replay,text=True,capture_output=True,check=True,timeout=120)
    assert native.stdout.splitlines()==answers
    result={'status':'independent_checks_passed','queries':len(queries),'coverage_cells':len(covered),
            'result_sha256':R.sha(R.CACHE/'result.json'),'outcomes':outcomes,
            'scope':'Finite TRAIN/VAL bridge; not Python refinement or CUDA verification'}
    (R.CACHE/'independent.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    (R.ROOT/'docs/paper/b97_lean.json').write_text(json.dumps({**z,'independent':result},indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))

if __name__=='__main__':main()
