"""Bounded TRAIN-only scalar fits, then S1 checks; no outer TEST access."""
import data98 as D
import json,subprocess,sys,time
import numpy as np
import model98 as M
sys.path.insert(0,str(D.HERE/'lead_checks'))
import recurrence98 as Q

def sources():
    files=[*D.HERE.glob('*.py'),*(D.HERE/'lead_checks').glob('*.py'),D.HERE.parent/'PREREGISTRATION_B98.md']
    for p in files:
        rel=p.relative_to(D.ROOT).as_posix()
        subprocess.run(['git','cat-file','-e','HEAD:'+rel],cwd=D.ROOT,check=True)
        subprocess.run(['git','diff','--quiet','HEAD','--',rel],cwd=D.ROOT,check=True)
    return {p.relative_to(D.ROOT).as_posix():D.sha(p) for p in files}

def main():
    pins=sources();D.CACHE.mkdir(parents=True,exist_ok=True)
    assert not (D.CACHE/'started.json').exists(),'Prior attempt exists; no automatic restart'
    start=time.time();(D.CACHE/'started.json').write_text(json.dumps({'sources':pins,'started':start}),encoding='utf8')
    units=[]
    for unit in D.UNITS:
        began=time.time();groups,meta=D.load(unit);models={}
        for kind in ('M','H','L'):
            models[kind]=M.fit(groups['train'],kind)
            print(json.dumps({'unit':unit,'model':kind,'tau':models[kind]['tau'],'train_mse':models[kind]['train_mse'],'seconds':time.time()-began}),flush=True)
        checks=Q.check(groups['val'],models,M.predict);preds=[];truth=None;mapping=None
        for kind in ('M','H','L'):
            p,y,idx=M.predict(groups['val'],models[kind]);preds.append(p)
            if truth is None:truth=y;mapping=idx
            else:assert np.array_equal(truth,y) and np.array_equal(mapping,idx)
        hold=[];clusters=[];shots=[];gi=[];source_gi=[]
        for segment,row,rule in mapping:
            s=groups['val'][segment];hold.append(s['ti'][s['anchor'][row,rule]]);clusters.append(s['cluster']);shots.append(s['shot'])
            gi.append(s['gi'][row]);source_gi.append(s['gi'][s['anchor'][row,rule]])
        path=D.CACHE/(unit+'.npz')
        np.savez_compressed(path,pred=np.column_stack([*preds,hold]),y=truth,mapping=mapping,
                            cluster=np.asarray(clusters),shot=np.asarray(shots),gi=np.asarray(gi),source_gi=np.asarray(source_gi))
        record={'unit':unit,'models':models,'checks':checks,'metadata':meta,'prediction_sha256':D.sha(path),'seconds':time.time()-began}
        (D.CACHE/(unit+'.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf8');units.append(unit)
    result={'sources':pins,'units':units,'all_checks':True,'seconds':time.time()-start,'test_constructed':False,'scope':'Representative-unit internal VAL only; not outer TEST or physical identification'}
    assert pins==sources()
    (D.CACHE/'fit_complete.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'fit_complete':True,'units':len(units),'seconds':result['seconds']}))

if __name__=='__main__':main()
