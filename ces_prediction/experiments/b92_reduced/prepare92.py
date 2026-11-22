"""Precache actual train/validation arrays; test segments are never constructed."""
import json,time
import numpy as np
import torch
import common92 as M
import data88 as D
def raw_identity():
    return {p.name:M.sha(p) for p in (M.DATA/'.b73/grid.npz',M.DATA/'.b73/rows.npz')}
def prepare_unit(name):
    pins=M.guard();u=next(u for u in M.select_units() if u['name']==name)
    G,R=M.load_sources();manifest=D.unit_manifest(u);stats=D.fit_stats(manifest['train'],G,R)
    identity={'sources':M.sources(),'frozen':pins,'raw':raw_identity(),'manifest':manifest,'stats':stats,'width':35}
    key=M.digest(identity);path=M.CACHE/'prepared'/f'{name}__{key}.pt'
    if path.exists():
        meta=json.loads(path.with_suffix('.json').read_text());assert meta['key']==key and meta['sha256']==M.sha(path),'Prepared artifact changed'
        z=torch.load(path,map_location='cpu',weights_only=False);assert z['identity']==identity
        return {**z,'artifact_sha256':meta['sha256']}
    started=time.perf_counter();train={};val={}
    for rule in M.TRAIN_RULES:
        train[rule]=D.build_segments(manifest['train'],stats,rule,G,R)
        val[rule]=D.build_segments(manifest['val'],stats,rule,G,R)
    keys=sorted((s['fidx'],s['segment']) for s in train[M.TRAIN_RULES[0]])
    assert keys and all(sorted((s['fidx'],s['segment']) for s in train[r])==keys for r in M.TRAIN_RULES)
    assert not set(manifest['train'])&set(manifest['val']) and not (set(manifest['train'])|set(manifest['val']))&set(manifest['test'])
    z={'identity':identity,'key':key,'train':train,'val':val,'keys':keys,'prep_seconds':time.perf_counter()-started}
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(z,tmp);tmp.replace(path)
    artifact=M.sha(path);M.save(path.with_suffix('.json'),{'key':key,'sha256':artifact})
    return {**z,'artifact_sha256':artifact}
def batches(segments,device,batch_size=16,shuffle_seed=None):
    # Same random.Random shuffle/membership as the inherited B85 batch builder.
    import random
    if shuffle_seed is not None:
        order=list(range(len(segments)));random.Random(shuffle_seed).shuffle(order);segments=[segments[i] for i in order]
    result=[]
    for start in range(0,len(segments),batch_size):
        ss=segments[start:start+batch_size];lengths=[len(s['x']) for s in ss];n=max(lengths)
        x=np.zeros((len(ss),n,35),np.float32);y=np.zeros((len(ss),n,2),np.float32);mask=np.zeros((len(ss),n,2),bool)
        for j,s in enumerate(ss):
            k=lengths[j];x[j,:k]=s['x'];y[j,:k]=s['y'];mask[j,:k]=s['mask']
        result.append({'x':torch.from_numpy(x).to(device),'y':torch.from_numpy(y).to(device),'mask':torch.from_numpy(mask).to(device),'lengths':torch.tensor(lengths,dtype=torch.long)})
    return result
def epoch_segments(prepared,unit,seed,epoch):
    lookup={r:{(s['fidx'],s['segment']):s for s in prepared['train'][r]} for r in M.TRAIN_RULES}
    return [lookup[M.B85.rule_for(unit,seed,epoch,seg)][(fid,seg)] for fid,seg in prepared['keys']]
