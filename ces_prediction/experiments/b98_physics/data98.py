"""Reuse registered prepared TRAIN/VAL only; no outer TEST construction."""
import hashlib,json,os
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
os.environ.update(CES_DROP_STUCK_TARGETS='1',CES_TI_SPIKE_CUT_EV='3000',CES_PER_SHOT_NORM='0',CES_SEQ_NORM_EXPANDING='0',CES_SEQ_ZERO_MC='1',CES_SEQ_ZERO_VT_HISTORY='1',CES_SEQ_ZERO_VT='1',CUDA_VISIBLE_DEVICES='')
import numpy as np
import torch
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;CACHE=ROOT/'data/.b98'
RULES=['D10','D50','D100','D300','D650','D1000','S']
UNITS=['V1_r2_k0','V2_s1_k0','V3_q3','V4a']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(unit):
    paths=list((ROOT/'data/.b94/prepared').glob(unit+'__*.pt'));assert len(paths)==1
    path=paths[0];side=json.loads(path.with_suffix('.json').read_text());assert sha(path)==side['sha256']
    z=torch.load(path,map_location='cpu',weights_only=False);identity=z['identity'];m=identity['manifest'];stats=identity['stats']
    assert stats['train_files']==sorted(m['train']) and stats['ti_sd']>0
    assert not set(m['train'])&set(m['val']) and not (set(m['train'])|set(m['val']))&set(m['test'])
    grid=ROOT/'data/.b73/grid.npz';beam=ROOT/'data/.b79/beam_cols.npz';validity=ROOT/'data/.b79/checks/validity.json'
    assert identity['raw']['grid.npz']==sha(grid)
    bad=set(json.loads(validity.read_text())['invalid_files'])
    with np.load(grid,allow_pickle=True) as g:
        time=g['time'];ti=g['ti'];shots=g['shot'];fidx=g['fidx'];seg=g['seg']
    assert len(shots)==int(fidx.max())+1 and all(isinstance(x,int) for x in bad)
    assert bad<=set(map(int,shots)),'B79 validity IDs must be delivered shot numbers'
    raw_hashes={}
    with np.load(beam) as nbi:
        assert len(nbi['P'])==len(time)==len(nbi['t'])
        take=np.flatnonzero(np.isin(fidx,m['train']+m['val']))
        assert np.all(np.abs(time[take]-nbi['t'][take])<=1e-4)
        power=np.full(len(time),np.nan)
        raw=nbi['P'][take];power[take]=np.where(np.isfinite(raw),raw,0).sum(axis=1)
        columns=[s+'_P' for s in ['NB1B','NB1C','NB2A','NB2B','NB2C']]
        for fid in m['train']+m['val']:
            if int(shots[fid]) in bad:continue
            path_raw=ROOT/'data/nbi'/('s'+str(int(shots[fid]))+'_NBI.csv')
            frame=pd.read_csv(path_raw,usecols=['time',*columns]);at=np.flatnonzero(fidx==fid)
            assert len(frame)==len(at)
            assert np.all(np.abs(frame['time'].to_numpy()-time[at])<=1e-4)
            assert np.array_equal(frame[columns].to_numpy(),nbi['P'][at],equal_nan=True)
            raw_hashes[path_raw.relative_to(ROOT).as_posix()]=sha(path_raw)
    with np.load(ROOT/'data/.b73/rows.npz') as rows:
        rf=rows['fidx'];rc=rows['clus']
        cluster={int(f):int(rc[np.flatnonzero(rf==f)[0]]) for f in set(m['train']+m['val']) if np.any(rf==f)}
    order=np.argsort(shots,kind='stable');expected_cluster=np.zeros(len(shots),dtype=int)
    expected_cluster[order]=np.r_[0,np.cumsum(np.diff(shots[order])>3)]
    assert all(cluster[f]==expected_cluster[f] for f in cluster)
    groups={};drops={};dropped_files={}
    for part in ('train','val'):
        groups[part]=[];drops[part]={'invalid_nbi_segments':0,'empty_common_segments':0,'nonfinite_context_segments':0}
        dropped_files[part]=set()
        lookup={r:{(s['fidx'],s['segment']):s for s in z[part][r]} for r in RULES}
        for row in z[part]['D10']:
            fid=row['fidx'];sg=row['segment'];ids=row['gi'];tm=time[ids];yt=ti[ids]
            if int(shots[fid]) in bad:
                drops[part]['invalid_nbi_segments']+=1;dropped_files[part].add(fid);continue
            assert fid in m[part] and fid not in m['test'] and np.all(fidx[ids]==fid) and np.all(seg[ids]==sg)
            assert np.all(np.diff(tm)>0)
            assert len(tm)<2 or np.max(np.diff(tm))<.5
            assert np.all(yt[np.isfinite(yt)]<=3000)
            fast=np.asarray(row['x'][:,:13],dtype=np.float64)
            u=np.column_stack([np.ones(len(ids)),fast[:,9:13].mean(1),fast[:,:9].mean(1),power[ids]])
            if not np.isfinite(u).all():drops[part]['nonfinite_context_segments']+=1;continue
            obs=np.flatnonzero(np.isfinite(yt));anchor=[]
            for rule in RULES:
                if rule=='S':
                    dt=np.r_[0.,np.diff(tm)];cut=np.maximum.accumulate(np.where((dt>.015)&(dt<.045),np.arange(len(tm))-1,-1))
                    pos=np.searchsorted(obs,cut,side='right')-1
                else:
                    pos=np.searchsorted(tm[obs],tm-int(rule[1:])/1000.+.003,side='right')-1
                    pos=np.minimum(pos,np.searchsorted(obs,np.arange(len(tm)),side='left')-1)
                src=np.full(len(tm),-1,dtype=int);ok=pos>=0
                if len(obs):src[ok]=obs[pos[ok]]
                original=lookup[rule][(fid,sg)]
                assert np.array_equal(original['gi'],ids)
                assert np.array_equal(original['mask'],row['mask']) and np.array_equal(original['y'],row['y'])
                assert np.array_equal(original['x'][:,:13],row['x'][:,:13])
                assert np.array_equal(ok,original['x'][:,18].astype(bool))
                if ok.any():
                    assert np.all(src[ok]<np.flatnonzero(ok))
                    assert np.allclose((yt[src[ok]]-stats['ti_mu'])/stats['ti_sd'],original['x'][ok,16],rtol=1e-5,atol=1e-5)
                    assert np.allclose(np.log1p(tm[ok]-tm[src[ok]]),original['x'][ok,17],rtol=1e-5,atol=1e-6)
                anchor.append(src)
            anchor=np.column_stack(anchor)
            target=row['y'][:,0].astype(np.float64)*stats['ti_sd']+stats['ti_mu']
            labeled=row['mask'][:,0].astype(bool)
            assert np.allclose(target[labeled],yt[labeled],rtol=1e-5,atol=.002)
            mask=labeled&(anchor>=0).all(axis=1)&np.isfinite(yt)
            if not mask.any():drops[part]['empty_common_segments']+=1;continue
            groups[part].append({'t':tm,'ti':yt,'target':yt.copy(),'u':u,'anchor':anchor,'mask':mask,
                                 'file':int(fid),'shot':int(shots[fid]),'cluster':cluster[int(fid)],'segment':int(sg),'gi':ids})
        assert groups[part],part+' has no eligible real segments'
        assert dropped_files[part]=={f for f in m[part] if int(shots[f]) in bad}
    meta={'unit':unit,'manifest':m,'normalization':stats,'drops':drops,
          'hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in (path,path.with_suffix('.json'),grid,beam,validity)},
          'raw_nbi_hashes':raw_hashes,'dropped_file_ids':{k:sorted(v) for k,v in dropped_files.items()},
          'train_val_cluster_overlap':len({s['cluster'] for s in groups['train']}&{s['cluster'] for s in groups['val']}),
          'env':{k:v for k,v in os.environ.items() if k.startswith('CES_')},
          'scope':'Existing normalization includes original TRAIN files; regression fits exclude invalid-NBI files; no outer TEST segments'}
    return groups,meta
