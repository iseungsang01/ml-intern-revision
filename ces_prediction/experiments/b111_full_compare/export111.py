"""CPU-only exact shared-input export; never fit or score an estimator."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import shutil,sys,zipfile
import numpy as np,torch
from common111 import *
sys.path.insert(0,str(HERE.parent/'b106_lstm'));import common106 as C
import data88 as D
from model106 import create_model
sys.path.insert(0,str(HERE.parent/'b108_colab'));from portable108 import TemperatureLSTM
sys.path.insert(0,str(HERE.parent/'b75_availability'));import stage_fits as SF

def aligned(path,idx,head=None):
    z=npz(path);pos=np.searchsorted(z['idx'],idx);assert (pos<len(z['idx'])).all() and np.array_equal(z['idx'][pos],idx)
    a=z['est'][pos];return a if head is None else a[:,head]

def main():
    guard();C.guard();C.inputs_guard();assert read(ROOT/'docs/paper/b109_fit_audit.json')['all_pass']
    assert not OUT.exists(),'Preserve prior attempt';OUT.mkdir();b=OUT/'bundle';b.mkdir()
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    selected=read(ROOT/'data/colab_b109/selected_paths.json');assert len(selected)==20
    G,R=C.M.load_sources();nr=len(R['gi']);ng=len(G['time']);ctx=SF.Ctx()
    rng=np.random.default_rng(20261124);codes=rng.permutation(nr);back=np.argsort(codes)
    fids=np.unique(G['fidx']);file_codes=dict(zip(fids.tolist(),rng.permutation(len(fids)).tolist()))
    clusters=np.unique(R['clus']);cluster_codes=dict(zip(clusters.tolist(),rng.permutation(len(clusters)).tolist()))
    row=np.full(ng,-1,np.int64);row[R['gi']]=codes
    dt=np.r_[0,np.diff(G['time'])];reset=np.r_[True,G['seg'][1:]!=G['seg'][:-1]];dt[reset]=0
    d={'fast':G['fast'].astype(np.float32),'ti':G['ti'].astype(np.float64),'dt_log':np.log1p(dt).astype(np.float32),'row':row}
    for rule in RULES:
        a=D._d85.anchors(rule,G=G);ok=a>=0;ii=np.arange(ng)[ok]
        assert (a[ok]<ii).all() and np.array_equal(G['seg'][a[ok]],G['seg'][ii]) and np.array_equal(G['fidx'][a[ok]],G['fidx'][ii])
        ages=np.zeros(ng,np.float32);ages[ok]=np.log1p(G['time'][ii]-G['time'][a[ok]])
        d['anchor_'+rule]=a.astype(np.int32);d['age_log_'+rule]=ages
    np.savez_compressed(b/'grid.npz',**d)
    common={'truth':R['tg_F'][back],'tstep':R['tstep'][back],'discharge':np.array([file_codes[int(x)] for x in R['fidx'][back]],np.int32),'cluster':np.array([cluster_codes[int(x)] for x in R['clus'][back]],np.int32)}
    reference_hashes={};rule_arrays={}
    for rule in RULES:
        root=ROOT/'data'/('.b78' if rule=='K10' else '.b75');p=root/'inputs'/(rule+'.npz');I=npz(p)
        rule_arrays[rule]=I
        for name in ('H','G','age'):common[rule+'_'+name]=I[name][back]
        reference_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    np.savez_compressed(b/'rows.npz',**common)
    tasks={};units={};private={};cached=0;builder_checks=0
    for ui,u in enumerate(UNITS):
        tag='U'+str(ui);outer=C.prepared(u+'__outer');om=outer['identity']['manifest'];fast,_,_=ctx.unit_arrays({'fit':om['train']})
        np.savez_compressed(b/(tag+'_fast.npz'),fast=fast[back])
        held_sets=[]
        for role in ('inner0','inner1','inner2','inner3','outer'):
            case=u+'__'+role;prep=C.prepared(case);m=prep['identity']['manifest'];s=prep['identity']['stats'];own=selected[case];weight=ROOT/own['path']/'best.pt';assert sha(weight)==own['best_sha256']
            assert s['train_files']==sorted(m['train']) and not set(m['train'])&set(m['held']+m['val']+m['test']) and not set(m['val'])&set(m['test'])
            if role!='outer':held_sets+=m['held'];assert set(m['held'])<=set(om['train'])
            if u.startswith('V2') and role!='outer':assert not set(R['clus'][np.isin(R['fidx'],m['held'])])&set(R['clus'][np.isin(R['fidx'],m['train'])])
            stats={k:v for k,v in s.items() if k!='train_files'}
            for split in ('train','val'):
                for seg in prep[split]['D10']:
                    assert np.array_equal(inputs(d,stats,seg['gi'],'D10'),seg['x'][:,:22]);cached+=1
            files=m['test'] if role=='outer' else m['held'];segments=[]
            for fid in files:
                ids=np.flatnonzero(G['fidx']==fid)
                for segment in np.unique(G['seg'][ids]):segments.append(ids[G['seg'][ids]==segment])
            gi=np.concatenate(segments);lengths=np.array([len(a) for a in segments],np.int64)
            assert len(np.unique(gi))==len(gi) and np.isin(G['fidx'][gi],files).all()
            task=tag+'_'+role
            for rule in RULES if role=='outer' else TRAIN_RULES:
                original=D.build_segments([files[0]],s,rule,G,R)
                for seg in original:
                    assert np.array_equal(inputs(d,stats,seg['gi'],rule),seg['x'][:,:22]);builder_checks+=1
            model=create_model('LSTM_F',{'dense':True},1701);model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));model.eval()
            portable=TemperatureLSTM();portable.load_state_dict(model.state_dict());portable.eval()
            xx=torch.from_numpy(inputs(d,stats,segments[0][:64],'D10')).unsqueeze(0)
            with torch.no_grad():ref=model(xx).numpy();got=portable(xx).numpy()
            assert np.allclose(ref,got,atol=1e-6,rtol=0)
            np.savez_compressed(b/(task+'.npz'),gi=gi.astype(np.int32),lengths=lengths,ref_x=xx.numpy(),ref_prediction=ref)
            shutil.copyfile(weight,b/(task+'.pt'))
            entry={'unit':tag,'role':role,'stats':stats,'weights_sha256':sha(weight),'manifest_sha256':C.digest(m),'stats_sha256':C.digest(s),'prepared_sha256':prep['artifact_sha256'],'files':{k:[file_codes[int(x)] for x in m[k]] for k in ('train','val','held','test')},'prefix_equal':True}
            tasks[task]=entry;private[task]={'case':case,'selected':own,'manifest':m,'stats':s}
            print('B111 exported '+task,flush=True)
        assert sorted(held_sets)==sorted(om['train']) and len(held_sets)==len(set(held_sets))
        units[tag]={'name':u,'outer':tag+'_outer','inner':[tag+'_inner'+str(k) for k in range(4)]}
        for rule in RULES:
            root=ROOT/'data'/('.b78' if rule=='K10' else '.b75');sets=npz(root/'sets.npz');mask=sets[('S_' if rule=='S' else 'common_')+'F']&np.isin(R['fidx'],om['test']);idx=np.flatnonzero(mask);assert len(idx)>0
            order=np.argsort(codes[idx]);idx=idx[order];arr={'idx':codes[idx],'truth':R['tg_F'][idx],'cluster':common['cluster'][codes[idx]],'discharge':common['discharge'][codes[idx]],'GP_only':rule_arrays[rule]['G'][idx],'hold':rule_arrays[rule]['H'][idx]}
            paths={n:(root/'fits'/f'{u}__{key}__{rule}__F.npz',None) for n,key in [('GP_history_boost','T0'),('GP_fast_boost','TX')]}
            paths.update({n:(ROOT/'data/.b105/predictions'/f'{u}__{n}__test__{rule}.npz',0) for n in ('TCN22','TCN35')})
            paths['GP_fair_mixture']=(ROOT/'data/.b105/trees'/u/f'{rule}__F__test__mixture.npz',None)
            if rule!='K10':paths['GP_fair_specialized']=(ROOT/'data/.b105/trees'/u/f'{rule}__F__test__specialized.npz',None)
            for n,(p,h) in paths.items():arr[n]=aligned(p,idx,h);reference_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
            assert all(np.isfinite(a).all() for a in arr.values())
            np.savez_compressed(b/(tag+'_'+rule+'_reference.npz'),**arr)
    for p in (HERE/'common111.py',HERE/'run111.py',HERE/'stats111.py',HERE.parent/'b108_colab/portable108.py'):shutil.copyfile(p,b/p.name)
    manifest={'registration':'PREREGISTRATION_B111.md','sources':sources(),'tasks':tasks,'units':units,'files':{p.name:sha(p) for p in b.iterdir()},'cached_D10_segments_bit_equal':cached,'original_builder_segments_bit_equal':builder_checks,'new_LSTM_fits':0,'post_fits':8,'reference_sha256':reference_hashes,'TEST_exported':True,'anonymized':True}
    write(b/'manifest.json',manifest);write(OUT/'private_map.json',{'tasks':private,'row_codes':codes.tolist(),'row_inverse':back.tolist(),'file_codes':file_codes,'cluster_codes':cluster_codes})
    archive=OUT/'ces_b111_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in b.iterdir():z.write(p,p.name)
    write(OUT/'export_verification.json',{'all_pass':True,'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'sources':sources(),'cached_segments':cached,'builder_segments':builder_checks,'TEST_exported':True,'local_fits':0})
    print('B111 bundle bytes='+str(archive.stat().st_size),flush=True)
if __name__=='__main__':main()
