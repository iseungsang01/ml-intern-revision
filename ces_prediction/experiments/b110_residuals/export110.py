"""CPU serialization only, using pinned real validation sequences and controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import shutil,sys,zipfile
import numpy as np,torch
from common110 import *

sys.path.insert(0,str(HERE.parent/'b106_lstm'))
import common106 as C
from model106 import create_model
sys.path.insert(0,str(HERE.parent/'b108_colab'))
from portable108 import TemperatureLSTM

def take(p,idx,head=None):
    z=npz(p);j=np.searchsorted(z['idx'],idx)
    assert (j<len(z['idx'])).all() and np.array_equal(z['idx'][j],idx)
    a=z['est'][j];return a if head is None else a[:,head]

def main():
    guard();C.guard();C.inputs_guard()
    assert read(ROOT/'docs/paper/b109_fit_audit.json')['all_pass']
    assert not OUT.exists(),'Prior attempt exists; inspect, do not overwrite'
    OUT.mkdir();bundle=OUT/'bundle';bundle.mkdir()
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    selection=read(ROOT/'data/colab_b109/selected_paths.json');assert len(selection)==20
    G,R=C.M.load_sources();I=npz(ROOT/'data/.b75/inputs/D10.npz');sets=npz(ROOT/'data/.b75/sets.npz')
    lookup=np.full(len(G['time']),-1,np.int64);lookup[R['gi']]=np.arange(len(R['gi']))
    tasks={};private={}
    for k,u in enumerate(UNITS):
        tag='U'+str(k);case=u+'__outer';p=C.prepared(case);m=p['identity']['manifest'];s=p['identity']['stats']
        assert not set(m['train'])&set(m['val']+m['test']) and not set(m['val'])&set(m['test'])
        assert s['train_files']==sorted(m['train'])
        selected=selection[case];weights=ROOT/selected['path']/'best.pt';assert sha(weights)==selected['best_sha256']
        base=ROOT/'data/.b105/trees'/u/'D10__F__val.npz';idx=npz(base)['idx']
        assert len(idx)>0 and (np.diff(idx)>0).all() and np.isin(R['fidx'][idx],m['val']).all()
        controls={name:take(ROOT/'data/.b105/predictions'/f'{u}__{name}__val__D10.npz',idx,0) for name in ('TCN22','TCN35')}
        controls.update(fair_specialized=npz(base)['est'],fair_mixture=take(base.with_name('D10__F__val__mixture.npz'),idx),GP=I['G'][idx],hold=I['H'][idx])
        ss=p['val']['D10'];lengths=np.array([len(x['x']) for x in ss],np.int64)
        seqidx=np.concatenate([lookup[x['gi']] for x in ss]);valid=seqidx>=0
        assert len(np.unique(seqidx[valid]))==valid.sum()
        sorter=np.argsort(seqidx[valid]);sorted_idx=seqidx[valid][sorter];j=np.searchsorted(sorted_idx,idx)
        assert (j<len(sorted_idx)).all() and np.array_equal(sorted_idx[j],idx)
        positions=np.flatnonzero(valid)[sorter[j]]
        train=np.isin(R['fidx'],m['train'])&sets['common_F']
        change=np.abs(I['G']-I['H']);assert train.any() and np.isfinite(change[train]).all()
        q=np.quantile(change[train],.75,method='linear')
        x=np.concatenate([v['x'][:,:22] for v in ss]);truth=R['tg_F'][idx]
        y=np.concatenate([v['y'][:,0] for v in ss])[positions]*s['ti_sd']+s['ti_mu']
        assert np.allclose(y,truth,atol=.001,rtol=1e-5),'VAL F target mismatch'
        original=create_model('LSTM_F',{'dense':True},1701);original.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));original.eval()
        portable=TemperatureLSTM();portable.load_state_dict(original.state_dict());portable.eval()
        xx=torch.from_numpy(ss[0]['x'][:64]).unsqueeze(0)
        with torch.no_grad():
            ref=original(xx).numpy();got=portable(xx[...,:22]).numpy()
        assert np.allclose(ref,got,atol=1e-6,rtol=0)
        arrays={'x':x,'lengths':lengths,'positions':positions,'truth':truth,'idx':np.arange(len(idx),dtype=np.int64),'cluster':np.unique(R['clus'][idx],return_inverse=True)[1],'discharge':np.unique(R['fidx'][idx],return_inverse=True)[1],'age':I['age'][idx],'history_change':change[idx],'change_q75':np.array(q),'ref_x':xx[...,:22].numpy(),'ref_prediction':ref,'mu':np.array(s['ti_mu']),'sd':np.array(s['ti_sd']),**controls}
        assert all(np.isfinite(a).all() for a in arrays.values())
        assert all(arrays[n].shape==truth.shape for n in controls)
        np.savez_compressed(bundle/(tag+'.npz'),**arrays);shutil.copyfile(weights,bundle/(tag+'.pt'))
        tasks[tag]={'unit':u,'weights_sha256':sha(weights),'manifest_sha256':C.digest(m),'stats_sha256':C.digest(s),'prepared_sha256':p['artifact_sha256'],'rows':len(idx),'segments':len(ss),'prefix_pass':True,'VAL_only_verified':True}
        private[tag]={'case':case,'global_idx':idx.tolist(),'selected':selected,'manifest':m,'stats':s,'q75_TRAIN_only':float(q)}
        print('Exported validation '+tag+' rows='+str(len(idx)),flush=True)
    for src in (HERE/'run110.py',HERE/'common110.py',HERE.parent/'b108_colab/portable108.py'):shutil.copyfile(src,bundle/src.name)
    manifest={'registration':'PREREGISTRATION_B110.md','sources':sources(),'tasks':tasks,'files':{p.name:sha(p) for p in bundle.iterdir()},'models':MODELS,'groups':GROUPS,'draws':DRAW,'TEST_exported':False,'new_fits':0}
    write(bundle/'manifest.json',manifest);write(OUT/'private_map.json',private)
    archive=OUT/'ces_b110_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in bundle.iterdir():z.write(p,p.name)
    write(OUT/'export_verification.json',{'all_pass':True,'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'sources':sources(),'tasks':tasks,'TEST_exported':False,'new_fits':0})
    print('B110 serialized bytes='+str(archive.stat().st_size),flush=True)

if __name__=='__main__':main()
