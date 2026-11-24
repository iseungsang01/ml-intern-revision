"""CPU-only anonymous export of four preserved resumes and five fresh paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'data/colab_b109'
sys.path.insert(0,str(HERE.parent/'b107_patience'));import common107 as C
sys.path.insert(0,str(HERE.parent/'b108_colab'));from portable108 import TemperatureLSTM
from model106 import create_model
from prepare92 import batches as original_batches
from train109 import batches as portable_batches
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,z):p.write_text(json.dumps(z,indent=2),encoding='utf8')
def main():
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False;C.guard();C.inputs_guard()
    sources=[*HERE.glob('*.py'),HERE.parent/'PREREGISTRATION_B109.md',ROOT/'ces_prediction/research/colab_b109_operator.py']
    for p in sources:C.OLD.PREV.OLD.committed(p)
    stop=C.load(C.CACHE/'author_stop.json');assert stop['reason']=='author_requested_pause';assert not OUT.exists();OUT.mkdir();bundle=OUT/'bundle';bundle.mkdir()
    interrupted={r['case']:r for r in stop['stopped_children']};completed={};fresh=[]
    for case in C.CASES:
        if case in interrupted:continue
        if (C.new_rd(case)/'complete.json').exists():
            r=C.verify_new(case);assert r['patience_met'];completed[case]={'path':C.new_rd(case).relative_to(ROOT).as_posix(),'best_sha256':r['best_sha256']}
        elif (C.OLD.rd(case)/'metrics.json').exists():
            r=C.OLD.verify_run(case);assert r['stop']=='converged';completed[case]={'path':C.OLD.rd(case).relative_to(ROOT).as_posix(),'best_sha256':r['best_sha256']}
        else:assert not (C.OLD.rd(case)/'child_start.json').exists();fresh.append(case)
    assert len(completed)==11 and len(fresh)==5 and len(interrupted)==4
    original=create_model('LSTM_F',{'dense':True},1701);initial=original.state_dict();torch.save(initial,bundle/'initial.pt');tasks={};local={}
    for i,case in enumerate([*interrupted,*fresh]):
        tag='C'+str(i).zfill(2);prep=C.prepared(case);identity=prep['identity'];m=identity['manifest'];stats=identity['stats']
        assert not set(m['train'])&set(m['held']+m['val']+m['test']) and stats['train_files']==sorted(m['train'])
        arrays={}
        for split in ('train','val'):
            ss=prep[split]['D10'];arrays[split+'_lengths']=np.array([len(s['x']) for s in ss],np.int64)
            arrays[split+'_x']=np.concatenate([s['x'][:,:22] for s in ss]);arrays[split+'_y']=np.concatenate([s['y'] for s in ss]);arrays[split+'_mask']=np.concatenate([s['mask'] for s in ss])
            assert np.isfinite(arrays[split+'_x']).all() and not arrays[split+'_mask'][:,1].any() and arrays[split+'_mask'][:,0].any()
        entry={'mode':'resume' if case in interrupted else 'fresh','weights':tag+'_resume.pt' if case in interrupted else 'initial.pt','manifest_hash':C.digest(m),'stats_hash':C.digest(stats),'prepared_sha256':prep['artifact_sha256'],'OOF_exclusion_verified':True,'train_segments':len(prep['train']['D10']),'val_segments':len(prep['val']['D10'])}
        model=create_model('LSTM_F',{'dense':True},1701)
        if case in interrupted:
            record=interrupted[case];saved=ROOT/record['preserved'];checkpoint=saved/'latest_preserved.pt';assert sha(checkpoint)==record['checkpoint_sha256']
            w=torch.load(checkpoint,map_location='cpu',weights_only=False);history=C.load(saved/'validated_history.json');assert w['validated'] and history[-1]['epoch']==w['epoch'] and history[-1]['updates']==w['updates']
            assert w['config_hash']==record['config_hash'];w['history']=history;torch.save(w,bundle/entry['weights']);entry['best']=tag+'_best.pt';shutil.copyfile(saved/'best.pt',bundle/entry['best']);entry['parent_sha256']=sha(checkpoint);entry['start_epoch']=len(history);model.load_state_dict(w['model'])
        else:entry['start_epoch']=0;model.load_state_dict(initial)
        for split,epochs in [('train',sorted({0,entry['start_epoch']})),('val',[None])]:
            for epoch in epochs:
                old=original_batches(prep[split]['D10'],'cpu',shuffle_seed=None if epoch is None else epoch*10000+1701)
                new=list(portable_batches(arrays,split,epoch,device='cpu'));assert len(old)==len(new)
                for a,b in zip(old,new):
                    assert torch.equal(a['x'][...,:22],b[0]) and torch.equal(a['y'],b[1]) and torch.equal(a['mask'],b[2]),'Changed original batch membership/order'
        model.eval();portable=TemperatureLSTM();portable.load_state_dict(model.state_dict());portable.eval();x=torch.from_numpy(prep['val']['D10'][0]['x'][:64]).unsqueeze(0)
        with torch.no_grad():
            ref=model(x);got=portable(x[...,:22]);assert torch.allclose(ref,got,atol=1e-6,rtol=0)
        arrays.update(ref_x=x[...,:22].numpy(),ref_prediction=ref.numpy(),mu=np.array(stats['ti_mu']),sd=np.array(stats['ti_sd']))
        np.savez_compressed(bundle/(tag+'.npz'),**arrays);tasks[tag]=entry;local[tag]={'case':case,'mode':entry['mode'],'manifest':m,'stats':stats,'original_identity_hash':C.digest(identity),'prefix_equal':True}
        print('Exported '+tag+' '+entry['mode']+' from epoch '+str(entry['start_epoch']),flush=True)
    shutil.copyfile(HERE/'train109.py',bundle/'train109.py');shutil.copyfile(HERE.parent/'b108_colab/portable108.py',bundle/'portable108.py')
    manifest={'registration':'PREREGISTRATION_B109.md','tasks':tasks,'retained_completed_count':11,'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},'files':{p.name:sha(p) for p in bundle.iterdir()},'TEST_exported':False,'local_GPU_fits':0}
    save(bundle/'manifest.json',manifest);save(OUT/'local_case_map.json',{'tasks':local,'retained':completed,'local_pause':True})
    archive=OUT/'ces_b109_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in bundle.iterdir():z.write(p,p.name)
    save(OUT/'export_verification.json',{'all_pass':True,'archive_sha256':sha(archive),'archive_MiB':archive.stat().st_size/1024**2,'tasks':9,'resumes':4,'fresh':5,'retained':11,'local_GPU_fits':0,'TEST_exported':False})
    print('B109 bundle MiB '+str(round(archive.stat().st_size/1024**2,2)),flush=True)
if __name__=='__main__':main()
