"""B88 prediction plumbing, locally corrected from rejected Claude draft."""
import hashlib,json
import numpy as np
import torch
import common88 as M
import data88 as D
from model88 import create_model
from adapter88 import activate,guarded_ready,dependency_ready,source_inventory
def prediction_hash():return hashlib.sha256(json.dumps(source_inventory(),sort_keys=True).encode()).hexdigest()
def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def validate_predictions():
    assert dependency_ready();guarded_ready();t=activate();source=prediction_hash()
    for u in M.select_units():
        expected=None
        for seed in M.INITS:
            for var in M.VARIANTS:
                cfg=json.loads((t.get_run_dir(u['name'],seed,var)/'config.json').read_text(encoding='utf8'))
                for rule in M.RULES:
                    z=npz(M.CACHE/'predictions'/f"{u['name']}__{seed}__{var}__{rule}.npz")
                    assert z['idx'].ndim==1 and len(z['idx']) and np.all(np.diff(z['idx'])>0)
                    assert z['est'].shape==(len(z['idx']),2) and np.isfinite(z['est']).all()
                    assert str(z['config_hash'])==cfg['config_hash'] and str(z['source_hash'])==source
                    if expected is None:expected=z['idx']
                    else:assert np.array_equal(z['idx'],expected),'Prediction row support differs'
    return True
def main():
    M.assert_frozen();assert dependency_ready();guarded_ready();t=activate()
    G,R=M.load_sources();lookup={int(g):i for i,g in enumerate(R['gi'])}
    assert len(lookup)==len(R['gi'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');outdir=M.CACHE/'predictions';outdir.mkdir(parents=True,exist_ok=True)
    source=prediction_hash()
    for u in M.select_units():
        for seed in M.INITS:
            for var in M.VARIANTS:
                rd=t.get_run_dir(u['name'],seed,var);assert t.verify_complete(rd)
                cfg=json.loads((rd/'config.json').read_text(encoding='utf8'));stats=json.loads((rd/'stats.json').read_text(encoding='utf8'));manifest=json.loads((rd/'manifest.json').read_text(encoding='utf8'))
                assert manifest==D.unit_manifest(u)
                model=create_model(var,seed).to(device);model.load_state_dict(torch.load(rd/'best.pt',map_location=device,weights_only=True));model.eval()
                for rule in M.RULES:
                    indices=[];estimates=[]
                    with torch.no_grad():
                        for s in D.build_segments(manifest['test'],stats,rule,G,R):
                            x=torch.from_numpy(s['x']).unsqueeze(0).to(device)
                            est=model(x,torch.tensor([len(s['x'])])).cpu().numpy()[0]*stats['ti_sd']+stats['ti_mu']
                            keep=[i for i,g in enumerate(s['gi']) if int(g) in lookup]
                            if keep:
                                indices.extend(lookup[int(s['gi'][i])] for i in keep);estimates.extend(est[keep])
                    idx=np.asarray(indices,dtype=np.int64);est=np.asarray(estimates,dtype=np.float32);order=np.argsort(idx);idx,est=idx[order],est[order]
                    assert len(idx) and np.all(np.diff(idx)>0) and np.isfinite(est).all()
                    path=outdir/f"{u['name']}__{seed}__{var}__{rule}.npz";tmp=path.with_suffix('.tmp.npz')
                    np.savez(tmp,idx=idx,est=est,config_hash=np.array(cfg['config_hash']),source_hash=np.array(source));tmp.replace(path)
                print('Predicted '+rd.name,flush=True)
    # Real held-out inputs, frozen train statistics; both fitted variants.
    u=M.select_units()[0];seed=M.INITS[0]
    for var in M.VARIANTS:
        rd=t.get_run_dir(u['name'],seed,var);stats=json.loads((rd/'stats.json').read_text());manifest=json.loads((rd/'manifest.json').read_text())
        altered=dict(R);altered['tg_F']=R['tg_F']+500;altered['tg_L5']=R['tg_L5']+500
        a=D.build_segments(manifest['test'][:2],stats,'D100',G,R);b=D.build_segments(manifest['test'][:2],stats,'D100',G,altered)
        model=create_model(var,seed).to(device);model.load_state_dict(torch.load(rd/'best.pt',map_location=device,weights_only=True));model.eval()
        assert len(a)==len(b) and a
        with torch.no_grad():
            for left,right in zip(a,b):
                assert np.array_equal(left['x'],right['x'])
                x=torch.from_numpy(left['x']).unsqueeze(0).to(device);y=torch.from_numpy(right['x']).unsqueeze(0).to(device)
                assert torch.allclose(model(x),model(y),atol=1e-5,rtol=0)
    validate_predictions()
    M.C.save_json(M.CACHE/'prediction_checks.json',{'all_pass':True,'source_hash':source,'label_mutation_invariance':True,'same_rows_all_rules_initializations_variants':True})
if __name__=='__main__':main()
