"""Test inference only after immutable validation promotion and all required fits."""
import json
import common90 as M
import data85 as D
from adapter90 import activate,guarded_ready,dependency_ready

def npz(path):return M.C.load_npz(path)
def prediction_hash():return M.hashlib.sha256(json.dumps({'sources':M.source_inventory(),'promotion':M.sha(M.CACHE/'promotion.json')},sort_keys=True).encode()).hexdigest()
def path(method,unit,seed,var,rule):return M.CACHE/'predictions'/method/f'{unit}__{seed}__{var}__{rule}.npz'

def validate_predictions():
    import numpy as np
    assert dependency_ready();guarded_ready();source=prediction_hash();supports={}
    for m,u,s,v in M.all_jobs():
        t=activate(m);cfg=json.loads((t.get_run_dir(u,s,v)/'config.json').read_text())
        for rule in M.RULES:
            z=npz(path(m,u,s,v,rule));idx=z['idx']
            assert idx.ndim==1 and len(idx) and np.all(np.diff(idx)>0)
            assert z['est'].shape==(len(idx),2) and np.isfinite(z['est']).all()
            assert str(z['config_hash'])==cfg['config_hash'] and str(z['source_hash'])==source
            if u in supports:assert np.array_equal(idx,supports[u])
            else:supports[u]=idx
    return True

def main():
    assert dependency_ready();guarded_ready() # includes locked promotion recomputation before any test inference
    import numpy as np
    import torch
    from models90 import create_model
    G,R=M.load_sources();lookup={int(g):i for i,g in enumerate(R['gi'])};assert len(lookup)==len(R['gi'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');source=prediction_hash()
    for method,unit,seed,var in M.all_jobs():
        t=activate(method);rd=t.get_run_dir(unit,seed,var);assert t.verify_complete(rd)
        stats=json.loads((rd/'stats.json').read_text());manifest=json.loads((rd/'manifest.json').read_text());cfg=json.loads((rd/'config.json').read_text())
        model=create_model(method=method,variant=var,init_seed=seed).to(device);model.load_state_dict(torch.load(rd/'best.pt',map_location=device,weights_only=True));model.eval()
        for rule in M.RULES:
            target=path(method,unit,seed,var,rule)
            if target.exists():
                old=npz(target);assert str(old['source_hash'])==source and str(old['config_hash'])==cfg['config_hash'];continue
            indices=[];estimates=[]
            with torch.no_grad():
                for s in D.build_segments(manifest['test'],stats,rule,G,R):
                    x=torch.from_numpy(s['x']).unsqueeze(0).to(device)
                    means=model(x,torch.tensor([len(s['x'])])).cpu().numpy()[0]*stats['ti_sd']+stats['ti_mu']
                    keep=[j for j,g in enumerate(s['gi']) if int(g) in lookup]
                    if keep:indices.extend(lookup[int(s['gi'][j])] for j in keep);estimates.extend(means[keep])
            idx=np.asarray(indices,dtype=np.int64);est=np.asarray(estimates,dtype=np.float32);order=np.argsort(idx);idx,est=idx[order],est[order]
            assert len(idx) and np.all(np.diff(idx)>0) and np.isfinite(est).all()
            M.C.save_npz(target,idx=idx,est=est,config_hash=np.array(cfg['config_hash']),source_hash=np.array(source))
        print('Predicted '+method+'/'+rd.name,flush=True)
    validate_predictions()
    M.save(M.CACHE/'prediction_checks.json',{'all_pass':True,'source_hash':source,'same_rows_all_methods_rules_seeds_variants':True,'promotion_hash':M.sha(M.CACHE/'promotion.json')})
if __name__=='__main__':main()
