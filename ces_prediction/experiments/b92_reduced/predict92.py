"""Selected-pair TEST inference only after all fresh confirmation completions."""
import json
import numpy as np
import torch
import common92 as M
import data88 as D
from train92 import verify_complete,checks_ready
from runtime92 import decision
from model92 import create_model
from select92 import lock_selection
def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def guarded_ready():
    M.guard();checks_ready();runtime=decision();s=lock_selection();assert s['chosen'],'No candidate advanced; TEST forbidden'
    jobs=M.confirmation_jobs(s['chosen'])
    for job in jobs:assert verify_complete(M.run_dir(*job))
    return s,runtime,jobs
def validate_predictions():
    s,_,jobs=guarded_ready();expected={}
    for stage,u,seed,code in jobs:
        cfg=json.loads((M.run_dir(stage,u,seed,code)/'config.json').read_text())
        for rule in M.RULES:
            z=npz(M.CACHE/'predictions'/f'{u}__{seed}__{code}__{rule}.npz')
            assert str(z['config_hash'])==cfg['config_hash'] and str(z['sources_hash'])==M.digest(M.sources())
            assert z['est'].shape==(len(z['idx']),2) and len(z['idx']) and np.all(np.diff(z['idx'])>0) and np.isfinite(z['est']).all()
            if u not in expected:expected[u]=z['idx']
            else:assert np.array_equal(expected[u],z['idx'])
    return True
def main():
    selected,runtime,jobs=guarded_ready();G,R=M.load_sources();lookup={int(g):i for i,g in enumerate(R['gi'])};assert len(lookup)==len(R['gi'])
    folder=M.CACHE/'predictions';folder.mkdir(parents=True,exist_ok=True)
    for stage,u,seed,code in jobs:
        rd=M.run_dir(stage,u,seed,code);cfg=json.loads((rd/'config.json').read_text());stats=json.loads((rd/'stats.json').read_text());manifest=json.loads((rd/'manifest.json').read_text())
        model=create_model(code,runtime,seed).to(runtime['device']);model.load_state_dict(torch.load(rd/'best.pt',map_location=runtime['device'],weights_only=True));model.eval()
        for rule in M.RULES:
            idx=[];est=[]
            with torch.no_grad():
                for seg in D.build_segments(manifest['test'],stats,rule,G,R):
                    out=model(torch.from_numpy(seg['x']).unsqueeze(0).to(runtime['device']),torch.tensor([len(seg['x'])])).cpu().numpy()[0]*stats['ti_sd']+stats['ti_mu']
                    for i,g in enumerate(seg['gi']):
                        if int(g) in lookup:idx.append(lookup[int(g)]);est.append(out[i])
            idx=np.asarray(idx,dtype=np.int64);est=np.asarray(est,dtype=np.float32);order=np.argsort(idx);idx,est=idx[order],est[order]
            assert len(idx) and np.all(np.diff(idx)>0) and np.isfinite(est).all()
            p=folder/f'{u}__{seed}__{code}__{rule}.npz';tmp=p.with_suffix('.tmp.npz');np.savez(tmp,idx=idx,est=est,config_hash=cfg['config_hash'],sources_hash=M.digest(M.sources()));tmp.replace(p)
        # Labels remain outside the predictor, checked on actual held-out traces.
        altered=dict(R);altered['tg_F']=R['tg_F']+500;altered['tg_L5']=R['tg_L5']+500
        left=D.build_segments(manifest['test'][:1],stats,'D100',G,R);right=D.build_segments(manifest['test'][:1],stats,'D100',G,altered)
        with torch.no_grad():
            for a,b in zip(left,right):
                assert np.array_equal(a['x'],b['x'])
                x=torch.from_numpy(a['x']).unsqueeze(0).to(runtime['device']);y=torch.from_numpy(b['x']).unsqueeze(0).to(runtime['device'])
                assert torch.allclose(model(x),model(y),atol=1e-5,rtol=0)
        print('Predicted '+rd.name,flush=True)
    validate_predictions();M.save(M.CACHE/'prediction_checks.json',{'all_pass':True,'sources':M.sources(),'selection_hash':M.sha(M.CACHE/'selection.json'),'labels_not_inputs':True})
if __name__=='__main__':main()
