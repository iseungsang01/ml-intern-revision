"""Frozen LSTM held-file/test prediction; no scoring during calibration inference."""
import argparse
import numpy as np,torch
import common106 as C
import data88 as D
from model106 import create_model
from prepare92 import batches

def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main(split):
    C.guard();C.inputs_guard();assert C.load(C.CACHE/'fit_checks.json')['all_pass']
    if split=='test':assert C.load(C.CACHE/'correction_checks.json')['all_pass']
    G,R=C.M.load_sources();lookup=np.full(len(G['time']),-1,np.int64);lookup[R['gi']]=np.arange(len(R['gi']))
    rt=C.runtime();torch.set_num_threads(rt['threads'])
    cases=[c for c in C.CASES if ('__inner' in c)==(split=='held')]
    for case in cases:
        rd=C.rd(case);manifest=C.load(rd/'manifest.json');stats=C.load(rd/'stats.json');model=create_model('LSTM_F',rt,C.SEED).to('cuda')
        expected=C.load(rd/'complete.json')['best_sha256'];assert C.sha(rd/'best.pt')==expected
        model.load_state_dict(torch.load(rd/'best.pt',map_location='cuda',weights_only=True));model.eval()
        for rule in C.TRAIN_RULES if split=='held' else C.RULES:
            segments=D.build_segments(manifest[split],stats,rule,G,R);idx=[];out=[]
            with torch.no_grad():
                for start in range(0,len(segments),8):
                    ss=segments[start:start+8];b=batches(ss,'cuda',8)[0];pred=model(b['x'],b['lengths']).cpu().numpy()[...,0]*stats['ti_sd']+stats['ti_mu']
                    for j,s in enumerate(ss):
                        ii=lookup[s['gi']];keep=ii>=0;idx.extend(ii[keep]);out.extend(pred[j,:len(ii)][keep])
            ii=np.asarray(idx,np.int64);v=np.asarray(out,np.float64);order=np.argsort(ii);ii,v=ii[order],v[order]
            assert len(ii) and (np.diff(ii)>0).all() and np.isfinite(v).all()
            p=C.CACHE/'predictions'/f'{case}__{split}__{rule}.npz';p.parent.mkdir(exist_ok=True)
            np.savez(p,idx=ii,est=v,model_sha256=expected,source_hash=C.digest(C.sources()),manifest_hash=C.digest(manifest))
        print('Predicted '+case+' '+split,flush=True)
        del model;torch.cuda.empty_cache()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=('held','test'));a=p.parse_args();main(a.split)
