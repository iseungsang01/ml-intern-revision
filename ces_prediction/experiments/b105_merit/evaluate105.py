"""Prediction only after normal fit audit; candidate TEST requires the fixed VAL gate."""
import argparse
import numpy as np,torch
import common105 as C
import common92 as M
import data88 as D
from model105 import create_model
from model92 import create_model as old_model

def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def predictions(split):
    assert C.load(C.CACHE/'fit_checks.json')['all_pass'];C.guard();C.inputs_guard();rt=C.runtime();torch.set_num_threads(rt['threads'])
    if split=='test':assert C.load(C.CACHE/'gate.json')['all_pass']
    G,R=M.load_sources();lookup=np.full(len(G['time']),-1,np.int64);lookup[R['gi']]=np.arange(len(R['gi']))
    for unit in C.UNITS:
        prep=C.prepared(unit);stats=prep['identity']['stats'];manifest=prep['identity']['manifest']
        for label in ('TCN35','TCN22'):
            path=C.rd(unit)/'best.pt' if label=='TCN35' else C.DATA/'.b94/confirm'/f'{unit}__1701__TX/best.pt'
            expected=C.load(C.rd(unit)/'complete.json')['best_sha256'] if label=='TCN35' else C.prov()['prepared'][unit]['original_best_sha256']
            assert C.sha(path)==expected
            model=(create_model('A35',rt,C.SEED) if label=='TCN35' else old_model('TX',rt,C.SEED)).to('cuda')
            model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));model.eval()
            for rule in C.TRAIN_RULES if split=='val' else C.RULES:
                segs=prep['val'][rule] if split=='val' else D.build_segments(manifest['test'],stats,rule,G,R)
                ii=[];pp=[]
                with torch.no_grad():
                    for s in segs:
                        x=torch.from_numpy(s['x']).unsqueeze(0).to('cuda');p=model(x).cpu().numpy()[0]*stats['ti_sd']+stats['ti_mu']
                        ix=lookup[s['gi']];keep=ix>=0;ii.extend(ix[keep]);pp.extend(p[keep])
                idx=np.asarray(ii,np.int64);est=np.asarray(pp,np.float32);order=np.argsort(idx);idx,est=idx[order],est[order]
                assert len(idx) and (np.diff(idx)>0).all() and np.isfinite(est).all()
                out=C.CACHE/'predictions'/f'{unit}__{label}__{split}__{rule}.npz';out.parent.mkdir(exist_ok=True)
                np.savez(out,idx=idx,est=est,model_sha256=expected,sources_sha256=C.digest(C.sources()))
                print('Predicted '+unit+' '+label+' '+split+' '+rule,flush=True)
            del model;torch.cuda.empty_cache()

def aligned(p,idx,head=None):
    z=npz(p);pos=np.searchsorted(z['idx'],idx);assert (pos<len(z['idx'])).all() and np.array_equal(z['idx'][pos],idx)
    est=z['est'][pos];return est if head is None else est[:,head]

def gate():
    G,R=M.load_sources();per={};items={}
    for unit in C.UNITS:
        vals={x:[] for x in ('TCN35','TCN22','specialized','mixture')};sd=C.prepared(unit)['identity']['stats']['ti_sd']
        for rule in C.TRAIN_RULES:
            for h,t in enumerate(('F','L5')):
                p=C.CACHE/'trees'/unit/f'{rule}__{t}__val.npz';ref=npz(p);idx=ref['idx'];truth=R['tg_'+t][idx]
                assert np.isfinite(truth).all() and len(idx)>=30 and len(np.unique(R['clus'][idx]))>=2
                estimates={'specialized':ref['est'],'mixture':aligned(C.CACHE/'trees'/unit/f'{rule}__{t}__val__mixture.npz',idx)}
                for name in ('TCN35','TCN22'):estimates[name]=aligned(C.CACHE/'predictions'/f'{unit}__{name}__val__{rule}.npz',idx,h)
                cell={name:float(np.mean((v.astype(np.float64)-truth)**2)/sd**2) for name,v in estimates.items()}
                assert all(np.isfinite(x) for x in cell.values()) and cell['specialized']>0
                items[unit+'/'+rule+'/'+t]={'rows':len(idx),'normalized_MSE':cell}
                for name in vals:vals[name].append(cell[name])
        means={k:float(np.mean(v)) for k,v in vals.items()};ratio=means['TCN35']/means['specialized']
        per[unit]={'mean_normalized_MSE':means,'candidate_to_specialized_ratio':ratio,'pass':bool(ratio<=1.05)}
    z={'all_pass':all(v['pass'] for v in per.values()),'rule':'ratio of means across7rules x2heads <=1.05 in EVERY fixed unit',
       'not_statistical_noninferiority':True,'VAL_selected_checkpoint_optimism':True,'units':per,'cells':items,'sources':C.sources()}
    C.save(C.CACHE/'gate.json',z);print('B105 VAL allocation gate '+str(z['all_pass']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=('val','test','gate'));a=p.parse_args()
    if a.part=='gate':gate()
    else:predictions(a.part)
