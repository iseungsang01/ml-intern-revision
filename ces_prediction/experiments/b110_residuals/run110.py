"""T4 inference, followed by cloud-only frozen validation residual readout."""
import os,time,zipfile
import numpy as np,torch
from common110 import *
from portable108 import TemperatureLSTM
WORK=Path(__file__).resolve().parent

def interval(v):
    if not np.isfinite(v).all():return {'available':False,'reason':'nonfinite draw; no redraw'}
    lo,hi=np.quantile(v,[.025,.975]);return {'available':True,'ci95':[float(lo),float(hi)],'excludes_zero':bool(lo>0 or hi<0)}

def summarize(d,unit):
    est=np.column_stack([d[n] for n in MODELS]).astype(np.float64);err=est-d['truth'][:,None];result={}
    for group,mask in zip(GROUPS,group_masks(d)):
        e=err[mask];n=len(e);cell={'rows':n,'models':{},'contrasts':{}}
        if not n:cell.update(available=False,reason='empty group');result[group]=cell;continue
        mse=(e*e).mean(0);bias=e.mean(0);center=np.mean((e-bias)**2,axis=0)
        assert np.allclose(mse,bias*bias+center,rtol=1e-12,atol=1e-7)
        for j,name in enumerate(MODELS):cell['models'][name]={'MSE':float(mse[j]),'bias':float(bias[j]),'centered_MSE':float(center[j])}
        for a,b in PAIRS:
            j,k=MODELS.index(a),MODELS.index(b);cell['contrasts'][a+'/'+b]={'point':float(1-mse[j]/mse[k]) if mse[k]>0 else None}
        for form in ('cluster','discharge'):
            labels=d[form][mask];_,inv=np.unique(labels,return_inverse=True);ng=inv.max()+1;cell[form+'_count']=int(ng)
            if n<2 or ng<2:
                for z in [*cell['models'].values(),*cell['contrasts'].values()]:z[form]={'available':False,'reason':'fewer than two rows or resampling units'}
                continue
            counts=np.bincount(inv);s1=np.stack([np.bincount(inv,weights=e[:,j],minlength=ng) for j in range(len(MODELS))],axis=1);s2=np.stack([np.bincount(inv,weights=e[:,j]**2,minlength=ng) for j in range(len(MODELS))],axis=1)
            rng=rng_for(unit,group,form);bs=[];bm=[]
            for offset in range(0,DRAW,500):
                w=rng.multinomial(ng,np.full(ng,1/ng),min(500,DRAW-offset));den=w@counts
                bs.append((w@s1)/den[:,None]);bm.append((w@s2)/den[:,None])
            bb=np.concatenate(bs);mm=np.concatenate(bm);cc=mm-bb*bb
            for j,name in enumerate(MODELS):cell['models'][name][form]={'MSE':interval(mm[:,j]),'bias':interval(bb[:,j]),'centered_MSE':interval(cc[:,j])}
            for a,b in PAIRS:
                j,k=MODELS.index(a),MODELS.index(b)
                cell['contrasts'][a+'/'+b][form]=interval(1-mm[:,j]/mm[:,k]) if (mm[:,k]>0).all() else {'available':False,'reason':'zero reference MSE; no redraw'}
        result[group]=cell
    return result

def main():
    assert not (WORK/'started.json').exists(),'No scientific retry';write(WORK/'started.json',{'time':time.time()})
    m=read(WORK/'manifest.json');assert m['registration']=='PREREGISTRATION_B110.md'
    for name,h in m['files'].items():assert sha(WORK/name)==h
    pins={'CES_DROP_STUCK_TARGETS':'1','CES_TI_SPIKE_CUT_EV':'3000','CES_PER_SHOT_NORM':'0','CES_SEQ_NORM_EXPANDING':'0','CES_SEQ_ZERO_MC':'1','CES_SEQ_ZERO_VT_HISTORY':'1','CES_SEQ_ZERO_VT':'1','CES_SEQ_DEVICE':'cuda','CES_B110_TARGET':'F','CES_B110_RULE':'D10'}
    assert all(os.environ.get(k)==v for k,v in pins.items())
    torch.set_num_threads(2);torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
    report={'registration':m['registration'],'sources':m['sources'],'exploratory':True,'VAL_selection_optimism':True,'TEST_scored':False,'new_fits':0,'pointwise_not_familywise':True,'draws':DRAW,'models':MODELS,'groups':GROUPS,'units':{},'pins':pins,'gpu':torch.cuda.get_device_name(0)}
    for tag,t in m['tasks'].items():
        d=npz(WORK/(tag+'.npz'));model=TemperatureLSTM().cuda();model.load_state_dict(torch.load(WORK/(tag+'.pt'),map_location='cuda',weights_only=True));model.eval()
        with torch.no_grad():
            ref=model(torch.from_numpy(d['ref_x']).cuda()).cpu().numpy();assert np.allclose(ref,d['ref_prediction'],atol=1e-4,rtol=1e-4)
            lengths=d['lengths'];starts=np.r_[0,np.cumsum(lengths)];out=[]
            for i in range(0,len(lengths),8):
                lens=lengths[i:i+8];x=np.zeros((len(lens),int(lens.max()),22),np.float32)
                for j,le in enumerate(lens):x[j,:le]=d['x'][starts[i+j]:starts[i+j+1]]
                p=model(torch.from_numpy(x).cuda()).cpu().numpy()[...,0]
                out.extend(p[j,:le] for j,le in enumerate(lens))
        pred=np.concatenate(out)[d['positions']].astype(np.float64)*float(d['sd'])+float(d['mu']);assert pred.shape==d['truth'].shape and np.isfinite(pred).all()
        row={k:d[k] for k in ('idx','truth','cluster','discharge','age','history_change','change_q75',*MODELS[1:])};row['LSTM']=pred
        np.savez_compressed(WORK/(tag+'_rows.npz'),**row)
        report['units'][t['unit']]={'tag':tag,'prefix_pass':True,'input_sha256':sha(WORK/(tag+'.npz')),'weights_sha256':sha(WORK/(tag+'.pt')),'groups':summarize(row,t['unit'])}
        write(WORK/'progress.json',{'completed':len(report['units']),'total':4,'state':'running'});print('B110 completed '+tag,flush=True)
        del model;torch.cuda.empty_cache()
    write(WORK/'results.json',report)
    files=[WORK/'results.json',*[WORK/(tag+'_rows.npz') for tag in m['tasks']]]
    write(WORK/'result_hashes.json',{p.name:sha(p) for p in files})
    with zipfile.ZipFile(WORK/'b110_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in [*files,WORK/'result_hashes.json']:z.write(p,p.name)
    write(WORK/'progress.json',{'completed':4,'total':4,'state':'complete'})

if __name__=='__main__':main()
