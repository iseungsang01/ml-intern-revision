"""Real-data original-branch, OOF exclusion, memory and fit arithmetic checks."""
import argparse,subprocess,math
import numpy as np,torch
import common106 as C
from model106 import create_model,SeqCESLSTMv2
from prepare92 import batches
def inputs():
    C.guard();C.inputs_guard();torch.set_num_threads(1);rows=[];example=None
    for u in C.UNITS:
        parent=C.prepared(u+'__outer')['identity']['manifest'];held=[]
        for role in ('outer','inner0','inner1','inner2','inner3'):
            case=u+'__'+role;z=C.prepared(case);m=z['identity']['manifest'];s=z['identity']['stats']
            assert s['train_files']==sorted(m['train']) and not set(m['train'])&set(m['val']+m['test']+m['held'])
            if role!='outer':held+=m['held']
            for split in ('train','val'):
                assert set(z[split])=={'D10'}
                for seg in z[split]['D10']:
                    assert np.isfinite(seg['x']).all() and not seg['mask'][:,1].any()
                    assert seg['fidx'] in m[split] and np.all(seg['gi'][1:]>seg['gi'][:-1])
            rows.append(case)
            if example is None:
                a=next(s for s in z['val']['D10'] if s['mask'][:,0].any());n=min(len(a['x']),max(64,int(np.flatnonzero(a['mask'][:,0])[0])+1))
                example=[{**a,**{k:a[k][:n] for k in ('x','y','mask','gi')}}]
        assert len(held)==len(set(held)) and set(held)==set(parent['train'])
    rt={'dense':True};model=create_model('LSTM_F',rt,C.SEED);torch.manual_seed(C.SEED);original=SeqCESLSTMv2(n_in=22,n_fast=15,n_extra_vt=0,n_extra_ti=0)
    b=batches(example,'cpu')[0];model.eval();original.eval()
    with torch.no_grad():
        f=model(b['x'],b['lengths']);ref=original(b['x'][...,:22],b['lengths'])[...,:1]
        assert torch.allclose(f[...,:1],ref,atol=1e-5,rtol=0)
        state=None;out=[]
        for x in b['x'][0]:p,state=model.forward_step(x.reshape(1,1,35),state);out.append(p)
        assert torch.allclose(f,torch.stack(out,1),atol=1e-5,rtol=0)
        k=len(b['x'][0])//2;x=b['x'].clone();x[:,k:]+=1
        assert torch.allclose(f[:,:k],model(x,b['lengths'])[:,:k],atol=1e-5,rtol=0)
    C.save(C.CACHE/'input_checks.json',{'all_pass':True,'sources':C.sources(),'cases':rows,'original_SeqCESLSTMv2_Ti_branch_equal':True,'stream_causal_prefix':True,'OOF_no_held_file_in_train_or_val':True,'TEST_read':False})
    print('B106 real input/branch/OOF checks passed',flush=True)
def memory():
    from runtime92 import configure
    from train92 import vector_loss
    C.guard();rt=C.runtime();configure(rt);peaks=[]
    for u in C.UNITS:
        z=C.prepared(u+'__outer');ss=sorted(z['train']['D10'],key=lambda s:len(s['x']),reverse=True)[:16]
        model=create_model('LSTM_F',rt,C.SEED).to('cuda');model.train();torch.cuda.reset_peak_memory_stats();b=batches(ss,'cuda')[0]
        p=model(b['x'],b['lengths']);s=z['identity']['stats'];v=vector_loss(p,b['y'],b['mask'],torch.tensor(s['ti_mu'],device='cuda'),torch.tensor(s['ti_sd'],device='cuda'));v.backward();torch.cuda.synchronize()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());peaks.append(torch.cuda.max_memory_reserved())
        del model,b,p,v,z;torch.cuda.empty_cache()
    budget=max(1200,math.ceil((max(peaks)/1024**2+768)*1.25));free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    workers=min(4,free//budget);assert workers>=1
    C.save(C.CACHE/'memory_checks.json',{'all_pass':True,'sources':C.sources(),'workers':workers,'per_worker_MiB':budget,'peaks_bytes':peaks,'free_MiB':free,'no_optimizer_updates':True})
    print('B106 reserved GPU slots '+str(workers),flush=True)
def fits():
    C.guard();result={case:C.verify_run(case) for case in C.CASES}
    C.save(C.CACHE/'fit_checks.json',{'all_pass':True,'sources':C.sources(),'fits':result});print('B106 all20 fit/exit/VAL arithmetic checks passed',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=('inputs','memory','fits'));a=p.parse_args();globals()[a.part]()
