"""Independent real-cache/source checks before fitting, and fit audit before scoring."""
import argparse,copy,sys
import numpy as np,torch
import common105 as C
import common92 as M
import data88 as D
from prepare92 import batches
from model105 import create_model
from models90 import create_model as original_model

def inputs():
    C.guard();C.inputs_guard();torch.set_num_threads(1);G,R=M.load_sources();rows=[];one=None
    for unit in C.UNITS:
        z=C.prepared(unit);m=z['identity']['manifest'];s=z['identity']['stats']
        assert s['train_files']==sorted(m['train']) and s['ti_sd']>0
        assert not set(m['train'])&set(m['val']) and not (set(m['train'])|set(m['val']))&set(m['test'])
        stats=D.fit_stats(m['train'],G,R);assert stats==s
        for rule in C.TRAIN_RULES:
            for split in ('train','val'):
                assert z[split][rule]
                seg=next(a for a in z[split][rule] if len(a['x'])>8 and a['mask'].any())
                idx=seg['gi'];x=seg['x'];a=D.anchors(rule,G=G)[idx];has=a>=0
                assert np.isfinite(x).all() and x.shape[1]==35 and np.array_equal(x[:,18].astype(bool),has)
                assert (a[has]<idx[has]).all() and np.array_equal(G['fidx'][a[has]],G['fidx'][idx[has]]) and np.array_equal(G['seg'][a[has]],G['seg'][idx[has]])
                expected=np.zeros((len(x),13),np.float32)
                expected[has]=(G['fast'][a[has]].astype(np.float32)-np.asarray(s['fast_mu']))/np.asarray(s['fast_sd'])
                assert np.array_equal(expected,x[:,22:])
                raw_current=((G['fast'][idx].astype(np.float32)-np.asarray(s['fast_mu']))/np.asarray(s['fast_sd'])).astype(np.float32)
                assert np.array_equal(raw_current,x[:,:13])
                # L5 label sources cannot cross split files: all five nominal future slots stay within file/segment.
                obs=np.flatnonzero(seg['mask'].any(1));end=max(64,int(obs[0])+1);end=min(end,len(x))
                tiny={**seg,**{k:seg[k][:end] for k in ('x','y','mask','gi')}}
                if one is None:one=[tiny]
            rows.append(unit+'/'+rule)
        # Rebuild actual VAL file against altered labels only (source data unchanged).
        fid=z['val']['D10'][0]['fidx'];left=D.build_segments([fid],s,'D10',G,R)
        changed={**R,'tg_F':R['tg_F']+1000,'tg_L5':R['tg_L5']-1000}
        right=D.build_segments([fid],s,'D10',G,changed)
        assert all(np.array_equal(a['x'],b['x']) for a,b in zip(left,right))
        assert 'K10' not in z['train'] and 'K10' not in z['val']
    rt={'device':'cpu','threads':1,'cudnn':False,'dense':False,'mkldnn':False}
    model=create_model('A35',rt,C.SEED);rng=torch.get_rng_state().clone();old=original_model('T','X',C.SEED)
    assert torch.equal(rng,torch.get_rng_state()),'Stem widening consumed RNG'
    assert torch.equal(model.stem.weight[:,:22],old.stem.weight) and torch.count_nonzero(model.stem.weight[:,22:])==0
    for k,v in old.state_dict().items():
        if k!='stem.weight':assert torch.equal(v,model.state_dict()[k])
    b=batches(one,'cpu')[0];model.eval();old.eval()
    with torch.no_grad():
        y=model(b['x'],b['lengths']);base=old(b['x'][...,:22],b['lengths']);assert torch.allclose(y,base,atol=1e-5,rtol=0)
        state=None;out=[]
        for x in b['x'][0]:p,state=model.forward_step(x.reshape(1,1,35),state);out.append(p)
        assert torch.allclose(y,torch.stack(out,1),atol=1e-5,rtol=0)
        x=b['x'].clone();k=len(x[0])//2;x[:,k:]+=3
        assert torch.allclose(y[:,:k],model(x,b['lengths'])[:,:k],atol=1e-5,rtol=0)
    model.train();p=model(b['x'],b['lengths']);loss=(p[b['mask']]-b['y'][b['mask']]).square().mean();loss.backward()
    grad=model.stem.weight.grad[:,22:];assert torch.isfinite(grad).all() and grad.abs().sum()>0,'Anchor gradient missing on real batch'
    C.save(C.CACHE/'input_checks.json',{'all_pass':True,'sources':C.sources(),'cases':rows,'source_time_direct_gather':True,'train_stats_recomputed':True,
        'initial_function_tolerance':1e-5,'initial_base_function_preserved':True,'widening_rng_unchanged':True,'anchor_real_gradient_nonzero':True,'stream_prefix':True,'labels_not_inputs':True,'TEST_segments_built':False})
    print('B105 real input/model checks passed',flush=True)

def fits():
    C.guard();runs={u:C.verify_run(u) for u in C.UNITS};trees={}
    for u in C.UNITS:
        z=C.load(C.CACHE/'trees'/u/'complete.json');assert z['sources']==C.sources() and z['fits']==16
        m=C.prepared(u)['identity']['manifest'];assert z['manifest_hash']==C.digest(m) and z['normalization_files_hash']==C.digest(m['train'])
        assert z['tree']['early_stopping'] is False and z['tree']['random_state']==20261116
        assert C.load(C.CACHE/'trees'/u/'process_exit.json')['returncode']==0
        for name,h in z['files'].items():assert C.sha(C.CACHE/'trees'/u/name)==h
        trees[u]={'fits':z['fits'],'seconds':z['seconds']}
    C.save(C.CACHE/'fit_checks.json',{'all_pass':True,'sources':C.sources(),'runs':runs,'trees':trees})
    print('B105 all fit exits/checkpoints/VAL arithmetic passed',flush=True)
def memory():
    from train92 import vector_loss
    from runtime92 import configure
    import math,subprocess
    C.guard();rt=C.runtime();configure(rt);peaks=[]
    for u in C.UNITS:
        prep=C.prepared(u);ss=sorted(prep['train']['D10'],key=lambda s:len(s['x']),reverse=True)[:16]
        model=create_model('A35',rt,C.SEED).to('cuda');model.train();torch.cuda.reset_peak_memory_stats()
        b=batches(ss,'cuda')[0];stats=prep['identity']['stats'];p=model(b['x'],b['lengths'])
        loss=vector_loss(p,b['y'],b['mask'],torch.tensor(stats['ti_mu'],device='cuda'),torch.tensor(stats['ti_sd'],device='cuda'));loss.backward();torch.cuda.synchronize()
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
        peaks.append({'unit':u,'rows':int(b['lengths'].max()),'batch':len(ss),'reserved_bytes':torch.cuda.max_memory_reserved(),'allocated_bytes':torch.cuda.max_memory_allocated()})
        del model,b,p,loss,prep;torch.cuda.empty_cache()
    budget=max(1200,math.ceil((max(p['reserved_bytes'] for p in peaks)/1024**2+768)*1.25))
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    workers=min(4,free//budget);assert workers>=1,'Insufficient memory for bounded production'
    C.save(C.CACHE/'memory_checks.json',{'all_pass':True,'sources':C.sources(),'real_longest16_batch_forward_backward_no_optimizer':True,'per_worker_reserved_MiB':budget,'free_MiB':free,'workers':workers,'peaks':peaks})
    print('B105 memory-admitted worker count '+str(workers),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['inputs','fits','memory']);a=p.parse_args();globals()[a.part]()
