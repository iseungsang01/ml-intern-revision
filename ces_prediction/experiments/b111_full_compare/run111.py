"""Colab OOF inference -> eight post fits -> outer replay -> all-cell readout."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse,subprocess,sys,time,zipfile
import numpy as np
from common111 import *
W=Path(__file__).resolve().parent

def path(task,rule):return W/'predictions'/(task+'__'+rule+'.npz')
def task_data(task):
    m=read(W/'manifest.json');t=m['tasks'][task];d=npz(W/'grid.npz');p=npz(W/(task+'.npz'));return m,t,d,p
def infer(task):
    import torch
    from portable108 import TemperatureLSTM
    m,t,d,p=task_data(task);out=W/'inference'/task;out.mkdir(parents=True,exist_ok=False)
    assert sha(W/(task+'.pt'))==t['weights_sha256']
    torch.set_num_threads(1);torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
    model=TemperatureLSTM().cuda();model.load_state_dict(torch.load(W/(task+'.pt'),map_location='cuda',weights_only=True));model.eval()
    with torch.no_grad():assert np.allclose(model(torch.from_numpy(p['ref_x']).cuda()).cpu().numpy(),p['ref_prediction'],atol=1e-4,rtol=1e-4)
    gi=p['gi'];lengths=p['lengths'];starts=np.r_[0,np.cumsum(lengths)];assert starts[-1]==len(gi)
    rr=d['row'][gi];keep=rr>=0;idx=rr[keep];order=np.argsort(idx);idx=idx[order];assert (np.diff(idx)>0).all()
    files={}
    for rule in RULES if t['role']=='outer' else TRAIN_RULES:
        xx=inputs(d,t['stats'],gi,rule);values=[]
        with torch.no_grad():
            for a in range(0,len(lengths),8):
                lens=lengths[a:a+8];x=np.zeros((len(lens),int(lens.max()),22),np.float32)
                for j,le in enumerate(lens):x[j,:le]=xx[starts[a+j]:starts[a+j+1]]
                y=model(torch.from_numpy(x).cuda()).cpu().numpy()[...,0]
                # Preserve original B106 float32 physical denormalization.
                y=y*t['stats']['ti_sd']+t['stats']['ti_mu']
                values.extend(y[j,:le] for j,le in enumerate(lens))
        est=np.concatenate(values)[keep][order].astype(np.float64);assert np.isfinite(est).all()
        dest=path(task,rule);dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,idx=idx,est=est)
        files[dest.name]=sha(dest);write(out/'progress.json',{'task':task,'rule':rule,'completed_rules':len(files),'updated':time.time()})
    write(out/'complete.json',{'task':task,'weights_sha256':t['weights_sha256'],'files':files,'prefix_pass':True})

def tree_values(model):
    out={'baseline':np.asarray(model._baseline_prediction).reshape(-1)};assert out['baseline'].shape==(1,)
    for k,iteration in enumerate(model._predictors):
        assert len(iteration)==1;nodes=iteration[0].nodes;assert not nodes['is_categorical'].any()
        for field in ('left','right','feature_idx','num_threshold','value','is_leaf','missing_go_to_left'):out[f'{k}_{field}']=nodes[field]
    out['trees']=np.array(len(model._predictors),np.int64);return out

def tree_predict(z,X):
    out=np.full(len(X),float(z['baseline'][0]));rows=np.arange(len(X))
    for tree in range(int(z['trees'])):
        pre=str(tree)+'_';node=np.zeros(len(X),np.int64)
        while True:
            active=~z[pre+'is_leaf'][node].astype(bool)
            if not active.any():break
            ii=rows[active];nn=node[active];xx=X[ii,z[pre+'feature_idx'][nn].astype(int)];left=np.where(np.isnan(xx),z[pre+'missing_go_to_left'][nn].astype(bool),xx<=z[pre+'num_threshold'][nn])
            node[ii]=np.where(left,z[pre+'left'][nn],z[pre+'right'][nn])
        out+=z[pre+'value'][node]
    return out

def make_features(rows,fast,rule,idx,est):return features(fast,rows['tstep'],rows[rule+'_H'],rows[rule+'_G'],rows[rule+'_age'],idx,est)

def post(unit):
    import joblib,sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    m=read(W/'manifest.json');u=m['units'][unit];outer=m['tasks'][u['outer']];rows=npz(W/'rows.npz');fast=npz(W/(unit+'_fast.npz'))['fast'];dest=W/'post'/unit;dest.mkdir(parents=True,exist_ok=False)
    pieces={};coverage={};bindings={}
    for rule in TRAIN_RULES:
        X=[];Y=[];indices=[]
        for task in u['inner']:
            t=m['tasks'][task];f=t['files'];assert not set(f['held'])&set(f['train']+f['val']+f['test']) and set(f['held'])<=set(outer['files']['train'])
            p=path(task,rule);z=npz(p);idx=z['idx'];assert np.isin(rows['discharge'][idx],f['held']).all()
            xx=make_features(rows,fast,rule,idx,z['est']);yy=rows['truth'][idx]-z['est'];ok=np.isfinite(xx).all(1)&np.isfinite(yy)
            X.append(xx[ok]);Y.append(yy[ok]);indices.append(idx[ok]);bindings[p.name]=sha(p)
        ix=np.concatenate(indices);assert len(ix)==len(np.unique(ix));xx=np.concatenate(X);yy=np.concatenate(Y);assert len(yy)>=30
        pieces[rule]=(xx,yy);coverage[rule]={'rows':len(yy),'files':len(np.unique(rows['discharge'][ix])),'X_sha256':hashlib.sha256(xx.tobytes()).hexdigest(),'y_sha256':hashlib.sha256(yy.tobytes()).hexdigest()}
    models={}
    for label,rules in [('frozen_D10',('D10',)),('mixed_post',TRAIN_RULES)]:
        X=np.concatenate([pieces[r][0] for r in rules]);y=np.concatenate([pieces[r][1] for r in rules]);weights=np.concatenate([np.full(len(pieces[r][1]),len(y)/(len(rules)*len(pieces[r][1]))) for r in rules])
        assert np.isclose(weights.sum(),len(y));model=HistGradientBoostingRegressor(random_state=20261116,early_stopping=False,max_iter=100).fit(X,y,sample_weight=weights)
        portable=tree_values(model);assert np.allclose(model.predict(X[:512]),tree_predict(portable,X[:512]),rtol=1e-12,atol=1e-9)
        joblib.dump(model,dest/(label+'.joblib'));np.savez_compressed(dest/(label+'_trees.npz'),**portable)
        models[label]={'params':model.get_params(),'rows':len(y),'iterations':int(model.n_iter_),'joblib_sha256':sha(dest/(label+'.joblib')),'portable_sha256':sha(dest/(label+'_trees.npz')),'portable_equal':True}
        print('B111 fitted '+unit+' '+label+' n='+str(len(y)),flush=True)
    write(dest/'complete.json',{'unit':unit,'coverage':coverage,'OOF_sources':bindings,'models':models,'sklearn':sklearn.__version__,'TEST_fit_rows':0,'VAL_fit_rows':0})

def summarize():
    import joblib
    from stats111 import compute
    m=read(W/'manifest.json');rows=npz(W/'rows.npz');cells={};(W/'estimates').mkdir()
    for unit,u in m['units'].items():
        fast=npz(W/(unit+'_fast.npz'))['fast'];mods={n:joblib.load(W/'post'/unit/(n+'.joblib')) for n in ('frozen_D10','mixed_post')};trees={n:npz(W/'post'/unit/(n+'_trees.npz')) for n in mods}
        for rule in RULES:
            ref=npz(W/(unit+'_'+rule+'_reference.npz'));p=npz(path(u['outer'],rule));idx=ref['idx'];j=np.searchsorted(p['idx'],idx)
            assert (j<len(p['idx'])).all() and np.array_equal(p['idx'][j],idx)
            raw=p['est'][j];X=make_features(rows,fast,rule,idx,raw);ok=np.isfinite(X).all(1);arr={**ref,'LSTM':raw,'fallback':~ok}
            for name,mod in mods.items():
                pred=mod.predict(X[ok]);assert np.allclose(pred,tree_predict(trees[name],X[ok]),rtol=1e-12,atol=1e-9)
                est=raw.copy();est[ok]+=pred;arr['LSTM_'+name]=est
            assert all(np.isfinite(v).all() for v in arr.values());out=W/'estimates'/(unit+'_'+rule+'.npz');np.savez_compressed(out,**arr)
            key=u['name']+'/'+rule+'/F';cells[key]=compute(arr,key)
            write(W/'progress.json',{'state':'readout','completed_cells':len(cells),'total_cells':32,'updated':time.time()})
    report={'registration':'PREREGISTRATION_B111.md','inherited_scientific_protocol':'PREREGISTRATION_B106.md','sources':m['sources'],'target':'current_report_F','exploratory':True,'units':UNITS,'seeds':[1701],'draws':20000,'bootstrap_seed':20261106,'cells':cells,'new_LSTM_fits':0,'post_fits':8,'reference_sha256':m['reference_sha256'],'limits':['One unit per design and one initialization; no unused campaign.','TRAIN-file OOF, not prospective temporal inner evaluation.','F/D10 LSTM versus mixed-rule F/L5 TCN is not an architecture-only comparison.','Disclosed B107/B109 mixed-runtime continuations; no bitwise historical reproduction.','Pointwise fixed-fit intervals omit refit,initialization and selection uncertainty.','Both post policies reported without TEST selection; K10 absent from fitting.']}
    report['statements']={'S1_frozen_LSTM_post_over_GP_all_fresh':all(cells[u+'/D10/F']['pairs']['LSTM_frozen_D10_over_GP_fast_boost']['cluster']['ci95'][0]>0 for u in UNITS),'S2_mixed_post_over_GP_all_scenarios':all(c['pairs']['LSTM_mixed_post_over_GP_fast_boost']['cluster']['ci95'][0]>0 for c in cells.values()),'S3_GP_fast_increment_all_fresh':all(cells[u+'/D10/F']['pairs']['GP_fast_boost_over_GP_history_boost']['cluster']['ci95'][0]>0 for u in UNITS)}
    write(W/'readout.json',report)

def execute_phase(phase,tags,limit):
    import torch
    pending=list(tags);active=[];done=[];failure=None
    logs=W/'processes';logs.mkdir(exist_ok=True)
    while active or pending:
        for e in list(active):
            timed=time.time()-e['started']>43200
            if timed and e['p'].poll() is None:e['p'].kill()
            if e['p'].poll() is None:continue
            code=e['p'].wait();e['log'].close();write(logs/(e['tag']+'_'+phase+'_exit.json'),{'returncode':code,'timeout':timed,'elapsed':time.time()-e['started']})
            if code!=0 or timed:failure=failure or phase+'/'+e['tag']
            else:
                q=W/('post' if phase=='post' else 'inference')/e['tag']/'complete.json';assert q.exists();record=read(q)
                if phase=='post':
                    assert set(record['models'])=={'frozen_D10','mixed_post'} and record['TEST_fit_rows']==record['VAL_fit_rows']==0
                    for label,v in record['models'].items():
                        assert v['iterations']==100 and sha(q.parent/(label+'.joblib'))==v['joblib_sha256'] and sha(q.parent/(label+'_trees.npz'))==v['portable_sha256']
                else:
                    assert record['prefix_pass'] and len(record['files'])==(8 if phase=='outer' else 7)
                    for name,h in record['files'].items():assert sha(W/'predictions'/name)==h
                done.append(e['tag'])
            active.remove(e)
        if failure:pending=[]
        while pending and len(active)<limit and not failure:
            if phase!='post' and torch.cuda.mem_get_info()[0]<1630*1024**2:break
            tag=pending.pop(0);env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1' if phase=='post' else os.environ.get('CUDA_VISIBLE_DEVICES','0'),'CES_SEQ_DEVICE':'cpu' if phase=='post' else 'cuda','CES_B111_PHASE':phase,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
            cmd=[sys.executable,'-u',str(W/'run111.py'),'--post' if phase=='post' else '--infer',tag];log=(logs/(tag+'_'+phase+'.log')).open('wb');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            started=time.time();write(logs/(tag+'_'+phase+'_start.json'),{'pid':p.pid,'started':started,'phase':phase,'tag':tag})
            active.append({'tag':tag,'p':p,'log':log,'started':started})
        write(W/'progress.json',{'state':phase,'completed':len(done),'total':len(tags),'pending':len(pending),'active':[{'tag':e['tag'],'pid':e['p'].pid} for e in active],'failure':failure,'updated':time.time()})
        if active or pending:time.sleep(1)
    assert not failure,'Scientific failure; no retry: '+str(failure)
    write(W/(phase+'_complete.json'),{'tags':done,'all_pass':True,'completed':time.time()})

def run():
    import torch,sklearn
    assert not (W/'started.json').exists(),'No scientific restart';write(W/'started.json',{'time':time.time(),'torch':torch.__version__,'sklearn':sklearn.__version__})
    m=read(W/'manifest.json');assert m['registration']=='PREREGISTRATION_B111.md'
    for n,h in m['files'].items():assert sha(W/n)==h
    pins={'CES_DROP_STUCK_TARGETS':'1','CES_TI_SPIKE_CUT_EV':'3000','CES_PER_SHOT_NORM':'0','CES_SEQ_NORM_EXPANDING':'0','CES_SEQ_ZERO_MC':'1','CES_SEQ_ZERO_VT_HISTORY':'1','CES_SEQ_ZERO_VT':'1','CES_SEQ_DEVICE':'cuda','CES_B111_TARGET':'F','CES_B111_RULE':'REGISTERED_REPLAY'}
    assert all(os.environ.get(k)==v for k,v in pins.items());assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
    for unit,u in m['units'].items():
        outer=m['tasks'][u['outer']]['files'];held=[]
        for task in u['inner']:
            f=m['tasks'][task]['files'];assert not set(f['held'])&set(f['train']+f['val']+f['test']);held+=f['held']
        assert len(held)==len(set(held)) and set(held)==set(outer['train'])
    execute_phase('held',[k for k,t in m['tasks'].items() if t['role']!='outer'],4)
    execute_phase('post',list(m['units']),2)
    execute_phase('outer',[u['outer'] for u in m['units'].values()],4)
    summarize()
    files=[W/'readout.json',W/'started.json',W/'held_complete.json',W/'post_complete.json',W/'outer_complete.json']
    for folder in ('predictions','post','inference','estimates','processes'):files.extend(p for p in (W/folder).rglob('*') if p.is_file())
    write(W/'result_hashes.json',{p.relative_to(W).as_posix():sha(p) for p in files})
    with zipfile.ZipFile(W/'b111_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in [*files,W/'result_hashes.json']:z.write(p,p.relative_to(W))
    write(W/'progress.json',{'state':'complete','completed_cells':32,'total_cells':32,'post_fits':8,'updated':time.time()});print('B111 all OOF/post/scenario stages complete.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--infer');p.add_argument('--post');a=p.parse_args()
    if a.infer:infer(a.infer)
    elif a.post:post(a.post)
    else:run()
