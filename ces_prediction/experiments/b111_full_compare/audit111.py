"""Independent local CPU reconstruction; no fitting or neural inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,zipfile,zlib
import numpy as np
from common111 import *
sys.path.insert(0,str(HERE.parent/'b106_lstm'));import common106 as C
sys.path.insert(0,str(HERE.parent/'b75_availability'));import stage_fits as SF

def predict_nodes(tree,X):
    result=np.repeat(float(tree['baseline'][0]),len(X))
    for k in range(int(tree['trees'])):
        prefix=str(k)+'_';queue=[(0,np.arange(len(X)))];value=np.empty(len(X))
        while queue:
            node,ids=queue.pop()
            if tree[prefix+'is_leaf'][node]:value[ids]=tree[prefix+'value'][node];continue
            v=X[ids,int(tree[prefix+'feature_idx'][node])];left=v<=tree[prefix+'num_threshold'][node]
            left[np.isnan(v)]=bool(tree[prefix+'missing_go_to_left'][node])
            if left.any():queue.append((int(tree[prefix+'left'][node]),ids[left]))
            if (~left).any():queue.append((int(tree[prefix+'right'][node]),ids[~left]))
        result+=value
    return result

def reference(p,idx,head=None):
    z=npz(p);lookup={int(r):j for j,r in enumerate(z['idx'])};pos=np.array([lookup[int(r)] for r in idx]);v=z['est'][pos];return v if head is None else v[:,head]

def stats_check(arr,cell,key):
    names=list(cell['models']);err=np.stack([arr[n]-arr['truth'] for n in names],axis=1);sq=err**2;mean=sq.sum(0)/len(err)
    assert cell['rows']==len(err)
    for j,n in enumerate(names):assert np.isclose(mean[j],cell['models'][n]['MSE'],rtol=1e-12,atol=1e-8) and np.isclose(np.sqrt(mean[j]),cell['models'][n]['RMSE'],rtol=1e-12)
    for form in ('cluster','discharge'):
        labels=np.unique(arr[form]);assert len(labels)==cell['clusters' if form=='cluster' else 'discharges'];inv=np.searchsorted(labels,arr[form]);ng=len(labels)
        sums=np.zeros((ng,len(names)));count=np.zeros(ng);np.add.at(sums,inv,sq);np.add.at(count,inv,1)
        rng=np.random.default_rng([20261106,zlib.crc32((key+'/'+form).encode())]);draws=[]
        for _ in range(40):
            w=rng.multinomial(ng,np.repeat(1/ng,ng),500);draws.append(np.einsum('bg,gc->bc',w,sums)/np.einsum('bg,g->b',w,count)[:,None])
        mm=np.concatenate(draws);ci=np.percentile(mm,[2.5,97.5],axis=0);assert np.isfinite(mm).all() and (mm>0).all()
        for j,n in enumerate(names):
            assert np.allclose(ci[:,j],cell['models'][n][form+'_MSE95'],rtol=1e-9,atol=1e-7)
            assert np.allclose(np.sqrt(ci[:,j]),cell['models'][n][form+'_RMSE95'],rtol=1e-9,atol=1e-7)
        assert set(cell['pairs'])=={a+'_over_'+b for a,b in PAIRS}
        for a,b in PAIRS:
            ia,ib=names.index(a),names.index(b);v=1-mm[:,ia]/mm[:,ib];q=np.percentile(v,[2.5,97.5]);p=cell['pairs'][a+'_over_'+b]
            assert np.isclose(1-mean[ia]/mean[ib],p['point'],atol=1e-12) and np.allclose(q,p[form]['ci95'],rtol=1e-9,atol=1e-10)
            assert p[form]['excludes_zero']==bool(q[0]>0 or q[1]<0)

def main():
    guard();C.guard();C.inputs_guard();state=read(ROOT/'data/.colab_connection/b111_remote.json');assert state['downloaded']
    archive=OUT/'b111_results.zip';assert sha(archive)==state['results_sha256'];dest=OUT/'retrieved';assert not dest.exists();dest.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n in z.namelist():assert (dest/n).resolve().is_relative_to(dest.resolve())
        z.extractall(dest)
    for name,h in read(dest/'result_hashes.json').items():assert sha(dest/name)==h
    m=read(OUT/'bundle/manifest.json');private=read(OUT/'private_map.json');report=read(dest/'readout.json');assert m['sources']==report['sources']
    for rel,h in report['reference_sha256'].items():assert sha(ROOT/rel)==h
    G,R=C.M.load_sources();ctx=SF.Ctx();codes=np.array(private['row_codes']);back=np.array(private['row_inverse']);assert np.array_equal(codes[back],np.arange(len(back)))
    rows=npz(OUT/'bundle/rows.npz');assert np.array_equal(rows['truth'],R['tg_F'][back],equal_nan=True)
    selected=read(ROOT/'data/colab_b109/selected_paths.json');prediction_count=0
    for phase,count in [('held',16),('post',4),('outer',4)]:
        complete=read(dest/(phase+'_complete.json'));assert complete['all_pass'] and len(complete['tags'])==count
        for tag in complete['tags']:
            e=read(dest/'processes'/(tag+'_'+phase+'_exit.json'));s=read(dest/'processes'/(tag+'_'+phase+'_start.json'));assert e['returncode']==0 and not e['timeout']
            if phase=='post':assert s['started']>=read(dest/'held_complete.json')['completed']
            if phase=='outer':assert s['started']>=read(dest/'post_complete.json')['completed']
    for task,t in m['tasks'].items():
        original=private['tasks'][task];case=original['case'];w=selected[case];assert sha(ROOT/w['path']/'best.pt')==t['weights_sha256']==w['best_sha256']
        task_meta=npz(OUT/'bundle'/(task+'.npz'));grid=npz(OUT/'bundle/grid.npz') if task==next(iter(m['tasks'])) else grid
        expected=np.sort(grid['row'][task_meta['gi']][grid['row'][task_meta['gi']]>=0]);done=read(dest/'inference'/task/'complete.json');assert done['prefix_pass'] and done['weights_sha256']==t['weights_sha256']
        assert len(done['files'])==(8 if t['role']=='outer' else 7)
        for rule in RULES if t['role']=='outer' else TRAIN_RULES:
            p=dest/'predictions'/(task+'__'+rule+'.npz');assert sha(p)==done['files'][p.name];z=npz(p);assert np.array_equal(z['idx'],expected) and np.isfinite(z['est']).all();prediction_count+=1
    for unit,u in m['units'].items():
        original=private['tasks'][u['outer']]['manifest'];fast,_,_=ctx.unit_arrays({'fit':original['train']})
        assert np.array_equal(npz(OUT/'bundle'/(unit+'_fast.npz'))['fast'],fast[back],equal_nan=True)
        done=read(dest/'post'/unit/'complete.json');assert done['TEST_fit_rows']==done['VAL_fit_rows']==0
        assert set(done['models'])=={'frozen_D10','mixed_post'}
        for rule in TRAIN_RULES:
            X=[];Y=[];IDs=[]
            for task in u['inner']:
                mm=private['tasks'][task]['manifest'];assert not set(mm['held'])&set(mm['train']+mm['val']+mm['test']) and set(mm['held'])<=set(original['train'])
                p=dest/'predictions'/(task+'__'+rule+'.npz');assert sha(p)==done['OOF_sources'][p.name];z=npz(p);ii=back[z['idx']];assert np.isin(R['fidx'][ii],mm['held']).all()
                I=npz(ROOT/'data/.b75/inputs'/(rule+'.npz'));c=z['est'];h=I['H'][ii];gp=I['G'][ii]
                xx=np.concatenate((fast[ii],R['tstep'][ii,None],I['age'][ii,None],h[:,None],(c-h)[:,None],(c-gp)[:,None],np.abs(c-h)[:,None],np.abs(c-gp)[:,None]),axis=1)
                yy=R['tg_F'][ii]-c;ok=np.isfinite(xx).all(1)&np.isfinite(yy);X.append(xx[ok]);Y.append(yy[ok]);IDs.append(ii[ok])
            xx=np.concatenate(X);yy=np.concatenate(Y);ids=np.concatenate(IDs);v=done['coverage'][rule]
            assert len(ids)==len(np.unique(ids))==v['rows'] and len(np.unique(R['fidx'][ids]))==v['files']
            assert hashlib.sha256(xx.tobytes()).hexdigest()==v['X_sha256'] and hashlib.sha256(yy.tobytes()).hexdigest()==v['y_sha256']
        trees={}
        for label,v in done['models'].items():
            assert v['iterations']==100 and v['params']['random_state']==20261116 and not v['params']['early_stopping'] and v['params']['max_iter']==100 and v['portable_equal']
            assert sha(dest/'post'/unit/(label+'.joblib'))==v['joblib_sha256'] and sha(dest/'post'/unit/(label+'_trees.npz'))==v['portable_sha256']
            trees[label]=npz(dest/'post'/unit/(label+'_trees.npz'))
            assert v['rows']==(done['coverage']['D10']['rows'] if label=='frozen_D10' else sum(vv['rows'] for vv in done['coverage'].values()))
        for rule in RULES:
            root=ROOT/'data'/('.b78' if rule=='K10' else '.b75');sets=npz(root/'sets.npz');mask=sets[('S_' if rule=='S' else 'common_')+'F']&np.isin(R['fidx'],original['test']);ii=np.flatnonzero(mask);ii=ii[np.argsort(codes[ii])];idx=codes[ii]
            arr=npz(dest/'estimates'/(unit+'_'+rule+'.npz'));assert np.array_equal(arr['idx'],idx) and np.array_equal(arr['truth'],R['tg_F'][ii])
            base=npz(dest/'predictions'/(u['outer']+'__'+rule+'.npz'));pos={int(r):j for j,r in enumerate(base['idx'])};raw=base['est'][np.array([pos[int(r)] for r in idx])];assert np.array_equal(arr['LSTM'],raw)
            I=npz(root/'inputs'/(rule+'.npz'));h=I['H'][ii];gp=I['G'][ii]
            X=np.concatenate((fast[ii],R['tstep'][ii,None],I['age'][ii,None],h[:,None],(raw-h)[:,None],(raw-gp)[:,None],np.abs(raw-h)[:,None],np.abs(raw-gp)[:,None]),axis=1);ok=np.isfinite(X).all(1);assert np.array_equal(arr['fallback'],~ok)
            for label,tree in trees.items():
                expected=raw.copy();expected[ok]+=predict_nodes(tree,X[ok]);assert np.allclose(expected,arr['LSTM_'+label],rtol=1e-12,atol=1e-8)
            assert np.array_equal(arr['GP_only'],gp) and np.array_equal(arr['hold'],h)
            for name,L in [('GP_history_boost','T0'),('GP_fast_boost','TX')]:assert np.array_equal(arr[name],reference(root/'fits'/f"{u['name']}__{L}__{rule}__F.npz",ii))
            for label in ('TCN22','TCN35'):assert np.array_equal(arr[label],reference(ROOT/'data/.b105/predictions'/f"{u['name']}__{label}__test__{rule}.npz",ii,0))
            for label,refname in [('GP_fair_mixture','mixture'),('GP_fair_specialized','specialized')]:
                if rule=='K10' and refname=='specialized':assert label not in arr;continue
                assert np.array_equal(arr[label],reference(ROOT/'data/.b105/trees'/u['name']/f'{rule}__F__test__{refname}.npz',ii))
            assert np.array_equal(arr['cluster'],rows['cluster'][idx]) and np.array_equal(arr['discharge'],rows['discharge'][idx])
            key=u['name']+'/'+rule+'/F';stats_check(arr,report['cells'][key],key)
            print('B111 audited '+key,flush=True)
    assert prediction_count==144 and len(report['cells'])==32
    c=report['cells'];s=report['statements'];assert s['S1_frozen_LSTM_post_over_GP_all_fresh']==all(c[u+'/D10/F']['pairs']['LSTM_frozen_D10_over_GP_fast_boost']['cluster']['ci95'][0]>0 for u in UNITS)
    assert s['S2_mixed_post_over_GP_all_scenarios']==all(v['pairs']['LSTM_mixed_post_over_GP_fast_boost']['cluster']['ci95'][0]>0 for v in c.values())
    assert s['S3_GP_fast_increment_all_fresh']==all(c[u+'/D10/F']['pairs']['GP_fast_boost_over_GP_history_boost']['cluster']['ci95'][0]>0 for u in UNITS)
    report['independent_audit']={'all_pass':True,'prediction_groups':prediction_count,'MSE_cells':32,'post_fits':8,'OOF_X_y_rebuilt':True,'post_tree_predictions_rebuilt':True,'all_points_intervals_recomputed':True,'archive_sha256':sha(archive),'local_fits':0,'local_GPU_work':0}
    write(OUT/'independent_audit.json',report['independent_audit']);write(ROOT/'docs/paper/b111_comparison.json',report)
    lines=['# B111: LSTM + OOF correction, all scenarios','','Exploratory fixed-fit comparison on reused TEST data; all four designs reported.','',*report['limits'],'','MSE eV^2, RMSE eV; brackets are pointwise95% shot-cluster intervals. Discharge intervals are in JSON.','','| Unit / rule | Model | Rows | MSE [95%] | RMSE |','|---|---|---:|---|---:|']
    for key,cell in c.items():
        for name,v in cell['models'].items():
            q=v['cluster_MSE95'];lines.append(f"| {key} | {name} | {cell['rows']} | {v['MSE']:.3f} [{q[0]:.3f}, {q[1]:.3f}] | {v['RMSE']:.3f} |")
    lines+=['','## Paired skill','','Skill=1-MSE(candidate)/MSE(reference). Positive means smaller candidate MSE.','','| Unit / rule | Contrast | Point [95% cluster] | Excludes zero |','|---|---|---|---|']
    for key,cell in c.items():
        for name,v in cell['pairs'].items():q=v['cluster']['ci95'];lines.append(f"| {key} | {name} | {v['point']:+.6f} [{q[0]:+.6f}, {q[1]:+.6f}] | {v['cluster']['excludes_zero']} |")
    lines+=['','Statements: '+json.dumps(s),'','An unfavorable conclusion could change after an independently specified matched comparison on unused observations. No architecture impossibility,new physics or algorithm originality is established.']
    (ROOT/'docs/paper/b111_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('B111 independent full audit passed.',flush=True)

if __name__=='__main__':main()
