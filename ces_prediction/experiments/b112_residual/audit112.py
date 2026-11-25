"""Independent result arithmetic and portable-model audit; no fitting/CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='2'

import hashlib
import json
import sys
import zipfile
import zlib
from pathlib import Path
import numpy as np
from scipy.special import erf
from common112 import *

sys.path.insert(0,str(HERE.parent/'b106_lstm'))
import common106 as C
sys.path.insert(0,str(HERE.parent/'b75_availability'))
import stage_fits as SF
sys.path.insert(0,str(HERE.parent/'b78_sparse'))
import common78


def digest_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def predict_nodes(tree,x):
    """Traverse each exported tree recursively, independently of sklearn."""
    result=np.repeat(float(tree['baseline'][0]),len(x))
    for k in range(int(tree['trees'])):
        prefix=str(k)+'_';queue=[(0,np.arange(len(x)))];value=np.empty(len(x))
        while queue:
            node,ids=queue.pop()
            if tree[prefix+'is_leaf'][node]:
                value[ids]=tree[prefix+'value'][node]
                continue
            v=x[ids,int(tree[prefix+'feature_idx'][node])]
            left=v<=tree[prefix+'num_threshold'][node]
            left[np.isnan(v)]=bool(tree[prefix+'missing_go_to_left'][node])
            if left.any():queue.append((int(tree[prefix+'left'][node]),ids[left]))
            if (~left).any():queue.append((int(tree[prefix+'right'][node]),ids[~left]))
        result+=value
    return result


def stats_check(arr,cell,key,seed,pairs):
    """Rebuild grouping with add.at and einsum, not the producer reduction."""
    names=list(cell['models'])
    sq=np.stack([(arr[n]-arr['truth'])**2 for n in names],axis=1)
    means=sq.sum(axis=0)/len(sq)
    assert cell['rows']==len(sq)
    for j,n in enumerate(names):
        assert np.isclose(means[j],cell['models'][n]['MSE'],rtol=1e-12,atol=1e-8)
        assert np.isclose(np.sqrt(means[j]),cell['models'][n]['RMSE'],rtol=1e-12)
    assert set(cell['pairs'])=={a+'_over_'+b for a,b in pairs}
    for form in ('cluster','discharge'):
        labels=np.unique(arr[form]);ng=len(labels)
        assert ng>=2 and ng==cell['clusters' if form=='cluster' else 'discharges']
        inv=np.searchsorted(labels,arr[form]);sums=np.zeros((ng,len(names)));count=np.zeros(ng)
        np.add.at(sums,inv,sq);np.add.at(count,inv,1)
        rng=np.random.default_rng([seed,zlib.crc32((key+'/'+form).encode())])
        draws=[]
        for _ in range(40):
            w=rng.multinomial(ng,np.repeat(1/ng,ng),500)
            draws.append(np.einsum('bg,gc->bc',w,sums)/np.einsum('bg,g->b',w,count)[:,None])
        mm=np.concatenate(draws);assert np.isfinite(mm).all() and (mm>0).all()
        ci=np.percentile(mm,[2.5,97.5],axis=0)
        for j,n in enumerate(names):
            assert np.allclose(ci[:,j],cell['models'][n][form+'_MSE95'],rtol=1e-9,atol=1e-7)
            assert np.allclose(np.sqrt(ci[:,j]),cell['models'][n][form+'_RMSE95'],rtol=1e-9,atol=1e-7)
        for a,b in pairs:
            ia,ib=names.index(a),names.index(b);p=cell['pairs'][a+'_over_'+b]
            q=np.percentile(1-mm[:,ia]/mm[:,ib],[2.5,97.5])
            assert np.isclose(1-means[ia]/means[ib],p['point'],atol=1e-12)
            assert np.allclose(q,p[form]['ci95'],rtol=1e-9,atol=1e-10)
            assert p[form]['excludes_zero']==bool(q[0]>0 or q[1]<0)


def audit_inputs(m,private):
    """Rebuild split membership and physical features from frozen originals."""
    prior=ROOT/'data/colab_b111'
    original_map=read(prior/'private_map.json')
    assert sha(prior/'private_map.json')==private['mapping_source_sha256']==m['mapping_source_sha256']
    codes=np.asarray(original_map['row_codes'],dtype=np.int64)
    back=np.asarray(original_map['row_inverse'],dtype=np.int64)
    assert np.array_equal(codes[back],np.arange(len(back)))
    rows=npz(prior/'bundle/rows.npz')
    ctx=SF.Ctx();R=ctx.R
    assert np.array_equal(rows['truth'],R['tg_F'][back],equal_nan=True)
    for tag,u in m['units'].items():
        own=private['units'][tag];man=own['manifest'];st=own['stats']
        assert C.digest(man)==u['manifest_sha256'] and C.digest(st)==u['original_stats_sha256']
        assert st['train_files']==sorted(man['train'])
        assert C.digest(sorted(man['train']))==u['normalization_files_sha256']
        expected=C.prepared(u['name']+'__outer')['identity']
        assert expected['manifest']==man and expected['stats']==st
        files={s:set(man[s]) for s in ('train','val','test')}
        for a,b in (('train','val'),('train','test'),('val','test')):
            assert not files[a]&files[b]
        if u['name'].startswith('V2'):
            cc={s:set(R['clus'][np.isin(R['fidx'],list(v))]) for s,v in files.items()}
            for a,b in (('train','val'),('train','test'),('val','test')):
                assert not cc[a]&cc[b]
        fast,mb,me=ctx.unit_arrays({'fit':man['train']})
        for rule in RULES:
            z=npz(OUT/'bundle'/(tag+'_'+rule+'.npz'))
            assert all(np.isfinite(v).all() for v in z.values())
            if rule=='K10':
                I=common78.load_inputs(rule)
                H=SF.K.history_matrix(R['tstep'],I)
                A=SF.K.anchor_scalars(R['gi'],I['anc_g'],mb,me,ctx.mr_b,ctx.mr_e)
                gp=I['G'];sets=npz(ROOT/'data/.b78/sets.npz')
            else:
                I=SF.K.load_inputs(rule)
                H=SF.K.history_matrix(R['tstep'],I)
                A=SF.K.anchor_scalars(R['gi'],I['anc_g'],mb,me,ctx.mr_b,ctx.mr_e)
                gp=I['G'];sets=ctx.S
            mask=sets[('S_' if rule=='S' else 'common_')+'F']
            for split,number in (('train',0),('val',1),('test',2)):
                kept=z['split']==number
                if rule=='K10' and number!=2:
                    assert not kept.any() and split not in own['rows'][rule]
                    continue
                ii=np.flatnonzero(mask&np.isin(R['fidx'],man[split]))
                ii=ii[np.argsort(codes[ii])]
                assert np.array_equal(ii,np.array(own['rows'][rule][split],dtype=np.int64))
                xx=np.concatenate((H[ii],fast[ii],A[ii]),axis=1)
                assert z['x'].dtype==np.float64 and np.array_equal(z['x'][kept],xx)
                assert np.array_equal(z['idx'][kept],codes[ii])
                assert np.array_equal(z['y'][kept],R['tg_F'][ii])
                assert np.array_equal(z['gp'][kept],gp[ii])
                assert np.array_equal(z['cluster'][kept],rows['cluster'][codes[ii]])
                assert np.array_equal(z['discharge'][kept],rows['discharge'][codes[ii]])
                v=u['split_hashes'][rule][split]
                assert v['rows']==len(ii) and v['original_rows_sha256']==digest_array(ii)
                assert v['files']==len(np.unique(z['discharge'][kept]))
                assert v['clusters']==len(np.unique(z['cluster'][kept]))
                for n in ('x','y','gp','idx'):
                    assert v[n+'_sha256']==digest_array(z[n][kept])
            ref=npz(OUT/'bundle'/(tag+'_'+rule+'_reference.npz'))
            old=npz(prior/'retrieved/estimates'/(tag+'_'+rule+'.npz'))
            for name,val in ref.items():assert np.array_equal(val,old[name])
        print('B112 original inputs audited '+tag,flush=True)


def mlp_algebra(z,x,gp):
    """Independent NumPy/erf calculation, not a PyTorch model invocation."""
    output=[]
    for start in range(0,len(x),16384):
        a=((x[start:start+16384]-z['x_mu'])/z['x_sd']).astype(np.float32)
        for layer in (0,2,4):
            a=a@z[str(layer)+'_weight'].T+z[str(layer)+'_bias']
            if layer!=4:
                a=np.float32(.5)*a*(np.float32(1)+erf(a/np.float32(np.sqrt(2))))
        output.append(a[:,0])
    result=np.concatenate(output).astype(np.float64)*float(z['y_sd'])+float(z['y_mu'])
    if bool(z['residual']):result+=gp
    assert np.isfinite(result).all()
    return result


def audit_models(m,dest,report):
    fitted=0
    for tag,u in m['units'].items():
        modeldir=dest/'models'/tag;done=read(modeldir/'complete.json')
        assert done['MLP_fits']==6 and done['HGB_fits']==2
        assert done['TEST_fit_rows']==done['VAL_fit_rows']==0
        assert done['sources']==m['sources']
        data={r:npz(OUT/'bundle'/(tag+'_'+r+'.npz')) for r in RULES}
        train=[(r,data[r]['split']==0) for r in TRAIN_RULES]
        x=np.concatenate([data[r]['x'][ok] for r,ok in train])
        y=np.concatenate([data[r]['y'][ok] for r,ok in train])
        gp=np.concatenate([data[r]['gp'][ok] for r,ok in train])
        weights=np.concatenate([np.full(int(ok.sum()),len(y)/(7*int(ok.sum()))) for _,ok in train])
        assert np.isclose(weights.sum(),len(y))
        # Equal-rule means/second central moments are calculated separately
        # from the producer's pooled weighted np.average arithmetic.
        mu=sum(data[r]['x'][ok].mean(axis=0) for r,ok in train)/7
        sd=np.maximum(np.sqrt(sum(((data[r]['x'][ok]-mu)**2).mean(axis=0) for r,ok in train)/7),1e-8)
        sx=npz(modeldir/'stats.npz')
        assert np.allclose(sx['x_mu'],mu,rtol=1e-11,atol=1e-10)
        assert np.allclose(sx['x_sd'],sd,rtol=1e-11,atol=1e-10)
        coverage={}
        for r,ok in train:
            coverage[r]={n+'_sha256':digest_array(data[r][n][ok]) for n in ('idx','x','y','gp')}
            coverage[r]['rows']=int(ok.sum())
        assert done['TRAIN_coverage']==coverage
        mlps={}
        for form in ('direct','residual'):
            target=y if form=='direct' else y-gp
            my=np.average(target,weights=weights)
            sy=max(float(np.sqrt(np.average((target-my)**2,weights=weights))),1.)
            for seed in SEEDS:
                label='MLP_'+form+'_s'+str(seed)
                config=read(modeldir/(label+'_config.json'))
                hist=read(modeldir/(label+'_history.json'))
                complete=read(modeldir/(label+'_complete.json'))
                portable=npz(modeldir/(label+'.npz'))
                assert config['unit']==tag and config['form']==form and config['seed']==seed
                assert config['architecture']==[x.shape[1],64,64,1]
                for n,v in {'activation':'GELU_exact','dropout':0,'optimizer':'AdamW','lr':1e-3,
                            'weight_decay':1e-4,'batch_size':8192,'max_epochs':80,'patience':8,
                            'VAL_fit_rows':0,'TEST_fit_rows':0,'dtype':'float32','AMP':False,'TF32':False}.items():
                    assert config[n]==v
                assert config['gpu']=='Tesla T4' and config['env']==ENV_PINS
                assert config['TRAIN_coverage']==coverage
                assert config['target_mu']==float(my) and config['target_sd']==sy
                assert np.array_equal(portable['x_mu'],sx['x_mu']) and np.array_equal(portable['x_sd'],sx['x_sd'])
                assert float(portable['y_mu'])==float(my) and float(portable['y_sd'])==sy
                assert bool(portable['residual'])==(form=='residual')
                for layer,shape in ((0,(64,x.shape[1])),(2,(64,64)),(4,(1,64))):
                    assert portable[str(layer)+'_weight'].shape==shape
                    assert portable[str(layer)+'_weight'].dtype==np.float32
                    assert portable[str(layer)+'_bias'].shape==(shape[0],)
                assert all(np.isfinite(a).all() for a in portable.values())
                assert sha(modeldir/(label+'.npz'))==complete['weights_sha256']
                assert complete['epochs']==len(hist) and 1<=len(hist)<=80
                best=float('inf');stale=0;best_epoch=None
                for epoch,h in enumerate(hist):
                    assert h['epoch']==epoch and set(h['VAL_rule_MSE'])==set(TRAIN_RULES)
                    assert np.isfinite(h['train_normalized_weighted_MSE']) and h['train_normalized_weighted_MSE']>=0
                    assert np.isclose(h['VAL_MSE'],np.mean(list(h['VAL_rule_MSE'].values())),rtol=1e-12)
                    assert h['improved']==bool(h['VAL_MSE']<best)
                    if h['improved']:best=h['VAL_MSE'];best_epoch=epoch;stale=0
                    else:stale+=1
                    assert stale<8 or epoch==len(hist)-1
                assert complete['best_epoch']==best_epoch and complete['best_VAL_MSE']==best
                assert complete['VAL_fit_rows']==complete['TEST_fit_rows']==0
                assert (complete['stopping']=='patience8' and stale==8) or (complete['stopping']=='max80' and len(hist)==80 and stale<8)
                validation=[]
                for r in TRAIN_RULES:
                    ok=data[r]['split']==1
                    pred=mlp_algebra(portable,data[r]['x'][ok],data[r]['gp'][ok])
                    actual=float(np.mean((pred-data[r]['y'][ok])**2))
                    assert np.isclose(actual,hist[best_epoch]['VAL_rule_MSE'][r],rtol=2e-5,atol=.1)
                    validation.append(actual)
                assert np.isclose(np.mean(validation),best,rtol=2e-5,atol=.1)
                mlps[label]=portable;fitted+=1
        trees={}
        for form in ('direct','residual'):
            label='HGB_'+form;v=read(modeldir/(label+'_complete.json'))
            assert v['TRAIN_coverage']==coverage and v['VAL_fit_rows']==v['TEST_fit_rows']==0
            assert v['portable_equal'] and v['iterations']==100
            assert v['params']['random_state']==20261116 and not v['params']['early_stopping'] and v['params']['max_iter']==100
            assert sha(modeldir/(label+'.joblib'))==v['joblib_sha256']
            assert sha(modeldir/(label+'_trees.npz'))==v['portable_sha256']
            trees[label]=npz(modeldir/(label+'_trees.npz'))
            assert int(trees[label]['trees'])==100
            fitted+=1
        for rule in RULES:
            p=dest/'estimates'/(tag+'_'+rule+'.npz')
            assert sha(p)==done['estimates'][p.name]
            arr=npz(p);z=data[rule];test=z['split']==2
            ref=npz(OUT/'bundle'/(tag+'_'+rule+'_reference.npz'))
            for n,v in ref.items():assert np.array_equal(arr[n],v)
            assert np.array_equal(arr['idx'],z['idx'][test]) and np.array_equal(arr['truth'],z['y'][test])
            for label,portable in mlps.items():
                expected=mlp_algebra(portable,z['x'][test],z['gp'][test])
                assert np.allclose(expected,arr[label],rtol=1e-5,atol=.01),label+' algebra mismatch'
            for form in ('direct','residual'):
                label='HGB_'+form;expected=predict_nodes(trees[label],z['x'][test])
                if form=='residual':expected+=z['gp'][test]
                assert np.allclose(expected,arr[label],rtol=1e-12,atol=1e-8)
                ensemble=np.mean([arr['MLP_'+form+'_s'+str(s)] for s in SEEDS],axis=0)
                assert np.array_equal(ensemble,arr['MLP_'+form+'_ensemble'])
            key=u['name']+'/'+rule+'/F'
            assert set(report['cells'][key]['models'])=={n for n in MODELS if n in arr}
            stats_check(arr,report['cells'][key],key,20261125,PAIRS)
            print('B112 prediction/statistics audited '+key,flush=True)
        del data,x,y,gp,weights,mlps,trees
    assert fitted==32


def main():
    guard();C.guard();C.inputs_guard()
    state=read(ROOT/'data/.colab_connection/b112_remote.json')
    assert state['downloaded']
    archive=OUT/'b112_results.zip';assert sha(archive)==state['results_sha256']
    dest=OUT/'retrieved';assert not dest.exists(),'Preserve existing extraction/audit attempt'
    dest.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n in z.namelist():assert (dest/n).resolve().is_relative_to(dest.resolve())
        z.extractall(dest)
    for n,h in read(dest/'result_hashes.json').items():
        assert (dest/n).resolve().is_relative_to(dest.resolve()) and sha(dest/n)==h
    m=read(OUT/'bundle/manifest.json');private=read(OUT/'private_map.json')
    assert m==read(dest/'manifest.json')
    assert m['sources']==sources() and m['registration']=='PREREGISTRATION_B112.md'
    for n,h in m['files'].items():assert sha(OUT/'bundle'/n)==h
    for rel,h in m['reference_sha256'].items():assert sha(ROOT/rel)==h
    report=read(dest/'readout.json')
    assert report['sources']==m['sources'] and report['registration']=='PREREGISTRATION_B112.md'
    assert report['MLP_fits']==24 and report['HGB_fits']==8 and report['seeds']==list(SEEDS)
    assert report['draws']==20000 and report['bootstrap_seed']==20261125 and report['exploratory']
    assert len(report['cells'])==32 and report['target']=='current_report_F'
    started=read(dest/'started.json')
    assert started['gpu']=='Tesla T4' and started['env']==ENV_PINS
    training=read(dest/'training_complete.json')
    assert training['all_pass'] and set(training['tags'])==set(m['units']) and len(training['tags'])==4
    for tag in training['tags']:
        exit_record=read(dest/'processes'/(tag+'_exit.json'))
        start=read(dest/'processes'/(tag+'_start.json'))
        assert exit_record['returncode']==0 and not exit_record['timeout'] and start['tag']==tag
    audit_inputs(m,private)
    audit_models(m,dest,report)
    c=report['cells'];fresh=[c[u+'/D10/F'] for u in UNITS]
    expected={
        'R1_residual_over_direct_all_fresh':all(v['pairs']['MLP_residual_ensemble_over_MLP_direct_ensemble']['cluster']['ci95'][0]>0 for v in fresh),
        'R2_residual_over_HGB_all_fresh':all(v['pairs']['MLP_residual_ensemble_over_HGB_residual']['cluster']['ci95'][0]>0 for v in fresh),
        'R3_residual_over_HGB_all_scenarios':all(v['pairs']['MLP_residual_ensemble_over_HGB_residual']['cluster']['ci95'][0]>0 for v in c.values())}
    assert report['statements']==expected
    publish(report,archive)
    print('B112 independent full audit passed.',flush=True)


def publish(report,archive):
    audit={'all_pass':True,'MSE_cells':32,'MLP_fits':24,'HGB_fits':8,
           'TRAIN_normalization_rebuilt':True,'split_roles_rebuilt':True,
           'portable_model_outputs_rebuilt':True,'all_points_intervals_recomputed':True,
           'archive_sha256':sha(archive),'local_fits':0,'local_GPU_work':0}
    report['independent_audit']=audit
    write(OUT/'independent_audit.json',audit)
    write(ROOT/'docs/paper/b112_comparison.json',report)
    lines=['# B112: matched direct and GP-residual learning','',
           'Exploratory reused-TEST fixed-fit comparison; all four validation designs.',
           *report.get('limits',[]),'',
           'MSE eV^2 and RMSE eV. Brackets are pointwise 95% shot-cluster intervals;',
           'discharge intervals and every seed are preserved in the JSON.','',
           '| Unit / rule | Model | Rows | MSE [95%] | RMSE |',
           '|---|---|---:|---|---:|']
    for key,c in report['cells'].items():
        for name,v in c['models'].items():
            q=v['cluster_MSE95']
            lines.append(f"| {key} | {name} | {c['rows']} | {v['MSE']:.3f} [{q[0]:.3f}, {q[1]:.3f}] | {v['RMSE']:.3f} |")
    lines+=['','## Paired skill','',
            'Skill=1-MSE(candidate)/MSE(reference). Positive means smaller candidate MSE.','',
            '| Unit / rule | Contrast | Point [95% cluster] | Excludes zero |',
            '|---|---|---|---|']
    for key,c in report['cells'].items():
        for name,v in c['pairs'].items():
            q=v['cluster']['ci95']
            lines.append(f"| {key} | {name} | {v['point']:+.6f} [{q[0]:+.6f}, {q[1]:+.6f}] | {v['cluster']['excludes_zero']} |")
    lines+=['','Statements: '+json.dumps(report.get('statements',{})),
            '', 'Intervals omit retraining, selection and initialization uncertainty. '
            'The seed ensemble does not constitute independent confirmation. '
            'A matched comparison on unused observations could reverse an unfavorable conclusion. '
            'No architecture impossibility, new physics or algorithm originality is established.']
    (ROOT/'docs/paper/b112_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
