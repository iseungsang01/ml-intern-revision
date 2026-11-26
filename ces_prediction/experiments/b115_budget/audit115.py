"""Independent B115 schedule/prefix checks, neural replay and paired readout.

No model fitting. Old/new predictions share the same registered group draws;
confidence intervals condition on fixed fits and previously used TEST support.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import json
import sys
import time
from pathlib import Path
import numpy as np
import common115 as C
sys.path.insert(0,str(C.HERE.parent/'b114_overnight'))
import audit114 as OLD
A=OLD.A
BASE=C.ROOT/'data/.b114'
SEEDS=C.SEEDS
LONG=tuple('NN_long_s'+str(s) for s in SEEDS)
SHORT=tuple('NN_base_s'+str(s) for s in SEEDS)
MODELS=('NN_long_ensemble','NN_base_ensemble',*LONG,*SHORT,'GP_fast_boost','GP_fair_mixture','GP_only')
PAIRS=(('NN_long_ensemble','NN_base_ensemble'),('NN_long_ensemble','GP_fast_boost'),
       ('NN_base_ensemble','GP_fast_boost'))+tuple((a,b) for a,b in zip(LONG,SHORT))+tuple((a,'GP_fast_boost') for a in LONG+SHORT)


def lr_for(epoch):
    return 0.001 if epoch<80 else 0.0003 if epoch<160 else 0.0001 if epoch<240 else 0.00003


def verify_state(state,context,form,architecture,knots):
    _,y,gp,weights,mu,sd,_=context
    target=y if form=='direct' else y-gp
    ym=float(np.average(target,weights=weights))
    ys=float(max(np.sqrt(np.average((target-ym)**2,weights=weights)),1.0))
    assert np.array_equal(state['x_mu'],mu) and np.array_equal(state['x_sd'],sd)
    assert float(state['y_mu'])==ym and float(state['y_sd'])==ys
    assert bool(state['residual'])==(form=='residual')
    for name,array in state.items():
        assert np.isfinite(array).all(),name
    if architecture=='ple128':
        assert np.array_equal(state['model.knots'],knots)


def verify_history(history,receipt,old_history):
    assert 1<=len(history)<=300 and len(history)>=len(old_history)
    assert receipt['epochs']==len(history)
    assert receipt['best_epoch']==int(np.argmin([h['objective'] for h in history]))
    assert receipt['best_VAL_MSE']==min(h['objective'] for h in history)
    best=float('inf')
    for epoch,h in enumerate(history):
        assert h['epoch']==epoch and h['learning_rate']==lr_for(epoch)
        assert set(h['VAL_rule_MSE'])==set(C.TRAIN_RULES)
        score=float(np.mean(list(h['VAL_rule_MSE'].values())))
        assert np.isfinite(score) and score==h['VAL_MSE']==h['objective']
        assert np.isfinite(h['train_normalized_weighted_MSE'])
        assert h['improved']==(score<best)
        best=min(best,score)
    # The new policy replays the original trajectory before its old early stop.
    max_train=max_val=0.0
    for before,after in zip(old_history,history):
        for key,atol in [('train_normalized_weighted_MSE',1e-5),('VAL_MSE',0.01),('objective',0.01)]:
            assert np.isclose(before[key],after[key],rtol=1e-6,atol=atol),'Old-prefix mismatch: '+key
        for rule in C.TRAIN_RULES:
            assert np.isclose(before['VAL_rule_MSE'][rule],after['VAL_rule_MSE'][rule],rtol=1e-6,atol=.01)
        max_train=max(max_train,abs(before['train_normalized_weighted_MSE']-after['train_normalized_weighted_MSE']))
        max_val=max(max_val,abs(before['VAL_MSE']-after['VAL_MSE']))
    assert receipt['prefix_check']['all_pass'] and receipt['prefix_check']['epochs']==len(old_history)
    # No early stopping in the registered extended arm.
    assert len(history)==300 or history[-1]['seconds']>=300,'Premature extended stop'
    return {'old_epochs':len(old_history),'new_epochs':len(history),'max_abs_train_loss':max_train,
            'max_abs_VAL_MSE':max_val,'old_best_epoch':int(np.argmin([h['objective'] for h in old_history])),
            'new_best_epoch':receipt['best_epoch'],'new_stopping':receipt['stopping'],'new_seconds':receipt['seconds']}


def cluster_loss_variation(arr):
    """Descriptive cluster heterogeneity; this is not another significance test.

    Cluster mean differences get equal weight in their SD. The global mean
    remains row-weighted and is a loss difference in eV^2 (GP minus candidate).
    Row-weighted cluster contributions sum to the global loss difference.
    """
    labels,inv=np.unique(arr['cluster'],return_inverse=True)
    count=np.bincount(inv);result={}
    gp_loss=(np.asarray(arr['GP_fast_boost'],dtype=np.float64)-arr['truth'])**2
    for form,name in [('base','NN_base_ensemble'),('long','NN_long_ensemble')]:
        loss=(np.asarray(arr[name],dtype=np.float64)-arr['truth'])**2
        delta=gp_loss-loss
        total=np.bincount(inv,weights=delta,minlength=len(labels))
        means=total/count;contribution=total/len(delta)
        global_mean=float(np.mean(delta))
        assert np.isclose(contribution.sum(),global_mean,rtol=1e-12,atol=1e-7)
        result[form]={'clusters':len(labels),'rows':len(delta),
                      'estimand':'loss_GP_minus_loss_candidate_eV2',
                      'cluster_mean_SD_weighting':'equal_cluster_ddof1_descriptive',
                      'cluster_mean_SD':float(np.std(means,ddof=1)),
                      'cluster_mean_min':float(means.min()),'cluster_mean_max':float(means.max()),
                      'row_weighted_global_mean':global_mean,
                      'row_weighted_cluster_contribution_SD':float(np.std(contribution,ddof=1)),
                      'row_weighted_cluster_contribution_min':float(contribution.min()),
                      'row_weighted_cluster_contribution_max':float(contribution.max()),
                      'cluster_size_min':int(count.min()),'cluster_size_max':int(count.max()),
                      'cluster_size_median':float(np.median(count))}
    return result


def render(report):
    lines=['# B115: extended scheduled neural training versus frozen B114 control','',
           'Exploratory reused-TEST fixed-fit comparison; the architecture and direct/residual form remain the original per-unit B114 VAL choices.',
           'No early stopping in the new arm: up to300epochs or300seconds, checked after a complete epoch and VAL; earliest strict VAL minimum retained.',
           'Long/base intervals use identical paired resampling draws. More training does not add independent TEST clusters or guarantee narrower intervals.',
           'MSE in eV^2; RMSE in eV. Intervals are pointwise95% shot-cluster, with discharge intervals and every seed in JSON.','',
           '| Unit | Architecture / form | Old / new epochs by seed | New best epochs |',
           '|---|---|---|---|']
    for tag,unit in report['units'].items():
        checks=report['training_checks'][tag]
        epochs=', '.join(str(checks[str(s)]['old_epochs'])+'/'+str(checks[str(s)]['new_epochs']) for s in SEEDS)
        best=', '.join(str(checks[str(s)]['new_best_epoch']) for s in SEEDS)
        lines.append(f"| {unit['name']} | {unit['architecture']}/{unit['form']} | {epochs} | {best} |")
    lines+=['','| Unit / rule / F | Model | Rows | MSE [95%cluster] | RMSE |','|---|---|---:|---|---:|']
    for key,cell in report['cells'].items():
        for name,model in cell['models'].items():
            lo,hi=model['cluster_MSE95']
            lines.append(f"| {key} | {name} | {cell['rows']} | {model['MSE']:.3f} [{lo:.3f}, {hi:.3f}] | {model['RMSE']:.3f} |")
    lines+=['','| Unit / rule / F | Paired MSE skill | Point [95%cluster] | Excludes zero |','|---|---|---|---|']
    for key,cell in report['cells'].items():
        for name,value in cell['pairs'].items():
            if '_ensemble_over_' in name:
                lo,hi=value['cluster']['ci95']
                lines.append(f"| {key} | {name} | {value['point']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {value['cluster']['excludes_zero']} |")
    lines+=['','| Unit / rule / F | Cluster skill-CI width ratio long/base vs GP+fast | Discharge ratio |','|---|---:|---:|']
    for key,widths in report['CI_width_comparison'].items():
        def fmt(value):return 'undefined' if value is None else f'{value:.6f}'
        lines.append('| '+key+' | '+fmt(widths['cluster']['ratio_long_to_base'])+' | '+fmt(widths['discharge']['ratio_long_to_base'])+' |')
    lines+=['','Cluster error variation in JSON reports equal-cluster SD/min/max of mean(lossGP-lossCandidate), separately from the row-weighted global mean. Its units are eV^2; it is descriptive, without a new test or interval.']
    lines+=['','Registered statements: '+json.dumps(report['statements']),'',
            'Interval width ratios below1 indicate narrower intervals for that same frozen support/draws, not additional independent information.',
            'Unfavorable or unresolved results can be overturned by a matched adequate-budget comparison on unused observations.',
            'Intervals omit refitting and selection uncertainty; no architecture impossibility, latent physical truth or10ms latency claim is made.']
    return '\n'.join(lines)+'\n'


def main():
    sources=C.source_guard();manifest=C.verify_inputs()
    old_audit=C.read(BASE/'audit.json');assert old_audit['all_pass']
    old_report=C.read(C.ROOT/'docs/paper/b114_comparison.json');assert old_report['independent_audit']==old_audit
    OLD.checked_files(C.ROOT,old_audit['sources'])
    selection=C.read(BASE/'selection.json');assert C.sha(BASE/'selection.json')==old_audit['selection_sha256']
    fit=C.read(C.OUT/'fit_complete.json');complete=C.read(C.OUT/'prediction_complete.json')
    assert fit['all_pass'] and fit['fits']==12 and fit['sources']==sources
    assert fit.get('TEST_predictions',0)==0 and len(fit['receipts'])==12
    assert complete['all_pass'] and complete['cells']==len(complete['files'])==32
    assert complete['sources']==sources and complete['selection_sha256']==C.sha(BASE/'selection.json')
    OLD.checked_files(C.OUT,fit['receipts']);OLD.checked_files(C.OUT,fit['val_files']);OLD.checked_files(C.OUT,complete['files'])
    old_complete=C.read(BASE/'neural/prediction_complete.json')
    assert old_complete['all_pass'] and old_complete['selection_sha256']==C.sha(BASE/'selection.json')
    OLD.checked_files(BASE/'neural',old_complete['files'])
    A.MODELS,A.PAIRS=MODELS,PAIRS
    cells={};replay={};training={};units={};widths={};runtime=[];variation={}
    for tag,unit in manifest['units'].items():
        choice=selection['neural']['by_unit'][tag];arch,form=choice['architecture'],choice['form']
        units[tag]={'name':unit['name'],'architecture':arch,'form':form}
        context=OLD.training_contract(tag);knots=OLD.expected_knots(context)
        base_states={};long_states={};training[tag]={}
        for seed in SEEDS:
            stem=form+'_s'+str(seed)
            old_dir=BASE/'neural/models'/tag/arch;new_dir=C.OUT/'models'/tag/arch
            before=C.read(old_dir/(stem+'_complete.json'));after=C.read(new_dir/(stem+'_complete.json'))
            h_before=C.read(old_dir/(stem+'_history.json'));h_after=C.read(new_dir/(stem+'_history.json'))
            OLD.verify_receipt(before,context,form);OLD.verify_receipt(after,context,form)
            for folder,receipt in [(old_dir,before),(new_dir,after)]:
                assert C.sha(folder/(stem+'.npz'))==receipt['weights_sha256']
                assert C.sha(folder/(stem+'_history.json'))==receipt['history_sha256']
            assert after['prefix_check']['old_history_sha256']==C.sha(old_dir/(stem+'_history.json'))
            assert after['prefix_check']['old_weights_sha256']==C.sha(old_dir/(stem+'.npz'))
            base_states[seed]=C.npz(old_dir/(stem+'.npz'));long_states[seed]=C.npz(new_dir/(stem+'.npz'))
            OLD.verify_neural_state(base_states[seed],before,h_before,context,form,arch,knots)
            verify_state(long_states[seed],context,form,arch,knots)
            assert after['params']['AMP'] is False and after['params']['TF32'] is False
            assert after['params']['early_stopping'] is False
            assert after['params']['max_epochs']==300 and after['params']['seconds_budget']==300
            assert after['params']['learning_rates']==[[0,.001],[80,.0003],[160,.0001],[240,.00003]]
            assert after['params']['batch_size']==8192 and after['params']['weight_decay']==.0001
            assert after['params']['architecture']==arch and after['form']==form
            assert after['stopping'] in ('max_epochs','seconds300')
            training[tag][str(seed)]=verify_history(h_after,after,h_before)
            training[tag][str(seed)]['old_stopping']=before['stopping']
            training[tag][str(seed)]['old_seconds']=before['seconds']
            runtime.append({'unit':tag,'seed':seed,'base_seconds':before['seconds'],'long_seconds':after['seconds']})
        # The retained best model must reproduce its saved VAL predictions and score.
        val_scores={str(seed):[] for seed in SEEDS}
        for rule in C.TRAIN_RULES:
            original=C.npz(C.PRIOR/f'{tag}_{rule}.npz');mask=original['split']==1
            arr=C.npz(C.OUT/'val'/f'{tag}_{rule}.npz')
            for name,source in [('idx','idx'),('truth','y'),('gp','gp'),('cluster','cluster'),('discharge','discharge')]:
                assert np.array_equal(arr[name],original[source][mask]),'VAL support changed'
            for seed in SEEDS:
                pred=OLD.neural_predict(long_states[seed],arch,original['x'][mask],original['gp'][mask] if form=='residual' else None)
                assert np.allclose(pred,arr['NN_long_s'+str(seed)],atol=.01,rtol=1e-5)
                val_scores[str(seed)].append(float(np.mean((pred-original['y'][mask])**2)))
                pred=OLD.neural_predict(base_states[seed],arch,original['x'][mask],original['gp'][mask] if form=='residual' else None)
                assert np.allclose(pred,arr['NN_base_s'+str(seed)],atol=.01,rtol=1e-5)
            for prefix in ('NN_long','NN_base'):
                ens=np.mean([arr[prefix+'_s'+str(seed)] for seed in SEEDS],axis=0)
                assert np.array_equal(ens,arr[prefix+'_ensemble'])
        for seed in SEEDS:
            receipt=C.read(C.OUT/'models'/tag/arch/(form+'_s'+str(seed)+'_complete.json'))
            assert np.isclose(np.mean(val_scores[str(seed)]),receipt['best_VAL_MSE'],rtol=1e-4,atol=1e-3)
        for rule in C.RULES:
            original=C.npz(C.PRIOR/f'{tag}_{rule}.npz');mask=original['split']==2
            if rule=='K10':assert mask.all()
            arr=C.npz(C.OUT/'estimates'/f'{tag}_{rule}.npz')
            old=C.npz(BASE/'neural/estimates'/f'{tag}_{rule}.npz')
            for name,source in [('idx','idx'),('truth','y'),('cluster','cluster'),('discharge','discharge')]:
                assert np.array_equal(arr[name],original[source][mask]) and np.array_equal(old[name],arr[name])
            errors={}
            for seed in SEEDS:
                old_name='NN_selected_s'+str(seed);base_name='NN_base_s'+str(seed);long_name='NN_long_s'+str(seed)
                assert np.array_equal(arr[base_name],old[old_name])
                for name,state in [(base_name,base_states[seed]),(long_name,long_states[seed])]:
                    pred=OLD.neural_predict(state,arch,original['x'][mask],original['gp'][mask] if form=='residual' else None)
                    errors[name]=float(np.max(np.abs(pred-arr[name])))
                    assert np.allclose(pred,arr[name],atol=.01,rtol=1e-5),name
            for prefix in ('NN_long','NN_base'):
                assert np.array_equal(np.mean([arr[prefix+'_s'+str(seed)] for seed in SEEDS],axis=0),arr[prefix+'_ensemble'])
            assert np.array_equal(arr['NN_base_ensemble'],old['NN_selected_ensemble'])
            for name in ('GP_fast_boost','GP_fair_mixture','GP_only'):
                assert np.array_equal(arr[name],old[name])
            key=unit['name']+'/'+rule+'/F';cells[key]=A.statistics(arr,key);replay[key]=errors
            variation[key]=cluster_loss_variation(arr)
            # The old-control arithmetic and intervals must agree with the original audit.
            prior=old_report['cells'][key]
            assert np.isclose(cells[key]['models']['NN_base_ensemble']['MSE'],prior['models']['NN_selected_ensemble']['MSE'],rtol=1e-12,atol=1e-7)
            widths[key]={}
            for grouping in ('cluster','discharge'):
                a=cells[key]['pairs']['NN_long_ensemble_over_GP_fast_boost'][grouping]['ci95']
                b=cells[key]['pairs']['NN_base_ensemble_over_GP_fast_boost'][grouping]['ci95']
                prior_ci=prior['pairs']['NN_selected_ensemble_over_GP_fast_boost'][grouping]['ci95']
                assert np.allclose(b,prior_ci,rtol=1e-10,atol=1e-10)
                wa,wb=float(a[1]-a[0]),float(b[1]-b[0])
                widths[key][grouping]={'long_width':wa,'base_width':wb,'ratio_long_to_base':wa/wb if wb>0 else None}
            print('B115 independent replay/readout '+key,flush=True)
    assert len(cells)==32
    fresh=[v for k,v in cells.items() if '/D10/' in k];assert len(fresh)==4
    statements={name:all(v['pairs'][pair]['cluster']['ci95'][0]>0 for v in fresh) for name,pair in [('R1','NN_long_ensemble_over_NN_base_ensemble'),('R2','NN_long_ensemble_over_GP_fast_boost')]}
    audit={'all_pass':True,'completed':time.time(),'fits':12,'cells':32,'sources':sources,
           'training_prefix_checks':training,'portable_replay_max_abs_ev':replay,
           'normalization_and_PLE_rebuilt':True,'checkpoint_history_rebuilt':True,
           'cluster_error_variation_verified':True,
           'bootstrap_draws':20000,'bootstrap_seed':20261126,'same_draws_old_new':True,
           'old_audit_sha256':C.sha(BASE/'audit.json'),'new_fit_complete_sha256':C.sha(C.OUT/'fit_complete.json'),
           'new_prediction_complete_sha256':C.sha(C.OUT/'prediction_complete.json'),
           'neural_tolerance':{'atol_ev':.01,'rtol':1e-5},'local_fits_in_audit':0}
    report={'registration':'PREREGISTRATION_B115.md','target':'current_report_F','exploratory':True,
            'units':units,'training_checks':training,'runtime':runtime,'cells':cells,'CI_width_comparison':widths,
            'cluster_error_variation':variation,
            'statements':statements,'independent_audit':audit,
            'limits':['Reused TEST; selection- and fixed-fit-conditional pointwise intervals.',
                      'No refit or hyperparameter-selection uncertainty in resampling intervals.',
                      'More epochs or seed fits do not add independent TEST clusters.',
                      'CI widths may shrink or grow; no data-dependent deletion or interval-level change.',
                      'No physical identification or10ms latency claim.']}
    C.save(C.OUT/'audit.json',audit)
    C.save(C.ROOT/'docs/paper/b115_comparison.json',report)
    (C.ROOT/'docs/paper/b115_comparison.md').write_text(render(report),encoding='utf8')


if __name__=='__main__':main()
