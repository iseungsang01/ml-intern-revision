"""Recompute native-CUDA ranking/concurrency; never consume CPU override evidence."""
import importlib.util,json,math,statistics,sys
import common94 as G
def decision():
    G.guard();path=G.DATA/'.b93/runtime_decision93.json';d=json.loads(path.read_text())
    folder=G.HERE.parent/'b93_gpu';sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('_b94_qualification93',folder/'profile93.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    assert p.verify_decision()==d and d['gpu_only'],'A current GPU-only B93 decision is required'
    eligible={};software={}
    def verify(row):
        out=G.DATA/'.b93/probes'/f"{row['path']}__{row['rep']}.json";exit=out.with_suffix('.exit.json')
        assert row['state']=='clean' and row['returncode']==0
        assert G.sha(out)==row['result_sha256'] and G.sha(exit)==row['exit_sha256']
        result=json.loads(out.read_text());att=json.loads(exit.read_text())
        assert result==row['result'] and result['all_pass'] and result['sources']==d['sources']
        assert result['input_binding']==d['input_binding'] and result['actual_batch_binding']==d['actual_batch_binding']
        checks=result['checks'];assert checks['actual_dropout']==.1 and checks['validation_rule_count']==7
        assert checks['training_updates']==result['actual_batch_binding']['train_batches']
        assert checks['all_training_losses_finite'] and checks['all_validation_sums_finite'] and checks['all_parameters_finite']
        assert not checks['test_inputs_read'] and not checks['scientific_loss_values_exported']
        assert result['config']==p.PATHS[row['path']] and result['config']['device']=='cuda' and not result['config']['cudnn']
        assert att['returncode']==0 and not att['timeout'] and att['result_sha256']==G.sha(out) and att['sources']==d['sources']
        assert att['path']==row['path'] and att['rep']==row['rep'] and att['phase']==row.get('phase','serial')
        if row['path'] in software:assert software[row['path']]==result['software']
        else:software[row['path']]=result['software']
    for name,cfg in p.PATHS.items():
        if cfg['device']!='cuda' or cfg['cudnn']:continue
        rows=[r for r in d['records'] if r['path']==name and r.get('phase','serial')=='serial']
        if len(rows)==3 and all(r['state']=='clean' and r['returncode']==0 for r in rows):
            for r in rows:verify(r)
            eligible[name]=statistics.median(r['result']['timing']['epoch_seconds'] for r in rows)
            assert all(math.isfinite(r['result']['timing']['epoch_seconds']) and r['result']['timing']['epoch_seconds']>0 for r in rows)
    assert eligible and eligible==d['eligible_epoch_seconds']
    selected=min(eligible,key=eligible.get)
    if 'cuda_packed_native' in eligible and eligible['cuda_packed_native']<=eligible[selected]*1.05:selected='cuda_packed_native'
    assert selected==d['selected_path'] and d['selected_config']==p.PATHS[selected]
    rows=[r for r in d['records'] if r['path']==selected and r.get('phase')=='parallel'];workers=1
    if len(rows)==2 and all(r['state']=='clean' and r['returncode']==0 for r in rows):
        for r in rows:verify(r)
        wall=d['parallel_two_jobs_wall_seconds'];assert math.isfinite(wall) and wall>0
        assert wall>=max(r['result']['timing']['epoch_seconds'] for r in rows)
        gain=2*eligible[selected]/wall
        assert math.isclose(gain,d['parallel_aggregate_gain'],rel_tol=1e-12)
        if gain>=1.10:workers=2
    assert workers==d['parallel_workers'] and workers in (1,2)
    import torch
    assert software[selected]['torch']==torch.__version__ and software[selected]['python']==sys.version
    assert software[selected]['cuda']==torch.version.cuda and not software[selected]['cudnn_enabled']
    return {**d['selected_config'],'b93_decision_sha256':G.sha(path),'parallel_workers':workers,'qualification':'B93 GPU-only native CUDA full epoch/normal exits','selected_path':selected}
