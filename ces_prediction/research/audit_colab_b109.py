"""Independent CPU-only audit of all retrieved continuation state bindings."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib,json,zipfile
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'data/colab_b109';PAPER=ROOT/'docs/paper'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    state=json.loads((ROOT/'data/.colab_connection/b109_remote.json').read_text());archive=DATA/'b109_colab_results.zip';assert sha(archive)==state['results_sha256'];dest=DATA/'retrieved';assert not dest.exists();dest.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():assert (dest/name).resolve().is_relative_to(dest.resolve())
        z.extractall(dest)
    manifest=json.loads((dest/'manifest.json').read_text());expected=json.loads((DATA/'bundle/manifest.json').read_text());assert manifest==expected
    mapping=json.loads((DATA/'local_case_map.json').read_text());fits={};selection=dict(mapping['retained']);assert len(selection)==11
    for tag,entry in manifest['tasks'].items():
        p=dest/'runs'/tag;read=lambda n:json.loads((p/n).read_text());exit=read('process_exit.json');cfg=read('config.json');complete=read('complete.json');m=read('metrics.json');ready=read('readiness.json')
        assert exit['returncode']==0 and not exit['timeout'] and m['stop']=='patience' and complete['patience_met']
        assert cfg['registration']=='PREREGISTRATION_B109.md' and cfg['cudnn'] and not cfg['tf32'] and cfg['dtype']=='float32' and cfg['dropout']==.1 and cfg['batch']==16
        for name,key in [('best.pt','best_sha256'),('latest.pt','latest_sha256'),('config.json','config_sha256')]:assert sha(p/name)==complete[key]
        assert ready['prefix_equal'] and ready['start_epoch']==entry['start_epoch']
        if entry['mode']=='resume':assert abs(ready['historical_best']-ready['restored_best_val'])<=1e-4*max(1,abs(ready['historical_best']))
        history=m['history'];best=float('inf');ni=0;last_updates=0
        for i,h in enumerate(history):
            assert h['epoch']==i and h['updates']>last_updates;last_updates=h['updates'];v=h['validation']['D10'];assert v['counts'][0]>0 and v['counts'][1]==0
            assert np.isclose(v['squared_sums'][0]/v['counts'][0],h['val_loss'],rtol=1e-10)
            improved=h['val_loss']<best;assert improved==h['improved']
            if improved:best=h['val_loss'];ni=0
            else:ni+=1
        assert ni>=6 and ni==m['trailing_no_improvement'] and len(history)==m['completed_epochs'] and len(history)>entry['start_epoch'] and best==complete['best_val']
        w=torch.load(p/'best.pt',map_location='cpu',weights_only=True);assert w and all(torch.isfinite(v).all() for v in w.values())
        latest=torch.load(p/'latest.pt',map_location='cpu',weights_only=False);assert latest['validated'] and latest['history']==history and latest['epoch']==len(history)-1 and latest['updates']==m['updates'] and latest['config_sha256']==complete['config_sha256']
        assert all(k in latest for k in ('rng_cpu','rng_cuda','rng_python','rng_numpy'))
        case=mapping['tasks'][tag]['case'];assert case not in selection;selection[case]={'path':p.relative_to(ROOT).as_posix(),'best_sha256':complete['best_sha256'],'source':'B109'}
        fits[case]={'mode':entry['mode'],'start_epoch':entry['start_epoch'],'epochs':len(history),'new_epochs':m['new_epochs'],'best_VAL_normalized_MSE':best,'cloud_wall_seconds':m['wall_seconds']}
    assert len(fits)==9 and len(selection)==20
    for case,s in selection.items():assert sha(ROOT/s['path']/'best.pt')==s['best_sha256']
    (DATA/'selected_paths.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
    report={'all_pass':True,'new_cloud_paths':9,'retained_paths':11,'total_selected_paths':20,'fits':fits,'archive_sha256':sha(archive),'TEST_scored':False,'post_corrections_fit':False,'local_GPU_fits':0}
    (PAPER/'b109_fit_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8');(DATA/'independent_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    text=['# B109 retrieved-state audit passed','','All20 selected LSTM paths are available; nine cloud paths met patience and eleven original completed paths are retained. No TEST MSE or correction fit in this batch.','', '| Case | Mode | Start epoch | Final epoch | New epochs | Cloud seconds |','|---|---|---:|---:|---:|---:|']
    for c,v in fits.items():text.append(f"| {c} | {v['mode']} | {v['start_epoch']} | {v['epochs']} | {v['new_epochs']} | {v['cloud_wall_seconds']:.1f} |")
    (PAPER/'b109_fit_audit.md').write_text('\n'.join(text)+'\n',encoding='utf8');print('B109 independent CPU artifact audit passed:20 selected,9 cloud,11 retained.',flush=True)
if __name__=='__main__':main()
