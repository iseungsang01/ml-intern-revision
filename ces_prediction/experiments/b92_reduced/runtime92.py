"""Consume the immutable B91 qualification without a silent fallback."""
import json,platform,statistics,sys
import torch
import common92 as M
def decision():
    M.guard();path=M.DATA/'.b91/runtime_decision.json';z=json.loads(path.read_text(encoding='utf8'))
    import importlib.util
    verifier=M.HERE.parent/'b91_fast/runtime91.py'
    sys.path.insert(0,str(verifier.parent))
    s=importlib.util.spec_from_file_location('_b92_decision91',verifier);v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
    verified=v.verify_decision();assert verified==z
    assert z.get('selected_path') and z.get('native_clean'),'No eligible B91 runtime'
    override_path=M.DATA/'.b92/production_override_r2.json';override=json.loads(override_path.read_text(encoding='utf8'))
    assert override['registration']=='PREREGISTRATION_B92_ADDENDUM2.md'
    reg=M.ROOT/'ces_prediction/experiments'/override['registration']
    import subprocess
    subprocess.run(['git','cat-file','-e','HEAD:'+reg.relative_to(M.ROOT).as_posix()],cwd=M.ROOT,check=True)
    subprocess.run(['git','diff','--quiet','HEAD','--',str(reg.relative_to(M.ROOT))],cwd=M.ROOT,check=True)
    assert override['original_decision_sha256']==M.sha(path)
    failed=(M.ROOT/override['failed_smoke_exit_path']).resolve();assert failed.is_relative_to(M.DATA/'.b92')
    assert M.sha(failed)==override['failed_smoke_exit_sha256']
    fail=json.loads(failed.read_text());assert fail['returncode']==3221226505 and not fail['timeout']
    eligible={};evidence={};configs={}
    for name in sorted({r['path'] for r in z['records']}):
        rows=[r for r in z['records'] if r['path']==name]
        if len(rows)!=3 or any(r['config'].get('cudnn',False) for r in rows):continue
        if not all(r['state']=='clean' and r['returncode']==0 for r in rows):continue
        entries=[]
        for r in rows:
            output=M.DATA/'.b91/probes'/f"{name}__{r['rep']}.json";attest=output.with_suffix('.exit.json')
            assert M.sha(output)==r['output_hash'] and M.sha(attest)==r['exit_hash']
            actual=json.loads(output.read_text());att=json.loads(attest.read_text())
            assert actual==r['result'] and actual['sources']==z['sources'] and actual['input_binding']==z['input_binding']
            assert actual['config']==r['config'] and actual['all_pass']
            assert actual['software']['torch']==z['torch_version']
            assert att['returncode']==0 and att['result_sha256']==r['output_hash'] and att['sources']==z['sources']
            assert att['path']==name and att['rep']==r['rep']
            assert r['config']==rows[0]['config']
            entries.append({'result_path':output.relative_to(M.ROOT).as_posix(),'result_sha256':M.sha(output),'exit_path':attest.relative_to(M.ROOT).as_posix(),'exit_sha256':M.sha(attest)})
        eligible[name]=statistics.median(r['result']['training']['median_batch_seconds'] for r in rows);evidence[name]=entries;configs[name]=rows[0]['config']
    assert eligible==override['eligible_non_cudnn_median_seconds']
    native='cuda_packed_native';assert native in eligible
    fastest=min(eligible,key=eligible.get)
    if eligible[native]<=eligible[fastest]*1.05:fastest=native
    chosen=fastest if eligible[native]/eligible[fastest]>=1.25 else native
    assert chosen==override['selected_path']=='cpu_dense_mkldnn8'
    config=configs[chosen];assert config==override['selected_config'] and config['device']=='cpu' and not config['cudnn']
    assert evidence[chosen]==override['selected_evidence']
    return {**config,'decision_sha256':M.sha(path),'production_override_sha256':M.sha(override_path),'decision':z}
def configure(z):
    torch.backends.cudnn.enabled=bool(z.get('cudnn',False));torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(int(z.get('threads',1)))
    torch.backends.mkldnn.enabled=bool(z.get('mkldnn',True))
    if z['device']=='cuda':assert torch.cuda.is_available(),'Qualified CUDA unavailable'
def synchronize(z):
    if z['device']=='cuda':torch.cuda.synchronize()
