"""Independent explicit non-cuDNN production override after the failed smoke."""
import sys,json,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common92 as M


def main():
    M.guard();p=M.DATA/'.b92/production_override_r2.json';o=json.loads(p.read_text());dp=M.DATA/'.b91/runtime_decision.json';d=json.loads(dp.read_text())
    assert o['registration']=='PREREGISTRATION_B92_ADDENDUM2.md' and o['original_decision_sha256']==M.sha(dp)
    failure=M.ROOT/o['failed_smoke_exit_path'];f=json.loads(failure.read_text())
    assert M.sha(failure)==o['failed_smoke_exit_sha256'] and f['returncode']==3221226505 and not f['timeout']
    allowed={}
    for name in d['eligible_median_batch_seconds']:
        rr=[r for r in d['records'] if r['path']==name]
        if rr[0]['config']['cudnn']:continue
        times=[]
        for r in rr:
            result=M.DATA/'.b91/probes'/f"{name}__{r['rep']}.json";att=result.with_suffix('.exit.json');a=json.loads(att.read_text());v=json.loads(result.read_text())
            assert r['state']=='clean' and a['returncode']==0 and r['returncode']==0
            assert M.sha(result)==r['output_hash']==a['result_sha256'] and M.sha(att)==r['exit_hash']
            times.append(statistics.median(s['batch_seconds'] for s in v['training']['steps']))
        allowed[name]=statistics.median(times)
    chosen=min(allowed,key=allowed.get);native='cuda_packed_native'
    if allowed[native]<=allowed[chosen]*1.05:chosen=native
    factor=allowed[native]/allowed[chosen]
    if factor<1.25:chosen=native
    assert chosen==o['selected_path']=='cpu_dense_mkldnn8' and allowed==o['eligible_non_cudnn_median_seconds']
    for e in o['selected_evidence']:
        assert M.sha(M.ROOT/e['result_path'])==e['result_sha256'] and M.sha(M.ROOT/e['exit_path'])==e['exit_sha256']
    report={'passes':True,'sources':M.sources(),'production_override_sha256':M.sha(p),'selected_path':chosen,'batch_speed_factor':factor,'qualification':'dropout0 timing and preserved failed cuDNN dropout.1 smoke; new CPU dropout.1 smoke still required'}
    M.save(M.CACHE/'runtime_override_lead_checks.json',report);M.C.save_text(M.PAPER/'b92_runtime_override_lead_checks.md','# B.92 explicit CPU runtime revision audit\n\n'+json.dumps(report,indent=2)+'\n')
    print('PASS: independent CPU override ranking and normal-exit evidence')


if __name__=='__main__':main()
