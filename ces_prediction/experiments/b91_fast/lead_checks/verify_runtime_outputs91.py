"""Independent exit/artifact, timing-median and validation-aggregation audit."""
import sys,json,statistics
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime91 as M


def main():
    z=json.loads((M.CACHE/'runtime_decision.json').read_text());assert z['sources']==M.sources()
    values={};count=0
    for name in M.PATHS:
        rr=[r for r in z['records'] if r['path']==name];assert len(rr)==3;durations=[]
        for r in rr:
            p=M.CACHE/'probes'/f"{name}__{r['rep']}.json";ap=p.with_suffix('.exit.json')
            child=json.loads(p.read_text());att=json.loads(ap.read_text())
            assert r['state']=='clean' and r['returncode']==att['returncode']==0 and not att['timeout']
            assert M.sha(p)==r['output_hash']==att['result_sha256'] and M.sha(ap)==r['exit_hash']
            assert child['sources']==z['sources'] and child['input_binding']==z['input_binding']
            steps=child['training']['steps'];assert len(steps)==5
            median=statistics.median(s['batch_seconds'] for s in steps)
            assert np.isclose(median,child['training']['median_batch_seconds'],rtol=0,atol=1e-12)
            durations.append(median)
            v=child['validation'];sq=np.asarray(v['squared_sums']);ct=np.asarray(v['counts']);means=[]
            assert len(v['rules'])==7
            for s,c in zip(sq,ct):
                good=c>0
                if good.any():means.append(float(np.mean(s[good]/c[good])))
            independent=float(np.mean(means))
            assert np.isclose(independent,v['cached_objective'],atol=1e-12,rtol=1e-12)
            assert np.isclose(independent,v['original_objective'],atol=1e-6,rtol=1e-5)
            count+=1
        values[name]=statistics.median(durations)
    assert values==z['eligible_median_batch_seconds']
    fastest=min(values,key=values.get);native='cuda_packed_native'
    if values[native]<=values[fastest]*1.05:fastest=native
    ratio=values[native]/values[fastest];chosen=fastest if ratio>=1.25 else native
    assert chosen==z['selected_path'] and np.isclose(ratio,z['speed_factor'],atol=1e-12)
    report={'passes':True,'sources':M.sources(),'clean_children':count,'selected_path':chosen,'batch_speed_factor':ratio,'median_batch_seconds':values,'scope':'dropout0 real-prefix batch; no full-epoch or production-dropout stability claim'}
    M.save(M.CACHE/'runtime_lead_checks.json',report);M.save(M.PAPER/'b91_runtime_lead_checks.json',report)
    M.PAPER.joinpath('b91_runtime_lead_checks.md').write_text('# B.91 independent artifact/timing audit\n\n'+json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('PASS: B91 independently audited21 clean exits, medians and7-rule objectives')


if __name__=='__main__':main()
