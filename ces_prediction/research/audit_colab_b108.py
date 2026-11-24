"""Recompute T4 timing arithmetic from every normal-exit archive record."""
import hashlib,json,math,statistics,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'data/colab_b108';PAPER=ROOT/'docs/paper'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    archive=DATA/'b108_colab_results.zip';state=json.loads((ROOT/'data/.colab_connection/b108_remote.json').read_text());assert state['results_sha256']==sha(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        def read(name):return json.loads(z.read(name))
        summary=read('results/summary.json');groups=summary['groups'];assert len(groups)==12
        child_count=0;recomputed={};shapes={};max_mem={}
        for g in groups:
            assert g['all_pass'] and len(g['children'])==g['workers']
            expected=[]
            for record in g['children']:
                unit=record['unit'];folder='results/'+g['name']+'/'
                actual=read(folder+unit+'.exit.json');r=read(folder+unit+'.json')
                assert actual==record and record['result']==r and record['returncode']==0 and not record['timeout']
                assert r['passed'] and r['backend']==g['backend'] and r['unit']==unit and r['gpu']=='Tesla T4'
                assert r['dtype']=='float32' and r['dropout']==.1 and r['tf32'] is False and r['threads']==2
                assert r['cudnn_enabled']==(g['backend']=='cudnn') and r['timed_steps']==5 and len(r['latency_seconds'])==5
                assert all(math.isfinite(x) and x>0 for x in r['latency_seconds'])
                assert math.isclose(statistics.median(r['latency_seconds']),r['median_step_seconds'],rel_tol=1e-12)
                assert r['timed_end']>r['timed_begin'] and math.isfinite(r['reference_max_abs_error'])
                shapes[unit]=r['train_shape'];key=(g['backend'],g['workers']);max_mem[key]=max(max_mem.get(key,0),r['peak_reserved_MiB'])
                expected.append(r);child_count+=1
            span=max(r['timed_end'] for r in expected)-min(r['timed_begin'] for r in expected)
            rate=sum(r['timed_steps'] for r in expected)/span;assert math.isclose(rate,g['synchronized_steps_per_second'],rel_tol=1e-12)
            assert g['launch_to_exit_seconds']>=span;recomputed.setdefault((g['backend'],g['workers']),[]).append(rate)
        assert child_count==30 and set(recomputed)=={(b,w) for b in ('native','cudnn') for w in (1,4)}
        rows=[]
        for (backend,workers),rates in recomputed.items():
            assert len(rates)==3;median=statistics.median(rates);reported=summary['configurations'][backend]
            assert reported['eligible_all_15_children_clean'] and reported['workers'+str(workers)]['median_steps_per_second']==median
            rows.append({'backend':backend,'workers':workers,'median_steps_per_second':median,'three_repetitions':rates,'max_child_reserved_MiB':max_mem[(backend,workers)]})
    rates={(r['backend'],r['workers']):r['median_steps_per_second'] for r in rows}
    report={'all_pass':True,'normal_children':30,'groups':12,'archive_sha256':sha(archive),'bundle_sha256':state['archive_sha256'],
            'gpu':summary['gpu'],'python':summary['python'],'torch':summary['torch'],'cpu_count':summary['cpu_count'],'rows':rows,'train_shapes':shapes,
            'cudnn_over_native_same_worker_count':{str(w):rates['cudnn',w]/rates['native',w] for w in (1,4)},
            'sum_group_launch_to_exit_seconds':sum(g['launch_to_exit_seconds'] for g in groups),
            'limits':['Three fixed repetitions; medians are descriptive, no95% interval or inferential speed claim.',
                      'Single worker uses unitA; four workers use A-D with different padded lengths. This does not isolate pure concurrency scaling.',
                      'Repeated-batch qualification is not full-epoch throughput or model MSE. No exact local/T4 speedup was measured.',
                      'Independent audit checks normal-exit records,hash bindings and timing arithmetic; GPU-prefix/finite checks remain producer attestations.']}
    (PAPER/'b108_results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    text=['# Colab T4 runtime qualification','',f"{report['normal_children']} normal children, {report['groups']} clean groups. GPU {report['gpu']}, torch {report['torch']}, CPU count {report['cpu_count']}.",'',
          '| Backend | Workers | Steps/s,median3 | Maximum child reserved MiB |','|---|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['workers'],r['backend'])):text.append(f"| {r['backend']} | {r['workers']} | {r['median_steps_per_second']:.3f} | {r['max_child_reserved_MiB']:.1f} |")
    text+=['',f"Within-T4 cuDNN/native ratios: workers1={report['cudnn_over_native_same_worker_count']['1']:.3f}, workers4={report['cudnn_over_native_same_worker_count']['4']:.3f}.",'',*report['limits']]
    (PAPER/'b108_results.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps({'all_pass':True,'normal_children':30,'rows':rows,'ratios':report['cudnn_over_native_same_worker_count'],'elapsed_probe_seconds':report['sum_group_launch_to_exit_seconds']},indent=2))
if __name__=='__main__':main()
