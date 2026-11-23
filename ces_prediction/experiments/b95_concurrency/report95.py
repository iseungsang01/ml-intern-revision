"""Publish runtime evidence only; no observations, predictions or training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];CACHE=ROOT/'data/.b95';PAPER=ROOT/'docs/paper'

def main():
    base=json.loads((CACHE/'decision95.json').read_text());audit=json.loads((CACHE/'audit95.json').read_text())
    assert audit['all_pass'];rows=[]
    for n in (2,3,4,6):
        cc=[c for c in base['cohorts'] if c['workers']==n]
        rows.append({'workers':n,'cohorts':len(cc),'all_clean':all(c['clean'] for c in cc),
                     'wall_seconds':[c['wall_seconds'] for c in cc],
                     'jobs_per_minute':[60*c['throughput'] for c in cc],
                     'peak_gpu_mib':max(c['peak_gpu_mib'] for c in cc)})
    refs=[c['throughput'] for c in base['cohorts'] if c['workers']==2 and c['clean']]
    six=[c['throughput'] for c in base['cohorts'] if c['workers']==6 and c['clean']]
    report={'registration':'PREREGISTRATION_B95.md; addenda1/2','selected_workers':audit['selected_workers'],
            'full_epoch_cohorts':rows,'six_worker_robust_ratio_to_fast_reference':min(six)/max(refs),
            'normal_exit_children':sum(c['workers'] for c in base['cohorts'] if c['clean']),
            'scope':base['scope'],'test_accessed':False,'audit':audit,
            'limits':['Observed cohort timings, not predictive confidence intervals or a whole-suite acceleration estimate.',
                      'Reference timing varied substantially; the registered decision compares the slower candidate cohort with the faster reference.',
                      'Memory feasibility alone does not establish useful throughput. Numeric fit code and fixed budgets remain unchanged.']}
    (PAPER/'b95_concurrency.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    lines=['# B.95 whole-workload concurrency qualification','','Source: `data/.b95/decision95.json`, `audit95.json`, `production_policy95.json` and preserved worker/exit files. No TEST scores were read.','',
           '| Global workers | Cohort wall seconds | Jobs/minute | Peak GPU MiB | All normal exits |','|---|---|---|---|---|']
    for r in rows:lines.append('| '+str(r['workers'])+' | '+', '.join(f'{v:.2f}' for v in r['wall_seconds'])+' | '+', '.join(f'{v:.3f}' for v in r['jobs_per_minute'])+' | '+str(r['peak_gpu_mib'])+' | '+str(r['all_clean'])+' |')
    lines+=['',f"Selected global capacity: **{audit['selected_workers']}**. Six workers were feasible, but their slower repeated throughput divided by the faster two-worker reference was {report['six_worker_robust_ratio_to_fast_reference']:.4f}, below the registered 1.10 threshold.",'',
            'The extra four-worker tie confirmation was cancelled before launch: its already-observed throughput bounds the minimum of any future pair below the required threshold. The mathematical certificate is preserved in `tie_cancelled95.json`; no fictitious extra cohort was reported.','',
            'The optimized continuation reuses two verified complete GPU pilot fits, all three clean smokes and prepared caches. Two author-interrupted partial attempts were archived; only those jobs restart with the original seed/budgets. The duplicate CPU confirmation queue stays stopped. See B.94 addenda2/3 and `b94_status.md`.','',*report['limits']]
    (PAPER/'b95_concurrency.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'selected_workers':report['selected_workers'],'normal_exit_children':report['normal_exit_children']}))

if __name__=='__main__':main()
