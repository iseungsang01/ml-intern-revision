"""Author-authorized source-only Claude review; never sends observed arrays."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ces_prediction/research'))
import claude_lean_scout_20261123 as S
S.CACHE=ROOT/'data/.b98/review'

if __name__=='__main__':
    S.CACHE.mkdir(parents=True,exist_ok=True)
    prompt='''Review a registered scalar physics-motivated nowcasting pilot BEFORE fitting. No execution, measurements, identity, unpublished scores or credentials are supplied. Find concrete blocking implementation errors, time-causality/label leakage, wrong ODE basis, invalid inference or independent-audit gaps. It uses real TRAIN/VAL only, no outer TEST. Existing cached scalar diagnostics and injected power are uncalibrated proxies. Do not propose more model searches, acceptance estimates or source changes to frozen training code. No web search is necessary. Rank at most5 actionable findings, distinguish actual errors from disclosed limitations, final answer under800words.\n'''
    files=['ces_prediction/experiments/PREREGISTRATION_B98.md',
           *['ces_prediction/experiments/b98_physics/'+n for n in ['model98.py','data98.py','run98.py','lead_checks/recurrence98.py','lead_checks/readout98.py']]]
    for f in files:prompt+='\nFILE '+f+'\n'+(ROOT/f).read_text(encoding='utf8')
    r=S.call('implementation_review',prompt,'claude-fable-5-2',S.API.credential())
    print(json.dumps({'stop':r['stop_reason'],'model':r['model'],'usage':r['usage']}))
