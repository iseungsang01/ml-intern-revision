"""Compact source-defined context correction, preserving original Claude draft."""
import datetime,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CACHE=ROOT/'data/.claude_posteval_lead_20261123'
sys.path.insert(0,str(HERE.parent/'posteval_claude'))
import evidence as E
import agent

def run():
    CACHE.mkdir(exist_ok=True);assert not (CACHE/'started.json').exists(),'No blind paid replay'
    result,binding=E.gate();original=E.read(ROOT/'docs/paper/posteval_claude_20261123.json')
    existing=list((E.CACHE/'agent').glob('*.request.json'));assert len(existing)+4<=16
    definition={
        'formula':'For each initialization: 1 - sum((TX-y)^2)/sum((TH-y)^2) on identical rows; reported point is mean across initializations. Positive means fast-enabled TCN has smaller MSE than history-only TCN. Negative means larger MSE. This is not TCN versus old LSTM/two-stage estimator.',
        'models':'TX=TCN with BES/ECEI snapshots plus CES history; TH=same TCN architecture with slots0:15 zeroed (13 diagnostic slots plus2 already-zero MC slots). Neither is the manuscript GP/tree TX. No NBI increment is isolated here.',
        'rules':'D10/D50/D100/D300/D650/D1000 are CES-history availability-delay rules in milliseconds, NOT forecast horizons. Source permits3ms grid tolerance but always strict past. Fast channels available through current timestamp.',
        'S':'CES anchor frozen according to the registered20-40ms gap rule; not generic random missingness.',
        'K10':'Retain every10th nominal CES frame, always strict-past visibility; this rule never selected the model.',
        'targets':'F=current-frame reported CES target. L5=future-five-frame report median with required observations, offline label only. Not independent physical temperature truth.',
        'support':'D10/F_full uses all eligible F rows in model prediction support. D10/F uses the registered common mask. They are different row sets. Do not assume common clusters across all four validation designs exist.',
        'controls':'Main paired skill is computed before supplemental trees. Supplementary TREE_TX is a different estimator/budget; supplementary comparison supports may differ.',
        'inference_limits':'All four designs and both scopes must be retained. Original four-stage reviewer uncertainty about this sign is resolved by source, not a new experiment. Changed significance alone is not evidence that effects differ.'}
    fragments={}
    ranges={'ces_prediction/experiments/b92_reduced/common92.py':(29,31),
            'ces_prediction/experiments/b92_reduced/readout92.py':(37,47),
            'ces_prediction/experiments/b85_unified/readout85.py':(13,26),
            'ces_prediction/experiments/b85_unified/data85.py':(94,110)}
    for rel,(a,b) in ranges.items():
        p=ROOT/rel;fragments[rel]={'sha256':E.sha(p),'excerpt':'\n'.join(p.read_text(encoding='utf8').splitlines()[a-1:b])}
    data=E.read(E.CACHE/'evidence_complete.json');data['context']['verified_definitions']=definition;data['context']['source_fragments']=fragments
    data['reviews']['original_final']=original['claude_proposal']
    E.save(CACHE/'evidence.json',data)
    E.save(CACHE/'started.json',{'time':time.time(),'original_binding':binding,'prior_calls':len(existing),'max_additional_calls':4,'automatic_training':False,'script_sha256':E.sha(HERE/'review.py')})
    task=(HERE/'PLAN.md').read_text(encoding='utf8')+'\nRead /context/verified_definitions first; source checks resolve the old unknown sign. Read all8 compact design/scope tables. Correct the original final plan. Reject cross-design cluster intersection unless feasibility is established, and do not infer changes from significance labels alone. B98 has very few nonoverlapping clusters; dropping overlapping VAL rows does not make it an independent test. Distinguish fixed-budget result from architectural inferiority. Keep at most3 concrete candidates and one immediate action, no automatic experiments. Each field one concise Korean sentence. Avoid another proposal just to verify a sign now supplied and source-bound.'
    plan=agent.run('lead_context_correction','claude-opus-5-5','plan',task,data,E.API.credential(),folder=CACHE/'agent',max_rounds=4)
    assert E.gate()[1]==binding
    calls=[E.read(p) for p in sorted((CACHE/'agent').glob('*.response.json'))]
    output={'status':'lead_context_corrected_exploratory_plan','original_audit_binding':binding,'verified_definitions':definition,'source_fragments':fragments,'claude_proposal':plan,'automatic_training':False,'original_draft_preserved':True,'api_usage':[r['response'].get('usage',{}) for r in calls],'total_substantive_calls':len(existing)+len(calls)}
    E.save(ROOT/'docs/paper/posteval_lead_review_20261123.json',output)
    lines=['# 검산 후 연구 계획 보완 검토','',
           '원래 네 단계 Claude 초안은 보존했다. 코드에서 확인한 skill 부호·모델·가용 규칙을 제공해 Opus가 후속 계획을 보완했다. 아래 제안은 TEST-informed 탐색이며 새 학습은 실행하지 않았다.','',
           '**비교 정의:** skill은 초기화별 `1 − MSE(TCN 진단 포함) / MSE(TCN 이력만)`의 평균이다. 양수는 진단 포함 모델의 MSE 감소, 음수는 증가를 뜻한다. 과거 LSTM 2단계 estimator와의 비교가 아니다. D 규칙은 미래 예측 시간이 아니라 CES 이력의 가용 지연이다.','',
           '## 보완된 연구 계획','',plan['summary'],'','**바로 다음 단계:** '+plan['immediate_next_step'],'']
    for c in sorted(plan['ranked_candidates'],key=lambda c:c['priority']):
        lines+=['### '+str(c['priority'])+'. '+c['title'],'']
        for k,label in [('hypothesis','가설'),('existing_work_overlap','기존 작업과 중복'),('minimal_experiment','최소 실험'),('train_val_only_development','개발 범위'),('independent_confirmation','독립 확인'),('falsifier','반증 측정'),('stop_rule','중단 기준'),('positive_sentence','양의 결과 문장'),('negative_sentence','음의 결과 문장'),('timing_basis','시간 근거')]:lines+=['- '+label+': '+c[k]]
        lines+=['']
    lines+=['## 지지되지 않는 주장','',*['- '+s for s in plan['claims_not_supported']],'','## 검산된 조건별 paired skill','',
            '두 범위와 네 검증 설계를 구분한다. 아래는 현재값 전체 support와 지연 조건의 예시이며, 전체 등록 조건은 b94_methods.md/json에 있다. point [95% cluster interval], % 단위. F/L5는 보고값 표적이며 구간은 재학습/모델선택 불확실성을 포함하지 않는다.','',
            '| Scope | Design | Cell | Skill % [95% interval] | 0 제외 |','|---|---|---|---|---|']
    for scope,s in result['scopes'].items():
        for d,record in s['designs'].items():
            for key in ('D10/F_full','D650/L5'):
                z=record['cells'][key]['skill'];lo,hi=z['cluster']
                lines += [f"| {scope} | {d} | {key} | {100*z['point']:.1f} [{100*lo:.1f}, {100*hi:.1f}] | {z['excludes_zero']} |"]
    (ROOT/'docs/paper/posteval_lead_review_20261123.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    E.save(CACHE/'status.json',{'state':'complete','time':time.time(),'total_substantive_calls':output['total_substantive_calls'],'report':'docs/paper/posteval_lead_review_20261123.md'})
    print(json.dumps({'lead_correction_complete':True,'total_substantive_calls':output['total_substantive_calls']}))

if __name__=='__main__':
    try:run()
    except BaseException as e:
        E.save(CACHE/'status.json',{'state':'failed','time':time.time(),'error':repr(e)});raise
