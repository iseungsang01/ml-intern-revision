"""New bounded Claude cause review; immutable B105 sources/results stay untouched.

Reads existing aggregate curves/MSE only, never raw traces, weights or predictions.
Consumer: docs/paper/b105_cause_review_20261124.md and HANDOVER.md.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CACHE=ROOT/'data/.b105_cause_review_20261124'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b85_unified'))
import claude_review85 as API

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,z):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,ensure_ascii=False,indent=2),encoding='utf8')

def packet():
    p=ROOT/'docs/paper/b105_merit.json';z=read(p);audit=read(ROOT/'data/.b105/audit.json')
    assert audit['all_pass'] and audit['report_sha256']==sha(p)
    curves={};file_counts={};tree_counts={}
    for u in z['units']:
        cases={}
        for label,folder in [('TCN35',ROOT/'data/.b105/confirm'/u),('TCN22',ROOT/'data/.b94/confirm'/f'{u}__1701__TX')]:
            m=read(folder/'metrics.json');h=m['history'];best=min(h,key=lambda e:e['val_loss'])
            cases[label]={'stop':m['stop'],'epochs':m['completed_epochs'],'wall_seconds':m['wall_seconds'],
                'best_epoch_one_based':best['epoch']+1,'best_training_objective':best['train_loss'],'best_VAL_objective':best['val_loss'],
                'curve_columns':['epoch_one_based','training_objective','balanced_VAL_MSE','learning_rate','updates'],
                'curve':[[e['epoch']+1,e['train_loss'],e['val_loss'],e['lr'],e['updates']] for e in h],
                'best_rule_aggregate_validation':best['validation']}
        curves[u]=cases
        manifest=read(ROOT/'data/.b105/confirm'/u/'manifest.json')
        file_counts[u]={k:len(manifest[k]) for k in ('train','val','test')}
        tree_counts[u]=read(ROOT/'data/.b105/trees'/u/'complete.json')['counts']
    table={};ci={}
    for name,c in z['cells'].items():
        unit,split,rule,target,pair=name.split('/')
        if split=='existing_test':continue
        key='/'.join((unit,split,rule,target));s=c['statistics'];ref=pair.removeprefix('TCN35_over_')
        row=table.setdefault(key,{'rows':c['rows'],'MSE_eV2':{'TCN35':s['MSE'][0]}})
        assert abs(row['MSE_eV2']['TCN35']-s['MSE'][0])<=1e-7
        row['MSE_eV2'][ref]=s['MSE'][1]
        if rule in ('D10','D650','S','K10') and ref in ('incumbent_outerfit','fair_specialized','fair_mixture','TCN22'):
            ci[name]={'point':s['point'],'cluster95':s['cluster']['ci95'],'zero_excluded':s['cluster']['excludes_zero']}
    paths={
        'model35':'ces_prediction/experiments/b105_merit/model105.py',
        'TCN_and_blocks':'ces_prediction/experiments/b90_methods/models90.py',
        'numerical_train_loop':'ces_prediction/experiments/b92_reduced/train92.py',
        'data22_actual_builder':'ces_prediction/experiments/b85_unified/data85.py',
        'anchor13_adapter':'ces_prediction/experiments/b88_anchor/data88.py',
        'fair_tree_training':'ces_prediction/experiments/b105_merit/trees105.py',
        'tree_summary_features':'ces_prediction/experiments/b73_targets/stage_trees.py',
        'causal_history_anchor_features':'ces_prediction/experiments/b75_availability/common75.py',
        'segment_rule_sampling':'ces_prediction/experiments/b92_reduced/prepare92.py',
    }
    source={k:(ROOT/v).read_text(encoding='utf8') for k,v in paths.items()}
    # Omit unrelated candidate architectures and CLI plumbing, preserving actual TCN path.
    source['TCN_and_blocks']='\n'.join(source['TCN_and_blocks'].splitlines()[:122])
    source['tree_summary_features']='\n'.join(source['tree_summary_features'].splitlines()[:96])
    source['causal_history_anchor_features']='\n'.join(source['causal_history_anchor_features'].splitlines()[67:201])
    e={'source_result_sha256':sha(p),'audit_passed':True,'scope':z['limits'],'file_counts_only':file_counts,
       'registered_statements':z['statements'],'allocation_gate':z['gate']['units'],'curves':curves,
       'all120_matching_support_MSE_cells':table,'fixed_overview_paired_intervals':ci,'tree_row_counts':tree_counts,
       'source_hashes':{k:sha(ROOT/v) for k,v in paths.items()},'source_code':source,
       'unavailable':['No per-row residual/MAE/bias/outlier stratification performed in this cause review.',
                      'No new35-channel history-only twin, no single-F-only TCN ablation.',
                      'No learned-weight/activation/gradient magnitude audit beyond original finite nonzero readiness gradient.',
                      'No new architecture tuning, refitting or TEST selection.']}
    save(CACHE/'evidence.json',e);return e

TASK='''You are a skeptical ML research lead asked WHY the trained TCN has higher MSE than the incumbent.
The author wants a substantive diagnosis, not another generic statement of failure or broad architecture menu.
Answer in Korean, about1800-2300 Korean words, with exact supporting source keys and a prioritized cause table.
Use only provided audited anonymous aggregates, stored optimization curves and actual code. No raw traces,
shot identifiers, credentials, author identity or private reasoning are supplied. No new fits/scoring authorized.
Do NOT invent residual distributions, bias, overfitting, calibration drift, effective information loss or bugs.
Separate VERIFIED fact, evidence-supported hypothesis, unidentifiable speculation and actual code defect.

Fixed research goal: pedestal-top current CES reported Ti nowcast every10ms. incumbent is causal GP(latest16
visible reports, row-local hyperparameters)+boosted residual on hand-engineered history/fast/anchor features.
Legacy LSTM+postcorrection is a DIFFERENT comparator and NOT verified here. F=current-frame report; L5=median
next5 reports, offline secondary target, not independent truth. Source-age D10...D1000 are NOT forecast horizons.
S is observed-gap availability; K10 is a held-out sparse cadence. Never conflate target with time availability.
Four prespecified units(one per validation design) and seed1701 ONLY; not full-fold or initialization robustness.
Existing TEST informed all follow-ups. No physical truth, architecture inferiority or journal/novelty claim.

The candidate TCN35 adds13 source-time-fast channels to old22; zero-initialized additional stem columns preserve
the seeded22 initial function and RNG. No old weights fine-tuned, fresh fits. B105 original tests prove finite
nonzero anchor gradients on actual readiness batch, causal source alignment, prefix/full-step parity, train-only
statistics, normal exits, frozen source/config/checkpoints. These checks reduce some implementation concerns but
are not a theorem of bug absence, nor evidence that anchor signals are used after learning. Uniform10ms nominal
grid has missing rows, so127-row receptive field is NOT always exactly1.27seconds. Held CES+age+flag at every row
means absence of explicit GP is not proof all past-CES information is inaccessible.

TCN35 budget3600seconds/40epochs/patience6 vs old22 budget1200, widths/dilations/loss otherwise same.
B104 already tried budget-only extension at one temporal unit: allFAST best checkpoints unchanged. This does
not prove everyB105 unit fully optimized. Distinguish budget_limited,epoch_cap,patience early-stop('converged'
in source means patience criterion, NOT mathematical convergence). Training objective includes negative-Ti
penalty/dropout/random per-segment rule and batch reduction; VAL is eval-mode pure MSE over each rule/head.
Therefore comparing training_objective directly with VAL_MSE is NOT a clean generalization gap estimate.

TCN loss jointly trains F/L5 and7availability regimes, with checkpoint chosen by balanced aggregate VAL loss.
Fair specialized trees use same INTERNAL TRAIN FILES as TCN, no automaticearlystop,max_iter100,per-rule/per-head;
fair mixture trees are also available (rule-balanced weights), so specialization cannot be the ONLY explanation.
Neural TRAIN includes all masked targets/cold-start cases; tree TRAIN uses common support masks, so same files
does NOT mean same training-row distribution. Reference outerfit trees use more fit files including innerVAL;
use fair tree comparisons to assess whether that difference alone explains gap, NOT causal-proof subtraction.
Engineered features/explicit GP/additive residual inductive bias vs direct raw-channel TCN is an important
candidate explanation. Determine what is present/absent from actual code, not assumptions.
No35history-only twin: poor incumbent-relative performance does not prove fast diagnostics intrinsically useless.

Task deliverables:
1. Lead with3-5 most likely causes, ranked by strength of AVAILABLE evidence, not fashionable modeling ideas.
2. Give MSE(TCN35),MSE(fair specialized),MSE(fair mixture),MSE(incumbent) for D10/F across all4units and at least
   one stale-history F and L5 example. Cite keys. Are increased fit files, specialty, time budget or source-anchor
   omission sufficient explanations? Identify which claims data weaken and which remain unresolved.
3. Explain why VAL gate passed while TEST-relative incumbent merit failed: ratio-of-means, different comparators,
   selection optimism/support and actual regime/head trade-offs. Distinguish expected math from distribution shift.
4. Does learned-anchor candidate improve over oldTCN22, and what cannot be attributed because budget changed?
5. Inspect actual model/loss/data code for definite bugs vs optimization/representation weaknesses. Do NOT label
   absent skip connection,zero-initialized anchor columns,finite receptive field or multi-head training a bug by
   themselves. _build_x_row is an unused legacy helper; actual build_segments computes dt_prev correctly.
6. Explicitly state what current aggregates CANNOT identify: e.g. systematic bias vs a few large residuals,
   noise-driven error, early segment concentration, extrapolation/calibration, task interference or optimization.
7. Propose at mostTHREE minimal diagnostics IN PRIORITY ORDER, with exact comparison/measurement and different
   predicted outcome under competing hypotheses. Prefer saved predictions/weights and CPU-only work BEFORE GPU;
   any new residual scoring/fit needs separate preregistration, no posthoc TEST optimization/claim of confirmation.
   One tightly specified GPU alternative only IF necessary. No execution, no giant candidate list, no promise.
8. Give a research decision: keep incumbent / pursue a narrow identifiable deficiency / stopTCN for now; separate
   accuracy objective from architecture novelty. Don't redesign paper framing or acceptance estimates.
'''

def call(stage):
    CACHE.mkdir(parents=True,exist_ok=True);e=packet();prompt=TASK+'\nEVIDENCE\n'+json.dumps(e,ensure_ascii=False,separators=(',',':'))
    model='claude-fable-5-2'
    if stage=='challenge':
        model='claude-opus-5-5'
        prior=(CACHE/'analysis.txt').read_text(encoding='utf8')
        # Challenge sees all numeric evidence and exact relevant code; fresh context.
        prompt+='\nFIRST FABLE ANALYSIS (unverified advice):\n'+prior+'''\nNow challenge that analysis, <=1100 Korean words. Prioritize false causal attribution, numeric/source errors, overfit/underfit claims, training-distribution mismatch, residual decomposition arithmetic and overstated physics. F and L5 have different row support/counts, so lower L5 MSE is not a controlled noise-variance comparison. Finite127-row context need not lose the latest CES value because that is explicitly carried at each timestep; long frozen-source S and sparse K10 may lose earlier CES values differently from a GP that explicitly queries16 historical observations, but prevalence/error attribution is not measured. Don't confuse global optimum with patience earlystop. Keep supported findings, correct unsupported ones. End with a final ranked explanation and three diagnostic priorities. No new fits; no author-approval flow needed for completing this review. Numerical evidence has120cells:56VAL+64TEST, not120independentreplications.'''
    req=CACHE/(stage+'.request.json');out=CACHE/(stage+'.response.json')
    if out.exists():
        assert read(req)['messages'][0]['content']==prompt,'Existing prompt mismatch';r=read(out)['response']
    else:
        assert not req.exists(),'Attempt exists; inspect without blind paid replay'
        assert len(list(CACHE.glob('*.request.json')))<2,'Two paid attempts only'
        payload={'model':model,'max_tokens':20000 if stage=='analysis' else 14000,
                 'thinking':{'type':'adaptive'},'output_config':{'effort':'low'},'messages':[{'role':'user','content':prompt}]}
        save(req,payload);start=time.time()
        try:r=API.request('/messages',API.credential(),payload)
        except Exception as ex:save(CACHE/(stage+'.error.json'),{'error':str(ex),'seconds':time.time()-start});raise
        save(out,{'response':r,'seconds':time.time()-start})
    text='\n'.join(b['text'] for b in r['content'] if b.get('type')=='text')
    (CACHE/(stage+'.partial.txt')).write_text(text,encoding='utf8')
    assert r.get('stop_reason')=='end_turn' and text,'Incomplete response preserved, not accepted'
    (CACHE/(stage+'.txt')).write_text(text,encoding='utf8')
    print(json.dumps({'stage':stage,'model':r['model'],'usage':r.get('usage'),'stop_reason':r['stop_reason']}))

def render():
    evidence=read(CACHE/'evidence.json');lines=['# TCN MSE 원인: Fable 상세 분석과 Opus 반론','',
        '기존 B.105 결과·학습 곡선·소스만 검토했다. 새 학습, 새 잔차 채점, 새 TEST 선택은 없다.',
        '아래 Claude 응답은 원인 가설이며 실험으로 입증된 인과 설명이 아니다.',
        '원본 결과 SHA256: `'+evidence['source_result_sha256']+'`.','',
        '## 작업자 최종 판정: 원인 가설과 확인된 차이를 분리','',
        '**두 Claude 응답의 아래 오류를 먼저 교정해 읽는다. 특히 Opus의 G=GRU 추론은 채택하지 않는다.**','',
        '1. 가장 유력한 가설은 GP/hold의 지역 수준을 보존한 잔차 학습과, 절대 온도를 직접 회귀하는 TCN의 귀납편향 차이다. '
        '회귀기는 GP와 이력 통계·fast 변화량을 입력으로 받고 GP 오차만 학습한다. TCN은 held Ti를 입력받지만 출력으로 그대로 더하는 경로가 없고 GP 추정값/동일 통계 요약도 입력받지 않는다. '
        '이는 확인된 구조 차이이며, 어느 요소가 MSE 격차를 만들었는지는 분리 실험 전에는 미확정이다.','',
        '2. 같은 내부 TRAIN 파일의 specialized 회귀기와 시나리오 혼합 회귀기에도 D10/F MSE가 낮게 남는다. '
        '따라서 기존 estimator의 추가 적합 파일이나 조건별 재적합만으로 격차를 설명하기 어렵다. '
        '하지만 학습 행 지원과 표현·손실 가중까지 같다는 뜻은 아니다.','',
        '3. 확인된 설계상 불일치는 checkpoint 선택과 평가 지원이다. 신경망은 cold-start 등을 포함한 전체 VAL로 checkpoint를 고르고 '
        '공통 지원에서 회귀기와 비교한다. V1 F의 선택 VAL은30396행, 공통 VAL은23214행이다. '
        '또한 자원 gate는14셀 평균의 비율이어서 D10 열세가 stale 조건의 VAL 개선으로 상쇄됐다. '
        'VAL 통과가 개별 현재값 조건의 우위나 통계적 비열등을 뜻하지 않았다. 이것은 통과 과정을 설명하지만 모델 오차의 단일 원인을 입증하지 않는다.','',
        '4. 과제 간섭·일부 큰 잔차·평균 회귀 편향·최적화 부족은 미측정 가설이다. TRAIN objective와 VAL MSE를 직접 빼서 과적합이라 할 수 없다. '
        '네 작업 모두 patience 조건으로 종료했지만 최적화 완료의 증거는 아니다. 입력 추가와 예산 연장이 동시에 달라져 기존 TCN 대비 개선도 원인을 분해할 수 없다.','',
        '### Claude 응답의 독립 교정','',
        '- **G는 이 결과표에서 인과 GP, H는 hold다.** `b105_merit/readout105.py:specs`는 inputs/{rule}.npz의 문자열 필드 G/H를 비교자로 지정하고, '
        '`arrays`가 그 필드 값을 직접 읽는다. `b75_availability/common75.py:file_inputs`는 G를 predict_gp_causal로 만든다. '
        '`models90.py`의 방법 코드 G(GRU)는 별도 namespace이며 이번 G 결과와 관계없다. Opus의 GRU 비교 추론은 폐기한다.','',
        '- **3600초 adapter는 실제로 확인됐다.** `b105_merit/train105.py:config`가 frozen train92의 config를 덮어쓰고 seconds=3600을 설정한다. '
        '등록·config/checkpoint hash와 별도 audit도 묶여 있다. 검토 요청에 adapter 본문이 빠진 것을 실제 실행 오류나 미해결 예산 불일치로 읽지 않는다.','',
        '- **S는 최신 관측 조건이 아니다.** 마지막으로 관측된 결측 이전 CES에서 source가 동결되는 조건이다. D10과 S를 함께 최신 정보 구간으로 묶은 문장은 부정확하다.','',
        '- **원본 MSE의 점추정은 행 가중이다.** MSE=평균오차 제곱+오차분산 분해도 동일한 행 가중을 써야 한다. '
        'cluster bootstrap으로 구간을 만든다는 이유로 점추정을 같은 비중의 군집 평균으로 바꾸면 다른 estimand가 된다. Opus의 군집 가중 강제 조언은 채택하지 않는다.','',
        '- rule_for는 CRC32 기반 modulo7의 결정적 규칙 배정이다. 균형 표집을 의도하지만 매 epoch의 실제 행·파일 노출이 정확히1/7이라는 뜻은 아니다.','',
        '- 127행 문맥은 불규칙한 격자에서 정확히1.27초가 아니다. 최신 CES는 매 행 held 입력으로 있으므로 무조건 잃지 않는다. '
        '긴 S 동결/K10의 더 오래된 CES 이력 차이가 실제 오차를 만들었는지는 별도 측정이 필요하다.','',
        '- anchor weight norm이나 0 치환 민감도만으로 학습된 원인 기여를 단정하지 않는다. 0 치환은 입력 결합분포를 바꿀 수 있으며 의존도 점검에 한정한다. '
        '동일 업데이트 지점의 개선 비율도 anchor 효과와 학습 예산의 인과 분해가 아니다.','',
        '### 다음 진단의 최소 순서 — 아직 실행하지 않음','',
        '1. 저장된 VAL 예측의 동일 행에서 signed bias, residual variance, paired 초과 제곱오차와 꼬리 점유율을 분해한다. '
        '꼬리 행 선택은 모델 공통 기준과 각자 기준을 구분한다. 초기 구간/관측 age별 집계를 함께 보되 오차 위치가 특정 기제를 증명한다고 해석하지 않는다.','',
        '2. 저장된 VAL 곡선과 예측으로 F/L5·가용 조건·선택 지원을 분리한다. '
        '지금 checkpoint가 현재값 정확도를 희생하는지와, 원래 전체 지원/공통 지원 차이를 확인한다. '
        'TEST로 checkpoint나 gate를 다시 선택하지 않는다.','',
        '3. CPU 진단이 직접 회귀의 넓은 오차를 지지할 때만, 명시적 hold 또는 GP 잔차 출력 한 변형을 별도 등록해 검증한다. '
        '현 결과만으로 새 GPU 적합을 시작하지 않는다. 이 검토의 권고는 기존 estimator 유지이며, 광범위한 TCN 탐색을 재개할 근거는 없다.','',
        '위 진단은 새 과학적 분석이므로 별도 사전등록 후 수행한다. 현재는 원인 가설 검토만 완료했다.','',
        '## 공식 API 실행 기록','',
        '| 단계 | 모델 | 종료 | 입력 토큰 | 출력 토큰 |','|---|---|---|---|---|']
    for stage in ('analysis','challenge'):
        r=read(CACHE/(stage+'.response.json'))['response'];u=r['usage'];assert r['stop_reason']=='end_turn'
        lines.append(f"| {stage} | {r['model']} | {r['stop_reason']} | {u['input_tokens']} | {u['output_tokens']} |")
    lines+=['','공식 Anthropic API에 익명 집계와 코드만 제공했다. 원시 trace·방전 식별자·개인정보·비밀키는 보내지 않았다.',
            '모든 요청·응답·usage는 ignored `data/.b105_cause_review_20261124/`에 보존한다. 아래 모델 응답에는 private reasoning을 포함하지 않는다.',
            '', '## Fable 상세 분석','',(CACHE/'analysis.txt').read_text(encoding='utf8'),
            '', '## 독립 Opus 반론·수정','',(CACHE/'challenge.txt').read_text(encoding='utf8')]
    (ROOT/'docs/paper/b105_cause_review_20261124.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Rendered completed two-model cause analysis; no new scientific fits/scores')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('packet','analysis','challenge','render'));a=p.parse_args()
    if a.stage=='packet':
        e=packet();print(json.dumps({'cells':len(e['all120_matching_support_MSE_cells']),'curves':len(e['curves']),'source_result_sha256':e['source_result_sha256']}))
    elif a.stage=='render':render()
    else:call(a.stage)
