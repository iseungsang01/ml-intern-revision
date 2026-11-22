"""Author-authorized redacted literature scouting through the official Claude API.

No shot arrays, unpublished result values, identifying information or credentials
are placed in prompts. No experiment or GPU operation is performed. Existing
responses are reused. Run from repository root with py; at most two requests run
concurrently. Source-level method leads require independent verification.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ces_prediction/experiments/b85_unified'))
import claude_review85 as API

CACHE = ROOT / 'data/.method_scout_20261122'
REPORT = ROOT / 'docs/paper/method_scout_claude_20261122.md'
MODEL = 'claude-opus-4-6'
CONTEXT = """Research task, with no identity, private arrays or unpublished performance values:
Estimate reported ion temperature on one pedestal-top charge-exchange spectroscopy
chord every 10 ms, causally, using past charge-exchange values and current/past BES
and ECEI samples. Real existing data are on a 10-ms grid with missing measurements.
No current or future CES value is available to the estimator at its own timestamp.
Inputs may contain the last received CES value, its age, and a has-observation flag.
Current registered baseline is one direct two-head causal LSTM trained across seven
availability conditions, with an identical history-only twin. Heads predict current
reported CES value F and the median of at least three values in the next five nominal
frames L5. L5 is an offline supervision target ONLY, not an input or independent truth.
A cadence condition (every tenth frame) is excluded from training/validation and used
to test availability transfer. Four validation designs are reported side by side:
random whole-file, clustered files, rolling origin and chronological files.
Earlier GP-plus-residual regression was tied to availability conditions; no numerical
outcomes are supplied. At inference only causal information may be used, with small
memory and a 10-ms processing budget on modest hardware. Data lack raw CES spectra,
per-frame fit errors, equilibria and per-shot ECEI settings. No new data are presumed.
The desired contribution can be strong useful pedestal diagnostic performance or a
testable physical use; architecture novelty is not required. Never claim physical
temperature recovery from sensor-derived targets alone. Acknowledge RTCAKENN and
Diag2Diag as existing diagnostic reconstruction/missing-input lineage. No journal
acceptance estimate, synthetic experiment, or unregistered training is requested.
"""

THEMES = {
 'dynamics': 'Irregular/missing-observation models: decay and time-aware recurrent models, continuous-time latent dynamics, controlled differential equations, recurrent state-space models, filtering/measurement updates, learned-gain or Kalman alternatives. Include simple credible models as well as modern ones.',
 'representation': 'Multimodal temporal representation: causal convolutions, diagnostic-specific encoders and fusion, masked or contrastive pretraining, teacher/student distillation, temporal super-resolution, structured state-space sequence models, compact causal attention where justified. Separate a task-ready method from a fashionable architecture name.',
 'observation': 'Observation and target modeling: heteroscedastic or heavy-tailed sensor likelihoods, robust losses, outlier/quality mixtures, non-random missingness, uncertain labels, calibration, causal conformal/selective prediction, reliability weighting and identifiability limits. Do not invent unavailable fit-error labels.',
 'transfer_physics': 'Generalization and physical use: causal system identification, latent/linear/Koopman dynamics, physics-informed hybrids with identifiable constraints, simulation-free self-supervision, domain generalization, online normalization/adaptation and change-point-aware state estimation. Explain what could reveal useful pedestal physics versus merely reproduce sensor artifacts.',
}
FAMILY_MAP = {
 'dynamics':['GRU-D','causal Neural CDE','ODE-RNN / latent ODE','continuous recurrent unit (CRU)','mTAN','KalmanNet / learned filtering gain','T-LSTM','decay plus linear/GP baseline'],
 'representation':['causal TCN','S4D / diagonal SSM','Mamba / selective SSM','modality-specific encoders and fusion','masked time-series pretraining','teacher-student distillation','temporal contrastive learning (TS2Vec)','compact causal attention'],
 'observation':['faithful heteroscedastic regression','beta-NLL','Student-t likelihood','EnbPI / time-series conformal','missing-data-augmentation conformal','inlier/outlier mixture likelihood','selective abstention','propensity/IPW weighting'],
 'transfer_physics':['Koopman latent-linear dynamics','S4/Mamba (overlap with representation)','Koopman/Granger exploratory analysis','online Bayesian last-layer adaptation','latent ODE/SDE (overlap with dynamics)','IRM/DANN domain regularization','online change-point-aware filtering','temporal self-supervised consistency'],
}


def prompt(theme):
    return CONTEXT + '\nPerspective: ' + THEMES[theme] + """
Use the web_search tool at least once to locate PRIMARY papers or author-maintained
official repositories, preferably surveys only as routes to original methods. Search
at most three times; do not add user location. Propose 8 distinct useful method families
(up to 10 if essential), not 8 slight variants of one architecture. For each provide:
name; canonical primary paper URL and official code URL if found; mechanism in one
sentence; concrete adaptation to this pedestal nowcast; whether current data suffice
or what extra data are required; likely inference/resource burden (qualitative, not an
invented benchmark); one minimal falsifiable comparison; main failure/leakage risk.
Use one compact table, around 70 words per row, then top three priorities and two methods
to avoid/defer. Mark each citation as SEARCH-VERIFIED versus UNVERIFIED LEAD; distinguish
method evidence from our application inference. Do not claim untested performance,
SOTA, physical discovery or that a named method solves this dataset. Prefer roughly
1800-2600 words maximum and finish rather than writing an implementation tutorial.
"""


def blocks_text(content):
    return '\n\n'.join(b.get('text', '') for b in content if b.get('type') == 'text')


def collect_sources(content):
    found = []
    def walk(x):
        if isinstance(x, dict):
            if x.get('url'):
                found.append({k:x[k] for k in ('url','title','type') if k in x})
            for value in x.values(): walk(value)
        elif isinstance(x, list):
            for value in x: walk(value)
    walk(content)
    return list({x['url']:x for x in found}.values())


def search_outcomes(content):
    errors=[]
    def walk(x):
        if isinstance(x,dict):
            if x.get('type')=='web_search_tool_result_error':
                errors.append(x.get('error_code','unknown_search_error'))
            for value in x.values():walk(value)
        elif isinstance(x,list):
            for value in x:walk(value)
    walk(content)
    return errors


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')
    temp.replace(path)


def run_theme(theme, key):
    path = CACHE / (theme + '.json')
    if path.exists():
        cached=json.loads(path.read_text(encoding='utf8'))
        cached['search_errors']=search_outcomes(cached.get('content',[]))
        save(path,cached)
        return cached
    p = prompt(theme)
    request = {'model':MODEL,'max_tokens':5000,'thinking':{'type':'disabled'},
               'tools':[{'type':'web_search_20250305','name':'web_search','max_uses':3}],
               'messages':[{'role':'user','content':p}]}
    started = time.time()
    try:
        response = API.request('/messages', key, request)
        content = response.get('content',[])
        record = {'theme':theme,'prompt':p,'model':response.get('model'),
                  'usage':response.get('usage',{}),'stop_reason':response.get('stop_reason'),
                  'seconds':round(time.time()-started,2),'content':content,
                  'text':blocks_text(content),'sources':collect_sources(content),
                  'queries':[b.get('input') for b in content if b.get('type')=='server_tool_use' and b.get('name')=='web_search'],
                  'search_requested':True,'search_used':any(b.get('type')=='server_tool_use' and b.get('name')=='web_search' for b in content),
                  'search_errors':search_outcomes(content)}
    except Exception as exc:
        # API helper never exposes raw bodies/headers/keys. No repeated blind call.
        record = {'theme':theme,'prompt':p,'model':MODEL,'error':type(exc).__name__+': '+str(exc),
                  'text':'','search_requested':True,'search_used':False}
    save(path,record)
    print(f"Scout {theme}: search_used={record.get('search_used')} stop={record.get('stop_reason')} error={bool(record.get('error'))}",flush=True)
    return record


def synthesis(records,key):
    path = CACHE/'synthesis.json'
    if path.exists():return json.loads(path.read_text(encoding='utf8'))
    summaries='\n\n'.join('PERSPECTIVE '+r['theme']+'\n'+r.get('text','') for r in records)
    p=CONTEXT+"""\nYou are the adversarial synthesis reviewer for four method searches below.
Deduplicate into about 25-35 genuinely distinct useful families. Return in Korean:
(1) a compact deduplicated family map, reusing supplied primary URLs;
(2) top five NEXT comparisons ordered by scientific value per implementation/runtime cost;
for each state mechanism, why it might address availability transfer/pedestal value,
exact causal information allowed, a minimal falsifier and why it might fail;
(3) methods that need unavailable data, methods likely redundant, and three specific
questions that would overturn the current priority order. Do not choose based on any
unpublished experimental performance (none was supplied). Do not require novelty of
architecture or invent citations. Separate searched references from unverified leads.
Do not add a new training run, acceptance prediction, or physical-truth claim.
Keep the output concise enough to finish within 4000 tokens.\n\n"""+summaries
    response=API.request('/messages',key,{'model':MODEL,'max_tokens':4000,'thinking':{'type':'disabled'},
                                         'messages':[{'role':'user','content':p}]})
    record={'prompt':p,'model':response.get('model'),'usage':response.get('usage',{}),
            'stop_reason':response.get('stop_reason'),'content':response.get('content',[]),
            'text':blocks_text(response.get('content',[]))}
    save(path,record)
    return record


def render(records, syn):
    syn_complete=syn.get('stop_reason')=='end_turn'
    lines=['# Claude API 기반 pedestal nowcasting 방법 탐색','',
      '문헌·방법 탐색이며 실험 결과가 아니다. B.85 학습 파일이나 GPU 작업은 변경하지 않았다. '
      '공식 Anthropic API에 식별 정보·원시 자료·미공개 성능 수치 없이 방법론 요약만 보냈다. '
      'Claude의 SEARCH-VERIFIED 표시는 API 검색 결과에 연결되었다는 뜻이며, 부모 에이전트의 별도 원문 검증과 구분한다.','',
      '모델 제안과 적용 가능성·연산 부담은 추론이며 실제 pedestal 성능이나 10 ms 충족을 입증하지 않는다. '
      '현재 센서값과 이후 센서 수준은 독립적인 참 온도가 아니다.','',
      '## 원시 Claude 응답을 읽을 때의 교정 사항','',
      '- 아래 원문에 나오는 "easily within 10 ms", "3–10× slower", "microseconds", "calibrated uncertainty", "no leakage risk"는 이 자료에서 측정·입증한 결론이 아니다. 실제 구현과 CPU latency·보정 검사가 필요하다.',
      '- KalmanNet은 원래 부분적으로 알려진 동역학/관측 모델을 활용한다. 동역학 모델 없이 그대로 적용 가능하다는 원문의 표현은 과도하다. 새로 학습하는 모델을 붙이면 별도 가정과 비교가 필요하다.',
      '- CRU의 부모 별도 확인 원문은 [PMLR](https://proceedings.mlr.press/v162/schirmer22a.html), 공식 코드는 [Continuous-Recurrent-Units](https://github.com/boschresearch/Continuous-Recurrent-Units)다. 원시 응답의 boschresearch/CRU 주소는 확인된 공식 주소로 채택하지 않는다.',
      '- 현재 기준은 Gaussian NLL이 아니라 MSE다. beta-NLL은 variance head 없이 단순 손실 치환으로 적용할 수 없다. Faithful heteroscedastic regression 원 논문의 stop-gradient/최적화 설계와 별도 mean-freeze 변형을 혼동하지 않는다.',
      '- Conformal 구간은 조건 없는 물리적 보장이 아니다. CES/L5 residual은 해당 측정·분석 결과가 실제 도착한 뒤에만 calibration에 사용할 수 있다. 시계열 의존성과 가용 조건 이동 아래 포함률은 별도로 검사한다.',
      '- Mixture의 outlier 성분이나 abstention이 실제 fit 실패를 식별했다고 간주하지 않는다. 선택한 행의 오차 감소는 같은 coverage의 기준과 비교하며, IPW로 MNAR 문제가 식별된다고 주장하지 않는다.',
      '- Attention weight는 물리적 설명이나 인과 기여의 증거가 아니다. 구조가 causal이어도 전처리·정규화·사전학습·teacher 구성에 미래 정보나 test 파일이 들어가면 누출된다.',
      '- 빠른 진단은 전체 자료에서 10 ms stamp 표본이다. 이것으로 관측하지 않은 고주파 스펙트럼·ELM 사건·수송을 복원했다고 주장하지 않는다. CES 지연만으로 ODE/CDE가 더 적합하다고 결론내리지 않는다.',
      '- 10 ms 표본만으로 kHz 물리 모드를 식별할 수 없다. Koopman/SSM 고유값이 곧 물리 모드라는 해석, Granger 관계가 곧 물리적 인과라는 해석, 근거 없는 시간 가역성 제약은 채택하지 않는다. Collisionality·ELM phase 등 없는 검증 변수를 쓰는 제안은 추가 자료가 필요한 방법으로 분류한다.',
      '- 온라인 보정은 CES가 도착한 시각의 feature가 아니라 그 CES가 실제 측정된 시각의 cached feature/예측과 짝지어야 한다. RTCAKENN의 sub-ms 속도·parameter 수에 관한 원시 응답은 이 보고서의 확인된 사실로 채택하지 않는다.',
      '- 제안된 임의 성능 문턱, 수치적 연산비, 자료 규모 표현은 채택한 실험 설계나 확인된 자료 사실이 아니다. 사전학습도 fit 파일만 사용하며 held-out cadence와 test 파일은 선택에 사용하지 않는다.',
      '- 짧은 attention/TCN 창의 길이는 최대 CES 지연과 fast-input anchor 정보를 보유할 수 있는지 따로 확인한다. 제공되지 않은 평형·PDE 계수·fit 오차가 필요한 방식은 현재 적용 가능 후보와 분리한다.','',
      '## API 실행 기록','','| 관점 | 모델 | 검색 사용 | 입력 토큰 | 출력 토큰 | 종료 |','|---|---|---|---:|---:|---|']
    for r in records:
        u=r.get('usage',{})
        lines.append(f"| {r['theme']} | {r.get('model')} | {r.get('search_used')} | {u.get('input_tokens','?')} | {u.get('output_tokens','?')} | {r.get('stop_reason',r.get('error'))} |")
    u=syn.get('usage',{})
    lines.append(f"| adversarial synthesis | {syn.get('model')} | prior four searches | {u.get('input_tokens','?')} | {u.get('output_tokens','?')} | {syn.get('stop_reason')} |")
    query_count=sum(len(r.get('queries',[])) for r in records)
    error_count=sum(len(r.get('search_errors',[])) for r in records)
    lines += ['',f'검색 도구 요청 {query_count}회가 기록되었고, 그중 제한 초과 오류 {error_count}회가 기록되었다. 오류도 아래 관점별 목록에 표시한다. 반환된 URL은 검색 inventory이며 모두 독립 확인한 1차 출처라는 뜻이 아니다.']
    lines += ['', '정확한 프롬프트·원시 응답·검색 질의·URL·사용량은 ignored `data/.method_scout_20261122/*.json`에 보존한다. '
              '재실행은 캐시를 재사용하며 자동 유료 재시도는 하지 않는다. 키는 읽기 전용 credential helper가 지정 경로에서 실행 시 읽으며 출력·커밋하지 않는다.','',
              '## 종합 판단과 완결성','',
              '네 관점의 검색 응답은 모두 완결되었고 각 여덟 항목, 합계 서른두 제안 항목을 냈다. S4/Mamba·ODE 계열 등이 관점 사이에 중복되므로 이를 서른두 개의 독립 방법이나 독립 증거로 세지 않는다. GRU-D/T-LSTM, S4D/S5/Mamba 등도 구현 예산을 정할 때 상위 기전 단위로 다시 묶는다.','',
              '마지막 Claude 종합 응답은 '+('완결되었다.' if syn_complete else '출력 한도에서 중단되었다. family map은 나왔지만 상위 다섯 비교와 비판은 완결되지 않았다. 따라서 그 미완성 순위를 채택하지 않는다. 원문과 종료 상태를 캐시에 보존했고 유료 재생성을 하지 않았다.'),'',
              '아래는 완결된 네 검색과 명시한 교정을 반영한 진행자의 요약이다. 아직 새 실험을 실행하거나 우위를 입증한 것은 아니다.','',
              '| 해결할 문제 | 검토할 기전 | 현재 자료에서의 최소 비교 | 제한 |',
              '|---|---|---|---|',
              '| CES 가용 조건 전이 | learned decay RNN(GRU-D/T-LSTM), 연속 시간 관측 갱신(CRU/RKN류) | 동일 입력·분할·head에서 기존 direct recurrent model과 비교; 학습하지 않은 cadence 포함 | 규칙을 많이 외운 것과 관측 갱신을 일반화한 것을 구분; 선형 dynamics 가정과 calibration을 검증 |',
              '| 빠른 진단의 시간 표현 | causal TCN, 작은 S4D, 진단별 encoder/fusion | 같은 수용 정보·parameter/연산 예산에서 과거 CES만 쓰는 쌍둥이와 함께 비교 | 10 ms stamp 자료에 없는 고주파 정보는 생성하지 않음; 모든 전처리와 RF를 인과적으로 고정 |',
              '| 표적 잡음과 신뢰도 | Student-t/Huber, faithful heteroscedastic head, 검증된 selective prediction | 평균 오차·꼬리 오차·구간 포함률·같은 coverage의 기준을 분리 | 낮은 오차나 outlier mixture는 실제 fit 오류 식별 또는 참 온도 복원의 증거가 아님 |',
              '| 방전/시기 이동 | timestamp-aligned online last-layer update, shift-aware calibration | 실제 CES 도착 시점만 갱신하고 chronological 설계에서 frozen 기준과 비교 | 늦게 도착한 CES는 원 측정 시각의 cached feature와 짝지어야 함; MNAR/shift 보장 없음 |',
              '| 물리적 쓸모 | 관측 가능한 하강/회복의 감시, 예측 기여·위상 지연의 통제 비교 | 입력으로 고정한 사건·조건에서 감시 지연과 거짓 경보를 같이 검증 | Granger·latent pole·attention만으로 인과 수송이나 물리 모드 발견을 선언하지 않음 |','',
              '부모가 추가로 제안한 적용 아이디어는 anchor/current Siamese encoder 또는 ΔTi/log-ratio 모델이다. 마지막 CES의 원 측정 시각에서의 fast feature와 현재 fast feature, age를 함께 써서 변화량을 직접 학습한다. 이는 이 Claude 네 응답에서 확인된 독립 문헌 결과가 아니라 프로젝트별 설계 제안이며, 기존 tree의 anchor-feature 계보를 인정하고 별도 등록·비교해야 한다.']
    for r in records:
        lines += ['', '## 탐색 관점: '+r['theme'],'',
                  '후보 family: '+', '.join(FAMILY_MAP.get(r['theme'],['Claude 종합 표와 원시 캐시 참조']))+'.','',
                  '정확한 Claude 원문은 해당 관점 JSON의 `text`와 `content`에 보존했다. 검색 원문의 긴 문구를 보고서에 재게시하지 않으며, 아래 URL 목록은 검색이 반환한 자료의 inventory다. 이 중 2차 자료·잘못 연결된 자료도 있으므로 원 논문 검증과 구분한다.','',
                  '### 실제 API 검색 질의','', '```json',json.dumps(r.get('queries',[]),ensure_ascii=False,indent=2),'```','',
                  '### API가 반환한 검색·인용 URL','']
        for source in r.get('sources',[]):
            title=source.get('title') or source['url']
            lines.append('- ['+title.replace('[','').replace(']','')+']('+source['url']+')')
        if r.get('error'):lines.append('오류: '+r['error'])
        if r.get('search_errors'):lines.append('검색 도구 오류: '+', '.join(r['search_errors']))
        if r.get('stop_reason')!='end_turn':lines.append('종료 상태에 유의: 완결된 최종 응답으로 자동 간주하지 않는다.')
    REPORT.write_text('\n'.join(lines).rstrip()+'\n',encoding='utf8')


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    all_cached=all((CACHE/(t+'.json')).exists() for t in THEMES) and (CACHE/'synthesis.json').exists()
    key=None if all_cached else API.credential()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(run_theme,t,key):t for t in THEMES}
        completed={}
        for future in as_completed(futures):
            result=future.result();completed[result['theme']]=result
    records=[completed[t] for t in THEMES]
    if not all(r.get('text') for r in records):
        syn={'model':MODEL,'text':'일부 API 응답이 없어 종합 검토를 실행하지 않았다. 오류 기록을 확인한다.'}
    else:syn=synthesis(records,key)
    render(records,syn)
    print('Report written; no B85 files or GPU jobs modified',flush=True)


if __name__=='__main__':main()
