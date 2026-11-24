# 인수인계 문서 (2026-11-22)

**GP 이론적 배경 보강(2026-11-24):** 한국어 HTML 이론적 배경 7절에 함수 사전·관측모형·조건부 평균/분산·Matérn-3/2·주변우도·가용정보·불확실성·GP+boosting 연결을 약7800자/8소절/9표시식/입력비교표로 추가했다. 기존 시간모형 개관은8절 유지. Rasmussen/Williams, Williams/Rasmussen, Roberts, Chilenski, Michoski, Friedman 인용을 연결하고 T0(CES)와 TX(CES+BES/ECEI)를 분리 설명했다. 원본 `research_reader/gp_background/section.json`, 서지/범위/검증은 동폴더README·sources·verification. Friedman 원문 새접근은 실패해 기존서지와 공식sklearn 수식대조 범위로 명시했다. 공식Fable 검토는 preflight에서 잔액부족으로 차단되어 유료요청/완료검토0이며 API상태/진단시도 기록을 남겼다. 기존5달러 상한은 증액하지 않았다. GP코드·학습·결과·영문PDF불변. 인용/MathML/모바일/오프라인/입력표 검증과 pytest89통과.

**TCN MSE 원인 상세 검토 완료(2026-11-24):** 저자의 요청으로 기존120개 VAL/TEST 집계·네 학습 곡선·소스를 Fable 상세 분석과 별도 Opus 반론에 제공했다. 두 응답 모두 end_turn. 산출물 `docs/paper/b105_cause_review_20261124.md`, 재생성 helper `research/b105_cause_review_20261124.py`, 모든 시도/usage ignored `.b105_cause_review_20261124/`. GP+잔차/요약 특징 대 직접 회귀의 귀납편향 차이가 유력한 가설이나 인과 미확정이다. 전체 VAL checkpoint 지원과 공통 평가 지원의 차이, 평균 gate의 조건 상쇄는 확인된 설계 차이다. Opus의 결과키 G=GRU 오독은 실제 readout/source로 G=GP,H=hold임을 확인해 폐기했다. 3600초 train105 adapter도 확인했고, cluster bootstrap이 행 가중 MSE 점추정을 군집 동일 가중으로 바꾸는 근거가 아님을 교정했다. 저장 VAL 잔차 분해·지원/과제별 곡선 점검을 우선 제안하지만 아직 등록·실행하지 않았다. 새 GPU/적합/잔차 채점/TEST 선택 없음. 기존 B105 frozen source/results 불변. pytest89 통과.

**L5 용어 확인·부록 편집 완료(2026-11-24):** 아래 최초 정의 검토의 ‘미반영’은 당시 상태다. 저자 후속 지시로 현재 한국어 HTML의 표적 설명을 짧게 줄이고 상세 정의를 접는 부록 A로 옮겼다. L5는 Lean과 무관한 프로젝트 표기다. 문헌 캐시254건/고유PDF253개의 추출기호 검색과 관련 표적 설명 확인에서는 같은 정의·표준 명칭을 찾지 못했다. Ross2026 preprint 4.2절에는 미래 구간 관측 중앙값으로 이진 레이블을 만드는 유사 사례가 있으나 동일 회귀표적은 아니다. 공식 Fable 용어 검토도 서술명 ‘후속 시간창 중앙값 표적’을 권고했다. 검색은 전체 재독/부재 증명 아님. `definition_audit/terminology_evidence.json`, `07_fable_L5_terminology.json`, 최신 `editorial_closeout.json`; 재생성 연결은 동폴더README. 기존 결과·등록·영문PDF·표적구현 불변, 새 적합/평가/GPU0. 공통 추가5달러 상한내7호출 누계추산3.294288달러. HTML 부록 링크/수식/모바일/오프라인 검증과 pytest89통과.

**정의 타당성 재검토 요청(2026-11-24):** 저자가 L5/임의정의 과다와 최신B105 TCN실패를 지적하여 별도 Opus비판→Fable반박을받았다. 읽기용보고서 `docs/paper/research_reader/definition_audit/index.html`, 원문/범위/검증은동폴더README/json. 공통권고는 현재보고값F중심,전방창L5보조화와물리적잡음제거해석철회이며 아직원고/프로토콜에적용하지않았다. L5는B105보다먼저정의되어이번TCN구제용이라고단정불가. B73의V1표적비교가이미있고 B105에도stale-F점수가있으므로새민감도요청전기존산출부터확인해야한다. B105 D10/F의incumbent대비TEST skill네개는음수; VAL은fair특화tree비교로다른기준. gate는14셀평균비율로조건별5%보장아님. 모델의일부'민감도없음/CPU가능/어느head상쇄'주장은확인범위와가설을구분했다. 이번추가2API정상종료,새적합/평가/GPU0,기존5달러상한안누계추산3.03128달러. 연구정의/등록/기존HTML본문은바꾸지않고피드백만저장.

**B.105 전체 완료(2026-11-24 04:30 KST,08:24 재확인):** 네 학습 exit0, 실제15.5–41.0분/12–40epoch. 공통 VAL 자원 게이트는 네 unit 모두 통과해 TEST까지 완료했다. 별도 audit512 paired 비교·20,000회 구간과 gate/source를 통과했고 게시 JSON/md hash도 독립 재확인했다. Fable 사후 검토 end_turn, supervisor exit0·완료 알림 CLI 접수 확인. S1=True,S2=False,S3=False; 현행 GP+부스팅 대비 TEST64셀 점추정 모두 음수, D10/F는 네 unit 모두 cluster 구간이0을 제외하는 음수다. 현재 후보의 교체 merit가 확립되지 않았으며 추가 fit은 없다. GPU0. 전체 결과 `docs/paper/b105_merit.json/md`, 검토 `b105_claude.md`, 상태 `b105_status.md`, ignored `.b105/audit.json`. Fable의 overview32셀 언급은 전체64셀과 구분하도록 작업자 보완을 붙였다. 기존 LSTM+후단 보정과의 비교는 여전히 manifest 미확인으로 unavailable이다. 아래 학습 중/준비 문단은 과거 경과다. 등록·실행 코드·raw prediction을 수정하거나 재학습하지 않는다.

**추가 Claude 근거·그림 검증(2026-11-24):** 저자의 추가 약5달러 요청으로 공식API 독립본문/공개원문검증(Opus), 실제집계그림+안전HTML카드검토(Fable), 교정근거재확인(Opus)3회 모두정상종료. 추산USD2.04744/상한5,청구서확인아님. 현재 `research_reader/index.html`의7개그림은원본plotPDF해시로결속한벡터SVG+한국어상세캡션+원본확대링크이며11개표는기존crop유지. EPED폭·높이안정성설명과무작위설계의중앙오차한정교정. 그림6가로막대는등가간격의cluster-bootstrap구간임을기존코드에서확인해초기'구간아님'판정정정. API에는개별방전trace제외,집계패널만제공;보호용으로가린축이름을모델이실제누락으로오판한제안은미적용. `docs/paper/research_reader/figure_verification/README.md`, LOCAL_FIGURE_QA.md, verification.json이최신근거/HTML해시. 본문교정은 integrate_longform.py에연결. 원본그래프·수치변경/적합/GPU/수락채점없음.

**한국어 HTML 장문·가독성 재작성(2026-11-24):** 저자가 개요와 작업 메모식 본문을 지적하여 `docs/paper/research_reader/index.html` 전체를 실제 논문 본문 약5만7천자로 교체했다. 기존 목차·NN비교→설계 흐름 유지, 결론 약3천4백자. Fable 장문 초안의 완결 필드와 호스트 서술을 합친 전체를 Opus로 검토하고 독립 원문/영문결과 점검을 반영했다. 추가 Fable 가독성 편집의 닫힌9교정과 정상 종료한 좁은 최종 검토도 기록했다. 표11개·그림7개를 원PDF에서 그대로 잘라 관련 소절에 배치했고,18–19쪽으로 밀렸던 TCN표도 포함했다. TCN시간표와GP시간표는분리. 본문문단수치4개초과0건(인용·식별자·수식·표제외), 수치나한글표기로회피하지않음. 최신입력은 `active_longform`이 있는 manuscript_sections.json, 재생성은 thesis_reader/integrate_longform.py→build_reader.py→verify_reader.py. 옛짧은통합기로돌아가지않는다. 전체자료재실험·GPU·새원고합격채점없음,영문PDF그대로. 장문·후속편집usage추산6.165712달러로최초30달러허용액내미사용분에서진행(원래심층/원고편집과합27.584164달러). 근거·교정·각API중단범위는 research_reader/longform/README.md와LONGFORM_QA.md,최종검증기록참조.

**B.105 GPU 네 개 학습 중(2026-11-24 03:19 KST):**1aee77b 사전등록/전체 코드 커밋·푸시 후03:16 durable suite 시작. 실자료 입력/모델 검증 통과, 공정한 내부 TRAIN CPU 회귀기64개와 CUDA smoke 정상 종료. 실제 최장16 segment forward/backward 메모리 검사로 네 reserved slot 허용.03:18:54 고정 V1/V2/V3/V4 네 production fit 병렬 시작,03:19:21 GPU74%·2133MiB·오류 없음. 독립 감시 task `B105_Merit_watch`, supervisor `B105_GPU_merit` Running과 알림 CLI 접수 확인. 기존 B94/B104 재실행이 아니다. `pipeline105.py status`, `.b105/{queue_status,supervisor_status,watch_status}.json`와 `confirm/*/process.log`에서 실제 상태 확인. 네 정상 종료 후 공통 VAL의 현행 estimator 대비 merit 판정, 통과 때만 후보 TEST, 독립 최종 audit·Fable 사후 검토까지 자동 연결. 아래 “실행 준비”는 과거 상태다. 실행 코드/등록 동결, source 수정 금지. 문서만 갱신 가능. UI config는 DPAPI 암호화했고 출력/전송/커밋 금지. pytest89·records 통과.

**B.105 실행 준비(2026-11-24, 저자의 학습·merit 확인 지시):** 계획을 실행으로 옮긴다. `PREREGISTRATION_B105.md`와 새 `b105_merit/`를 작성 중이며 코드/등록 커밋 전에는 적합하지 않는다. 후보는35채널 anchor TCN 하나, 고정 네 unit·seed1701 최대 네 fit. 같은 내부 TRAIN으로 적합한 specialized/mixture GP+부스팅을 먼저 준비하고, 같은 VAL 행의 비율 게이트가 네 unit에서 모두 통과할 때만 후보 TEST를 평가한다. 실패 시 추가 탐색 없음. 기존22TCN/B84 비교는 별도이며 과거 LSTM+후단 보정과 혼동하지 않는다. Fable 설계/구현 검토와 모든 시도 사용량은 ignored `data/.b105/review/`; 새 과학적 실행 상태는 `docs/paper/b105_status.md`, `data/.b105/queue_status.json` 등. `pipeline105.py prepare`→`register`가 공식 진입점이며 source/실자료 체크·CUDA smoke·독립 최종 audit가 자동 연결된다. 오늘 GPU 사용을 새로 승인받았으므로 아래 “오늘 새 학습 없음”은 과거 상태다. UI 인증 config는 출력/전송/커밋 금지.

**문헌·원고·HTML 통합 완료(2026-11-24):** `docs/paper/research_reader/index.html`이 절별 한국어 원고와 인용 근거의 진입점이다. 기존 HWP의 Dataset/수집기준/연도·장치 상태 분류/NN 선택·개조 이유 계층을 유지했다. 전체 보관 파일 중 공개 고유 출처 251개를 개별 검토(PDF 전체 제공41, 전체 추출문210; 중복2·기관 가이드1 별도)했으며 독립 연구 수나 모든 그림의 육안 확인을 뜻하지 않는다. 기준 기록은 `docs/paper/introduction_review/all_corpus_status.json`, 비용은 `cost_summary.json`. Fable 초안과 개별 Opus 감사 후 최종 통합문을 받아 정보시점·L5·온도식을 작업자가 교정했다. 영어 작업본은 `docs/paper/ppcf_revision/main_ppcf.pdf`22쪽, Results 본문은 기존과 동일하다. HTML에 Results10–17쪽을 내장했다. 새 적합/GPU/TEST 재평가 없음. 기존 grade3/novelty2는 이전 동결 PDF 평가로 최신 수정본 채점이 아니다. 추가 문헌/HTML 비용49.869956달러(상한50), 기존 원고 편집2.283684달러는 앞선 심층자문과 합쳐 최초30달러 범위 안. 전체 반환 usage 기준 추산71.288408달러이며 청구서 확인액은 아니다. 유료 호출은 종료했다. 무료 재생성·검증 배선은 문헌 폴더 README와 root README에 있다. 다음 과학 실험 우선순위는 아래 별도 TCN merit 계획을 따른다.

**최신 우선순위: TCN merit부터(2026-11-24, 저자의 내일 GPU 준비 지시):** `docs/paper/tcn_merit_plan_20261124.md`가 아래 P1/Stage0 우선순위를 대체한다. 공식 Fable 재검토 완료; 원문/모든 시도 사용량은 `tcn_merit_fable_20261124.md`, 원본 JSON의 기존 paired 비교 전체 표는 `tcn_merit_existing_20261124.md`. “직접 비교 없다”는 작업자의 앞선 설명은 오류: B94에는 GP+부스팅 대비가 이미 있다. 과거 LSTM+후단 보정과 구분한다. 오늘 새 GPU/적합/TEST 채점 없음. 기존 TCN 단순 연장은 종료; 기존 혼합 회귀기 B84(L5, V2 없음) 비교와 내부 TRAIN/VAL 공정성부터 확인한다. 조건부 후보는 35채널 anchor TCN 하나, 이전 고정 네 unit의 seed1701 최대 네 fit의 VAL 선별. 이는 등록/runner 준비 완료가 아니며 다음 세션에서 새 scorer·내부 TRAIN 회귀기 적합·독립 검증·사전등록 커밋 후 실행한다. 선별 실패 시 자동 확장/튜닝 없음. `research/tcn_merit_plan_20261124.py`는 최대 세 API 시도만, `research/render_tcn_merit_20261124.py`는 원본 구간 전사와 idx/mask 메타데이터 점검만 한다. 새 과학적 비교는 별도 등록 필요. GPU 예약 작업을 설치하지 않았고 B94/B104 큐를 재실행하지 않는다.

**B.104 완료(2026-11-24 01:11 KST, 최종 확인 02:36):** 여섯 학습 정상 종료, 예측24개와 네 조건의 독립 통계 검산, Fable/Opus 사후 검토까지 완료했다. supervisor exit0·자동 완료 알림을 확인했다. 진단 포함 FAST의 SHORT/LONG 가중치는 세 초기화 모두 동일하며, HISTORY1701만 LONG 체크포인트가 바뀌었다. 학습 연장으로 fast 증분이 회복되지는 않았다. 고정 초기화 구간과 초기화 혼합 구간의 해석을 구분해야 하며, 한 시간분할 unit의 탐색 결과다. 근거 `docs/paper/b104_budget.json/md`, `b104_claude.md`; 독립 최종 체크포인트 검증 `data/.b104/lead_completion.json` all_pass. 현재 GPU 유휴, 추가 학습 없음. 다음 준비 방향은 P1/Stage0 초안이며 새 등록·적합은 아직 하지 않았다. 아래 학습 중/준비 문단은 과거 경과다.

**심화 보고의 최신 문헌 정정도 적용:** DECISION 말미 NN-CES 전문 미확보 문장은 `fable_deep_review/NN_CES_CORRECTION.md`·`docs/reference/nnces_source.md`로 정정한다. 보관 전문은 이미 있었으며 특정 Fable 요청에 미제공된 사실과 구분한다. P1 우선순위는 유망한 준비 방향이지만 NN-CES 전문을 포함해 Fable이 다시 평가한 순위로 소급하지 않는다. Stage0 초안에 문헌 대비표 gate를 추가했고, 기존 조사 원문/과금 기록은 건드리지 않았다.

**후속 우선순위 보완(2026-11-24, 저자의 DECISION 확인 요청):** `docs/paper/fable_deep_review/DECISION.md`와12/13번 정정·METHOD_CHECKS·EXISTING_WORK를 직접 확인했다. B.104는 예산 혼동을 정리하는 현재 보조 실험으로 유지하며 중복/중단하지 않는다. 다음 본 방법론 우선순위는 source-time 일관 작은 강제응답 추정기에서 H/H+fast/H+NBI/H+NBI+fast를 같은 행에 두어 NBI를 조건으로 한 fast 증분을 구분하는 P1이다. 사건 갱신/재전파는 표준 구성요소이므로 독립 신규성 방향에서 내려 P1 내부 대조로 병합한다. B90 observer/robust-loss는 보류·미평가이지 실패한 결과가 아니다. 이후 후보는 고정 평균의 보고값 분산, 동결 인과 표현 탐침이다. `docs/paper/stage0_anchor_program_DRAFT.md`에 메타데이터-only Stage0 등록 전 초안을 작성했으며 새 적합·TEST 재채점은 실행하지 않았다. 이 초안은 margin/다중성/범위를 확정한 등록문이 아니다. B104 동결 실행 소스에는 손대지 않았다.

**NN-CES 원문·기억 정정(2026-11-24):** 사용자가 다시 지정한 `1-s2.0-S0920379625007148-main.pdf`는 기존 `data/literature/Lee2026NeuralNetworkbased.pdf`와 페이지별 정규화 추출문이 모두 같은 출판사 전문이다. 원문은 이미 있었고 과거 본문 검토도 있었다. 최근 '전문 미확보' 기록은 철회한다. 현재 원천·hash·보관 경로·본문 확인·API에 제공하지 않았던 범위는 `docs/reference/nnces_source.md/json`과 `docs/paper/fable_deep_review/NN_CES_CORRECTION.md` 참조. 다음 문헌 조사 전에 로컬 INDEX·PDF·이 기록을 먼저 확인한다. 삭제된 옛 지도나 옛 실험 결과는 복원하지 않는다. 프로젝트별 Claude 메모리도 같은 사실로 정정했다.

**공유 Git index 주의(2026-11-24):** 여러 pane이 같은 index를 쓰므로 후속 commit은 `git commit --only ... -- <명시한 파일들>`을 사용한다.8c7b129에는 B.104 상태 문서 외에 다른 pane이 동시에 stage한 fable_deep 연구 파일도 함께 포함됐다. 해당 내용은 수정/삭제하지 않고 보존했으며 이미 push된 이력은 바꾸지 않는다. B.104 동결 실행 소스는8dd55a6 그대로이고 GPU 학습은 정상이다. icml/발표 파일은 여전히 다른 작업 범위다.

**B.104 실제 GPU 학습 중(2026-11-24 00:36 KST):**8dd55a6 사전등록/전체 코드 커밋·푸시 후 prepare/register 실행. real-prefix smoke 네 개 exit0; 두 모델 모두 observer off/on의 두 epoch VAL과 best/final 전 파라미터가 bitwise 동일, snapshot 재검증 통과.00:29:59 HISTORY1701/1702/1703, FAST1701 네 경로 시작.00:36 각4epoch, GPU72–74%·3138MiB, 오류 없음. 뒤 FAST1702/1703는 슬롯이 비면 자동 실행. task `B93_GPU_B104_budget`(supervisor6500/queue17340 시작 관측), 감시 `B93_Research_B104` 알림 CLI 접수 확인. 실제 live 상태는 `.b104/queue_status.json/watch_status.json/dispatch_events.json`, 모델별 confirm/*/process.log; `pipeline104.py status`. Clock00:27:15→03:27:15 KST, 전 연구 마감은 불변. 이후24예측→네 endpoint 통계→별도 audit→명시적 json/md 게시→Fable/Opus 분석을 자동 연결했다. 새 실행·소스 수정하지 말고 먼저 live 상태 확인. 현 소스 frozen이므로 training/observer/statistics 코드 변경 금지. UI 인증 config 출력/전송/커밋 금지.89 tests·records 통과. untracked icml/fable_deep/발표 타 작업은 그대로 둔다.

**B.104 후속 준비(2026-11-24):** 저자의 계속 지시로 고정 V3_q3·기존 세 seed의 새 TCN FAST/HISTORY 여섯 경로에서 학습 예산 영향을 검사한다. Fable TRAIN/VAL 검토와 Opus 반론·전체 구현 검토 완료, 통계 코드는 Opus가 작성했다. 원래 numeric loop를 import하고 같은 경로에서1200s SHORT/최대3600s LONG을 저장한다.40epoch·earlystop·전처리·모델은 유지, 전역4개. 최대3h 새 연구 clock이며 옛 B94 clock을 바꾸지 않는다. 먼저 real-prefix smoke 네 개를 통과해야 full fit가 배정된다. 소스·등록 동결 후 `b104_budget/pipeline104.py prepare`→`register`.15초 감시·오류/단계 즉시·5분 로컬 알림. 상태는 `docs/paper/b104_status.md`, `.b104/queue_status.json/watch_status.json/dispatch_events.json`; UI 인증 config 출력/전송 금지. 이번 범위는 TEST-informed 한 unit 탐색이며 원래 네 설계 전체 성능 주장이 아니다. 원래 숫자·코드는 보존한다. 새 icml/fable_deep 연구·발표 untracked 파일은 타 작업이므로 건드리지 않는다.

**전체 분석 완료·lead 보완 확인(2026-11-23 23:31 KST):** 원래 Claude 네 단계는23:03:47 draft_ready로 종료했고 완료 알림 전달을 확인했다. 초안에 skill 부호/모델 정의 미확인과 가용 지연을 horizon으로 부르는 한계가 있어, 원본은 보존하고 `research/posteval_lead/{PLAN.md,review.py}`에서 동결 소스 정의·모든 scope/design을 제공해 Opus 보완을 받았다. 보완은23:29 완료, 원래10회+추가3회=13 substantive API calls로 기존16회 한도 안이다. 이후 원래 gate/hash를 다시 확인했고 셀별 구간 부호 분류를 원본에서 생성해 prose보다 우선하도록 했다. 최종 연구 제안은 `docs/paper/posteval_lead_review_20261123.md/json`, 수치 원본은 `b94_methods.md/json`, 이전 Claude 원문은 `posteval_claude_20261123.md/json`에 있다. `.claude_posteval_lead_20261123/lead_checks.json` 통과. 전체 학습/예측/집계/독립 감사/분석 완료 상태이며 현재 새 학습은 없다. 우선 후보는 학습 예산 영향의 한정 검증, 다음은 설계 내 support 기술·B98 cluster 수 feasibility다. 새 fit/readout은 별도 사전등록 전 실행하지 않는다. PK4.75에 해석 범위를 추가했다.

**최종 수치 감사·게시 완료, Claude 실제 분석 시작(2026-11-23 23:01 KST):** 원래 post audit가22:45에46 confirmation·136셀을 통과했다. B.102는 이후 게시 glob에 readiness 문서가 포함되어 멈췄고 로컬 알림은 즉시 전달됐다. B.103은 재계산/감사 재실행 없이 기존 감사·보존 binding을 확인해 methods JSON/md 두 파일만 게시했다. `.b103/published.json` all_pass와 원래 supervisor/queue normal exit0 및 Claude `E.gate()` true를 확인했다. `docs/paper/b94_methods.json/md`가 검증된 결과다. 새 watcher103도 전달 성공. 게시 큐가 완료 상태를 쓰기 전 초기 Claude 대기가 과거 failed queue를 읽은 race는 substantive 요청0을 확인한 후 전체 완료 gate가 true인 상태에서만 다시 시작했다. **23:01 Claude evidence_review 실제 request/response가 생성되었으며 현재 첫 단계 분석 중**이다. 감시는 `B93_Research_Watch103`, 공개 기록 `.b103/watch_status.json/watch_error.json`; watch_config는 인증정보라 출력 금지. 이후 evidence/adversarial/plan/final revision 및 lead review를 확인한다. 아래22:43 대기·이전 failed는 과거 관측이다.

**B.102 실제 검사 통과·감시 알림 전달(2026-11-23 22:43 KST):** postcheck-only 검사가 정상 종료했고 markdown 모든 지표 행·368예측·baseline/fit/selection·변경 없는 paired 통계를 확인했다. source425c9ad 실행 전 커밋/푸시, 테스트89·기록 검사 통과. task `B93_GPU_B94_postcheck102` supervisor31524/queue20612의 생존·새 heartbeat와 audit_post 단계를 확인했다. Claude는 새22:43 timestamp로 대기한다. **별도 task `B93_Research_Watch102`도 Running이며 watch_status의 notify_returncode0으로 실제 로컬 cvmux 알림 전달을 확인했다.**15초 poll·단계/오류 즉시·5분 상태 알림, 진단 없는 외부 평가 중단만1회 재시작한다. Claude 재시작/과금 replay는 감시기가 하지 않는다. `data/.b102/watch_config.json`은 로컬 UI 인증정보이므로 절대 출력/전송/커밋하지 않는다. 공개 상태는 watch_status/watch_error/restart_budget만 본다. 최종 post audit/Claude 결과는 아직 완료 주장 아님.

**B.101 최종 파일 검사 실패·B.102 준비(2026-11-23 22:34 확인):** B.101은136셀 계산과 report 저장, control/preservation 검사를 끝냈으나21:34 markdown hash lookup에서 `reports/b94_methods.md`와 Windows manifest key의 구분자 차이로 KeyError가 났다. 평가·Claude 모두 failed이며 아래 running은 과거 관측이다. 저자가 자동 복구/중간 보고를 명시 요청했다. B.102/addendum7·B102 addendum1은 계산 재생성 없이 manifest key를 메모리에서만 정규화해 남은 검사·원래 post audit를 수행한다. `b102_postcheck/watch102.py`는15초 생존/단계 확인, 즉시·5분 간격 로컬 cvmux 알림, 진단 오류 없는 외부 중단에 한해 평가-only 한 번 재시작을 맡는다. Claude 자동 재실행/과금 replay는 하지 않는다. 설정은 ignored `data/.b102/watch_config.json`으로 로컬 UI 인증정보가 있으므로 출력·API 전송·커밋 금지. 결과 상태는 `.b102/{validation,watch_status,watch_error,restart_budget}.json`과 원래 status93/Claude status를 확인한다. 새 학습/추론/bootstrap 및 마감 연장은 없다.

**B.101 실제 재개(2026-11-23 21:19 KST):** 공식 Fable 코드 검토(end_turn,6544 input/5304 output)를 받고 addendum1의 control·report/markdown·baseline/fit/selection hash 검사를 추가했다. 테스트89개 및 기록 검사 통과 후327785c까지 커밋/푸시했다. 준비 정상 종료,368예측과 실패 원본·hash 보존 확인. 새 task `B93_GPU_B94_readout101` supervisor26620/queue23272 생존·생성 시각·새 heartbeat와 readout 실행을 관측했다. Claude task도 새21:19 timestamp로 최종감사 대기 중이다. 복구 진행은 `.b94/readout.log`의 Repaired/Carried 행(총136,복구128), `.b101/{prepared,repair_checks,launch}.json` 및 status93로 본다. repair_checks는 복구 완료 후에만 생성된다. 아직 감사 통과나 최종 성능 확정 아님.

**21:04 검산 실패·B.101 복구 준비(2026-11-23):** B.100은 예측368개 검증을 통과했고 readout을20:28~21:04 수행했으나, original post audit가 main-pair metric 불일치를 발견해 중단했다. Claude는21:05 upstream 실패 gate로 멈췄으며 부분 TEST 결과를 받지 않았다. 원인은 `readout92.cell`의 neural TX와 supplementary tree TX가 `metrics['TX']`를 공유하여128/136셀에서 main metric을 덮어쓴 것이다. `skill`은 overwrite 전에 계산되지만 최종 재검산 전에는 검증된 결과로 취급하지 않는다. B.101/addendum6, `b101_readout_repair/repair101.py`는 실패 원본·예측 hash 보존 후 TREE_ namespace로 분리하고 덮어쓴 marginal metric만 원래 함수/RNG로 복구한다. 나머지 report 전체 equality와 원래 post audit를 필수로 한다. 새 학습/추론·마감 연장 없음. `data/.b101/`와 최신 `status93.py --cache data/.b94`를 확인한다. 아래20:18 running 문장은 당시 관측이다.

**실제 재개 확인(2026-11-23 20:18 KST):** B.100 준비 검증이 정상 종료했다. 독립 확인된confirmation46개와pilot 선택을 재사용하며, 완료 로그가 있는 예측344개(43묶음)는 hash로 고정했다. 중단 묶음의 기존6개 파일은 보존하고3묶음24개만 원래 predictor로 다시 계산한다. task `B93_GPU_B94_eval100`의 supervisor PID23360/queue PID4312 생성 시각·생존·heartbeat와 prediction 단계를 실제 확인했다. 새 학습은 없다. Claude task `B93_Claude_PostEval_20261123`도 Running 및 새20:18 대기 timestamp를 확인했다. 원래 마감은11월24일01:54:57 KST 그대로다. 현황은 `status93.py --cache data/.b94`, `data/.b100/{prepared,launch,prediction_resume}.json`, Claude `status.json`을 읽는다. `prediction_resume.json`은 전체 예측 검증 후에만 생긴다. 원래 readout/post audit/정상 종료 후에만 Claude 결과 분석이 시작된다.

**재부팅 후 평가 복구 준비(2026-11-23 20:13 KST):** 저자가18:24 일시중단 후 계속하라고 지시했다.20:06 OS 재부팅으로 기존 네 프로세스는 모두 소실되었다. 학습은 pilot12/12·confirmation46/46 완료 상태이고 예측350/368 파일이 보존되어 있다. 원래 scheduler는 TEST 이후 pilot gate를 다시 통과시키지 않으므로 그대로 재시작하지 않는다. B.100/B.94 addendum5와 `ces_prediction/experiments/b100_eval_resume/resume100.py`로 독립 fit 검증·로그가 있는 예측 묶음 보존·중단 묶음 재계산·원래 readout/post audit만 수행한다. 새 학습과 마감 연장은 없다. 복구 기록은 `data/.b100/`, live 상태는 `status93.py --cache data/.b94`. 기존 Claude worker는 새 supervisor/queue 생존 확인 후 재시작한다. 아래18:03 대기 및 이전 GPU 실행 문장은 당시 관측이다.

**Claude 기본 연동·평가 후 자동 정리 설정(2026-11-23 18:03 KST):** 저자는 주요 설계/해석/후속 연구 판단을 최대한 Claude API와 연동하라고 지시했다. AGENTS.md·CLAUDE.md에 선호를 반영했다. 숨김 일회성 task `B93_Claude_PostEval_20261123`과 worker 생존을 확인했으며 현재 원래 평가 최종 감사를 기다린다. `data/.claude_posteval_20261123/status.json`을 먼저 읽는다. 원래 정상 exit0+post audit+hash gate 후 Fable 근거 → Opus 반대 → Fable 계획 → Opus 최종 수정, 최대16 paid rounds로 제한한다. 설정 설명은 `docs/paper/posteval_claude_20261123_status.md`; 완료 초안은 같은 이름에서 `_status`를 뺀 md/json. partial TEST 성능은 보내지 않으며 새 연구는 TEST-informed 탐색·별도 사전등록 대상으로 정리한다. 실패 시 유료 호출을 무작정 재실행하지 않는다. 현재 평가/학습 코드는 바꾸지 않았다.

**수식 모형 구현·내부 판독 완료(B.98):** 강제1차 완화 M·이력 완화 H·정적 anchor 차분 L을 네 대표 unit의 TRAIN에만 적합하고 내부 VAL에서 모든 등록 규칙을 판독했다. 원자료·직접 재귀·미래/현재 CES 배제 검사를 통과했다. `docs/paper/b98_physics.md/json`, `b98_lead_checks.md` 참조. hold 대비 변화 상당 부분은 이력 완화이며 진단+NBI 강제항 추가 효과는 작거나 시간순 조건에서 불안정하다. M−H는 fast 단독 효과가 아니다. 군집 중복·작은 군집 수·다중 비교·재적합 불확실성 누락을 명시했다. outer TEST/L5/모형 채택은 하지 않았다. GPU 실제 재개는 아래 B.99를 따른다.

**GPU 실제 재개(B.99, 2026-11-23 15:41 KST):** 저자의 후속 재개 지시로 원래4-slot GPU 확인 학습이 실제 계산 중이다.23/46 완료를 재검증해 보존했고, 중단4개는 원본/hash를 보존한 뒤 부록4에 따라 새 trajectory로 시작했다.4개 실행·19개 대기 관측. 새 task `B93_GPU_B94_recovery99`; metadata 읽기/쓰기 오류만 격리한 감독기이며 frozen 수치/큐 코드는 그대로다. `docs/paper/b99_recovery.md/json`, `b94_status.md` 최상단 참조. 마감은 원래대로다. 아래14:27 ‘중단’ 문장은 이 후속 재개 전 기록이다.

**물리 수식 접근 조사(2026-11-23):** 저자 요청으로 Claude API 다섯 검토와 원천 문헌·기호 검산을 수행했다. `docs/paper/physics_formulations_20261123.md/json` 참조. 현 자료의 최소 후보는 강제1차 완화식의 다섯 계수 모형이며 τ는 물리 수송시간으로 식별된 값이 아니다. 더 직접적인 물리 후보인 pedestal 변위 분리·CES 선폭 예산은 기하/보정/원시 스펙트럼이 추가로 필요하다. OU 잡음·노출 평균·L5 중앙값 해석 오류를 수정했다. 실제 적합·TEST 탐색·GPU 재개는 하지 않았다. 후속 실험은 기존 특징/대조와 중복을 먼저 확인하고 별도 사전등록한다.

**Lean 후속 구현 완료(B.97):** 선택기 여섯 정리의 Lean 컴파일·공리 감사를 통과했고 등록 TRAIN/VAL 질의80개가 Python·Lean·별도 직접 탐색과 일치했다. 독립 검사의 경로 key 표기 오류는 부록1의 새 adapter로만 정규화했고 원래 결과·소스·실패를 보존했다. `docs/paper/b97_lean.md`, `b97_lean.json`, `b97_lean_checks.md` 참조. 전체 신경망/Python/CUDA 검증이 아니다. 아래 조사 당시 ‘Lean 증명 없음’은 이 후속 구현 이전 상태다. GPU 재개는 하지 않았다.

**현재 운영 상태(2026-11-23 14:27 KST 확인):** B.96 감독·큐가 `PermissionError`로 종료했고 네 trainer도 생존하지 않는다. confirmation은 stale 큐 기록상23/46 완료이며 중단된 네 partial에는 정상 종료 증거가 없다. 재시도하지 않고 보존했다. 원인과 복구 전 조건은 `docs/paper/b94_status.md` 최상단을 읽는다. 아래의 과거 ‘실행 중’ 문장을 현재 상태로 읽지 않는다.

**Lean 조사 완료:** 저자 요청으로 Claude 공식 API 여섯 검토(Fable/Opus 각 세 건)와 원천 문헌 교차 확인·기호 항등식 검산을 수행했다. 판단·정정은 `docs/paper/lean_feasibility_20261123.md`, 호출 감사는 `claude_lean_reviews_20261123.md`. 최우선 후보는 가용정보 비간섭의 좁은 명세이며 새로운 물리나 수학 정리를 확립하지 않았다. Lean 컴파일·증명은 아직 없고 현재 학습 방법·TEST 선택을 바꾸지 않았다. 새 age 실험은 기존 B.75/B.94와 중복을 먼저 확인하며 별도 사전등록 전 TEST 탐색을 하지 않는다.

이 저장소의 일을 다른 에이전트(Codex 등)나 사람이 이어받아 실험을 계속할 수 있게 쓴 문서다. 규칙의 원문은 `CLAUDE.md`와 `PAPER_DESIGN.md`에 있고, 이 문서는 그것들을 어떻게 쓰는지, 무엇이 어디 있는지, 다음에 무엇을 할지를 적는다. 수치의 원천은 `PROJECT_KNOWLEDGE.md`와 `docs/paper/`의 결과 파일이다. 이 문서에는 수치를 거의 적지 않는다.

## 0. 한눈에

**최신 저자 선택: 전역4개(B.96).** 저자가 보수적인10% 확대 기준과 최적 설정을 구별하고4개로 진행하도록 지시했다. B.95 원 결과/선택은 보존하고, 이미 정상 종료한4-worker cohort를 실행 가능성 근거로 사용한다. 이전 두 V3 C/A fit은 정상 완료·원 제어기의 exit0 기록 뒤 제어기만 교체했으며 학습을 버리거나 재시작하지 않았다. 새 task는 `B93_GPU_B94_four96`, 진입점 `b96_four/production96.py`; 실제 전역 수는 `.b94/dispatch_policy96.json`과 새 config의 `global_gpu_worker_limit=4`를 읽는다. 로컬 그룹의2를 전역 수로 해석하지 않는다. `docs/paper/b96_four.md`가 선택 근거와 보존 기록이다. 아래 B.95의2개 선택은 이전 운영 정책이다.

**동시 실행 최종 실측(B.95).** 2/3/4/6개로 실제 전체 epoch를 실행했고 모든 worker가 정상 종료했다. RAM·VRAM상 6개 실행도 가능했으나 반복 기준으로 정한 처리량 개선 문턱을 넘지 못해 전역 두 slot을 유지한다. `docs/paper/b95_concurrency.md`, `.b95/production_policy95.json`이 선택 근거다. 수학적으로 적격이 될 수 없는 추가4개 cohort는 부록2에 따라 실행 전에 취소했다. 새 one-shot task는 `B93_GPU_B94_resume_1117`, 실행 진입점은 `b95_concurrency/production95.py`; CPU 중복 confirmation은 재개하지 않는다. 동결 B.94 완료 fit·smoke·준비 자료를 재사용한다.

**현재 재개 지시(2026-11-23 오전).** 저자 요청으로 일시정지했던 프로세스는 이후 OS 재부팅으로 모두 소실되었다. B.94 부록2에 따라 정상 종료한 GPU pilot 두 fit·세 smoke·준비 캐시를 검증해 재사용하고, 중단된 두 partial만 `.b94/interrupted_20261123_resume/`에 보존한 뒤 동일 seed로 재시작한다. RNG 상태가 없어 exact checkpoint continuation이라고 쓰지 않는다. CPU B.92 중복 confirmation은 재개하지 않는다. 원래 suite 마감은 유지한다. 저자의 RAM 여유·동시 실행 확대 지시로 B.95에서 실제 전체 epoch의 2/3/4 및 조건부6 worker 처리량을 측정하고, 독립 감사된 수를 B.94 부록3의 전역 dispatcher에 적용한다. 새 실행 코드는 `b95_concurrency/production95.py` → `bootstrap95.py`이며 동결 B.94 수치 코드는 수정하지 않는다. 실제 전역 worker 수는 새 config의 `global_gpu_worker_limit`과 `global_worker_slot`, `dispatch_policy95.json`을 읽는다. 기존 `scheduled_concurrency`는 명시된 두 slot 이하 그룹의 수다. 아래 과거 PID·CPU 재개 설명은 현 상태로 사용하지 않는다.

**병렬 실행 개정(2026-11-23).** CPU_r2는 이전 프로세스 소실을 확인한 뒤 같은 검증된 큐로 재개했다. 단계 JSON만으로 실행 중이라고 판단하지 말고 `py ces_prediction/experiments/b93_gpu/status93.py --cache data/.b92/cpu_r2`로 PID·생성 시각·독립 watcher heartbeat를 확인한다. 관측기는 중단을 감지하지만 기존 큐를 자동 재시작하지 않는다. `docs/paper/b93_supervisor.md` 참조.

저자의 병렬/GPU 지시에 따라 B.93을 등록·커밋하고 기존 CPU 학습과 동시에 native CUDA의 실제 전체 epoch 및 두 작업 동시 실행을 검증 중이다. `data/.b93/progress93.json`, 최종 `runtime_decision93.json`이 실행 근거다. 첫 정상 종료만으로 전체 경로나 두 작업의 가속을 확정하지 않는다. B.94는 해당 결정의 독립 감사와 실제 dropout smoke를 통과한 경우에만 별도 캐시에서 후보 축소 실험을 실행하도록 사전 등록했다. B.92 소스·fit·CPU override를 바꾸거나 GPU fit과 합치지 않고 원래 suite 마감을 유지한다. `PREREGISTRATION_B93.md`, `PREREGISTRATION_B94.md` 참조.

- **무엇을 하는가.** KSTAR 2022 캠페인의 pedestal top 측정선(R = 2.198 m)에서 CES 이온 온도를, 다음 CES 값을 기다리지 않고 10 ms마다 추정한다(nowcasting). 입력은 그 측정선의 CES 이력과 같은 위치의 빠른 진단(BES 9채널, ECEI 4채널)이다.
- **지금 상태.** 원고(`docs/paper/ppcf/main_ppcf.tex`, 본문 18쪽, 보충 29쪽)는 저자의 틀로 쓰여 있고 빌드와 수치 검사를 통과한다. B.87까지 완료 기록이 있다. B.85와 이전 전체 큐는 저자의 2026-11-23 속도·후보 축소 지시로 checkpoint를 보존하고 중단했다. 현재 B.91 runtime 실측 뒤 B.92 축소 비교를 진행한다. 방법·상태는 `docs/paper/efficiency_decision_20261123.md`, `b91_runtime.md`, `b92_status.md`. 독립 채점은 PPCF 13–20 %에 머문다(`docs/paper/review_loop/`).
- **GPU·후속 대기열 승인(2026-11-22).** 저자가 GPU와 Claude API 사용을 승인했고, 이어 우선 후보를 모두 자동 대기열에 설정하라고 지시했다. 현재 B.85를 유지하고 B.89 실행 속도 점검 → B.88 anchor/current → B.90 후보 일곱의 비교·추가 반복을 연결한다. 등록과 source-bound 검사를 따른다. B.69와 MAST의 옛 대기 학습은 이 승인으로 자동 재개하지 않는다. 실제 상태와 명령은 `docs/paper/queued_experiments_20261122.md`.

**최신 지시(2026-11-23, 위 전체 큐를 대체).** 저자가 현재 학습 가속과 확실한 소수 후보로 축소를 Fable API로 각각 분석하고 실행하라고 지시했다. 실제 claude-fable-5-2 두 완료 검토와 별도 설계 검토 후 B.91/B.92를 등록·푸시했다. B.85는 부록4에 따라 네 clean fit와 현재 epoch2 partial checkpoint를 보존하고 연구 소유 작업만 중단했다. 옛 suite와 B.88/B.89/B.90 전체 배정은 superseded이며 자동 재개하지 않는다. B.91의 일곱 경로·세 번 실측은 완료됐고, B.92는 C/A/TCN의 열두 pilot와 선택 pair의 최대 마흔여섯 추가 fit로 제한한다. `docs/paper/efficiency_decision_20261123.md`, `b91_runtime.md`, `b92_status.md`가 현재 판단·실측·실행 상태다. 전체15unit 단일 seed와 네 unit 세 seed를 섞어 전체 초기화 안정성이라고 쓰지 않는다.

**실행 경로 수정(B.92 부록2).** CUDA dense cuDNN의 dropout0 배치는 빨랐지만 실제 dropout.1 smoke는 계산 후 비정상 종료했다. 성공한 학습으로 재사용하지 않고 원 기록을 보존했다. Pilot 전에 CPU dense MKLDNN8로 명시적 revision을 등록했다. 현재 namespace는 `data/.b92/cpu_r2/`, 선택 증거는 `.b92/production_override_r2.json`; 원래 B.91 결정과 실패한 `.b92/smoke/`를 덮어쓰지 않는다. 새 readiness/독립 검사와 dropout.1 정상 종료가 필요하다. 상태는 `docs/paper/b92_status.md` 및 cpu_r2의 queue_status.json을 읽고, 원래24h wall clock을 이어받는다.

**실제 재개:** CPU_r2 readiness와 독립 감사를 통과했고 C/A/T의 실제 dropout.1 fresh smoke 모두 exit0을 확인했다. 2026-11-23 02:06:13 KST supervisor PID954168이 pilot에 들어갔다. 코드/등록/기록은 푸시했다. 상태 확인은 `Get-Content data/.b92/cpu_r2/queue_status.json`, 학습 로그는 같은 폴더의 `pilot/*/process.log`다. 이전 CUDA/전체 큐를 다시 띄우지 않는다. 끝난 fit와 진행 중인 fit, pilot 선택과 TEST 결과를 구분한다.

주장별 강건성과 차별점 검토는 `docs/paper/claim_robustness_20261123.md`에 있다. 기존 결과·반대 증거·판단을 바꿀 측정을 함께 적었으며, 진행 중 NN의 TEST 성능이나 새로운 물리의 확인으로 읽지 않는다.

## 1. 먼저 읽을 것 (이 순서로)

병렬 GPU 큐는 `docs/paper/b94_status.md`와 `py ces_prediction/experiments/b93_gpu/status93.py --cache data/.b94`로 확인한다. B.93 전체 workload 동시 실행은 두 worker 기준을 통과했고, 별도 B.94 큐가 원래 마감으로 시작되었다. 상태 문서의 시작 사실과 현재 실제 학습 단계는 구분한다.

0. `README.md`: 저장소의 배선(무엇이 무엇을 읽는가)
1. `AGENTS.md` (Codex가 자동으로 읽는 요약) 또는 `CLAUDE.md` (같은 규칙의 원문)
2. `PAPER_DESIGN.md` 1–10절: 논문의 목적, 용어, 표적, estimator, 검증 설계, 서술 규칙, 논문 지도
3. `PROJECT_KNOWLEDGE.md` 1–3절(목적, 자료, 설계), 4절(쓰는 결과: 4.48절부터), 6절(규칙)
4. `ces_prediction/experiments/README.md`: 배치 디렉터리와 상태 표
5. 이 문서의 9절(열린 일)과 10절(확장 방향)

`THESIS_RESULTS.md`는 시간순 기록이다(B.70부터). 한 실험의 세부를 볼 때만 연다. 거기에만 있는 수치는 근거로 쓰지 않는다.

**2026-11-21의 정리.** 저자의 지시로 옛 연구를 저장소에서 걷어냈다: 지금 줄기의 코드가 import하거나 읽지 않는 실험 디렉터리 47개와 그 등록·결과·시험, 옛 문서, 이전 원고의 스크립트, `PROJECT_KNOWLEDGE.md`의 4.1–4.47절, `THESIS_RESULTS.md`의 B.70 앞 기록. 모두 git 이력에 있다(지우기 전 마지막 커밋 `e86adaf`). 되살리지 않는다. 새 파일을 만들면 무엇이 그것을 읽는지 `README.md`의 배선 표에 적는다.

## 2. 논문의 틀 (바꾸지 않는다)

저자가 정한 틀(`PAPER_DESIGN.md` 1절):

> pedestal top에서의 nowcasting 문제를 푼다. 정확히는 보간법 등에 견주어 가장 좋은 모델을 찾는다. 지금 가진 자원으로 10 ms 실시간이 되어야 하고, 적용했을 때 더 잘 작동함을 보이면 더 좋다.

동기는 "pedestal top에서는 CES가 약하다(값이 비고, 흩어지고, fit이 실패한다)"이다. 논문의 질문은 다섯이다: 어느 estimator가 가장 좋은가(hold, 인과 GP, 그리고 다음 값을 기다리는 PCHIP·offline GP 대비) → 빠른 진단이 더하는 것 → 방전을 달리 나눠도·시간 순으로도·실시간으로 → 적용했을 때(CES가 늦거나 드물 때, 둘째 장치) → 성립하지 않는 곳.

**하지 말 것.** 비판이나 채점 결과에 답해 줄거리를 새로 짜지 않는다. 2026-11-20 낮에 줄거리를 "늦은 CES 값을 빠른 진단으로 잇는다"로 옮겼다가 저자가 되돌렸다. 새 결과는 위 다섯 질문의 어느 자리에 들어가는지로만 정리한다. 주장을 pedestal top에서 떼지 않는다. "같은 반경이라서 된다"고 쓰지 않는다(B.82). 채점 수치를 올리려고 주장을 바꾸지 않는다.

## 3. 환경

| 항목 | 값 |
|---|---|
| OS, 셸 | Windows 11, PowerShell과 Git Bash |
| 인터프리터 | `py` (Python 3.14). 모든 스크립트는 **저장소 루트에서** 실행 |
| 패키지 | `pyproject.toml`의 numpy, scipy, pandas, torch에 더해 scikit-learn(`HistGradientBoostingRegressor`), matplotlib, pypdf, python-pptx |
| LaTeX | MiKTeX, `latexmk -pdf` (`docs/paper/ppcf/`에서 `supp_ppcf` 먼저, 그다음 `main_ppcf`; 둘이 서로를 참조한다) |
| GPU | 노트북 GPU 하나(8 GB). 다른 프로세스(`llama-server.exe`)가 쓰고 있을 수 있다. 그 프로세스는 건드리지 않는다 |
| 시험 | `py -m pytest -q` (89개 통과가 정상) |

함정:

- **콘솔은 cp949다.** 스크립트의 `print()`에는 ASCII만 쓴다. 한글을 찍으려면 `sys.stdout.reconfigure(encoding="utf-8")`.
- **Bash heredoc 안의 백슬래시가 바뀐다.** LaTeX나 정규식을 다루는 Python은 heredoc으로 넘기지 말고 파일로 써서 실행한다(또는 `chr(92)`로 백슬래시를 만든다). `sed -i`로 `\usepackage`를 고치면 `\u`가 대문자 변환으로 읽힌다.
- **샌드박스가 GPU를 가린다.** GPU 작업은 샌드박스 밖에서 돌려야 한다.
- 학습 스크립트(`train.py`, `experiments/seq/train_seq.py`)는 `CES_*` 환경 변수로 설정한다. 자료 처리 변수(`CES_DROP_STUCK_TARGETS`, `CES_TI_SPIKE_CUT_EV` 등)는 기본값에 맡기지 말고 러너의 env로 못박는다. 본보기는 `experiments/b64_two_stage/run_b64.py`, `experiments/b39_primary/run_b39.py`다.
- `ces_prediction/model_iter009.py`, `model_iter002.py`는 고치지 않는다(SHA-256으로 고정). GPU 배치가 도는 동안에는 기존 학습 코드를 고치지 않는다.

## 4. 자료와 캐시

`data/`는 git에 올리지 않는다(`data/splits/`만 예외). KSTAR 자료는 저자의 연구실을 통해 받은 것이고 다시 배포할 수 없다. 새 기계에서는 아래를 제자리에 둔다.

| 위치 | 내용 | 출처 |
|---|---|---|
| `data/s{shot}.csv` (641개) | 10 ms 방전 파일: time, CES_TI, CES_VT, BES 9, ECEI 4, MC 2 | KSTAR (제공) |
| `data/nbi/s{shot}_NBI.csv` (884개) | 같은 행의 빔 source별 출력·전압 | KSTAR (제공, 2026-11-20) |
| `data/raw_mus/` | 방전 12개의 원시 스트림(500 kHz BES 등) | KSTAR (제공) |
| `data/mast/` | 공개 FAIR-MAST 자료 | `experiments/mast/fetch_*.py`로 다시 받을 수 있다 |
| `data/.b70` … `data/.b86` | 배치별 캐시(행 표, 입력, 저장된 추정) | 아래 순서로 다시 만들 수 있다 |

파일 규칙(`docs/paper/file_rules.md`): 파일의 첫 행과 끝 행은 격자 행이 아니다. 빠른 채널의 값은 시각 표지에서의 원시 표본 하나다. 20–40 ms 간격은 빠진 프레임(NBI-1의 notch)이다. 3 keV 넘는 T_i와 직전 값과 끝자리까지 같은 값(유지값)은 어디서나 결측으로 다룬다(자료 정의 D1, `PAPER_DESIGN.md` 3절).

캐시의 의존 순서(뒤의 배치는 앞의 캐시를 읽는다):

```
B.70 (D1의 신경망 추정, PCHIP·offline GP 기준)  data/.b70*
  └ B.73 rows.npz, grid.npz, trees/ (D1 행 표, 표적, 신경망 없는 estimator)  data/.b73
      ├ B.75 inputs/{rule}.npz, sets.npz, fits/ (가용 조건별 입력과 추정)  data/.b75
      │    ├ B.78 (K 규칙)  ├ B.79 (빔 스칼라; data/nbi 필요)  ├ B.80  ├ B.81  ├ B.82  └ B.84
      └ B.83 (보간 기준; B.70의 코드 경로)
B.76 (MAST 행 표와 추정)  data/.b76  └ B.86 (반경 대역별)
```

각 배치 디렉터리의 단계 스크립트를 이름 순서(입력/행 → fit → check → readout → render)로 돌리면 캐시와 `docs/paper/b{NN}_*.{md,json}`이 다시 나온다. 정확한 실행 순서와 인자는 각 배치의 `PREREGISTRATION_B{NN}_ADDENDUM1.md`에 적혀 있다. 캐시가 이미 있으면 다시 돌릴 필요가 없다.

## 5. 실험 한 배치의 절차

모든 실험은 같은 절차를 따른다. 본보기는 B.81–B.84다.

1. **등록.** `ces_prediction/experiments/PREREGISTRATION_B{NN}.md`를 쓴다: 왜 하는가(알려진 것과 모르는 것), 자료와 행, estimator, 구간, 번호 붙인 진술마다 판정 규칙과 예상, 판독 전 점검, **결과에 따라 논문에 들어갈 문장**, 산출물. 실행 전에 커밋하고 푸시한다. 등록 파일은 그 뒤 고치지 않는다.
2. **부록.** 등록이 열어 둔 선택은 실행하는 쪽이 `PREREGISTRATION_B{NN}_ADDENDUM1.md`에 판독 전에 적는다.
3. **실행.** 새 디렉터리 `experiments/b{NN}_{name}/`에 새 파일만 만든다. 기존 코드는 import해서 쓴다. 순서는 입력 → fit → 점검 → 판독이다. 점검이 통과하기 전에는 오차나 skill을 계산하지 않는다. 캐시는 `data/.b{NN}/`.
4. **산출물.** `docs/paper/b{NN}_{name}.md`와 `.json`(진술마다 값, 군집 구간, 방전 구간, 판정, 예상), `b{NN}_checks.md`.
5. **독립 재계산.** 실행한 쪽과 다른 코드로 입력이나 진술 값을 다시 계산한다: `experiments/b{NN}_{name}/lead_checks/verify_b{NN}.py`, 기록은 `docs/paper/b{NN}_lead_checks.md`.
6. **기록.** `PROJECT_KNOWLEDGE.md` 4절에 새 소절(진술 표, 보고, 읽는 법), `THESIS_RESULTS.md` 끝에 항목, `experiments/README.md`의 상태, `ces_prediction/check_records.py`의 `SOURCES`에 새 md. 설계가 바뀌면 `PAPER_DESIGN.md` 11절 변경 기록.
7. **검사와 커밋.** `py ces_prediction/check_records.py`(출처 없는 수치 0개여야 한다), `py -m pytest -q`. 그다음 커밋하고 푸시한다.

통계 규칙(`PAPER_DESIGN.md` 7절): 검증 설계 넷(V1 무작위, V2 shot 군집 차단, V3 rolling origin, V4 시간순)을 통째로 싣는다. 구간은 95 %, 20,000회, shot 군집(MAST는 session)을 다시 뽑는다. 층은 입력이나 시간 구조로만 정의한다. 판독을 본 뒤 바꾼 처리는 민감도로 나란히 싣는다.

서술 규칙(`PAPER_DESIGN.md` 8절): 결과는 점추정, 95 % 구간, 0 제외 여부로 적는다. "이긴다", "beats", "first", "novel"을 쓰지 않는다. 수치는 표에 둔다. 음의 결과에는 그것을 뒤집을 측정을 함께 적는다.

## 6. 최근 배치 (무엇을 물었고 어디 있는가)

| 배치 | 물음 | 코드 | 결과 |
|---|---|---|---|
| B.98 | 저차 물리 동기 응답식의 유한 구현과 내부 VAL | `experiments/b98_physics/` | `docs/paper/b98_physics.md/json`, `b98_lead_checks.md`; 새 물리 식별 아님 |
| B.99 | metadata 장애 후 원래 GPU 큐 재개 | `experiments/b99_recovery/` | `docs/paper/b99_recovery.md/json`; 정상 결과 보존·원래 마감 |
| B.97 | 가시 source 선택 명세와 실제 trace 연결 | `formal/visibility/`, `experiments/b97_lean/` | `docs/paper/b97_lean.md`, 경로 key 정규화 후 독립 검사 완료 |
| B.70 | 자료 정의 D1에서 신경망·2단계 estimator와 기준들 | `experiments/b70_d1/` | `docs/paper/b70_d1.md` |
| B.71 | CES 갱신이 멈출 때(신경망과 filter의 재생) | `b71_scenarios/` | `b71_scenarios.md` |
| B.73 | 표적 둘(그 프레임의 값, 이후 수준)과 설계 넷; 신경망 없는 estimator | `b73_targets/` | `b73_targets.md` |
| B.74 | 한 프레임만 튀는 값과 노출 안의 버스트 | `b74_exposure/` | `b74_exposure.md` |
| B.75 | CES 이력이 묵었을 때(가용 규칙별) | `b75_availability/` | `b75_availability.md` |
| B.76 | MAST: 묵은 CXRS 이력과 Thomson | `b76_mast_availability/` | `b76_mast_availability.md` |
| B.77 | 프레임의 값을 읽은 뒤의 보정 | `b77_correct/` | `b77_correct.md` |
| B.78 | CES 값이 k번째 프레임마다 | `b78_sparse/` | `b78_sparse.md` |
| B.79 | 빔 파형을 입력으로 | `b79_beam_inputs/` | `b79_beam_inputs.md` |
| B.80 | ECEI 채널의 상관을 조건으로 | `b80_ecei_gate/` | `b80_ecei_gate.md` |
| B.81 | 대조 둘, ECEI만·BES만 | `b81_controls/` | `b81_controls.md` |
| B.82 | notch 뒤 경과 시간별, 견고한 지표, Mirnov 대조, seed | `b82_localize/` | `b82_localize.md` |
| B.83 | **주 표: 다음 값을 기다리는 보간 대비** | `b83_interp/` | `b83_interp.md` |
| B.84 | 2차 estimator가 조건에 묶여 있는가 | `b84_onefit/` | `b84_onefit.md` |
| B.86 | MAST에서 중심부터 가장자리까지 | `b86_mast_radius/` | `b86_mast_radius.md` |
| B.87 | 같은 회귀기의 미래 CES 허용 시점별 오차 | `b87_latency/` | `b87_latency.md`, `b87_lead_checks.md` |
| B.85 | 가용 조건을 섞어 학습하는 직접 출력 순환 모델 | `b85_unified/` | `b85_*` readiness/smoke/status; 전체 배치 후 unified 판독 |

B.85는 등록·구현·정상 종료 smoke를 마치고 본 배치를 실행 중이다. 9.1절과 등록 파일·부록을 따른다.

## 7. 원고

모든 수치는 스크립트가 넣는다. 손으로 옮기지 않는다.

```
py docs/paper/ppcf/make_tables_v2.py     # 결과 json → gen_v2.tex(표와 수치 매크로), gen_v2_tokens.json(출처), gen_v2_README.md
py docs/paper/ppcf/fig_v2.py             # figv2_*.pdf/png/json (예시 방전은 규칙으로 고른 31076)
py docs/paper/ppcf/make_supp_fragment.py . # 보충 자료의 자료 절(이전 판에서 잘라 옴)
cd docs/paper/ppcf && latexmk -pdf supp_ppcf && latexmk -pdf main_ppcf && latexmk -pdf supp_ppcf
py docs/paper/ppcf/check_numbers_v2.py   # tex에 손으로 친 수치가 출처에 있는지; --self는 gen_v2.tex 자체
```

- `gen_v2_README.md`에 표(`\TabXxx`, 캡션 `\CapXxx`), 매크로, 그림의 목록과 출처가 있다. 새 배치의 표가 필요하면 `make_tables_v2.py`에 함수를 더하고 `check_numbers_v2.py`의 `SOURCES_MD`에 md를 더한다.
- 본문은 `PAPER_DESIGN.md` 9절의 논문 지도를 따른다. 본문 16–18쪽, 그림 6–8개, 초록의 결과 수치 다섯 이내, 본문 표에 배치 번호를 쓰지 않는다.
- `docs/paper/ppcf/earlier_line/`은 2026-11-19까지의 원고를 얼려 둔 것이고 보충 자료의 자료 절(`make_supp_fragment.py`)이 거기서 온다. 고치지 않는다. `fig_paper.py`는 그 원고의 그림 스크립트로, 보충 자료의 그림 둘(`fig_geometry`, `fig_rows`)과 포스터 그림을 만든다. 그 원고의 표·검사 스크립트는 2026-11-21에 지웠다.

## 8. 독립 채점

원고의 수락 가능성은 진행자가 추정하지 않는다. 고정한 PDF 두 개만 받은 별도의 채점자 에이전트가 낸다. 지시문은 `docs/paper/review_loop/GRADER_PROMPT.md`에 있고, 비교가 되도록 그대로 쓴다. 채점자에게 저장소, 기록, 앞선 채점 수치를 주지 않는다. 결과는 `docs/paper/review_loop/grade{NN}_{날짜}.md`에 적고, 채점자가 표와 어긋난다고 짚은 문장은 고친다. 채점 수치를 올리려고 줄거리를 바꾸지는 않는다.

이력: 11–18차 PPCF 13, 20, 19, 16, 16, 15, 13, 15 % (`PROJECT_KNOWLEDGE.md` 4.13, 4.66, 4.67, 4.70절). 18차(저자의 틀로 다시 쓴 원고, 커밋 ab0bf1a)의 권고는 major revision이고, 짚인 문장은 고쳤다.

채점자들이 공통으로 꼽는, 가진 파일로 풀 수 없는 것: 실제 빔 변조 일정을 시간 순으로 채점하면 남는 결과가 없다; 보고값이 온도인지 가를 자료(스펙트럼, fit 오차)가 없다; 방전별 설정(toroidal field, ECEI 설정)이 없어 채널 위치를 확인할 수 없다.

## 9. 열린 일

### 9.1 GPU (B.85 재개 승인; 다른 배치는 대기)

**현재는 B.91→B.92(2026-11-23).** 아래 B.85 유지·옛 통합 supervisor 실행 설명은 이전 승인 기록이다. 저자의 더 빠르고 작은 계획 지시가 우선한다. `b91_fast/checkpoint_stop91.py`는 현재 연구 PID·경로·checkpoint/RNG/source를 확인하고 중단 전에 보존했다. 완료된 B.85 네 fit는 clean, 중단 partial은 nonzero/incomplete로 남으며 기존 로그를 지우지 않는다. B.91은 dropout0 실제 배치에서 dense cuDNN을 선택했지만 그 배수를 전체 epoch나 실제 dropout.1 안정성으로 옮기지 않는다. B.92의 새 source-bound 입력/모델/loss 검증과 dropout.1 fresh smoke가 통과해야 fit를 진행한다. CPU/CUDA 결정·정상 종료·24h 상한과 무승격 중단은 새 등록을 따른다. 기존 B.69/MAST는 여전히 이 큐 밖이다.

**후속 승인 추가(2026-11-22 밤).** 저자의 "다 세팅해줘"에 따라 새 통합 supervisor `ces_prediction/experiments/queued_suite_20261122.py`가 B.85 정상 완료를 기다린 뒤 B.89→B.88→B.90을 순차 실행한다. 각 사전등록·부록은 실행 전에 커밋했다. B.88은 명시적 측정 시점 anchor의 A/C 비교, B.90은 감쇠 GRU·causal TCN·두 대각 observer·Huber/Student-t/Gaussian loss 비교다. B.90의 추가 초기화 대상은 테스트를 읽기 전에 validation만으로 고정한다. 기존 B.85 소스·프로세스와 다른 GPU 프로세스는 건드리지 않는다. 상태·재개·실측 시간 해석은 `docs/paper/queued_experiments_20261122.md`, 병목 근거는 `docs/paper/runtime_plan_20261122.md`에 있다. 아래의 다른 미승인 배치 대기는 계속 적용한다.

**(가) B.85: 시나리오를 섞어 학습한 순환 신경망.** 2026-11-22 저자가 GPU 사용을 승인했고 등록·푸시했다(`PREREGISTRATION_B85.md`). 아래는 인계 당시 제안이며 실제 고정 설계는 등록을 따른다: 두 head로 F/L5 직접 출력, 빠른 진단 없는 쌍둥이, V2를 포함한 네 검증 설계, 학습에 없는 K10 평가. Claude API로 설계 검토와 신규 코드 작성을 진행하며 기존 학습 파일은 고치지 않는다. 실제 shot 자료와 비밀키는 외부 요청에 넣지 않는다.

- **왜.** 지금 논문의 estimator는 인과 GP 위에 얹은 boosted tree(손으로 만든 스칼라 24개)이고 가용 조건마다 다시 맞춰야 한다. 조건 하나에 맞춘 것을 다른 조건에 쓰면 빠른 진단의 증분이 거의 사라지고, 본 적 없는 조건으로 옮겨 가지 않으며, 원시 창을 주면 못 배운다(B.84, `PROJECT_KNOWLEDGE.md` 4.69절). 5-parameter filter는 갱신 없이 돌리면 1 s 뒤에 발산한다(B.71). 저자의 말: 2차 estimator를 rule로 얹은 것은 시나리오 연결에 좋지 않다.
- **무엇을.** 순환 신경망(`experiments/seq/model_seq_v2.py`; 입력은 `seq_data.py`의 22채널: 빠른 채널, 시간 간격, 표적별 "마지막 관측값, 그 뒤 지난 시간, 관측 여부")을 **CES 이력의 가용 조건을 학습 중에 무작위로 바꿔 가며** 한 번 학습한다. 조건은 B.75의 규칙들이다: 직전 프레임까지(D10), 지연 d(50–1000 ms), 빔 변조 일정(S). segment마다 조건 하나를 뽑아 "마지막으로 보이는 값"과 "그 값의 나이"를 그 조건대로 만든다. k번째 프레임 규칙(K10)은 학습에 넣지 않고 시험에만 쓴다(본 적 없는 시나리오).
- **견줄 것.** 같은 행에서: 조건별로 맞춘 TX와 T0(`data/.b75/fits/`), 한 번 맞춘 TXall(`data/.b84/fits/`), 직전 프레임 조건의 TX[F](`data/.b73/trees/`), 빠른 채널을 뺀 쌍둥이 신경망. 표적은 그 프레임의 값과 이후 수준 둘 다(출력 head 둘 또는 따로 학습; 등록에서 정한다).
- **단위.** V1의 재분할 r2(5 fold), V3(4), V4a(1)부터. 초기화는 셋. 학습 설정(epoch 상한, 조기 종료, D1 처리)은 `experiments/b64_two_stage/run_b64.py`의 env를 그대로 쓴다.
- **미리 정할 진술(안).** (1) 직전 프레임 조건, 그 프레임의 값: 신경망이 TX보다 5 % 넘게 나쁘지 않다. (2) D650과 S, 이후 수준: 조건별 TX보다 5 % 넘게 나쁘지 않다. (3) K10: 그 조건에 맞춘 T0보다 낫다(옮겨 가는가). (4) V3에서 (2). (5) 쌍둥이 대비 증분이 D650에서 0 위. (1)–(3)이 성립하면 논문의 estimator를 이 신경망으로 바꾸고 "모델 하나가 nowcast와 모든 시나리오를 잇는다"고 쓴다. 아니면 지금 것을 유지하고 그 결과를 한계에 적는다.
- **구현 주의.** 기존 학습 코드를 고치지 말고 새 디렉터리에서 `seq_data`의 블록을 감싸 조건별 이력 입력을 만든다. B.75의 `inputs/{rule}.npz`는 D1 행에 대한 것이라, 격자의 모든 행에 대해 보이는 값을 다시 계산해야 한다(B.75 부록 1의 정의대로). 누출 점검: 보이지 않아야 할 값을 1000 eV 올려도 입력이 같아야 한다.

**(나) B.69의 남은 15회.** 2단계 estimator의 시간 분할 확증 시험이다(`PREREGISTRATION_B69.md`, 40회 가운데 25회에서 멈춤). 등록된 것이므로 끝내 보고해야 한다. 채점자가 바깥 run 여덟 개를 이미 읽었다는 것을 보고에 밝힌다.

**(다) MAST에서 같은 순환 신경망, 장치 간 전이, 층별 비교.** 저자가 정한 흐름이다(`PROJECT_KNOWLEDGE.md` 7절의 표). (가) 뒤에 한다.

**B.85 실행 인계(2026-11-22).** 입력 검사와 별도 `lead_checks/verify85.py`가 통과했고, 판독 알고리즘은 `lead_checks/verify_readout85.py`로 기존 실자료의 예측을 명시적으로 재표집해 별도 검증했다. Claude 작성·수정 이력과 실패 기록은 `docs/paper/b85_implementation_provenance.md`에 있다. 본 실험 성능 판독은 아직 없다. CUDA smoke에서 계산 검사와 한 epoch는 끝났으나 자식 프로세스가 비정상 종료해 그 산출물은 격리했다. 깨끗한 X/H smoke 재시도를 모두 통과해야 본 배치가 시작된다. 다른 `llama-server.exe`의 VRAM 점유는 건드리지 않는다.

`queue85.py`는 smoke → 등록한 본 배치 → 예측 → 판독을 연결하며, VRAM이 부족하면 대기하고 자식 프로세스 실패 시 중단한다. 자동 실행 상태는 `data/.b85/queue_status.json`, `docs/paper/b85_smoke.md`, `b85_status.md`, `data/.b85/logs/`를 확인한다. `py ces_prediction/experiments/b85_unified/run85.py --status`는 읽기 전용 상태 확인이다. 재개 전에는 기존 queue PID와 lock을 확인해 중복 실행하지 않는다. 학습 파일·정규화·설정·소스 해시가 맞는 완료 산출물만 재사용한다. 전체 배치가 끝나고 독립 판독 검증을 마치기 전에는 논문의 estimator를 바꾸지 않는다.

**B.85 GPU 재개 기록.** `docs/paper/b85_runtime_fix.md` 참조. 실자료로 cuDNN LSTM 경로의 종료 오류를 분리했고 등록 부록에 따라 native ATen CUDA로 고정했다(CPU 학습 아님). 새 X/H smoke가 모두 정상 종료했으며 본 배치가 실행 중이다. 현재 queue 로그는 `data/.b85/queue_aten.log`와 `queue_aten.stderr.log`; 기존 `queue.log`는 과거 실패·대기 이력이다. `process_exit.json`의 종료 코드와 설정 해시가 맞아야 완료·재사용·판독을 허용한다. 실패한 산출물은 격리 보존했다. 실행 중에는 학습 소스를 고치지 않는다.

### 9.2 CPU로 할 수 있는 것

**방법 후보 조사(저자 요청).** Claude API의 실제 web search를 이용해 결측·불규칙 시계열, 진단 융합·표현 학습, 관측 잡음·불확실도, 장치/조건 변화·관측기 계열을 별도로 탐색한다. 원문·사용량은 `docs/paper/method_scout_claude_20261122.md`, 별도 원천 확인과 선택 근거는 `docs/paper/method_selection_20261122.md`. 이는 문헌 조사이며 새 학습 실험이 아니다. B.85 실행을 유지하며 새 후보는 자료 요구·인과성·실행 비용과 판정 규칙을 정해 따로 등록한다.

2026-11-22: `docs/paper/research_direction_20261122.md`에 현 기여와 선행연구의 겹침, 반증 조건, 저널 적합성 판단을 정리했다. 기다림의 값어치는 B.87로 등록·푸시한 뒤 실행·독립 검증을 완료했다(`experiments/b87_latency/`, `PROJECT_KNOWLEDGE.md` 4.72절). 현재 빠른 진단이 미래 CES를 대체한다는 시간순 근거는 없고, 동일 학습기에서도 기다림의 정보가 더한다. 원고 반영은 아직 하지 않았다. 다음 CPU 작업은 표적 견고성의 별도 등록, 그다음 사건 시연이다. GPU는 재개하지 않았다.

1. **B.86은 끝났고 원고에 들어갔다**(`PROJECT_KNOWLEDGE.md` 4.71절; 본문의 MAST 절 문단, 보충 자료의 대역별 표).
2. **사건 기준의 쓸모(설계 초안).** 저자의 목적에 있는 "적용했을 때 더 잘 작동함"을 수준이 급히 떨어지는 사건으로 보인다. 사건은 이후 수준이 앞선 300 ms의 중앙값보다 정해 둔 비율 넘게 떨어지는 첫 행으로 정의한다. 견줄 것은 (a) 빔 변조 일정대로 늦게 오는 CES 값만 보는 감시, (b) T0, (c) TX가 같은 문턱을 넘는 시각이다. 보고는 놓친 사건의 몫, 거짓 경보율, 검출 지연의 중앙값. 저장된 추정(`data/.b75/fits/`)으로 된다. 등록부터 한다.
3. **기다림의 값어치: B.87 완료.** 같은 boosted regression에 이후 CES를 허용해 정보 시점과 학습 여부를 분리했다. 결과와 곡선은 `docs/paper/b87_latency.*`, 해석은 `PROJECT_KNOWLEDGE.md` 4.72절. 원고 반영 전이다.
4. **표적 검증(17차·18차 채점자의 권고).** 다섯 프레임을 둘로 나눈 잡음 바닥, notch 뒤 100 ms 안의 행과 BES 수준이 가장 낮은 행을 뺀 주 표의 재채점, 컷을 2 keV와 4 keV로 바꾼 판독, 절대 오차를 손실로 한 fit.
5. **`experiments/seq/`의 옛 변형 정리.** `seq_models.py`가 지운 배치들의 구조 변형(v4k, v2d, b28, b3, tcn, xfmr 등)을 한 사전에 import한다. 지금 줄기는 v2, v2h, v2e·v2b·v2n만 쓴다. 변형 파일과 그 시험을 지우려면 `seq_models.py`를 고쳐야 하고, 저장된 학습 결과가 그대로 읽히는지(B.70의 채점 경로) 확인해야 한다. GPU 배치가 돌지 않을 때 한다.
6. **구간의 보정.** 군집이 14–19개인 시간순 설계에서 섞은 입력으로 구간의 포함률을 잰다(B.81의 대조를 여러 seed로).

### 9.3 저자의 결정을 기다리는 것

- B.69 잔여 학습과 MAST GPU 확장을 언제 재개할지(9.1). B.85와 새 B.88/B.89/B.90 후속 대기열은 2026-11-22 승인됨.
- 원고를 12쪽 안팎의 짧은 논문으로 줄일지(17차 채점자의 권고).
- 포스터와 발표 덱(저자의 바탕화면에 있는 `중간 보고 2_final.pptx`)은 저자가 "수정"을 말할 때만 고친다. 고치기 전에 저자가 손본 곳이 있는지 확인한다.

## 10. 덱의 흐름을 바탕으로 한 확장 방향 (제안, 2026-11-21)

저자가 이해하고 있는 흐름은 발표 덱의 순서다: 배경(결측) → 문제(선행 연구, KSTAR 병목) → 진단과 자료 → estimator와 채점(2단계 estimator, 5-parameter filter) → 결과 → 정보를 나르는 입력 → 한계와 전망. 저자의 평: 5-parameter robust filter는 우리가 낸 값을 믿을 수 있게 해 주는 정도이고 시나리오에서는 쓸모가 없었다. 아래는 그 흐름의 각 자리를 넓히는 것이지 흐름을 바꾸는 것이 아니다.

| 덱의 자리 | 지금 덱에 있는 것 | 넓힐 것 | 근거 · 상태 |
|---|---|---|---|
| estimator | 2단계 estimator, 5-parameter filter | **시나리오를 잇는 estimator 하나.** filter는 "간단한 모델로도 같은 증분이 나온다"는 확인으로만 둔다. 2차 estimator(tree)는 조건에 묶여 있으므로, 순환 신경망을 조건을 섞어 학습해 모델 하나로 nowcast와 시나리오를 잇는다 | B.71, B.84 (끝남); B.85 (GPU, 9.1) |
| 결과 | hold, 인과 GP, offline GP 대비 (무작위 분할) | 보간(PCHIP, offline GP) 대비를 네 설계로; 견고한 지표(중앙값, 10 % 안의 몫); 신경망 없는 estimator가 같은 오차 | B.83, B.82, B.73 (끝남, 원고 반영) |
| 정보를 나르는 입력 | "파일에 빔 출력 없음: 빔의 기여는 배제 못 함" | 빔 출력을 받았다: 빔 파형을 다 준 뒤에도 증분이 남는다. 섞은 입력으로는 0. Mirnov도 4분의 1을 낸다(반경 특이성은 주장하지 않는다) | B.79, B.81, B.82 (끝남, 원고 반영) |
| 한계와 전망 → "적용했을 때" | 컷 모집단, 측정선 하나, 실시간 사용 조사 중 | CES가 늦거나 드물 때 빠른 진단의 몫이 커진다; 둘째 장치(MAST); **MAST에서 중심부터 가장자리까지**; **사건 기준의 쓸모**(급한 하강을 얼마나 일찍 잡는가) | B.75, B.78, B.76 (끝남); B.86 (실행); 사건 시연 (9.2) |
| 제어에서의 값어치 | 요구 문헌 표 | pedestal top T_i는 제어가 원하는 양이지만 지금 정확도(직전 프레임 조건에서 수준의 10 % 안에 드는 추정이 세 프레임에 둘)는 요구에 못 미친다. 원고는 전망까지만 쓴다. MAST 반경별 표가 "어디서 10 % 안에 드는가"를 보인다 | B.82, B.86 |

우선순위: (1) B.86을 원고에 넣기, (2) 사건 기준의 쓸모 시연, (3) GPU가 풀리면 B.85. 셋 다 저자의 목적의 뒷부분("적용했을 때 더 잘 작동함")을 채운다.

## 11. 건드리지 않는 것

- `cellfie` 프로세스와 `llama-server.exe`.
- `docs/presentation/`의 추적되지 않는 파일들(`KSTAR_CES_의사결정.pptx`, `PPT_4STEP.md`, `audit_deck_geometry.py`, `build_pptx_decision.py`, `check_deck_rules.py`), `preview_pptx.py`.
- 저자의 바탕화면에 있는 덱들(읽기만 한다). PowerPoint가 열려 있으면 종료하지 않는다.
- `git push --force`, 이미 푸시한 이력의 수정, `--no-verify`. PR은 요청받을 때만 연다.
- 메일은 읽기만 한다. 제공자에게 보낼 글은 요청받을 때만 쓴다.
- 타인의 그림은 공개 저장소에 올리지 않는다.

## 12. 저자와 일하는 방식

- 답은 한국어로 한다. 결과는 표로, 점추정과 95 % 구간과 0 제외 여부로 적는다. 용어는 흔한 영어 용어를 쓴다(estimator, nowcasting; "모수"는 쓰지 않는다).
- 고르고 실행한다. 정말 저자만 정할 수 있는 것(틀, GPU, 제출, 바깥으로 나가는 행동)만 묻는다.
- 검증된 작업은 알아서 커밋하고 푸시한다. 실험은 실자료로만 한다(합성 자료 금지).
- 설계와 등록은 진행자가, 실행은 하위 에이전트가, 재현과 기록은 다시 진행자가 한다.
- 논문의 주축은 물리와 측정이다. AutoML 이야기는 꺼내지 않는다.
