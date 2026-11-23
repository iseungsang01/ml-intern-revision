# 작업자 별도 확인

## 이번에 새로 확보한 전문

- RTCAKENN, DOI [10.1088/1741-4326/ad142f](https://doi.org/10.1088/1741-4326/ad142f): 출판사 원문 PDF 전체를 확보했다. PDF p.4–6의 input/output 표, 모델 학습과 dropout, PCS의 최신 유효 데이터 재사용을 확인했다. CES/CER를 입력으로 쓰는 정상 운전과 해당 진단이 없는 상태의 평가를 구분해야 한다. 단순한 계산시간 주장에 그치지 않고 PCS 구현을 설명하므로, 우리 연구의 실제 배포가 없다는 점과 구별한다.
- COMPASS, DOI [10.1088/1361-6587/ae72c6](https://doi.org/10.1088/1361-6587/ae72c6): 원문 PDF 전체를 확보했다. PDF p.5–6에서 대상은 TS 기반 전자온도·밀도이며, 방전 단위 학습/검증 분리와 교차검증을 명시한다. 따라서 '이 계보에 방전 단위 검증이 없다'는 주장은 쓰면 안 된다. 이것만으로 시간순 외부 캠페인 검증이 확인되는 것은 아니다.
- Diag2Diag의 원문은 이전에 확보한 원래 XML body를 사용한다. Methods의 파생 특징·시간 정렬 및 검증 절을 함께 읽는다. 논문이 연결한 Zenodo DOI [10.5281/zenodo.16748363](https://doi.org/10.5281/zenodo.16748363)의 공식 코드 archive도 확보했다.
- RKN·CRU·KalmanNet은 앞선 감사의 일부 페이지가 아니라 확보한 전체 PDF와 공식 저장소의 핵심 소스를 제공한다. GitHub source commit 및 blob SHA를 보존한다. 코드의 원 논문 버전과 정리된 공개 버전 차이는 구분한다. 코드를 실행한 것은 아니다.
- Faithful heteroscedastic regression, β-NLL, TS2Vec, generalized distillation의 원문 PDF를 확보해 해당 방법의 실제 가정과 원 알고리즘을 확인하도록 했다.

## 확보하지 못한 내용

KSTAR NN-CES 출판사 PDF 접근은 거절됐고 공개 Elsevier API는 rate limit 응답이었다. 해당 전문을 확보했다고 쓰지 않는다. RTCAKENN/COMPASS를 확보했다는 사실이 NN-CES 전문 검토까지 뜻하는 것은 아니다. 원문 외 추가 자료·공식 코드가 없으면 그 부재를 특정 구현의 결함으로 해석하지 않는다.

## 기존 실험에 관한 핵심 정정

`PREREGISTRATION_B92.md`는 기존 B.88/B.89/B.90 실행 일정을 대체한다고 명시한다. 완료된 작은 pilot은 anchor encoder A, matched no-anchor encoder C, TCN T만 비교했다. B.90의 discrete/continuous observer, decay GRU, Huber, Student-t, mean/variance Gaussian 모델이 전체 실험에서 부정적인 결과를 냈다고 말할 수 없다. **등록됨·구현됨·pilot에서 평가됨·외부 TEST에서 평가됨을 따로 기록해야 한다.**

`EXISTING_WORK.md`는 source·registration을 읽은 후 작성한 methodology crosswalk다. 성능과 최신 결과에 대한 판단은 원래 독립 감사 gate를 통과한 익명 aggregate packet에만 근거한다. 이 조사 작업은 B.104를 실행·정지·수정하지 않는다.

## TS2Vec 공식 코드의 인과 추론 경로

공식 저장소가 `yuezhihan/ts2vec`에서 `zhihanyue/ts2vec`으로 이동한 것을 확인하고, commit `b0088e14a99706c05451316dc6db8d3da9351163`의 소스를 고정했다. `ts2vec.py:encode`의 기본값은 causal=False, sliding_length=None이다. sliding 분기에서만 오른쪽 문맥을 causal flag에 따라 제한한다. README는 causal=True, sliding_length=1로 각 시점의 과거 창을 인코딩하는 예를 준다. 따라서 causal=True라는 인자 하나만 주거나, 여러 시점을 포함한 chunk의 모든 출력을 그대로 사용하는 것으로 시점별 미래 차단이 보장된다고 쓰지 않는다. 실제 사용하는 출력 시점과 입력 창 경계를 확인해야 한다.

`models/encoder.py`의 dilated convolution과 사전학습 자체가 자동으로 시점별 causal deployment를 보장하는 것은 아니다. TRAIN 안의 양방향 context 사전학습은 배포 입력에 미래가 없고 TEST/VAL 누출이 없다면 원천적으로 금지되는 것도 아니다. 최종 배포 경로의 prefix 불변성과 matched-compute 비교가 별도로 필요하다. 이 소스 확인은 학습·벤치마크 실행이 아니다.

## 마지막 지연측정 문헌 보완

Mina·Marinello·Christian의 [Remarks on stochastic cloning and delayed-state filtering](https://arxiv.org/abs/2508.21260v2)은 내려받은 PDF 안의 v2 표기(2026-05-11)를 확인한 29쪽 전문이다. 고정한 파일 hash는 source manifest에 포함한다. 상대 상태에 의존하는 측정과 상관을 다루는 논문이므로 CES 절대 스칼라 지연과 같은 문제라고 가정하지 않는다.

Koch 등의 [Relative multiplicative extended Kalman filter for observable GPS-denied navigation](https://doi.org/10.1177/0278364920903094), §4.3의 출판사 HTML 본문을 직접 확인했다. 과거 상태·공분산을 복원하고 중간 입력과 관측을 다시 처리하는 유한 버퍼 구현을 설명한다. 따라서 재전파 자체는 기존 방법이다. 이 절은 로봇 필터의 구현 근거이며 CES 성능이나 정확한 비선형 베이즈 필터링의 증명이 아니다. Fable에 이 작업자 요약과 새 PDF를 전달해 P2를 별도 연구로 세울 가치까지 다시 검토했다.

## 비용 산정 절차

사용자가 허용한 추가 연구 비용 상한은 USD 30이다. 공식 [Fable 모델 단가](https://claude.com/pricing?currency=USD)와 [token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)을 확인했다. API 생성 요청마다 무료 count_tokens 결과에 여유를 더한 입력 비용과 최대 출력 비용을 미리 예약한다. 이번 paid 요청에는 server-side search/fetch를 넣지 않아 검색 중 입력이 자동 증가하지 않는다. 정확한 account 청구서가 아닌 공개 단가 기준 추정임을 `cost_ledger.json`에 명시한다.
