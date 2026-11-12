# Round 2 판정: 프로그램 위원장 (2026-11-13 02:45–03:20)

대상: `docs/paper/short/main_short.tex` (커밋 4739b59 기준, 688행, 본문 8쪽 중 7쪽에서 끝남).
심사위원 3명(hostile, method, reader)의 비평 23건과 저자 측 반박, `round_1.md`, `round_1_verify.json`의
남은 쟁점을 읽었다. 다툼이 있는 사실은 저장소 파일로 직접 확인했다(§1). 이미 4739b59에서 고친 항목
("trails", "no detectable", "reaches the same mean error", "ahead of", "equals")은 다시 올리지 않았다.
새 실험은 돌리지 않았고 GPU도 쓰지 않았다. 확인에는 기존 JSON 읽기와 사칙연산만 썼다.

## 0. 한 단락 요약

T_i 결과는 그대로 선다. 전체 입력 인과 추정기는 CES만 읽는 causal GP 대비 skill +0.113 [+0.085, +0.132]
(cut, pooled), 분할별로는 4/4 분할 × 2모집단에서 0을 제외한다. 바뀐 것은 해석이다. 이 여유 중 BES/ECEI
몫은 분리되지 않았다. 그런데 Claim 1, 제목, 초록, 결론은 모두 그 몫을 fast 채널 공으로 돌린다. 이번 라운드의
가장 큰 과제다. V_rot 쪽에는 round 1 수정이 새 문제를 들여왔다. 초록과 결론이 기대는 1,000 km/s 규칙은
review 중에 사후로 정했고 채점 단계에만 적용하는데, 본문은 이를 밝히지 않는다. 그 규칙 아래 causal GP 대비
여유는 pooled 판독에서만 0을 제외하고 분할별로는 cut 1/4, inclusive 2/4다. persistence 대비만이 분할별
4/4 × 2모집단에서 0을 제외한다(T = 500·1000). 방향은 이렇다. 제목은 질문 하나로 줄이고 surrogate는 해석
도구로 내린다. V_rot는 "fast 채널 결합의 상한 + 이력만으로 persistence 위"라는 범위 결과로 쓴다. surrogate
retention은 PCHIP 기준(72–78 %, 등록 막대)과 causal GP 기준(31–50 %)을 함께 적는다. 이번 수정은 모두
문장·표 정리로 끝나며 새 실험이 필요 없다.

## 1. 위원장이 직접 확인한 사실

| # | 확인 내용 | 근거 |
|---|---|---|
| G1 | T = 1000 km/s에서 V_rot 대 causal GP의 분할별 결과. cut은 s42 +0.088 [+0.019, +0.145], s1 +0.131 [−0.007, +0.259], s7 +0.057 [−0.041, +0.142], s123 +0.085(구간이 0 포함)로 **1/4**다. inclusive는 s42 [+0.004, …]와 s1 [+0.011, …]만 0을 제외해 **2/4**다. 점추정은 4/4 × 2모집단 모두 양수다. pooled 반복별 값은 cut 0.121–0.185, inclusive 0.112–0.179로 8/8 반복이 양수다. 규칙 아래 fold-level 구간은 파일에 없다. | `docs/paper/vrot_failfit_sensitivity.json` `pops.*.T1000` |
| G2 | V_rot 대 persistence(분할별). T = 1000과 T = 500에서 4/4 × 2모집단 모두 0을 제외한다. T = 2000에서는 3/4 × 2모집단이다. PCHIP 분할별은 T = 1000에서 cut 1/4, inclusive 2/4다. offline GP는 T = 500 inclusive에서 3/4 분할이 0 아래로 구간을 둔다. | 같은 파일 |
| G3 | surrogate와 backbone의 causal GP 대비 skill(RMSE에서 계산, cut). s42 +0.024 / +0.078, s1 +0.054 / +0.133, s7 +0.068 / +0.138, s123 +0.033 / +0.105다. 비율은 **0.31, 0.40, 0.50, 0.31**이다. surrogate의 paired 구간은 s42 [−0.012, +0.057], s123 [−0.004, +0.069]로 0을 포함하고 s1·s7에서만 0을 제외한다(2/4). offline GP 대비는 cut 4/4에서 구간이 0 아래다. | `docs/paper/b28_decomposition.json` `b_surrogate` |
| G4 | A3의 T_i 하한. lag 0 구간 하한이 BES w = 10/20/50/100/200 ms에서 0.127/0.152/0.221/0.253/0.256, ECEI에서 0.131/0.137/0.164/0.187/0.218이다. 최소값 +0.127은 w = 10 ms에서 나오므로 **모든 window(10–200 ms)에서 ≥ +0.127**이 성립한다. 본문의 "windows up to 50 ms"는 과소 범위다. 최댓값이 lag 0에서 나오므로 T_i 하한은 lag 0에 대한 진술이다. | `docs/paper/fast_target_coupling_bound.json` `summary.mad` |
| G5 | 강건 판독(§8ax, backbone 대 PCHIP, pooled). cut T_i는 Huber +0.156*, 1 % trim +0.210*이다. inclusive T_i는 Huber +0.130*, trim +0.057(0 포함)이다. V_rot는 Huber −0.060 / −0.056, trim −0.212 / −0.200(cut / inclusive)이다. | `THESIS_RESULTS.md` l.4451–4456 |
| G6 | 라우팅 근거. −0.044/−0.072/−0.067/−0.023은 seq v1(shared encoder)을 held-free 윈도 대조군(iter009)과 짝지은 값이다. 같은 대조군 대비 seq_v2_nops(라우팅만)는 1/4 유의(−0.025, −0.009, −0.049*, −0.011), seq_v2는 0/4(−0.026, −0.003, −0.039, −0.013)다. 날짜는 2026-08-09로 확정 프로토콜 이전이다. 같은 표에서 per-shot 표준화의 T_i 기여는 평균 paired +0.035 → +0.045(대조군 대비)다. routed-vs-shared를 직접 짝지은 구간은 없다. | `THESIS_RESULTS.md` l.1531–1546 |
| G7 | CES만 읽는 학습형 gap 의존 필터(§8bi(b) OU anchor)는 aggregate MSE에서 "ties the causal GP"다. LSTM 잔차가 그 위에 T_i cut +0.11…+0.19를 더한다. 기록에는 구간 없이 동사만 있다. | `THESIS_RESULTS.md` l.5590–5594 |
| G8 | 본문 Table 1의 V_rot 열과 l.437–438의 "18 % (V_rot) below the causal GP"(1 − 23.6/28.8)는 규칙 없이 계산한 값이다. 같은 s42 cut에서 규칙으로 2행을 빼면 causal GP +0.088, PCHIP +0.215, persistence +0.329가 된다. | G1 파일 `per_split`, 본문 l.448–471 |

## 2. 비평별 판정

판정 기준: STANDS = 핵심 주장이 기록과 일치하고 수정이 필요하다. PARTLY = 일부만 맞거나 이미 기록·본문에
답이 있다. REFUTED = 기록이 반증한다.

### 2.1 hostile

| id | 판정 | 근거 |
|---|---|---|
| H2-1 | PARTLY (fatal → major) | 소비자 미명시와 "cannot wait" 틀은 STANDS다(l.109, l.671; verifier 5). "평가 모델이 인과가 아니다"는 과장이다. CES 입력은 t 이전만 읽고, 비인과 단계는 fast 채널 15개의 방전별 affine z뿐이다(round 1 F1). per-shot 효과는 T_i 평균 paired +0.01 수준(G6)이지만 이전 프로토콜이고 대조군 대비 값이다. causal GP 대비 인과 정규화 변형은 측정되지 않았다. 수정: 인과 정규화 변형이 이 프로토콜에서 미검이라는 점을 초록 수준까지 올리고, 소비자 문장은 빼거나 "untested"로 한정한다. |
| H2-2 | PARTLY | (ii) 분할별 개수 누락과 (iv) 사후 규칙 미공개는 STANDS다(G1, verifier 1–2). (i) V_rot running-mean arm이 없다는 점은 사실이다. 다만 causal GP 자체가 같은 16개 과거 CES의 noise-aware smoother이므로 "smoother와 구별 불가"는 과장이다(PARTLY). (iii) offline GP 열세는 이미 l.537–539에 있다(REFUTED). "scoring only라 backbone에 불리할 뿐"이라는 반박도 맞다. 그래도 공개는 필요하다. |
| H2-3 | PARTLY | "1 % 행에만 산다"는 G5로 반박된다. cut T_i는 1 % trim +0.210*, Huber +0.156*로 남는다. 다만 inclusive trim +0.057은 0을 포함하므로 cut 한정으로 적어야 한다. "4 %"는 RMSE 기준 s42 값이다. MSE skill은 s42 +0.078, pooled +0.113이다. 사건 분류(ELM·L-H)는 D-alpha·NBI 신호가 없어 불가하다(데이터 요청 5). r/a 미기재와 l.95–96의 pedestal/ELM 동기는 STANDS다. |
| H2-4 | STANDS | 굵은 한 줄 구분은 결측 행에 노출이 없다고 가정한다. 파일에는 결측 원인이 없다(l.198–199). 조건부 문장으로 바꾸고 per-exposure 메타데이터 한계를 Related work로 옮긴다. "그러면 이 논문이 열등하다"는 반대 주장도 미검이므로 양쪽 모두 hedge한다. |
| H2-5 | PARTLY | 주 모델에서 history-only·memoryless 비교가 없다는 점과 Table 3 −0.125의 OOD 해석은 STANDS다. "fast 몫을 전혀 말할 수 없다"는 과장이다. A3 하한(G4), 사건 행 근거(l.571–573), CES-only gap 의존 필터가 causal GP 수준이라는 기록(G7, A2 running mean −0.060…+0.003)이 간접 지시다. 이 셋은 지시이지 귀속이 아니다. "We found no precedent for that conjunction" 문장은 삭제한다(부재-프레임). 확장 3축 서술 자체는 유지한다. |
| H2-6 | STANDS (수치) / PARTLY (해석) | causal GP 기준 retention 0.31–0.50은 G3로 확인했다. 72–78 %는 등록 막대(0.70)에 대한 판독이므로 "부풀림"은 아니다. 하지만 PCHIP 기준만 쓰면 CES 잡음 제거가 상을 받는다. 두 비율을 나란히 적는다. "no low-order function ... reproduces"는 삭제한다(verifier 7, 이번에도 남아 있음). 제목에서 surrogate를 뺀다. |
| H2-7 | PARTLY | 비교 대상 미명시와 이전 프로토콜이라는 점은 STANDS다(G6). "순수 cross-architecture뿐"이라는 주장은 PARTLY다. 같은 대조군을 거친 간접 사다리(v1 4/4 → v2_nops 1/4 → v2 0/4)가 있다. 다만 구간이 없고 가지 크기도 함께 바뀐다. torque 논증을 "증거"에서 "기대"로 옮기라는 요구는 수용한다. |
| H2-8 | STANDS | 제목 약 24단어에 3항목, 초록 약 22개 수치(round 1 목표 약 9). "only one survives as a positive"는 과장이다. V_rot 대 persistence는 G2로 0을 제외한다. 데이터 진술과 elsarticle은 저자 몫이다. |

### 2.2 method

| id | 판정 | 근거 |
|---|---|---|
| r2-1 | STANDS | G1. 저자 반박의 "pooled는 8개 반복 8/8 양수"는 맞다. 그래도 규칙 사본에서 +0.151은 분할별 개수 없이 쓸 수 없다. l.539와 결론을 고친다. |
| r2-2 | STANDS | 저자도 인정한다. 사후, review 중 정의, 채점 단계만 적용, history 입력 미정리라는 네 가지를 공개해야 한다. 초록의 "anchored on fit failures"는 "target or anchor exceeds"로 바꾼다. |
| r2-3 | PARTLY | Table 1 V_rot 열과 "18 %"는 STANDS다(G8). campaign split V_rot는 이미 PCHIP 대비 0 포함으로 적혀 있다. shared-encoder 손실은 모델 대 모델 차이라 anchor 논리가 그대로 옮겨오지 않는다. 그래도 "규칙 없음, 이전 프로토콜"이라는 꼬리표는 붙인다. |
| r2-4 | STANDS (문구) / PARTLY (귀속 추론) | l.436–437 "do not read the future"와 Table 1 2단 구분이 offline 표준화 모델을 인과 블록에 넣는다. 세 블록으로 나눈다. "대조군 +0.041 vs backbone +0.113의 차이가 offline 단계 몫일 수 있다"는 추론은 G6(+0.01 수준)로 대부분 반박된다. 다만 이전 프로토콜이다. surrogate 고역 항이 평균 불변이라는 점은 한 문장으로 넣는다. |
| r2-5 | STANDS | Claim 1·제목·결론의 귀속 문구가 문제다. Table 3 해석은 sensitivity로 바꾼다. surrogate 행에는 주석을 단다. "the three forms tried"로 한정한다. 저자의 Jacobian·사건 행 반박은 "fast 성분이 0은 아니다"를 지지할 뿐 몫을 재지는 않는다. |
| r2-6 | STANDS | H2-7과 같다. 두 행(v1, seq_v2)과 날짜를 적고 "loses skill on 4/4"를 Claim 2에서 뺀다. |
| r2-7 | PARTLY | Comparators 논리상 V_rot의 결정 비교는 causal GP이고, persistence가 헤드라인이면 사소함을 함께 적어야 한다. 이 부분은 STANDS다. 다만 41.1 %가 직전 값 반복인 신호에서 persistence는 운영 기준선이고, 규칙 아래 분할별로 버티는 유일한 비교다(G2). 둘 다 쓰고 V_rot 중앙값 열세(4.3 vs 3.7 km/s, median-AE vs PCHIP −0.488 cut)를 Claim 2에 넣는다. |
| r2-8 | PARTLY | 결론의 "sits at a fit-noise floor"는 STANDS다(verifier도 지적, 아직 남음). 초록 T_i 하한은 REFUTED에 가깝다. G4로 모든 window에서 ≥ +0.127이 성립하고 0.13은 반올림이다. 정확히 하려면 "+0.127, lag 0, every window"로 쓴다. √2 해석은 PARTLY다. 비율 검정과 기울기 검정이 서로 다른 가정을 시험한다는 저자 반박은 타당하다. 그러나 N과 T_i의 상관은 측정되지 않았으므로 "floor"는 쓰지 않는다. |

### 2.3 reader

| id | 판정 | 근거 |
|---|---|---|
| R2-1 | STANDS (문구) / PARTLY (반박 수용) | Claim 1과 Comparators의 "deciding comparison for a fast-diagnostic contribution"은 고친다. 저자가 든 CES-only 필터 세 가지(A2 running mean, §8bi(b) OU anchor, hold-or-jump)는 "학습형 gap 처리만으로 여유가 나온다"는 설명을 약하게 만든다. 다만 G7은 구간 없는 동사이고 비선형 CES-only 시퀀스 모델은 미검이다. 본문에는 지시로만 쓴다. |
| R2-2 | PARTLY | 공개 문장과 문구는 STANDS다. "등록 판독에서도 약하다"는 반박된다. 등록된 pooled causal GP는 +0.361 [+0.045, +0.701] cut으로 0을 제외한다(Table 2). "persistence는 모든 판독에서 깨끗하다"도 틀렸다. 등록 4분할은 3/4이고 T = 2000은 3/4다(G2). |
| R2-3 | PARTLY | (1) T_i 하한 범위는 REFUTED다(G4). 범위 차이는 window가 아니라 lag다. (2) 초록에 PCHIP 중앙값 39.4 eV와 −0.198 [−0.259, −0.135]가 빠진 점은 STANDS다(round 1 ml-7, verifier 5). (3) Limitations가 초록에 없는 hold-or-jump 필터에 기대는 점은 STANDS다. |
| R2-4 | STANDS | G6. |
| R2-5 | PARTLY | 21k rung, Student-t/mixture head, raw-Mirnov 훈련 세부, stage 1–2 서술은 자른다. 결합 상한·라우팅·surrogate 단독 skill은 §4에서 §5로 옮긴다. 반면 gate 문장은 선택 공개(같은 test 분할 재사용)이므로 한 절로 남긴다. 인과 표준화 수치는 round 1 phys-1의 유일한 측정 근거이므로 Limitations에 수치와 함께 남긴다. linear interp 한 절도 PCHIP 허수아비 반론의 답이므로 남긴다. |
| R2-6 | STANDS | 제목에서 surrogate를 빼고 초록에서 한 번만 쓴다("post hoc"). Table 1 행에는 주석을 단다(A2 판정과 verifier 1). |
| R2-7 | STANDS | 초록 약 10개 수치로 줄인다. 결측률은 서론·Fig. 1에서만 쓴다. "TEST rows"(l.357, l.360), "on TRAIN"(l.646)은 소문자로 바꾼다. 주장 문장에서 구간을 전부 빼라는 제안은 수용하지 않는다(결과는 구간으로 쓴다는 규칙). 주장마다 헤드라인 구간 하나만 남긴다. |

## 3. 출판 가능성 (위원장 추정)

round 1 표의 "이번 라운드 후" 값은 round 1 수정이 V_rot 사후 규칙을 공개하지 않은 채 들여온 점을 반영해
"현재" 열에서 약간 낮췄다.

| venue | 판정 | round 1 후(예상) | 현재(4739b59) | round 2 수정 후 | 변화 이유 |
|---|---|---|---|---|---|
| Nuclear Fusion | reject | ~6 % | ~5 % | ~6 % | 단일 chord·캠페인, 소비자 없음, fast 몫 미분리는 그대로다. |
| PPCF | reject | ~6 % | ~5 % | ~6 % | 같은 이유. |
| Fusion Engineering and Design | major revision | ~35 % | ~30 % | ~40 % (이후 작업 1–2 후 ~55 %) | 사후 규칙 미공개와 귀속 과장이 지금 가장 큰 감점이다. 둘 다 문구로 고쳐진다. 남는 것은 인과 정규화 재학습과 history-only 백본이다. |
| Nuclear Engineering and Technology | major revision | ~40 % | ~36 % | ~43 % | 같은 이유, 심사 강도는 낮다. |
| J. Korean Physical Society | major revision | ~45 % | ~42 % | ~48 % | 같은 이유. |
| Machine Learning: Sci. & Tech. | reject / major | ~25 % | ~22 % | ~25 % | 방법론 신규성 부족은 문구로 바뀌지 않는다. |
| NeurIPS ML4PS 2027 workshop | accept 가능권 | ~55 % | ~52 % | ~58 % | 범위를 정직하게 좁히면 워크숍에는 충분하다. |
| ICLR 2027 workshop | 가능권 | ~45 % | ~42 % | ~47 % | 같은 이유. |
| KPS / KNS 2027 춘계 초록 | 거의 확실 | ~95 % | ~95 % | ~95 % | 변화 없음. |
| RSI / HTPD | 범위 불일치 | <5 % | <5 % | <5 % | 계측기 기여 없음. |

## 4. 방향 변경

1. **Claim 1을 귀속에서 비교로 바꾼다.** "BES/ECEI가 +0.113을 더한다"를 "전체 입력 인과 추정기가 CES-only
   causal GP를 +0.113 [+0.085, +0.132] 넘는다. 그중 fast 채널 몫은 재학습한 history-only 백본으로 분리되지
   않았다"로 바꾼다. fast 성분의 존재는 세 지시로 쓴다. A3 하한 ≥ +0.127(모든 window, lag 0), 사건 행
   (+0.116/+0.269 vs −0.011/+0.088, seed 42/1), CES-only running mean이 causal GP와 같은 수준(−0.060…+0.003)이라는
   점이다. 제목에서도 "from Co-located Fast Diagnostics"라는 귀속 표현을 뺀다.
2. **Claim 2는 범위 결과로 쓴다.** 순서는 (a) fast 채널과 V_rot 증분 결합의 잡음 보정 상한 |r| ≤ 0.044,
   (b) 이력만 쓴 추정의 persistence 대비 +0.265 [+0.198, +0.326](사소한 기준선임을 명시), (c) causal GP 대비
   pooled +0.151 [+0.038, +0.290](분할별 1/4 cut), (d) PCHIP 대비는 확립되지 않음, (e) 중앙값 행은
   persistence보다 나쁨이다. "V_rot는 양성"이라는 기존 결정은 persistence 대비로 유지되므로 이 결정과 충돌하지
   않는다. 다만 memory의 "V_rot vs PCHIP 0 제외(B.27c)"는 사후 규칙 아래에서 성립하지 않는다(round 1 이후 작업
   10과 같은 갱신 필요).
3. **surrogate는 해석 도구로 내린다.** 제목에서 빼고 초록에서는 한 번만 쓴다("summarized post hoc").
   retention은 PCHIP 기준 0.72–0.78(등록 막대)과 causal GP 기준 0.31–0.50을 함께 적는다.
4. **인과성 틀.** 소비자를 지어내지 않는다. "다음 CES 노출을 기다릴 수 없는 추정"은 질문의 정의로만 남기고,
   배치형 변형은 이 프로토콜에서 미검이라고 같은 문장에서 말한다.
5. 유지: 두 모집단, shot 군집 bootstrap, two-level 판독(등록 순서 공개), 확장 3축 novelty 서술(부재 문장만
   삭제), gate 선택 공개, 인과 표준화 수치(Limitations로).

## 5. main_short.tex 수정 목록 (영문 원본, 최대 10개; 한글판은 이후 동일 반영)

공통 규칙: 수치는 괄호 안 출처에서만 가져온다. 결과는 점추정 + 95 % 구간 + 0 제외 여부로 쓴다. "first"와
부재-프레임은 금지한다. 내부 라벨(B.xx, §8xx, P0x, stage n, TEST/TRAIN, s42 같은 표기)은 쓰지 않는다
(본문에는 "seed 42"로 쓴다). 본문은 8쪽 안에 둔다. E10의 삭제분이 E2–E9의 추가분을 상쇄한다.

**E1. 제목과 `pdftitle` (l.51, l.55–57).** 질문 하나짜리 제목으로 바꾸고 surrogate를 뺀다. 예:
"Filling 10 ms Charge-Exchange Spectroscopy Gaps in KSTAR: What Co-located Fast Diagnostics Add to the
Ion-Temperature Past, and a Bound for Rotation". 출처: 없음(문구).

**E2. 초록 재작성 (l.67–89), 약 10개 수치.** 내용과 순서는 다음과 같다.
- 결측 8.2 % / 23.9 %.
- T_i: 전체 입력 인과 추정기 대 CES-only causal GP +0.113 [+0.085, +0.132](cut; inclusive도 0 제외).
  "fast 채널 몫은 분리되지 않음"을 같은 문장에 둔다.
- PCHIP 대비 +0.221 [+0.188, +0.248], 그 절반가량은 running mean이 이미 얻는다.
- offline GP 대비 −0.010 [−0.043, +0.014].
- 중앙값 행: 47.2 eV, persistence 46.1 eV, PCHIP 39.4 eV(median-AE skill vs PCHIP −0.198 [−0.259, −0.135]).
- surrogate는 "summarized post hoc by a 20-parameter surrogate" 한 번만 쓰고 retention 숫자는 뺀다.
- V_rot: |r| ≤ 0.044(lag 0–200 ms, window 10–200 ms; T_i 하한은 lag 0에서 모든 window에 +0.127).
- V_rot 대 persistence +0.265 [+0.198, +0.326]. "target 또는 anchor가 1,000 km/s를 넘는 0.05 % 행을 빼는,
  review 중 정한 규칙 아래"라고 쓴다. causal GP와 PCHIP 대비는 "established only in a pooled readout / not
  established"로 한 절에 쓴다.
- 인과 표준화 변형은 미검이라는 한 절.
출처: Table 2(`b27c_hier.json`), A1 `vrot_failfit_sensitivity.json`, A3 `fast_target_coupling_bound.json`,
§8ax(THESIS l.4453), `b28_decomposition.json`.

**E3. 서론 (l.95–96 동기, l.109–111 용도, l.119–120 Comparators, l.127–137 Claims).**
- l.95–96: "pedestal structure, ELM behaviour" 동기를 빼거나 "the chord is at R = 2.198 m on the outboard side;
  its r/a is not mapped here"를 붙인다.
- l.109–111: 용도 문장 뒤에 "with fast channels standardized per discharge, an offline step; a causally
  standardized variant is untested under this protocol"을 붙인다.
- l.119–120: "the deciding comparison for a fast-diagnostic contribution"을 "the deciding deployable
  comparison"으로 바꾼다.
- Claim 1: "the full-input causal estimator exceeds the CES-only causal GP by +0.113 [+0.085, +0.132] (cut;
  +0.118 [+0.076, +0.157] inclusive); the fast-channel share of that margin is not isolated by a retrained
  history-only model". 72–78 % 절은 빼고 §5.4를 가리킨다.
- Claim 2: |r| ≤ 0.044 상한을 앞에 둔다. "letting them reach V_rot loses skill on 4/4 splits"는 삭제한다.
  persistence +0.265 [+0.198, +0.326](review 중 정한 규칙 아래; persistence는 직전 값 복사라 smoother라면
  넘는 기준선임을 명시), causal GP는 pooled +0.151 [+0.038, +0.290](분할별 cut 1/4), 중앙값 행 4.3 vs
  3.7 km/s를 넣는다. "noise floor"를 "consistent with Gaussian line-fit errors (√2 ratio; the √T_i scaling is
  not observed)"로 바꾼다.
출처: Table 2, A1 JSON(G1·G2), 본문 §5.6 수치.

**E4. Related work (l.155–157, l.172–175).**
- NN-CES 굵은 문장을 조건부로 바꾼다. "Rows with no CES exposure are the target of this work; rows whose
  spectrum exists but whose fit was rejected belong to spectral re-analysis such as NN-CES. The files do not
  record which case a missing row is, so the split between the two is not measured."
- Limitations l.668–669의 per-exposure 메타데이터 문장을 여기로 옮긴다.
- "We found no precedent for that conjunction, and 'not found' is not 'does not exist'."를 삭제한다. 확장 3축
  문장은 유지한다.
출처: 없음(문구).

**E5. §4.2 라우팅과 결합 상한 (l.262–274).**
- torque 문장(l.262–265)을 "no torque, beam-power or NTV channel exists here, so little V_rot information is
  expected from these inputs"라는 기대로 바꾼다.
- 결합 상한 수치(l.267–271)는 §5.3으로 옮긴다. T_i 하한은 "+0.127 at lag 0 for every window 10–200 ms"로
  고친다.
- 라우팅 문장은 다음으로 바꾼다. "Under an earlier protocol (before the fit-failure cut), a full-grid variant
  with a shared encoder had paired V_rot skill −0.044, −0.072, −0.067, −0.023 against the windowed control
  (4/4 splits excluding zero); with the branches split at the encoder it had −0.026, −0.003, −0.039, −0.013
  (0/4). No paired shared-vs-split comparison exists under the present protocol, and these values predate
  the V_rot fit-failure rule."
출처: G4(A3 JSON), G6(THESIS l.1536–1539).

**E6. Table 1과 §5.1 (l.436–471).**
- Table 1을 세 블록으로 나눈다: "past CES only", "past CES + fast channels standardized per discharge (offline
  step)", "past + future CES". caption의 "Lower rows read the future"도 이에 맞춘다.
- l.436–437 "Among the methods that do not read the future"는 "Among the methods that do not read future CES
  values"로 바꾼다.
- "18 % (V_rot) below the causal GP"를 삭제한다. caption에 다음을 단다. "V_rot RMSEs are without the
  fit-failure rule of §5.2; under it (2 rows dropped on this split) the backbone's V_rot skill is +0.088
  [+0.019, +0.145] over the causal GP, +0.215 [+0.071, +0.289] over PCHIP and +0.329 [+0.265, +0.368] over
  persistence."
- surrogate 행에 주석을 단다. "an interpretation of the backbone, not a standalone-estimator claim; paired
  skill over the causal GP on this split +0.024 [−0.012, +0.057]". 공간이 모자라면 행을 삭제한다.
출처: A1 JSON `cut.T1000.per_split` s42, `b28_decomposition.json` `b_surrogate` seed 42.

**E7. §5.2 V_rot 단락 (l.523–539).**
- l.528 규칙 도입 직후에 다음 문장을 넣는다. "This rule was set during review, after the share carried by
  these rows was known; it is applied when scoring only, so the history inputs still contain the fit failures
  and no model was retrained."
- +0.151 / +0.132 옆에 분할별 결과를 쓴다: "per split +0.057 to +0.131 (cut, interval excluding zero on 1/4)
  and +0.053 to +0.143 (inclusive, 2/4), every point estimate positive; a fold-level interval under the rule
  was not computed".
- persistence 분할별 결과: 4/4 × 2모집단(T = 500, 1000), 3/4(T = 2000).
- PCHIP 문장 끝에 "so a V_rot margin over PCHIP is not established"를 붙인다.
- l.539는 "The V_rot margin is established against persistence on every split; against the causal GP only in
  the pooled readout."으로 바꾼다.
출처: A1 JSON(G1·G2).

**E8. §5.3 입력 민감도와 귀속 (l.565–576, Table 3 caption).**
- "shows that history is indispensable"과 "removing the fast channels drops that model's skill below the
  interpolator"를 다음으로 바꾼다. "Zeroing a group at evaluation time feeds the control inputs it never saw;
  the No-fast cut row (−0.125) lies below a CES-only running mean (+0.05 to +0.17 over PCHIP), so the table
  measures sensitivity to out-of-distribution inputs, not the contribution of a group."
- 귀속 지시를 한 문장으로 둔다. "A VAL-tuned running mean of the CES past sits at −0.060 to +0.003 against the
  causal GP, so the backbone's margin over the causal GP is close to its margin over that running mean
  (+0.113 to +0.137, 4/4); event rows concentrate it."
- 결합 상한 수치(E5에서 옮긴 것)를 이 절의 V_rot 근거로 둔다.
출처: `b28_decomposition.json` `a_decomposition`(A2), 본문 Table 3, A3 JSON.

**E9. surrogate (§4.5 l.344–389, §5.4 l.615–625).**
- §4.5 stage (1)–(2) 서술을 한 문장으로 줄인다. "A frozen-row readout and a first trained increment form
  fell below the registered 0.70 retention bar; the backbone's Jacobian on the test rows then suggested Eq. 3."
  "TEST rows"는 "test rows"로 바꾼다.
- 단독 skill 문장과 "These are measurements of the model"은 §5.4로 옮긴다.
- §5.4: 0.72–0.78(PCHIP 기준, 등록 막대 0.70) 옆에 "measured against the causal GP, the surrogate keeps 0.31 to
  0.50 of the backbone's skill (cut, point estimates)"를 넣는다.
- "which no low-order function of the fast-channel history reproduces"를 "which the three forms tried here do
  not reproduce"로 바꾼다.
- "d_{c,t} is invariant to the per-discharge mean, so the surrogate's offline dependence is through the
  per-discharge σ only"를 한 문장 넣는다.
출처: `b28_decomposition.json` RMSE에서 계산한 skill 비(G3), 본문 기존 수치.

**E10. 정리: §4.4·§5.3·§5.6·결론 (공간 확보).**
- 삭제: "Size and cost"의 21,498-parameter rung 문장(지연 0.97/1.49 ms는 한 절로 남길 수 있음),
  Student-t/mixture predictive head 문장(l.653–656), raw-Mirnov 훈련 시험 세부. raw-Mirnov는 24/24 coil, lag 0
  snapshot 사실만 남긴다.
- §5.6 hold-or-jump 필터는 두 문장으로 줄인다. "on TRAIN"은 "on the training files"로 바꾼다.
- §5.6에 cut T_i 강건 판독 한 문장을 추가한다. "under a symmetric 1 % trim and a Huber loss at the noise bound
  the skill over PCHIP stays +0.210 and +0.156 (both intervals excluding zero, cut); in the inclusive
  population the trimmed skill is +0.057 (interval including zero)."
- Limitations:
  - "the median is read from the hold-or-jump filter"를 "typical rows are not improved over PCHIP"로 바꾼다.
  - 사후 규칙 두 가지를 나란히 쓴다(surrogate 형태, 1,000 km/s V_rot 규칙).
  - 인과 표준화 한 절: "under an earlier protocol a training-statistics variant kept +0.22 to +0.30 over PCHIP
    (4/4), but it was not scored against the causal GP".
- 결론: 첫 문장을 "the full-input causal estimator exceeds the CES-only causal GP ...; the fast-channel share is
  not isolated; fast channels standardized per discharge"로 바꾼다. V_rot 문장은 E7의 l.539 문구로 맞춘다.
  "its 10 ms scatter sits at a fit-noise floor"는 "is consistent with Gaussian line-fit errors"로 바꾼다.
출처: §8ax(THESIS l.4453), round 1 F2(`data/.seq_v2_nops_lstm_s*/bootstrap_summary.json`), 본문 기존 수치.

## 6. 이후 작업 (round 1 목록 대비 변경분)

- **새로 추가, CPU, 기존 npz만, 각 25분 이내:**
  - (a) V_rot running-mean(VAL 조정 EWMA) arm을 A1 규칙 아래 채점한다. backbone과의 paired 구간을 낸다.
    0을 포함하면 Claim 2는 "CES-history smoother 수준"으로 더 좁힌다.
  - (b) surrogate의 a_c = 0 변형(K(s)만 쓰는 4-parameter 필터)을 frozen 행에서 causal GP·backbone과 짝짓는다.
    fast 몫을 재는 가장 싼 측정이다.
  - (c) Table 1 V_rot 열을 규칙 아래 RMSE로 다시 낸다.
  - (d) per_shot = 0 산출물(`data/.seq_v2_nops_lstm_s*`)에 causal GP arm을 붙여 다시 채점한다.
- **우선순위 유지(GPU):** round 1 이후 작업 1(인과 정규화 seq_v2, 확정 프로토콜)과 2(history-only seq_v2)가
  여전히 FED 수락의 관문이다. 2는 이제 Claim 1 귀속을 직접 결정하므로 1과 같은 순위로 올린다.
- **새로 추가(GPU, 낮은 순위):** 확정 프로토콜 아래 seq_v2의 V_rot 가지에 22채널을 모두 넣는 직접 라우팅
  ablation. 이게 있어야 E5의 "earlier protocol" 꼬리표를 뗄 수 있다.
- **데이터 요청 5 강화:** 결측·held·fit-failure 행 중 원 노출이 있는 비율. NN-CES와의 구분(E4)을 측정으로
  바꾼다.
- **저자 결정:** 데이터 진술 단락(l.680–681), elsarticle 이행, 1,000 km/s 규칙을 다음 배치부터 사전 등록할지
  (round 1 이후 작업 6).
- **memory 갱신:** "V_rot vs PCHIP 0 제외(B.27c)"에 "사후 1,000 km/s 규칙 아래에서는 성립하지 않음(cut
  [+0.0003, +0.170], inclusive 0 포함)"을 덧붙인다.
