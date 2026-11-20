# 인수인계 문서 (2026-11-21)

이 저장소의 일을 다른 에이전트(Codex 등)나 사람이 이어받아 실험을 계속할 수 있게 쓴 문서다. 규칙의 원문은 `CLAUDE.md`와 `PAPER_DESIGN.md`에 있고, 이 문서는 그것들을 어떻게 쓰는지, 무엇이 어디 있는지, 다음에 무엇을 할지를 적는다. 수치의 원천은 `PROJECT_KNOWLEDGE.md`와 `docs/paper/`의 결과 파일이다. 이 문서에는 수치를 거의 적지 않는다.

## 0. 한눈에

- **무엇을 하는가.** KSTAR 2022 캠페인의 pedestal top 측정선(R = 2.198 m)에서 CES 이온 온도를, 다음 CES 값을 기다리지 않고 10 ms마다 추정한다(nowcasting). 입력은 그 측정선의 CES 이력과 같은 위치의 빠른 진단(BES 9채널, ECEI 4채널)이다.
- **지금 상태.** 원고(`docs/paper/ppcf/main_ppcf.tex`, 본문 17쪽, 보충 28쪽)는 저자의 틀로 쓰여 있고 빌드와 수치 검사를 통과한다. 등록 배치는 B.86까지 있다. 독립 채점은 PPCF 13–20 %에 머문다(`docs/paper/review_loop/`).
- **GPU는 꺼 둔 상태다.** 저자의 지시(2026-11-21): GPU 실험은 이 문서에 적어 두고 나중에 돌린다(9절).

## 1. 먼저 읽을 것 (이 순서로)

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

B.85는 9절의 GPU 배치를 위해 비워 둔 번호다.

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

### 9.1 GPU (나중에 돌린다; 저자의 지시)

**(가) B.85: 시나리오를 섞어 학습한 순환 신경망.** 가장 먼저 할 GPU 배치다. 아직 등록하지 않았다. 돌리기 전에 5절대로 등록한다.

- **왜.** 지금 논문의 estimator는 인과 GP 위에 얹은 boosted tree(손으로 만든 스칼라 24개)이고 가용 조건마다 다시 맞춰야 한다. 조건 하나에 맞춘 것을 다른 조건에 쓰면 빠른 진단의 증분이 거의 사라지고, 본 적 없는 조건으로 옮겨 가지 않으며, 원시 창을 주면 못 배운다(B.84, `PROJECT_KNOWLEDGE.md` 4.69절). 5-parameter filter는 갱신 없이 돌리면 1 s 뒤에 발산한다(B.71). 저자의 말: 2차 estimator를 rule로 얹은 것은 시나리오 연결에 좋지 않다.
- **무엇을.** 순환 신경망(`experiments/seq/model_seq_v2.py`; 입력은 `seq_data.py`의 22채널: 빠른 채널, 시간 간격, 표적별 "마지막 관측값, 그 뒤 지난 시간, 관측 여부")을 **CES 이력의 가용 조건을 학습 중에 무작위로 바꿔 가며** 한 번 학습한다. 조건은 B.75의 규칙들이다: 직전 프레임까지(D10), 지연 d(50–1000 ms), 빔 변조 일정(S). segment마다 조건 하나를 뽑아 "마지막으로 보이는 값"과 "그 값의 나이"를 그 조건대로 만든다. k번째 프레임 규칙(K10)은 학습에 넣지 않고 시험에만 쓴다(본 적 없는 시나리오).
- **견줄 것.** 같은 행에서: 조건별로 맞춘 TX와 T0(`data/.b75/fits/`), 한 번 맞춘 TXall(`data/.b84/fits/`), 직전 프레임 조건의 TX[F](`data/.b73/trees/`), 빠른 채널을 뺀 쌍둥이 신경망. 표적은 그 프레임의 값과 이후 수준 둘 다(출력 head 둘 또는 따로 학습; 등록에서 정한다).
- **단위.** V1의 재분할 r2(5 fold), V3(4), V4a(1)부터. 초기화는 셋. 학습 설정(epoch 상한, 조기 종료, D1 처리)은 `experiments/b64_two_stage/run_b64.py`의 env를 그대로 쓴다.
- **미리 정할 진술(안).** (1) 직전 프레임 조건, 그 프레임의 값: 신경망이 TX보다 5 % 넘게 나쁘지 않다. (2) D650과 S, 이후 수준: 조건별 TX보다 5 % 넘게 나쁘지 않다. (3) K10: 그 조건에 맞춘 T0보다 낫다(옮겨 가는가). (4) V3에서 (2). (5) 쌍둥이 대비 증분이 D650에서 0 위. (1)–(3)이 성립하면 논문의 estimator를 이 신경망으로 바꾸고 "모델 하나가 nowcast와 모든 시나리오를 잇는다"고 쓴다. 아니면 지금 것을 유지하고 그 결과를 한계에 적는다.
- **구현 주의.** 기존 학습 코드를 고치지 말고 새 디렉터리에서 `seq_data`의 블록을 감싸 조건별 이력 입력을 만든다. B.75의 `inputs/{rule}.npz`는 D1 행에 대한 것이라, 격자의 모든 행에 대해 보이는 값을 다시 계산해야 한다(B.75 부록 1의 정의대로). 누출 점검: 보이지 않아야 할 값을 1000 eV 올려도 입력이 같아야 한다.

**(나) B.69의 남은 15회.** 2단계 estimator의 시간 분할 확증 시험이다(`PREREGISTRATION_B69.md`, 40회 가운데 25회에서 멈춤). 등록된 것이므로 끝내 보고해야 한다. 채점자가 바깥 run 여덟 개를 이미 읽었다는 것을 보고에 밝힌다.

**(다) MAST에서 같은 순환 신경망, 장치 간 전이, 층별 비교.** 저자가 정한 흐름이다(`PROJECT_KNOWLEDGE.md` 7절의 표). (가) 뒤에 한다.

### 9.2 CPU로 할 수 있는 것

1. **B.86은 끝났다**(`PROJECT_KNOWLEDGE.md` 4.71절). 남은 것: 원고의 MAST 절에 대역별 표가 들어갔는지 확인한다(`make_tables_v2.py`의 표, `main_ppcf.tex`의 문장).
2. **사건 기준의 쓸모(설계 초안).** 저자의 목적에 있는 "적용했을 때 더 잘 작동함"을 수준이 급히 떨어지는 사건으로 보인다. 사건은 이후 수준이 앞선 300 ms의 중앙값보다 정해 둔 비율 넘게 떨어지는 첫 행으로 정의한다. 견줄 것은 (a) 빔 변조 일정대로 늦게 오는 CES 값만 보는 감시, (b) T0, (c) TX가 같은 문턱을 넘는 시각이다. 보고는 놓친 사건의 몫, 거짓 경보율, 검출 지연의 중앙값. 저장된 추정(`data/.b75/fits/`)으로 된다. 등록부터 한다.
3. **기다림의 값어치(18차 채점자의 권고; 논문의 첫 질문에 들어간다).** 같은 boosted regression에 다음 값 1, 2, 5개를 보여 준 기준을 만들어, 허용 지연 0, 1, 2, 5프레임에 따른 오차를 한 그림으로 낸다. 지금의 비교는 학습한 인과 estimator 대 학습하지 않은 양쪽 보간이라 두 차이가 섞여 있다. 저장된 행 표(`data/.b73/rows.npz`)와 B.83의 기준으로 된다. 등록부터 한다.
4. **표적 검증(17차·18차 채점자의 권고).** 다섯 프레임을 둘로 나눈 잡음 바닥, notch 뒤 100 ms 안의 행과 BES 수준이 가장 낮은 행을 뺀 주 표의 재채점, 컷을 2 keV와 4 keV로 바꾼 판독, 절대 오차를 손실로 한 fit.
5. **`experiments/seq/`의 옛 변형 정리.** `seq_models.py`가 지운 배치들의 구조 변형(v4k, v2d, b28, b3, tcn, xfmr 등)을 한 사전에 import한다. 지금 줄기는 v2, v2h, v2e·v2b·v2n만 쓴다. 변형 파일과 그 시험을 지우려면 `seq_models.py`를 고쳐야 하고, 저장된 학습 결과가 그대로 읽히는지(B.70의 채점 경로) 확인해야 한다. GPU 배치가 돌지 않을 때 한다.
6. **구간의 보정.** 군집이 14–19개인 시간순 설계에서 섞은 입력으로 구간의 포함률을 잰다(B.81의 대조를 여러 seed로).

### 9.3 저자의 결정을 기다리는 것

- GPU를 언제 다시 켤지(9.1).
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
