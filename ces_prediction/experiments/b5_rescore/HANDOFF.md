# B.5 인계 (2026-08-16 00:40 중단 시점)

## 어디까지 됐나

`run_b5.py --stage all --resume` 배치를 승상님 지시로 중단. 완료된 GPU 단계(전부 재개 시 스킵됨):

| 단계 | 상태 | 산출물 |
|---|---|---|
| incl_window (포함 모집단 윈도우 재채점) | **4/4 완료**, 얼린 스윕 npz와 bit-identical 검증 | `data/.b5i_w2_s*` |
| incl_backbone (포함 모집단 seq_v2) | **4/4 완료** (val+test) | `data/.b5i_seqv2_s*` |
| val_backbone (컷 백본 val 채점 + bootstrap) | **4/4 완료** | `data/.b1_seqv2_s*_i*` |
| ladder_incl (포함 모집단 anchor·b3k8) | **4/4 완료** | `data/.b5i_anchor_s*`, `data/.b5i_b3k8_s*` |
| campaign (시간 분할, 두 모집단) | **20/20 완료** | `data/.b5_camp_{cut,incl}_{win,winps,seq}_s*` |
| cut_sens (2.5/4 keV 컷 백본) | **4/8** — 2500 완료(4), 4000 미완 | `data/.b5c2500_seqv2_s*` (완), `.b5c4000_*` |
| ablate (윈도우 modality ablation) | 0/16 | — |
| analyze (CPU) | 컷 모집단 dry-run만 확인됨 | `data/.b5_summary.json`(부분) |

## 재개 명령 (repo root, 그대로 실행)

```powershell
py ces_prediction/experiments/b5_rescore/run_b5.py --stage cut_sens --resume
py ces_prediction/experiments/b5_rescore/run_b5.py --stage ablate --resume
py ces_prediction/experiments/b5_rescore/run_b5.py --stage analyze
```
(또는 한 번에 `--stage all --resume` — 완료된 단계는 자동 스킵. 예상 남은 시간 ≈ 1.5–2 h.)
백그라운드로 띄울 때 `| grep`으로 파이프하지 말 것 — stdout이 버퍼링돼 진행이 안 보임. 진행은
`ls -d data/.b5_abl_*/paired_vs_base.json | wc -l` 같은 산출물 카운트로 확인.

## 이미 나온 결과 (기록용 요지)

- **포함(p100) 헤드라인**: seq_v2 `T_i` vs PCHIP PASS 4/4(+0.225/+0.238/+0.292/+0.316), vs 인과 GP PASS 4/4,
  윈도우 대비 paired 4/4 양수(2/4 유의). 컷(B.1)과 합쳐 §1.1 무조건부 기준 충족.
- **포함 모집단 사다리**: b3k8은 anchor 대비 4/4 유의 우세지만 **seq_v2 대비 −0.160/−0.200/−0.203/−0.214 4/4
  유의 열세**, 윈도우 대비 3/4 열세. 스파이크-anchor 행(persistence 오차 >2 keV, 0.6–1.3%)이 b3 T_i SSE의
  73–83% → §8z "21k = 백본"은 **컷 모집단 조건부**로 재서술 필요.
- **캠페인(컷, 완료 8/12 시점 요약)**: 윈도우 OFF `T_i` vs PCHIP 1/3 PASS(오프라인 우위 붕괴 재현), vs persistence
  전부 PASS; 윈도우 ON 3/3 PASS(OFF 대비 +0.08\*/+0.02/+0.10\*, §8s 재확인); seq_v2 2/2 PASS(OFF 대비 +0.16\*/+0.09\*).
  전체 20 run 수치는 `analyze` 단계가 `.b5_summary.json`의 `campaign` 블록으로 산출.
- **스파이크 구조 진단(감사 메모)**: `T_i > 3 keV` 1,197행 = 951 run, 85% 단일행, 78% 고립 스파이크(양옆 <2 keV),
  이웃 대비 중앙값 15× 점프, 지속 구간(≥5행) 17 run(2%), 274 shot에 분산. 단, 한 샘플 이상치는 **양방향**
  (상향 ≥2× 2,805행 vs 하향 dip ≥2× 4,687행)이고 3 keV 컷은 상향 꼬리 22%만 제거 → 값 컷은 부분적·비대칭.
  fit 품질 메타데이터 도착 시 대체(사전등록 §1). 모집단 선택(p100 단일 헤드라인 여부)은 **승상님 결정 대기**.

## 완료 후 할 일

1. `data/.b5_summary.json` 확인 → THESIS_RESULTS.md **§8ab** 작성(두 모집단 표: 헤드라인·MNAR·conformal·peak·
   large-gap·캠페인·컷 민감도·ablation·커버리지/PR2; 사다리 조건부 재서술; 스파이크 진단 메모).
2. `PREREGISTRATION_W2.md` §6 B.5에 집행 결과 블록, `experiments/README.md` b5 행 갱신(pending → verdict),
   PROJECT_KNOWLEDGE.md 교훈, 메모리 갱신, 커밋·푸시.
3. 승상님 결정 사항: ① p100 단일 헤드라인 vs 공동 1차 유지, ② `V_rot` 스파이크 감사/컷 규칙, ③ B.6 kHz Mirnov 특징
   도착 여부(현재 `data/`에 없음).
