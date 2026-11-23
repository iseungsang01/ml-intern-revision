# B.99 — GPU 확인 학습 재개

2026-11-23 15:42 KST 관측. [기록 JSON](b99_recovery.json)이 source/clock/검사/실제 GPU PID의 근거다. 사전등록과 B.94 부록4는 `2a674b5`, 새 metadata 감독기는 `58a2438`에서 실행 전에 커밋·푸시했다.

**원래 네 slot GPU 확인 학습을 재개했다.** 15:41에는 네 trainer PID가 모두 NVIDIA compute-process 목록에 있었고 GPU 사용률76%를 관측했다. 15:42 기록은73%,2891MiB다. 확인 학습23/46개 완료를 보존했고,4개 실행·19개 대기다. Pilot12개와 production smokes도 원래 정상 종료 증거로 재검증했다. `TX/TH`는 여기서 TCN fast/history 쌍이며 원고의 기존 GP/tree TX와 다르다.

## 원인 조사와 보완

과거 실패 기록에는 PermissionError만 남아 있어 정확한 파일·잠금 주체를 확정할 수 없다. 기존 감독기는 queue phase 읽기의 PermissionError를 처리하지 않고, 감독기 예외 발생 시 소유한 tree를 종료하는 경로가 있다. 실제 Windows exclusive-sharing handle로 원래 read 실패를 재현했다. 이는 가능한 실패 경로의 재현이며 과거 원인을 입증한 것은 아니다.

새 `monitor99.py`는 원래 동결 감독기를 호출하되 metadata 경로만 보완한다. phase read는 유한 시간 재시도 후 `metadata_unavailable`로 표시하고, bounded atomic-write 이후 heartbeat PermissionError는 별도 local fallback/로그로 남긴다. telemetry 실패만으로 과학 계산을 종료하지 않는다. 그 밖의 실패 정책·원래 suite 마감·자식 hard timeout은 유지한다. 전체 traceback을 새 local 감독기 stderr에 보존한다. frozen numeric/controller 파일은 바꾸지 않았다.

실제 timestamp JSON 잠금과 무해한 timer child를 사용한 검사가 통과했다. read/write가 막힌 동안 child가 생존하고 정상 종료했으며 잠금 해제 후 복구됐다. plasma 관측·모형·GPU probe를 생성한 검사가 아니다.

## 보존과 한정 재시작

중단된 네 fit `V3_q2__1701__TH`, `V3_q3__1701__TX`, `V3_q3__1701__TH`, `V3_q4__1701__TX`는 전체 디렉터리·hash manifest를 `data/.b94/interrupted_permission99/`에 보존했다. 완료/정상 종료 증거가 없으므로 완료 결과로 세지 않는다. RNG/elapsed-budget의 정확한 복구 상태가 없어 저자 지시와 부록4에 따라 이 네 fit만 원래 seed로 새로 시작했다. 기존 completed fit은 재학습하지 않았다. partial validation 값을 선택에 사용하지 않았다.

원래 verifier로 완료 결과를 검증했고, 독립 검사에서 동결 소스·완료/exit hash·부분 결과 보존·마감·옛 process 부재를 다시 확인했다. 새 one-shot hidden task는 `B93_GPU_B94_recovery99`다. 재시도 설정은 없고 원래 `b96_four/production96.py`를 실행한다. 초기 재검증 중 GPU가 비어 있었던 시간과 실제 학습 재개를 구별한다.

원래 마감은 **2026-11-24 01:54:57 KST**로 연장하지 않았다. 새로운 예측 우열·TEST 결론·물리 발견은 이 복구에서 나오지 않는다. [현재 B.94 상태](b94_status.md), 별도의 수식 모형 pilot B.98을 구별해 읽는다.
