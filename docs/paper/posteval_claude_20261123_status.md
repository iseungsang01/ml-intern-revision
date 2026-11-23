# 평가 후 Claude 연구 정리 — 자동 실행 설정

2026-11-23 18:03 KST에 일회성 숨김 작업 `B93_Claude_PostEval_20261123`을 등록·시작했다. Python worker process와 작업의 Running 상태를 확인했다. 당시 원래 평가 단계는 prediction이므로 연구 검토는 `waiting_for_audited_completion`으로 대기 중이다. **평가나 연구 검토가 완료됐다는 보고가 아니다.**

계약·구현은 `84e0807`에서 등록 전에 커밋·푸시했다. 사용자의 주요 설계·해석·연구 판단에 Claude API를 연결하는 선호는 AGENTS.md와 CLAUDE.md에도 반영했다. 단순 파일 확인까지 유료 호출을 강제하는 규칙은 아니다.

## 실행 순서

1. 원래 큐 완료, 감독기 정상 exit0, 원래 post audit 통과, confirmation46개와 selection/readout hash 및 복사된 원본 결과 일치를 확인한다. 실패·중단·불일치면 부분 TEST 결과를 Claude에 보내지 않는다.
2. Fable이 근거를 읽고 검토 기록을 제출한다.
3. Opus가 통계·지원 행·학습 예산·물리 해석의 반대 검토를 제출한다.
4. Fable이 기존 실험과의 중복을 확인해 후속 연구를 최대 세 개, 즉시 할 일 하나로 줄인다.
5. Opus가 최종 수정한다. host가 근거 pointer를 검사하고, 원본 수치표와 제안된 후속 연구를 구분해 저장한다.

단계당 최대 네 paid API round, 본 검토 최대16회다. 요청·응답·사용량을 local ignored cache에 보존한다. Claude는 명시된 익명 집계만 조회하고 자기 단계의 구조화 초안만 제출할 수 있다. shell·임의 파일 접근·학습·현재 원고 수정 도구는 없다. 실패한 불확실 유료 호출을 자동 반복하지 않는다. 새 연구는 TEST-informed 탐색으로 표시하며 실행 전 별도 사전등록과 필요한 독립 확인을 정리한다.

## 확인한 것과 아직 남은 것

실제 Fable API 연결 시험에서 context 조회와 구조화 findings 제출이 두 paid 호출로 동작했다. 이 시험은 이미 감사된 Lean 상태와 현재 평가 phase 등 운영 metadata만 전달했으며, 진행 중 B.94 성능은 노출하지 않았다. 초기 시험의 응답·사용량은 보존했고, 본 worker는 요청 원문도 저장하도록 보완했다. Claude가 지적한 Lean 감사와 원래 post 감사의 혼동 가능성을 반영해 gate boolean을 분리하고, setup에서 없는 B94 경로를 안내하지 않도록 했다.

현재 분석 결과가 없으므로 이를 이용한 의미적 판독이나 연구 후보 검증은 아직 수행하지 않았다. host의 pointer 검증이 새로운 가설의 타당성을 증명하지는 않는다. 원래 평가가 끝나면 정상 종료·최종 감사 gate를 먼저 만족해야 한다. 완료 후 문서는 연구 초안이며, 원고의 수치·물리 주장으로 자동 승격하지 않는다.

현재 live 상태: `data/.claude_posteval_20261123/status.json`. 작업의 stdout/stderr와 모든 API attempt는 같은 ignored 폴더에 있다. 완료 예정 산출물: `docs/paper/posteval_claude_20261123.md`와 `.json`. 새 실험을 실행하거나 현재 평가 마감을 연장하지 않는다.
# 2026-11-23 23:31 KST: all stages and lead correction complete

Original four-stage workflow completed23:03:47 with10 paid substantive calls, saved all requests/responses and generated original draft; local completion notification delivered. Lead review found missing skill formula/sign confirmation and availability-delay versus forecast-horizon ambiguity. Original draft remains unchanged. A separate bounded Opus tool review received compact frozen-source definitions and both scopes/all designs; completed23:29 in3 additional calls, total13 within16. Original audit/publication/normal-exit gate rechecked after the response. Exact per-cell interval classifications were generated as an independent reference for broad prose claims. The corrected plan is `posteval_lead_review_20261123.md/json`; it is exploratory and no automatic training occurred. Local lead check: `data/.claude_posteval_lead_20261123/lead_checks.json`.

## Earlier23:01 running observation

The original final audit, explicit publication and normal queue/supervisor exit completed; the unchanged original E.gate is true. After confirming no previous substantive requests, the worker was explicitly started against completed evidence. Real evidence_review call1/call2 requests/responses and call3 request were observed23:01. This is actual paid evidence analysis, no longer a waiting-only task. Original models and maximum16 rounds remain bound. Independent notifier103 delivered the stage notification and monitors errors/5min progress. The early publication waiter had stopped on stale failed queue metadata before completion, with zero substantive requests; it was not a paid-call replay.

## Earlier22:43 wait

B.102 artifact validation passed without recalculating statistics. A live new supervisor/queue was established, then the existing Claude task was explicitly restarted after source binding and absence of prior substantive requests were checked. Fresh22:43 waiting status observed. The separate local notifier is also live and delivery-tested; it alerts on Claude failure but never restarts or replays a paid Claude stage. Original independent final audit, normal exit and matching report hashes remain mandatory.

## Earlier21:34 stop

All136 repair cells completed, but the evaluation stopped on a Windows manifest-key mismatch before the final audit. The Claude waiter stopped21:34:21; no substantive performance review started. B.102 may explicitly restart the original source-bound waiter only after new queue liveness and absence of prior substantive requests are checked. The new notifier never automatically launches Claude or replays a paid request.

## Earlier21:19 observation

After registered B.101 preparation and live new supervisor/queue checks, the same source-bound Claude task was explicitly restarted. Fresh status21:19 is waiting_for_audited_completion/readout. No prior substantive research-review request exists; no uncertain paid call was replayed. Original final audit, matching copies and normal exit remain mandatory. Below21:05 describes the previous stopped attempt.

## Earlier21:05 stop

The original post audit rejected B.100 at21:04. The watcher correctly stopped with `Upstream evaluation stopped; no partial-result research review`. No substantive request was made. B.101 separates a neural/tree metric namespace collision and must pass the original audit and normal-completion gates before this worker can analyze any result. Restart is explicit after a live recovery queue is established; committed worker code and paid-round limits remain unchanged.

## Earlier20:18 observation

After B.100 fit/cache verification and a live new evaluation supervisor/queue were established, the existing Claude task was explicitly enabled/started. Observed Running with fresh20:18 status `waiting_for_audited_completion`, upstream `prediction`. Its committed sources match the original config. No prior substantive agent request exists, so this does not replay a paid research-review stage. Original final audit/normal-exit/hash gates and maximum16 substantive rounds remain in force. This is a running wait, not a completed research report.

## 20:13 recovery preparation

The18:24 author pause suspended this watcher as well as evaluation and disabled both tasks. The20:06 OS reboot destroyed the paused processes. Evaluation-only recovery B.100 will establish a live supervisor/queue before the existing one-shot watcher is explicitly restarted. Its code, source binding, original wait deadline and maximum16 substantive paid rounds remain unchanged. No partial B.94 metrics or automatic new training are authorized. Current local status may still contain the pre-reboot waiting timestamp until restart.

