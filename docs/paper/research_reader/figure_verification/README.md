# 추가 USD5 상한의 근거·그림 검증

저자의 추가 요청에 따라 공식 Claude API를 새 문맥에서 세 번 호출했다. **Opus의 본문/원문 검증 → Fable의 실제 집계 그림·HTML 카드 검토 → Opus의 교정 근거 재확인** 순서다. 세 응답 모두 정상 종료했고 사용량·요청/응답 해시와 상한은 `api_usage.json`에 남겼다. 실제 반환 usage의 공개 단가 추산은 USD2.04744이며 청구서 확인액은 아니다. 추가 요청은 종료했다. 합격률이나 학술지 수락 판정은 하지 않았다.

## 실제로 제공하고 확인한 범위

- 첫 Opus에는 현재 한국어 원고 전체, 익명화한 기존 영문 방법·결과·고찰·결론, 인용 문헌별 감사와 NN-CES·RTCAKENN·EPED의 공개 원문 PDF 전체를 제공했다. 다른 문헌은 기존 감사 범위의 확인이지 새 PDF 전문 검토가 아니다.
- Fable에는 집계 그림 1(b), 도식 2, 그림 4·5·6, 집계 7(a)를 실제 PNG로 제공했다. 원자료 시계열이 있는 그림 1(a)와 3은 제외했고, 보수적으로 7(b)도 외부 제공에서 제외했다. 그림 2·4는 실제 새 HTML 카드 화면도 제공했다. 외부 제공 이미지는 허용 패널만 렌더한 픽셀이므로 가려진 원자료가 SVG/PDF 내부에 남지 않는다. 안전 입력 목록과 해시는 `api_safe_figures.json`에 있다.
- 보호용으로 지운 V4 적합 경계와 방전 블록 이름을 Fable이 실제 그림의 누락으로 해석했다. 실제 원본과 최종 HTML에는 그 라벨이 있으므로 이 제안을 그대로 적용하지 않았다. `caption_review_decisions.json`에 판정을 남겼다.
- 그림 일곱 개 전부는 로컬에서 원본·compiled PDF·HTML을 직접 대조했다. `LOCAL_FIGURE_QA.md`는 원본 PDF 그림 stream과 기존 crop의 일치, 외부 제공 가능 범위와 세부 확인을 기록한다. 원래 plot의 데이터와 축을 다시 계산하거나 바꾸지 않았다.

## 반영한 교정과 표시 방식

`applied_corrections.json`의 본문 교정은 EPED의 폭·높이가 안정성 제약을 함께 만족한다는 설명과, 중앙 절대오차 비교가 무작위 검증에 해당한다는 한정이다. 실제 EPED 원문과 기존 Results/Table3를 독립 대조했다. 기존 실험의 성능 수치를 새로 생성하지 않았다.

그림은 원고 페이지 crop 대신 **원본 standalone plot PDF에서 변환한 벡터 SVG**를 독립 이미지로 내장한다. SVG를 이미지 데이터 URL로 격리하여 그림 간 글꼴/클립 ID가 충돌하지 않으며, 글자는 경로로 보존한다. 원본 PDF 경로와 SHA는 `../longform/exhibits.json`의 `asset_pdf`, `asset_sha256`으로 결속한다. 관련 소절, 그림 번호와 확대 링크는 유지하고 한국어 캡션에서 패널·기호·표적·기준선·검증 설계·해석 범위를 설명했다. 표 열한 개는 검증된 원고의 기존 crop을 그대로 유지했다.

특히 그림 6의 가로 오차막대는 **등가 보고 간격의 묶음 bootstrap 구간**이다. RMSE의 세로 구간이 아니다. `fig_v2.py:fig_sparse`와 `b78_sparse/readout78.py:eqrate`를 확인하여 각 재표집에서 등가 간격을 다시 구한 뒤 분위수를 사용한다는 정의까지 바로잡았다. 초기 로컬 검토의 단순한 '구간이 아님' 판정은 이 확인으로 정정했다. 세 번째 Opus 응답은 이 코드와 교정 캡션을 직접 대조했다.

## 재생성

```powershell
py ces_prediction/research/thesis_reader/integrate_longform.py
py ces_prediction/research/thesis_reader/build_reader.py
py ces_prediction/research/thesis_reader/verify_reader.py
```

`integrate_longform.py`가 이번 본문 교정도 읽으므로 이전 문장으로 돌아가지 않는다. `build_reader.py`는 figure 자산 해시와 원고 PDF 결속을 검사한 뒤 SVG를 생성한다. `verify_reader.py`는 그림 일곱 개의 벡터 표시, 표·그림 총 열여덟 개, 인용·탐색·오프라인·모바일과 본문 문단의 수치 상한을 검사한다. `verification.json`이 현재 HTML 해시와 최종 검증·비용의 기준 기록이다. 앞선 `../longform/final_verification.json`의 HTML 해시는 이번 그림 변경 전 버전이다.

유료 검증을 다시 실행할 필요는 없다. `verify_with_claude.py`는 이번 bounded workflow의 요청/응답을 보존하며 자료나 요청이 바뀌면 묵시적으로 재사용·재결제하지 않는다. 정확한 요청/응답과 보호용 이미지는 ignored `data/.thesis_figure_verification/`에만 보존한다. 새 적합·GPU·과학적 판독은 없다.
