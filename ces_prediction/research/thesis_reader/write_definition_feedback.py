"""Readable report of reviewer advice; does not edit scientific paper/protocol."""
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs/paper/research_reader/definition_audit'

def read(path):return json.loads(path.read_text(encoding='utf8'))

def main():
    opus=read(OUT/'04_opus_definition_validity.json')
    fable=read(OUT/'05_fable_definition_challenge.json')
    result=read(ROOT/'docs/paper/b105_merit.json')
    rows=[]
    names={'V1_r2_k0':'무작위 파일','V2_s1_k0':'군집 차단','V3_q3':'Rolling origin','V4a':'마지막 시간 블록'}
    for key,cell in result['cells'].items():
        if '/test/D10/F/TCN35_over_incumbent_outerfit' not in key:continue
        stat=cell['statistics'];lo,hi=stat['cluster']['ci95'];point=stat['point']
        assert point<0 and lo<hi<0
        rows.append('<tr><td>'+names[key.split('/')[0]]+'</td><td>'+f'{100*point:+.1f}% [{100*lo:+.1f}, {100*hi:+.1f}]'+'</td></tr>')
    originals=''.join('<details><summary>'+label+'</summary><pre>'+html.escape(json.dumps(review,ensure_ascii=False,indent=2))+'</pre></details>'for label,review in [('Opus의 정의 타당성 검토',opus),('Fable의 반박·교차 점검',fable)])
    body='''<h1>연구 정의와 TCN 결과에 대한 Claude 피드백</h1>
<p class="status">검토 의견과 작업자 확인을 정리한 문서다. 후속 저자 지시로 본문의 표적 정의만 짧게 줄여 부록으로 이동했다. 기존 등록·평가 규칙과 결과는 그대로다.</p>
<h2>핵심 의견</h2>
<p><strong>연구자가 운용 정의를 정하는 것은 가능하다. 그러나 그 정의가 현재 물리량의 참값이거나 잡음 제거라는 해석까지 보장하지는 않는다.</strong> 두 검토는 현재 보고값을 추정한다는 목적에 비해 L5가 원고의 중심에 너무 자주 등장한다고 지적했다.</p>
<p>L5는 현재 프레임을 포함하지 않는 전방 창의 보고값 중앙값이다. 현재까지의 입력만 쓴다는 규칙과, 무엇을 정답으로 삼는지는 서로 다른 문제다. 입력 누출의 증거가 있다는 뜻이 아니라, F와 L5가 서로 다른 추정 대상이라는 뜻이다. 현재 온도를 대리한다는 해석에는 별도 근거가 필요하다.</p>
<p>권고는 <strong>현재 프레임 보고값 F를 본문 중심에 두고, L5는 전방 창 보고 수준을 평가하는 보조 결과로 정리</strong>하는 것이다. 후속 저자 지시에 따라 본문 정의를 짧게 줄이고 상세 정의를 부록 A로 옮겼다. 결과 전체를 재편하는 권고는 아직 적용하지 않았으며 기존 그림과 비교는 유지했다.</p>
<h2>너무 많은 정의는 어떻게 줄일 것인가</h2>
<table><thead><tr><th>묶음</th><th>남길 내용과 주의할 해석</th></tr></thead><tbody>
<tr><td>추정 대상</td><td>현재 보고값 F를 분명히 정의한다. L5는 미래 창의 별도 표적으로 설명한다.</td></tr>
<tr><td>자료 제외 규칙</td><td>온도 상한·정확 반복 제거·구간 절단은 분석 규칙으로 제시한다. 값의 크기나 동일성만으로 물리적 무효를 입증했다고 쓰지 않는다.</td></tr>
<tr><td>가용성 시나리오</td><td>고정 지연·notch 기반 일정·성긴 보고는 명시한 가정 아래의 평가로 기술한다. 실제 보고 도착 시각의 측정과 구분한다.</td></tr>
</tbody></table>
<p>커널·이력 길이·특징 창·묶음 간격 같은 설정이 연구 고유라는 이유만으로 잘못된 것은 아니다. 선택 근거, 적용 범위와 필요한 민감도를 밝혀야 한다. 등록과 구현의 일치는 재현성을 보여 주지만 물리 해석을 검증한 것과는 다르다.</p>
<h2>TCN에 대해서는 무엇을 말할 수 있는가</h2>
<p>아래는 원본 B105 결과 파일에서 읽은 D10/F TEST 비교다. 음수는 현재 TCN 후보의 MSE가 기존 GP·boosting보다 컸음을 뜻한다. 설계마다 고정 unit과 초기화가 하나인 탐색 결과이며, TCN 계열 전체의 결론은 아니다.</p>
<table><thead><tr><th>검증 설계</th><th>MSE 감소율 [조건부 군집 구간]</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table>
<p>집계 VAL 문턱은 조건별 성능 보장이 아니다. 여러 규칙과 두 표적의 평균 손실 비율이 통과하면, 기본 현재값 조건에서의 악화와 함께 통과할 수 있다. 이 통과는 계산 배분 결정을 뜻하며 통계적 비열등 판정이 아니다. VAL의 공정 대조와 TEST의 incumbent는 적합 범위도 다르므로 같은 비교로 묶어 쓰지 않는다.</p>
<p>원본 incumbent 대비 TEST 비교는 모든 셀의 점추정이 음수다. 다만 모든 구간이 영을 제외한다는 뜻은 아니다. 이 기록을 남기고, 현재 TCN을 주 추정기로 채택할 근거가 부족하다는 범위에서 해석한다. 정확한 원인은 이번 비교에서 분리되지 않았다.</p>
<h2>기존 근거와 빠뜨리지 말아야 할 점</h2>
<p>B73에는 이미 여러 표적을 비교한 결과가 있다. 이를 없는 실험처럼 다시 요구해서는 안 된다. 다만 무작위 설계·tree 비교이며 창 길이뿐 아니라 평균/중앙값, 결측 기준과 평가 행이 함께 달라져 순수한 창 길이 비교나 독립적인 물리 검증은 아니다.</p>
<p>L5는 새 TCN 실험보다 먼저 정의됐다. 따라서 이번 TCN을 구제하려고 새로 만들었다는 비판은 맞지 않는다. 반대로 TCN 결과를 본 뒤 유리한 표적이나 판정 규칙을 골라 주장을 바꾸는 것은 피해야 한다. 이미 있는 현재 F 결과와 기준선 비교를 먼저 정리하는 것이 두 검토의 공통 권고다.</p>
<h2>작업자 확인 및 아직 권고인 부분</h2>
<p>Opus의 일부 표현은 그대로 채택하지 않았다. VAL과 TEST의 비교 대상이 다르고, 조건부 구간은 재적합·모형 선택·초기화 변동을 모두 포함하지 않는다. 어느 표적이 gate 통과에 얼마를 기여했는지는 제공된 집계만으로 원인을 단정하지 않는다.</p>
<p>새 민감도 분석이 반드시 GPU 없이 된다는 조언도 확정할 수 없다. 단순 집계 재확인과, 제외 기준을 바꾸어 이력·정규화·학습 표적부터 다시 만드는 비교는 다르다. 후자는 재적합이 필요할 수 있다. 먼저 기존 결과의 범위를 확인해야 하며 새로운 분석은 별도 등록 대상이다. 이번 작업에서는 추가 학습·평가·검산·프로토콜 변경을 실행하지 않았다.</p>
<p><a href="../index.html">현재 논문 HTML</a> · <a href="04_opus_definition_validity.json">Opus 원문</a> · <a href="05_fable_definition_challenge.json">Fable 원문</a> · <a href="verification.json">출처·사용량 기록</a></p>
<h2>원문 피드백 펼치기</h2>'''+originals
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>연구 정의 검토</title><style>body{max-width:920px;margin:45px auto;padding:0 22px;font:17px/1.9 sans-serif;color:#203638;background:#fbfcfa}h1{font-size:28px}h2{margin-top:45px;font-size:22px}p{margin:20px 0}.status{padding:15px;background:#eaf0ec;font-size:14px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border-bottom:1px solid #ccd6d1;text-align:left;padding:12px}th{background:#edf2ef}pre{white-space:pre-wrap;word-break:break-word;font:14px/1.8 sans-serif}details{border:1px solid #d5dfda;padding:15px;margin:16px 0}a{color:#126564}@media(max-width:600px){body{font-size:15px;margin-top:20px}h1{font-size:23px}table{font-size:12px}}</style>'+body+'</html>'
    terminology=OUT/'07_fable_L5_terminology.json'
    if terminology.exists():
        t=read(terminology)
        note='<section><h2>L5는 다른 연구의 표준 명칭인가</h2><p>'+html.escape(t['answer_ko'])+'</p>'
        note+='<p>권장 설명: <strong>후속 시간창 중앙값 표적(future-window median target)</strong>. 표준 고유 명칭이 아니라 연산을 풀어 쓴 표현이다.</p>'
        note+='<p>Ross 등은 미래 구간의 관측 중앙값으로 이진 판정 레이블을 구성했다. 직접적인 중앙값 회귀 학습, 다섯 격자·최소 세 관측 규칙과는 다르다. 이 사례가 현재 CES 참값의 근거는 아니다. <a href="https://arxiv.org/html/2603.08242v1#S4.SS2">원문 4.2절</a></p>'
        note+='<p>시간적 집계라는 넓은 범주는 기존 문헌에 있으나, 이것을 L5의 정확한 표준 이름으로 대체하지 않는다. <a href="https://doi.org/10.1002/nav.21546">Rostami-Tabar 등 원문 초록</a></p>'
        note+='<p><a href="07_fable_L5_terminology.json">Fable 용어 검토</a> · <a href="terminology_evidence.json">검색 범위·원문 위치</a></p></section>'
        page=page.replace('<h2>원문 피드백 펼치기</h2>',note+'<h2>원문 피드백 펼치기</h2>')
    (OUT/'index.html').write_text(page,encoding='utf8')
    print('DEFINITION FEEDBACK REPORT WRITTEN')

if __name__=='__main__':main()
