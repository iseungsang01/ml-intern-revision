"""Integrate the completed source review while retaining the author's hierarchy."""
import json

from build_reader import REVIEWS, load_sources


def main():
    path=REVIEWS/'manuscript_sections.json'
    document=json.loads(path.read_text(encoding='utf8'))
    review=json.loads((REVIEWS/'final_manuscript_sections_review.json').read_text(encoding='utf8'))
    sections={section['id']:section for section in document['sections']}
    intro=sections['introduction']
    intro['content_markdown']=review['introduction_ko']['general']
    for subsection,key in zip(intro['subsections'],['a_objectives','b_rationale','c_effects','d_distinction']):
        subsection['content_markdown']=review['introduction_ko'][key]
    objectives=intro['subsections'][0]
    objectives['content_markdown']=objectives['content_markdown'].replace(
        '비교는 같은 평가 행, 같은 정보 집합, 같은 계산 예산에서 한다.',
        '후속 구조 비교는 같은 평가 행, 같은 정보 집합, 같은 계산 예산에서 수행하는 것을 원칙으로 한다. 기존에 완료한 모든 비교가 이 조건을 충족했다는 뜻은 아니다.')
    objectives['content_markdown']=objectives['content_markdown'].replace(
        'L5는 이후 다섯 보고값 가운데 세 개 이상의 중앙값으로 오프라인에서 정의한다.',
        'L5는 현재 프레임 다음의 다섯 격자 행(t+1부터 t+5까지)에서 실제 측정값이 세 개 이상 존재할 때, 그 측정값의 중앙값으로 오프라인에서 정의한다. 다음에 관측되는 보고값 다섯 개를 모으는 정의와는 다르다.')
    background=sections['background']
    for subsection,key in zip(background['subsections'],['pedestal','ces_spectrum','temporal_information','learning_models']):
        subsection['content_markdown']=review['background_ko'][key]
    background['subsections'][1]['content_markdown']=background['subsections'][1]['content_markdown'].replace('미세구조 같은 선형 효과','미세구조 같은 선 구조의 영향')
    background['subsections'][2]['content_markdown']+='\n\n본 자료에는 실제 보고 도착 시각이 기록되어 있지 않다. 따라서 이 정보 집합은 평가에서 정한 가용성 규칙을 나타내며, 실제 도착 시각을 관측해 재현한 운용 결과를 뜻하지 않는다. 현재 CES 스펙트럼도 입력으로 사용하지 않는다.'
    for section in (intro,background):
        section['status']='공개 문헌 원문 검토에 근거한 Opus 통합 초안 · 저자 검토 전'
    methods=sections['materials-methods']
    dataset=methods['subsections'][0]
    dataset['subsections'][0]['content_markdown']='KSTAR 2022 캠페인에서 제공받은 한 CES 측정선의 탄소 불순물 이온온도 보고값과, 10 ms 격자의 BES·ECEI 전압 표본을 사용한다. CES는 관례적 분석 표지이며 BES·ECEI 표본은 보정된 전자온도·밀도가 아니다. 같은 시간 격자의 NBI 출력·전압 자료는 빔이 설명하는 정보를 분리하는 비교에 사용한다. 데이터의 전체 보유 범위, 학습에 사용 가능한 범위, 각 표적에서 채점 가능한 행을 구분한다. 실제 수량과 처리 결과는 검증된 원고의 데이터 표를 따른다.'
    for entry in dataset['subsections'][1]['subsections']:
        if entry['title'].startswith('c.'):
            entry['content_markdown']=entry['content_markdown'].replace('현재 연구 자료는 하나의 2022 캠페인이므로,','현재 연구 자료는 이미 재사용한 하나의 2022 캠페인이며, 한 번도 사용하지 않은 독립 자료는 없다. 따라서')
    neural=methods['subsections'][1]['subsections'][0]
    neural['subsections'][0]['content_markdown']='단일 구조를 먼저 채택하기보다 CNN·인과적 TCN과 가벼운 시계열 모델을 이력-only 모델 및 비신경망 기준선과 비교하는 방식으로 구성한다. GP·boosting·LSTM과 예산이 제한된 TCN 비교는 각 실험의 완료 범위를 밝혀 제시한다. 과거 CNN 비교는 같은 정보와 계산 예산을 맞춘 구조 우열의 증거로 간주하지 않는다. 추가 후보는 같은 입력·표적·평가 행·검증 설계와 계산 예산 조건에서 비교한 뒤 선정한다.'
    methods['subsections'][2]['content_markdown']='F는 현재 프레임의 CES 보고값이다. L5는 다음 다섯 격자 행(t+1부터 t+5까지)에 실제 측정된 CES 값이 세 개 이상 있을 때 그 값들의 중앙값으로 정의하는 오프라인 수준이다. 다음에 관측되는 다섯 보고값의 중앙값이 아니다. 둘 다 평가 표적이며 현재 입력에 포함하지 않는다. 실제 보고 도착 시각은 기록되지 않았으므로, 이력의 가용성은 각 평가의 명시적 규칙으로 설정한다. Hold·인과 GP는 인과적 기준선이고, 다음 보고값을 사용하는 양측 보간은 오프라인 기준선이다. 네 검증 설계를 동등하게 제시한다. 오차 차이는 같은 행의 직접 비교와 구간으로 보고하며, 별도 구간의 겹침만으로 차이를 판정하지 않는다. 현재 평가로 10 ms보다 빠른 실제 이온온도 변화나 sub-ms 시간해상도의 물리적 복원을 검증한 것은 아니다.'
    results=sections['results']
    results['pdf_preview_section']='Results'
    results['status']='기존 생성 표·원고 PDF의 결과 페이지'
    results['content_markdown']='완료된 결과는 기존 LaTeX 원고의 생성 표와 PDF 페이지를 아래에 그대로 표시한다. GP·boosting·LSTM 및 예산이 제한된 TCN 비교, 빠른 진단의 추가 정보, 이력 가용성 조건, 서로 다른 검증 설계의 결과를 나눠 읽는다. 새 CNN 또는 재설계 NN의 성능은 실행·검증 전까지 완료 결과에 포함하지 않는다. 기존 CNN 비교를 동일 정보·동일 계산 예산에서 구조 우열을 확정한 결과로 해석하지 않는다.'
    sections['discussion']['status']='기존 결과 해석 초안 · 저자 검토 전'
    sections['conclusion']['status']='완료 범위와 후속 계획 구분 · 저자 검토 전'
    sections['references']['status']='전체 공개 문헌 원문 검토 기록 연결'
    sources,_=load_sources()
    counts={mode:sum(source['mode']==mode for source in sources) for mode in ['full_pdf','full_text','summary','held','duplicate']}
    sections['references']['content_markdown']='254개 보관 PDF 파일은 중복 보관본 두 개와 외부 검토를 보류한 기관 가이드 한 개를 포함한다. 나머지 고유 공개 소스 파일 251건의 원문 검토 기록을 연결한다. 동일 연구의 preprint·출판 버전이 포함될 수 있으므로 이를 서로 독립된 연구 251개로 해석하지 않는다. 전체 PDF로 검토한 파일 '+str(counts['full_pdf'])+'건과 페이지 그림 없이 전체 추출문으로 검토한 파일 '+str(counts['full_text'])+'건을 구분한다. 논문별 근거 위치와 채택 가능한 주장·주의할 해석은 인용 버튼 또는 문헌·인용 근거에서 확인할 수 있다. 원문의 그림·도표를 추가로 확인해야 하는 항목은 검토 기록에 남긴다.'
    document['status_note']='전체 공개 문헌의 원문 검토 기록에 근거한 한국어 작업본이다. 서론·이론적 배경은 Opus 통합 초안을 반영했으며 저자 검토 전이다. 기존 실험 결과·실행하지 않은 후속 계획·확인되지 않은 메타데이터를 구분한다. 문헌 검토는 원고의 독립 채점이나 실험 결과의 독립 검증을 대신하지 않는다.'
    path.write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('KOREAN MANUSCRIPT INTEGRATED; PDF build intentionally deferred')


if __name__=='__main__':
    main()
