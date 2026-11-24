"""Presentation names only: never rename frozen result keys or numerical files."""
import re

GP={'T0':'GP-Boost-CES','TX':'GP-Boost-Multi','TE':'GP-Boost-ECEI',
    'TB':'GP-Boost-BES','TN':'GP-Boost-CES+NBI','TXN':'GP-Boost-Multi+NBI'}
TCN={'TH':'TCN-CES','TX':'TCN-Multi'}

def replace_names(text,tcn=False):
    mapping={**GP,**(TCN if tcn else {})}
    # Korean particles may follow an identifier; longer ASCII identifiers must not match.
    return re.sub(r'(?<![A-Za-z0-9_])(?:TXN|T0|TX|TE|TB|TN|TH)(?![A-Za-z0-9_])',
                  lambda m:mapping.get(m[0],m[0]),text)

def apply_names(sections):
    def walk(node,tcn=False):
        tcn=tcn or '고정 계산 예산의 TCN 비교' in node.get('title','')
        text=node.get('content_markdown','')
        paragraphs=text.split('\n\n')
        for i,p in enumerate(paragraphs):
            if '감사된 결과 파일과 추론 시간 표' in p and 'TH' in p:
                p=p[:p.index('감사된 결과 파일과 추론 시간 표')]
                p+='이 두 신경망은 본문에서 각각 TCN-CES와 TCN-Multi로 부른다. GP-Boost-CES 및 GP-Boost-Multi와 구조와 학습 절차가 다른 비교 후보다. 기존 산출물의 기호 대응은 모델 명칭 표에 정리하였다.'
            paragraphs[i]=replace_names(p,tcn=tcn)
        node['content_markdown']='\n\n'.join(paragraphs)
        node['title']=replace_names(node.get('title',''),tcn=tcn)
        for child in node.get('subsections',[]):walk(child,tcn)
    for section in sections:walk(section)
    methods=next(s for s in sections if s['id']=='materials-methods')
    methods['content_markdown']+='''

### 모델 명칭과 기존 결과 파일의 대응

모델 이름은 구조와 입력을 함께 나타내도록 구분한다. GP-Boost는 CES 이력의 GP 평균에 boosting 잔차 회귀를 더한 구성이고, TCN은 시간 합성곱 신경망이다. Multi는 CES 이력에 현재까지의 다른 진단 특징을 더한 입력을 뜻한다. KSTAR에서는 BES·ECEI, MAST 적용에서는 Thomson 자료에 해당한다. CES 보고값을 언제까지 읽는지 정하는 가용성 규칙은 모델 이름과 별도로 표시한다.

| 본문 모델명 | 구성과 추가 입력 | 기존 결과 파일의 기호 |
|---|---|---|
| GP-Boost-CES | GP + CES 이력 특징의 잔차 회귀 | GP 계열 T0 |
| GP-Boost-Multi | GP + CES 이력·추가 진단 특징의 잔차 회귀 | GP 계열 TX |
| GP-Boost-ECEI | GP-Boost-CES에 ECEI 특징 추가 | GP 계열 TE |
| GP-Boost-BES | GP-Boost-CES에 BES 특징 추가 | GP 계열 TB |
| GP-Boost-CES+NBI | GP-Boost-CES에 빔 특징 추가 | GP 계열 TN |
| GP-Boost-Multi+NBI | GP-Boost-Multi에 빔 특징 추가 | GP 계열 TXN |
| TCN-CES | CES 이력을 읽는 TCN | TCN 계열 TH |
| TCN-Multi | CES 이력과 빠른 진단을 읽는 TCN | TCN 계열 TX |

기존 GP 결과와 TCN 결과에서 TX가 서로 다른 모델을 가리켰으므로, 본문에서는 TX를 단독 모델명으로 쓰지 않는다. 원본 표·그림과 결과 파일을 대조할 때만 해당 계열의 기존 기호를 사용한다. 명칭 정리는 구조·입력·성능 결과를 바꾸지 않는다.
'''

def exhibit_legend(item):
    tcn=item['id'] in ('table9','table10','table11')
    if tcn:
        return '원본 표의 기호 대응: TH → TCN-CES, TX → TCN-Multi. 이 표의 TX는 GP-Boost-Multi가 아니다.'
    if item['kind']=='figure' or item['id'] in ('table2','table3','table4','table5','table6','table7','table8'):
        return '원본 표·그림의 기호 대응: T0 → GP-Boost-CES, TX → GP-Boost-Multi. TN/TXN은 각각 빔 특징을 추가한 GP-Boost-CES+NBI/GP-Boost-Multi+NBI다.'
    return ''
