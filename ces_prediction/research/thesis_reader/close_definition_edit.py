"""Free closeout for authorized appendix edit and completed terminology review."""
import hashlib
import json
import subprocess

import verify_with_claude as V

OUT=V.ROOT/'docs/paper/research_reader/definition_audit'

def main():
    ledger=V.ledger()
    stages=['04_opus_definition_validity','05_fable_definition_challenge',
            '06_fable_concise_L5_appendix','07_fable_L5_terminology']
    for stage in stages:
        raw=V.read(V.CACHE/(stage+'.response.json'))
        assert raw['stop_reason']=='end_turn'
        assert (OUT/(stage+'.json')).exists()
    reader=V.ROOT/'docs/paper/research_reader/index.html'
    document=V.read(V.ROOT/'docs/paper/introduction_review/manuscript_sections.json')
    assert len(document['appendices'])==1
    assert '학습·평가의 표적' in document['appendices'][0]['content_markdown']
    unchanged=[]
    for relative in ['docs/paper/ppcf/main_ppcf.tex','docs/paper/ppcf/supp_ppcf.tex',
                     'docs/paper/ppcf_revision/main_ppcf.pdf','docs/paper/b105_merit.json',
                     'ces_prediction/experiments/b73_targets/common.py',
                     'ces_prediction/experiments/PREREGISTRATION_B73.md',
                     'ces_prediction/experiments/PREREGISTRATION_B105.md']:
        original=subprocess.check_output(['git','show','HEAD:'+relative],cwd=V.ROOT)
        current=(V.ROOT/relative).read_bytes()
        # Git normalizes text line endings; binary files compared byte-for-byte.
        assert original.replace(b'\r\n',b'\n')==current.replace(b'\r\n',b'\n'),relative
        unchanged.append(relative)
    V.save(OUT/'editorial_closeout.json',{
      'scope':'Author requested concise main definition and appendix; then checked established terminology with public primary sources and Fable.',
      'completed_review_stages':stages,'unchanged_scientific_sources':unchanged,
      'reader_sha256':hashlib.sha256(reader.read_bytes()).hexdigest(),
      'ui_checks':V.read(V.ROOT/'docs/paper/research_reader/qa_status.json'),
      'cost_shared_cap_usd':ledger['cap_usd'],
      'cost_shared_estimate_usd':ledger['estimated_or_reserved_usd'],
      'new_scientific_fits':0,'new_evaluations':0,'gpu_runs':0,
      'terminology_scope':V.read(OUT/'terminology_evidence.json')['scope'],
      'not_claimed':'No universal absence of precedent, physical truth validation, or standard proper name established.'})
    p=OUT/'README.md';text=p.read_text(encoding='utf8').replace('研究','연구')
    if '## 후속 용어 확인과 부록 편집' not in text:
        text+='''\n## 후속 용어 확인과 부록 편집\n\n저자의 후속 지시로 본문의 표적 정의를 짧게 줄이고 세부를 접을 수 있는 부록 A로 이동했다. 기존 결과·그림과 등록 정의는 유지한다. 최초 검토 시점의 `verification.json`과 달리 이 편집은 실제 HTML에 반영됐다. 최신 상태는 `editorial_closeout.json`이다.\n\n`06_fable_concise_L5_appendix.json` → `integrate_longform.py` → 본문·부록, `build_reader.py` → HTML로 연결된다. 호스트는 같은 파일·연속 구간/사용 가능한 보고값 조건을 명료화하고, 학습과 평가의 표적이라는 표현을 바로잡았다. `check_l5_terminology.py`는 공개 문헌의 추출 텍스트 검색·표적 설명·웹 원문 위치를 `terminology_evidence.json`에 기록하고 공식 Fable 검토 `07_fable_L5_terminology.json/txt`를 받는다. 254개 캐시 레코드(고유 PDF 해시 253개)의 기호 검색이며 전체 재독은 아니다. 정확한 L5 선례·표준 이름은 확인하지 못했다. Ross 등의 미래 구간 중앙값 기반 이진 레이블은 유사 사례일 뿐 동일한 회귀 표적은 아니다.\n\n`write_definition_feedback.py`가 이 용어 검토도 읽기용 보고서에 넣는다. `close_definition_edit.py`는 추가 API 호출 없이 완료 응답·공통 비용 상한·원래 과학 소스 불변·HTML 검증을 기록한다. 비용은 figure_verification/api_usage.json의 같은 5달러 상한에 합산하며 청구서 확인액은 아니다.\n'''
    p.write_text(text,encoding='utf8')
    print('EDITORIAL CLOSEOUT PASSED',ledger['estimated_or_reserved_usd'])

if __name__=='__main__':main()
