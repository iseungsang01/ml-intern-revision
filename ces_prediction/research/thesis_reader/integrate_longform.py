"""Integrate long-form prose only after checking exact Opus/host corrections."""
import json
import re
from pathlib import Path

from build_reader import ROOT, load_sources

OUT=ROOT/'docs/paper/research_reader/longform'
REV=ROOT/'docs/paper/introduction_review'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def nodes(section):
    yield section
    for child in section.get('subsections',[]):
        yield from nodes(child)


def main():
    draft=read(OUT/'draft_longform.json')
    sections=draft['sections']
    by_id={section['id']:section for section in sections}
    assert set(by_id)=={'introduction','background','materials-methods','results','discussion','conclusion'}
    applied=[]
    review_corrections=read(OUT/'03_opus_source_and_prose_check.json')['corrections']
    decisions_path=OUT/'review_decisions.json'
    decisions=read(decisions_path) if decisions_path.exists() else {}
    corrections=[]
    for index,correction in enumerate(review_corrections):
        if index in decisions.get('excluded_indices',[]):
            continue
        correction=dict(correction)
        if str(index) in decisions.get('replacement_new_by_index',{}):
            correction['new']=decisions['replacement_new_by_index'][str(index)]
            correction['host_refinement']='review_decisions.json'
        corrections.append(correction)
    host=OUT/'host_corrections.json'
    if host.exists():
        corrections+=read(host).get('corrections',[])
    readability=OUT/'04_fable_readability_recovered.json'
    if readability.exists():
        corrections+=read(readability)['corrections']
    readability_host=OUT/'readability_host_corrections.json'
    if readability_host.exists():
        corrections+=read(readability_host)['corrections']
    figure_verification=ROOT/'docs/paper/research_reader/figure_verification/applied_corrections.json'
    if figure_verification.exists():
        corrections+=read(figure_verification)['corrections']
    skipped=[]
    for correction in corrections:
        candidates=[n for n in nodes(by_id[correction['section_id']])
                    if correction['old_exact'] in n.get('content_markdown','')]
        if not candidates and correction.get('if_still_present'):
            skipped.append({'old_exact':correction['old_exact'],'reason':'Already replaced by preceding source-review correction; host must inspect final prose'})
            continue
        assert len(candidates)==1,correction
        node=candidates[0]
        assert node['content_markdown'].count(correction['old_exact'])==1
        node['content_markdown']=node['content_markdown'].replace(correction['old_exact'],correction['new'])
        applied.append(correction)
    current=read(REV/'manuscript_sections.json')
    old={s['id']:s for s in current['sections']}
    for section in sections:
        for key in ['pdf_path','pdf_page','pdf_label','pdf_preview_section']:
            if key in old[section['id']]:
                section[key]=old[section['id']][key]
        section['status']='장문 논문 본문 · 출처 및 기존 결과 대조 후 저자 검토본'
        if section['id']=='results':
            section['pdf_preview_after_text']=True
            section['pdf_preview_collapsed']=True
        if section['id']=='materials-methods':
            section.update(pdf_path='../ppcf_revision/main_ppcf.pdf',pdf_page=4,
                           pdf_label='자료·방법의 원본 표와 식 열기')
    exhibits_path=OUT/'exhibits.json'
    if exhibits_path.exists():
        for exhibit in read(exhibits_path)['items']:
            placement=exhibit['placement']
            prefix=placement['subsection_title_prefix']
            matches=[n for n in nodes(by_id[placement['section_id']]) if n.get('title','').startswith(prefix)]
            assert len(matches)==1,(exhibit['id'],prefix)
            matches[0].setdefault('pdf_exhibits',[]).append(exhibit['id'])
        runtime=next(n for n in nodes(by_id['results']) if n.get('title','').startswith('3.'))
        runtime['content_markdown']+='\n\n본선 GP·boosting 계산 시간의 측정 조건과 분위수는 [보충 표 S13](../ppcf_revision/supp_ppcf.pdf#page=11)에 제시하였다. TCN의 계산 비용은 별도 신경망 비교 절의 표 11에서 확인할 수 있다.'
    counts={sid:sum(len(n.get('content_markdown','')) for n in nodes(s)) for sid,s in by_id.items()}
    assert counts['introduction']>=5000,counts
    assert counts['background']>=5000,counts
    assert counts['materials-methods']>=5000,counts
    assert counts['results']>=2000,counts
    assert counts['discussion']>=3000,counts
    assert counts['conclusion']>=1800,counts
    sources,aliases=load_sources()
    citations={k for s in sections for n in nodes(s)
               for k in re.findall(r'\[@([^\]]+)\]',n.get('content_markdown',''))}
    assert citations<=set(aliases),citations-set(aliases)
    by_source={s['key']:s for s in sources}
    references=[]
    for key in sorted({aliases[k] for k in citations},key=lambda k:by_source[k]['title'].lower()):
        source_record=by_source[key]
        citation=source_record['review']['citation']
        authors=source_record['authors']
        title=source_record['title'];year=source_record['year']
        venue=citation.get('venue','')
        volume=str(citation.get('volume',''))
        pages=str(citation.get('pages',''))
        suffix=', '.join(x for x in [str(venue),volume,pages] if x)
        link=' [DOI](https://doi.org/'+source_record['doi']+').' if source_record['doi'] else ''
        references.append('[@'+key+'] '+authors+' ('+year+'). *'+title+'*. '+suffix+'.'+link)
    old['references']['content_markdown']='\n\n'.join(references)
    old['references']['status']='본문 인용 문헌 · 출처 감사의 서지정보에서 자동 생성'
    # Method definitions must not displace scientific theory exposition.
    theory='\n'.join(n.get('content_markdown','') for n in nodes(by_id['background']))
    assert not re.search(r'filtering에 해당|smoothing에 해당|F와 L5|정보 집합 I_t',theory)
    manuscript={**current,'sections':sections+[old['references']],
       'active_longform':'docs/paper/research_reader/longform/draft_longform.json',
       'status_note':'기존 졸업논문 목차를 보존한 장문 한국어 논문 초안이다. Fable 본문 작성 후 Opus 검토와 기존 영문 결과·문헌 근거 대조를 거쳐 반영했다. 결과의 수치·표·그림은 기존 검증 PDF를 함께 제시한다.'}
    save(REV/'manuscript_sections.json',manuscript)
    save(OUT/'integration_status.json',{'section_characters':counts,
       'total_prose_characters_excluding_references':sum(counts.values()),
       'citation_keys':sorted(citations),'applied_corrections':applied,
       'generated_cited_reference_entries':len(references),
       'conditional_corrections_already_superseded':skipped,
       'review_decisions':decisions,
       'new_fits':0,'gpu_runs':0,'english_pdf_unchanged':True,
       'entire_original_corpus_not_reread_in_this_step':True,
       'independent_acceptance_grade':False})
    print('LONGFORM INTEGRATED',sum(counts.values()),'characters',counts)


if __name__=='__main__':
    main()
