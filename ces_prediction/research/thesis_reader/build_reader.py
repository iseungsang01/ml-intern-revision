"""Build the local, self-contained thesis and source-audit reader."""
import argparse
import base64
import html
import json
import re
import urllib.parse
from pathlib import Path

import fitz
import markdown
try:
    import latex2mathml.converter
except ImportError:
    latex2mathml = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'docs/paper/research_reader'
REVIEWS = ROOT / 'docs/paper/introduction_review'
PAPER = ROOT / 'docs/paper/ppcf_revision'
CITATION_LABELS = {}
PDF_PREVIEWS = []


def read_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf8'))


def safe_text(value):
    if value is None:
        return ''
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def safe_link(value):
    value=html.unescape(value).strip()
    scheme=urllib.parse.urlsplit(value).scheme.lower()
    if scheme not in ('','https','http') or value.startswith('//') or any(ord(c)<32 for c in value):
        return '#'
    return value


def tex_to_markdown(text):
    text = re.sub(r'(?m)^\s*%.*$', '', text)
    text = text.replace(r'\Ti{}', 'Tᵢ').replace(r'\Ti', 'Tᵢ').replace(r'\Te{}', 'Tₑ')
    text = re.sub(r'\\(?:sub)*section\*?\{([^{}]+)\}', r'\n### \1\n', text)
    text = re.sub(r'\\label\{[^}]+\}', '', text)
    text = re.sub(r'\\(?:emph|textbf|textit)\{([^{}]+)\}', r'**\1**', text)
    text = re.sub(r'\\(?:S|ref|eqref|Supp|SuppTab|SuppFig)\{([^}]+)\}', r'(원문 참조: \1)', text)
    text = re.sub(r'\\begin\{(figure|table)\}.*?\\end\{\1\}', '\n[표·그림은 연결된 PDF에서 확인]\n', text, flags=re.S)
    text = text.replace(r'\,', ' ').replace(r'\%', '%').replace('~', ' ')
    return text.strip()


def quantitative_tokens(text):
    text=re.sub(r'\[@[^\]]+\]','',text)
    text=re.sub(r'\$\$.+?\$\$','',text,flags=re.S)
    text=re.sub(r'\$(?!\$).+?\$','',text)
    text=re.sub(r'\\\(.+?\\\)','',text)
    return re.findall(r'(?<![A-Za-z0-9_])\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?',text)


def render_content(text):
    exhibit_links={'tab:dataset':('table1','표 1'),'fig:problem':('figure1','그림 1'),
        'tab:inputs':('table2','표 2'),'fig:rules':('figure2','그림 2'),
        'tab:ladder':('table3','표 3'),'tab:vsref':('table4','표 4'),
        'fig:nowcast':('figure3','그림 3'),'tab:rules':('table5','표 5'),
        'fig:ladder':('figure4','그림 4'),'tab:increment':('table6','표 6'),
        'fig:designs':('figure5','그림 5'),'fig:sparse':('figure6','그림 6'),
        'tab:robust':('table7','표 7'),'tab:blocks':('table8','표 8'),'fig:limits':('figure7','그림 7')}
    for key,(target,label) in exhibit_links.items():
        text=text.replace(key,'['+label+'](#exhibit-'+target+')')
    citations=[]
    maths=[]
    def math_fragment(match,display='inline'):
        expression=match[1].replace(r'\Ti{}','T_i').replace(r'\Te{}','T_e')
        try:
            markup=latex2mathml.converter.convert(expression,display=display) if latex2mathml else '<code>'+html.escape(expression)+'</code>'
        except Exception:
            markup='<code>'+html.escape(expression)+'</code>'
        token='MATHTOKEN'+str(len(maths))+'END'; maths.append(markup)
        return token
    text=re.sub(r'!\[([^\]]*)\]\([^)]+\)',r'[그림: \1 — 연결된 PDF 참조]',text)
    text=re.sub(r'\$\$(.+?)\$\$',lambda m:math_fragment(m,'block'),text,flags=re.S)
    text=re.sub(r'\\\[(.+?)\\\]',lambda m:math_fragment(m,'block'),text,flags=re.S)
    text=re.sub(r'\\\((.+?)\\\)',math_fragment,text,flags=re.S)
    text=re.sub(r'(?<!\\)\$(?!\$)(.+?)(?<!\\)\$',math_fragment,text)
    def cited(match):
        keys=[x.strip() for x in re.split(r'[,;]\s*@?',match[1]) if x.strip()]
        token='CITATIONTOKEN'+str(len(citations))+'END'
        buttons=[]
        for key in keys:
            label,title=CITATION_LABELS.get(key,('[?]',key))
            buttons.append('<button type="button" class="citation" data-source="'+
                html.escape(key,quote=True)+'" aria-label="'+html.escape(label+' '+title+' 인용 근거 보기',quote=True)+
                '">'+html.escape(label)+'</button>')
        citations.append(' '.join(buttons))
        return token
    text=re.sub(r'\\cite\w*\{([^}]+)\}',cited,text)
    text=re.sub(r'\[@([^\]]+)\]',cited,text)
    # All manuscript/source text is data. Literal tags from an upstream model are
    # escaped; only the renderer's citation buttons and Markdown are admitted.
    rendered=markdown.markdown(html.escape(text), extensions=['tables','fenced_code','sane_lists'])
    rendered=re.sub(r'href="([^"]*)"',lambda m:'href="'+html.escape(safe_link(m[1]),quote=True)+'"',rendered)
    for i, markup in enumerate(citations):
        rendered=rendered.replace('CITATIONTOKEN'+str(i)+'END',markup)
    for i, markup in enumerate(maths):
        rendered=rendered.replace('MATHTOKEN'+str(i)+'END',markup)
    return rendered


def unwrap_review(record):
    for name in ('audit','review','result'):
        if isinstance(record.get(name),dict):
            return {**record,**record[name]}
    return record


def load_sources():
    catalogue=read_json(REVIEWS/'catalogue.json',{'papers':[]})['papers']
    reviews={}
    for pattern in ('paper_*.json','background_*.json'):
        for path in sorted(REVIEWS.glob(pattern)):
            record=unwrap_review(read_json(path,{}))
            for claim in record.get('claim_checks',[]):
                if isinstance(claim,dict):
                    if claim.get('allowed_intro_or_theory_claim') and not claim.get('allowed_intro_sentence'):
                        claim['allowed_intro_sentence']=claim['allowed_intro_or_theory_claim']
                    if claim.get('original_pdf_locator') and not claim.get('locator'):
                        claim['locator']=claim['original_pdf_locator']
            key=record.get('key') or path.stem.removeprefix('paper_').removeprefix('background_')
            reviews[key]=record
    sources=[]
    for row in catalogue:
        key=row['key']; review=reviews.get(key)
        scope=(review or {}).get('scope_read',{})
        source_mode=safe_text((review or {}).get('supplied_source_mode') or
                             (review or {}).get('source_mode') or (review or {}).get('source_scope') or '')
        if not row.get('external_review_allowed',True):
            mode='held'; mode_label='외부 검토 보류'
        elif row.get('duplicate_of_prior'):
            mode='duplicate'; mode_label='중복 보관본'
        elif review:
            if scope.get('full_pdf_supplied') is True or source_mode.lower().startswith(('full_pdf','full_original_pdf','scan_pdf','full pdf')):
                mode='full_pdf'; mode_label='전체 PDF'
            else:
                mode='full_text'; mode_label='전체 추출문'
        else:
            mode='summary'; mode_label='목록·요약 선별'
        citation=(review or {}).get('citation',{})
        if not isinstance(citation,dict):
            citation={'title':safe_text(citation)}
        title=citation.get('title') or row.get('metadata_title') or row.get('citation') or key
        choices=[safe_text(citation.get('doi_or_arxiv')),safe_text(row.get('identifier'))]
        doi=next((m for v in choices if (m:=re.search(r'10\.\d{4,9}/[^\s\)\]}>;,]+',v))),None)
        doi=doi[0].rstrip('.') if doi else ''
        authors=citation.get('authors','')
        if isinstance(authors,list):
            authors=' · '.join(safe_text(author) for author in authors)
        source={'key':key,'title':safe_text(title),'authors':safe_text(authors),
            'year':safe_text(citation.get('year','')),'pages':row.get('pages'),
            'mode':mode,'mode_label':mode_label,'doi':doi,
            'path':'../../../'+row.get('path',''),'relevance':safe_text((review or {}).get('relevance','')),
            'review':review,'review_file':('../introduction_review/'+('paper_' if (REVIEWS/('paper_'+key+'.json')).exists() else 'background_')+key+'.json') if review else ''}
        sources.append(source)
    aliases={s['key']:s['key'] for s in sources}
    # Exact DOI matching provides aliases for existing BibTeX keys only where the
    # source identity is unambiguous; a reference-list DOI is never used here.
    own_doi={}
    for s in sources:
        if s['doi']:
            own_doi.setdefault(s['doi'].lower(),[]).append(s['key'])
    bib=(PAPER/'refs.bib').read_text(encoding='utf8') if (PAPER/'refs.bib').exists() else ''
    for entry in re.split(r'(?=@\w+\s*\{)',bib):
        key=re.match(r'@\w+\s*\{\s*([^,\s]+)',entry)
        doi=re.search(r'\bdoi\s*=\s*\{([^}]+)\}',entry,re.I)
        if key and doi and len(own_doi.get(doi[1].lower(),[]))==1:
            aliases[key[1]]=own_doi[doi[1].lower()][0]
    for name in ('02_fable_source_corrected_introduction.json','final_manuscript.json'):
        obj=read_json(REVIEWS/name,{})
        for mapping in obj.get('citation_map',[]):
            if mapping.get('corpus_key') in aliases:
                aliases[mapping['bib_key']]=mapping['corpus_key']
    return sources,aliases


def render_pdf_exhibits(section):
    identifiers=section.get('pdf_exhibits',[])
    if not identifiers:
        return ''
    manifest=read_json(OUT/'longform/exhibits.json')
    path=PAPER/'main_ppcf.pdf'
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['pdf_sha256']:
        raise ValueError('PDF changed: recheck exhibit captions and crop boundaries')
    by_id={item['id']:item for item in manifest['items']}
    result=[]
    with fitz.open(path) as pdf:
        for identifier in identifiers:
            item=by_id[identifier];page=pdf[item['page']-1];rect=fitz.Rect(item['clip'])
            if not page.rect.contains(rect) or item['caption_prefix'] not in page.get_text(clip=rect):
                raise ValueError('Missing caption or invalid exhibit boundary: '+identifier)
            pixels=page.get_pixmap(clip=rect,dpi=160,alpha=False)
            encoded=base64.b64encode(pixels.tobytes('png')).decode('ascii')
            label=('표 ' if item['kind']=='table' else '그림 ')+str(item['number'])+'. '+item['title_ko']
            result.append('<figure class="source-exhibit" id="exhibit-'+html.escape(identifier)+'"><figcaption><strong>'+html.escape(label)+'</strong><p>'+html.escape(item['note_ko'])+'</p></figcaption><a href="../ppcf_revision/main_ppcf.pdf#page='+str(item['page'])+'" target="_blank" rel="noopener" aria-label="'+html.escape(label,quote=True)+' 원본 확대"><img loading="lazy" width="'+str(pixels.width)+'" height="'+str(pixels.height)+'" src="data:image/png;base64,'+encoded+'" alt="'+html.escape(label,quote=True)+'"></a><p class="exhibit-source">기존 생성 표·그림 그대로 · PDF '+str(item['page'])+'쪽 · 그림을 눌러 원본 확대</p></figure>')
    return ''.join(result)


def render_pdf_preview(section):
    """Embed only this project's manuscript, locating section pages on every build."""
    wanted=section.get('pdf_preview_section')
    if not wanted:
        return ''
    path=(OUT/section.get('pdf_path','')).resolve()
    if path != (PAPER/'main_ppcf.pdf').resolve():
        raise ValueError('PDF preview is restricted to the existing project manuscript')
    with fitz.open(path) as pdf:
        toc=pdf.get_toc()
        position=next((i for i,row in enumerate(toc) if row[1].strip().casefold()==wanted.casefold()),None)
        if position is None:
            raise ValueError('Requested manuscript section missing from PDF bookmarks: '+wanted)
        level,heading,start=toc[position]
        stop=next((row[2] for row in toc[position+1:] if row[0]<=level),len(pdf)+1)
        if stop<=start:
            raise ValueError('PDF section boundaries share a page; manual review is required')
        figures=[]
        for number in range(start,stop):
            page=pdf[number-1]
            pixels=page.get_pixmap(dpi=130,alpha=False)
            encoded=base64.b64encode(pixels.tobytes('png')).decode('ascii')
            label='기존 PPCF 원고 · '+heading+' · PDF '+str(number)+'쪽'
            figures.append('<figure class="pdf-page"><img width="'+str(pixels.width)+'" height="'+str(pixels.height)+
                '" src="data:image/png;base64,'+encoded+'" alt="'+html.escape(label,quote=True)+
                '" loading="lazy"><figcaption>'+html.escape(label)+'</figcaption></figure>')
        PDF_PREVIEWS.append({'section':heading,'pdf':'docs/paper/ppcf_revision/main_ppcf.pdf','pages':list(range(start,stop)),'dpi':130})
    opened='' if section.get('pdf_preview_collapsed') else ' open'
    return '<details class="pdf-preview"'+opened+'><summary>영문 결과 원고 페이지 함께 보기 ('+str(len(figures))+'쪽 발췌)</summary>'+\
        '<p class="preview-note">기존 원고의 원본 페이지를 그대로 표시합니다. 시작 페이지에는 앞 절의 끝부분이 포함될 수 있습니다. 수치·표·그림은 재작성하지 않았습니다.</p>'+''.join(figures)+'</details>'


def fallback_sections():
    draft=read_json(OUT/'outline_mapping.json',{})
    if draft.get('sections'):
        sections=draft['sections']
    else:
        titles=[('introduction','Ⅰ. 서론'),('background','Ⅱ. 이론적 배경'),
                ('materials-methods','Ⅲ. 재료 및 방법'),('results','Ⅳ. 결과'),
                ('discussion','Ⅴ. 고찰'),('conclusion','Ⅵ. 결론'),('references','Ⅶ. 참고문헌')]
        sections=[{'id':i,'title':t,'status':'초안 준비 중','content_markdown':'절별 원고 입력을 기다리고 있습니다.'} for i,t in titles]
    intro_path=next((p for p in [REVIEWS/'03_opus_checked_introduction.tex',REVIEWS/'02_source_corrected_introduction.tex',REVIEWS/'01_initial_introduction.tex'] if p.exists()),None)
    for section in sections:
        if section['id']=='introduction' and intro_path:
            section['subsections']=section.get('subsections',[])+[{
                'id':'current-english-intro','title':'현재 영문 서론 작업본','status':'문헌 대조 중',
                'content_latex':intro_path.read_text(encoding='utf8')}]
        if section['id']=='results':
            section.update({'pdf_path':'../ppcf_revision/main_ppcf.pdf','pdf_label':'검증된 기존 PPCF 결과 원고 열기'})
    return {'title':draft.get('title','KSTAR CES nowcasting 연구 원고'),
            'subtitle':'졸업논문 구조와 문헌 근거를 함께 읽는 작업본',
            'status_note':'자료는 현재 검토 진행 상태입니다. 완료된 결과와 향후 계획을 구분합니다.',
            'sections':sections}


def render_section(section,level=2,fallback_id='section',seen=None):
    if seen is None:
        seen=set()
    sid=section.get('id') or fallback_id
    base=sid; suffix=2
    while sid in seen:
        sid=base+'-'+str(suffix); suffix+=1
    seen.add(sid)
    title=section.get('title','')
    status=section.get('status','')
    status={'mixed':'기존 결과·향후 계획 구분','structural':'구조 초안','planned':'연구 계획',
            'evaluated':'기존 평가 범위','metadata-unavailable':'메타데이터 미확인'}.get(status,status)
    body=section.get('content_markdown') or tex_to_markdown(section.get('content_latex',''))
    result='<section class="manuscript-section" id="'+html.escape(sid,quote=True)+'">'
    result+='<header class="section-heading"><h'+str(level)+'>'+html.escape(title)+'</h'+str(level)+'>'
    if status:
        result+='<span class="status">'+html.escape(status)+'</span>'
    result+='</header><div class="prose">'+render_content(body)+'</div>'
    if level==2 and section.get('subsections'):
        result+='<nav class="section-toc" aria-label="이 절의 구성">'
        for i,child in enumerate(section['subsections']):
            target=child.get('id') or sid+'-sub-'+str(i+1)
            result+='<a href="#'+html.escape(target,quote=True)+'">'+html.escape(child.get('title',''))+'</a>'
        result+='</nav>'
    if section.get('pdf_path'):
        path=section['pdf_path']; page=section.get('pdf_page')
        href=safe_link(path+('#page='+str(page) if page else ''))
        result+='<p><a class="document-link" href="'+html.escape(href,quote=True)+'" target="_blank" rel="noopener">'+html.escape(section.get('pdf_label','연결된 원문 PDF 열기'))+'</a></p>'
    if not section.get('pdf_preview_after_text'):
        result+=render_pdf_preview(section)
    for i,sub in enumerate(section.get('subsections',[])):
        result+=render_section(sub,min(level+1,5),sid+'-sub-'+str(i+1),seen)
    if section.get('pdf_preview_after_text'):
        result+=render_pdf_preview(section)
    result+=render_pdf_exhibits(section)
    return result+'</section>'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=REVIEWS/'manuscript_sections.json')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    document=read_json(args.input) if args.input.exists() else fallback_sections()
    sources,aliases=load_sources()
    for number,source in enumerate(sorted(sources,key=lambda row:row['title'].casefold()),1):
        source['number']=number
    by_key={source['key']:source for source in sources}
    CITATION_LABELS.clear()
    PDF_PREVIEWS.clear()
    for alias,key in aliases.items():
        source=by_key[key]
        CITATION_LABELS[alias]=('['+str(source['number'])+']',source['title'])
    nav=''.join('<button type="button" class="section-nav" data-section="'+html.escape(s['id'],quote=True)+'">'+html.escape(s['title'])+'</button>' for s in document['sections'])
    seen=set()
    manuscript=''.join(render_section(s,fallback_id='part-'+str(i+1),seen=seen) for i,s in enumerate(document['sections']))
    manifest={'title':document.get('title','KSTAR CES 연구 원고'),
        'subtitle':document.get('subtitle',''),'status_note':document.get('status_note',''),
        'sources':sources,'aliases':aliases,'section_ids':[s['id'] for s in document['sections']]}
    template=(HERE/'reader_template.html').read_text(encoding='utf8')
    result=template.replace('<!--SECTION_NAV-->',nav).replace('<!--MANUSCRIPT-->',manuscript)
    result=result.replace('/*READER_DATA*/',json.dumps(manifest,ensure_ascii=False).replace('</','<\\/'))
    result=result.replace('<!--TITLE-->',html.escape(manifest['title'])).replace('<!--SUBTITLE-->',html.escape(manifest['subtitle']))
    result=result.replace('<!--STATUS_NOTE-->',html.escape(manifest['status_note']))
    (OUT/'index.html').write_text(result,encoding='utf8')
    summary={'input':str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else args.input.name,
        'parent_content_loaded':args.input.exists(),'sections':len(document['sections']),
        'source_records':len(sources),'reading_states':{m:sum(s['mode']==m for s in sources) for m in ['full_pdf','full_text','summary','held','duplicate']},
        'self_contained_css_js':True,'automatic_network_requests':False,'pdf_previews':PDF_PREVIEWS,
        'source_exhibit_ids':re.findall(r'id="exhibit-([^"]+)"',result)}
    (OUT/'build_status.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('READER BUILT:',len(sources),'source records;',len(document['sections']),'sections')


if __name__=='__main__':
    main()
