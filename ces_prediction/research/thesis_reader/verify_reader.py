"""Exercise the real local reader without network access or invented study data."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from build_reader import ROOT,quantitative_tokens,load_sources

OUT = ROOT / 'docs/paper/research_reader'
QA = ROOT / 'data/.thesis_reader/qa'


def main():
    QA.mkdir(parents=True,exist_ok=True)
    results={}
    manuscript=json.loads((ROOT/'docs/paper/introduction_review/manuscript_sections.json').read_text(encoding='utf8'))
    if manuscript.get('active_longform'):
        counts=[]
        def inspect(section):
            for paragraph in section.get('content_markdown','').split('\n\n'):
                if paragraph.strip() and not paragraph.lstrip().startswith('|'):
                    counts.append(len(quantitative_tokens(paragraph)))
            for child in section.get('subsections',[]):inspect(child)
        for section in manuscript['sections']:
            if section['id']!='references':inspect(section)
        assert max(counts)<=4,'Body paragraph exceeds author numerical-density limit'
        results['body_numeric_tokens_max']=max(counts)
        results['body_paragraphs_checked']=len(counts)
    with sync_playwright() as play:
        browser=None
        for kwargs in ({'headless':True},{'headless':True,'channel':'chrome'},{'headless':True,'channel':'msedge'}):
            try:
                browser=play.chromium.launch(**kwargs);break
            except Exception:
                continue
        assert browser is not None,'No installed headless Chromium available'
        context=browser.new_context(viewport={'width':1440,'height':1000})
        context.set_offline(True)
        page=context.new_page(); errors=[]; network=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request',lambda request:network.append(request.url) if request.url.startswith(('http://','https://')) else None)
        page.goto((OUT/'index.html').resolve().as_uri(),wait_until='load')
        assert page.locator('.paper-view>.manuscript-section').count()==7
        assert page.evaluate('window.readerTest.sourceCount')==len(load_sources()[0])
        ids=page.locator('[id]').evaluate_all('(els)=>els.map(e=>e.id)')
        assert len(ids)==len(set(ids)),'Duplicate element IDs'
        assert page.locator('h1').inner_text()
        assert 'GP-Boost-Multi' in page.locator('#materials-methods').inner_text()
        assert 'TCN-Multi' in page.locator('#exhibit-table11 .exhibit-source').inner_text()
        assert 'GP-Boost-Multi' in page.locator('#exhibit-figure3 .exhibit-source').inner_text()
        results['model_names_distinguish_architecture_and_inputs']=True
        page.screenshot(path=str(QA/'reader_desktop.png'))
        page.get_by_role('button',name='문헌·인용 근거',exact=True).click()
        page.locator('#source-search').fill('NN-CES')
        page.locator('#mode-filter').select_option('full_pdf')
        assert page.locator('.source-row').count()>=1
        page.locator('.source-row[data-key="Lee2026NeuralNetworkbased"]').click()
        assert page.locator('#source-dialog').is_visible()
        assert page.locator('#source-dialog-title').inner_text()
        assert 'Lee2026' in page.locator('#dialog-key').inner_text()
        assert page.locator('.claim-note').count()>=1
        page.screenshot(path=str(QA/'reader_source_proof.png'))
        page.keyboard.press('Escape')
        assert not page.locator('#source-dialog').is_visible()
        page.locator('#source-search').fill('this-specific-source-does-not-exist')
        assert page.locator('.source-row').count()==0
        assert page.locator('.empty').is_visible()
        page.get_by_role('button',name='원고 읽기',exact=True).click()
        page.locator('.section-nav[data-section="materials-methods"]').click()
        assert page.locator('#materials-methods').is_visible()
        assert 'Neural Network' in page.locator('#materials-methods').inner_text()
        page.screenshot(path=str(QA/'reader_methods.png'))
        if page.locator('#gp-theory').count():
            gp=page.locator('#gp-theory')
            assert gp.locator('.display-math').count()==9
            assert gp.locator('table').count()==1
            gp.locator('.citation[data-source="Rasmussen2006GPML"]').first.click()
            assert page.get_by_role('link',name='공개 원문 열기').get_attribute('href').startswith('https://gaussianprocess.org/')
            assert '[?]' not in gp.inner_text()
            page.keyboard.press('Escape')
            page.locator('#gp-theory-sub-4').screenshot(path=str(QA/'reader_gp_kernel.png'))
            page.locator('#gp-theory-sub-8').screenshot(path=str(QA/'reader_gp_multimodal.png'))
            results['gp_equations_and_input_comparison_and_citation']=True
        citation=page.locator('.citation').first
        if citation.count():
            assert citation.inner_text().startswith('[')
            assert citation.inner_text().endswith(']')
            assert citation.get_attribute('aria-label')
            citation.click()
            assert page.locator('#source-dialog').is_visible()
            page.keyboard.press('Escape')
        page.evaluate("window.readerTest.showSource('Haskey2017DeuteriumCharge')")
        if page.get_by_text('서지 확인 메모',exact=True).count():
            page.get_by_text('서지 확인 메모',exact=True).click()
            assert page.locator('#dialog-content details[open]').inner_text()
            results['bibliographic_note']=True
        page.keyboard.press('Escape')
        previews=page.locator('.pdf-preview')
        exhibits=page.locator('.source-exhibit')
        if exhibits.count():
            assert exhibits.count()==18
            if page.locator('#exhibit-figure1 img').get_attribute('src').startswith('data:image/svg+xml;base64,'):
                for number in range(1,8):
                    assert page.locator('#exhibit-figure'+str(number)+' img').get_attribute('src').startswith('data:image/svg+xml;base64,')
                results['original_vector_figures']=7
            assert page.locator('#exhibit-table9').count()==1
            assert page.locator('#exhibit-table10').count()==1
            assert page.locator('#exhibit-table11').count()==1
            assert page.locator('#results-sub-6 #exhibit-table11').count()==1
            for target in page.locator('.section-toc a').evaluate_all('(links)=>links.map(a=>a.hash.slice(1))'):
                assert page.locator('[id="'+target+'"]').count()==1
            results['source_tables_and_figures']=exhibits.count()
        if previews.count():
            preview=previews.first
            if preview.get_attribute('open') is None:
                preview.locator('summary').click()
            images=preview.locator('img')
            assert images.count()>0
            preview.scroll_into_view_if_needed()
            images.first.scroll_into_view_if_needed()
            page.wait_for_function('(img)=>img.complete && img.naturalWidth>0',arg=images.first.element_handle(),timeout=10000)
            assert images.first.evaluate('(img)=>img.complete && img.naturalWidth>0')
            assert images.first.get_attribute('src').startswith('data:image/png;base64,')
            page.screenshot(path=str(QA/'reader_results_preview.png'))
            preview.locator('summary').click()
            assert not preview.get_attribute('open')
            page.emulate_media(media='print')
            page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
            assert preview.get_attribute('open') is not None
            page.emulate_media(media='screen')
            results['embedded_results_pages']=images.count()
        page.evaluate("window.readerTest.showSource('unmapped-citation-key')")
        assert page.locator('#source-dialog-title').inner_text()
        page.keyboard.press('Escape')
        results.update({'desktop_sections':7,'source_records':len(load_sources()[0]),'search_and_filters':True,
                        'compact_citation_labels':True,
                        'citation_dialog':True,'escape_and_focus':True,'empty_state':True,
                        'unknown_citation_state':True,'section_navigation':True})
        if manuscript.get('appendices'):
            page.locator('a[href="#appendix-l5"]').first.click()
            assert page.locator('#appendix-l5').get_attribute('open') is not None
            assert page.locator('#appendix-l5 math').count()==1
            page.locator('#appendix-l5').screenshot(path=str(QA/'reader_appendix.png'))
            results['appendix_link_opens_definition']=True
        page.set_viewport_size({'width':390,'height':844})
        page.evaluate('window.scrollTo(0,0)')
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),'Mobile horizontal overflow'
        page.screenshot(path=str(QA/'reader_mobile.png'))
        page.get_by_role('button',name='문헌·인용 근거',exact=True).click()
        page.locator('#source-search').fill('NN-CES')
        page.locator('#source-list .source-row').first.click()
        assert page.locator('#source-dialog').is_visible()
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        results['mobile_390px_no_overflow']=True
        assert not errors,errors
        assert not network,network
        results.update({'js_errors':errors,'automatic_http_requests':network,'offline_passed':True})
        browser.close()
    (OUT/'qa_status.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('READER UI CHECKS PASSED')


if __name__=='__main__':
    main()
