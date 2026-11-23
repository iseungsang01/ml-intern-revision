"""Author-expanded full public-corpus reading after the36 original-PDF audits.

All remaining paper texts are supplied in full, page tagged. Poor/scanned text
falls back to the whole original PDF. Figure-dependent claims are flagged for
visual source checks, never inferred from extracted equations or captions alone.
"""
import concurrent.futures
import json

import introduction_corpus_review as R


def audit(row):
    key=row['key']
    # The final source was read in the bounded manuscript-writing request.
    # Reuse that verified record instead of paying for a second source audit.
    saved=R.OUT/('background_'+key+'.json')
    if saved.exists():
        completed=json.loads(saved.read_text(encoding='utf8'))
        if completed.get('funding_ledger')=='docs/paper/ppcf_revision/nnces_edit_usage.json':
            assert completed['key']==key
            assert completed['supplied_pdf_sha256']==row['sha256']==R.D.sha(R.ROOT/row['path'])
            assert completed['supplied_source_mode']=='full_original_pdf'
            return completed
    source=json.loads((R.CACHE/'texts'/(key+'.json')).read_text(encoding='utf8'))
    assert source['sha256']==row['sha256']==R.D.sha(R.ROOT/row['path'])
    nonempty=sum(len(p.strip())>120 for p in source['pages'])
    pdf_mode=not row['text_extractable'] or nonempty < .6*len(source['pages'])
    if pdf_mode:
        content=[R.doc(R.ROOT/row['path'],key+' full original PDF '+str(row['pages'])+'pages')]
        mode='full_original_pdf'
    else:
        body='\n\n'.join('ORIGINAL_PDF_PAGE '+str(i+1)+'\n'+p for i,p in enumerate(source['pages']))
        content=[R.D.textblock('ENTIRE EXTRACTED ORIGINAL PAPER TEXT, ALL PAGES. No page images were supplied. Equations/figures may extract imperfectly.\n'+body)]
        mode='full_extracted_text_no_page_images'
    task=r'''Individually read the entire supplied original source and judge its fit to our paper. This is one paper, not an abstract screen. For text-only sources, no plots/page images were supplied: do not claim to have visually checked figures. No raw study data is supplied. Do not repeat generic field summaries. Only specific source-grounded information.
Return concise strict JSON {key,citation:{title,authors,year,venue,volume,pages,doi_or_arxiv},relevance:direct|supporting|peripheral|unsuitable,topic,own_question,actual_method_and_evidence,claim_checks:[{allowed_intro_or_theory_claim,original_pdf_locator,qualification,figure_or_equation_visual_check_needed}],input_target_timing_if_relevant,validation_if_relevant,forbidden_inferences:[...],could_change_intro_or_theory:boolean,reason_to_include_or_exclude,bibtex_if_worth_citing}.
At most2 claim_checks. Peripheral/unsuitable papers: <=220words total, it is fine to give no claim and no bibtex. Direct/supporting: <=400words total. Unknown split/arrival time/truth stays unknown; do not fill missing evidence. A requirement or author's interpretation is not achieved performance or proof of physics. Distinguish impurity versus main-ion Ti, electron versus ion diagnostics, offline processing versus online control, current-report nowcasting versus future prediction. Do not generalize any single paper to all prior work. Scientific theory and concepts can support our theoretical-background section without becoming a new implemented method. A DOI in the reference list is not the source's own DOI. Do not quote more than15words or output private reasoning. SOURCE RECORD:\n'''
    content += [R.D.textblock(R.CONTEXT),R.D.textblock(task+json.dumps({
        'key':key,'citation_hint':row.get('citation'),'identifier_hint':row.get('identifier'),
        'pages':row['pages'],'source_mode':mode},ensure_ascii=False))]
    result=R.call('background_'+key,'claude-opus-5-5',content,max_tokens=4096,effort='low')
    if result.get('key')!=key:
        result['source_label_as_returned']=result.get('key')
        result['key']=key
    result['supplied_source_mode']=mode
    result['supplied_pdf_pages']=row['pages']
    result['supplied_pdf_sha256']=row['sha256']
    result['source_scope_note']=('Original PDF supplied including all pages' if pdf_mode else
        'All extracted page texts supplied; figure/equation visual validation remains separate')
    R.save(R.OUT/('background_'+key+'.json'),result)
    return result


def main():
    rows=json.loads((R.OUT/'catalogue.json').read_text(encoding='utf8'))['papers']
    eligible=[r for r in rows if r['external_review_allowed'] and not r['duplicate_of_prior']]
    original={r['key'] for r in eligible if (R.OUT/('paper_'+r['key']+'.json')).exists()}
    assert len(original)==36
    remaining=[r for r in eligible if r['key'] not in original]
    R.save(R.OUT/'budget_allocation.json',{'author_additional_ceiling_usd':50,
         'main_literature_workflow_ceiling_usd':49.218,'parallel_thesis_reader_closed_paid_usd':.781884,
         'scope':'All unique public papers; institutional guide held local; HTML/thesis structure in parallel'})
    done,errors={},{}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(audit,row):row['key'] for row in remaining}
        for future in concurrent.futures.as_completed(futures):
            key=futures[future]
            try:
                done[key]=future.result()
            except Exception as error:
                errors[key]=type(error).__name__+': '+str(error)
            R.save(R.OUT/'all_corpus_progress.json',{'state':'individual_full_source_reading',
                'original_full_pdf_reviews':36,'remaining_planned':len(remaining),
                'completed_additional':list(done),'errors':errors,'scientific_runs':0})
    assert not errors, 'Unresolved source audits; no invented completion'
    R.save(R.OUT/'all_corpus_status.json',{'state':'full_source_reading_complete_synthesis_pending',
          'complete':False,'inventory_files':len(rows),'unique_public_papers':len(eligible),
          'original_full_pdf_reviews':36,'additional_source_reviews':len(done),
          'additional_full_pdf_reviews':sum(r['supplied_source_mode']=='full_original_pdf' for r in done.values()),
          'additional_extracted_text_reviews':sum(r['supplied_source_mode']=='full_extracted_text_no_page_images' for r in done.values()),
          'excluded_from_external_review':['KFE2026DiagnosticsDataUserGuide'],
          'scientific_runs':0,'estimated_api_usd':R.ledger()['estimated_returned_usage_usd']})
    print('ALL PUBLIC SOURCES REVIEWED',len(eligible),'SYNTHESIS PENDING',flush=True)


if __name__=='__main__':
    main()
