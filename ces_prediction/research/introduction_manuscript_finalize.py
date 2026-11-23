"""Complete the source-grounded manuscript within the original PPCF-edit allowance.

The corpus+reader branch remains below its additional USD50 cap. This writing
request uses the existing USD3 manuscript-edit cap (within the earlier USD30
allowance), including its final remaining theoretical-background source check.
"""
import json
import re

import fitz
import introduction_corpus_review as R
import ppcf_nnces_edit as P


def snippet(row, pages):
    path=R.ROOT/row['path']
    assert R.D.sha(path)==row['sha256']
    dest=R.CACHE/'writing_snippets'/(row['key']+'.pdf')
    dest.parent.mkdir(parents=True,exist_ok=True)
    original=fitz.open(path); out=fitz.open()
    if dest.exists():
        cached=fitz.open(dest)
        assert len(cached)==len(pages)
        assert all(cached[i].get_text()==original[p-1].get_text() for i,p in enumerate(pages))
        return R.doc(dest,'SOURCE_KEY '+row['key']+' ORIGINAL_PDF_PAGES '+','.join(map(str,pages)))
    for page in pages:
        assert 1<=page<=len(original)
        out.insert_pdf(original,from_page=page-1,to_page=page-1)
    out.save(dest,garbage=4,deflate=True)
    return R.doc(dest,'SOURCE_KEY '+row['key']+' ORIGINAL_PDF_PAGES '+','.join(map(str,pages)))


def main():
    rows=json.loads((R.OUT/'catalogue.json').read_text(encoding='utf8'))['papers']
    by_key={r['key']:r for r in rows}
    eligible=[r for r in rows if r['external_review_allowed'] and not r['duplicate_of_prior']]
    audits={}
    for r in eligible:
        paths=[R.OUT/('paper_'+r['key']+'.json'),R.OUT/('background_'+r['key']+'.json')]
        existing=next((p for p in paths if p.exists()),None)
        if existing:
            audits[r['key']]=json.loads(existing.read_text(encoding='utf8'))
    missing=[r for r in eligible if r['key'] not in audits]
    assert len(missing)<=1, 'Complete the other source audits before manuscript writing'
    # Full detailed evidence for relevant papers; brief exclusion records for the
    # rest still preserve the fact that every paper was individually considered.
    evidence=[]
    for key,a in audits.items():
        item={'key':key,'citation':a.get('citation'),'relevance':a.get('relevance')}
        if a.get('relevance') in ['direct','supporting']:
            item.update({k:a[k] for k in ['claim_checks','input_target_timing','validation',
                'runtime_or_deployment','forbidden_inferences','actual_method_and_evidence',
                'reason_to_include_or_exclude','visible_trailing_response_note'] if k in a})
        else:
            item.update({'topic':a.get('topic'),'reason':a.get('reason_to_include_or_exclude')})
        evidence.append(item)
    docs=[]
    # Recheck the key scientific premises against actual publisher PDF pages.
    pages={'Lee2026NeuralNetworkbased':[1,2,3,4,5,6,7],
           'Young2008PlasmaMeasurements':[9,10],
           'Donne2007ChapterDiagnostics':[9,17,18],
           'Shousha2023MachineLearningbased':[3,4,5,6,13,14,15],
           'Jalalvand2025MultimodalSuperresolution':[3,4,5,9,10],
           'Imrek2026TemporalSuperresolution':[4,5,6,8,9]}
    for key, chosen in pages.items():
        docs.append(snippet(by_key[key],chosen))
    if missing:
        row=missing[0]
        docs.append(R.doc(R.ROOT/row['path'],'FINAL_SINGLE_SOURCE '+row['key']+' FULL_ORIGINAL_PDF '+str(row['pages'])+'pages'))
    prior=json.loads((R.OUT/'02_interim_36_source_package.json').read_text(encoding='utf8'))
    skeleton=json.loads((R.OUT/'manuscript_sections.json').read_text(encoding='utf8'))
    outline={s['id']:{'title':s['title'],'subsections':s.get('subsections',[])}
             for s in skeleton['sections'] if s['id'] in ['introduction','background','materials-methods']}
    task=r'''FINAL source-grounded manuscript writing/editing. You are Opus following Fable's draft and separate per-paper original-source audits. An incomplete structured Fable response contained a complete Introduction field, recovered below; do not treat incomplete request as a completed review. The whole repository corpus was inventoried, screened, and individually read; actual evidence for relevant papers is supplied. Text-only audits did not inspect figures. Key premises are now supplied again as original PDF-page snippets with ORIGINAL_PDF_PAGES mappings. These source pages override erroneous reviewer summaries.

Preserve the author's original Korean graduation-thesis structure while strengthening the science. Top level: Introduction / Theoretical background / Materials and methods / Results / Discussion / Conclusion / References. Introduction subheadings a specific plan/objectives, b why needed, c expected effects, d difference from prior work. Materials and methods must retain 1Data set: collection criteria incl scenarios/plasma state, usable time intervals, campaign/device-state discharge classification; 2Design: NeuralNetwork a which NN architecture (compare several) b advantages/reasons for modification decided from measured performance. Current data are one2022campaign; no shot-level H/L/RMP/device-state labels have been established. Do not invent a multiyear analysis, operational taxonomy, or device modification. Validation groups are not physical regimes.

Author model-development flow: first compare CNN/lightweight models and references to test whether other diagnostic data provide useful temporal completion of CES reports; then design a NN using those results. Preserve this constructive progression as a research plan, distinguishing completed GP/boosting/LSTM/TCN comparisons from unrun architecture redesign. Do NOT call existing10ms-grid report agreement demonstrated submillisecond physical resolution recovery. No new fits, no fabricated accuracy, no global CNN superiority, no new experimental evidence. Performance comparison means regression errors and uncertainty on matched rows/information/budgets. Both targetF and offline future-derived L5 remain targets, not inputs. Use 'nowcasting current reported temperature', not forecasting future physical temperature. NN-CES upstream/current-spectrum reader and downstream temporal estimate are complementary potential stages, not implemented coupling. Do not claim every old method is electrons-only, noncausal, memoryless or lacks shot-wise validation. Preserve all4 validation designs equally. No acceptance estimates.

For theoretical background explain: pedestal/ion-temperature context; CES spectrum->Doppler parameters->carbon-impurity report, a simplified width/temperature relation only with clear Gaussian/instrumental-correction/energy-unit definitions; spectrum reading versus report-time availability; causal information sets/currentF and offlineL5; what local temporal CNN/TCN and recurrent/history models are candidates for, with no unverified implementation claim. Explain before equations and avoid unnecessary complexity. Direct paper claims, published design requirements, author's interpretation and our proposals differ. No new performance numbers; manuscript results/tables remain unchanged.

Return ONLY strict JSON, no fences:
{remaining_source_audit: null OR {key,citation:{title,authors,year,venue,doi_or_arxiv},relevance,topic,own_question,actual_method_and_evidence,claim_checks:[{allowed_intro_or_theory_claim,original_pdf_locator,qualification,figure_or_equation_visual_check_needed}],reason_to_include_or_exclude},
introduction_en_tex:complete English sectionIntroduction plus labelsec:intro,800-1100words,
theory_en_tex:complete English sectionTheoretical background plus labelsec:background,450-700words,
introduction_ko:{general,a_objectives,b_rationale,c_effects,d_distinction},
background_ko:{pedestal,ces_spectrum,temporal_information,learning_models},
materials_methods_corrections:[{subheading,correction}],
source_claim_map:[{section,claim,corpus_keys,original_pdf_locators,qualification}],
important_corrections:[...], korean_summary}.
English cite only audited source CORPUS_KEYS using \citep{CorpusKey}; Korean uses [@CorpusKey]. Host will map to existing BibTeX by DOI/title. At most25 distinct citations in the Introduction, prefer primary direct sources over huge reviews. Theory can add a few supported foundations. Every cited key must exist in supplied evidence or the final single-source PDF. Avoid dumping250citations. Korean introduction fields collectively about2200-3200Korean characters, background fields about2500-3500Korean characters. source_claim_map at most14 concise entries. Do not repeat all bibliographies or the raw audit catalogue. If FINAL_SINGLE_SOURCE is present, audit that ONE full original PDF first (<=250words) before integrating; all other original papers were already audited separately. No fabricated metadata if absent. Give complete final text before elaborating side issues.

EVIDENCE PACKET:
'''
    content=docs+[R.D.textblock(R.CONTEXT),
        R.D.textblock('FABLE DRAFT\n'+prior['introduction_tex']),
        R.D.textblock('AUTHOR OUTLINE AND CURRENT METHODS PLAN\n'+json.dumps(outline,ensure_ascii=False)),
        R.D.textblock(task+json.dumps(evidence,ensure_ascii=False))]
    result=P.call('03_source_grounded_thesis_sections','claude-opus-5-5',content,max_tokens=24576)
    R.save(R.OUT/'final_manuscript_sections_review.json',result)
    if missing:
        row=missing[0]; a=result['remaining_source_audit']
        assert a and a['key']==row['key']
        a.update({'supplied_source_mode':'full_original_pdf',
                  'supplied_pdf_pages':row['pages'],'supplied_pdf_sha256':row['sha256'],
                  'source_scope_note':'Full original PDF checked during final theoretical-background writing request',
                  'funding_ledger':'docs/paper/ppcf_revision/nnces_edit_usage.json'})
        R.save(R.OUT/('background_'+row['key']+'.json'),a)
    (R.OUT/'final_introduction_en.tex').write_text(result['introduction_en_tex'].rstrip()+'\n',encoding='utf8')
    (R.OUT/'final_theory_en.tex').write_text(result['theory_en_tex'].rstrip()+'\n',encoding='utf8')
    R.save(R.OUT/'manuscript_writing_provenance.json',{
        'workflow':'ces_prediction/research/introduction_manuscript_finalize.py',
        'funding':'Existing PPCF manuscript-edit ceiling USD3, within unused original USD30 allowance',
        'usage_record':'docs/paper/ppcf_revision/nnces_edit_usage.json',
        'corpus_reader_funding_separate':True,'original_source_snippets':pages,
        'remaining_full_pdf_check':[r['key'] for r in missing],
        'host_integration_pending':True,'scientific_runs':0,'whole_manuscript_grade':False})
    print('SOURCE-GROUNDED THESIS TEXT COMPLETE; HOST INTEGRATION PENDING',flush=True)


if __name__=='__main__':
    main()
