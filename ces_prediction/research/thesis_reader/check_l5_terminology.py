"""Bounded literature terminology audit; text search is not full rereading."""
import json
import re

import verify_with_claude as V

OUT=V.ROOT/'docs/paper/research_reader/definition_audit'

def main():
    documents=[];hits=[]
    for path in sorted((V.ROOT/'data/.introduction_corpus/texts').glob('*.json')):
        d=V.read(path)
        documents.append({'key':path.stem,'sha256':d['sha256'],'pages':len(d['pages'])})
        for page,text in enumerate(d['pages'],1):
            for match in re.finditer(r'\bL\s*[_ {]?\s*5\b',text):
                hits.append({'key':path.stem,'page':page,'context':text[max(0,match.start()-80):match.end()+100]})
    sources=[
      {'source':'Lee et al., NN-CES','url':'https://doi.org/10.1016/j.fusengdes.2025.115518','locator':'p3, losses Eq6; p4 model outputs',
       'finding':'Targets include conventional-analysis Doppler shift and broadening; no exact future-five-grid median target identified in the examined target description.'},
      {'source':'Jalalvand et al., Diag2Diag','url':'https://doi.org/10.1038/s41467-025-63492-1','locator':'pp9-11, time-series development and evaluation',
       'finding':'Targets are TS diagnostic channels, with heteroscedastic output. No exact future-five-grid median target identified in the examined target description.'},
      {'source':'Ross et al., 2026 preprint','url':'https://arxiv.org/html/2603.08242v1#S4.SS2','locator':'Section4.2, paragraph beginning We divide',
       'finding':'Future12-hour window divided into four3-hour bins. Observed medians within bins determine a binary switch-readiness label. Missing bins excluded for that variable. This IS an analogous use of future-window observed medians in label construction, but not the same scalar regression target, five-slot rule, missingness threshold, or plasma claim. No L5 name is given there. Forecast model itself is trained on target observations, not directly on a scalar bin median.'},
      {'source':'Rostami-Tabar et al., 2013','url':'https://doi.org/10.1002/nav.21546','locator':'Publisher abstract',
       'finding':'Uses established category temporal aggregation for lower-frequency demand buckets. Abstract does not establish median aggregation, a specific L5 operation or its physical interpretation.'}
    ]
    packet={'scope':'254 cached PDF text records, not necessarily254 unique studies; extracted-text symbol search plus targeted source passages, not a fresh full reread or exhaustive proof of absence.',
      'documents':documents,'unique_pdf_hashes':len({d['sha256']for d in documents}),
      'symbol_hits':hits,'host_hit_classification':'All returned hits were Nature volume/page typography or reference page range L1-L5, not the study target.',
      'our_exact_definition':'Within same file/contiguous block, median of usable observed Ti in next five nominal10ms grid slots, current frame excluded, >=3valid; missing slots do not extend window. Label only, not online input. L5 is following-level label used in this project, unrelated to Lean.',
      'primary_source_findings':sources,
      'candidate_descriptive_name':'후속 시간창 중앙값 표적 / future-window median target. Descriptive phrase, not asserted conventional unique proper name.',
      'avoid_confusions':['median of predictive distribution','median ensemble of forecasts','classification label smoothing','ordinary centered or trailing median filter','physical denoised current truth']}
    V.save(OUT/'terminology_evidence.json',packet)
    # Send public paraphrases and aggregate scope, not full corpus/identity/observations.
    external={k:v for k,v in packet.items()if k not in ['documents','symbol_hits']}
    task='''Author asks whether other studies define L5, and whether it has an established alternative name. Audit supplied evidence skeptically. Do not claim an exhaustive negative literature result. Distinguish exact five-grid >=3 median from analogous future-bin observed median and generic temporal aggregation. The Ross preprint is a real analogous LABEL construction, not direct training of scalar median regression; it cannot validate currentCES truth. Recommend plain Korean/English wording without inventing standard terminology. No renaming frozen code, no experiments. Return strict JSON {answer_ko,established_L5_evidence,analogous_study_and_difference,recommended_descriptive_term:{ko,en,status},terms_to_avoid,proposed_appendix_sentence_ko,limitations}. <=650visible words. Evidence follows:\n'''
    V.call('07_fable_L5_terminology','claude-fable-5-2',[V.block(task+json.dumps(external,ensure_ascii=False))],10000,output_dir=OUT)

if __name__=='__main__':main()
