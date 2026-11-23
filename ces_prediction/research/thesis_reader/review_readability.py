"""User-requested Fable editorial pass on actual long-form manuscript (no fits)."""
import json
import re
import sys

from rewrite_longform import OUT,REV,ROOT,call,save
from build_reader import quantitative_tokens


def numeric_tokens(text):
    return quantitative_tokens(text)


def paragraphs(document):
    result=[]
    def visit(section,sid):
        for paragraph in section.get('content_markdown','').split('\n\n'):
            if paragraph.strip():
                result.append({'section_id':sid,'subsection_title':section.get('title'),
                               'text':paragraph,'numeric_tokens':numeric_tokens(paragraph)})
        for child in section.get('subsections',[]):visit(child,sid)
    for section in document['sections']:
        if section['id']!='references':visit(section,section['id'])
    return result


def main():
    if '--closeout' in sys.argv:
        document=json.loads((REV/'manuscript_sections.json').read_text(encoding='utf8'))
        sample={s['id']:s for s in document['sections'] if s['id']=='conclusion'}
        dense=json.loads((OUT/'numeric_density_after.json').read_text(encoding='utf8'))
        first=json.loads((OUT/'04_fable_readability_recovered.json').read_text(encoding='utf8'))
        exhibits=json.loads((OUT/'exhibits.json').read_text(encoding='utf8'))
        packet={'prior_full_context_Fable_assessment':first['adequacy_assessment'],
           'prior_request_incomplete_but_only_closed_nine_edits_used':True,
           'implemented_edits':first['corrections'],
           'host_remaining_paragraph_edits':json.loads((OUT/'readability_host_corrections.json').read_text(encoding='utf8')),
           'numeric_density_after':dense,'counts':json.loads((OUT/'integration_status.json').read_text(encoding='utf8'))['section_characters'],
           'source_exhibit_inventory':exhibits,
           'final_conclusion':sample,
           'host_layout_override':'TCN runtime Table11 is placed in TCN Results6, not GP runtime Results3; caption explicitly says TCN. All18PDF crops individually rendered/visually checked by host team; no page images supplied to this request.'}
        result=call('05_fable_editorial_closeout','claude-fable-5-2',
          'Complete a NARROW final editorial check after the previous output-limited full-context pass. Do not re-review all source science or claim to have viewed HTML/PDF images. Check the supplied actual edit operations, density audit, full final conclusion and exhibit inventory. Is there sufficient substantial prose for author reading/cutting, are results and explanatory exhibits matched, do the paragraph splits/table transfer preserve meaning, and are there any remaining concrete usability problems? No new data/results, no acceptance grade. Distinguish measured runtime of TCN from GP+boosting and keep all4designs. Return strict JSON with keys scope,editorial_verdict,volume_assessment,numerical_prose_assessment,exhibit_organization,conclusion_assessment,necessary_fixes:[{section_id,old_exact,new,reason}],remaining_author_choices. At most2 indispensable fixes; no whole-manuscript rewrite, no restatement of all evidence, no more than900words visible output. DATA:\n'+json.dumps(packet,ensure_ascii=False),10000)
        print('NARROW FABLE EDITORIAL CLOSEOUT COMPLETE',flush=True)
        return
    manuscript=json.loads((REV/'manuscript_sections.json').read_text(encoding='utf8'))
    dense=[p for p in paragraphs(manuscript) if len(p['numeric_tokens'])>4]
    save(OUT/'numeric_density_before.json',{'method':'Arabic quantitative tokens excluding citation keys and display math; identifiers such as L5/V1 are not numbers. Reference entries are exempt.',
                                        'paragraphs_over_four':dense})
    tex=(ROOT/'docs/paper/ppcf_revision/main_ppcf.tex').read_text(encoding='utf8')
    exhibit_captions=re.findall(r'\\caption\{(.+?)\}\s*\\label\{([^}]+)\}',tex,flags=re.S)
    packet={'actual_current_manuscript':manuscript,'numeric_dense_paragraphs':dense,
       'existing_figure_and_table_labels_and_captions':exhibit_captions,
       'planned_layout':'Existing generated PDF tables/figures will be cropped directly with caption+all rows and placed beside relevant Methods/Results subsections. No retyping result values. Tables9/10/11 float to PDF18/19 and will be included explicitly. Main original Results pages remain optional appendix overview.',
       'actual_character_counts':json.loads((OUT/'integration_status.json').read_text(encoding='utf8'))['section_characters']}
    prompt='''Korean thesis manuscript editorial revision requested by author AFTER the source/Opus/host correction pass. The actual full manuscript is attached as text. Check whether there is enough explanatory substance for the author to read and later shorten, whether tables/figures support the relevant argument close to its prose, and what makes reading awkward. Author explicitly wants NO MORE THAN3-4 quantitative numbers in a body paragraph, preferably<=3. Detailed numerical comparisons belong to source-bound tables/figures. Reference entries, model names/identifiers F/L5/V1/D650 and equation indices are not numerical-result prose. Do not disguise density by spelling all numbers in Korean. Reduce it by explanatory paragraph organization or transferring details to the relevant exhibit.

Preserve the substantial long-form writing, all7top-level sections and original Dataset/collectioncriteria/scenario/time/campaign-state/NNdesign/architecture/modification/stage1-2 hierarchy. Do not compress the whole paper into an outline. Do not replace actual conclusions with instructions on what a conclusion should say. Keep positive scientific explanation, reduce repetitive rebuttals/caveats and long mechanical enumerations. Review especially literature comparisons and conclusion. No new results or experiment, no new physical mechanism, no broad absence/novelty claim, no new acceptance grade. Do not reverse verified source corrections: this is current CES report nowcasting, not forecasting future physical Ti nor mathematical identity with filtering. F=currentframe;L5=median measured values next5gridrows >=3; both targets not currentinputs. MSE eV²;RMSE/MAE eV; difference null0,ratio null1. Four validation designs equal. GP+boosting main and budgetTCN have different support/budgets. D650fixed rule differsfromSschedule. Actualarrivalunknown. Sourceaudits and originalSnyderverifyEPED'sPB/KBMconstraints.

No performance value may be invented or typed into a new prose table. Preserve actual PDF table/figure labels. You may propose moving already present publication specifications or a theoretical worked example into a small display table ONLY if values come verbatim from the supplied paragraph and exact citation remains. Host verifies those separately. Prefer replacing a dense paragraph by several explanatory paragraphs referencing existing exhibits, without deleting needed definitions. Display equations may be separate from prose. At most14 material replacements; prioritize all numeric_dense_paragraphs plus the worst readability problems. old_exact must occur exactly once in a content_markdown string. new can contain paragraph breaks and a Markdown table if needed. Each new prose paragraph MUST contain<=4 quantitative numeric tokens (citation keys excluded). Do NOT repeat source contents or return a complete manuscript.

Return strict JSON {adequacy_assessment:{substance,explanation_gaps,conclusion,layout},corrections:[{section_id,old_exact,new,reason,source_or_existing_exhibit}],exhibit_placement_recommendations:[{section_id,subsection,existing_label,purpose,reading_order}],reading_discomforts:[...],remaining_numeric_density_issues:[...]}. Be concrete, keep all complete output within12000 tokens including thinking, no fences. DATA:\n'''
    result=call('04_fable_readability_and_exhibits','claude-fable-5-2',prompt+json.dumps(packet,ensure_ascii=False),14000)
    print('FABLE READABILITY REVIEW READY; HOST APPLY REQUIRED',flush=True)


if __name__=='__main__':
    main()
