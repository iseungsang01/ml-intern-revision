"""Author-requested concise main-text target description with appendix detail."""
import json

import verify_with_claude as V

OUT=V.ROOT/'docs/paper/research_reader/definition_audit'

def main():
    document=V.read(V.ROOT/'docs/paper/introduction_review/manuscript_sections.json')
    def walk(node):
        yield node
        for child in node.get('subsections',[]):yield from walk(child)
    target=[n for s in document['sections']for n in walk(s)if '표적 F와 L5'in n.get('title','')]
    assert len(target)==1
    lean=(V.ROOT/'docs/paper/b97_lean.md').read_text(encoding='utf8')
    lean=lean[lean.index('## 증명과 경계'):lean.index('## 실제 trace')]
    task='''Author asks in Korean: "Is L5 a Lean-based definition? Keep it brief; should this detail be in an appendix? Do it with Claude." Apply an EDITORIAL relocation, not a new scientific protocol. L5 means following-five-grid-report median, not Lean. Existing Lean proves a visible-source selector's index/eligibility/latest/prefix/readout properties, not that L5 is current physical temperature/noise-free. Original target implementation stays: next5GRIDrows t+1..t+5, >=3measured, median. No new fits, no deletion of results, no changed frozen registration or numerical outcome. Existing L5 figures and performance comparisons remain explicitly L5. Main actual nowcast targetF is current report. Do not equate future-label use with input leakage.
Return strictJSON {lean_answer_ko,main_replacement_ko,appendix:{id,title,content_markdown},changes_check}. main_replacement_ko should be2-3short sentences, <=230Koreancharacters, definingF and saying L5is auxiliary following-window report level with a Markdown link [부록 A](#appendix-l5). No detailed5/3conditions, irreducible-noise claims, defensive repetition or longlist. Appendix id=appendix-l5,title=부록 A. 보조 표적 L5의 정의; explain exact formula/validity, gridvsnextobservations, missingness, why it is separate from currentF, and why existing2sidedreferences that read label-windowvalues weren't scored onL5. That exclusion is a protocol choice, not a theorem that no offlinebaseline can ever be constructed. Use1displayequation and2-3briefexplanatoryparagraphs or shorttable;450-900Koreancharacters total. No claims currenttruth/denoising/Leanjustifieslabel. No research-instructionplaceholders. Lean distinction is for author reply, don't introduce Lean into paper body/appendix as though it groundsL5. This is not the broad F-focused rewrite recommended by previous reviewers; only shorten definition and move detail, keep results. DATA:\n'''
    data={'existing_target_section':target[0],'source_selector_Lean_scope':lean,
          'author_objective':'currentCESreportnowcasting','already_received_feedback':'Operationaldefinitionlegitimate; futurewindowestimandnotindependentcurrenttruth. ExistingtargetvariantstudyonlyV1/tree,notphysicalvalidation.'}
    V.call('06_fable_concise_L5_appendix','claude-fable-5-2',[V.block(task+json.dumps(data,ensure_ascii=False))],10000,output_dir=OUT)

if __name__=='__main__':main()
