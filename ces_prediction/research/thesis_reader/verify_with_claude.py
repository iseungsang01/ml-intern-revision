"""Fresh, bounded USD5 manuscript/source + actual figure review. No scientific fits."""
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ces_prediction/research'))
import fable_deep_review as D

OUT=ROOT/'docs/paper/research_reader/figure_verification'
CACHE=ROOT/'data/.thesis_figure_verification'
PRICES={'claude-fable-5-2':(8,40),'claude-opus-5-5':(4,20)}
CAP=5.0

def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def read(p):return json.loads(p.read_text(encoding='utf8'))

def ledger():
    calls=[]
    for p in sorted(CACHE.glob('*.reservation.json')):
        row=read(p);response=CACHE/(row['stage']+'.response.json')
        if response.exists():
            result=read(response);u=result['usage'];ip,op=PRICES[row['model']]
            assert not u.get('cache_creation_input_tokens') and not u.get('cache_read_input_tokens')
            row.update(usage=u,stop_reason=result['stop_reason'],
               estimated_usd=(ip*u['input_tokens']+op*u['output_tokens'])/1e6,
               response_sha256=D.sha(response))
        calls.append(row)
    total=sum(c.get('estimated_usd',c['reserved_usd'])for c in calls)
    assert total<=CAP
    result={'cap_usd':CAP,'authorization':'Author requested an additional approximately USD5 Claude verification and proper figure placement',
       'estimated_or_reserved_usd':total,'invoice_verified':False,'prices_per_million':PRICES,
       'pricing_source':'https://claude.com/pricing?currency=USD','calls':calls}
    if any(c['stage'].startswith(('04_','05_')) for c in calls):
        result['followup_authorization']='Author then explicitly challenged custom definitions/L5 and supplied completed B105 results; skeptical definition audit uses unused part of same USD5 cap.'
    save(OUT/'api_usage.json',result)
    return result

def block(text):return {'type':'text','text':text}

def document(path,title):
    return {'type':'document','source':{'type':'base64','media_type':'application/pdf',
            'data':base64.b64encode(path.read_bytes()).decode()},'title':title}

def image(path):
    return {'type':'image','source':{'type':'base64','media_type':'image/png',
            'data':base64.b64encode(path.read_bytes()).decode()}}

def call(stage,model,content,max_tokens,output_dir=None):
    payload={'model':model,'max_tokens':max_tokens,'output_config':{'effort':'low'},
             'messages':[{'role':'user','content':content}]}
    request=CACHE/(stage+'.request.json');response=CACHE/(stage+'.response.json')
    if response.exists():
        assert read(request)==payload,'Changed task: no silent paid replay'
        result=read(response)
    else:
        assert not request.exists(),'Uncertain prior request: no replay'
        assert len(list(CACHE.glob('*.reservation.json')))<7
        count=D.api('/messages/count_tokens',{'model':model,'messages':payload['messages']})
        ip,op=PRICES[model];reserve=count['input_tokens']*1.1*ip/1e6+max_tokens*op/1e6+.02
        assert ledger()['estimated_or_reserved_usd']+reserve<=CAP
        save(request,payload);save(CACHE/(stage+'.reservation.json'),{'stage':stage,'model':model,
           'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
           'counted_input_tokens':count['input_tokens'],'reserved_usd':reserve})
        ledger();result=D.api('/messages',payload);save(response,result)
    ledger();text=D.txt(result).strip()
    destination=output_dir or OUT
    destination.mkdir(parents=True,exist_ok=True)
    (destination/(stage+(''if result['stop_reason']=='end_turn'else'_INCOMPLETE')+'.txt')).write_text(text+'\n',encoding='utf8')
    assert result['stop_reason']=='end_turn','Incomplete response preserved, not a completed audit'
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
    value=json.loads(text);save(destination/(stage+'.json'),value)
    print(stage,'COMPLETE',flush=True)
    return value

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manuscript=read(ROOT/'docs/paper/introduction_review/manuscript_sections.json')
    if '--closeout' in sys.argv:
        source=(ROOT/'docs/paper/ppcf/fig_v2.py').read_text(encoding='utf8')
        sparse=source[source.index('def fig_sparse():'):source.index('\n# ---',source.index('def fig_sparse():'))]
        readout=(ROOT/'ces_prediction/experiments/b78_sparse/readout78.py').read_text(encoding='utf8')
        equation=readout[readout.index('    def eqrate('):readout.index('\n\n',readout.index('        return out',readout.index('    def eqrate('))) ]
        packet={'applied_body_corrections':read(OUT/'applied_corrections.json'),
          'final_figure_captions':[i for i in read(ROOT/'docs/paper/research_reader/longform/exhibits.json')['items']if i['kind']=='figure'],
          'existing_sparse_plot_function_no_data_arrays':sparse,
          'existing_equivalent_interval_definition_no_data_arrays':equation,
          'local_figure_identity_verification':'All7 standalone plot PDF decoded streams match compiled PDF; local source labels V4 and shot-block labels are present. Their absence in earlier API images is intentional identifier redaction, not missing labels in final HTML.',
          'privacy_scope':'Individual-observation traces were never supplied to these calls. Local artifact retains original7figures. Final graph6 horizontal bar is cluster-bootstrap interval of equivalent spacing, not a vertical RMSE interval.',
          'html_validation':read(ROOT/'docs/paper/research_reader/qa_status.json')}
        call('03_opus_corrected_evidence_closeout','claude-opus-5-5',[
            block('Narrow independent final check of corrections and exact figure caption semantics. This is NOT whole-paper grading. Validate EPED joint stability constraints, random-only median comparison qualifier, and caption descriptions against supplied code/metadata. In particular decide what fig6 horizontal errorbar means from fig_sparse and eqrate, not prior reviewer suggestions. Missing V4/shot labels in previous external images were privacy redactions; final local labels are present, do not request changing scientific plots. No new figures or stats or physics result. Return strict JSON {scope,body_corrections_check,figure6_interval_check,caption_findings:[{figure_id,old_exact,new,reason}],remaining_material_errors:[...],overall}. At most3 indispensable corrections, <=1000visible words. DATA:\n'+json.dumps(packet,ensure_ascii=False))],10000)
        return
    if '--figures' in sys.argv:
        safe=read(OUT/'api_safe_figures.json')
        content=[]
        for row in safe['images']:
            content.extend([block(row['label']+'\n'+row['scope']),image(ROOT/row['path'])])
        content.append(block('Review the ACTUAL provided aggregate figures/schematic, captions and their manuscript passages. No individual observation traces were supplied. Figure1a,Figure3 andFigure7b are excluded from this request; do not claim to have visually checked them. We preserved original figure PDF data and render it as isolated vector SVG images in HTML, not manuscript-page screenshots. Check labels/axes/units/legends, whether conclusions match what plots show, F/L5,4designs,D650vsS,fastGPvsTCN distinction, panel organization and Korean caption sufficiency/readability. Caption should explain panels/markers/comparators, conditions, what evidence means. Do not infer significance without intervals or a causal physical mechanism. No new result, acceptance grade or broad rewrite. Return strict JSON {scope,figure_checks:[{figure_id,visual_check,caption_check,necessary_fix}],material_prose_fixes:[{section_id,old_exact,new,reason}],layout_fixes:[...],unverified_figures:[...],overall}. At most6 essential fixes, <=1500visible words. DATA:\n'+json.dumps({'manifest':read(ROOT/'docs/paper/research_reader/longform/exhibits.json'),
            'relevant_manuscript_sections':[s for s in manuscript['sections']if s['id'] in ['materials-methods','results','discussion','conclusion']]},ensure_ascii=False)))
        call('02_fable_aggregate_figure_review','claude-fable-5-2',content,16000)
        return
    tex=(ROOT/'docs/paper/ppcf_revision/main_ppcf.tex').read_text(encoding='utf8')
    tex=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}','',tex,flags=re.S)
    tex=tex[tex.index('\\section{Materials and methods}'):tex.index('\\bibliographystyle')]
    tex=re.sub(r'(?i)shots?\s+[0-9][0-9,\\\-– ]*','discharges [IDs redacted]',tex)
    audits=[]
    keys=read(ROOT/'docs/paper/research_reader/longform/integration_status.json')['citation_keys']
    for key in keys:
        folder=ROOT/'docs/paper/introduction_review'
        paths=[folder/('paper_'+key+'.json'),folder/('background_'+key+'.json')]
        audits.append(read(next(p for p in paths if p.exists())))
    content=[document(ROOT/'data/literature/Lee2026NeuralNetworkbased.pdf','NN-CES full original published PDF'),
             document(ROOT/'data/literature/Shousha2023MachineLearningbased.pdf','RTCAKENN full original published PDF'),
             document(ROOT/'data/literature/Snyder2011FirstPrinciples.pdf','EPED original published PDF')]
    content.append(block('Fresh adversarial verification of actual Korean thesis manuscript. No previous reviews are supplied. Read the complete manuscript and supplied original public PDFs; other cited papers have source-specific prior original-reading audits, which are fallible evidence, not fresh original PDFs. Check every section for factual contradictions, overclaiming, target/information-time errors, false novelty/absence, incorrect priorwork distinction, dimensional math errors, and actual-vs-planned experiments. Crosscheck own-study result directions against original anonymous English manuscript. All4validation designs equal;Fcurrent report,L5median measured next5GRIDrows>=3; both targets notinputs; currentfastsamples10msgrid uncalibrated, no actualarrivalmeasurements, no independentTi truth. No NN-CEScoupling, physicalsubmsrestoration or futurephysicalforecast done. Keep familiar model and fixed author framing. No new fits/GPU or new source assertions. Do NOT demand another storyline or acceptance grade. Do not mistake missing detail in an audit summary for absence from original source: EPED original PDF explicitly gives PB/KBMconstraints. Numeric/result verification is also performed by host; your prose is not verified fact. Preserve substantial long-form writing; identify only material errors, not stylistic preference. Return strict JSON {scope,verdict,critical_findings:[{section_id,old_exact,new,why,evidence_locator}],checked_result_claims:[{claim,verdict,evidence_locator,qualification}],checked_prior_work:[{source,claim,verdict,original_pdf_locator_or_audit_scope}],math_unit_checks:[...],remaining_unverified:[...]}. At most10 indispensable findings, exact oldtext must occur once in a content_markdown. Do not quote source text >15words/paper. Complete response in <=1800visible words; prioritize actual errors. DATA:\n'+json.dumps({'actual_manuscript':manuscript,'original_anonymous_methods_results_discussion_conclusion_tex':tex,'source_audits':audits},ensure_ascii=False)))
    call('01_opus_independent_source_verification','claude-opus-5-5',content,14000)

if __name__=='__main__':main()
