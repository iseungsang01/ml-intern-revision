"""Independent source/code/UI closeout; does not request paid review or fit models."""
import hashlib
import json
import subprocess

from build_reader import ROOT,load_sources,quantitative_tokens

OUT=ROOT/'docs/paper/research_reader/gp_background'

def main():
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    save=lambda p,d:p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    draft=read(OUT/'section.json');sources,aliases=load_sources()
    nodes=[draft]+draft['subsections']
    paragraphs=[p for n in nodes for p in n['content_markdown'].split('\n\n') if not p.startswith('|')]
    assert max(map(lambda p:len(quantitative_tokens(p)),paragraphs))<=4
    import re
    text='\n'.join(n['content_markdown']for n in nodes)
    cites=set(re.findall(r'\[@([^\]]+)\]',text));assert cites<=set(aliases)
    assert len(draft['subsections'])==8
    assert text.count('$$')==18
    for fragment in [r'\sigma_n^2 I',r'v_*+\sigma_n^2',r'\sqrt 3',r'\log|C|',r'r_t=y_t-g_t']:
        assert fragment in text,fragment
    immutable=['docs/paper/ppcf/main_ppcf.tex','docs/paper/ppcf/supp_ppcf.tex',
       'docs/paper/ppcf_revision/main_ppcf.pdf','ces_prediction/baselines_interpolation.py',
       'ces_prediction/experiments/b73_targets/common.py','docs/paper/b105_merit.json']
    for relative in immutable:
        previous=subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT)
        assert previous.replace(b'\r\n',b'\n')==(ROOT/relative).read_bytes().replace(b'\r\n',b'\n'),relative
    preflight=read(OUT/'api_preflight_status.json')
    assert preflight['reason_category']=='insufficient_credit'
    save(OUT/'review_attempt.json',{'completed_claude_review':False,'paid_message_sent':False,
      'attempts':[{'endpoint':'/messages/count_tokens','purpose':'full draft preflight','result':'HTTPError'},
                  {'endpoint':'/messages/count_tokens','purpose':'minimal diagnostic','http_status':400,'error_type':'invalid_request_error'},
                  {'endpoint':'/messages/count_tokens','purpose':'redacted reason classification','http_status':400,'reason':'insufficient_credit'}],
      'secrets_or_error_body_saved':False,'new_paid_cost':0,
      'draft_and_source_metadata_prepared_for_review':True,'no_automatic_paid_retry':True})
    save(OUT/'verification.json',{'section_characters':sum(len(n['content_markdown'])for n in nodes),
      'subsections':len(draft['subsections']),'display_equations':text.count('$$')//2,'comparison_tables':1,
      'citations':sorted(cites),'manuscript_source_records':len(sources),
      'source_scope':'GPML chapters2/4/5, Williams original section2, Roberts sections3/4; existing local Chilenski and Michoski reading records. Friedman original fresh access failed; gradient formulation independently crosschecked against official scikit-learn mathematical documentation.',
      'boosting_crosscheck':'https://scikit-learn.org/stable/modules/ensemble.html#mathematical-formulation',
      'independent_checks':['Posterior mean/covariance and LML against primary equations; noise variance vs latent variance distinguished.',
        'Matérn3/2 form and mean-square differentiability checked against GPML chapter4.',
        'Existing code unit standardized amplitude, fixed length/noise grid, LML, Cholesky and mean-only return verified.',
        'T0/TX input division crosschecked with current manuscript methods; no causal physics claim from residual association.'],
      'unchanged_scientific_sources':immutable,'ui_checks':read(ROOT/'docs/paper/research_reader/qa_status.json'),
      'reader_sha256':hashlib.sha256((ROOT/'docs/paper/research_reader/index.html').read_bytes()).hexdigest(),
      'claude_review_completed':False,'new_fits':0,'new_evaluations':0,'gpu_runs':0})
    print('GP_BACKGROUND_CLOSEOUT_PASSED',len(nodes),len(cites))

if __name__=='__main__':main()
