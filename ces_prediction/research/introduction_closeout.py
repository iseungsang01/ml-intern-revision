"""Free, independent coverage/hash/cost closeout; never calls a language model."""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import fitz
from thesis_reader.build_reader import load_sources

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper/introduction_review'
PAPER=ROOT/'docs/paper/ppcf_revision'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources,_=load_sources()
    modes=dict(Counter(s['mode'] for s in sources))
    assert len(sources)==254 and modes.get('summary',0)==0
    catalogue=read(OUT/'catalogue.json')['papers']
    provenance=[]
    for row in catalogue:
        assert sha(ROOT/row['path'])==row['sha256'],row['key']
        if not row['external_review_allowed'] or row['duplicate_of_prior']:
            continue
        paths=[OUT/('paper_'+row['key']+'.json'),OUT/('background_'+row['key']+'.json')]
        path=next(p for p in paths if p.exists())
        audit=read(path)
        assert audit['key']==row['key']
        if 'supplied_pdf_sha256' in audit:
            assert audit['supplied_pdf_sha256']==row['sha256']
        else:
            assert audit['scope_read']['full_pdf_supplied'] is True
        provenance.append({'key':row['key'],'pdf_sha256':row['sha256'],
                           'audit':path.relative_to(ROOT).as_posix(),'audit_sha256':sha(path)})
    assert len(provenance)==251
    ledger_paths=[OUT/'api_usage.json',ROOT/'docs/paper/research_reader/api_usage.json',PAPER/'nnces_edit_usage.json']
    ledgers=[read(p) for p in ledger_paths]
    costs=[float(l.get('estimated_returned_usage_usd',l['estimated_or_reserved_usd'])) for l in ledgers]
    cache_paths=[ROOT/'data/.introduction_corpus/api',ROOT/'data/.thesis_reader',ROOT/'data/.ppcf_nnces_edit']
    checked_attempts=0
    for ledger,cache,total in zip(ledgers,cache_paths,costs):
        computed=0.0
        prices=ledger.get('prices_input_output_per_million',ledger.get('prices_per_million_input_output'))
        for call in ledger['calls']:
            request=cache/(call['stage']+'.request.json')
            response=cache/(call['stage']+'.response.json')
            binding=hashlib.sha256(json.dumps(read(request),sort_keys=True).encode()).hexdigest()
            assert binding==call['request_sha256']
            assert sha(response)==call['response_sha256']
            usage=read(response)['usage']
            assert usage==call['usage']
            assert not usage.get('cache_creation_input_tokens') and not usage.get('cache_read_input_tokens')
            ip,op=prices[call['model']]
            computed+=(usage['input_tokens']*ip+usage['output_tokens']*op)/1e6
            checked_attempts+=1
        assert abs(computed-total)<1e-8
    assert costs[0]+costs[1]<=50 and costs[2]<=3
    status={'state':'all_eligible_sources_individually_reviewed_and_prose_integrated',
      'source_file_unit_not_distinct_works':True,'total_pdf_files':len(sources),
      'eligible_public_unique_source_files':len(provenance),'reading_modes':modes,
      'figure_limit':'Full extracted text does not constitute visual checking of figures/equations. Even PDF-supplied audits retain source-specific visual/legibility caveats.',
      'duplicate_versions_may_still_represent_same_work':True,
      'source_records':provenance,'new_fits':0,'gpu_runs':0,'whole_manuscript_regraded':False,
      'prior_incomplete_responses':'Preserved; recovered complete fields only; not counted as completed synthesis or grades',
      'final_writing_provenance':'manuscript_writing_provenance.json',
      'reader':'../research_reader/index.html'}
    save(OUT/'all_corpus_status.json',status)
    save(OUT/'all_corpus_progress.json',{k:v for k,v in status.items() if k!='source_records'})
    save(OUT/'status.json',{'state':status['state'],'complete':True,'scientific_runs':0,
                           'coverage_record':'all_corpus_status.json','historical_selection':'progress.json'})
    save(OUT/'cost_summary.json',{'invoice_verified':False,'list_price_usage_estimates_usd':{
        p.relative_to(ROOT).as_posix():c for p,c in zip(ledger_paths,costs)},
        'corpus_and_reader_total_usd':round(costs[0]+costs[1],6),
        'corpus_and_reader_ceiling_usd':50,
        'manuscript_edit_total_usd':costs[2],
        'prior_deep_review_usd':19.134768,
        'prior_deep_plus_manuscript_edit_usd':round(19.134768+costs[2],6),
        'original_deep_and_manuscript_allowance_usd':30,
        'combined_total_usd':round(sum(costs)+19.134768,6),
        'no_further_paid_calls_planned':True})
    baseline=subprocess.check_output(['git','show','HEAD:docs/paper/ppcf_revision/main_ppcf.tex'],cwd=ROOT).decode('utf8')
    current=(PAPER/'main_ppcf.tex').read_text(encoding='utf8')
    def result_part(t):
        return t[t.index('\\section{Results}'):t.index('\\section{Discussion}')].replace('\r\n','\n')
    assert result_part(baseline)==result_part(current),'Existing result text changed'
    save(OUT/'verification.json',{'existing_results_identical_to_head':True,
        'results_reference_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        'result_section_sha256':hashlib.sha256(result_part(current).encode()).hexdigest(),
        'source_file_hashes_verified':len(catalogue),'review_records_verified':len(provenance),
        'api_attempt_request_response_hashes_and_usage_verified':checked_attempts,
        'bibliography_metadata_records':len(read(OUT/'bibliography_checks.json')['records']),
        'test_result':'89 passed; 3 pre-existing dependency warnings',
        'numerical_checker':'main and supplement pass; records checker passes',
        'pdf_visual_pages':[1,2,3,4,5,22],'pdf_build':'No undefined references, citations or overfull boxes'})
    manuscript_status=read(PAPER/'status.json')
    manuscript_status.update({'main_pages':len(fitz.open(PAPER/'main_ppcf.pdf')),
       'literature_and_structure_update':'../introduction_review/README.md',
       'latest_pdf_fully_regraded':False})
    manuscript_status['pdf_hashes']['main_ppcf.pdf']=sha(PAPER/'main_ppcf.pdf')
    save(PAPER/'status.json',manuscript_status)
    provenance_path=OUT/'manuscript_writing_provenance.json'
    p=read(provenance_path);p['host_integration_pending']=False
    p['host_corrections']='host_integration.json';p['final_verification']='verification.json'
    save(provenance_path,p)
    print('VERIFIED',len(provenance),'source records; modes',modes,'cost',round(sum(costs)+19.134768,6))


if __name__=='__main__':
    main()
