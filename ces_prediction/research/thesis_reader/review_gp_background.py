"""Primary-source anchored GP theory review; no data or fits sent/run."""
import json

import verify_with_claude as V

OUT=V.ROOT/'docs/paper/research_reader/gp_background'

def main():
    definitions=[
      ('Rasmussen2006GPML','Rasmussen, C. E.; Williams, C. K. I.','2006','Gaussian Processes for Machine Learning','MIT Press','','','',
       'https://gaussianprocess.org/gpml/',
       'Author-hosted chapters2,4,5: conditional Gaussian regression Eq2.22-2.24; Matérn Eq4.17; marginal likelihood Eq5.8. Relevant passages checked, not full-volume reread.'),
      ('Williams1996GaussianRegression','Williams, C. K. I.; Rasmussen, C. E.','1996','Gaussian Processes for Regression','Advances in Neural Information Processing Systems','8','514-520','',
       'https://papers.neurips.cc/paper/1048-gaussian-processes-for-regression.pdf',
       'Original paper pp514-516, Section2 Eqs1-2: GP definition, conditional mean/variance, covariance/noise, hyperparameters and dense complexity. Conference1995, proceedings1996.'),
      ('Roberts2013GPTimeSeries','Roberts, S.; Osborne, M.; Ebden, M.; Reece, S.; Gibson, N.; Aigrain, S.','2013','Gaussian processes for time-series modelling','Philosophical Transactions of the Royal Society A','371','20110550','10.1098/rsta.2011.0550',
       'https://www.robots.ox.ac.uk/~sjrob/Pubs/Phil.%20Trans.%20R.%20Soc.%20A-2013-Roberts-.pdf',
       'Author-hosted full PDF; Sections3-4: function prior, time covariance design and conditional prediction. Relevant source passages checked, not figures/new performance claims.'),
      ('Friedman2001GreedyFunction','Friedman, J. H.','2001','Greedy function approximation: A gradient boosting machine','The Annals of Statistics','29','1189-1232','10.1214/aos/1013203451',
       'https://statweb.stanford.edu/~jhf/ftp/trebst.pdf',
       '기존 BibTeX 서지를 연결했다. 이번 요청에서 출판사 원문은 보안 화면, 저자 PDF는 연결 실패로 새 본문 확인을 완료하지 못했다. 함수 공간 경사 하강·잔차학습의 인용이며 CES 성능 근거가 아니다.')]
    sources=[]
    for key,authors,year,title,venue,volume,pages,doi,url,scope in definitions:
        citation=dict(title=title,authors=authors,year=year,venue=venue,volume=volume,pages=pages,doi_or_arxiv=doi,url=url)
        sources.append(dict(key=key,title=title,authors=authors,year=year,pages=None,
           mode='summary',mode_label='기존 서지 · 접근 범위 명시' if key.startswith('Friedman') else '관련 이론 절 확인',doi=doi,path=url,url=url,
           relevance='GP 이론과 잔차학습의 근거',review_file='gp_background/sources.json',
           review={'citation':citation,'validation':scope,
             'claim_checks':[{'proposed_claim':'GP 이론·구현 연결의 근거','source_finding':scope,'locator':url}],
             'forbidden_inferences':'이 문헌이 본 연구의 CES 정확도·물리적 복원·신규성을 입증한 것으로 쓰지 않는다.'}))
    V.save(OUT/'sources.json',{'scope':'New targeted primary-source checks, separate from historical whole-corpus API reading provenance.','sources':sources})
    draft=V.read(OUT/'section.json')
    baseline=(V.ROOT/'ces_prediction/baselines_interpolation.py').read_text(encoding='utf8')
    start=baseline.index('def _gp_fit_predict(')
    code=baseline[start:baseline.index('def predict_gp_causal(',start)]
    # Published public-source paragraphs + self-written theory + implementation, no observations.
    packet={'draft':draft,'source_checks':sources,
      'existing_fusion_source':V.read(V.ROOT/'docs/paper/introduction_review/background_michoski2024.json'),
      'existing_profile_source':V.read(V.ROOT/'docs/paper/introduction_review/paper_chilenski2015.json'),
      'actual_GP_code_no_observations':code,
      'existing_method_scope':'Past visibleCES only, local standardized Matérn3/2, unit signal variance in standardized space, fixed grid length/noise LML, posterior MEAN only. Causal vs offline differ visible set, TX=GP+boosted regression on CEShistory+BES/ECEIscalars,T0=historyfeaturesonly. No final probability calibration or GPvariance output currently. Scientific code/results unchanged.'}
    task='''Review detailed Korean thesis theoretical-background addition on Gaussian process regression. Check all math/dimensions, latent vs reported value uncertainty, Matérn3/2 mean-square differentiability vs path regularity, empirical-Bayes/grid selection, actual code, causality as information eligibility not GPname, extrapolation to local mean, GP+boosting vs CES-only distinction. No new novelty/results, no arbitrary L5theory. Preserve detailed pedagogical writing. Point out only material corrections. Do not assume unpublished data, final calibratedvariance or independenttruth. Public-source scope is explicit; source citations are not new whole-corpus reviews. Return strict JSON {verdict,corrections:[{subsection_title,old_exact,new,reason}],equation_checks,source_checks,remaining_limitations}. <=1400visible words, max8corrections. Exact text replacement must occur once. No final acceptance grade. DATA:\n'''
    if '--sources-only' in __import__('sys').argv:
        print('GP_SOURCE_RECORDS_UPDATED_WITHOUT_API')
        return
    try:
        V.call('08_fable_GP_background_review','claude-fable-5-2',[V.block(task+json.dumps(packet,ensure_ascii=False))],12000,output_dir=OUT)
    except RuntimeError:
        V.save(OUT/'review_attempt.json',{'stage':'preflight token count','completed_claude_review':False,
            'paid_message_sent':(V.CACHE/'08_fable_GP_background_review.request.json').exists(),
            'status':'API request failed; see api_preflight_status.json. Do not claim review or retry paid request blindly.'})
        raise

if __name__=='__main__':main()
