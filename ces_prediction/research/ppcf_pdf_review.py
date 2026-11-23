"""Independent frozen-PDF feedback for recovered PPCF draft."""
import base64,concurrent.futures,json,sys
import ppcf_revision as P
W=P.W;ROOT=P.ROOT;OUT=P.OUT;W.CACHE=P.CACHE

def review(name,prompt,search=False):
    content=[]
    for pdf in ['main_ppcf','supp_ppcf']:
        content.append({'type':'document','title':pdf,'source':{'type':'base64','media_type':'application/pdf','data':base64.b64encode((OUT/(pdf+'.pdf')).read_bytes()).decode()}})
    content.append({'type':'text','text':prompt})
    payload={'model':'claude-opus-5-5','max_tokens':6500,'output_config':{'effort':'low'},'messages':[{'role':'user','content':content}]}
    if search:payload['tools']=[{'type':'web_search_20250305','name':'web_search','max_uses':3}]
    r=W.call(name,payload);text=W.answer(r);(OUT/(name+'.md')).write_text(text,encoding='utf8');print(name,'complete',flush=True)

def main():
    suffix=sys.argv[1] if len(sys.argv)>1 else '1'
    fixed=(ROOT/'docs/paper/review_loop/GRADER_PROMPT.md').read_text(encoding='utf8').split('```')[1]
    fixed=fixed.replace('<path to manuscript.pdf>','attached main manuscript PDF').replace('<path to supplement.pdf>','attached supplementary PDF').replace('(<n> pages)','(17 pages)').replace('(<m> pages)','(26 pages)')
    novelty='''You are an independent experienced PPCF novelty referee. Read ONLY these two frozen PDFs. No author opinion or previous grade is supplied. Search at most3times PRIMARY sources for closest prior work on CES raw-spectrum neural analysis, Ti virtual diagnostics, delayed/missing-sensor nowcasting and diagnostic fusion. Do not infer novelty from unsuccessful search. Identify precisely established components, application-specific evidence and genuinely supported distinct contribution. Cite exact sections/tables and supporting literature URLs. Distinguish rewrite-only issues from necessary new analysis, new fits/GPU, and new measurements. Preserve author framing pedestal-top nowcasting versus interpolation and other estimators, fast diagnostic information, four validation designs and runtime. Do not demand a new framing. Assess newest TCN evidence separately from GP/tree evidence and distinguish one-seed full scope from three-seed subset. Recommend accept, weak accept, borderline, weak reject, reject or strong reject; never inflate to achieve a target. Give up to3 ranked changes with falsifiers and an honest statement of whether rewriting alone could get to weak reject or better. This is simulated feedback, not a conference/journal decision. No acceptance percentage; about900 words.'''
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(review,'grade'+suffix,fixed),pool.submit(review,'novelty'+suffix,novelty,True)]
        for j in jobs:j.result()
    W.save(OUT/'api_usage.json',[{'call':p.stem,'model':(x:=json.loads(p.read_text(encoding='utf8'))).get('model'),'usage':x.get('usage'),'stop_reason':x.get('stop_reason')} for p in P.CACHE.glob('*_response.json')])

if __name__=='__main__':main()
