"""Author-authorized bounded PPCF writing/novelty loop; no scientific fits."""
import base64,json,re,shutil,subprocess,sys
from pathlib import Path
import icml_revision as W
E=W.E
ROOT=W.ROOT
OUT=ROOT/'docs/paper/ppcf_revision'
CACHE=ROOT/'data/.ppcf_revision'

def build():
    for name in ['supp_ppcf','main_ppcf','supp_ppcf','main_ppcf']:
        p=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=OUT,capture_output=True,timeout=180)
        (OUT/(name+'_build.txt')).write_bytes(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError('Compile failed: '+name)

def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    W.CACHE=CACHE
    if (OUT/'status.json').exists():
        status=json.loads((OUT/'status.json').read_text(encoding='utf8'))
        if status.get('requested_threshold_met_by_novelty_referee'):
            print('PPCF novelty threshold already reached; preserved reviews, no paid replay.')
            return
    r,b=E.gate();assert r
    packet=E.packet(r,b)
    compact={'context':packet['context'],'paired':packet['b94']['paired_by_design'],'budgets':packet['b94']['budget_summary'],'latency':packet['b94']['latency']}
    W.save(OUT/'evidence_binding.json',b)
    src=ROOT/'docs/paper/ppcf'
    if not (OUT/'initial_copy.json').exists():
        for p in src.iterdir():
            if p.is_file() and p.suffix in ['.tex','.bib','.pdf','.sty','.cls']:
                shutil.copy2(p,OUT/p.name)
        W.save(OUT/'initial_copy.json',{'source':'docs/paper/ppcf','source_hashes':{p.name:E.sha(p) for p in src.glob('*.tex')}})
    # Numbers/tables always generated directly from source JSON.
    prior=W.OUT;W.OUT=OUT;W.generate_tables(r);W.OUT=prior
    tables=(OUT/'generated_tables.tex').read_text()
    original=(OUT/'main_ppcf.tex').read_text(encoding='utf8')
    # Only redacted manuscript text is sent for editorial review.
    redacted=re.sub(r'\\url\{https://github.com/[^}]+\}',r'\\textit{anonymized repository}',original)
    redacted=re.sub(r'\\author\{.*?\}',r'\\author{Anonymous}',redacted,flags=re.S)
    task='''You are a scientific editor revising an existing PPCF article. Preserve the author framing: pedestal-top CES causal nowcasting, best estimator against interpolation/hold/GP, fast diagnostic increment, four validation designs equally,10-ms computation, applications then limits. Latest independently audited selected TCN results below must be added without replacing the established GP/tree estimator: neural TX/TH are distinct from tree TX/T0. FULL15 one seed vs fixed FOUR three-seed subset are separate. D delays are history availability, not forecast horizon. No independent campaign or true-temperature truth. Fixed fits, no refit/selection/multiplicity uncertainty. IMPORTANT NOVELTY: explicitly distinguish established GP/boosting/TCN methods, application-specific empirical contribution, and unverified originality. Do not call standard components novel or establish priority from search absence. Contrast raw-spectrum NN-CES (Lee et al2026 doi10.1016/j.fusengdes.2025.115518), UFCES hardware(Jang et al2021 doi10.1063/5.0043824), virtual diagnostics, and general missing/delayed sensor estimation. Preserve adverse/latest outcomes. No new experiments. Use only supplied/verified references; no invented citations. Return strict JSON {tex:complete revised main_ppcf.tex,plan_markdown:Korean ranked next experiments and writing work with novelty gap, overlap, falsifier, GPU needed, preregistration and both-outcome wording,changes:summary}. All new performance numbers ONLY via input{generated_tables}; never manually type values in prose. Include generated_tables.tex once near new budgeted TCN results. Existing gen_v2 macros must remain. Keep existing references, graphics, crossreferences compiling. Do not transmit identity or local paths. Do not claim TCN wins or adopt it globally. Do not create new physics framing. Old paper prose can be edited for consistency and concision. Preserve main/sub supplement references. Formal selector proof and internal VAL grey-box are narrow supporting evidence only, not inflated novelty.'''
    prompt=task+'\nAUDITED EVIDENCE\n'+json.dumps(compact)+'\nGENERATED TABLES\n'+tables+'\nCURRENT PAPER\n'+redacted
    cached=CACHE/'draft_response.json'
    if cached.exists() and json.loads(cached.read_text(encoding='utf8')).get('stop_reason')!='end_turn':
        raise RuntimeError('Incomplete draft preserved; use the recovered reviewed manuscript, not an automatic replay.')
    draft=W.obj(W.call('draft',{'model':'claude-fable-5-2','max_tokens':24000,'output_config':{'effort':'low'},'messages':[{'role':'user','content':prompt}]}))
    history=[]
    for i in range(1,5):
        (OUT/'main_ppcf.tex').write_text(draft['tex'],encoding='utf8')
        p=OUT/'main_ppcf.tex'
        s=p.read_text(encoding='utf8').replace('Seungsang Lee','Anonymous').replace('iseungsang01','anonymous')
        p.write_text(s,encoding='utf8')
        (OUT/'NEXT_EXPERIMENTS.md').write_text(draft['plan_markdown'],encoding='utf8')
        build()
        for name in ['main_ppcf','supp_ppcf']:
            shutil.copy2(OUT/(name+'.pdf'),OUT/(name+f'_round{i}.pdf'))
        fixed=(ROOT/'docs/paper/review_loop/GRADER_PROMPT.md').read_text(encoding='utf8').split('```')[1]
        # Frozen fixed prompt; actual attached PDFs substitute its file placeholders.
        fixed=fixed.replace('<path to manuscript.pdf>','attached main manuscript PDF').replace('<path to supplement.pdf>','attached supplementary PDF').replace('(<n> pages)','(read all pages)').replace('(<m> pages)','(read all pages)')
        content=[]
        for name in ['main_ppcf','supp_ppcf']:
            content.append({'type':'document','title':name,'source':{'type':'base64','media_type':'application/pdf','data':base64.b64encode((OUT/(name+'.pdf')).read_bytes()).decode()}})
        content.append({'type':'text','text':fixed})
        report=W.answer(W.call(f'grade{i}',{'model':'claude-opus-5-5','max_tokens':9000,'output_config':{'effort':'low'},'messages':[{'role':'user','content':content}]}))
        (OUT/f'grade{i}.md').write_text(report,encoding='utf8')
        # Separate novelty referee receives only the same frozen PDFs and no previous reviews.
        nov=W.answer(W.call(f'novelty{i}',{'model':'claude-opus-5-5','max_tokens':6500,'output_config':{'effort':'low'},'tools':[{'type':'web_search_20250305','name':'web_search','max_uses':3}],'messages':[{'role':'user','content':content[:-1]+[{'type':'text','text':'Independent PPCF novelty referee. Read both attached PDFs. Search at most3times primary sources for closest prior work on CES raw-spectrum NN analysis, virtual Ti diagnostics, missing/delayed sensor nowcasting, and fast diagnostic fusion. No author expectations or earlier grades supplied. Give cited closest works and precise overlap, supported distinct contribution, missing comparison/data versus rewrite-only fixes, skeptical recommendation accept / weak accept / borderline / weak reject / reject / strong reject. This is simulated feedback, not actual acceptance. Never manufacture originality to achieve a target; negative evidence is not a new method. Return strict JSON {recommendation,report_markdown,needs_new_evidence,priority_changes}. No acceptance percentages.'}]}]}))
        n=json.loads(nov.strip().removeprefix('```json').removesuffix('```').strip())
        W.save(OUT/f'novelty{i}.json',n);history.append(n['recommendation'])
        W.save(OUT/'status.json',{'round':i,'novelty_recommendation':n['recommendation'],'target_met':n['recommendation'] in ['accept','weak accept','weak reject'],'history':history,'gpu_run':False})
        print('PPCF ROUND',i,n['recommendation'],flush=True)
        if n['recommendation'] in ['accept','weak accept','weak reject'] or i==4:break
        draft=W.obj(W.call(f'revise{i}',{'model':'claude-fable-5-2','max_tokens':24000,'output_config':{'effort':'low'},'messages':[{'role':'user','content':task+'\nCURRENT PAPER\n'+draft['tex']+'\nFIXED REFEREE REPORT\n'+report+'\nINDEPENDENT NOVELTY REPORT\n'+nov+'\nAUDITED EVIDENCE\n'+json.dumps(compact)+'\nGENERATED TABLES\n'+tables}]}))
    W.save(OUT/'api_usage.json',[{'call':p.stem,'model':(x:=json.loads(p.read_text(encoding='utf8'))).get('model'),'usage':x.get('usage'),'stop_reason':x.get('stop_reason')} for p in CACHE.glob('*_response.json')])

if __name__=='__main__':main()
