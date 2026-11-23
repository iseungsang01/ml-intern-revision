"""Bounded author-authorized ICML writing loop; no fitting or GPU execution."""
import base64, hashlib, io, json, shutil, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper/icml'
CACHE=ROOT/'data/.icml_revision'
sys.path.insert(0,str(Path(__file__).parent/'posteval_claude'))
import evidence as E
API=E.API

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')

def call(name,payload):
    request=CACHE/(name+'_request.json'); response=CACHE/(name+'_response.json')
    if response.exists():
        assert json.loads(request.read_text(encoding='utf8'))==payload
        return json.loads(response.read_text(encoding='utf8'))
    if request.exists(): raise RuntimeError('Uncertain paid attempt: '+name)
    save(request,payload)
    try:r=API.request('/messages',API.credential(),payload)
    except Exception as e:
        save(CACHE/(name+'_error.json'),{'error':str(e)});raise
    save(response,r)
    if r.get('stop_reason')!='end_turn':raise RuntimeError('Incomplete response preserved: '+name)
    return r

def answer(r):return '\n'.join(b['text'] for b in r['content'] if b['type']=='text')
def obj(r):
    t=answer(r).strip()
    if t.startswith('```'):t=t.split('\n',1)[1].rsplit('```',1)[0]
    return json.loads(t)

def build():
    for cmd in [['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']]:
        r=subprocess.run(cmd,cwd=OUT,capture_output=True,timeout=150)
        (OUT/'build.log.txt').write_bytes(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('LaTeX compile failed; inspect build.log.txt')

def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    result,binding=E.gate();assert result
    packet=E.packet(result,binding)
    # Use only the already whitelisted anonymous packet; avoid complete repository files.
    def compact(x):
        if isinstance(x,dict):return {k:compact(v) for k,v in x.items() if k not in ['per_init','discharge']}
        if isinstance(x,list):return [compact(v) for v in x]
        return x
    evidence=compact(packet)
    save(OUT/'evidence_binding.json',binding)
    save(CACHE/'evidence.json',evidence)
    if not (OUT/'icml2026.sty').exists():
        url='https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip'
        blob=urllib.request.urlopen(url,timeout=30).read()
        z=zipfile.ZipFile(io.BytesIO(blob))
        for n in z.namelist():
            if Path(n).name in ['icml2026.sty','icml2026.bst','fancyhdr.sty']:
                (OUT/Path(n).name).write_bytes(z.read(n))
        save(OUT/'template_source.json',{'url':url,'sha256':hashlib.sha256(blob).hexdigest()})
    protocol='''Write an anonymous English ICML-format research paper about causal nowcasting of CES reported ion temperature at the KSTAR pedestal top. Preserve the author's five questions: best estimator vs hold/GP/interpolation; incremental fast diagnostic information; all four validation designs and 10-ms compute; applications; limits. Do not invent novelty, results, references, architectures, equations not supported by evidence, or untouched confirmation. Never call availability age a forecast horizon. F is current report, L5 is offline median of next five nominal frames with at least three reports. Neural TX/TH are selected causal TCN fast/history twins, not GP/tree TX. FULL15 one seed and FOUR subset three seeds must remain separate. Preserve adverse findings, fixed-budget confounding, same-row support caveats, sensor-derived targets and exploratory reused campaign. Do not claim superiority from marginal intervals. Every manuscript performance number MUST come exclusively from generated_tables.tex, which you include verbatim using input; write no performance numbers by hand. Do not put internal batch IDs in scientific prose. No new data, CPU fits or GPU work. Review recommendations for experiments are proposals only and TEST-informed exploratory. Use official icml2026 submission style, anonymous authors, letter paper, main text at most eight pages then references/appendix in same PDF. Include impact statement and reproducibility limitations. Cite only verified references listed below, using a thebibliography environment with author-year bibitems compatible with natbib. Return strict JSON with tex (complete compilable main.tex), plan_markdown (ranked rewriting/CPU/GPU/new-data needs, preregistration requirement, falsifiers, both-outcome wording), and changes. No markdown fences. Do not send identity or repository paths. Aim for a substantive six-to-eight-page main text, not a short outline.'''
    refs='''Verified references: Ko et al. (2010), Charge exchange spectroscopy system calibration for ion temperature measurement in KSTAR, RSI81 10D740, doi10.1063/1.3496991; Yun et al.(2014), Quasi3D ECE imaging system for study of MHD instabilities in KSTAR, RSI85 11D820, doi10.1063/1.4890401; Jang et al.(2021), Development of an ultrafast charge exchange spectroscopy system on the KSTAR tokamak, RSI92 053525, doi10.1063/5.0043824 (10us capability was simulated design performance, not paired ground truth for this paper); Lee et al.(2026), Neural network-based analysis of charge exchange spectra in KSTAR, Fusion Engineering and Design222 115518, doi10.1016/j.fusengdes.2025.115518 (raw-spectrum analysis, different task); Bai, Kolter, Koltun(2018), An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, arXiv1803.01271; Rasmussen and Williams(2006), Gaussian Processes for Machine Learning, MIT Press; Fritsch and Carlson(1980), Monotone Piecewise Cubic Interpolation, SIAM J Numer Anal17 238-246 doi10.1137/0717021. No exhaustive novelty claim from this list.'''
    generate_tables(result)
    table=(OUT/'generated_tables.tex').read_text()
    draft=call('draft',{'model':'claude-fable-5-2','max_tokens':14000,'output_config':{'effort':'low'},'messages':[{'role':'user','content':protocol+'\n'+refs+'\nGENERATED TABLES\n'+table+'\nANONYMOUS AUDITED EVIDENCE\n'+json.dumps(evidence)}]})
    manuscript=obj(draft)
    if (CACHE/'STOP_AFTER_DRAFT').exists():
        save(CACHE/'preserved_draft.json',manuscript)
        print('Target changed to PPCF; ICML draft preserved, no ICML review.',flush=True)
        return
    reports=[]
    for i in range(1,5):
        (OUT/'main.tex').write_text(manuscript['tex'],encoding='utf8')
        (OUT/'NEXT_EXPERIMENTS.md').write_text(manuscript['plan_markdown'],encoding='utf8')
        build()
        shutil.copy2(OUT/'main.pdf',OUT/f'round{i}.pdf')
        shutil.copy2(OUT/'main.tex',OUT/f'round{i}.tex')
        review_prompt='''You are a fresh independent experienced ICML referee. Read ONLY the attached frozen manuscript PDF including appendix. No author conclusions or prior grades are supplied. Assess ICML significance, originality, methodological soundness, evidence, clarity and reproducibility. Do not inflate grades to satisfy an author target. Cite concrete sections/tables for criticisms. Distinguish writing fixes, analysis of existing data, new fits/GPU and new measurements. Return strict JSON: recommendation (one of accept, weak accept, borderline, weak reject, reject, strong reject), confidence, report_markdown, blocking_evidence, rewriting_can_resolve (boolean). No acceptance probability. This is a simulated review, not an actual conference decision.'''
        review=obj(call(f'review{i}',{'model':'claude-opus-5-5','max_tokens':6500,'output_config':{'effort':'low'},'messages':[{'role':'user','content':[{'type':'document','source':{'type':'base64','media_type':'application/pdf','data':base64.b64encode((OUT/'main.pdf').read_bytes()).decode()}},{'type':'text','text':review_prompt}]}]}))
        save(OUT/f'review{i}.json',review);(OUT/f'review{i}.md').write_text(review['report_markdown'],encoding='utf8');reports.append(review)
        save(OUT/'status.json',{'round':i,'recommendation':review['recommendation'],'complete':True,'target_met':review['recommendation'] in ['accept','weak accept','weak reject'],'reason':'target met' if review['recommendation'] in ['accept','weak accept','weak reject'] else 'reviewed; revision or evidence needed'})
        print('ROUND',i,review['recommendation'],flush=True)
        if review['recommendation'] in ['accept','weak accept','weak reject']:break
        if i==4:break
        manuscript=obj(call(f'revision{i}',{'model':'claude-fable-5-2','max_tokens':14000,'output_config':{'effort':'low'},'messages':[{'role':'user','content':protocol+'\n'+refs+'\nCorrect the paper and update plan in response to review. Never fabricate missing evidence; explicitly leave experiments pending. Audited table source cannot change.\nTABLES\n'+table+'\nCURRENT TEX\n'+manuscript['tex']+'\nREVIEW\n'+json.dumps(review)+'\nEVIDENCE\n'+json.dumps(evidence)}]}))
    usage=[]
    for p in CACHE.glob('*_response.json'):
        x=json.loads(p.read_text(encoding='utf8'));usage.append({'call':p.stem,'model':x.get('model'),'usage':x.get('usage'),'stop_reason':x.get('stop_reason')})
    save(OUT/'api_usage.json',usage)

def generate_tables(r):
    lines=['% Generated from independently audited result JSON; never edit by hand.']
    for scope,sc in r['scopes'].items():
        lines += ['\\begin{table*}[t]','\\centering\\scriptsize',r'\begin{tabular}{llrrr}\toprule',r'Design & Availability / target & Fast TCN RMSE & History TCN RMSE & Paired skill [95\% interval]\\\midrule']
        for design,d in sc['designs'].items():
            for cell in ['D10/F','D650/L5','S/L5','K10/F']:
                c=d['cells'].get(cell)
                if not c:continue
                skill=c.get('skill')
                if skill is None:
                    # Discover only explicit paired comparison, never derive a CI from marginals.
                    matches=[v for k,v in c.items() if isinstance(v,dict) and 'point' in v and 'cluster' in v]
                    if len(matches)==1:skill=matches[0]
                tx=c['metrics']['TX']['RMSE']['point'];th=c['metrics']['TH']['RMSE']['point']
                s='not reported'
                if isinstance(skill,dict) and 'point' in skill and 'cluster' in skill:
                    lo,hi=skill['cluster'];s=f"{skill['point']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                lines.append(f'{design} & {cell} & {tx:.2f} & {th:.2f} & {s}'+r'\\')
        label='Full validation suite, one initialization' if scope=='FULL15_one_seed' else 'Fixed unit subset, three initializations'
        lines += [r'\bottomrule\end{tabular}',r'\caption{'+label+r'. RMSE in eV. Paired skill is fast relative to history; intervals condition on fixed fits. Designs are not ranked. Selected rows summarize availability conditions; scopes remain separate.}',r'\end{table*}']
    lines += [r'\begin{table}[t]\centering\small',r'\begin{tabular}{lrr}\toprule Model & Median ms & p99 ms\\\midrule']
    for model,v in r['latency'].items():
        label={'TH':'History TCN','TX':'Fast TCN'}.get(model,model)
        lines.append(f"{label} & {v['median_ms']:.3f} & {v['p99_ms']:.3f}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\caption{Single-thread streaming inference on a held-out trace. Acquisition and transfer excluded; preprocessing measured separately.}\end{table}']
    (OUT/'generated_tables.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':main()
