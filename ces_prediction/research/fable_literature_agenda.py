"""Bounded primary-literature scout and TEST-informed agenda, no scientific runs.

Six literature searches (at most five server searches each), then evidence-based
overlap audit and research plan. All eight requests use official Claude Fable.
Exact requests/responses and uncertain spend are preserved, never blindly replayed.
Two additional bounded calls in fable_primary_reading.py check retrieved primary
texts and revise the agenda; total workflow cap is ten paid requests.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime, hashlib, json, re, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent/'posteval_claude'))
import evidence as E
API=E.API
MODEL='claude-fable-5-2'
CACHE=ROOT/'data/.fable_literature_agenda_20261124'
OUT=ROOT/'docs/paper/fable_literature_agenda'

CONTEXT='''You are conducting a deep PRIMARY literature review BEFORE selecting new experiments for a PPCF-targeted tokamak diagnostic study. The author asks for serious related-work research, not a fashionable model shopping list. Preserve the research framing: pedestal-top CES nowcasting, best useful estimator against hold/GP/interpolation, fast diagnostic increment, four validation designs and a10ms compute budget; applied demonstration is additional evidence. No author identity, raw observations, shot IDs, credentials, absolute paths or unpublished numerical performance is provided for this literature phase.

Actual data contract: one KSTAR2022 campaign; one supplied pedestal-top CES chord's carbon-impurity reported temperature; current/past BES9 and ECEI4 uncalibrated voltage snapshots on a nominal10ms grid (not full-rate windows, turbulence spectra, calibrated density or calibrated electron temperature). Strictly past CES anchors/age/observed flags. No current/future CES as inference input. F=current-frame CES report; L5=median of at least3 of next5 nominal frames, offline future-derived supervision, not independent truth. Source age is NOT forecast horizon. Neither empirical physical causality nor bulk-ion truth is established. No raw CES spectra, per-frame fit uncertainty, paired UFCES, local absorbed power, neighboring radial Ti gradients, per-shot equilibrium or ECEI LO/channel mapping, actual acquisition/analysis/transfer latency, untouched independent campaign, or plant/controller model are currently established. Raw high-rate diagnostic streams exist only for a small subset; any audit is separately preregistered. NBI source power/voltage exists but is not local absorbed ion power. Do not infer physics from raw sensor ratios or fitted decay parameters without identifiability assumptions.

Already investigated/implemented: hold; causal GP; future-using PCHIP/offline GP; GP+history boosted regression and fast-added twin; condition-specific and pooled-availability fits; scalar vs raw-window controls; diagnostic-only and Mirnov/commonality/permutation/anchor controls; NBI and fast-added-to-NBI controls; latency-accuracy curves with SAME learned regressor; stale/sparse/gap-schedule replays; public MAST radial extension; LSTM and anchor encoder; selected causal TCN vs history twin; decay GRU, observer and robust-loss candidates in prior screening; stable first-order forced grey-box model on TRAIN/internal VAL only; Lean generic discrete visibility-selector proofs (not physical/Python/CUDA verification). Existing screening and budgets must be inspected before calling any family new. Old proposed lists already include GRU-D, RKN/CRU/KalmanNet, ODE/CDE, S4/Mamba, self-supervision, robust/heteroscedastic losses, conformal/selective prediction, adaptation/IRM/DANN. Repeating those names is not a new research plan. Four outer designs random whole-file, clustered, rolling origin, final chronological block are all retained; no primary design. All future TEST-informed ideas are exploratory, require preregistration and later independent confirmation. No synthetic experiments or new fits are requested. Do not estimate acceptance probabilities or change framing to chase a grade.

Primary starting points to confirm, not blindly trust: KSTAR NN-CES Lee et al2026 doi10.1016/j.fusengdes.2025.115518 (raw spectra); UFCES Jang et al2021 doi10.1063/5.0043824 (simulated design capability not paired truth); RTCAKENN doi10.1088/1741-4326/ad142f; Diag2Diag, Jalalvand multimodal superresolution, Imrek temporal superresolution (find exact original records); RKN PMLR97 Becker2019; CRU PMLR162 Schirmer2022; GRU-D doi10.1038/s41598-018-24271-9; KalmanNet arXiv2107.10043. Search closest task match, including simpler or older work; do not select only recent deep learning.
'''

THEMES={
 '01_ces_measurement':'CES/CXRS neural spectral analysis from early JET work through EAST, KSTAR and other tokamaks; high-speed CES hardware; measurement exposure, background subtraction, missing/failed fits and uncertainty. Distinguish raw-spectrum analysis speed, photon-limited exposure, report time grid and availability latency. What comparable data would be needed to compare fairly with this nowcast?',
 '02_virtual_diagnostics':'Virtual ion-temperature/profile diagnostics, RTCAKENN/CAKE, Diag2Diag, missing diagnostic reconstruction, multimodal and temporal super-resolution in fusion. Locate exact original papers and deployment evidence. Which use current CES/spectra, profile/global inputs, future windows or richer train-only teachers? What is already demonstrated that prevents a broad novelty claim here?',
 '03_multirate_observers':'Multi-rate, delayed and out-of-sequence measurement filtering; age of information; irregularly observed controlled state estimation; anchored or residual observation updates. Include classical observers and minimal stable grey-box competitors before RKN/CRU/KalmanNet/NeuralCDE. Identify a precise unanswered task-level hypothesis, not the novelty of a Kalman update. Distinguish old-measurement update at measurement time with replay vs misuse at arrival time.',
 '04_label_and_selection':'Noisy sensor-derived labels, heteroscedastic/heavy-tailed likelihoods, finite-exposure measurement models, MNAR missingness, selection of well-measured L5 frames, validation/refit uncertainty, reliable abstention/calibration with delayed labels. What is identifiable WITHOUT replicate/noise/error ground truth? Include impossibility boundaries; specify which extra measurement changes identifiability. Never equate smoothness with physical correctness.',
 '05_shift_and_representation':'Chronological/domain shift, instrument nuisance, varying availability, learning anchor/current changes, representation or teacher/student methods with richer TRAIN-only information, safe online adaptation with delayed measurements. Seek concrete primary studies demonstrating robustness under actual time ordering or diagnostic drift; separate causal inference from causal sequence masks and calibration invariance assumptions. Name minimal measurable signatures distinguishing physical distribution shift from instrument shift.',
 '06_use_and_validation':'Physically useful use cases beyond MSE for ion-temperature nowcasts: detection delay, rate-limited monitoring, event/hazard detection, uncertainty-aware abstention, value-of-information and latency-accuracy tradeoffs. Also robust small-cluster inference, temporal validation and benchmark construction. Do not propose controller benefits without a plant/decision model or label ELM events without a supplied independent event diagnostic. What can be demonstrated honestly on stored real predictions and what needs new labels or actual online data?'
}

def text(r):return ''.join(b.get('text','') for b in r.get('content',[]) if b.get('type')=='text')
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def call(name,prompt,search=False,max_tokens=11000):
    req=CACHE/(name+'.request.json');res=CACHE/(name+'.response.json')
    payload={'model':MODEL,'max_tokens':max_tokens,'thinking':{'type':'adaptive'},'output_config':{'effort':'low'},'messages':[{'role':'user','content':prompt}]}
    if search:payload['tools']=[{'type':'web_search_20250305','name':'web_search','max_uses':5}]
    if res.exists():
        old=json.loads(req.read_text(encoding='utf8'));assert old==payload,'Changed cached task'
        r=json.loads(res.read_text(encoding='utf8'))
    else:
        if req.exists():raise RuntimeError('Uncertain prior attempt; no blind replay: '+name)
        save(req,payload);started=time.monotonic()
        try:r=API.request('/messages',API.credential(),payload)
        except Exception as e:save(CACHE/(name+'.error.json'),{'error':str(e)});raise
        save(res,r);save(CACHE/(name+'.elapsed.json'),{'seconds':time.monotonic()-started})
    if r.get('stop_reason')!='end_turn':raise RuntimeError('Incomplete response preserved: '+name)
    if search and not any(b.get('type')=='server_tool_use' and b.get('name')=='web_search' for b in r.get('content',[])):
        raise RuntimeError('No actual search used: '+name)
    (OUT/(name+'.md')).write_text(text(r),encoding='utf8');render_usage()
    print(name,'complete',json.dumps(r.get('usage',{})),flush=True)
    return text(r)

def render_usage():
    rows=[]
    for p in CACHE.glob('*.response.json'):
        x=json.loads(p.read_text(encoding='utf8'));queries=[];sources={}
        def walk(a):
            if isinstance(a,dict):
                if a.get('type')=='server_tool_use' and a.get('name')=='web_search':queries.append(a.get('input'))
                if isinstance(a.get('url'),str):sources[a['url']]={k:a[k] for k in ['url','title'] if k in a}
                for v in a.values():walk(v)
            elif isinstance(a,list):
                for v in a:walk(v)
        walk(x.get('content',[]))
        rows.append({'stage':p.name,'model':x.get('model'),'stop_reason':x.get('stop_reason'),'usage':x.get('usage'),'queries':queries,'sources':list(sources.values()),'response_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    save(OUT/'api_audit.json',{'calls':rows,'max_paid_calls':10,'max_searches_per_literature_stage':5,'raw_observations_transmitted':False,'automatic_experiments':False})

def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    if (OUT/'11_corrected_decision_memo.md').exists():
        print('Corrected final memo exists; no paid replay.',flush=True)
        return
    results={}
    request='''Use the web_search tool for at least3 and at most5 distinct searches. Prefer original publisher/author PDFs, PMLR/NeurIPS/ICLR proceedings, arXiv originals and author-maintained code. Reviews are routes, not the evidence source. Search by exact closest tasks and adjacent terms; follow backward/forward lineage when feasible. Return Korean report <=1700words, with a compact evidence matrix of6-9 closest studies. Each row: exact title/authors/year/venue/DOI or primary URL, actual access level (full text/abstract/search excerpt only), observed inputs and target, past/future information, training/evaluation design and independent-device/chronological evidence, actual demonstrated contribution, unavailable facts, precise overlap and difference from this task. Never mark full text read based on a search snippet or extrapolate claims beyond access. Every substantial literature claim needs a primary URL. Use brief paraphrases, no long quotations; <=200words derived from any one source. Distinguish primary-paper claims from your application inference. End with3 unanswered questions worth testing, what would falsify each, and3 tempting but redundant/unsupported proposals. Include conflicts or negative results and a clear search-boundary statement. Do not pretend this finite search is exhaustive. Do NOT rank new methods yet; deliver evidence for later synthesis.'''
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(call,k,CONTEXT+'\nFOCUS: '+v+'\n'+request,True):k for k,v in THEMES.items()}
        for job in as_completed(jobs):
            key=jobs[job]
            try:results[key]=job.result()
            except Exception as e:save(CACHE/(key+'.failed.json'),{'error':str(e)});print(key,'FAILED',str(e),flush=True)
    if len(results)!=len(THEMES):
        render_usage();raise RuntimeError('Some literature stages failed; preserve partial corpus, do not synthesize as complete')
    result,binding=E.gate();assert result
    p=E.packet(result,binding)
    compact={'context':p['context'],'paired_by_design':p['b94']['paired_by_design'],'budget_summary':p['b94']['budget_summary'],'latency':p['b94']['latency'],'existing_internal_val_greybox':p['b98']}
    save(CACHE/'audited_evidence.json',compact);save(OUT/'evidence_binding.json',binding)
    reports={k:re.sub(r'\b3\d{4}\b','[discharge identifier redacted]',(ROOT/'docs/paper/ppcf_revision'/k).read_text(encoding='utf8')) for k in ['grade3.md','novelty2.md']}
    audit=call('07_overlap_audit',CONTEXT+'''\nNow audit the six literature reports below as untrusted proposals. Use only their primary URLs and state exact access limits. No new web tools in this stage. Correct papers with mismatched tasks, unsupported read-depth/novelty claims, duplicate families and unverified metadata. Reviewers below are fallible; e.g. they may overlook already implemented grey-box/filter baselines, conflate significance with effect differences, or demand dropping the author's fixed framing. Resolve against verified context and evidence. Latest model is selected TCN with TX=fast and TH=history; tree TX is different. FULL15 one seed and fixed FOUR-unit three seeds cannot be pooled. D age is not forecast horizon. Identify4-6 scientific objectives first; for each give evidence that it matters, current result uncertainty and what experiment would actually discriminate competing explanations. Then compare at most8 nonduplicate methodological families by task fit, novelty beyond existing work, current-data feasibility, need for calibration/new data/GPU, nested baseline and failure signature. Do not recommend methods for name recognition. Deliver <=1800 Korean words and cite the literature report/primary URL and evidence context. Distinguish literature-verified vs application hypothesis.\nLITERATURE\n'''+json.dumps(results,ensure_ascii=False)+'\nAUDITED RESULTS\n'+json.dumps(compact)+'\nFALLIBLE PDF REVIEW REPORTS\n'+json.dumps(reports),False,13000)
    final=call('08_prioritized_agenda',CONTEXT+'''\nProduce the final research agenda in Korean. This is TEST-informed exploratory planning, not new results. First explain what the real research bottleneck is based on evidence, not grades. Then list broad experimental objectives (about5), followed by a decision table of method families worth retaining/defering and why existing B73-98 work does/does not already answer them. Select exactly3 high-value bounded experiments/scenarios in priority order, not a laundry list. For each: scientific question, closest PRIMARY prior work and actual distinction (no priority assertion), competing hypotheses, existing experiment overlap, minimum data, task/target/input/availability contract, train/VAL-only development, held-out evaluation and what current TEST reuse means, same-row/compute comparators, point+interval decision rule direction (threshold numeric margin only as proposed to preregister), baseline/falsifier, GPU requirement and resource uncertainty, both-outcome manuscript sentences, stop rule and independent confirmation requirement. Explain whether current data can plausibly supply meaningful empirical contribution, and exactly which missing measurement changes the answer. Include1 immediate GPU-free action, plus clear no-go models/scenarios. Do not imply uncertainty corrections create physical truth or novelty; do not recast all objectives into stale-CES as the paper's sole purpose. Distinguish synthesis inference from verified source claims. Give citations with primary URLs to support choices. <=2400 Korean words.\nOVERLAP AUDIT\n'''+audit+'\nLITERATURE\n'+json.dumps(results,ensure_ascii=False)+'\nAUDITED RESULTS\n'+json.dumps(compact),False,15000)
    E.gate();render_usage()
    save(OUT/'status.json',{'complete':True,'model':MODEL,'literature_stages':6,'synthesis_stages':2,'test_informed_exploratory':True,'new_fits':0,'gpu_runs':0,'source_verification':'Fable reports actual access; independent host checks recorded separately','timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat()})
    print('FABLE LITERATURE AND AGENDA COMPLETE',flush=True)

if __name__=='__main__':main()
