"""Author-authorized Fable reviews: runtime and principled candidate reduction.

Only redacted methodological contracts and public runtime observations are sent.
No arrays, identifiers, scientific test results or credentials leave the machine.
"""
import concurrent.futures,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b85_unified'))
import claude_review85 as API
CACHE=ROOT/'data/.fable_efficiency_20261123'
REPORT=ROOT/'docs/paper/fable_efficiency_20261123.md'
COMMON='''Act as a skeptical senior collaborator on an actual small diagnostic nowcasting study.
We need substantially faster progress and a defensible SMALL method shortlist, not an architecture zoo.
Task: causal scalar temperature nowcast at one pedestal-top measurement chord using fast diagnostics
and only previously available temperature measurements. Direct two-head normalized F=current report
and L5=median of at least3 subsequent5 reports, a future-derived TRAINING LABEL ONLY, not truth.
Real previously used data: about246k grid rows,639 files, mostly10ms, segments reset at gaps>=.5s.
22 input channels:13 fast plus2 zero placeholders; log1p(dt), carried past temperature, log1p(age),
has-history and unused rotation slots. Train-file-only normalization. Seven availability rules mixed
by segment/epoch; an every10th-frame rule is excluded from training/validation. No current/future CES
as input, no test-time normalization. Four designs random-file,clusterblocked,rollingorigin,chronological,
none primary. Existing boosted-tree/history baselines exist. No new architecture/physics discovery or
journal acceptance claim is needed. Standard regression is eligible if useful against fair baselines.
Training:2layer160 LSTM,LayerNorm,two64GELU heads,AdamW.001,batch16 full segments,cap80,plateaupatience2,
earlystop6,masked equal-head MSE+.1negativepenalty. Train about170krows/1328segments, val34k/235segments
in firstunit. Val replays ALL7 rules EVERYepoch, equalrule/head MSE chooses checkpoint. Split units15:
V1 fivefold,V2 fivefold,V3 four origins,V4a finaltime;3seeds;X/H historytwin =>90fits/model family.
Hardware:8GB laptopGPU,Windows,Python3.14,torch2.11CUDA12.8. Baseline uses packed variable-length LSTM.
cuDNN path computed then interpreter aborted on teardown; nativeATenCUDA clean but lowGPUuse~10%,oneCPUcore.
Eachfit about40min; training83batches/epoch versus7xvalidation105batches/epoch. No profiler breakdown yet.
Frozen ongoing files cannot be changed; existing completed fits remain historical. New implementation
and revised allocation require committed preregistration, independent real-input/loss/gradient checks,
normal interpreter exit0 (no os._exit bypass) and source/config bound artifacts. Own queued supervisors
can be superseded; never stop unrelated GPU holders. No synthetic observations or external private data.
Currently queued:anchor/current shared13->32->16encoder, lastvisibleCES MEASUREMENT-time anchor, matched
no-anchor control, originalLSTM fused39;90fits. Then7candidates:lagdecayGRU,causalTCN128/6blocks/RF127,
small diagonal discrete/continuous learned observers(NOT actual RKN/CRU reproductions),LSTM Huber,
fixedStudentt4,LSTM Gaussianmean/logvar;210one-seedscreenfits15unitsXH,validation-onlytop2+120fits.
This cost is excessive. No test score has been used to choose candidates. Need practical preregistered
revision with honest reused-data exploration, no inflated claims from reduced validation/seed budgets.
Use web_search at most3times for PRIMARY papers/official technical docs, cite exact URLs. Give concrete
best recommendations and failure tests; do not promise unmeasured speedups or recite30methods.
Answer under2200words so the response finishes. Tables okay. No code implementation necessary.
'''
PROMPTS={
'training':COMMON+'''
Analyze ONLY current model TRAINING methodology/runtime, independently of shrinking candidate count.
Rank concrete bottleneck fixes by expected benefit/engineeringrisk and classify arithmetic-equivalent
implementation change vs statistical-method change. Examine packed vs dense CAUSAL right-padded LSTM
with supervision masking (no backward crosspadding), cuDNN/trainingflag/dropout/cleanexit/version issues,
precache validation tensors and batching rules, equal-rule/head aggregation, avoidingscalar sync,
lengthbucketing, batchsize, mixedprecision/compile, boundedcontextTBPTT and validationfrequency.
Choose the smallest highconfidence implementation/profiling plan that can run after the current fit,
not after90fits. Specify realtrace batch/gradient/optimizerstep/stream equivalence tests, tolerances,
maxbenchmarkbudget, decision gates, rollback and preserving completed results. Is fused dense likely
worth testing before handrolledcell/observer models? Identify misleading acceleration checks, including
evalmode cuDNN backward and dropoutRNG. Propose a fixed training budget for a NEW batch if exact
baseline budget is too expensive; don't silently call changedstatisticssamecomparison. End with exact
recommended next3actions and the minimum uncertainty that must be resolved by measurement.
''',
'selection':COMMON+'''
Analyze ONLY CANDIDATE REDUCTION methodology. Choose at most THREE model configurations INCLUDING
the baseline/control, not three broad families with dozens of hidden subvariants. Account for redundant
handrolled observers, lack of fit-error labels and weak calibration evidence; distinguish precision gain
from addingphysicalvalue. Protect explicitanchor hypothesis and a strongsimpletemporalbaseline ifjustified.
Should robust losses or uncertainty be deferred rather than searched? Compare fixedsmallpilot vs
successivehalving/Hyperband/ASHA, learningcurveearlyrankmisordering, samevalidationfit reuse, temporal
designs, splitunits and pairedX/H controls. Recommend an EXACT small budget: whichoneunit/design/seed,
epochsorwallbudget, valonlyranking/stopcriterion, numberpromoted, expanded4designcoverage, totalfits,
testlock and honest intervals/modelselectionlimitations. We can report all4designs without three seeds
for every weakcandidate; don't assert unchanged evidence strength. Do not select using alreadyseen
testmetrics or dropdifficulttemporaldesigns. Include whatmeasurementwouldre-open a deferredcandidate.
End with ONE preferred portfolio, exacteligibility/rankingrules and a succinct protocol we can register.
'''
}


def one(theme,key,model):
    path=CACHE/(theme+'_answer.json')
    final_prompt=PROMPTS[theme]+'\nIMPORTANT: Deliver your final actionable answer now, at most900words. Use at most ONE web search, then finish. Prioritize the concrete plan over extended deliberation.'
    if path.exists():
        record=json.loads(path.read_text(encoding='utf8'))
        if record['prompt']!=final_prompt:
            record['prompt']=final_prompt
            record['prompt_record_note']='Exact deterministic request suffix restored; response and usage unchanged.'
            path.write_text(json.dumps(record,indent=2),encoding='utf8')
        return record
    payload={'model':model,'max_tokens':8000,'thinking':{'type':'adaptive'},'output_config':{'effort':'low'},
             'tools':[{'type':'web_search_20250305','name':'web_search','max_uses':1}],
             'messages':[{'role':'user','content':final_prompt}]}
    started=time.time();response=API.request('/messages',key,payload)
    record={'requested_model':model,'model':response.get('model'),'theme':theme,'prompt':final_prompt,
            'stop_reason':response.get('stop_reason'),'usage':response.get('usage',{}),
            'elapsed_seconds':time.time()-started,'content':response.get('content',[])}
    path.write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps({'theme':theme,'model':record['model'],'stop':record['stop_reason'],'usage':record['usage']}),flush=True)
    return record


def main():
    CACHE.mkdir(parents=True,exist_ok=True);key=API.credential();catalog=API.request('/models?limit=1000',key)
    models=[m['id'] for m in catalog.get('data',[]) if 'fable' in m['id'].lower()]
    assert models,'Requested Fable unavailable; do not silently substitute'
    model='claude-fable-5-2' if 'claude-fable-5-2' in models else models[0]
    records={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(one,t,key,model):t for t in PROMPTS}
        for future in concurrent.futures.as_completed(futures):records[futures[future]]=future.result()
    lines=['# Fable training efficiency and candidate reduction review','',
           'Author-authorized official API. Redacted protocols only, no raw observations, identity, credentials or scientific test scores. Independent methodological advice, not evidence of performance or novelty. Exact prompts/responses and usage retained locally. Initial thinking-disabled requests were rejected by this model. Two adaptive/high-effort requests exhausted their output budgets without final answers; their exact records remain training.json/selection.json and are not treated as completed reviews. The bounded final-answer requests use adaptive/low effort on the SAME Fable model, one search maximum.','']
    for theme in PROMPTS:
        r=records[theme];lines+=['## '+theme,'',f"Returned model `{r['model']}`, stop `{r['stop_reason']}`, usage `{json.dumps(r['usage'])}`.",'',
                               '### Exact prompt','',r['prompt'],'','### Response','']
        lines+=['\n\n'.join(b.get('text','') for b in r['content'] if b.get('type')=='text'),'']
    REPORT.write_text('\n'.join(line.rstrip() for line in '\n'.join(lines).splitlines())+'\n',encoding='utf8');print('Fable report written',flush=True)


if __name__=='__main__':main()
