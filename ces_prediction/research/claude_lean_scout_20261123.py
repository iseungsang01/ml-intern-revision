"""Author-authorized redacted Lean/theory scouting; never touches active training.

Independent official-API reviews plus a fresh adversarial synthesis. Exact
requests/responses/usage remain in ignored local data; credentials are not logged.
"""
import concurrent.futures,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b85_unified'))
import claude_review85 as API
CACHE=ROOT/'data/.claude_lean_20261123'
COMMON='''You are a skeptical research collaborator. Find concrete, useful mathematical/formal directions for an existing plasma diagnostic nowcasting paper. Do not estimate journal acceptance, invent results, call an established theorem new, or change the author's research framing. Standard regression is acceptable; a new architecture is not required. We seek useful prediction against hold/interpolation/GP, causal inference inputs on a pedestal-top chord, and a 10-ms compute budget. A physical mechanism or control demonstration would be additional evidence, not an assumed result.

Only this redacted methodological description is transmitted. It contains no identity, raw measurements, shot IDs, unpublished performance scores or credentials. Do not search for the author/project. Use at most TWO web searches, PRIMARY papers and official Lean/mathlib documentation; give exact supporting URLs and distinguish verified sources from search suggestions. End with a final answer under1000words. A theorem sketch is not a Lean-checked proof. No sorry/axiom-based claim of completion. No Lean toolchain is currently available locally.

Task/data contract: scalar carbon-impurity CES reported temperature on one supplied pedestal-top chord, from CES history and present BES/ECEI proxies. F=current-frame report. L5=median of at least3 valid subsequent5 nominal10ms reports within the same segment; offline future-derived label only, not input or independent physical truth. CES exposure starts at its timestamp; actual acquisition/analysis/arrival delays are not measured. We cannot equate an impurity report with bulk-ion truth, infer true values in unobserved missing rows, or prove location-specific transport physics from prediction alone. No closed-loop controller/plant model, independently measured temperature truth, or equilibrium-based per-shot radial verification is supplied.

Availability: latest finite STRICTLY EARLIER indexed CES in the SAME shot/segment, subject to prescribed age cutoffs with3ms timestamp tolerance, observed-gap schedule, or held-out cadence. No current CES as input. Segment reset for time gaps>=.5s. Sampling gaps inside a segment can make row count differ from elapsed time. Training-only normalization; disjoint whole-file train/validation/test manifests. Four validation designs: random, clustered, rolling-origin, final chronological block; none primary. Random splits establish within-shot feature causality, not prospective chronological fitting. The campaign has been reused; no untouched new campaign. Fixed-fit shot-cluster bootstrap does not include refitting/selection uncertainty. Real data only; new empirical work must be preregistered and cannot inspect current TEST to choose methods. Active training must not be changed.

Models: GP plus learned scalar boosted regression is the existing estimator, not handwritten temperature rules. Availability protocols/per-condition fits are a separate generalization issue. A current screen compares shared-encoder LSTM anchor/no-anchor controls and a small causal TCN; no TEST finding is provided here. Anchor model computes shared nonlinear phi(current fast) and phi(fast at last visible CES measurement time), CONCATENATES both with history/age, then a unidirectional LSTM. It does not subtract the embeddings or constrain offset invariance. A shared GELU encoder alone does not guarantee nuisance cancellation. TCN: six ONE-convolution residual blocks, kernel3, dilations1,2,4,8,16,32, left padding only, pointwise per-row LayerNorm/GELU, total127-row receptive field. Inference dropout off. Streaming buffers reset per segment. History-only twin zeros fast channels; fast-enabled model can represent this twin by zeroing the relevant stem columns. Training is finite-budget, so expressivity nesting is not trained-risk superiority. Reported numerical comparisons may have differing reached epochs.

Existing executable assurance: real-array source/config/hash binding; visibility/age/segment assertions; mutations to invisible CES and future fast data must leave the prediction-input prefix unchanged with frozen statistics; label-only test-array mutation must not change predictions; model full/stream/reset/right-padding checks; normal-process-exit requirement; independent metric/bootstrap recomputation. These are finite executable checks, not a Lean proof of the Python/CUDA implementation. Exact-real function proofs do not automatically cover floating-point/backend shape-dependent rounding or the Python-to-Lean refinement gap.

Public primary starting points (content must be checked, not treated as proof of our claims):
https://lean-lang.org/theorem_proving_in_lean4/Introduction/
https://leanprover-community.github.io/mathlib4_docs/Mathlib/MeasureTheory/Function/ConditionalExpectation/CondexpL2.html
https://arxiv.org/abs/2611.18220 (Lean for system identification; distinction almost surely versus every realization)
https://arxiv.org/abs/2607.19727 (machine-checked Lyapunov/small-gain foundations)
https://arxiv.org/abs/1902.03552 (OU remote estimation/Age of Information)
https://doi.org/10.1088/1741-4326/ad142f (RTCAKENN already treats diagnostic absence)
'''
PROMPTS={
'formal_invariants':COMMON+'''
Focus ONLY formal verification. Rank at most3 precise theorems worth formalizing, including a noninterference contract for visibility/label isolation and inference streaming/causality. For each: exact mathematical statement, minimum assumptions, what the code bridge requires, concrete counterexample if an assumption is dropped, current tests already covering examples, and honest value to THIS applied paper. Analyze index causality vs exposure/arrival causality, fixed-stat/weight assumptions, right padding only on valid outputs, and127ROWS versus seconds. Give an isolated minimal Lean4 milestone and clear stop condition; do not create a formal-methods side project or pretend a tautological spec proves Python. Identify any existing claim that formalization could sharpen rather than decorate. Cite concrete library support, label uncompiled API names.
''',
'staleness_theory':COMMON+'''
Focus ONLY useful theory for information age, proxy observations, and estimation error. Find at most3 candidate propositions with proof sketches and falsifiers. Separate known facts from potential task-specific contributions. Examine Bayes MSE risk reduction under nested information (conditional expectation projection), why finite-history summaries/availability-conditioned selected rows need not be nested, and why actual fitted risk can worsen. Analyze a minimal linear-Gaussian/OU example with stale noisy CES and fast noisy proxy; derive an explicit gain-age relationship under stated assumptions, and counterexamples to unconditional monotonic gain with age. Consider anchored proxies with unknown shot offsets: linear difference cancellation versus nonlinear shared-encoder concatenation actually implemented. Is there a nontrivial, falsifiable, identifiable proposition worth a small methods appendix, or only textbook Kalman/AoI theory? Never assume BES/ECEI is a calibrated Ti thermometer. Return the best single proposal with exact data requirements and an empirical protocol using only future preregistered evaluation, not current TEST tuning.
''',
'physics_value':COMMON+'''
Focus ONLY physical/control value and practical originality. Actively try to find a theory-plus-measurement direction that is stronger than proving generic software invariants, while refusing speculative pedestal causal claims. Rank at most3 options: identifiability limits under proxy/beam nuisance and missing-not-at-random labels; uncertainty/abstention or validity-domain guarantees; downstream observer/control utility. Distinguish mathematical validity conditional on a model from empirical validity of the model. Give a useful counterexample/impossibility statement if truth cannot be identified from the available observations. Explain whether Lyapunov/small-gain tools can say anything without a plant/controller and estimator-error bound, and what missing measurement would make that meaningful. Compare against existing RTCAKENN/missing-diagnostic work and AoI estimation theory, without declaring absence of prior art from search failure. State what Lean could and could not strengthen, whether it changes novelty, and a bounded priority plan preserving the current nowcasting storyline.
'''
}

def text_answer(record):
    return '\n\n'.join(b.get('text','') for b in record.get('content',[]) if b.get('type')=='text')

def call(theme,prompt,model,key):
    path=CACHE/(theme+'.json')
    if path.exists():
        r=json.loads(path.read_text());assert r['prompt']==prompt and r['requested_model']==model
        assert r['stop_reason']=='end_turn' and text_answer(r);return r
    for attempt in (1,2):
        actual=prompt if attempt==1 else prompt+'\nDeliver a final answer NOW under600words, without additional searches. Separate proven facts from proposals.'
        payload={'model':model,'max_tokens':8000,'thinking':{'type':'adaptive'},'output_config':{'effort':'low'},
                 'messages':[{'role':'user','content':actual}]}
        if attempt==1:payload['tools']=[{'type':'web_search_20250305','name':'web_search','max_uses':2}]
        started=time.time()
        response=API.request('/messages',key,payload)
        r={'theme':theme,'requested_model':model,'model':response.get('model'),'prompt':prompt,'actual_prompt':actual,'attempt':attempt,
           'usage':response.get('usage',{}),'stop_reason':response.get('stop_reason'),'elapsed_seconds':time.time()-started,'content':response.get('content',[])}
        (CACHE/f'{theme}_attempt{attempt}.json').write_text(json.dumps(r,indent=2),encoding='utf8')
        print(json.dumps({'theme':theme,'attempt':attempt,'model':r['model'],'stop':r['stop_reason'],'usage':r['usage'],'elapsed_seconds':r['elapsed_seconds']}),flush=True)
        if r['stop_reason']=='end_turn' and text_answer(r):
            path.write_text(json.dumps(r,indent=2),encoding='utf8');return r
    raise RuntimeError('No complete final answer for '+theme+'; partial attempts preserved, not treated as review')

def render():
    """Publish audit metadata only; final advice needs independent adjudication."""
    usage={'calls':[]}
    for p in sorted(CACHE.glob('*_attempt*.json')):
        r=json.loads(p.read_text(encoding='utf8'))
        usage['calls'].append({**{k:r[k] for k in ('theme','attempt','model','stop_reason','usage','elapsed_seconds')},
                               'record_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for field in ('input_tokens','output_tokens'):
        usage[field]=sum(c['usage'].get(field,0) for c in usage['calls'])
    usage['web_search_requests']=sum(c['usage'].get('server_tool_use',{}).get('web_search_requests',0) for c in usage['calls'])
    (CACHE/'usage.json').write_text(json.dumps(usage,indent=2),encoding='utf8')
    lines=['# Claude API Lean review audit','',
           'Official Anthropic API, authorized by the author. Only redacted methodology and public sources were transmitted. No raw measurements, shot IDs, identity, credentials or unpublished predictive scores were sent. Exact responses, including private reasoning blocks, remain in the ignored local cache and are not reproduced here. These records establish API use, not correctness or originality.','',
           'Adjudicated findings, source checks, mathematical corrections and the bounded recommendation: [Lean feasibility](lean_feasibility_20261123.md). No Lean proof was compiled and no training method or TEST allocation was changed.','',
           '| Review | Returned model | Stop | Input tokens | Output tokens | Seconds | Local record SHA-256 |',
           '|---|---|---|---:|---:|---:|---|']
    for c in usage['calls']:
        u=c['usage'];lines.append(f"| {c['theme']} attempt {c['attempt']} | `{c['model']}` | `{c['stop_reason']}` | {u.get('input_tokens',0)} | {u.get('output_tokens',0)} | {c['elapsed_seconds']:.2f} | `{c['record_sha256']}` |")
    lines += ['',f"Total: {len(usage['calls'])} paid attempts; {usage['input_tokens']} input tokens; {usage['output_tokens']} output tokens; {usage['web_search_requests']} server web searches. Output includes billed thinking tokens where reported. No price estimate is inferred.",'',
              'Three initial reviews covered formal invariants, staleness theory and physical/control value. A fresh adversarial synthesis challenged those proposals. Two focused follow-ups checked the TorchLean implementation bridge and disputed mathematics. The final audit explicitly corrected earlier reviews; agreement among API responses was never treated as proof.','',
              'Prompts and credential-safe API orchestration are in `ces_prediction/research/claude_lean_scout_20261123.py`, `claude_lean_bridge_20261123.py` and `claude_lean_math_audit_20261123.py`. Regenerate this metadata without an API request by importing the scout module and calling `render()`. Its `main()` performs the paid research workflow and reuses exact completed prompt/model caches.']
    (ROOT/'docs/paper/claude_lean_reviews_20261123.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return usage

def main():
    CACHE.mkdir(parents=True,exist_ok=True);key=API.credential();catalog=API.request('/models?limit=1000',key)
    ids=[m['id'] for m in catalog.get('data',[])];assert 'claude-fable-5-2' in ids and 'claude-opus-5-5' in ids
    (CACHE/'model_catalog.json').write_text(json.dumps({'available_relevant':[i for i in ids if 'fable' in i or 'opus' in i],'recorded_at':time.time()},indent=2),encoding='utf8')
    records={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(call,t,p,'claude-opus-5-5' if t=='physics_value' else 'claude-fable-5-2',key):t for t,p in PROMPTS.items()}
        for future in concurrent.futures.as_completed(jobs):records[jobs[future]]=future.result()
    critique=COMMON+'''\nYou are a fresh adversarial cross-reviewer. The three independent reviews below are proposals, not facts. Check every attractive theorem for hidden assumptions, known prior art, circular proofs and mismatch with the actual implemented model. Reject decorative Lean use, unverifiable citations and unidentifiable physics. Recommend at most ONE theoretical direction and ONE assurance milestone, or explicitly recommend no Lean work now. Provide an exact corrected theorem statement, counterexample, evidence needed, and a go/no-go plan in Korean. Explicitly distinguish algorithm/spec proof, Lean checking, Python refinement, floating-point execution, and empirical physical truth. Do not estimate journal acceptance. Search at most2times; finish under1200words.\n'''
    for t in PROMPTS:critique+='\n--- REVIEW '+t+' ---\n'+text_answer(records[t])
    records['adversarial_synthesis']=call('adversarial_synthesis',critique,'claude-opus-5-5',key)
    usage=render();print(json.dumps({'completed_reviews':len(records),'input_tokens':usage['input_tokens'],'output_tokens':usage['output_tokens']}),flush=True)

if __name__=='__main__':main()
