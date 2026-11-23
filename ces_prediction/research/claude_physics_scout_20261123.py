"""Author-authorized simple physical models; no private observations transmitted.

Research only: no fitting, TEST inspection, predictive experiments or GPU work.
Final advice is independently checked before publishing a synthesized report.
"""
import concurrent.futures,hashlib,json
from pathlib import Path
import claude_lean_scout_20261123 as S

ROOT=Path(__file__).resolve().parents[2]
S.CACHE=ROOT/'data/.claude_physics_20261123'
COMMON='''Find genuinely useful, SIMPLE physics/mathematical approaches for an existing pedestal-top CES nowcasting study. The author asks for physical reasoning beyond Lean software proofs. Preserve the task: compare useful estimators to hold/interpolation/causal GP under a10ms compute budget, causal inputs at the provided pedestal-top chord. A standard regressor is acceptable. Do not replace the paper's framing, invent results, claim a theorem or architecture is new, estimate publication acceptance, or recommend an unbounded PINN/large-parameter search. A simple physical relation with falsifiable assumptions is preferable to decorative constraints.

This is a redacted methodology-only request: no author identity, raw observations, shot IDs, credentials or unpublished predictive scores. Search public PRIMARY diagnostic/physics papers and official diagnostic documentation, at most TWO searches. Cite exact supporting source URLs and say what was actually checked versus suggested. Distinguish textbook identities, known application, proposed hypothesis and established empirical result. End under1000words.

Available working data: one carbon-impurity CES Doppler-temperature report plus toroidal velocity at a supplied edge chord;9 BES channels and4 ECEI channels; all working fast values are ONE raw voltage/sample at nominal10ms timestamps, NOT window-averaged turbulence statistics. Time intervals can have gaps. No calibrated ne, calibrated Te, magnetic equilibrium, q95, B-field, plasma current, radial gradients, channel geometric mapping, local absorbed power or raw CES spectra/instrument-function/intensity-fit uncertainty are established in these working files. NBI source power/voltage exists on the same grid, but injected power is not local absorbed ion heating. Calibration/optical thickness are not established. Do not call BES intensity density or ECEI voltage Te without a forward model and assumptions. Original high-rate BES/other diagnostic streams exist for only a small subset of discharges in a separate raw folder; not part of the full-campaign model inputs. A raw-subset audit is possible after separate registration; no high-frequency turbulence, ELM heat flux, cross-phase or spectra can be inferred from50Hz-Nyquist grid snapshots.

Geometry: provided chord called pedestal top; no per-discharge equilibrium-confirmed normalized radius, no established independent neighboring Ti gradients. Fast sensors cannot be assumed co-located with each other or CES just because they were supplied together. Do not infer location-specific transport mechanisms or causal heating from predictive association. One reused campaign, no untouched independent campaign or independent bulk-ion truth.

Targets: F=current-frame reported impurity temperature; L5=median of at least3 of the next5 nominal10ms reported values, offline future-derived target only, not physical truth. CES has an approximately10ms exposure; exact acquisition/analysis/arrival timing must not be assumed from the timestamp. At online decision t, no current/future CES report is input. Available inputs are current/past fast samples, past NBI power, and latest permitted strictly earlier CES at anchor s in same segment; weights/stats fitted on TRAIN only, segment reset at gaps>=.5s. For F the future portion of exposure may make an exact frame-integrated physical label unavailable at t. Distinguish measurement convolution from pointwise state. Missing reports and fit artifacts may correlate with beam/state (MNAR). Independent report noise and Ti=Tcarbon=main-ion temperature are assumptions, not verified.

Current comparisons already include GP+boosted scalar regression on history/fast, fast levels and anchor differences/log ratios, ECE-only/BES-only, NBI-only and fast-added-to-NBI controls, history-only twins, input-age/cadence rules, spatial commonality and beam/gap controls. A simple Te-ratio estimator was considered on a different public device and did not establish a universal Ti/Te relation; do not recycle it as a discovery here or extrapolate its measured performance. An anchor LSTM concatenates nonlinear current/anchor embeddings; it does NOT enforce differencing or nuisance cancellation. A small causal TCN is another comparator. Lean now proves a generic latest-visible selector, not physical laws or PyTorch correctness. New exploration must first compare to already existing inputs/controls to avoid redundant experiments. No new empirical result is requested now. Future real-data fits/diagnostic audits require a committed registration and TRAIN/VAL-only development, all four outer validation designs (random/clustered/rolling/final-block), paired common-row scoring and shot-cluster intervals; no current TEST tuning.

Deliver each candidate with a short equation and units, assumptions, directly observed versus unobserved quantities, identifiability or gauge freedom, simplest nontrivial comparator, failure/negative-result falsifier, minimum extra measurement, and what prediction-only success could honestly show. Separate feasible NOW from feasible after a bounded raw-subset or calibration audit and infeasible with current data.
'''

PROMPTS={
 'measurement_physics':COMMON+'''
Focus on the physical FORWARD MODELS of CES, BES, ECEI and nuisance-invariant simple estimators. Derive line-mixture/finite-exposure second-moment broadening (thermal plus velocity variance plus instrument) carefully for wavelength/velocity units and intensity-weighted mixtures. Does the existing single per-frame velocity report suffice to subtract unresolved intra-frame variance? Derive optically thick/thin ECE radiative transfer and BES neutral-beam/excitation dependencies, and analyze whether anchor differencing, ratios or multichannel contrasts cancel beam/gain nuisance without canceling the desired state. Unknown additive offsets/negative voltages, varying neutral density, attenuation and noncollocation matter. Rank at most2 practical hypotheses, not generic warnings. Identify the smallest useful raw/calibration metadata audit and an estimator implementable with the current scalar data if honest. Do not claim common-mode removal isolates physics automatically.
''',
 'energy_balance':COMMON+'''
Focus on simple energy/particle dynamics: two-temperature energy exchange and transport, carbon-versus-main-ion relaxation, local0D thermal response and stable low-order anchor models. Start from the correct Eulerian energy equation including density/compression/advection terms, then show EXACTLY which assumptions permit a1- or2-pole scalar approximation and an integral/closed-form causal predictor. Is any parameter physically identifiable from uncalibrated fast voltages and unknown transport/heating? Separate an effective stable grey-box response from Spitzer collision time, diffusivity or physical power balance. Can positivity, dissipation, nondimensionalization, semigroup consistency or constrained kernels give a useful low-parameter model WITHOUT forcing wrong cross-channel signs? Do not propose Te drives Ti with a universally positive gain, equate NBI injected power to local heating, or infer heating direction from lag. Rank at most2 feasible approaches including fixed-fit minimal baselines and exact testable added value beyond existing GP/trees/TCN. State time-resolution/exposure limitations explicitly.
''',
 'pedestal_geometry':COMMON+'''
Focus on physical relationships at a fixed pedestal measurement chord: radial displacement of steep profiles versus real heating, linearized delta T=-xi*dT/dr, electron/ion gradient ratios, polytropic/entropy relations and pressure balance. Distinguish Lagrangian adiabatic perturbations from Eulerian fixed-chord changes; missing n/T calibration, gradients, flow and B cannot be wished away. Analyze a small latent displacement/thermodynamic model and identifiability/gauge ambiguities from the present9+4voltages and one CES scalar. Could the raw high-rate subset (geometry still unverified) falsify a coherent-mode/displacement hypothesis cheaply, versus a simple anchor deltaTi~beta_e deltaECEI+beta_b deltaBES model being merely regression? Rank at most2 useful candidates. Give a proposed measured signature that distinguishes hypotheses, confounders that give the SAME signature, and minimal added diagnostic needed to call it physical rather than a low-rank statistical representation. Explain where a physical formulation can simplify the model even if no transport mechanism is discovered.
'''
}

def audit():
    """Publish usage/hashes and symbolic-check metadata, never raw API advice."""
    records=[]
    for p in sorted(S.CACHE.glob('*_attempt*.json')):
        x=json.loads(p.read_text(encoding='utf8'))
        records.append({**{k:x[k] for k in ('theme','model','attempt','stop_reason','usage','elapsed_seconds')},'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    usage={'calls':records,'input_tokens':sum(c['usage'].get('input_tokens',0) for c in records),
           'output_tokens':sum(c['usage'].get('output_tokens',0) for c in records),
           'web_search_requests':sum(c['usage'].get('server_tool_use',{}).get('web_search_requests',0) for c in records)}
    (S.CACHE/'usage.json').write_text(json.dumps(usage,indent=2),encoding='utf8')
    algebra=S.CACHE/'algebra_checks.json'
    publication={**usage,'scope':'Research proposals, no empirical fit or TEST analysis; raw responses stay local',
                 'algebra':json.loads(algebra.read_text(encoding='utf8')) if algebra.exists() else None,
                 'research_sources_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('claude_physics_math_audit_20261123.py'),Path(__file__).with_name('check_physics_formulations_20261123.py')]}}
    (ROOT/'docs/paper/physics_formulations_20261123.json').write_text(json.dumps(publication,indent=2)+'\n',encoding='utf8')
    return usage

def main():
    S.CACHE.mkdir(parents=True,exist_ok=True)
    key=S.API.credential();catalog=S.API.request('/models?limit=1000',key)
    ids=[m['id'] for m in catalog.get('data',[])];assert {'claude-fable-5-2','claude-opus-5-5'}<=set(ids)
    reviews={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(S.call,t,p,'claude-opus-5-5' if t=='pedestal_geometry' else 'claude-fable-5-2',key):t for t,p in PROMPTS.items()}
        for f in concurrent.futures.as_completed(jobs):reviews[jobs[f]]=f.result()
    prompt=COMMON+'''
You are a fresh adversarial physical-methods reviewer. The three independent proposals below are not evidence. Find mathematical, units, identifiability, timing and data-contract mistakes. Recommend ONE easiest physical formulation worth a bounded future experiment and optionally ONE raw-subset diagnostic audit. If no present-data physical identification is possible, still give the simplest honest physics-motivated predictor with explicit gauge/assumption limits, rather than stopping at impossibility. A generic estimator can be useful without a new physics claim. Provide its closed-form update and parameter count, known prior art, explicit nested comparator and what would overturn the usefulness hypothesis. Do not assume the previous reviewers verified their sources. Do not invent results. Say which extra measurement changes the interpretation materially. Distinguish temperature truth from reported scalar and causal inputs from physical causality. Separate current deterministic input-evaluation from F's finite future exposure and L5's offline future median. Rank proposals and say why not to run the others now. Answer in Korean under1200words, at most2source searches.
'''
    for t in PROMPTS:prompt+='\nREVIEW '+t+'\n'+S.text_answer(reviews[t])
    reviews['adversarial_choice']=S.call('adversarial_choice',prompt,'claude-opus-5-5',key)
    usage=audit()
    print(json.dumps({'completed_reviews':len(reviews),'input_tokens':usage['input_tokens'],'output_tokens':usage['output_tokens']}),flush=True)

if __name__=='__main__':main()
