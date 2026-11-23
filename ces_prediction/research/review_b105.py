"""Bounded redacted official Claude review for the author's B105 execution."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'data/.b105/review'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b85_unified'))
import claude_review85 as A

def call(stage,prompt):
    CACHE.mkdir(parents=True,exist_ok=True)
    req=CACHE/(stage+'.request.json');out=CACHE/(stage+'.response.json')
    if out.exists():
        assert json.loads(req.read_text(encoding='utf8'))['messages'][0]['content']==prompt,'Cached review prompt mismatch'
        r=json.loads(out.read_text(encoding='utf8'))['response']
    else:
        assert not req.exists(),'Inspect prior attempt, no blind replay'
        assert len(list(CACHE.glob('*.request.json')))<4,'Four bounded attempts including incomplete implementation review'
        payload={'model':'claude-fable-5-2','max_tokens':16000 if stage=='implementation_final' else 11000,'thinking':{'type':'adaptive'},'output_config':{'effort':'low'},'messages':[{'role':'user','content':prompt}]}
        req.write_text(json.dumps(payload,indent=2),encoding='utf8');start=time.time()
        try:r=A.request('/messages',A.credential(),payload)
        except Exception as e:
            (CACHE/(stage+'.error.json')).write_text(json.dumps({'error':str(e),'seconds':time.time()-start}),encoding='utf8');raise
        out.write_text(json.dumps({'response':r,'seconds':time.time()-start},indent=2),encoding='utf8')
    text='\n'.join(b['text'] for b in r['content'] if b['type']=='text')
    assert r.get('stop_reason')=='end_turn' and text,'Incomplete response preserved'
    (CACHE/(stage+'.txt')).write_text(text,encoding='utf8')
    print(json.dumps({'stage':stage,'model':r['model'],'usage':r['usage'],'stop_reason':r['stop_reason']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['design','implementation','implementation_final','post']);a=p.parse_args()
    if a.stage=='design':
        prompt='''Review this execution contract BEFORE preregistration/GPU. <=800 Korean words; concrete blocking flaws/fixes only. Do not demand additional author permission, invent results, claim novelty or estimate acceptance. Author authorizes four GPU fits and aggregate-only official API reviews. No raw observations, identity or secrets sent.
Pedestal-top CES report nowcast, current F and offline next5-median L5. Same fixed4 units from old protocol (random files, clustered files, rolling, finalblock), seed1701 only. No primary design; this is TEST-informed exploratory screening, not initialization robustness/independent confirmation. Existing22-channel TCN did not establish merit over GP+boosted residual and time extension did not fix fast increment. GPU only one new35-channel TCN, same widths/dilations/loss/dropout/batching/optimizer/earlystop6/40epoch, soft3600s (older22 budget1200s, disclosed). All four fits run; no architecture search. Input extends22 with13 source-time fast channels, existing prepared real TRAIN/VAL cached35 arrays. Proposed initialization: preserve seeded22 stem/blocks/heads exactly, widen stem with extra13 columns initially zero, preserve RNG state across widening. All added columns trainable. Finite nonzero anchor gradient on actual real batch checked. This isolates information opening at initialization, not pure architecture effect. Frozen original train92 numeric loop imported via new adapter, existing source never edited.
Critical fairness: old GP+tree uses all outerfit files including neural innerVAL, so cannot use old predictions to select new model. Refit identical HistGradientBoosting/defaults per rule+head on neural INTERNAL TRAIN only; its fast-window normalization from those TRAIN files only. GP is row-local optimized from <=16 visible past CES, no pooled hyperparameters, existing causal input cache safe with source checks. New CPU trees: specialized7rules x2heads x4units=56, plus one mixture rule-balanced tree/head/unit=8 (64 total). No K10 used for fitting/VAL. Training-row support can differ (tree common masks vs neural allmasked) and must be disclosed. Compare GP/tree and bothTCNs on same common rows and source info policy, not artificial global intersection of designs.
Checkpoint selection remains original equal7rule x2head masked normalizedMSE over innerVAL. Separate allocation gate recomputes on matching tree commonVAL rows: mean7rules x2heads physicalMSE/ti_sd^2, candidate/fair specializedtree <=1.05 in EACH of4units. This is deterministic compute-allocation gate, not statistical noninferiority. Gate after allfits+normal exits+independentfitchecks. If fails, NO candidateTEST inference, stop no furtherfits. Existing pairedTEST evidence remains available. If passes, frozen35 candidateTEST all8rules x2heads at4units, compare innerTRAIN specializedtree+mixture, old outerfitTREE_TX, original22TCN, hold/GP; optional saved B84TXall L5 only where available(noV2), old D10-applied comparator never alone. No silent immediate15unit/3seed expansion. For existing22 vsB84 compute new pairs only after committed preregistration; report all available designs/cells, missing stays missing. Old LSTM+postcorrection D1predictions allowed only if explicit manifest/row/target/source binding matches, D10random-file unit only; otherwise unavailable. No restoration of removed earlier research.
Results point+95% cluster and discharge20000 draws, deterministic shared paired rows; one init fixed, no refit/init/selection uncertainty. Allcells report; exploratory no posthoc endpointselection, Holm only if claiming individual discovery; global claim requires allprespecifiedcases and all4designs. No latency superiority from old forward-only timing. Stream/reset/causal mutation/trainstat/target-noninterference, new35smokeCUDA finitegradient+normalexit, max4globalworkers andVRAMguard. Whole suite150min after dispatch start, perchild hard4020s includes600ish startup; CPUbaseline preparation capped45min; no unbounded retry, failed artifacts preserved. Independent monitor and actual progress notifications. Register+commit source BEFORE any scientific fit, CPU included. Separate independent audit reconstructs errors/points/intervals, source/provenance checks before publication or Claudepost. Need adjustments to gate/CPUmixture row weighting/modelinitialization/CPUparallel budget?'''
    else:
        folder=ROOT/'ces_prediction/experiments/b105_merit'
        files=[*sorted(folder.glob('*.py')),ROOT/'ces_prediction/experiments/PREREGISTRATION_B105.md']
        if a.stage=='implementation_final':files=[folder/n for n in ('pipeline105.py','common105.py','train105.py','model105.py','audit105.py')]
        prompt='Review this implementation for blocking scientific/operational flaws. Return <=1100 Korean words. No raw observations/identity/secrets included. Author asks train+verify merit. Existing files immutable; only new B105 code shown. Distinguish actual bugs from limitations; no new broad search or permission requests. Focus source-time alignment, information leakage, common rows, ratio/gate math, proper process exits, frozen code/provenance, GPU dispatch, independent audit. No acceptance estimates.\n'
        if a.stage=='post':
            report=json.loads((ROOT/'docs/paper/b105_merit.json').read_text(encoding='utf8'))
            packet={k:report[k] for k in ('exploratory','units','seeds','candidate_TEST_evaluated','statements','limits')}
            packet['allocation']=report['gate']['units']
            packet['overview']={k:v for k,v in report['cells'].items() if any('/'+r+'/' in k for r in ('D10','D650','S','K10')) and ('over_fair_specialized' in k or 'over_incumbent_outerfit' in k or 'over_B84_mixture' in k)}
            prompt='Interpret these independently audited anonymous aggregate results; <=700 Korean words. Is there practical merit over incumbent? Distinguish VAL screening and TEST, no new physics or architecture inferiority, no automatic extra fits. Give one next priority.\n'+json.dumps(packet)
        else:
            prompt+='\n\n'.join('FILE '+p.name+'\n'+p.read_text(encoding='utf8') for p in files)
            if a.stage=='implementation_final':prompt+='''\nPrevious review reached max_tokens after identifying two bugs: step env was unpinned (NOW fixed) and S2/S3 omitted (NOW computed and independently audited). Please finish a focused <=600 Korean-word review of these corrected runtime/provenance files, not a new methodology search. Full producer uses mean14cell candidate/tree gate<=1.05 every4units, source-row matched, 20000multinomial cluster/discharge percentile intervals, no candidateTEST if failed. All source code and registration must commit before any real fit/check. No need to re-review those statistical definitions. Frozen train92 imports common92, uses overridden T.create_model for35; batches produce35; same loop numeric code; generic checks92.model_contract accepts35 stream signature. CPU reference sklearn trees=64fits, internalTRAIN normalization, early_stoppingFalse, unit parallel workers4/OMP2; before GPU group; models retained and hashes checked before conditionalTEST. CausalGP source uses last16 visible per row not global fit. Review definite blockers only, differentiate already fixed items, don't assume unseen imports are necessarily broken. Output concise with PASS/conditional on specific checks/blocking flaws, never acceptance score.'''
    call(a.stage,prompt)
