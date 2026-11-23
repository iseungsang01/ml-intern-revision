"""One additional bounded Fable correction of identified planning errors."""
import json
import fable_literature_agenda as L

def main():
    old=(L.OUT/'10_final_research_agenda.md').read_text(encoding='utf8')
    prompt=L.CONTEXT+'''\nA host audit found concrete errors in your final agenda. Correct them rather than repeating or defending them. No new web search, no fits. Return Korean corrected decision memo <=1200words, preserving5 objectives, up to8 family decisions and exactly3 prioritized experiments with compact hypothesis/overlap/comparator/decision/falsifier/resources/stop fields, one immediate action and source pointers already verified. Include a clear list of retracted claims. Corrections REQUIRED:
1. Separate CI overlap does NOT establish no difference. For E1 use a directly paired contrast of effects/loss differences on same rows and resampled clusters. Failure to reject is inconclusive, not support for equality/confounding.
2. A difference CI including zero is NOT evidence of robustness/equivalence. E2 needs a proposed TRAIN/VAL-grounded equivalence margin fixed in preregistration and the entire paired interval inside that margin, or report inconclusive. Define estimand/scale across different targets/supports; don't compare raw labels as same physical truth.
3. No significant harm in other designs is NOT noninferiority. If E3 uses a noninferiority criterion define direction and preregistered margin with full CI bound; otherwise report tradeoffs. Define whether positive contrast means loss increase or skill gain.
4. Old expanding past-only standardization already exists in maintained seq_data.py (expanding_standardize). Current-line B84 already evaluated dropping absolute levels TXrel, pooled availability TXall and raw-window controls. Do not label past-only normalization or removing levels new; first map exact data/feature/model overlap. NO old pre-cleanup scientific numbers are supplied or may be quoted.
5. RKN/CRU learn latent dynamics and do not require the exact known physical f/h assumption of KalmanNet. Separate technical applicability from lack of new question/priority. S4/Mamba/self-supervision work on10ms data; full-rate raw data is not technically mandatory.
6. 'Only empirical contribution is possible with current data' is not a proven impossibility of new algorithms. Say current evidence establishes no algorithmic novelty, and new methods require a falsifiable task-specific hypothesis and existing-screen overlap audit.
7. Do not prescribe stopping at lack of significance as proof of no effect. Specify precision/power/equivalence or inconclusive outcomes. No invented thresholds or planned sample-size adequacy claims.
8. Do not say new DATA alone changes conclusions; experiments/analysis may also change task-specific conclusions. Distinguish unidentifiable physical claims needing new measurement from predictive hypotheses evaluable on current real data.
9. Keep the author's fixed aim: pedestal-top NOWCAST performance vs hold/GP/interpolation, diagnostic increment, four designs and computation, applications optional. Never make stale CES the sole paper storyline.
10. Source access limits remain. Diag2Diag/Jalalvand one paper, Imrisek metadata corrected; source09 incomplete and host corrections prevail. Do not imply broad time-order absence across all literature from5 partial texts.
Do not invent new publications or numerical performance. Preserve disagreements with reviewer recommendations when already addressed. Give actionable overall experimental goals and what family should be considered NEXT CONDITIONAL on which observations, not only a list of things forbidden. GPU fits may be proposed with approval/preregistration requirements but not launched.\nYOUR PLAN TO CORRECT\n'''+old
    text=L.call('11_corrected_decision_memo',prompt,False,11000)
    L.render_usage()
    p=L.OUT/'api_audit.json';x=json.loads(p.read_text(encoding='utf8'));x['max_paid_calls']=11;x['extra_call_reason']='Host identified invalid CI-overlap/equivalence reasoning and existing normalization overlap; one bounded correction, no blind replay.';L.save(p,x)
    p=L.OUT/'status.json';x=json.loads(p.read_text(encoding='utf8'));x.update({'max_paid_calls':11,'primary_text_audit_complete':False,'primary_text_notes_corrected_in_final':True,'final_plan':'11_corrected_decision_memo.md','prior_plan':'10_final_research_agenda.md','additional_correction_calls':1,'new_fits':0,'gpu_runs':0});L.save(p,x)
    print('CORRECTED FABLE AGENDA COMPLETE',flush=True)

if __name__=='__main__':main()
