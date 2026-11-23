"""Critique target/protocol validity, not just compliance with prior choices. No fits."""
import hashlib
import json
import re
import sys

import verify_with_claude as V

ROOT=V.ROOT
OUT=ROOT/'docs/paper/research_reader/definition_audit'

def readtext(path):return (ROOT/path).read_text(encoding='utf8')

def packet():
    manuscript=V.read(ROOT/'docs/paper/introduction_review/manuscript_sections.json')
    registration=readtext('ces_prediction/experiments/PREREGISTRATION_B73.md')
    registration=registration[registration.index('## 2.'):]
    common=readtext('ces_prediction/experiments/b73_targets/common.py')
    targets=common[common.index('def later_values('):common.index('def window_feats(')]
    dataset=readtext('ces_prediction/dataset.py')
    masks=dataset[dataset.index('    def _nan_out_ti_spikes'):dataset.index('    def _build_column_slices')]
    b73=V.read(ROOT/'docs/paper/b73_targets.json')
    # Current-line aggregate tree comparisons only; legacy/as-built rows excluded.
    variants={}
    for key in ['V1|F','V1|N1','V1|L3','V1|L5','V1|L10']:
        row=b73['increments'][key]
        variants[key]={k:row[k] for k in ['rows','discharges','clusters']}
        variants[key]['tree_fast_increment']=row['cells']['TX/T0']
    b105=V.read(ROOT/'docs/paper/b105_merit.json')
    cells={}
    for key,row in b105['cells'].items():
        if '/test/D10/F/TCN35_over_incumbent_outerfit' in key or '/val/D10/F/TCN35_over_fair_specialized' in key:
            stats=row['statistics']
            cells[key]={'rows':row['rows'],'point':stats['point'],
              'cluster_ci95':stats['cluster']['ci95'],'excludes_zero':stats['cluster']['excludes_zero']}
    reg105=readtext('ces_prediction/experiments/PREREGISTRATION_B105.md')
    gate=reg105[reg105.index('## Fixed allocation gate'):reg105.index('\n## ',reg105.index('## Fixed allocation gate')+5)]
    gate105=b105['gate']
    data={'author_objective':'Current reported CES carbon-impurity temperature on one provider-selected KSTAR2022 pedestal-top chord, from past CES+current/past BES/ECEI. Useful estimator vs hold/causalGP/offline interpolation;4validation designs equally;10ms computation. Model architecture and operational definitions may be criticized; do not defend them as fixed scientific truths.',
      'actual_manuscript':manuscript,
      'current_protocol_record_not_authoritative_scientific_proof':readtext('PAPER_DESIGN.md'),
      'existing_target_registration_relevant_sections':registration,
      'current_implementation_target_definition':targets,
      'current_implementation_cut_and_repeat_masks':masks,
      'existing_target_variant_study':variants,
      'registration_timing':'B73 targets registered2026-11-20, but same campaign had been seen; not untouched confirmatory work. New B105 TCN screen registered2026-11-24, so L5 predates this particular TCN. Registration chronology does not establish physical validity or rule out earlier data-informed choices.',
      'B105_new_result_scope':b105['limits'],
      'B105_current_target_comparisons':cells,
      'B105_gate_definition':gate,
      'B105_gate_unit_aggregates':gate105['units'],
      'B105_interpretation':'All4 selected unit D10/F TCN-vs-incumbent TEST skills negative with cluster CI entirelynegative. VALunit mean gate passed. One unit/design,one initialization; not architecture-universal finding. No further training allocated. Future changes are TEST-informed exploratory.'}
    # Only report aggregates, not individual observation records or identities.
    for key in ['current_protocol_record_not_authoritative_scientific_proof','existing_target_registration_relevant_sections','B105_gate_definition']:
        data[key]=re.sub(r'(?i)shot(?:s)?\s+[0-9][0-9,\\\-– ]*','discharges [identifiers redacted]',data[key])
    return data


TASK='''The AUTHOR is challenging our scientific definitions, not asking for another copyedit. They say L5 appears constantly, too many quantities/rules seem arbitrarily invented, and the latest TCN cannot even match the existing GP+boosting estimator on current-report D10/F. Be a skeptical methodology reviewer. Do not mechanically validate a definition merely because it is registered, implemented consistently, or already used in prior manuscript. Do not be biased by any previous reviews (none supplied). Preserve the practical author objective while challenging target/metric/protocol choices themselves. Do not invent literature support, acceptance probabilities or experiments/results.

Answer the hard questions: Is it scientifically legitimate to define a following-five-grid median with >=3values as L5? Does that estimate the current frame or a different future-window estimand? What evidence would be required to call it noise reduction/current level rather than future target? Does input causality make its label a nowcast? Does excluding interpolators from L5 fix an unfair benchmark or merely define a different question? Are 'onlyFcanbecompared'/'frame scatter cannot be predicted' scientific facts or overstatements? Assess nonstationarity, physical ELM transients, missingness/minimumcount/censoring-induced target shifts, asymmetric temporal support, common-support selection, target-dependent training, and using future median as validation truth.

Audit ALL other custom definitions that materially matter: temperature cutoff3keV and exact-repeat removal (cannot infer invalidity solely from magnitude/equality), gap0.5s, clusters byshotgap<=3, choice/history window/kernel/normalization and anchorfeatures, matchedrows, assumed availability schedules and D650median/zeroanalysis time, stalecost/recoveredfraction ratios, cluster CIs conditioning on fits/selection, reused TEST. Categorize established convention vs legitimate study-specific operational definition vs choice needing evidence/sensitivity vs unjustified physical interpretation. A custom definition need not have a prior paper to be legitimate; distinguish formal definition, empirical support and physical interpretation. Minimum necessary set, which should be shortened or moved to supplement, which should not support the main claim.

Existing B73 already compared N1/L3/L5/L10 underV1 with trees; acknowledge this evidence but do not call it independent physical validation, nor imply full4design sensitivity or pure window-length ablation (different means/medians/missingness/support). Do not request the exact same completed comparison as though absent.

TCN B105: confirm negative sign convention and limited scope. Does VAL aggregate14cells gate<=1.05 guarantee per-condition adequacy? It does not: check supplied definition and D10/FVAL cells. Is the aggregate goal aligned with current-report objective, or can stale/auxiliary-target improvement compensate deterioration of current-F? Does failure justify newL5/target/gate changes to rescueTCN? Should manuscript retain baseline method, present TCN as comparison, or change actual scientific question? Distinguish recommendation from currently demonstrated fact. No new run, threshold, outcome or cause identification; proposed changes require separate preregistration and remain TEST-informed.

Return strictJSON {overall_verdict_ko,most_important_answer_ko,L5_assessment:{formal_validity,actual_estimand,current_report_fit,physical_interpretations_supported_or_not,existing_sensitivity_scope,recommendation},definition_register:[{definition,category,scientific_status,actual_evidence,what_is_missing,recommendation:keep|qualify|demote|remove|sensitivity_needed,why}],TCN_and_gate_assessment:{actual_result,gate_meaning,goal_alignment,what_not_concluded,next_decision},minimum_maintext_design:[...],statements_to_retract_or_rephrase:[{old_claim,new_claim,reason}],necessary_experiments_only:[{question,already_done_related_work,what_new_comparison_would_answer,priority,needs_new_data_or_GPU,preregistration_note}],author_facing_feedback_ko}. <=2500visible words, concise enough to COMPLETE. Do not rewrite the whole paper. Verdict must be candid even if current framing is methodologically weak. DATA:\n'''


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if '--challenge' in sys.argv:
        review=V.read(OUT/'04_opus_definition_validity.json')
        evidence=packet()
        reduced={k:evidence[k]for k in ['author_objective','current_implementation_target_definition','existing_target_variant_study','registration_timing','B105_new_result_scope','B105_current_target_comparisons','B105_gate_definition','B105_gate_unit_aggregates']}
        b105=V.read(ROOT/'docs/paper/b105_merit.json')
        incumbent=[r['statistics']['point']for k,r in b105['cells'].items()if '/test/'in k and k.endswith('/TCN35_over_incumbent_outerfit')]
        reduced['existing_B105_incumbent_TEST_summary']={'cells':len(incumbent),
          'negative_point_estimates':sum(x<0 for x in incumbent),
          'nonnegative_point_estimates':sum(x>=0 for x in incumbent),
          'note':'Counts of already published comparisons only; does not assert all intervals exclude zero or prove architecture-wide inferiority.'}
        content='''Critically check the supplied Opus methodology feedback, not merely agree. Author objects to too many arbitrary definitions and L5 dominating current-report nowcasting; latest TCN underperforms incumbent in D10/F. Identify where reviewer critiques are correct, too broad, or ignore existing evidence. No new fits, acceptance grades, invented source support or rescue tuning. Formal operational definitions are allowed but physical interpretation must be justified. Existing B73 target sensitivity and B105 aggregate gate are supplied. Distinguish lack of physical truth, input/target leakage, different estimands and mere narrative repetition. Recommend an actionable MINIMUM paper design preserving honest current-report objective and all4validation designs; do not delete alreadycomputed results to hide negatives. If moving L5 to auxiliary/supplement explain what main current-F results remain and what claims must change. Any new sensitivity test is future registered/exploratory, not run. Return strictJSON {agreements:[...],corrections_to_opus:[{claim,correction,evidence}],recommended_decision_ko,minimum_changes_now:[...],future_checks_only:[...],author_message_ko}. <=1200visible words. DATA:\n'''
        result=V.call('05_fable_definition_challenge','claude-fable-5-2',[V.block(content+json.dumps({'opus_feedback':review,'evidence':reduced},ensure_ascii=False))],16000,output_dir=OUT)
    else:
        data=packet()
        V.save(OUT/'evidence_manifest.json',{'source_files':['PAPER_DESIGN.md','docs/paper/b73_targets.json','docs/paper/b105_merit.json','ces_prediction/experiments/PREREGISTRATION_B73.md','ces_prediction/experiments/PREREGISTRATION_B105.md','ces_prediction/experiments/b73_targets/common.py','ces_prediction/dataset.py'],
            'source_hashes':{p:V.D.sha(ROOT/p)for p in ['PAPER_DESIGN.md','docs/paper/b73_targets.json','docs/paper/b105_merit.json']},
            'individual_observations_supplied':False,'new_runs':0,'funding':'Unused part of existing USD5 verification cap; shared api_usage ledger in ../figure_verification/'})
        result=V.call('04_opus_definition_validity','claude-opus-5-5',[V.block(TASK+json.dumps(data,ensure_ascii=False))],16000,output_dir=OUT)
    print('DEFINITION FEEDBACK COMPLETE; NO SCIENTIFIC PROTOCOL CHANGED',flush=True)

if __name__=='__main__':main()
