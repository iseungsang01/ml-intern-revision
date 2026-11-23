"""Resolve specific host/adversary disagreements; no fits or raw observations."""
import json
import fable_deep_review as D


def main():
    previous = json.loads((D.OUT / 'status.json').read_text(encoding='utf8'))
    assert previous['complete']
    result, binding = D.E.gate()
    assert result
    names = [
        '11_final_korean_decision.md', '10_adversarial_program_review.md',
        'EXISTING_WORK.md', 'PRIMARY_CHECKS.md', 'METHOD_CHECKS.md',
    ]
    content = [D.textblock(D.CONTEXT)]
    content += [D.textblock(name + '\n' + (D.OUT / name).read_text(encoding='utf8'))
                for name in names]
    content.append(D.textblock('''NEW bounded corrective task, not a repeat of the previous review.
Provide a complete Korean actionable final memo <=1800words, superseding the prior final memo.
Resolve concrete source/logic disagreements:
1. Adversary E6 is incorrect: TS2Vec official code was acquired later, host PRIMARY_CHECKS documents the pinned commit and encode path. Do not downgrade a checked source merely because the earlier dossier did not have it. Preserve causal sliding limitations.
2. Paired metric sign is now independently checked against actual readout and bootstrap sources in METHOD_CHECKS. No need to indefinitely demand this same check. The aggregate packet lacks stored row arrays; absence from that packet does NOT prove the repo has no paired baseline evidence. A future common-support readout is new work to register, not permission to inspect TEST now.
3. Apply METHOD_CHECKS exact nonlinear-offset and delayed-state/covariance corrections. Delayed formula and fresh-anchor shrinkage cannot establish fast information by construction. Same slots is not same information. A failed finite model does not establish that data carry no information.
4. Empirical report-level coverage is evaluable, but no automatic distribution-free guarantee with dependent shots. NLL is a legitimate predictive score despite temporal dependence; uncertainty/generalization needs separate treatment. Faithful identity requires actual implementation controls, not just the loss name.
5. 'Only future fast is nonduplicated privileged information' is too absolute: architectural use, cross-fitting, masking and training mechanism differ from a future CES label/B87 reference. State the exact new contrast rather than assuming all CES-informed teachers redundant. Do not assume alpha replay exact for general nonlinear filters.
6. No automatic inferiority/equivalence from non-significance. An inconclusive result may justify stopping for cost, but does not falsify information availability or establish history-only bookkeeping. Distinguish the tested hypothesis from model-class limitations. No unregistered more-than-one-design promotion threshold. Treat all4 designs equally with predeclared multiplicity and margins.
7. Preserve adverse B98 V3 internalVAL evidence and B104 budget uncertainty. They constrain, but do not veto all estimator investigation. Don't turn the agenda back into only audits. Novelty should be a specific task/evidence distinction, not standard model renaming or an unsubstantiated no-prior-art statement.
Deliver: key corrected literature findings with primary links and locators;5 concrete scientific objectives; exactly3 ranked bounded substantive programs (with hypotheses, standard vs actually new empirical question, simple then conditional learned model, same-input/support/budget comparisons, information/timing/target contract, TRAIN/VAL selection, exploratory reusedTEST and independent confirmation, positive/negative/inconclusive outcomes, compute uncertainty/stop);2 deferred options; one next feasible action. State unanswered source-access gaps. Do not start work, grade acceptance, invent numerical results, or spend remaining budget merely to spend it.'''))
    D.call('12_corrected_korean_plan', content, max_tokens=49152, effort='medium')
    D.E.gate()
    with D.LOCK:
        ledger = D.ledger()
    previous.update({
        'final_report': '12_corrected_korean_plan.md',
        'superseded_final_report': '11_final_korean_decision.md',
        'estimated_list_price_usd': ledger['estimated_spent_usd'],
        'paid_calls': len(ledger['calls']),
        'final_gate_binding': binding,
    })
    D.save(D.OUT / 'status.json', previous)
    print('CORRECTED PLAN COMPLETE USD', ledger['estimated_spent_usd'], flush=True)


if __name__ == '__main__':
    main()
