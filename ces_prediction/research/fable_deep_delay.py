"""Close the delayed-measurement primary-source gap within the same USD30 cap."""
import json
import fable_deep_review as D


def main():
    previous = json.loads((D.OUT / 'status.json').read_text(encoding='utf8'))
    assert previous['complete']
    r, _ = D.E.gate()
    assert r
    content = [D.textblock(D.CONTEXT),
               D.doc(D.SOURCE / 'delayed_state.pdf',
                     'Mina et al Remarks on stochastic cloning and delayed-state filtering arXiv2508.21260v2 11May2026 full29pages'),
               D.textblock((D.OUT / '12_corrected_korean_plan.md').read_text(encoding='utf8')),
               D.textblock('''Host primary publisher check (paraphrase, not a quotation): Koch et al2020, Relative multiplicative extended Kalman filter for observable GPS-denied navigation, https://doi.org/10.1177/0278364920903094 section4.3 describes retaining time-ordered inputs, measurements, states and error covariances in a finite buffer; restoring the measurement-time state when a delayed observation arrives; applying that update; then replaying intervening inputs AND observations to current time. Observations older than the buffer are discarded. Section8/Fig13 describes the hardware manifestation. This is a robotics implementation, not CES evidence and not a proof of exact nonlinear Bayesian filtering. Host read the publisher full HTML, not a summary site.
NEW NARROW TASK: Korean <=1000words. Close the exact P2 primary-literature gap; do not redo all prior tasks. Read full supplied MinaPDF and discriminate correlated relative-state measurements/stochastic cloning from our delayed absolute scalar CES and from purely missing/held reports. Give at most4 precise page/equation locators and3 restrained conclusions, including assumptions behind any claimed equivalence. Explain what is standard, whether P2 is a valid experiment versus an obvious repeated-measurement strawman, what the strongest fair comparator must be, and if P2 should be demoted/merged with P1 rather than called a separate scientific objective. State whether P1 or P3 priority changes. An every-row latent pseudo-observation trained end-to-end (B90) is NOT automatically the same as repeatedly treating the same CES scalar as independent with a fixed Gaussian likelihood: do not imply that it is. No live arrival logs; distinguish constructed source-age schedules from random/out-of-order arrivals. No proof of physical information or algorithmic novelty from successful replay. Define paired loss difference negative-is-better, and an upper bound below -delta as the superiority direction (previous Korean wording was ambiguous). End with revised3-program ordering, optionally promoting a previously deferred representation or operational option with its bounded discriminating experiment if P2 merges. No fits, TEST readout, acceptance grade or raw data. Primary sources only. No generic novelty claims.''')]
    D.call('13_delayed_measurement_source_addendum', content,
           max_tokens=24576, effort='medium')
    D.E.gate()
    D.source_manifest()
    with D.LOCK:
        ledger = D.ledger()
    previous.update({'source_addendum': '13_delayed_measurement_source_addendum.md',
                     'estimated_list_price_usd': ledger['estimated_spent_usd'],
                     'paid_calls': len(ledger['calls'])})
    D.save(D.OUT / 'status.json', previous)
    print('DELAY SOURCE COMPLETE USD', ledger['estimated_spent_usd'], flush=True)


if __name__ == '__main__':
    main()
