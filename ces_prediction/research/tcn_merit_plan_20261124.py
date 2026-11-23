"""Bounded official Fable planning; published aggregates/metadata only, no fits.

Consumer: docs/paper/tcn_merit_plan_20261124.md and HANDOVER.md.
Raw API attempts/usage stay in ignored data/.tcn_merit_plan_20261124/.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'data/.tcn_merit_plan_20261124'
sys.path.insert(0, str(ROOT / 'ces_prediction/experiments/b85_unified'))
import claude_review85 as API


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def evidence():
    report_path = ROOT / 'docs/paper/b94_methods.json'
    z = json.loads(report_path.read_text(encoding='utf8'))
    cells = []
    for scope, sr in z['scopes'].items():
        for design, dr in sr['designs'].items():
            for key, c in dr['cells'].items():
                pairs = {}
                for learner, p in c['supplementary_tree_pairs'].items():
                    pairs[learner] = {k: p[k] for k in
                        ('point', 'cluster', 'excludes_zero', 'rows', 'excluded_from_cell')}
                cells.append({'scope': scope, 'design': design, 'cell': key,
                              'rows': c['rows'], 'tcn_over_tree': pairs})
    inv = {}
    for name in ('.b73/trees', '.b75/fits', '.b78/fits', '.b84/fits',
                 '.b94/predictions', '.b94/confirm', '.b104/confirm'):
        p = ROOT / 'data' / name
        inv[name] = {'exists': p.exists(), 'npz_count': len(list(p.glob('*.npz'))),
                     'checkpoint_count': len(list(p.glob('*/best.pt')))}
    result = {'source_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
              'published_cells': cells, 'latency': z['latency'],
              'artifact_counts_only': inv,
              'limitations': z['limitations']}
    save(CACHE / 'evidence.json', result)
    return result


TASK = '''You are the methodological lead preparing TOMORROW'S MINIMAL GPU plan.
The author is frustrated: we trained TCN and then extended its budget before clearly
answering whether it is useful relative to the existing two-stage estimator. Do not
propose another broad search. Produce a concrete executable scientific decision plan
in Korean, <=1800 Korean words. No fabricated results, novelty or acceptance claims.
No GPU fit is authorized today; this call is planning only. All proposals informed by
previous TEST results are exploratory; no new independent confirmation claims.

CORRECTION: a previous working-agent reply wrongly said no direct comparison exists.
Published B94 JSON ALREADY contains paired TCN vs condition-specific GP+boosted-tree
comparisons, on matching scored rows, under supplementary_tree_pairs. Markdown only
prints marginal metrics, so these paired results were overlooked. Use them FIRST.
Do not conflate fast-input increment TCNfast vs TCNhistory with TCN vs incumbent.
Two different 'two-stage' estimators must be explicitly distinguished:
(A) incumbent manuscript GP on latest16 visible CES values + HistGradientBoosting
residual, with history/fast/anchor features; condition-specific fits in B73/B75/B78.
(B) historical causal LSTM + a second-stage correction (legacy B64/B70), available
at D10 only in current lineage; no apples-to-apples delayed replay is established.
We cannot assume which of these the author meant; handle A and B explicitly in the
plan without blocking preparation. Never restore removed earlier research or numbers.

Scientific purpose fixed: pedestal-top CES reported Ti nowcasting at10ms; best useful
estimator vs hold/causal GP/offline interpolation. F=current report target; L5=median
next5 reports (offline label, not future physical truth). D10/D50/... denote minimum
history source age, not forecast horizon. Four validation designs V1 random files,
V2 clusters,V3 rolling,V4 finalblock; none primary. FULL15 scope=one initialization;
FOUR units=3 initializations on one prespecified unit of each design; never pool them.
B94 TCN trained one fixed model per split with random segmentwise availability among
D10,D50,D100,D300,D650,D1000,S; K10 held out from train/VAL. Best checkpoint selected by
equal mean validation loss across7rules x2heads. AdamW/loss/dropout .1/batch16/40epochs/
patience6; many1200s time limits. Neural inner VAL reserves15% of outerfit files;
old tree used all outerfit files, not identical fit budgets. Tree specialized by
rule+target vs unified TCN. Input features also differ: summaries vs full channels.
Negative comparison is conditional on training/info/selection differences.
B104 fixed one temporal unit,3inits fresh FAST/HISTORY paths, original frozen loop,
only1200->3600s cap changed; allFAST SHORT/LONG weights exactly same, onlyoneHISTORY
changed. It did NOT recover fast skill. Stop 'just train longer' as default.

Evidence below is anonymized published aggregate paired results only, no raw rows,
observations, shot identifiers, identity or secrets. Existing pairing skill is
1-MSE(TCNfast)/MSE(tree) positive=lowerTCNerror. Point,95%clusterinterval,zero exclusion;
scope cluster/init resampling limitations persist. No selecting favorable endpoints
after TEST. TCN latency is CPU single-thread forward-step only, not acquisition,
not a latency comparison against tree. Can't claim measured simpler/faster yet.

Deliver: (1) honest existing merit table across ALL4designs and relevant scenarios,
refer to exact source keys; distinguish potential interpretation vs known facts.
(2) CPU/no-fit-first work to reuse existing checkpoints and predictions; identify
what is ALREADY answered and what is missing. No need GPU for reading published JSON.
(3) bounded tomorrowGPU priority, ideally zero or <=4 new fits first-stage, exact
arms, fixed units/seeds selected by prior protocol not TEST, comparators, targets,
fit/VAL parity, availability, deployment-frozen vs condition-refit controls, fairness
and source-time checks, uncertainty/multiplicity and go/no-go thresholds. If further
seeds/all4designs cannot fit <=4, explicitly separate screening from final evidence.
(4) concrete training recipe IF warranted, no huge architecture menu; consider
whether GP/hold residualization or anchor differences are justified from TRAIN/VAL,
not a promise. Don't switch targets/objective to rescue test scores silently.
(5) a single decision tree with hard stop, maximumwork budget and no automatic
expansion. Model merit can be accuracy, same accuracy with measured operational
benefit, or conditional scenario benefit; not architecture novelty by itself.
(6) distinguish implementable plan vs code already ready; don't invent commands.
Avoid spending tomorrow's GPU on a question CPU can answer tonight. New fits require
committed preregistration+implementation+independent checks BEFORE dispatch. No new
fits today, no pretending the plan is a completed preregistration.
'''


def call(stage):
    CACHE.mkdir(parents=True, exist_ok=True)
    req = CACHE / (stage + '.request.json')
    res = CACHE / (stage + '.response.json')
    if res.exists():
        r = json.loads(res.read_text(encoding='utf8'))['response']
    else:
        if req.exists():
            raise RuntimeError('Prior attempt exists; inspect before any paid replay')
        if len(list(CACHE.glob('*.request.json'))) >= 3:
            raise RuntimeError('Three-attempt bound reached, including HTTP400')
        e = evidence()
        if stage.startswith('challenge'):
            e = dict(e)
            e['published_cells'] = [c for c in e['published_cells'] if c['cell'] in
                                    ('D10/F', 'D650/L5', 'S/L5', 'K10/L5')]
        prompt = TASK + '\nAGGREGATE EVIDENCE:\n' + json.dumps(e)
        if stage.startswith('challenge'):
            prompt += '''\nINDEPENDENT SOURCE CHECK AFTER FIRST REQUEST:
B84 ALREADY implemented single-mixture GP+tree: TXall/T0all, and raw-window
TXraw/T0raw on L5; frozen D10-applied TXa/T0a also exist in B75. Therefore
'unified scenarios' is NOT an untested operational merit unique to TCN.
Relevant verified source: docs/paper/b84_onefit.md and B84 code. B84 has the
same7-rule mixture and K10 held out; each outer split fits once for TXall.
B94 omitted TXall from paired output, despite an older B85 helper recognizing
that baseline. B94 already has paired TCN vs condition-specialized TREE_TX;
new comparisons against TXall/TXa/legacy LSTM require preregistration BEFORE
rescoring/bootstrap, explicit common-support+manifest checks, and disclosure
that prior TEST was already inspected. Use full frozen B84 results, never
cherry-pick a bad D10-applied comparator to manufacture scenario merit.
Do not repeat the original wrong claim 'direct comparison absent'.
Important implementation check: current TCN slices input to22 channels,
omitting the13 anchor-fast channels used by learned anchor candidate. Its
fast block is15slots, history/timing7slots; incumbent TREE_TX has anchor
summaries. Architecture-only superiority cannot be inferred from unequal
information/fit sets. Full incumbent is a valid deployment challenger;
matched-information explanatory comparison is a separate question.
Published old LSTM+correction B70 stage2 files exist, but row/manifest parity
with new TCN folds and causal scenario replay is NOT verified. Don't quote
obsolete pre-cleanup results or run old scripts by default.
'''
            first = json.loads((CACHE / 'design.response.json').read_text(encoding='utf8'))['response']
            first_text = '\n'.join(b['text'] for b in first['content'] if b.get('type') == 'text')
            prompt += '\nFIRST FABLE PARTIAL PROPOSAL (max_tokens; incomplete, NOT accepted advice):\n' + first_text
            prompt += '''\nNow adversarially correct that proposal instead of repeating it.
Prioritize whether new GPU training is actually justified; prevent unsupported claims
from selected favorable cells. Give one final minimal recommendation and corrections.
Explicitly identify unresolved estimator identity A/B and missing scientific contract.
Use <=1000 Korean words. Do not invent numeric outcomes or claim code is runnable.'''
        payload = {'model': 'claude-fable-5-2', 'max_tokens': 9500,
                   'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'medium'},
                   'messages': [{'role': 'user', 'content': prompt}]}
        if stage == 'challenge_final':
            payload['max_tokens'] = 15000
            payload['output_config'] = {'effort': 'low'}
        save(req, payload)
        started = time.time()
        try:
            r = API.request('/messages', API.credential(), payload)
        except Exception as exc:
            save(CACHE / (stage + '.error.json'), {'error': str(exc), 'seconds': time.time() - started})
            raise
        save(res, {'response': r, 'seconds': time.time() - started})
    content = '\n\n'.join(b['text'] for b in r.get('content', []) if b.get('type') == 'text')
    if r.get('stop_reason') != 'end_turn' or not content:
        raise RuntimeError('Incomplete API response; preserved, not accepted')
    (CACHE / (stage + '.txt')).write_text(content, encoding='utf8')
    print(json.dumps({'stage': stage, 'model': r.get('model'), 'usage': r.get('usage'),
                      'stop_reason': r.get('stop_reason')}, ensure_ascii=True))


def render():
    lines = ['# Fable: TCN merit planning review', '',
             'Generated by `ces_prediction/research/tcn_merit_plan_20261124.py render`.',
             'Planning advice only. No scientific fits or new error scoring were performed.',
             'Official Anthropic API, model `claude-fable-5-2`.',
             'Only redacted published aggregates and protocol/code summaries were transmitted.',
             'No credentials, identity, raw observations or private reasoning were sent.',
             'Every attempt remains in ignored `data/.tcn_merit_plan_20261124/`.', '',
             '| Attempt | Outcome | Input tokens | Output tokens |',
             '|---|---|---|---|']
    for name in ('design', 'challenge', 'challenge_final'):
        p = CACHE / (name + '.response.json')
        if p.exists():
            r = json.loads(p.read_text(encoding='utf8'))['response']
            u = r.get('usage', {})
            lines.append(f"| {name} | {r.get('stop_reason')} | {u.get('input_tokens')} | {u.get('output_tokens')} |")
        else:
            assert (CACHE / (name + '.error.json')).exists()
            lines.append(f'| {name} | HTTP400 rejected configuration; no usage returned | unknown | unknown |')
    lines += ['', 'The first draft reached its output limit and was not accepted as completed advice.',
              'The HTTP400 attempt used an unsupported request configuration; it is preserved, not hidden.',
              'The final corrected request completed with end_turn. No further calls are queued.',
              'The final request included the four fixed overview cells across both scopes and all designs;',
              'the first request included all136 cells. See host corrections in the final direction document.',
              'Do not treat this prose as a verified result or a ready-to-run registration.', '',
              '## Completed Fable response (verbatim; host corrections take precedence)', '',
              (CACHE / 'challenge_final.txt').read_text(encoding='utf8'), '',
              '## Host disposition', '',
              'See `tcn_merit_plan_20261124.md`, especially the final correction list.',
              'The source-derived full matrix is `tcn_merit_existing_20261124.md`.',
              'No uncompleted draft, model thinking or local credential material is published.']
    (ROOT / 'docs/paper/tcn_merit_fable_20261124.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('Rendered completed Fable advice and all-attempt usage ledger')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('evidence', 'design', 'challenge', 'challenge_final', 'render'))
    args = ap.parse_args()
    if args.stage == 'evidence':
        e = evidence()
        print(json.dumps({'cells': len(e['published_cells']), 'artifacts': e['artifact_counts_only']}))
    elif args.stage == 'render':
        render()
    else:
        call(args.stage)
