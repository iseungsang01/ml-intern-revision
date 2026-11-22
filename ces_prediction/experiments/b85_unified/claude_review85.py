"""One redacted Claude methodological review for B85; never prints credentials.

The author explicitly authorized reading the sibling repository credential and
using Claude for this review. Only PROMPT below is transmitted, not repo files.
Run from repository root with py. Reuses a completed response unless --refresh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / 'data/.b85/claude_review.json'
REPORT = ROOT / 'docs/paper/b85_claude_design_review.md'
KEY_FILE = ROOT.parent / '2026_cvpr/.env'
BASE = 'https://api.anthropic.com/v1'
VERSION = '2023-06-01'
PROMPT = """Review a proposed preregistered experiment as a skeptical ML/time-series methods collaborator. Give blocking flaws, exact leakage tests, and feasible fixes BEFORE implementation/training. Do not estimate journal acceptance, claim novelty, or invent results. This summary has no author identity, shot IDs, raw data, or experimental performance results.

Task: one causal estimator of an edge diagnostic's reported temperature from its past measurements and current fast diagnostic channels. Existing condition-specific regression is a comparator. We want one model without condition-specific refitting or post-fit correction.

Data: real previously used campaign data, exploratory rather than untouched confirmation. Full timestamp grid includes unlabeled context. Supervision uses a measured current-frame scalar F and L5, median of the next five nominal 10-ms frames with at least three measured values in the same segment. L5 is a future-derived offline label only, not independent physical truth. Labels appear only in loss/scoring.

Model: two-layer 160-unit unidirectional LSTM, layernorm and 64-unit GELU MLP, two scalar heads F/L5 sharing a state. Input slots: fast channels; log1p elapsed time; last-visible temperature normalized by training-file-only mean/std; log1p age; has-visible flag. Unused channels zeroed. Identical-parameter history-only twin zeros all fast channels; paired initializations/augmentation. No scenario ID, self-prediction feedback, target-row measurement, or future normalization. State resets at original segment boundaries (time gap at least 0.5 seconds). Train/test inference use whole segments.

Availability: one constant rule per segment; a new rule is sampled uniformly for each segment each training epoch, deterministic and paired for twins. Rules: last measurement at least 10,50,100,300,650,1000 ms old; or only values preceding the most recent ALREADY OBSERVED 15-45-ms gap in that segment (before any such gap: no history). Every visible source index must be strictly before the current row, with 3-ms timestamp tolerance. Availability limit is nondecreasing through a fixed-rule segment. Held-out cadence rule (measurements at every tenth nominal frame, past only) is excluded from both training and validation. At test each entire segment is replayed separately under each rule with state reset and unchanged weights. No switching rules mid-state.

Splits: four designs reported side by side, none primary: random whole-file folds, clustered-file folds, rolling-origin, final chronological block. Three initialization seeds, full and twin models. Within each outer fit set reserve 15% files for validation: seeded random for non-temporal designs, latest files for temporal designs; training normalization excludes validation/test. Exact manifests saved and disjoint. Comparators used all original fit files, so network has fewer training files; disclose rather than claiming training budgets identical.

Optimization fixed before readout: AdamW 0.001, weight decay 0.0001, batch16 full segments, clip1, dropout0.1, cap80, plateau scheduler factor0.5 patience2; stop after six non-improving epochs. Average masked MSE over the two heads, plus 0.1 negative-temperature ReLU penalty in normalized units. Select best validation mean MSE with equal weights for seven training rules and two heads. Validation deterministically replays every rule. Held-out cadence never selects settings. One-epoch real-data CUDA smoke is separate, plumbing only.

Checks before test errors: match cached anchors/ages/flags of independently implemented condition-specific pipeline; frozen-stat mutation of all values not yet visible at a sampled scored time by removal and large additive offset must leave the entire input prefix unchanged; future fast-input mutation leaves prefix unchanged; current target excluded; no cross-segment carry; deterministic augmentation; held-out cadence excluded; normalization provenance; disjoint manifests; equal labels; finite forward/backward on real CUDA batch; prediction invariant under mutation of the test label arrays alone. Any failed check aborts.

Evaluation: exact same rows as stored comparator predictions, intersection masks explicit and baseline inventory fixed before scoring. RMSE, MAE, median absolute error, share within 10% on targets above fixed floor, out-of-range predictions, age bins and cold-start rows. Conditional 95% percentile bootstrap: initialization sampled first (paired twins), then shot clusters; discharge resampling beside it, 20000 draws, deterministic per-cell seeds. Fixed fitted weights, no refitting; no multiplicity adjustment and small cluster counts disclosed. Each initialization and its mean reported, no post-hoc ensemble. The model replaces the existing estimator only if fresh-history and delayed/schedule errors meet a prespecified 5% MSE noninferiority margin and held-out cadence improves over its condition-specific history-only comparator in every design; fast-input increment must also persist in time-ordered designs. Otherwise retain comparator and report failures. CPU single-step inference latency is measured separately from preprocessing and actual diagnostic acquisition latency.

Return: (1) blocking design/implementation ambiguities, (2) most important additional tests, (3) minimal pre-readout clarifications, (4) which interpretations remain unsupported even if all thresholds pass. Prioritize a short, actionable response. Distinguish actual leakage from limitations and avoid proposing a large architecture search."""


def credential():
    if not KEY_FILE.is_file():
        raise RuntimeError('credential file missing')
    for line in KEY_FILE.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line.startswith('export '):
            line = line[7:].lstrip()
        name, sep, value = line.partition('=')
        if sep and name.strip() == 'ANTHROPIC_API_KEY':
            value = value.strip().strip(chr(34)).strip(chr(39))
            if value:
                return value
    raise RuntimeError('ANTHROPIC_API_KEY missing or empty')


def request(path, key, payload=None):
    headers = {'x-api-key': key, 'anthropic-version': VERSION,
               'content-type': 'application/json'}
    req = urllib.request.Request(BASE + path,
        data=None if payload is None else json.dumps(payload).encode('utf-8'),
        headers=headers, method='GET' if payload is None else 'POST')
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Do not echo response bodies, request headers or URLs from exceptions.
        raise RuntimeError(f'Claude API HTTP {exc.code}') from None
    except urllib.error.URLError:
        raise RuntimeError('Claude API network connection failed') from None


def render(record):
    usage = json.dumps(record['usage'], ensure_ascii=True)
    text = ('# B.85 Claude methodological design review\n\n'
        'The author explicitly authorized this external review. Only the redacted methodological prompt below '
        'was sent to the official Anthropic API; no repository file, identity, raw data, credential, or '
        'unpublished numerical result was transmitted. This is not a manuscript acceptance grade.\n\n'
        f"Model returned: `{record['model']}`. API version: `{VERSION}`. Usage: `{usage}`. "
        f"Stop reason: `{record['stop_reason']}`.\n\n"
        '## Exact transmitted prompt\n\n' + PROMPT + '\n\n## Response\n\n' + record['response'] + '\n')
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='explicitly request a further paid call')
    args = ap.parse_args()
    if CACHE.exists() and not args.refresh:
        record = json.loads(CACHE.read_text(encoding='utf-8'))
        render(record)
        print('Reused completed redacted design review')
        return
    key = credential()
    catalog = request('/models?limit=1000', key)
    ids = [m['id'] for m in catalog.get('data', [])]
    preferred = [m for m in ids if 'opus' in m.lower()]
    if not preferred:
        preferred = [m for m in ids if 'fable' in m.lower()]
    if not preferred:
        raise RuntimeError('No Opus or Fable model available in official model catalog')
    model = 'claude-opus-4-6' if 'claude-opus-4-6' in preferred else preferred[0]
    response = request('/messages', key, {'model': model, 'max_tokens': 5000, 'thinking': {'type': 'disabled'},
        'messages': [{'role': 'user', 'content': PROMPT}]})
    record = {'prompt': PROMPT, 'model': response.get('model', model),
        'usage': response.get('usage', {}), 'stop_reason': response.get('stop_reason'),
        'response': '\n\n'.join(b['text'] for b in response.get('content', []) if b.get('type') == 'text')}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    if not record['response']:
        print(json.dumps({'model':record['model'], 'usage':record['usage'],
            'stop_reason':record['stop_reason'], 'response_keys':list(response),
            'content_types':[b.get('type') for b in response.get('content',[])]}))
        raise RuntimeError('Claude API returned no text; safe metadata saved')
    render(record)
    print('Review completed with model ' + record['model'])
    print('Usage ' + json.dumps(record['usage'], ensure_ascii=True))
    print('Report saved; credentials were not logged')


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1)
