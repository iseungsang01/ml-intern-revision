"""Build source catalogue and free-token-count budget before paid corpus review."""
import concurrent.futures
import json
from pathlib import Path

import fable_deep_review as D
import introduction_corpus_inventory as I


def catalogue():
    inventory = json.loads((I.OUT / 'corpus_inventory.json').read_text(encoding='utf8'))
    known = {}
    for p in sorted(I.SOURCE.glob('INDEX*.md')):
        for line in p.read_text(encoding='utf8').splitlines():
            if not line.startswith('|'):
                continue
            fields = [v.strip() for v in line.split('|')[1:-1]]
            if len(fields) == 7 and fields[0] not in ['Key', 'key']:
                known[fields[0]] = {'citation': fields[1], 'identifier': fields[2]}
    seen, rows = set(), []
    for row in inventory['papers']:
        key = row['key']
        record = {**row, **known.get(key, {})}
        record['duplicate_of_prior'] = row['normalized_text_sha256'] in seen
        seen.add(row['normalized_text_sha256'])
        # The institutional user guide has not been cleared as a public paper.
        record['external_review_allowed'] = key != 'KFE2026DiagnosticsDataUserGuide'
        record['scope_note'] = ('Local-only institutional guide pending public-source verification'
                                if not record['external_review_allowed'] else
                                'Published literature supplied by author; bibliographic identities only')
        rows.append(record)
    I.save(I.OUT / 'catalogue.json', {'inventory_only': True, 'papers': rows})
    return rows


def count(index, body):
    target = I.CACHE / 'counts' / (str(index)+'.json')
    if target.exists():
        return json.loads(target.read_text())
    request = {'model': 'claude-opus-5-5',
               'messages': [{'role': 'user', 'content': body}]}
    counted = D.api('/messages/count_tokens', request)
    result = {'group': index, 'input_tokens': counted['input_tokens'], 'paid': False}
    I.save(target, result)
    return result


def main():
    rows = catalogue()
    groups, current, words = [], [], 0
    for row in rows:
        if row['duplicate_of_prior'] or not row['external_review_allowed'] or not row['text_extractable']:
            continue
        source = json.loads((I.CACHE/'texts'/(row['key']+'.json')).read_text(encoding='utf8'))
        body = '\n'.join('PDF_PAGE '+str(i+1)+'\n'+s for i,s in enumerate(source['pages']))
        if current and words + row['text_words'] > 100000:
            groups.append('\n\n'.join(current)); current, words = [], 0
        current.append('SOURCE_KEY '+row['key']+'\n'+body)
        words += row['text_words']
    if current:
        groups.append('\n\n'.join(current))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(lambda pair: count(*pair), enumerate(groups)))
    total = sum(x['input_tokens'] for x in counts)
    eligible = [r for r in rows if not r['duplicate_of_prior'] and
                r['external_review_allowed'] and r['text_extractable']]
    result = {'method': 'Free official token counts over full extracted paper texts, not PDF-image review',
              'paid_generation_calls': 0, 'input_tokens': total, 'groups': counts,
              'eligible_unique_text_papers': len(eligible),
              'opus_input_only_usd_at_4_per_million': total*4/1e6,
              'opus_max_output_usd_at_1024_tokens_per_paper': len(eligible)*1024*20/1e6,
              'exclusions': [r['key'] for r in rows if r not in eligible],
              'not_included': 'Per-paper task prompt, figures, scan-only PDF, Fable draft, Opus synthesis and potential corrections'}
    I.save(I.OUT / 'budget_estimate.json', result)
    print('FREE TOKEN COUNT COMPLETE', total, 'tokens;', len(eligible), 'unique papers;',
          round(total*4/1e6, 3), 'USD input only', flush=True)


if __name__ == '__main__':
    main()
