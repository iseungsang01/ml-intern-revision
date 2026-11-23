"""Inventory every local literature PDF before bounded source-to-claim review."""
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/literature'
CACHE = ROOT / 'data/.introduction_corpus'
OUT = ROOT / 'docs/paper/introduction_review'


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    for attempt in range(20):
        try:
            temp.replace(p)
            return
        except PermissionError:
            if attempt==19:
                raise
            time.sleep(.1)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.monotonic()
    for i, p in enumerate(sorted(SOURCE.glob('*.pdf'))):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        extracted = CACHE / 'texts' / (p.stem + '.json')
        if extracted.exists():
            text = json.loads(extracted.read_text(encoding='utf8'))
            assert text['sha256'] == digest
        else:
            with fitz.open(p) as doc:
                text = {'sha256': digest, 'pages': [page.get_text() for page in doc],
                        'metadata': doc.metadata}
            save(extracted, text)
        head = '\n'.join(text['pages'][:2])
        full = '\n'.join(text['pages'])
        doi = sorted(set(x.rstrip('.,);]') for x in re.findall(
            r'10\.\d{4,9}/[^\s<>"{}]+', head)))
        normalized = re.sub(r'\s+', ' ', full).strip()
        rows.append({'key': p.stem, 'path': p.relative_to(ROOT).as_posix(),
                     'sha256': digest, 'pages': len(text['pages']),
                     'text_chars': len(full), 'text_words': len(full.split()),
                     'normalized_text_sha256': hashlib.sha256(normalized.encode()).hexdigest(),
                     'doi_candidates': doi, 'metadata_title': text['metadata'].get('title', ''),
                     'text_extractable': len(full.strip()) > 300,
                     'review_state': 'inventoried_not_reviewed'})
        if (i+1) % 25 == 0:
            save(CACHE / 'inventory_progress.json', {'files': i+1, 'elapsed_s': time.monotonic()-started})
            print('Inventoried', i+1, flush=True)
    groups = {}
    for row in rows:
        groups.setdefault(row['normalized_text_sha256'], []).append(row['key'])
    result = {'scope': 'All local corpus PDFs; inventory is not full-text review',
              'pdf_files': len(rows), 'total_pages': sum(x['pages'] for x in rows),
              'total_text_words': sum(x['text_words'] for x in rows),
              'unique_normalized_texts': len(groups),
              'text_unavailable': [x['key'] for x in rows if not x['text_extractable']],
              'duplicate_text_groups': [g for g in groups.values() if len(g)>1], 'papers': rows}
    save(OUT / 'corpus_inventory.json', result)
    print('INVENTORY COMPLETE', result['pdf_files'], 'PDFs', result['total_pages'], 'pages',
          result['total_text_words'], 'words', flush=True)


if __name__ == '__main__':
    main()
