"""Independent publisher-record checks of audited source identities (no paid API)."""
import concurrent.futures
import difflib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import introduction_corpus_inventory as I


def entries(text):
    result={}
    for match in re.finditer(r'@\w+\s*\{\s*([^,\s]+)\s*,',text):
        depth=1; i=match.end()
        while i<len(text) and depth:
            if text[i] in '{}':
                depth += 1 if text[i]=='{' else -1
            i+=1
        result[match[1]]=text[match.start():i]
    return result


def field(entry,name):
    m=re.search(r'\b'+re.escape(name)+r'\s*=\s*\{',entry,re.I)
    if not m:
        return ''
    depth=1; i=m.end(); start=i
    while i<len(entry) and depth:
        if entry[i] in '{}':
            depth += 1 if entry[i]=='{' else -1
        i+=1
    return entry[start:i-1]


def normalized(text):
    return re.sub(r'[^a-z0-9]+','',text.lower())


def check(audit, existing, catalogue):
    key=audit['key']; citation=audit['citation']
    choices=[str(citation.get('doi_or_arxiv','')),field(existing.get(key,''),'doi'),
             catalogue.get(key,{}).get('identifier','')]
    matches=[re.search(r'10\.\d{4,9}/[^\s\)\]}>;,]+',s) for s in choices]
    candidates=list(dict.fromkeys(m[0].rstrip('.') for m in matches if m))
    result={'key':key,'pdf_citation':citation,'doi_candidates':candidates,
            'existing_bib_year':field(existing.get(key,''),'year'),
            'metadata_only_not_claim_verification':True}
    if not candidates:
        result['state']='no_own_doi_verified_from_local_record'
        return result
    doi=candidates[0]
    target=I.CACHE/'crossref'/(key+'.json')
    if target.exists():
        payload=json.loads(target.read_text(encoding='utf8'))
    else:
        url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
        request=urllib.request.Request(url,headers={'User-Agent':'LiteratureSourceVerification/1.0',
                                                   'Accept':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=40) as response:
                payload=json.load(response)
            I.save(target,payload)
        except Exception as error:
            result.update({'state':'crossref_unavailable','error_type':type(error).__name__})
            return result
    record=payload['message']
    title=' '.join(record.get('title',[]))
    score=difflib.SequenceMatcher(None,normalized(str(citation.get('title',''))),normalized(title)).ratio()
    result.update({'state':'title_match' if score>=.8 else 'manual_identity_check_required',
                   'crossref_doi':record.get('DOI'),'crossref_title':title,'title_similarity':score,
                   'venue':record.get('container-title'), 'volume':record.get('volume'),
                   'page':record.get('page'),'article_number':record.get('article-number'),
                   'issued':record.get('issued'),'published_print':record.get('published-print'),
                   'published_online':record.get('published-online'),
                   'authors':[{'given':a.get('given'),'family':a.get('family')} for a in record.get('author',[])],
                   'source_url':'https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')})
    return result


def main():
    existing=entries((I.ROOT/'docs/paper/ppcf_revision/refs.bib').read_text(encoding='utf8'))
    catalogue={r['key']:r for r in json.loads((I.OUT/'catalogue.json').read_text(encoding='utf8'))['papers']}
    audits=[json.loads(p.read_text(encoding='utf8'))
            for pattern in ['paper_*.json','background_*.json']
            for p in sorted(I.OUT.glob(pattern))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda a:check(a,existing,catalogue),audits))
    I.save(I.OUT/'bibliography_checks.json',{'method':'Own-DOI Crossref title/date crosscheck against original PDF audit; not verification of scientific claims',
                                           'records':results})
    print('BIBLIOGRAPHY CHECKED',len(results),'records',flush=True)


if __name__=='__main__':
    main()
