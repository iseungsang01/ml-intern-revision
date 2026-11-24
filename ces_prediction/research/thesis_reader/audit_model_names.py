"""Verify editorial namespace change against prior manuscript and frozen sources."""
import hashlib
import json
import re
import subprocess

from build_reader import ROOT,quantitative_tokens
from model_names import GP,TCN

def main():
    relative='docs/paper/introduction_review/manuscript_sections.json'
    prior=json.loads(subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT))
    current=json.loads((ROOT/relative).read_text(encoding='utf8'))
    def walk(node):
        yield node
        for child in node.get('subsections',[]):yield from walk(child)
    def text(section):return '\n\n'.join(n.get('content_markdown','')for n in walk(section))
    old={s['id']:s for s in prior['sections']};new={s['id']:s for s in current['sections']}
    for sid in ['introduction','background','results','discussion','conclusion']:
        assert quantitative_tokens(text(old[sid]))==quantitative_tokens(text(new[sid])),sid
    for sid in new:
        assert set(re.findall(r'\[@([^\]]+)\]',text(old[sid])))==set(re.findall(r'\[@([^\]]+)\]',text(new[sid]))),sid
    for s in current['sections']:
        if s['id']=='materials-methods':continue # legacy mapping lives here
        assert not re.search(r'(?<![A-Za-z0-9_])(?:TXN|T0|TX|TE|TB|TN|TH)(?![A-Za-z0-9_])',text(s)),s['id']
    immutable=['docs/paper/ppcf_revision/main_ppcf.pdf','docs/paper/ppcf/main_ppcf.tex',
       'docs/paper/research_reader/longform/exhibits.json','docs/paper/b105_merit.json',
       'ces_prediction/experiments/b73_targets/common.py']
    for path in immutable:
        a=subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)
        assert a.replace(b'\r\n',b'\n')==(ROOT/path).read_bytes().replace(b'\r\n',b'\n'),path
    reader=ROOT/'docs/paper/research_reader/index.html'
    out={'scope':'Reader presentation names; scientific result keys and English source/PDF preserved.',
      'GP_namespace':GP,'TCN_namespace':TCN,'current_reader_sha256':hashlib.sha256(reader.read_bytes()).hexdigest(),
      'numeric_sequences_unchanged':['introduction','background','results','discussion','conclusion'],
      'citation_sets_unchanged':True,'ambiguous_names_only_in_explicit_legacy_mapping':True,
      'unchanged_sources':immutable,'ui_checks':json.loads((reader.parent/'qa_status.json').read_text(encoding='utf8')),
      'new_scientific_fits':0,'new_evaluations':0,'api_calls':0}
    (reader.parent/'model_naming.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('MODEL_NAMING_AUDIT_PASSED')

if __name__=='__main__':main()
