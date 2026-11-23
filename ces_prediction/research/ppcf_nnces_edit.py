"""Source-bound NN-CES framing edit and fresh focused review; no new grading/fits."""
import base64
import hashlib
import json
from pathlib import Path

import fable_deep_review as D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/paper/ppcf_revision'
CACHE = ROOT / 'data/.ppcf_nnces_edit'
PRICES = {'claude-fable-5-2': (8, 40), 'claude-opus-5-5': (4, 20)}
CAP = 3.0


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def ledger():
    rows = []
    for p in sorted(CACHE.glob('*.reservation.json')):
        z = json.loads(p.read_text(encoding='utf8'))
        response = p.with_name(p.name.replace('.reservation.json', '.response.json'))
        if response.exists():
            result = json.loads(response.read_text(encoding='utf8'))
            u = result['usage']
            assert not u.get('cache_creation_input_tokens') and not u.get('cache_read_input_tokens')
            ip, op = PRICES[z['model']]
            z.update({'usage': u, 'stop_reason': result['stop_reason'],
                      'estimated_usd': (ip*u['input_tokens'] + op*u['output_tokens'])/1e6,
                      'response_sha256': D.sha(response)})
        rows.append(z)
    spent = sum(x.get('estimated_usd', x['reserved_usd']) for x in rows)
    prior = json.loads((ROOT / 'docs/paper/fable_deep_review/cost_ledger.json').read_text())
    assert spent <= CAP and spent + prior['estimated_spent_usd'] <= 30
    result = {'cap_usd': CAP, 'pricing_source': 'https://claude.com/pricing?currency=USD',
              'prices_per_million_input_output': PRICES, 'invoice_verified': False,
              'estimated_or_reserved_usd': spent, 'prior_deep_review_usd': prior['estimated_spent_usd'],
              'combined_usd': spent + prior['estimated_spent_usd'], 'calls': rows}
    save(OUT / 'nnces_edit_usage.json', result)
    return result


def call(stage, model, content, max_tokens=8192):
    payload = {'model': model, 'max_tokens': max_tokens,
               'output_config': {'effort': 'low'},
               'messages': [{'role': 'user', 'content': content}]}
    request = CACHE / (stage + '.request.json')
    response = CACHE / (stage + '.response.json')
    binding = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if response.exists():
        assert json.loads(request.read_text(encoding='utf8')) == payload
        result = json.loads(response.read_text(encoding='utf8'))
    else:
        assert not request.exists(), 'Uncertain prior attempt; do not replay'
        assert len(list(CACHE.glob('*.reservation.json'))) < 3
        count = D.api('/messages/count_tokens', {'model': model, 'messages': payload['messages']})
        ip, op = PRICES[model]
        reserved = 1.1*count['input_tokens']*ip/1e6 + max_tokens*op/1e6 + .02
        assert ledger()['estimated_or_reserved_usd'] + reserved <= CAP
        save(request, payload)
        save(CACHE / (stage+'.reservation.json'),
             {'stage': stage, 'model': model, 'request_sha256': binding, 'reserved_usd': reserved})
        ledger()
        result = D.api('/messages', payload)
        save(response, result)
    ledger()
    assert result['stop_reason'] == 'end_turn', 'Incomplete output preserved; not a review'
    text = D.txt(result).strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        if error.msg!='Extra data':
            raise
        head,end=json.JSONDecoder().raw_decode(text)
        if isinstance(head,dict) and text[end:].lstrip().startswith(','):
            # A completed response closed the root before its remaining fields.
            # Remove that one premature brace; retain every field and raw bytes.
            repaired=json.loads(text[:end-1]+text[end:])
            repaired['json_format_repair_note']='Removed one premature root-closing brace before remaining fields. Original complete response unchanged.'
            return repaired
        raise


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    before = CACHE / 'main_before.tex'
    current = OUT / 'main_ppcf.tex'
    if not before.exists():
        before.write_bytes(current.read_bytes())
        (CACHE / 'main_before.pdf').write_bytes((OUT / 'main_ppcf.pdf').read_bytes())
    text = before.read_text(encoding='utf8')
    paragraphs = text.split('\n\n')
    prefixes = {'motivation': 'The KSTAR chord studied here',
                'earlier_work': 'Earlier work meets the problem',
                'discussion': r'\emph{Relation to readers of the spectrum.}'}
    original = {k: next(p for p in paragraphs if p.startswith(v)) for k, v in prefixes.items()}
    source_record = json.loads((ROOT / 'docs/reference/nnces_source.json').read_text())
    pdf = ROOT / source_record['user_supplied_copy']
    assert D.sha(pdf) == source_record['user_pdf_sha256']
    doc = {'type': 'document', 'title': 'Lee et al NN-CES published full text7pages',
           'source': {'type': 'base64', 'media_type': 'application/pdf',
                      'data': base64.b64encode(pdf.read_bytes()).decode()}}
    task = r'''Edit a few English paragraphs of an existing PPCF draft using the attached actual NN-CES full PDF, not an abstract. No private reasoning, no acceptance grades, no new results or invented novelty.
Author's constructive interpretation: NN-CES is an upstream spectrum reader (current measured CES spectrum -> reported Ti/rotation); our nowcast can use previously available CES values plus current/past BES/ECEI to estimate the current CES report before a new report is available. These can be connected in principle. We have only conventional-analysis CES labels, not NN-CES outputs or paired raw spectra. We have NOT coupled the models, retrained on NN-CES outputs, measured their arrival timing, or validated a controller. An upstream reader could refresh anchors sooner; this may reduce rather than guarantee the marginal need for nowcasting. It does not deliver a future temperature before its measurement. Do not claim our study is inherently more advanced or all prior spectrum readers are noncausal. Distinguish frame F from L5 (offline median of subsequent reports): our primary description here is current-report nowcasting, not future-temperature forecasting or high-rate physical reconstruction. Faster spectrum reading may reduce availability gaps; no actual pipeline benefit is established. Raw-spectrum availability and causal temporal information sets differ. A smooth output is not proof of physical truth.
Keep the paper aim: pedestal-top chord nowcasting against hold/GP/interpolation. Leave existing findings and numbers unchanged. Preserve citation keys and TeX macros. No new timing numbers. Use \citep{Lee2026NeuralNetworkbased}. Keep constructive prose before the qualification.
Return strict JSON, no fences:
{motivation: revised existing motivation paragraph, earlier_work: existing earlier-work paragraph with ONLY its overbroad sentence about all readers inheriting edge weakness corrected, intro_bridge: new120word-or-shorter paragraph to insert AFTER earlier_work explaining spectrum-to-report and report-to-current-nowcast complementarity, discussion: replace the misleading existing discussion paragraph with <=150words on possible coupling, faster anchor refresh, no validated coupling or superiority, korean_summary: concise explanation, primary_locators: source pages/sections used}.
Motivation correction: the old 'delivers about a second later' is not our measured fixed latency; describe the paper's need for a later beam-off background under conventional analysis without assigning our files a measured arrival delay. Do not lead with limitations. Preserve first two motivation sentences about this chord exactly. CURRENT PARAGRAPHS:
'''
    draft = call('01_source_edit', 'claude-fable-5-2',
                 [doc, D.textblock(task + json.dumps(original))])
    save(CACHE / 'draft.json', draft)
    review_task = '''Fresh focused source-and-logic review, not an acceptance grade. Read the attached published NN-CES PDF and proposed paragraphs. Do they correctly represent a possible upstream spectral reader and downstream causal current-report nowcast, keep F versus future-derived L5 distinct, avoid actual integration/forecasting/edge-failure claims, preserve the constructive author framing, and avoid asserting that a faster reader guarantees more nowcast value? Independent source check: PDF p5 timing is single-batch throughput, not single-frame online latency. Identify exact source locators. Do not require a new physics storyline or acceptance-driven reframing. Return strict JSON {verdict: pass or corrections_required, corrections: [{field,reason,replacement}], source_checks: concise list, korean_assessment: concise Korean}. Corrections should replace whole affected fields, only where materially needed. No study outcome/number changes. PROPOSED PARAGRAPHS:\n'''
    review = call('02_focused_review', 'claude-opus-5-5',
                  [doc, D.textblock(review_task + json.dumps(draft))], max_tokens=6144)
    save(CACHE / 'review.json', review)
    for correction in review.get('corrections', []):
        assert correction['field'] in ['motivation', 'earlier_work', 'intro_bridge', 'discussion']
        draft[correction['field']] = correction['replacement']
    # Independent host check: F and L5 are both evaluation targets; F is not an
    # input. The paper's conventional pipeline is not an arrival log for our rows.
    draft['intro_bridge'] = r'''Spectral analysis and nowcasting can therefore form complementary stages. NN-CES maps a measured CES spectrum to ion temperature and toroidal rotation without beam modulation~\citep{Lee2026NeuralNetworkbased}. Its reports could supply the upstream CES history for a temporal estimator. Our estimator uses already available CES reports and current and past BES and ECEI to estimate the current report before that report becomes available; it does not require the current raw CES spectrum. This is nowcasting rather than forecasting a future physical temperature. We evaluate both the current-frame report F and the future-derived level L5, which is an offline target defined below, not an input available online. The present evaluation uses conventional-analysis reports; coupling to NN-CES remains a possible implementation.'''
    draft['discussion'] = r'''\emph{Relation to readers of the spectrum.} A possible modular system would use NN-CES~\citep{Lee2026NeuralNetworkbased} to analyse each acquired spectrum and update the CES history when its result becomes available. Between updates, the temporal estimator would combine that history with the available BES and ECEI to maintain a nowcast. Faster spectral analysis could refresh the anchors sooner, while also shortening the interval over which a nowcast is needed. The usefulness of the combined system would therefore have to be measured with its actual report timing and outputs. Here the history and targets come from conventional analysis: NN-CES outputs were not used to fit or evaluate the estimator, and no coupled pipeline was tested. The two methods address complementary estimation stages; their relative accuracy is not established by this study.'''
    host_check = ('The focused reviewer conflated the causal input set with target F '
                  'and future-derived target L5. Both F and L5 are evaluation targets; '
                  'neither current F nor future L5 is an online input. The host corrected '
                  'this wording and removed an unsupported assertion about the exact '
                  'background-processing timing of every supplied label. No new data '
                  'or result was introduced. Final paragraphs retain hypothetical coupling.')
    for key in ['motivation', 'earlier_work', 'discussion']:
        assert text.count(original[key]) == 1
        text = text.replace(original[key], draft[key])
    assert text.count(draft['earlier_work']) == 1
    text = text.replace(draft['earlier_work'], draft['earlier_work']+'\n\n'+draft['intro_bridge'])
    # Use the already-present verified BibTeX record instead of a hand-written DOI
    # in prose; this also removes inherited DOI tokens from the number checker.
    explicit = r'(UFCES; Jang \emph{et al} 2021, \href{https://doi.org/10.1063/5.0043824}{doi:10.1063/5.0043824})'
    assert text.count(explicit) == 1
    text = text.replace(explicit, r'(UFCES)~\citep{Jang2021DevelopmentUltrafast}')
    assert r'\citep{Lee2026NeuralNetworkbased}' in draft['intro_bridge']
    assert text.count(r'\input{generated_tables}') == before.read_text(encoding='utf8').count(r'\input{generated_tables}')
    current.write_text(text, encoding='utf8')
    save(CACHE / 'applied.json', draft)
    report = '# NN-CES upstream-reader framing update\n\n'
    report += draft['korean_summary'] + '\n\n## Focused independent source review\n\n'
    report += review['korean_assessment'] + '\n\n'
    report += '## Independent host correction of reviewer advice\n\n' + host_check + '\n\n'
    report += 'Source locators: ' + json.dumps(draft['primary_locators'], ensure_ascii=False) + '\n\n'
    report += 'This is a focused paragraph review, not a new fixed-prompt manuscript grade. Historical grade3.md and novelty2.md apply to their frozen PDFs. No NN-CES coupling, new fits or TEST readout was performed.\n\n'
    report += '## Applied paragraphs\n\n' + '\n\n'.join(draft[k] for k in ['motivation', 'intro_bridge', 'discussion']) + '\n'
    (OUT / 'NN_CES_EDIT.md').write_text(report, encoding='utf8')
    save(OUT / 'nnces_edit_status.json', {'source_record': 'docs/reference/nnces_source.json',
         'source_pdf_sha256': D.sha(pdf), 'before_tex_sha256': D.sha(before),
         'after_tex_sha256': D.sha(current), 'focused_review': review,
         'independent_host_check': host_check,
         'new_full_manuscript_grade': False, 'scientific_runs': 0, 'compiled': False})
    print('SOURCE EDIT AND FOCUSED REVIEW COMPLETE', flush=True)


if __name__ == '__main__':
    main()
