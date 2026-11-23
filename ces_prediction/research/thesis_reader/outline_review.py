"""Two bounded, redacted official Claude calls for the thesis-outline mapping."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import fable_deep_review as D
from extract_outline import CACHE, OUT

PRICES = {'claude-fable-5-2': (8, 40), 'claude-opus-5-5': (4, 20)}
CAP = 1.10


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def ledger():
    rows=[]
    for p in sorted(CACHE.glob('*.reservation.json')):
        z=json.loads(p.read_text(encoding='utf8'))
        response=p.with_name(p.name.replace('.reservation.json','.response.json'))
        if response.exists():
            result=json.loads(response.read_text(encoding='utf8')); u=result.get('usage',{})
            assert not u.get('cache_creation_input_tokens') and not u.get('cache_read_input_tokens')
            ip,op=PRICES[z['model']]
            z.update({'usage':u,'stop_reason':result.get('stop_reason'),
                'estimated_usd':(u.get('input_tokens',0)*ip+u.get('output_tokens',0)*op)/1e6,
                'response_sha256':D.sha(response)})
        rows.append(z)
    total=sum(x.get('estimated_usd',x['reserved_usd']) for x in rows)
    assert total<=CAP
    result={'cap_usd':CAP,'scope':'Thesis outline mapping and artifact design; separate allocation from parent corpus budget',
        'prices_input_output_per_million':PRICES,'pricing_source':'https://claude.com/pricing?currency=USD',
        'invoice_verified':False,'estimated_or_reserved_usd':total,
        'estimated_returned_usage_usd':sum(x.get('estimated_usd',0) for x in rows),'calls':rows}
    save(OUT/'api_usage.json',result)
    return result


def call(stage,model,text,max_tokens):
    payload={'model':model,'max_tokens':max_tokens,'output_config':{'effort':'low'},
             'messages':[{'role':'user','content':text}]}
    request=CACHE/(stage+'.request.json'); response=CACHE/(stage+'.response.json')
    if response.exists():
        assert json.loads(request.read_text(encoding='utf8'))==payload
        result=json.loads(response.read_text(encoding='utf8'))
    else:
        assert not request.exists(),'Uncertain prior attempt; no blind replay'
        assert len(list(CACHE.glob('*.reservation.json')))<2
        count=D.api('/messages/count_tokens',{'model':model,'messages':payload['messages']})
        ip,op=PRICES[model];reserve=count['input_tokens']*1.1*ip/1e6+max_tokens*op/1e6+.02
        assert ledger()['estimated_or_reserved_usd']+reserve<=CAP
        save(request,payload)
        save(CACHE/(stage+'.reservation.json'),{'stage':stage,'model':model,'reserved_usd':reserve,
            'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()})
        ledger()
        result=D.api('/messages',payload);save(response,result)
    ledger()
    assert result.get('stop_reason')=='end_turn','Incomplete output preserved'
    text=D.txt(result).strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
    result=json.loads(text);save(OUT/(stage+'.json'),result)
    return result


def main():
    headings=json.loads((OUT/'original_outline.json').read_text(encoding='utf8'))['headings']
    context='''An author asks to preserve an existing graduation-thesis outline while organizing a PPCF-targeted CES study. Do not output private reasoning. No identity, institutional affiliation, raw observations or personal correspondence is supplied. Actual study: causal current-report nowcasting on one provider-selected KSTAR2022 pedestal-top chord, strictly past CES reports plus current/past BES/ECEI snapshots on a10ms grid; compare hold/causalGP/interpolation and model families under four equally reported validation designs. F is current-report target and L5 is offline future-derived level target; neither currentF nor futureL5 is an online input. Conventional CES reports are labels, not independent physical truth. NN-CES maps a measured raw spectrum to Ti/rotation and could supply upstream reports to our downstream temporal nowcaster; no coupled experiment exists. Existing validated paper uses GP+boosted regression; LSTM/TCN comparisons exist but no global adoption of a CNN is established. Do not invent a result or accuracy. Existing CNN/TCN limitations and adverse findings must be retained. Future fits need separate preregistration, no execution here.
The author wants: introduction / theoretical background / materials and research methods / results / discussion / conclusion / references as distinct sections. In methods retain1.Data set, collection/selection criteria, a.plasma/scenario classification, b.time-series interval selection, c.discharge classification by experimental year/device state. Current campaign is2022; confinement mode, device-state and equilibrium metadata are not established, so these headings must remain as documented criteria/unknown classification, not invented L/H/I/FIRE labels.
Retain2.장치 설계 > 가.Neural Network > a.which NN architecture? compare several model families; b.existing advantages or modifications should be decided from evaluated performance. The intended two-stage research flow is first assess multimodal temporal estimation with CNN/lightweight comparators, then design a task-specific NN from the evidence. Treat unperformed parts as plans; current10ms-grid evaluations do not prove sub-ms physical Ti reconstruction, resolution multiplication or UFCES ground truth. Original headings are historical structure, not verified scientific claims. Preserve the names/places but correct overclaims in new wording. No need to rewrite the author's framing around a referee grade.
Return Korean strict JSON {mapping:[{old_heading,new_section_id,new_path,action,reason}],sections:[{id,title,status,content_markdown,subsections:[{id,title,status,content_markdown}]}],notes:[...]}. ids introduction/background/materials-methods/results/discussion/conclusion/references. Sections are a structural working draft with concise purpose paragraphs, not a complete invented thesis. Use Markdown and [@CorpusKey] only where clearly applicable; numbers are only headings/known study contract, never made-up performance. In Results only link existing validated PDF or say source-linked, no numerical transcription. Distinguish evaluated/planned/metadata-unavailable status. Schema will accept parent-provided fuller prose later. Original non-identifying heading list:\n'''
    draft=call('01_fable_outline','claude-fable-5-2',context+json.dumps(headings,ensure_ascii=False),16384)
    review='''Fresh source/logic and structure review of an artifact outline. No acceptance grade. The original headings and context were provided to the draft, but no thesis body, raw observations or identity was sent. Check preservation of the original Dataset/collection criteria/year-device classification and2.장치설계/가NN/aNNchoice/bmodification reasons. Do not invent device-state metadata or say a campaign year partition equals physical-mode classification. Check current F/future-derivedL5 are targets,10ms-grid nowcasting is not demonstrated sub-ms physical reconstruction, and future CNN/lightweight comparison then NN-design stays a plan. A negative model result should inform choice, not be overwritten with a success story. Keep constructive prose. Return strict JSON {assessment,corrections:[{path,problem,correction}],approved_with_caveats,corrected_sections:complete corrected sections array}. No numerical results or new models. CONTEXT:\n'''
    checked=call('02_opus_structure_check','claude-opus-5-5',review+context+'\nDRAFT:\n'+json.dumps(draft,ensure_ascii=False),12288)
    final={'title':'KSTAR CES nowcasting 연구 원고','subtitle':'졸업논문 목차를 보존한 절별 작업본',
           'status_note':'문헌 근거 보강과 초안 편집 중. 완료된 결과와 향후 계획을 구분합니다.',
           'sections':checked['corrected_sections'],'source_outline':'original_outline.json',
           'mapping':draft['mapping'],'structure_review':checked}
    titles={'introduction':'Ⅰ. 서론','background':'Ⅱ. 이론적 배경',
            'materials-methods':'Ⅲ. 재료 및 방법','results':'Ⅳ. 결과',
            'discussion':'Ⅴ. 고찰','conclusion':'Ⅵ. 결론','references':'Ⅶ. 참고문헌'}
    for section in final['sections']:
        section['title']=titles[section['id']]
    final['host_formatting_note']='최신 요청의 일곱 상위 절로 번호만 정리했다. 원래 Dataset/수집 기준/장치 설계/NN의 하위 자리는 유지한다. 원 API 응답은 바꾸지 않았다.'
    save(OUT/'outline_mapping.json',final)
    lines=['# 원래 HWP 목차와 현재 절별 구성','',final['host_formatting_note'],'',
           '원본의 개인 정보와 본문은 공개 기록이나 API 요청에 넣지 않았다. 아래는 목차만 대조한 구조 기록이며, 과거 제안 문구가 완료된 실험을 뜻하지 않는다.','',
           '| 원래 제목 | 현재 위치 | 처리와 이유 |','|---|---|---|']
    for item in final['mapping']:
        lines.append('| '+str(item.get('old_heading','')).replace('|','/')+' | '+
                     str(item.get('new_path','')).replace('|','/')+' | '+
                     str(item.get('action','')).replace('|','/')+' — '+
                     str(item.get('reason','')).replace('|','/')+' |')
    (OUT/'outline_mapping.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('OUTLINE REVIEW COMPLETE; USD',round(ledger()['estimated_returned_usage_usd'],6))


if __name__=='__main__':
    main()
