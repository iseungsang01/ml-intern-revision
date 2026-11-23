"""One-shot durable wait -> gated Claude tool workflow -> local research draft."""
import argparse,contextlib,datetime,json,subprocess,sys,time,traceback
import evidence as E
import agent

STAGES=[
 ('evidence_review','claude-fable-5-2','analysis','Read both scopes and all four designs. Extract supported claims, failures and limits. Distinguish thresholds from zero, actual conditional intervals, baseline support and budgets. Do not choose a new model.'),
 ('adversarial_review','claude-opus-5-5','analysis','Challenge evidence_review against original evidence, especially temporal designs, three-seed subset, budget/convergence, supplementary support, physical interpretation and B98 scope. Prior notes are proposals.'),
 ('research_plan','claude-fable-5-2','plan','Use both reviews and evidence to rank at most3 bounded follow-up candidates, only1 immediate next action. Minimize runtime and redundant work. Include overlap with prior B98/Lean/age controls; TRAIN/VAL-only development; new independent confirmation; registered positive/negative sentences and stop rules. All are TEST-informed exploratory.'),
 ('final_revision','claude-opus-5-5','plan','Revise the proposed research_plan critically against both evidence scopes and adversarial findings. Keep at most3 candidates and1 immediate action. Remove unsupported physics/novelty, conflated supports, unverifiable timings and redundant experiments. Return the final structured exploratory plan, not an acceptance estimate.')]

def sources():
    paths=[*E.HERE.glob('*.py'),E.HERE/'PLAN.md',
           E.ROOT/'ces_prediction/experiments/b85_unified/claude_review85.py',
           E.ROOT/'ces_prediction/experiments/b93_gpu/atomic93.py',
           E.ROOT/'ces_prediction/experiments/b93_gpu/status93.py']
    return {p.relative_to(E.ROOT).as_posix():E.sha(p) for p in paths}

def status(state,**extra):E.save(E.CACHE/'status.json',{'state':state,'updated_kst':datetime.datetime.now().astimezone().isoformat(),**extra})

def render(evidence,bindings,final):
    E.save(E.CACHE/'evidence_complete.json',evidence)
    ledger=[]
    for scope,s in evidence['b94']['scopes'].items():
        for design,d in s['designs'].items():
            for name,c in d['cells'].items():
                v=c['skill'];ledger.append({'scope':scope,'design':design,'cell':name,'rows':c['rows'],'clusters':c['clusters'],**v})
    calls=[]
    for p in sorted((E.CACHE/'agent').glob('*.response.json')):
        r=E.read(p);calls.append({'record':p.name,'sha256':E.sha(p),'requested_model':r['model_requested'],'returned_model':r['response'].get('model'),'stop_reason':r['response'].get('stop_reason'),'usage':r['response'].get('usage',{}),'seconds':r['seconds']})
    output={'status':'draft_ready_for_lead_review','original_audit_binding':bindings,'original_evidence_sha256':E.sha(E.CACHE/'evidence.json'),
            'complete_evidence_sha256':E.sha(E.CACHE/'evidence_complete.json'),'reviews_sha256':E.sha(E.CACHE/'reviews.json'),
            'deterministic_paired_evidence':ledger,'claude_proposal':final,'api_calls':calls,
            'input_tokens':sum(c['usage'].get('input_tokens',0) for c in calls),'output_tokens':sum(c['usage'].get('output_tokens',0) for c in calls),
            'automatic_training':False,'semantic_verification':'Evidence pointers exist; independent original arithmetic passed. New hypotheses and literature/originality claims remain proposals.'}
    E.save(E.ROOT/'docs/paper/posteval_claude_20261123.json',output)
    lines=['# 평가 후 Claude 연구 정리 초안','',
           '원래 평가의 정상 종료와 독립 최종 감사가 통과한 뒤 실행했다. 아래 수치 표는 원본 JSON에서 직접 생성했으며, 후속 연구는 Claude가 제안한 TEST-informed 탐색 초안이다. 새 학습·원고 수정·물리 발견·수락률 추정은 실행하지 않았다.','',
           'Fable 근거 검토 → Opus 반대 검토 → Fable 연구 설계 → Opus 최종 수정. 원자료·개인 정보·키를 전송하지 않고 익명 집계와 명시된 한계만 제공했다. 실제 API 사용·근거 hash는 같은 이름의 JSON에 있다.','',
           '## 다음 연구 제안','',final['summary'],'','**바로 다음 단계:** '+final['immediate_next_step'],'']
    for c in sorted(final['ranked_candidates'],key=lambda a:a['priority']):
        lines += ['### '+str(c['priority'])+'. '+c['title'],'']
        for key,label in [('hypothesis','가설'),('existing_work_overlap','기존 작업과 중복'),('minimal_experiment','최소 실험'),('train_val_only_development','개발 범위'),('independent_confirmation','독립 확인'),('falsifier','반증 측정'),('stop_rule','중단 기준'),('positive_sentence','양의 결과 문장'),('negative_sentence','음의 결과 문장'),('timing_basis','시간 근거')]:
            lines += ['- '+label+': '+c[key]]
        lines += ['- 근거 JSON pointer: '+', '.join('`'+p+'`' for p in c['evidence_refs']),'']
    lines += ['## 지지되지 않는 주장','',*['- '+s for s in final['claims_not_supported']],'',
              '## 원본에서 생성한 paired skill','',
              '모든 설계/두 evidence scope를 함께 읽는다. point [95% cluster interval]; zero 제외 여부이며, 등록 statement의 비열등 threshold 판정과는 별개다.','',
              '| Scope | Design | Cell | Rows | Clusters | Point [interval] | Excludes zero |',
              '|---|---|---|---:|---:|---|---|']
    for v in ledger:
        lo,hi=v['cluster'];lines.append(f"| {v['scope']} | {v['design']} | {v['cell']} | {v['rows']} | {v['clusters']} | {v['point']:.6f} [{lo:.6f}, {hi:.6f}] | {v['excludes_zero']} |")
    (E.ROOT/'docs/paper/posteval_claude_20261123.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def execute(config):
    assert sources()==config['sources'],'Source drift after scheduled registration'
    status('waiting_for_audited_completion',wait_deadline=config['wait_deadline'])
    while time.time()<config['wait_deadline']:
        try:ready=E.gate()
        except (FileNotFoundError,PermissionError):ready=None
        if ready:break
        try:phase=E.read(E.UP/'queue_status.json')['state']
        except (FileNotFoundError,PermissionError):phase='metadata_unavailable'
        status('waiting_for_audited_completion',upstream_phase=phase,wait_deadline=config['wait_deadline'])
        time.sleep(30)
    else:raise RuntimeError('Timed out waiting for original audited completion; no partial-result review')
    result,binding=ready;assert sources()==config['sources']
    data=E.packet(result,binding);E.save(E.CACHE/'evidence.json',data);E.save(E.CACHE/'audit_binding.json',binding)
    key=E.API.credential();catalog=E.API.request('/models?limit=1000',key);ids={m['id'] for m in catalog['data']}
    assert {s[1] for s in STAGES}<=ids,'Requested Claude model unavailable; no silent fallback'
    for name,model,kind,instruction in STAGES:
        assert sources()==config['sources']
        status('claude_'+name,original_readout_sha256=binding['readout_sha256'])
        data['reviews'][name]=agent.run(name,model,kind,instruction,data,key)
        E.save(E.CACHE/'reviews.json',data['reviews'])
    # Recheck upstream result identity before saving any attributed draft.
    assert E.gate()[1]==binding
    render(data,binding,data['reviews']['final_revision'])
    status('draft_ready_for_lead_review',report='docs/paper/posteval_claude_20261123.md',automatic_training=False)

def register():
    E.CACHE.mkdir(parents=True,exist_ok=True)
    assert not (E.CACHE/'registration.json').exists(),'Existing scheduled workflow; inspect instead of duplicate launch'
    for rel in sources():
        subprocess.run(['git','cat-file','-e','HEAD:'+rel],cwd=E.ROOT,check=True)
        subprocess.run(['git','diff','--quiet','HEAD','--',rel],cwd=E.ROOT,check=True)
    upstream=E.read(E.UP/'supervisor_status.json');deadline=float(upstream['deadline']);name='B93_Claude_PostEval_20261123'
    config={'task_name':name,'sources':sources(),'wait_deadline':deadline+600,'automatic_training':False,'models':[s[1] for s in STAGES],'max_paid_rounds':16}
    E.save(E.CACHE/'config.json',config)
    sys.path.insert(0,str(E.ROOT/'ces_prediction/experiments/b93_gpu'));import durable93 as D
    pythonw=E.Path(sys.executable).with_name('pythonw.exe')
    args=['-u',str(E.HERE/'worker.py'),'execute'];quote=D.quote_ps
    script=f"""$ErrorActionPreference='Stop'
$action=New-ScheduledTaskAction -Execute {quote(pythonw)} -Argument {quote(subprocess.list2cmdline(args))} -WorkingDirectory {quote(E.ROOT)}
$settings=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds {int(deadline-time.time()+7200)}) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal=New-ScheduledTaskPrincipal -UserId ($env:USERDOMAIN+'\\'+$env:USERNAME) -LogonType Interactive -RunLevel Limited
if(Get-ScheduledTask -TaskName '{name}' -ErrorAction SilentlyContinue){{throw 'Task name already exists'}}
Register-ScheduledTask -TaskName '{name}' -Action $action -Settings $settings -Principal $principal | Out-Null
Start-ScheduledTask -TaskName '{name}'
"""
    r=D.powershell(script);assert r.returncode==0,r.stderr
    E.save(E.CACHE/'registration.json',{**config,'registered_at':time.time(),'restart_on_failure':0,'command':[str(pythonw),*args]})
    print(json.dumps({'registered_and_started':name,'max_paid_rounds_after_audit':16}))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','execute','check']);a=p.parse_args()
    if a.action=='register':return register()
    if a.action=='check':
        ready=E.gate();print(json.dumps({'evaluation_ready':ready is not None,'sources':sources()}));return
    E.CACHE.mkdir(parents=True,exist_ok=True)
    with (E.CACHE/'worker.stdout.log').open('a',encoding='utf8') as out,(E.CACHE/'worker.stderr.log').open('a',encoding='utf8') as err:
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:execute(E.read(E.CACHE/'config.json'))
            except BaseException as exc:
                status('failed',error=type(exc).__name__+': '+str(exc));traceback.print_exc();raise

if __name__=='__main__':main()
