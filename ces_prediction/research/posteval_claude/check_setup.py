"""Actual API/tool contract check on real audited metadata, no partial TEST review."""
import json
import evidence as E
import agent

def main():
    base=E.CACHE/'setup_check';base.mkdir(parents=True,exist_ok=True)
    b=E.read(E.ROOT/'docs/paper/b97_lean.json')
    sup=E.read(E.UP/'supervisor_status.json');queue=E.read(E.UP/'queue_status.json')
    data={'context':{'current_task':'Configure a wait-for-original-final-audit research workflow, not interpret current partial evaluation.',
                     'real_audited_lean_status':b['status'],
                     'upstream_current_stage':E.read(E.UP/'queue_status.json')['state'],
                     'original_evaluation_gate':{'queue_complete':queue['state']=='complete','supervisor_exit0':sup['state']=='complete' and sup.get('returncode')==0,'post_audit_passed':False,'selection_readout_hashes_equal':False},
                     'gate':'Require original queue complete, supervisor complete exit0, post audit passed, selection/readout hashes equal; otherwise no B94 metrics exposed.',
                     'tools':'Read-only explicit aggregate evidence pointers and scoped structured proposal submissions. No shell, credentials, raw data, training or arbitrary file edits.',
                     'scope':'At most4paid rounds per substantive stage; four roles. All TEST-informed followups exploratory with preregistration and independent confirmation.'},'reviews':{}}
    key=E.API.credential();catalog=E.API.request('/models?limit=1000',key)
    assert {'claude-fable-5-2','claude-opus-5-5'}<={m['id'] for m in catalog['data']}
    plan=agent.run('setup_review','claude-fable-5-2','analysis','This is ONLY workflow setup. Read /context, assess its safeguards and identify one useful correction if any. No b94/scopes or statistical results exist in this setup evidence. Submit1-3 findings with /context evidence refs; do not claim the upstream evaluation completed.',data,key,folder=base,max_rounds=4)
    assert plan['findings']
    E.save(base/'passed.json',{'actual_api_tool_check':True,'finding_count':len(plan['findings']),'metadata_only':True})
    print(json.dumps({'actual_api_tool_check':True,'findings':len(plan['findings'])}))

if __name__=='__main__':main()
