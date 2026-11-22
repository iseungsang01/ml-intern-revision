"""Immutable pilot allocation based solely on paired training-validation records."""
import json
import numpy as np
import common92 as M
from train92 import verify_complete
def compute_selection():
    M.guard();inputs={};values={}
    for stage,u,s,c in M.pilot_jobs():
        rd=M.run_dir(stage,u,s,c);assert verify_complete(rd)
        metrics=json.loads((rd/'metrics.json').read_text());value=json.loads((rd/'complete.json').read_text())['best_val_loss'];assert np.isfinite(value) and value>=0
        values[(u,c)]=value;inputs[u+'|'+c]={'best_val_loss':value,'files':{n:M.sha(rd/n) for n in ('config.json','complete.json','process_exit.json','metrics.json','best.pt','latest.pt')}}
    candidates={}
    for c in ('A','T'):
        assert all(values[(u,'C')]>0 for u in M.PILOT_UNITS),'Control validation denominator must be positive'
        ratios={u:values[(u,c)]/values[(u,'C')] for u in M.PILOT_UNITS};mean=float(np.mean(list(ratios.values())))
        candidates[c]={'ratios':ratios,'mean':mean,'eligible':bool(mean<.98 and ratios['V3_q3']<=1.05 and ratios['V4a']<=1.05)}
    eligible=[c for c in ('A','T') if candidates[c]['eligible']]
    chosen=None
    if len(eligible)==1:chosen=eligible[0]
    elif len(eligible)==2:chosen='A' if abs(candidates['A']['mean']-candidates['T']['mean'])<=.005 else min(eligible,key=lambda c:candidates[c]['mean'])
    return {'sources':M.sources(),'inputs':inputs,'candidates':candidates,'chosen':chosen,'test_accessed':False,'rule':'mean<.98, V3/V4<=1.05; absolute tie .005 prefersA'}
def lock_selection():
    z=compute_selection();p=M.CACHE/'selection.json'
    if p.exists():assert json.loads(p.read_text())==z,'Immutable pilot decision mismatch'
    else:M.save(p,z)
    return z
if __name__=='__main__':print(json.dumps(lock_selection(),indent=2))
