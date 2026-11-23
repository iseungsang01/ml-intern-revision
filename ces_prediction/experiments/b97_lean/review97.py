"""Author-authorized Claude cross-review; redacted source, never real traces."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b85_unified'))
import claude_review85 as API

def main():
    dest=ROOT/'data/.b97/claude_review.json'
    assert not dest.exists(),'Do not repeat a paid review silently'
    prompt='''Review this narrow Lean4 assurance implementation critically. It proves a discrete latest-eligible source selector, not full Python/CUDA correctness or physical truth. Identify concrete bugs, incorrect universal claims or gaps in the finite Python/native bridge. Do not assert that source code compiled; you cannot execute it. No synthetic measurements, extra experiments, private project search or acceptance estimate. Prefer at most5 actionable issues; distinguish actual errors from limitations. Answer under700words. Only redacted implementation source follows, with no observations, identity, credentials or performance scores.\n'''
    for name in ('formal/visibility/Visibility.lean','formal/visibility/Main.lean','ces_prediction/experiments/b97_lean/run97.py','ces_prediction/experiments/b97_lean/lead_checks/verify97.py'):
        prompt+='\nFILE '+name+'\n'+(ROOT/name).read_text(encoding='utf8')
    start=time.time()
    response=API.request('/messages',API.credential(),{'model':'claude-fable-5-2','max_tokens':8000,
        'thinking':{'type':'adaptive'},'output_config':{'effort':'low'},
        'messages':[{'role':'user','content':prompt}]})
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps({'prompt':prompt,'response':response,'seconds':time.time()-start},indent=2),encoding='utf8')
    print(json.dumps({'model':response.get('model'),'stop':response.get('stop_reason'),'usage':response.get('usage')}))
    assert response.get('stop_reason')=='end_turn','Incomplete response preserved; not a completed review'

if __name__=='__main__':main()
