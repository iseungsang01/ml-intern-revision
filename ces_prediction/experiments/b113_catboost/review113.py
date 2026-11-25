"""One bounded redacted design review; never sends data or private paths."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b85_unified'))
import claude_review85 as API
OUT=ROOT/'data/.b113_review';OUT.mkdir(exist_ok=True)
prompt='''Skeptical methodological review, <=500 words. No results to infer. Exploratory reused-TEST causal nowcasting of current reported scalar F. Compare GPU CatBoost direct F and residual F minus a fixed causal GP, identical numeric scalar inputs and internal TRAIN rows from four existing outer validation designs; seven equally rule-weighted availability rules in TRAIN, eighth cadence rule TEST-only. Three fixed seeds per form per design. Fixed depth6,1000 iterations,lr.03,RMSE,l2=3,border_count128; no VAL selection, no TEST tuning. All24 fits finish before TEST inference. Reuse audited same-support HGB direct/residual and GP+fast predictions. Report all32 cells, all seeds and arithmetic ensembles, paired shot-cluster and discharge bootstrap20000. Intervals conditional on fits; initialization spread separately. Compare GPU predictions independently using exported JSON tree traversal on CPU, verify all original input hashes/splits/support. GPU-only real TRAIN512-row 10-tree smoke before full fits, no retry on failure, no edits to running scientific code. Identify blocking flaws and minimal controls. No architectural superiority, novelty or physical truth claim. This is a fixed-budget learner comparison, no claim of optimized CatBoost.'''
payload={'model':'claude-fable-5-2','max_tokens':1400,'thinking':{'type':'disabled'},'messages':[{'role':'user','content':prompt}]}
p=OUT/'request.json';assert not p.exists(),'No repeated paid attempt'
p.write_text(json.dumps({'payload':payload,'max_attempts':1,'max_output_tokens':1400,'started':time.time()},indent=2))
try:
 r=API.request('/messages',API.credential(),payload)
 (OUT/'response.json').write_text(json.dumps(r,indent=2))
 text='\n'.join(x['text'] for x in r.get('content',[]) if x.get('type')=='text')
 report='# B113 bounded Claude design advice\n\nRedacted methodological prompt only; no raw observations or identity. One attempt, max1400 output tokens. Advice is not verified fact.\n\n'+text+'\n\nUsage: '+json.dumps(r.get('usage',{}))+'\nStop: '+str(r.get('stop_reason'))
except Exception as e:
 (OUT/'error.json').write_text(json.dumps({'error':str(e),'time':time.time()}))
 report='# B113 Claude review attempt\n\nOne bounded official API attempt failed: '+str(e)+'. No completed advice; no scientific data sent. Attempt retained under ignored data/.b113_review.\n'
(ROOT/'docs/paper/b113_claude.md').write_text(report,encoding='utf8')
print('B113 review attempt preserved')
