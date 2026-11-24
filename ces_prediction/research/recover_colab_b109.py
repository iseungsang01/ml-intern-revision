"""Transport-only recovery; exact prefix hashes and explicit no-launch guard."""
import argparse,base64,hashlib,json,shutil,sys,time
from pathlib import Path
from colab_t4_operator import call,update_run,new_cell,gpu_check
ROOT=Path(__file__).resolve().parents[2];CON=ROOT/'data/.colab_connection';OUT=ROOT/'data/colab_b109';STATE=CON/'b109_remote.json'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b93_gpu'))
from atomic93 import save_json
def save(p,z):return save_json(p,z)
def obj(text):
    for line in reversed(text.splitlines()):
        try:return json.loads(line)
        except json.JSONDecodeError:pass
    return None
def short_run(cell,code):
    r=obj(update_run(cell,code))
    for _ in range(30):
        if r is not None:return r
        time.sleep(.3);cc=call('get_cells',{'includeOutputs':True})['cells'];c=next(x for x in cc if x['id']==cell)
        outputs=c.get('outputs',[]);assert not any(x.get('output_type')=='error' for x in outputs),'Inspect recorded Colab error'
        text=''.join(''.join(x.get('text',[])) if isinstance(x.get('text'),list) else x.get('text','') for x in outputs if x.get('output_type')=='stream');r=obj(text)
    raise RuntimeError('Short inspection produced no response; no more writes')
def main(resume_transport=False):
    old=json.loads(STATE.read_text());backup=CON/'b109_transfer_attempt1.json'
    if backup.exists():assert resume_transport,'Explicit transport resume required'
    else:shutil.copyfile(STATE,backup)
    raw=(OUT/'ces_b109_bundle.zip').read_bytes();expected=old['archive_sha256'];assert hashlib.sha256(raw).hexdigest()==expected
    state={**old,'status':'recovering_transport','recovery_started':time.time(),'recovery_helper':'PREREGISTRATION_B109_ADDENDUM2.md'};save(STATE,state);root=state['root'];cell=state['transfer_cell']
    inspect="from pathlib import Path\nimport json,hashlib\nroot=Path("+repr(root)+")\np=root/'bundle.part'\nassert not (root/'launch.json').exists(),'Already launched; refuse duplication'\nprint(json.dumps({'root_exists':root.exists(),'bytes':p.stat().st_size if p.exists() else 0,'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else hashlib.sha256(b'').hexdigest()}))"
    checked=short_run(cell,inspect);offset=checked['bytes'];assert offset<=len(raw) and checked['sha256']==hashlib.sha256(raw[:offset]).hexdigest()
    gpu_check()
    if not checked['root_exists']:
        assert not resume_transport,'Second workspace loss; abort'
        assert offset==0;assert short_run(cell,"from pathlib import Path\nimport json\nPath("+repr(root)+").mkdir(exist_ok=False)\nprint(json.dumps({'created':True}))")['created']
    state.update(uploaded=offset,verified_prefix_bytes=offset,status='uploading');save(STATE,state);ambiguities=0;chunk=1024*1024
    while offset<len(raw):
        encoded=base64.b64encode(raw[offset:offset+chunk]).decode()
        code="from pathlib import Path\nimport json,base64\np=Path("+repr(root+"/bundle.part")+")\nassert p.exists() or "+str(offset)+"==0\nassert (p.stat().st_size if p.exists() else 0)=="+str(offset)+"\nwith p.open('ab') as f:f.write(base64.b64decode("+repr(encoded)+"))\nprint(json.dumps({'bytes':p.stat().st_size}))"
        ack=obj(update_run(cell,code))
        if ack is None:
            ambiguities+=1;assert ambiguities<=3,'Repeated transfer ambiguity; no scientific retry'
            checked=short_run(cell,inspect);assert checked['root_exists'],'Second workspace loss; abort'
            n=checked['bytes'];assert offset<=n<=min(len(raw),offset+chunk) and checked['sha256']==hashlib.sha256(raw[:n]).hexdigest();offset=n
        else:assert ack['bytes']==min(len(raw),offset+chunk);offset=ack['bytes']
        state.update(uploaded=offset,transport_ambiguities=ambiguities);save(STATE,state);print('Verified transfer '+str(offset)+'/'+str(len(raw)),flush=True)
    code="from pathlib import Path\nimport hashlib,zipfile,json\nroot=Path("+repr(root)+")\np=root/'bundle.part'\nassert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(expected)+"\nwork=root/'bundle'\nwork.mkdir(exist_ok=False)\nwith zipfile.ZipFile(p) as z:\n    assert all(Path(n).name==n and not n.startswith('.') for n in z.namelist())\n    z.extractall(work)\nprint(json.dumps({'bundle_verified':True}))"
    assert short_run(cell,code)['bundle_verified']
    pins={'CES_DATA_DIR':root+'/bundle','CES_DROP_STUCK_TARGETS':'1','CES_TI_SPIKE_CUT_EV':'3000','CES_PER_SHOT_NORM':'0','CES_SEQ_NORM_EXPANDING':'0','CES_SEQ_ZERO_MC':'1','CES_SEQ_ZERO_VT_HISTORY':'1','CES_SEQ_ZERO_VT':'1','CES_SEQ_BATCH':'16','CES_LR':'.001','CES_SEQ_TRAIN_CTX':'0','CES_SEQ_EVAL_CTX':'0','CES_SEQ_LOSS':'mse','CES_SEQ_SELECT':'mse','CES_SEQ_FIXED_EPOCHS':'0','CES_SEQ_DEVICE':'cuda','CES_SEQ_SEED':'1701','PYTHONUNBUFFERED':'1'}
    launch="from pathlib import Path\nimport torch,json,subprocess,sys,time,os\nroot=Path("+repr(root)+")\nassert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)\nassert not (root/'launch.json').exists()\nlog=(root/'run.log').open('wb')\nenv={k:v for k,v in os.environ.items() if not k.startswith('CES_')}\nenv.update("+repr(pins)+")\np=subprocess.Popen([sys.executable,'-u',str(root/'bundle/train109.py')],cwd=root/'bundle',stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)\ninfo={'pid':p.pid,'started':time.time()}\n(root/'launch.json').write_text(json.dumps(info))\nprint(json.dumps(info))"
    state['remote_process']=short_run(cell,launch);state['status']='training';save(STATE,state)
    poll="from pathlib import Path\nimport json\nroot=Path("+repr(root)+")\np=root/'bundle/status.json'\ns=json.loads(p.read_text()) if p.exists() else {'state':'starting'}\nfor e in s.get('active',[]):\n    q=root/'bundle/runs'/e['tag']/'progress.json'\n    e['progress']=json.loads(q.read_text()) if q.exists() else None\ns['zip_ready']=(root/'bundle/b109_results.zip').exists()\ns['log_tail']=(root/'run.log').read_text(errors='replace')[-1200:]\nprint(json.dumps(s))"
    state['poll_cell']=new_cell(poll);save(STATE,state);print('Actual T4 training launched after hash-verified transport recovery.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume-transport',action='store_true');main(p.parse_args().resume_transport)
