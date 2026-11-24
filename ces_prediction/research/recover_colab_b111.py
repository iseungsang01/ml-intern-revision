"""Exact-byte transport recovery; never retries scientific inference or fitting."""
import argparse,base64,hashlib,json,shutil,sys,time
from pathlib import Path
from colab_t4_operator import call,execute,new_cell,update_run
ROOT=Path(__file__).resolve().parents[2];CON=ROOT/'data/.colab_connection';OUT=ROOT/'data/colab_b111';STATE=CON/'b111_remote.json'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b93_gpu'))
from atomic93 import save_json

def object_from(text):
    for line in reversed(text.splitlines()):
        try:return json.loads(line)
        except json.JSONDecodeError:pass
    return None

def short(cell,code,required):
    # Confirm source is short before any read that includes cell sources.
    assert len(code)<12000
    call('update_cell',{'cellId':cell,'content':code})
    try:result=object_from(execute(cell))
    except AssertionError:result=None
    if isinstance(result,dict) and all(k in result for k in required):return result
    for _ in range(10):
        cc=call('get_cells',{'includeOutputs':True})['cells'];c=next(x for x in cc if x['id']==cell)
        outputs=c.get('outputs',[]);assert not any(o.get('output_type')=='error' for o in outputs),'Inspect Colab execution error'
        text=''.join(''.join(o.get('text',[])) if isinstance(o.get('text'),list) else o.get('text','') for o in outputs if o.get('output_type')=='stream');result=object_from(text)
        if isinstance(result,dict) and all(k in result for k in required):return result
        time.sleep(.5)
    raise RuntimeError('No acknowledged short inspection; do not write more')

def main(resume_transport=False):
    old=json.loads(STATE.read_text());assert old['status']=='uploading' and 'launch' not in old and 'poll_cell' not in old
    backup=CON/'b111_transfer_attempt1.json'
    if backup.exists():assert resume_transport,'Explicit --resume-transport required after inspection'
    else:shutil.copyfile(STATE,backup)
    raw=(OUT/'ces_b111_bundle.zip').read_bytes();expected=hashlib.sha256(raw).hexdigest();assert expected==old['archive_sha256']
    cell=old['cell'];root=old['root'];state={**old,'recovery_started':time.time(),'recovery_helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'operational_addendum_sha256':hashlib.sha256((ROOT/'ces_prediction/experiments/PREREGISTRATION_B111_ADDENDUM1.md').read_bytes()).hexdigest()}
    inspect=f"from pathlib import Path\nimport json,hashlib,torch\nroot=Path({root!r})\np=root/'bundle.part'\nassert not (root/'launch.json').exists() and not (root/'bundle/started.json').exists(),'Scientific execution exists; refuse recovery'\nprint(json.dumps({{'exists':root.exists(),'bytes':p.stat().st_size if p.exists() else 0,'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else hashlib.sha256(b'').hexdigest(),'T4':torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)}}))"
    info=short(cell,inspect,('exists','T4','bytes','sha256'));assert info['T4'];offset=info['bytes'];assert offset<=len(raw) and hashlib.sha256(raw[:offset]).hexdigest()==info['sha256']
    if not info['exists']:
        assert offset==0;assert short(cell,f"from pathlib import Path\nimport json\nPath({root!r}).mkdir(exist_ok=False)\nprint(json.dumps({{'created':True}}))",('created',))['created']
    state.update(workspace_was_lost=not info['exists'],verified_prefix_bytes=offset,uploaded=offset,status='uploading');save_json(STATE,state)
    chunk=1024*1024;ambiguities=0
    while offset<len(raw):
        blob=base64.b64encode(raw[offset:offset+chunk]).decode();end=min(len(raw),offset+chunk)
        code=f"from pathlib import Path\nimport base64,json\np=Path({(root+'/bundle.part')!r})\nassert (p.stat().st_size if p.exists() else 0)=={offset}\nwith p.open('ab') as f:f.write(base64.b64decode({blob!r}))\nprint(json.dumps({{'bytes':p.stat().st_size}}))"
        ack=object_from(update_run(cell,code))
        if ack is None:
            ambiguities+=1;assert ambiguities<=3;info=short(cell,inspect,('exists','T4','bytes','sha256'));assert info['exists'] and offset<=info['bytes']<=end
            assert hashlib.sha256(raw[:info['bytes']]).hexdigest()==info['sha256'];offset=info['bytes']
        else:assert ack['bytes']==end;offset=end
        state.update(uploaded=offset,transport_ambiguities=ambiguities);save_json(STATE,state);print('B111 verified transfer '+str(offset)+'/'+str(len(raw)),flush=True)
    verify=f"from pathlib import Path\nimport zipfile,hashlib,json\nroot=Path({root!r})\np=root/'bundle.part'\nassert hashlib.sha256(p.read_bytes()).hexdigest()=={expected!r}\nwork=root/'bundle'\nwork.mkdir(exist_ok=False)\nwith zipfile.ZipFile(p) as z:\n    assert all(Path(n).name==n and not n.startswith('.') for n in z.namelist())\n    z.extractall(work)\nprint(json.dumps({{'verified':True}}))"
    assert short(cell,verify,('verified',))['verified']
    pins={'CES_DROP_STUCK_TARGETS':'1','CES_TI_SPIKE_CUT_EV':'3000','CES_PER_SHOT_NORM':'0','CES_SEQ_NORM_EXPANDING':'0','CES_SEQ_ZERO_MC':'1','CES_SEQ_ZERO_VT_HISTORY':'1','CES_SEQ_ZERO_VT':'1','CES_SEQ_DEVICE':'cuda','CES_B111_TARGET':'F','CES_B111_RULE':'REGISTERED_REPLAY'}
    child="import subprocess,sys,json;from pathlib import Path;r=subprocess.run([sys.executable,'-u','run111.py']);Path('process_exit.json').write_text(json.dumps({'returncode':r.returncode}));sys.exit(r.returncode)"
    launch=f"from pathlib import Path\nimport subprocess,sys,os,json,time,torch\nroot=Path({root!r})\nassert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)\nassert not (root/'launch.json').exists() and not (root/'bundle/started.json').exists()\nenv={{k:v for k,v in os.environ.items() if not k.startswith('CES_')}}\nenv.update({pins!r})\nenv.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')\nlog=(root/'run.log').open('wb')\np=subprocess.Popen([sys.executable,'-u','-c',{child!r}],cwd=root/'bundle',stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)\ninfo={{'pid':p.pid,'started':time.time()}}\n(root/'launch.json').write_text(json.dumps(info))\nprint(json.dumps(info))"
    state['launch']=short(cell,launch,('pid','started'));state['status']='running';save_json(STATE,state)
    poll=f"from pathlib import Path\nimport json\nroot=Path({root!r})\np=root/'bundle/progress.json'\ns=json.loads(p.read_text()) if p.exists() else {{'state':'starting'}}\ne=root/'bundle/process_exit.json'\ns['exit']=json.loads(e.read_text()) if e.exists() else None\ns['zip_ready']=(root/'bundle/b111_results.zip').exists()\ns['log_tail']=(root/'run.log').read_text(errors='replace')[-1600:]\nprint(json.dumps(s))"
    state['poll_cell']=new_cell(poll);save_json(STATE,state);print('B111 OOF inference -> eight post fits -> all-scenario comparison launched.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume-transport',action='store_true');main(p.parse_args().resume_transport)
