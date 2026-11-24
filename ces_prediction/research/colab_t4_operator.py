"""Deliver the registered private bundle via already-connected official MCP."""
import argparse,base64,hashlib,json,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CON=ROOT/'data/.colab_connection';OUT=ROOT/'data/colab_b108';STATE=CON/'b108_remote.json'
def save(p,z):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(z,indent=2),encoding='utf8');tmp.replace(p)
def call(name,arguments):
    q=CON/'requests'/('108_'+str(time.time_ns())+'_'+uuid.uuid4().hex[:8]+'.json');save(q,{'action':'call','name':name,'arguments':arguments});r=q.with_suffix('.result.json');deadline=time.monotonic()+180
    while not r.exists():
        assert time.monotonic()<deadline,'MCP request not completed; inspect without repeating '+q.name;time.sleep(.1)
    z=json.loads(r.read_text());assert z['ok'],'MCP transport error; inspect '+r.name
    response=z['response'];assert not response.get('isError'),'MCP tool error; inspect '+r.name
    data=response.get('structuredContent')
    if data is None:
        text=''.join(c.get('text','') for c in response.get('content',[]) if c.get('type')=='text');data=json.loads(text) if text else {}
    return data
def cells():return call('get_cells',{'includeOutputs':False})['cells']
def source(c):return ''.join(c.get('source',[])) if isinstance(c.get('source'),list) else c.get('source','')
def execute(cell):
    data=call('run_code_cell',{'cellId':cell});outputs=data.get('outputs',[])
    assert not any(o.get('output_type')=='error' for o in outputs),'Colab code error; inspect recorded MCP result'
    return ''.join(''.join(o.get('text',[])) if isinstance(o.get('text'),list) else o.get('text','') for o in outputs if o.get('output_type')=='stream')
def update_run(cell,code):call('update_cell',{'cellId':cell,'content':code});return execute(cell)
def new_cell(code):
    marker='# B108_OWNED_'+uuid.uuid4().hex;code=marker+'\n'+code;before=cells();call('add_code_cell',{'cellIndex':len(before),'language':'python','code':code})
    found=[c for c in cells() if marker in source(c)];assert len(found)==1;return found[0]['id']
def gpu_check():
    found=[c for c in cells() if '"cuda_available"' in source(c) and 'get_device_name' in source(c)]
    assert len(found)==1,'Expected owned diagnostic cell'
    text=execute(found[0]['id']);rows=[]
    for line in text.splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    info=next(x for x in rows if isinstance(x,dict) and 'cuda_available' in x)
    save(CON/'gpu_check.json',{'time':time.time(),**info});assert info['cuda_available'] and 'T4' in (info['gpu'] or ''),'Select T4 GPU in the connected notebook first'
    return info
def start():
    assert not STATE.exists(),'Remote B108 already attempted; inspect state before any continuation'
    gpu=gpu_check();archive=OUT/'ces_t4_bundle.zip';expected=json.loads((OUT/'export_verification.json').read_text())['archive_sha256'];raw=archive.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected
    remote='/content/ces_b108_mcp';setup=f"from pathlib import Path\nimport json, torch\nassert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)\np=Path({remote!r})\np.mkdir(exist_ok=False)\nprint('B108_STAGE_READY')"
    transfer=new_cell(setup);assert 'B108_STAGE_READY' in execute(transfer)
    state={'created':time.time(),'gpu':gpu,'root':remote,'transfer_cell':transfer,'archive_sha256':expected,'uploaded_bytes':0,'status':'uploading'};save(STATE,state)
    chunk_size=384*1024
    for offset in range(0,len(raw),chunk_size):
        blob=base64.b64encode(raw[offset:offset+chunk_size]).decode('ascii')
        code=f"from pathlib import Path\nimport base64,json\np=Path({(remote+'/bundle.zip.part')!r})\nassert (p.stat().st_size if p.exists() else 0)=={offset}\nwith p.open('ab') as f: f.write(base64.b64decode({blob!r}))\nprint(json.dumps({{'uploaded':p.stat().st_size}}))"
        text=update_run(transfer,code);ack=json.loads(text.strip());assert ack['uploaded']==min(len(raw),offset+chunk_size)
        state['uploaded_bytes']=ack['uploaded'];save(STATE,state);print('Uploaded '+str(ack['uploaded'])+'/'+str(len(raw)),flush=True)
    code=f"from pathlib import Path\nimport hashlib,zipfile,json\np=Path({(remote+'/bundle.zip.part')!r})\nassert hashlib.sha256(p.read_bytes()).hexdigest()=={expected!r}\nwork=Path({(remote+'/bundle')!r})\nwork.mkdir(exist_ok=False)\nwith zipfile.ZipFile(p) as z:\n    assert all(Path(n).name==n and not n.startswith('.') for n in z.namelist())\n    z.extractall(work)\nprint('B108_BUNDLE_VERIFIED')"
    assert 'B108_BUNDLE_VERIFIED' in update_run(transfer,code);state['status']='uploaded';save(STATE,state)
    pins={'CES_DATA_DIR':remote+'/bundle','CES_DROP_STUCK_TARGETS':'1','CES_TI_SPIKE_CUT_EV':'3000','CES_PER_SHOT_NORM':'0','CES_SEQ_NORM_EXPANDING':'0','CES_SEQ_ZERO_MC':'1','CES_SEQ_ZERO_VT_HISTORY':'1','CES_SEQ_ZERO_VT':'1','CES_SEQ_BATCH':'16','CES_LR':'.001','CES_SEQ_TRAIN_CTX':'0','CES_SEQ_EVAL_CTX':'0','CES_SEQ_LOSS':'mse','CES_SEQ_SELECT':'mse','CES_SEQ_FIXED_EPOCHS':'0','CES_SEQ_DEVICE':'cuda','CES_SEQ_SEED':'1701'}
    launch=f"from pathlib import Path\nimport json,subprocess,sys,time,torch,os\nassert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)\nroot=Path({remote!r})\nassert not (root/'launch.json').exists()\nlog=(root/'run.log').open('wb')\nenv={{k:v for k,v in os.environ.items() if not k.startswith('CES_')}}\nenv.update({pins!r})\np=subprocess.Popen([sys.executable,'-u',str(root/'bundle/run108.py'),'--expect-t4'],cwd=root/'bundle',stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)\ninfo={{'pid':p.pid,'started':time.time()}}\n(root/'launch.json').write_text(json.dumps(info))\nprint(json.dumps(info))"
    state['launch_cell']=new_cell(launch);save(STATE,state);state['remote_process']=json.loads(execute(state['launch_cell']).strip());state['status']='running'
    poll=f"from pathlib import Path\nimport json,os\nroot=Path({remote!r})\ninfo=json.loads((root/'launch.json').read_text())\ntry:\n    os.kill(info['pid'],0)\n    alive=True\nexcept ProcessLookupError:\n    alive=False\npartial=root/'bundle/results/partial.json'\ngroups=json.loads(partial.read_text()) if partial.exists() else []\nprint(json.dumps({{'pid':info['pid'],'alive':alive,'groups_completed':len(groups),'last_group':groups[-1]['name'] if groups else None,'zip_ready':(root/'bundle/b108_results.zip').exists(),'log_tail':(root/'run.log').read_text(errors='replace')[-1200:]}}))"
    state['poll_cell']=new_cell(poll);save(STATE,state);print('B108 T4 qualification launched; no local GPU work.',flush=True)
def poll():
    state=json.loads(STATE.read_text());text=execute(state['poll_cell']);progress=json.loads(text.strip());save(CON/'b108_progress.json',{'observed':time.time(),**progress});print(json.dumps(progress,indent=2))
    if progress['zip_ready'] and not state.get('downloaded'):
        code=f"from pathlib import Path\nimport base64\nprint(base64.b64encode(Path({(state['root']+'/bundle/b108_results.zip')!r}).read_bytes()).decode('ascii'))"
        download=new_cell(code);encoded=execute(download).strip();raw=base64.b64decode(encoded,validate=True);dest=OUT/'b108_colab_results.zip';assert not dest.exists();dest.write_bytes(raw)
        state.update(status='results_downloaded',downloaded=True,results_sha256=hashlib.sha256(raw).hexdigest());save(STATE,state)
        call('update_cell',{'cellId':download,'content':"# B108 aggregate result archive retrieved by the local operator.\nprint('Results retrieved.')"})
        print('Aggregate results downloaded locally.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('check','start','poll'));a=p.parse_args()
    if a.action=='check':print(json.dumps(gpu_check(),indent=2))
    else:globals()[a.action]()
