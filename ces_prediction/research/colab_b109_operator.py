"""Registered T4 training delivery and bounded aggregate/checkpoint retrieval."""
import argparse,base64,hashlib,json,time
from pathlib import Path
from colab_t4_operator import call,cells,new_cell,update_run,execute,gpu_check
ROOT=Path(__file__).resolve().parents[2];CON=ROOT/'data/.colab_connection';OUT=ROOT/'data/colab_b109';STATE=CON/'b109_remote.json'
def save(p,z):t=p.with_suffix('.tmp');t.write_text(json.dumps(z,indent=2),encoding='utf8');t.replace(p)
def start():
    assert not STATE.exists(),'Prior B109 attempt; inspect, no duplicate launch';gpu=gpu_check();archive=OUT/'ces_b109_bundle.zip';v=json.loads((OUT/'export_verification.json').read_text());raw=archive.read_bytes();assert hashlib.sha256(raw).hexdigest()==v['archive_sha256']
    root='/content/ces_b109';transfer=new_cell(f"from pathlib import Path\nimport torch\nassert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)\nPath({root!r}).mkdir(exist_ok=False)\nprint('B109_READY')");assert 'B109_READY' in execute(transfer)
    state={'created':time.time(),'root':root,'gpu':gpu,'archive_sha256':v['archive_sha256'],'transfer_cell':transfer,'status':'uploading','uploaded':0};save(STATE,state)
    chunk=384*1024
    for offset in range(0,len(raw),chunk):
        encoded=base64.b64encode(raw[offset:offset+chunk]).decode();code=f"from pathlib import Path\nimport base64,json\np=Path({(root+'/bundle.part')!r})\nassert (p.stat().st_size if p.exists() else 0)=={offset}\nwith p.open('ab') as f:f.write(base64.b64decode({encoded!r}))\nprint(json.dumps({{'bytes':p.stat().st_size}}))"
        ack=json.loads(update_run(transfer,code).strip());assert ack['bytes']==min(len(raw),offset+chunk);state['uploaded']=ack['bytes'];save(STATE,state)
        if offset%(chunk*10)==0 or ack['bytes']==len(raw):print('B109 uploaded '+str(ack['bytes'])+'/'+str(len(raw)),flush=True)
    code=f"from pathlib import Path\nimport hashlib,zipfile,torch,subprocess,sys,json,time,os\nroot=Path({root!r})\np=root/'bundle.part'\nassert hashlib.sha256(p.read_bytes()).hexdigest()=={v['archive_sha256']!r}\nwork=root/'bundle'\nwork.mkdir(exist_ok=False)\nwith zipfile.ZipFile(p) as z:\n    assert all(Path(n).name==n and not n.startswith('.') for n in z.namelist())\n    z.extractall(work)\nassert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)\nassert not (root/'launch.json').exists()\nlog=(root/'run.log').open('wb')\nenv={{k:v for k,v in os.environ.items() if not k.startswith('CES_')}}\nenv.update({{'CES_DATA_DIR':str(work),'CES_DROP_STUCK_TARGETS':'1','CES_TI_SPIKE_CUT_EV':'3000','CES_PER_SHOT_NORM':'0','CES_SEQ_NORM_EXPANDING':'0','CES_SEQ_ZERO_MC':'1','CES_SEQ_ZERO_VT_HISTORY':'1','CES_SEQ_ZERO_VT':'1','CES_SEQ_BATCH':'16','CES_LR':'.001','CES_SEQ_TRAIN_CTX':'0','CES_SEQ_EVAL_CTX':'0','CES_SEQ_LOSS':'mse','CES_SEQ_SELECT':'mse','CES_SEQ_FIXED_EPOCHS':'0','CES_SEQ_DEVICE':'cuda','CES_SEQ_SEED':'1701','PYTHONUNBUFFERED':'1'}})\np=subprocess.Popen([sys.executable,'-u',str(work/'train109.py')],cwd=work,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)\ninfo={{'pid':p.pid,'started':time.time()}}\n(root/'launch.json').write_text(json.dumps(info))\nprint(json.dumps(info))"
    launch=json.loads(update_run(transfer,code).strip());state.update(status='training',remote_process=launch)
    poll=f"from pathlib import Path\nimport json\nroot=Path({root!r})\np=root/'bundle/status.json'\ns=json.loads(p.read_text()) if p.exists() else {{'state':'starting'}}\nfor e in s.get('active',[]):\n    x=root/'bundle/runs'/e['tag']/'progress.json'\n    e['progress']=json.loads(x.read_text()) if x.exists() else None\ns['zip_ready']=(root/'bundle/b109_results.zip').exists()\ns['log_tail']=(root/'run.log').read_text(errors='replace')[-1200:]\nprint(json.dumps(s))"
    state['poll_cell']=new_cell(poll);save(STATE,state);print('Nine remaining T4 training paths launched; local GPU remains paused.',flush=True)
def poll():
    state=json.loads(STATE.read_text());progress=json.loads(execute(state['poll_cell']).strip());save(CON/'b109_progress.json',{'observed':time.time(),**progress});print(json.dumps(progress,indent=2))
    if progress['zip_ready'] and not state.get('downloaded'):
        cell=new_cell(f"from pathlib import Path\nimport json,hashlib\np=Path({(state['root']+'/bundle/b109_results.zip')!r})\nprint(json.dumps({{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}))")
        info=json.loads(execute(cell).strip());dest=OUT/'b109_colab_results.zip';assert not dest.exists();part=OUT/'b109_colab_results.zip.part';assert not part.exists()
        for offset in range(0,info['bytes'],192*1024):
            code=f"from pathlib import Path\nimport base64\nwith Path({(state['root']+'/bundle/b109_results.zip')!r}).open('rb') as f:\n    f.seek({offset})\n    print(base64.b64encode(f.read({192*1024})).decode('ascii'))"
            raw=base64.b64decode(update_run(cell,code).strip(),validate=True)
            with part.open('ab') as f:f.write(raw)
        assert part.stat().st_size==info['bytes'] and hashlib.sha256(part.read_bytes()).hexdigest()==info['sha256'];part.replace(dest)
        state.update(status='results_downloaded',downloaded=True,results_sha256=info['sha256']);save(STATE,state)
        call('update_cell',{'cellId':cell,'content':"print('B109 validated states downloaded locally.')"});print('B109 checkpoint archive downloaded and hash verified.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('start','poll'));a=p.parse_args();globals()[a.action]()
