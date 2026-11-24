"""Bounded official-MCP delivery for frozen validation inference, no fitting."""
import argparse,base64,hashlib,json,sys,time
from pathlib import Path
from colab_t4_operator import call,new_cell,update_run,execute,gpu_check
ROOT=Path(__file__).resolve().parents[2]
CON=ROOT/'data/.colab_connection';OUT=ROOT/'data/colab_b111';STATE=CON/'b111_remote.json'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b93_gpu'))
from atomic93 import save_json

def start():
    assert not STATE.exists(),'Prior B111 attempt; inspect without duplicate launch'
    gpu=gpu_check();v=json.loads((OUT/'export_verification.json').read_text());raw=(OUT/'ces_b111_bundle.zip').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==v['archive_sha256']
    root='/content/ces_b111';cell=new_cell(f"from pathlib import Path\np=Path({root!r})\np.mkdir(exist_ok=False)\nprint('B111_READY')")
    assert 'B111_READY' in execute(cell)
    state={'status':'uploading','created':time.time(),'root':root,'cell':cell,'gpu':gpu,'archive_sha256':v['archive_sha256'],'uploaded':0};save_json(STATE,state)
    chunk=384*1024
    for offset in range(0,len(raw),chunk):
        b64=base64.b64encode(raw[offset:offset+chunk]).decode()
        code=f"from pathlib import Path\nimport base64,json\np=Path({(root+'/bundle.part')!r})\nassert (p.stat().st_size if p.exists() else 0)=={offset}\nwith p.open('ab') as f:f.write(base64.b64decode({b64!r}))\nprint(json.dumps({{'bytes':p.stat().st_size}}))"
        ack=json.loads(update_run(cell,code).strip());assert ack['bytes']==min(offset+chunk,len(raw));state['uploaded']=ack['bytes'];save_json(STATE,state)
        if offset%(chunk*10)==0 or ack['bytes']==len(raw):print('B111 upload '+str(ack['bytes'])+'/'+str(len(raw)),flush=True)
    verify=f"from pathlib import Path\nimport hashlib,zipfile\nroot=Path({root!r})\np=root/'bundle.part'\nassert hashlib.sha256(p.read_bytes()).hexdigest()=={v['archive_sha256']!r}\nwork=root/'bundle'\nwork.mkdir(exist_ok=False)\nwith zipfile.ZipFile(p) as z:\n    assert all(Path(n).name==n and not n.startswith('.') for n in z.namelist())\n    z.extractall(work)\nprint('B111_VERIFIED')"
    assert 'B111_VERIFIED' in update_run(cell,verify)
    pins={'CES_DROP_STUCK_TARGETS':'1','CES_TI_SPIKE_CUT_EV':'3000','CES_PER_SHOT_NORM':'0','CES_SEQ_NORM_EXPANDING':'0','CES_SEQ_ZERO_MC':'1','CES_SEQ_ZERO_VT_HISTORY':'1','CES_SEQ_ZERO_VT':'1','CES_SEQ_DEVICE':'cuda','CES_B111_TARGET':'F','CES_B111_RULE':'REGISTERED_REPLAY'}
    child="import subprocess,sys,json;from pathlib import Path;r=subprocess.run([sys.executable,'-u','run110.py']);Path('process_exit.json').write_text(json.dumps({'returncode':r.returncode}));sys.exit(r.returncode)"
    launch=f"from pathlib import Path\nimport subprocess,sys,os,json,time\nroot=Path({root!r})\nwork=root/'bundle'\nassert not (root/'launch.json').exists()\nenv={{k:v for k,v in os.environ.items() if not k.startswith('CES_')}}\nenv.update({pins!r})\nenv['PYTHONUNBUFFERED']='1'\nenv['OPENBLAS_NUM_THREADS']='2'\nenv['OMP_NUM_THREADS']='2'\nlog=(root/'run.log').open('wb')\np=subprocess.Popen([sys.executable,'-u','-c',{child!r}],cwd=work,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)\ninfo={{'pid':p.pid,'started':time.time()}}\n(root/'launch.json').write_text(json.dumps(info))\nprint(json.dumps(info))"
    state['launch']=json.loads(update_run(cell,launch).strip());state['status']='running';save_json(STATE,state)
    code=f"from pathlib import Path\nimport json\nroot=Path({root!r})\np=root/'bundle/progress.json'\ns=json.loads(p.read_text()) if p.exists() else {{'state':'starting'}}\ne=root/'bundle/process_exit.json'\ns['exit']=json.loads(e.read_text()) if e.exists() else None\ns['zip_ready']=(root/'bundle/b111_results.zip').exists()\ns['log_tail']=(root/'run.log').read_text(errors='replace')[-1600:]\nprint(json.dumps(s))"
    state['poll_cell']=new_cell(code);save_json(STATE,state);print('B111 T4 validation inference started; zero fits.',flush=True)

def poll():
    state=json.loads(STATE.read_text());p=json.loads(execute(state['poll_cell']).strip());save_json(CON/'b111_progress.json',{'observed':time.time(),**p});print(json.dumps(p),flush=True)
    if p.get('exit') is not None:assert p['exit']['returncode']==0,'Remote failure; no retry'
    if p['zip_ready'] and p.get('exit',{} ) and not state.get('downloaded'):
        cell=new_cell(f"from pathlib import Path\nimport hashlib,json\np=Path({(state['root']+'/bundle/b111_results.zip')!r})\nprint(json.dumps({{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}))")
        info=json.loads(execute(cell).strip());part=OUT/'b111_results.zip.part';dest=OUT/'b111_results.zip';assert not part.exists() and not dest.exists()
        for offset in range(0,info['bytes'],192*1024):
            code=f"from pathlib import Path\nimport base64\nwith Path({(state['root']+'/bundle/b111_results.zip')!r}).open('rb') as f:\n    f.seek({offset})\n    print(base64.b64encode(f.read({192*1024})).decode())"
            raw=base64.b64decode(update_run(cell,code).strip(),validate=True)
            with part.open('ab') as f:f.write(raw)
        assert part.stat().st_size==info['bytes'] and hashlib.sha256(part.read_bytes()).hexdigest()==info['sha256'];part.replace(dest)
        state.update(downloaded=True,status='downloaded',results_sha256=info['sha256']);save_json(STATE,state)
        call('update_cell',{'cellId':cell,'content':"print('B111 result archive retrieved.')"});print('B111 results hash verified.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('start','poll'));a=p.parse_args();globals()[a.action]()
