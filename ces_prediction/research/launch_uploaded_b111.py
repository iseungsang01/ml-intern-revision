"""Launch the frozen B111 archive delivered through Colab's native upload UI."""
import hashlib,json,sys,time
from pathlib import Path
from colab_t4_operator import new_cell,execute
ROOT=Path(__file__).resolve().parents[2];CON=ROOT/'data/.colab_connection';OUT=ROOT/'data/colab_b111';STATE=CON/'b111_remote.json'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b93_gpu'))
from atomic93 import save_json

def main():
    old=json.loads(STATE.read_text());assert 'launch' not in old and 'poll_cell' not in old,'Existing scientific launch; inspect'
    archive=OUT/'ces_b111_bundle.zip';v=json.loads((OUT/'export_verification.json').read_text());expected=hashlib.sha256(archive.read_bytes()).hexdigest();assert expected==v['archive_sha256']
    root='/content/ces_b111';upload='/content/ces_b111_bundle.zip';pins={'CES_DROP_STUCK_TARGETS':'1','CES_TI_SPIKE_CUT_EV':'3000','CES_PER_SHOT_NORM':'0','CES_SEQ_NORM_EXPANDING':'0','CES_SEQ_ZERO_MC':'1','CES_SEQ_ZERO_VT_HISTORY':'1','CES_SEQ_ZERO_VT':'1','CES_SEQ_DEVICE':'cuda','CES_B111_TARGET':'F','CES_B111_RULE':'REGISTERED_REPLAY'}
    child="import subprocess,sys,json;from pathlib import Path;r=subprocess.run([sys.executable,'-u','run111.py']);Path('process_exit.json').write_text(json.dumps({'returncode':r.returncode}));sys.exit(r.returncode)"
    code=f"from pathlib import Path\nimport hashlib,json,zipfile,torch,subprocess,sys,time,os\nassert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0),'Wait for actual T4 allocation'\nroot=Path({root!r})\np=Path({upload!r})\nassert p.stat().st_size=={archive.stat().st_size}\nassert hashlib.sha256(p.read_bytes()).hexdigest()=={expected!r}\nassert not (root/'launch.json').exists() and not (root/'bundle/started.json').exists()\nroot.mkdir(exist_ok=True)\nwork=root/'bundle'\nwork.mkdir(exist_ok=False)\nwith zipfile.ZipFile(p) as z:\n    assert all(Path(n).name==n and not n.startswith('.') for n in z.namelist())\n    z.extractall(work)\nenv={{k:v for k,v in os.environ.items() if not k.startswith('CES_')}}\nenv.update({pins!r})\nenv.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')\nlog=(root/'run.log').open('wb')\np=subprocess.Popen([sys.executable,'-u','-c',{child!r}],cwd=work,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)\ninfo={{'pid':p.pid,'started':time.time(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'archive_verified':True}}\n(root/'launch.json').write_text(json.dumps(info))\nprint(json.dumps(info))"
    backup=CON/'b111_before_browser_upload_launch.json';assert not backup.exists(),'Prior launch attempt; inspect without repetition';save_json(backup,old)
    cell=new_cell(code);save_json(CON/'b111_browser_launch_attempt.json',{'cell':cell,'created':time.time(),'archive_sha256':expected})
    launch=json.loads(execute(cell).strip());assert launch['archive_verified']
    state={**old,'status':'running','uploaded':archive.stat().st_size,'root':root,'launch':launch,'delivery':'Colab native file upload','operational_helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'operational_addendum_sha256':hashlib.sha256((ROOT/'ces_prediction/experiments/PREREGISTRATION_B111_ADDENDUM3.md').read_bytes()).hexdigest()}
    save_json(STATE,state)
    poll=f"from pathlib import Path\nimport json\nroot=Path({root!r})\np=root/'bundle/progress.json'\ns=json.loads(p.read_text()) if p.exists() else {{'state':'starting'}}\ne=root/'bundle/process_exit.json'\ns['exit']=json.loads(e.read_text()) if e.exists() else None\ns['zip_ready']=(root/'bundle/b111_results.zip').exists()\ns['log_tail']=(root/'run.log').read_text(errors='replace')[-1600:]\nprint(json.dumps(s))"
    state['poll_cell']=new_cell(poll);save_json(STATE,state);print('B111 started: T4 OOF inference -> eight Colab post fits -> all32 scenario cells.',flush=True)

if __name__=='__main__':main()
