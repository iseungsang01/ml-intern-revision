"""Prepare one concrete upload/run cell; never execute it or launch a fit."""
import hashlib,json,sys,time,uuid
from pathlib import Path
from colab_t4_operator import cells,source,call,new_cell
ROOT=Path(__file__).resolve().parents[2];CON=ROOT/'data/.colab_connection';OUT=ROOT/'data/colab_b111'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b93_gpu'))
from atomic93 import save_json

def main():
    state=json.loads((CON/'b111_remote.json').read_text());assert 'launch' not in state and 'poll_cell' not in state
    attempt=CON/'b111_manual_attempt.json';assert not attempt.exists(),'Keep original prepared cell; inspect rather than duplicate'
    archive=OUT/'ces_b111_bundle.zip';rawhash=hashlib.sha256(archive.read_bytes()).hexdigest();assert rawhash==state['archive_sha256']
    token=uuid.uuid4().hex;root='/content/ces_b111';pins={'CES_DROP_STUCK_TARGETS':'1','CES_TI_SPIKE_CUT_EV':'3000','CES_PER_SHOT_NORM':'0','CES_SEQ_NORM_EXPANDING':'0','CES_SEQ_ZERO_MC':'1','CES_SEQ_ZERO_VT_HISTORY':'1','CES_SEQ_ZERO_VT':'1','CES_SEQ_DEVICE':'cuda','CES_B111_TARGET':'F','CES_B111_RULE':'REGISTERED_REPLAY'}
    child="import subprocess,sys,json;from pathlib import Path;r=subprocess.run([sys.executable,'-u','run111.py']);Path('process_exit.json').write_text(json.dumps({'returncode':r.returncode}));sys.exit(r.returncode)"
    code=f'''# B111_MANUAL_OWNED_{token}
# Run once, then select ces_b111_bundle.zip when the upload widget appears.
import os,sys,json,time,hashlib,zipfile,subprocess
from pathlib import Path
import torch
from google.colab import files
root=Path({root!r})
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Wait for an allocated T4 runtime before running this cell.'
assert not (root/'launch.json').exists() and not (root/'bundle/started.json').exists(), 'Already launched; do not run twice.'
print('Select the existing ces_b111_bundle.zip file. No other file is needed.',flush=True)
os.chdir('/content')
uploaded=files.upload()
assert set(uploaded)=={{'ces_b111_bundle.zip'}}, 'Wrong archive; no scientific process started.'
p=Path('/content/ces_b111_bundle.zip')
assert p.stat().st_size=={archive.stat().st_size}
assert hashlib.sha256(p.read_bytes()).hexdigest()=={rawhash!r}, 'Archive mismatch; no scientific process started.'
root.mkdir(exist_ok=True)
work=root/'bundle'
work.mkdir(exist_ok=False)
with zipfile.ZipFile(p) as z:
    assert all(Path(n).name==n and not n.startswith('.') for n in z.namelist())
    z.extractall(work)
m=json.loads((work/'manifest.json').read_text())
for name,h in m['files'].items():
    assert hashlib.sha256((work/name).read_bytes()).hexdigest()==h
env={{k:v for k,v in os.environ.items() if not k.startswith('CES_')}}
env.update({pins!r})
env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
log=(root/'run.log').open('wb')
proc=subprocess.Popen([sys.executable,'-u','-c',{child!r}],cwd=work,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
info={{'pid':proc.pid,'started':time.time(),'root':str(root),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'archive_sha256':{rawhash!r},'attempt':{token!r},'archive_verified':True}}
(root/'launch.json').write_text(json.dumps(info))
print('B111_LAUNCHED='+json.dumps(info),flush=True)
print('OOF inference -> eight post-correction fits -> eight-scenario evaluation. Automatic retrieval/audit will attach.',flush=True)
'''
    (OUT/'COLAB_B111_CELL.py').write_text(code,encoding='utf8')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','colab':{'gpuType':'T4'},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(True)}]}
    (OUT/'COLAB_B111.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf8')
    cc=cells();owned=[c for c in cc if c['id']=='lIYdn1woOS1n' and ('cuda_available' in source(c) or not source(c).strip())]
    if len(owned)==1:
        cell=owned[0]['id'];call('update_cell',{'cellId':cell,'content':code})
    else:cell=new_cell(code)
    stored=next(source(c) for c in cells() if c['id']==cell);assert code.strip() in stored
    save_json(attempt,{'cell':cell,'attempt':token,'source_sha256':hashlib.sha256(stored.encode()).hexdigest(),'archive_sha256':rawhash,'root':root,'created':time.time(),'preparer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print('B111 upload/run cell prepared. No kernel execution or fit started. Cell='+cell,flush=True)
if __name__=='__main__':main()
