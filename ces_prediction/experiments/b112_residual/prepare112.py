"""Generate the registered one-cell Colab entrypoint; does not fit anything."""
import time,uuid
from common112 import *
def main():
    guard();v=read(OUT/'export_verification.json');assert v['all_pass']
    archive=OUT/'ces_b112_bundle.zip';h=sha(archive);assert h==v['archive_sha256']
    assert not (OUT/'standalone_attempt.json').exists(),'Preserve prepared attempt'
    attempt=uuid.uuid4().hex
    code=f'''# B112_OWNED_{attempt}
import os,sys,json,time,hashlib,zipfile,subprocess
from pathlib import Path
import torch
from google.colab import files
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0),'Allocated T4 required'
root=Path('/content/ces_b112')
assert not root.exists(),'Existing B112 workspace; inspect instead of launching twice'
print('Select ces_b112_bundle.zip only.',flush=True)
os.chdir('/content')
uploaded=files.upload()
assert set(uploaded)=={{'ces_b112_bundle.zip'}}
p=Path('/content/ces_b112_bundle.zip')
assert p.stat().st_size=={archive.stat().st_size} and hashlib.sha256(p.read_bytes()).hexdigest()=='{h}'
root.mkdir();work=root/'bundle';work.mkdir()
with zipfile.ZipFile(p) as z:
    assert all(Path(n).name==n and not n.startswith('.') for n in z.namelist())
    z.extractall(work)
m=json.loads((work/'manifest.json').read_text())
for name,digest in m['files'].items():
    assert hashlib.sha256((work/name).read_bytes()).hexdigest()==digest
env={{k:v for k,v in os.environ.items() if not k.startswith('CES_')}}
env.update({ENV_PINS!r})
env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
log=(root/'run.log').open('wb')
proc=subprocess.Popen([sys.executable,'-u','run112.py'],cwd=work,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
info={{'pid':proc.pid,'started':time.time(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'attempt':'{attempt}','archive_sha256':'{h}','archive_verified':True}}
(root/'launch.json').write_text(json.dumps(info))
print('B112_LAUNCHED='+json.dumps(info),flush=True)
last=None
while proc.poll() is None:
    status=work/'progress.json'
    if status.exists():
        try:
            current=status.read_text()
            if current!=last:print(current,flush=True);last=current
        except (ValueError,OSError):pass
    time.sleep(15)
log.close()
assert proc.returncode==0,'B112 failed; preserve run.log, do not rerun'
result=work/'b112_results.zip';assert result.exists()
receipt={{**info,'returncode':proc.returncode,'manifest_sha256':hashlib.sha256((work/'manifest.json').read_bytes()).hexdigest()}}
with zipfile.ZipFile(result,'a',zipfile.ZIP_DEFLATED) as z:
    assert 'standalone_delivery.json' not in z.namelist()
    z.writestr('standalone_delivery.json',json.dumps(receipt,indent=2))
print('B112_DOWNLOAD_READY bytes='+str(result.stat().st_size)+' sha256='+hashlib.sha256(result.read_bytes()).hexdigest(),flush=True)
files.download(str(result))
'''
    notebook={'nbformat':4,'nbformat_minor':0,'metadata':{'accelerator':'GPU','colab':{'gpuType':'T4'},'kernelspec':{'display_name':'Python 3','name':'python3'}},'cells':[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    write(OUT/'COLAB_B112.ipynb',notebook)
    write(OUT/'standalone_attempt.json',{'attempt':attempt,'prepared':time.time(),'archive_sha256':h,'manifest_sha256':sha(OUT/'bundle/manifest.json'),'notebook_sha256':sha(OUT/'COLAB_B112.ipynb'),'expected_download':str(Path.home()/'Downloads/b112_results.zip'),'scientific_process_started':False})
    print('B112 notebook prepared; no fits started.',flush=True)
if __name__=='__main__':main()
