"""CPU-only export of private real TRAIN/VAL workload and ready Colab notebook."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,sys,zipfile,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'data/colab_b108'
sys.path.insert(0,str(HERE.parent/'b107_patience'))
import common107 as C
import numpy as np,torch
from model106 import create_model
from prepare92 import batches
from train92 import vector_loss
from portable108 import TemperatureLSTM,loss108
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,z):Path(p).write_text(json.dumps(z,indent=2,ensure_ascii=False),encoding='utf8')
def main():
    C.OLD.guard();C.OLD.inputs_guard();torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    sources=[*HERE.glob('*.py'),HERE.parent/'PREREGISTRATION_B108.md']
    for p in sources:C.OLD.PREV.OLD.committed(p)
    assert not OUT.exists(),'Preserve prior export; no overwrite';OUT.mkdir();bundle=OUT/'bundle';bundle.mkdir();checks=[];local={}
    for tag,unit in zip('ABCD',C.UNITS):
        case=unit+'__outer';prep=C.OLD.prepared(case);C.OLD.verify_run(case);old=C.OLD.rd(case)
        original=create_model('LSTM_F',{'dense':True},1701);original.load_state_dict(torch.load(old/'best.pt',map_location='cpu',weights_only=True));original.eval()
        model=TemperatureLSTM();model.load_state_dict(original.state_dict());model.eval()
        arrays={}
        selections={'train':batches(prep['train']['D10'],'cpu',shuffle_seed=1701)[0],
                    'val':batches(prep['val']['D10'],'cpu')[0],
                    'long':batches(sorted(prep['train']['D10'],key=lambda s:len(s['x']),reverse=True)[:16],'cpu')[0]}
        for name,b in selections.items():
            for k in ('x','y','mask'):arrays[name+'_'+k]=b[k].numpy()
        x=selections['val']['x'][:1,:64]
        with torch.no_grad():
            ref=original(x);got=model(x);assert torch.allclose(ref,got,atol=1e-6,rtol=0)
        s=next(s for s in prep['train']['D10'] if s['mask'][:,0].any());n=min(len(s['x']),max(64,int(np.flatnonzero(s['mask'][:,0])[0])+1))
        xx=torch.from_numpy(s['x'][:n]).unsqueeze(0);y=torch.from_numpy(s['y'][:n]).unsqueeze(0);mask=torch.from_numpy(s['mask'][:n]).unsqueeze(0);stats=prep['identity']['stats']
        mu=torch.tensor(stats['ti_mu']);sd=torch.tensor(stats['ti_sd']);pred=model(xx).detach().requires_grad_(True)
        a=loss108(pred,y,mask,mu,sd);v=vector_loss(pred,y,mask,mu,sd);assert torch.allclose(a,v,atol=1e-7,rtol=1e-6)
        g1=torch.autograd.grad(a,pred,retain_graph=True)[0];g2=torch.autograd.grad(v,pred)[0];assert torch.allclose(g1,g2,atol=1e-7,rtol=1e-6)
        assert mask[:,:,0].any(),'Real prefix must contain F supervision'
        arrays.update(ref_x=x.numpy(),ref_prediction=ref.numpy(),mu=np.asarray(stats['ti_mu']),sd=np.asarray(stats['ti_sd']))
        np.savez_compressed(bundle/(tag+'.npz'),**arrays);torch.save(original.state_dict(),bundle/(tag+'.pt'))
        local[tag]={'unit':unit,'prepared_sha256':prep['artifact_sha256'],'checkpoint_sha256':sha(old/'best.pt'),'manifest_hash':C.digest(prep['identity']['manifest'])}
        checks.append({'unit':tag,'original_branch_equal':True,'loss_gradient_equal':True,'TEST_exported':False,'train_shape':list(arrays['train_x'].shape)})
        print('Exported real workload '+tag,flush=True)
    for n in ('portable108.py','probe108.py','run108.py'):shutil.copyfile(HERE/n,bundle/n)
    manifest={'registration':'PREREGISTRATION_B108.md','sources':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
              'files':{p.name:sha(p) for p in bundle.iterdir()},'checks':checks,'private_data':'Real TRAIN/VAL observations; keep the archive private. No identifiers or TEST data.'}
    save(bundle/'manifest.json',manifest);archive=OUT/'ces_t4_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in bundle.iterdir():z.write(p,p.name)
    code1="""from google.colab import files
from pathlib import Path
import hashlib, zipfile, os
uploaded = files.upload()  # select ces_t4_bundle.zip
name = next(n for n in uploaded if n.endswith('.zip'))
archive = Path(name)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'ARCHIVE_SHA', 'Different archive'
work = Path('/content/ces_b108')
work.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    assert all(Path(n).name == n and not n.startswith('.') for n in z.namelist())
    z.extractall(work)
del uploaded
print('Verified private bundle:', work)
""".replace('ARCHIVE_SHA',sha(archive))
    code2="""import subprocess, sys
p = subprocess.Popen([sys.executable, '-u', 'run108.py', '--expect-t4'], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end='')
assert p.wait() == 0, 'Input/runtime failure: inspect the output above'
"""
    cells=[('markdown','# T4 LSTM 속도·안정성 검증\n\n1. 런타임 → 런타임 유형 변경 → T4 GPU 선택.\n2. 모두 실행 후 ces_t4_bundle.zip 선택.\n3. 완료되면 b108_results.zip을 내려받아 전달.\n\n실제 TRAIN/VAL 기반의 짧은 실행 경로 검증입니다. 본 학습·TEST MSE 평가가 아닙니다. 개인 런타임에서만 사용하고 노트북/데이터를 공개하지 마세요. 실패한 경로도 로그에 남습니다. 예상 시간은 환경에 따라 달라지며 전체 측정 상한은 30분입니다.'),
           ('code',code1),('code',code2),('code',"import json\nprint(json.dumps(json.loads((work/'results/summary.json').read_text())['configurations'], indent=2))\nfiles.download(str(work/'b108_results.zip'))")]
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'COLAB_T4.ipynb'},'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}},
              'cells':[{'cell_type':kind,'id':'cell'+str(i),'metadata':{},'source':text.splitlines(keepends=True),**({'execution_count':None,'outputs':[]} if kind=='code' else {})} for i,(kind,text) in enumerate(cells)]}
    save(OUT/'COLAB_T4.ipynb',notebook)
    save(OUT/'export_verification.json',{'all_pass':True,'new_local_GPU_fits':0,'checks':checks,'archive_sha256':sha(archive),'archive_MiB':archive.stat().st_size/1024**2,'local_provenance':local,'notebook_sha256':sha(OUT/'COLAB_T4.ipynb')})
    print('Created COLAB_T4.ipynb and ces_t4_bundle.zip ('+str(round(archive.stat().st_size/1024**2,1))+' MiB)',flush=True)
if __name__=='__main__':main()
