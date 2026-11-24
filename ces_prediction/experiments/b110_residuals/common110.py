"""Isolated validation diagnostic; no model fitting or TEST predictions."""
import hashlib,json,subprocess,zlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'data/colab_b110'
UNITS=('V1_r2_k0','V2_s1_k0','V3_q3','V4a')
MODELS=('LSTM','TCN22','TCN35','GP','hold','fair_specialized','fair_mixture')
GROUPS=('all','age_10ms','age_over_10ms','history_change_low','history_change_high')
PAIRS=tuple((x,'GP') for x in MODELS if x!='GP')+(('TCN35','fair_specialized'),('LSTM','fair_specialized'),('LSTM','TCN35'))
DRAW=20000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,z):Path(p).write_text(json.dumps(z,indent=2,allow_nan=False),encoding='utf8')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sources():
    paths=[*sorted(HERE.glob('*.py')),HERE.parent/'PREREGISTRATION_B110.md',ROOT/'ces_prediction/research/colab_b110_operator.py',HERE.parent/'b108_colab/portable108.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
def guard():
    for rel in sources():
        subprocess.run(['git','cat-file','-e','HEAD:'+rel],cwd=ROOT,check=True)
        subprocess.run(['git','diff','--quiet','HEAD','--',rel],cwd=ROOT,check=True)
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','refs/remotes/origin/main'],cwd=ROOT,check=True)
def group_masks(d):
    age=d['age'];change=d['history_change'];q=float(d['change_q75'])
    return (np.ones(len(age),bool),age<=10.000001,age>10.000001,change<=q,change>q)
def rng_for(unit,group,form):return np.random.default_rng([20261124,zlib.crc32((unit+'/'+group+'/'+form).encode())])
