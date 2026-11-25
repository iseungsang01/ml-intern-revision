"""Source and data contract for the registered B112 Colab experiment."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'data/colab_b112'
UNITS=('V1_r2_k0','V2_s1_k0','V3_q3','V4a')
TRAIN_RULES=('D10','D50','D100','D300','D650','D1000','S');RULES=TRAIN_RULES+('K10',)
SEEDS=(1701,1702,1703)
ENV_PINS={'CES_DROP_STUCK_TARGETS':'1','CES_TI_SPIKE_CUT_EV':'3000','CES_PER_SHOT_NORM':'0','CES_SEQ_NORM_EXPANDING':'0','CES_SEQ_ZERO_MC':'1','CES_SEQ_ZERO_VT_HISTORY':'1','CES_SEQ_ZERO_VT':'1','CES_SEQ_DEVICE':'cuda','CES_B112_TARGET':'F','CES_B112_RULE':'MIXTURE7'}
MLPS=tuple('MLP_'+f+'_s'+str(s) for f in ('direct','residual') for s in SEEDS)
MODELS=('MLP_direct_ensemble','MLP_residual_ensemble',*MLPS,'HGB_direct','HGB_residual','GP_fast_boost','GP_fair_mixture','GP_fair_specialized','GP_history_boost','GP_only','hold','TCN22','TCN35','LSTM','LSTM_frozen_D10','LSTM_mixed_post')
PAIRS=(('MLP_residual_ensemble','MLP_direct_ensemble'),('HGB_residual','HGB_direct'))+tuple((a,b) for a in ('MLP_direct_ensemble','MLP_residual_ensemble') for b in ('HGB_residual','GP_fast_boost','GP_fair_mixture','GP_only','TCN35','LSTM'))+tuple((a,'HGB_residual') for a in MLPS)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,z):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(z,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sources():
    pp=[*sorted(HERE.glob('*.py')),*sorted(HERE.parent.glob('PREREGISTRATION_B112*.md'))]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in pp}
def guard():
    for rel in sources():
        subprocess.run(['git','cat-file','-e','HEAD:'+rel],cwd=ROOT,check=True);subprocess.run(['git','diff','--quiet','HEAD','--',rel],cwd=ROOT,check=True)
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','refs/remotes/origin/main'],cwd=ROOT,check=True)
