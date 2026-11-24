"""Shared data contract and constants for the frozen full comparator."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'data/colab_b111'
UNITS=('V1_r2_k0','V2_s1_k0','V3_q3','V4a');TRAIN_RULES=('D10','D50','D100','D300','D650','D1000','S');RULES=TRAIN_RULES+('K10',)
MODELS=('LSTM','LSTM_frozen_D10','LSTM_mixed_post','GP_history_boost','GP_fast_boost','TCN22','TCN35','GP_fair_mixture','GP_fair_specialized','GP_only','hold')
PAIRS=tuple((a,b) for a in MODELS[:3] for b in ('GP_fast_boost','TCN35','GP_only'))+(('LSTM_frozen_D10','LSTM'),('LSTM_mixed_post','LSTM'),('GP_fast_boost','GP_history_boost'),('TCN22','GP_fast_boost'),('TCN35','GP_fast_boost'),('TCN35','TCN22'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,z):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(z,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sources():
    pp=[*sorted(HERE.glob('*.py')),HERE.parent/'PREREGISTRATION_B111.md',HERE.parent/'b108_colab/portable108.py',ROOT/'ces_prediction/research/colab_b111_operator.py',ROOT/'ces_prediction/research/watch_colab_b111.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in pp}
def guard():
    for rel in sources():
        subprocess.run(['git','cat-file','-e','HEAD:'+rel],cwd=ROOT,check=True);subprocess.run(['git','diff','--quiet','HEAD','--',rel],cwd=ROOT,check=True)
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','refs/remotes/origin/main'],cwd=ROOT,check=True)
def inputs(d,stats,gi,rule):
    x=np.zeros((len(gi),22),np.float32)
    x[:,:13]=(d['fast'][gi].astype(np.float32)-np.array(stats['fast_mu']))/np.array(stats['fast_sd'])
    x[:,15]=d['dt_log'][gi];a=d['anchor_'+rule][gi];ok=a>=0
    x[ok,16]=(d['ti'][a[ok]]-stats['ti_mu'])/stats['ti_sd'];x[ok,17]=d['age_log_'+rule][gi[ok]];x[ok,18]=1
    assert np.isfinite(x).all();return x
def features(fast,tstep,h,gp,age,idx,est):
    return np.column_stack((fast[idx],tstep[idx],age[idx],h[idx],est-h[idx],est-gp[idx],np.abs(est-h[idx]),np.abs(est-gp[idx])))
