"""Frozen B114 data/source contract; same audited B112 payload, no fits."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'data/.b114';PRIOR=ROOT/'data/colab_b112/bundle'
sys.path.insert(0,str(HERE.parent/'b113_catboost'))
import run113 as R
C=R.C
UNITS=C.UNITS;RULES=C.RULES;TRAIN_RULES=C.TRAIN_RULES;SEEDS=C.SEEDS
ENV_PINS={**C.ENV_PINS,'CES_B114_TARGET':'F','CES_B114_RULE':'MIXTURE7'}
read=C.read;save=C.write;sha=C.sha;npz=C.npz

def load_unit(tag):
 R.B.W=PRIOR
 return R.B.load_unit(tag)

def source_guard():
 paths=[*sorted(HERE.glob('*.py')),*sorted(HERE.parent.glob('PREREGISTRATION_B114*.md'))]
 result={}
 for p in paths:
  rel=p.relative_to(ROOT).as_posix();committed=subprocess.run(['git','show','HEAD:'+rel],cwd=ROOT,capture_output=True,check=True).stdout
  assert hashlib.sha256(committed).hexdigest()==sha(p),'Uncommitted source: '+rel
  result[rel]=sha(p)
 assert (HERE.parent/'PREREGISTRATION_B114.md').is_file()
 return result

def verify_inputs():
 m=R.verify_inputs()
 prior=ROOT/'data/.b113';a=read(prior/'audit.json');assert a['all_pass']
 published=read(ROOT/'docs/paper/b113_comparison.json');assert published['independent_audit']==a
 complete=read(prior/'complete.json');assert complete['all_pass']
 assert a['complete_sha256']==sha(prior/'complete.json') and a['started_sha256']==sha(prior/'started.json')
 for rel,h in complete['estimates'].items():assert sha(prior/rel)==h,'B113 reference changed: '+rel
 return m
