"""B115 frozen data/source contract, no fits."""
import hashlib,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'data/.b115'
sys.path.insert(0,str(HERE.parent/'b114_overnight'))
import common114 as B
PRIOR=B.PRIOR;SEEDS=B.SEEDS;RULES=B.RULES;TRAIN_RULES=B.TRAIN_RULES;UNITS=B.UNITS
ENV_PINS={**B.ENV_PINS,'CES_B115_TARGET':'F','CES_B115_TREATMENT':'LONG_SCHEDULE'}
read=B.read;save=B.save;sha=B.sha;npz=B.npz;load_unit=B.load_unit

def source_guard():
 result={}
 for p in [*sorted(HERE.glob('*.py')),*sorted(HERE.parent.glob('PREREGISTRATION_B115*.md'))]:
  rel=p.relative_to(ROOT).as_posix();data=subprocess.run(['git','show','HEAD:'+rel],cwd=ROOT,capture_output=True,check=True).stdout
  assert hashlib.sha256(data).hexdigest()==sha(p),'Uncommitted source '+rel
  result[rel]=sha(p)
 assert (HERE.parent/'PREREGISTRATION_B115.md').is_file()
 audit=read(ROOT/'data/.b114/audit.json');assert audit['all_pass']
 for rel,h in audit['sources'].items():assert sha(ROOT/rel)==h,'Frozen B114 source changed'
 return {**audit['sources'],**result}

def verify_inputs():
 m=B.verify_inputs();a=read(ROOT/'data/.b114/audit.json');z=read(ROOT/'docs/paper/b114_comparison.json');assert z['independent_audit']==a and a['all_pass']
 assert sha(ROOT/'data/.b114/selection.json')==a['selection_sha256']
 for f in ('cat','neural'):
  base=ROOT/'data/.b114'/f;c=read(base/'prediction_complete.json');assert c['all_pass']
  for rel,h in c['files'].items():assert sha(base/rel)==h
 return m
