"""Source-bound B92 allocation; no test access in pilot/selection helpers."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
for d in ('b88_anchor','b90_methods'):sys.path.insert(0,str(HERE.parent/d))
import common88 as B88
import common85 as B85
DATA,PAPER,C=B85.DATA,B85.PAPER,B85.C;CACHE=DATA/'.b92'
RULES=tuple(B85.RULES);TRAIN_RULES=tuple(B85.TRAIN_RULES)
PILOT_UNITS=('V1_r2_k0','V2_s1_k0','V3_q3','V4a')
select_units,load_sources=B85.select_units,B85.load_sources
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf8');tmp.replace(path)
def guard():
    reg=ROOT/'ces_prediction/experiments/PREREGISTRATION_B92.md'
    subprocess.run(['git','diff','--quiet','HEAD','--',str(reg.relative_to(ROOT))],cwd=ROOT,check=True)
    subprocess.run(['git','cat-file','-e','HEAD:'+reg.relative_to(ROOT).as_posix()],cwd=ROOT,check=True)
    pin_path=HERE/'frozen_sources92.json'
    subprocess.run(['git','cat-file','-e','HEAD:'+pin_path.relative_to(ROOT).as_posix()],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--quiet','HEAD','--',str(pin_path.relative_to(ROOT))],cwd=ROOT,check=True)
    pins=json.loads(pin_path.read_text(encoding='utf8'));assert pins
    for rel,h in pins.items():
        p=(ROOT/rel).resolve();assert not Path(rel).is_absolute() and p.is_relative_to(ROOT)
        assert sha(p)==h,'Frozen source changed: '+rel
    return pins
def sources():return {p.name:sha(p) for p in sorted(HERE.glob('*.py'))}
def pilot_jobs():return [('pilot',u,1701,c) for u in PILOT_UNITS for c in ('C','A','T')]
def pair(chosen):return ('A','C') if chosen=='A' else ('TX','TH')
def confirmation_jobs(chosen):
    codes=pair(chosen)
    return [('confirm',u['name'],1701,c) for u in select_units() for c in codes]+[('confirm',u,s,c) for u in PILOT_UNITS for s in (1702,1703) for c in codes]
def run_dir(stage,unit,seed,code):return CACHE/stage/f'{unit}__{seed}__{code}'
def selection():
    z=json.loads((CACHE/'selection.json').read_text(encoding='utf8'));assert z['sources']==sources(),'Stale selection';return z
