"""B90 fixed two-stage allocation and source contracts."""
import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'b88_anchor'))
import common88 as B88
import common85 as B85
DATA,PAPER,C=B85.DATA,B85.PAPER,B85.C
CACHE=DATA/'.b90'
METHODS=tuple('GTRCUVN')
INITS=(1701,1702,1703)
VARIANTS=('X','H')
RULES=tuple(B85.RULES)
select_units,load_sources=B85.select_units,B85.load_sources
REG=ROOT/'ces_prediction/experiments/PREREGISTRATION_B90.md'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf8');tmp.replace(path)
def assert_registered():
    assert REG.is_file(),'B90 registration missing'
    subprocess.run(['git','ls-files','--error-unmatch',str(REG.relative_to(ROOT))],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--quiet','HEAD','--',str(REG.relative_to(ROOT))],cwd=ROOT,check=True)
def assert_frozen():
    B88.assert_frozen()
    pins=json.loads((HERE/'frozen_sources90.json').read_text(encoding='utf8'))
    assert pins,'B88 pins absent'
    for rel,digest in pins.items():
        target=(ROOT/rel).resolve()
        assert not Path(rel).is_absolute() and target.is_relative_to(ROOT),'Unsafe frozen path'
        assert sha(target)==digest,'Frozen source changed: '+rel
    return {**B88.assert_frozen(),**pins}
def source_inventory():
    return {p.name:sha(p) for p in sorted(HERE.glob('*.py'))}
def screen_jobs():return [(m,u['name'],1701,v) for m in METHODS for u in select_units() for v in VARIANTS]
def promotion():
    p=CACHE/'promotion.json';assert p.is_file(),'Validation promotion not locked'
    record=json.loads(p.read_text(encoding='utf8'))
    assert record['execution_sources']==source_inventory(),'Stale promotion source binding'
    assert len(record['promoted'])==2 and len(set(record['promoted']))==2
    assert set(record['promoted'])<=set(METHODS)
    return record
def all_jobs():
    return screen_jobs()+[(m,u['name'],seed,v) for m in promotion()['promoted'] for u in select_units() for seed in INITS[1:] for v in VARIANTS]

