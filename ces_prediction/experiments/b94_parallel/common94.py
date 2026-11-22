"""B94 namespace, committed source guards and the original non-resettable clock."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=ROOT/'data';CACHE=DATA/'.b94'
BASE=HERE.parent/'b92_reduced';sys.path.insert(0,str(BASE))
sys.path.insert(0,str(HERE.parent/'b93_gpu'))
from atomic93 import save_json
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(z):return hashlib.sha256(json.dumps(z,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,z):
    return save_json(p,z)
def committed(p):
    rel=Path(p).relative_to(ROOT).as_posix()
    subprocess.run(['git','cat-file','-e','HEAD:'+rel],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--quiet','HEAD','--',rel],cwd=ROOT,check=True)
def guard():
    committed(ROOT/'ces_prediction/experiments/PREREGISTRATION_B94.md');pin=HERE/'frozen_sources94.json';committed(pin)
    pins=json.loads(pin.read_text());assert pins
    owned=list(HERE.glob('*.py'))+list((HERE/'lead_checks').glob('*.py'))
    for p in owned:committed(p)
    base_pin=BASE/'frozen_sources92.json';committed(base_pin)
    closure=json.loads(base_pin.read_text())
    assert all(pins.get(rel)==h for rel,h in closure.items()),'Original B92 numeric dependency closure must remain pinned'
    required={p.relative_to(ROOT).as_posix() for p in BASE.glob('*.py')}
    required|={p.relative_to(ROOT).as_posix() for p in owned}
    required.add(base_pin.relative_to(ROOT).as_posix())
    required|={(HERE.parent/'b93_gpu'/n).relative_to(ROOT).as_posix() for n in ('profile93.py','worker93.py')}
    required|={(HERE.parent/'b93_gpu'/n).relative_to(ROOT).as_posix() for n in ('durable93.py','status93.py','atomic93.py')}
    assert required<=set(pins),'All inherited execution modules must be pinned'
    for rel,h in pins.items():
        p=(ROOT/rel).resolve();assert not Path(rel).is_absolute() and p.is_relative_to(ROOT)
        assert sha(p)==h,'Frozen source changed: '+rel
    return pins
def sources():return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(HERE.glob('*.py'))}
def clock():
    original=DATA/'.b92/suite_clock.json';z=json.loads(original.read_text());return {'started':float(z['started']),'deadline':float(z['started'])+86400,'original_clock_sha256':sha(original),'sources':sources()}
def remaining():return max(0,clock()['deadline']-time.time())
