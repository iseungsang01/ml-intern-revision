"""Claude-origin B88 contracts; local path/pinning compatibility repairs."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'b85_unified'))
import common85 as _c85
DATA,PAPER,C=_c85.DATA,_c85.PAPER,_c85.C
CACHE=DATA/'.b88'
INITS=list(_c85.INITS)
VARIANTS=['A','C']
RULES=list(_c85.RULES)
TRAIN_RULES=list(_c85.TRAIN_RULES)
load_sources=_c85.load_sources
select_units=_c85.select_units
def assert_frozen():
    pins=json.loads((HERE/'frozen_b85_sources.json').read_text(encoding='utf8'))
    assert pins,'Empty source inventory'
    for rel,expected in pins.items():
        p=(ROOT/rel).resolve()
        assert p.is_relative_to(ROOT) and not Path(rel).is_absolute(),'Unsafe source pin'
        assert p.is_file(),'Missing frozen source: '+rel
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,'Frozen source changed: '+rel
    return pins
