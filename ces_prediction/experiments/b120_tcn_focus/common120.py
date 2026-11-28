"""One registered TCN optimization diagnostic; frozen B105 data/model contract."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'b105_merit'))
import common105 as B
import train92 as T

OUT = ROOT / 'data/.b120'
FIT = OUT / 'fit'
UNIT, SEED = 'V1_r2_k0', 1701
RULES = B.TRAIN_RULES
ENV = {**B.env(UNIT), 'CES_SEQ_EPOCHS': '200', 'CES_B120_OUTPUT': str(OUT),
       'CES_B120_SCOPE': 'ONE_TCN_TRAIN_VAL_ONLY'}
MIN_EPOCHS, MAX_EPOCHS, FIT_SECONDS = 50, 200, 6 * 3600
save, read, sha = B.save, B.load, B.sha
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def sources():
    frozen = B.guard()
    files = [*sorted(HERE.glob('*.py')), HERE.parent / 'PREREGISTRATION_B120.md']
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        data = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT, creationflags=FLAGS)
        assert hashlib.sha256(data).hexdigest() == sha(p), 'Uncommitted source: ' + rel
        frozen[rel] = sha(p)
    return frozen


def prepared():
    B.inputs_guard()
    z = B.prepared(UNIT)
    assert z['identity']['width'] == 35
    m = z['identity']['manifest']
    assert not (set(m['train']) & set(m['val']))
    assert not ((set(m['train']) | set(m['val'])) & set(m['test']))
    assert set(z['train']) == set(z['val']) == set(RULES)
    assert B.verify_run(UNIT)['epochs'] == 40
    return z


def event(phase, **extra):
    import time
    save(OUT / 'progress.json', {'phase': phase, 'updated': time.time(), **extra})


def atomic_pt(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)
