"""B116 committed-source and original temporal-input contract; no fits."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'data/.b116'
PRIOR = ROOT / 'data/colab_b112/bundle'
UNITS = ('V1_r2_k0', 'V2_s1_k0', 'V3_q3', 'V4a')
SEEDS = (1701, 1702, 1703)
TRAIN_RULES = ('D10', 'D50', 'D100', 'D300', 'D650', 'D1000', 'S')
RULES = TRAIN_RULES + ('K10',)
ENV_PINS = {'CES_DATA_DIR': str(ROOT / 'data'), 'CES_DROP_STUCK_TARGETS': '1',
    'CES_TI_SPIKE_CUT_EV': '3000', 'CES_PER_SHOT_NORM': '0',
    'CES_SEQ_NORM_EXPANDING': '0', 'CES_SEQ_ZERO_MC': '1',
    'CES_SEQ_ZERO_VT_HISTORY': '1', 'CES_SEQ_ZERO_VT': '1',
    'CES_SEQ_BATCH': '16', 'CES_SEQ_DEVICE': 'cuda', 'CES_SEQ_TRAIN_CTX': '0',
    'CES_SEQ_EVAL_CTX': '0', 'CES_SEQ_LOSS': 'mse', 'CES_SEQ_SELECT': 'mse',
    'CES_LR': '.001', 'CES_SEQ_FIXED_EPOCHS': '0',
    'CES_B116_TARGET': 'F', 'CES_B116_RULE': 'MIXTURE7',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def source_guard():
    registration = HERE.parent / 'PREREGISTRATION_B116.md'
    assert registration.is_file(), 'Committed preregistration required'
    paths = [*sorted(HERE.glob('*.py')), *sorted(HERE.parent.glob('PREREGISTRATION_B116*.md'))]
    result = {}
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        committed = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT,
                                   capture_output=True, check=True).stdout
        assert hashlib.sha256(committed).hexdigest() == sha(path), 'Uncommitted source: ' + rel
        result[rel] = sha(path)
    return result


def verify_inputs():
    import data116 as D
    D.source_guard()
    manifest = read(PRIOR / 'manifest.json')
    assert manifest['registration'] == 'PREREGISTRATION_B112.md'
    assert manifest['target'] == 'current_report_F'
    assert tuple(v['name'] for v in manifest['units'].values()) == UNITS
    for name, expected in manifest['files'].items():
        assert Path(name).name == name
        assert sha(PRIOR / name) == expected, 'B112 artifact drift: ' + name
    for rel, expected in manifest['sources'].items():
        assert sha(ROOT / rel) == expected, 'B112 source drift: ' + rel
    assert read(ROOT / 'data/colab_b112/independent_audit.json')['all_pass']
    provenance = D.C105.prov()
    for unit in UNITS:
        item = provenance['prepared'][unit]
        assert sha(ROOT / item['path']) == item['sha256'], 'Prepared sequence drift: ' + unit
    D._row_mapping()
    return manifest
