"""B117 one-fit source and TRAIN/VAL input contract."""
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'b116_temporal'))
import common116 as B
OUT = ROOT / 'data/.b117_focused'
FIT = OUT / 'fits/U0_lstm_s1701'
UNIT, ARCH, SEED = 'V1_r2_k0', 'lstm', 1701
UNITS, RULES, TRAIN_RULES = (UNIT,), B.RULES, B.TRAIN_RULES
read, save, sha, npz = B.read, B.save, B.sha, B.npz
ENV_PINS = {**B.ENV_PINS, 'CES_B117_TARGET': 'F', 'CES_B117_SCOPE': 'ONE_U0_LSTM_TRAIN_VAL'}


def original_sources():
    z = read(ROOT / 'data/.b116/workflow/started.json')['sources']
    for rel, expected in z.items():
        assert sha(ROOT / rel) == expected, 'Frozen source drift: ' + rel
    return z


def source_guard():
    original = original_sources()
    paths = [*sorted(HERE.glob('*.py')), *sorted(HERE.parent.glob('PREREGISTRATION_B117*.md'))]
    assert (HERE.parent / 'PREREGISTRATION_B117.md').is_file()
    result = {}
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        data = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT, capture_output=True, check=True).stdout
        assert hashlib.sha256(data).hexdigest() == sha(path), 'Uncommitted B117 source: ' + rel
        result[rel] = sha(path)
    return {**original, **result}


def verify_inputs():
    import data116 as D
    D.source_guard()
    original_out = ROOT / 'data/.b116'
    audit = read(original_out / 'audit.json')
    assert audit['all_pass']
    assert sha(original_out / 'fit_complete.json') == audit['fit_complete_sha256']
    fit = read(original_out / 'fit_complete.json')
    assert fit['all_pass'] and fit['sources'] == original_sources()
    old = ROOT / 'data/.b116/fits' / UNIT / ARCH / ('s' + str(SEED))
    p = read(old / 'params.json')
    assert p['normal'] and p['completed_epochs'] == 17 and p['sources'] == original_sources()
    assert sha(old / 'history.json') == p['history_sha256']
    assert sha(old / 'best.npz') == p['best_sha256']
    receipt = [r for r in fit['fits'] if (r['unit'], r['architecture'], r['seed']) == (UNIT, ARCH, SEED)]
    assert len(receipt) == 1 and receipt[0]['normal']
    assert sha(old / 'params.json') == receipt[0]['sha256']
    assert (original_out / receipt[0]['path']).resolve() == (old / 'params.json').resolve()
    baseline_val = {}
    for rule in TRAIN_RULES:
        rel = 'val/' + UNIT + '_' + rule + '.npz'
        assert sha(original_out / rel) == fit['files'][rel], 'Original VAL baseline drift: ' + rel
        baseline_val[rel] = fit['files'][rel]
    smoke = read(ROOT / 'data/.b116/smoke.json')
    independent = read(ROOT / 'data/.b116/smoke_independent.json')
    assert smoke['all_pass'] and independent['all_pass']
    assert smoke['sources'] == independent['sources'] == original_sources()
    assert independent['smoke_sha256'] == sha(ROOT / 'data/.b116/smoke.json')
    for item in smoke['architectures'].values():
        for rel, expected in item['files'].items():
            assert sha(ROOT / 'data/.b116' / rel) == expected
    prepared = D.load_prepared(UNIT)
    assert prepared['artifact_sha256'] == p['prepared_artifact_sha256']
    return {'original_params_sha256': sha(old / 'params.json'),
            'original_history_sha256': sha(old / 'history.json'),
            'prepared_artifact_sha256': prepared['artifact_sha256'],
            'original_audit_sha256': sha(original_out / 'audit.json'),
            'original_fit_complete_sha256': sha(original_out / 'fit_complete.json'),
            'baseline_val_files': baseline_val}
