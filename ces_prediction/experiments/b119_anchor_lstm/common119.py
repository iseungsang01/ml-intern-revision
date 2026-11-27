"""One registered visible-CES residual LSTM and audited direct-control inputs."""
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'b118_lstm_eval'))
import common118 as P
OUT = ROOT / 'data/.b119'
FIT = OUT / 'fits/U0_anchor_lstm_s1701'
UNIT, ARCH, SEED = P.UNIT, 'anchor_lstm', 1701
UNITS, RULES, TRAIN_RULES = (UNIT,), P.RULES, P.TRAIN_RULES
read, save, sha, npz = P.read, P.save, P.sha, P.npz
ENV_PINS = {**P.P.B.ENV_PINS, 'CES_B119_TARGET': 'F', 'CES_B119_SCOPE': 'ONE_ANCHOR_RESIDUAL_LSTM'}
test_grid_indices, test_coordinates = P.test_grid_indices, P.test_coordinates


def source_guard():
    prior = P.source_guard()
    assert (HERE.parent / 'PREREGISTRATION_B119.md').is_file()
    paths = [*sorted(HERE.glob('*.py')), *sorted(HERE.parent.glob('PREREGISTRATION_B119*.md'))]
    result = {}
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        data = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT, capture_output=True, check=True).stdout
        assert hashlib.sha256(data).hexdigest() == sha(path), 'Uncommitted B119 source: ' + rel
        result[rel] = sha(path)
    return {**prior, **result}


def verify_inputs():
    inherited = P.verify_inputs()
    initial = P.P.OUT
    initial_smoke = read(initial / 'smoke.json')
    initial_independent = read(initial / 'smoke_independent.json')
    assert initial_smoke['all_pass'] and initial_independent['all_pass']
    assert initial_smoke['training_updates'] == initial_independent['training_updates'] == 0
    assert initial_smoke['TEST_accessed'] is False and initial_independent['TEST_accessed'] is False
    assert initial_smoke['sources'] == initial_independent['sources'] == P.P.source_guard()
    assert initial_independent['smoke_sha256'] == sha(initial / 'smoke.json')
    for rel, expected in initial_smoke['files'].items():
        assert sha(initial / rel) == expected
    audit = read(P.OUT / 'audit.json')
    assert audit['all_pass'] and audit['sources'] == P.source_guard()
    assert sha(P.OUT / 'audit.json') == sha(ROOT / 'docs/paper/b118_audit.json')
    assert sha(P.OUT / 'readout.json') == audit['readout_sha256']
    report = read(P.OUT / 'readout.json')
    complete = read(P.OUT / 'predictions_complete.json')
    assert sha(P.OUT / 'predictions_complete.json') == report['predictions_complete_sha256']
    assert complete['all_pass'] and complete['sources'] == P.source_guard()
    baseline = {}
    for rule in RULES:
        rel = 'test/' + UNIT + '_' + rule + '.npz'
        assert sha(P.OUT / rel) == complete['files'][rel]
        baseline[rel] = complete['files'][rel]
    return {'inherited': inherited, 'B118_audit_sha256': sha(P.OUT / 'audit.json'),
        'original_initial_smoke_sha256': sha(initial / 'smoke.json'),
        'original_initial_independent_sha256': sha(initial / 'smoke_independent.json'),
        'original_initial_weights_sha256': sha(initial / 'smoke/weights.npz'),
        'original_initial_prefix_sha256': sha(initial / 'smoke/prefix.npz'),
        'B118_readout_sha256': sha(P.OUT / 'readout.json'),
        'B118_predictions_complete_sha256': sha(P.OUT / 'predictions_complete.json'),
        'direct_control_files': baseline}
