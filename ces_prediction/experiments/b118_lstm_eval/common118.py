"""B118 evaluation-only binding of one already selected, audited checkpoint."""
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'b117_lstm_focus'))
import common117 as P
OUT = ROOT / 'data/.b118'
UNIT, ARCH, SEED = P.UNIT, P.ARCH, P.SEED
UNITS, RULES, TRAIN_RULES = (UNIT,), P.RULES, P.TRAIN_RULES
FIT, PRIOR = P.FIT, P.OUT
read, save, sha, npz = P.read, P.save, P.sha, P.npz
ENV_PINS = {**P.ENV_PINS, 'CES_B118_MODE': 'EVAL_ONLY'}


def source_guard():
    previous = P.source_guard()
    paths = [*sorted(HERE.glob('*.py')), *sorted(HERE.parent.glob('PREREGISTRATION_B118*.md'))]
    assert (HERE.parent / 'PREREGISTRATION_B118.md').is_file()
    result = {}
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        data = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT, capture_output=True, check=True).stdout
        assert hashlib.sha256(data).hexdigest() == sha(path), 'Uncommitted B118 source: ' + rel
        result[rel] = sha(path)
    return {**previous, **result}


def verify_inputs():
    inherited = P.verify_inputs()
    P.B.verify_inputs()
    audit = read(PRIOR / 'audit.json')
    assert sha(PRIOR / 'audit.json') == sha(ROOT / 'docs/paper/b117_audit.json')
    assert audit['all_pass'] and audit['sources'] == P.source_guard()
    assert sha(PRIOR / 'readout.json') == audit['readout_sha256']
    report = read(PRIOR / 'readout.json')
    assert sha(PRIOR / 'fit_complete.json') == report['fit_complete_sha256']
    complete = read(PRIOR / 'fit_complete.json')
    assert complete['all_pass'] and complete['production_fits'] == 1 and complete['TEST_accessed'] is False
    assert complete['sources'] == P.source_guard()
    assert sha(FIT / 'params.json') == complete['params_sha256']
    for rel, expected in complete['files'].items():
        assert sha(PRIOR / rel) == expected, 'Audited B117 artifact drift: ' + rel
    recovery = audit['recovery_checkpoint']
    for key, name in (('latest_sha256', 'latest.pt'), ('best_sha256', 'best.npz')):
        assert recovery[key] == sha(FIT / name)
    for rel, expected in report['provenance'].items():
        assert sha(ROOT / rel) == expected
    old = ROOT / 'data/.b116'
    old_audit = read(old / 'audit.json')
    assert sha(old / 'audit.json') == sha(ROOT / 'docs/paper/b116_audit.json')
    assert old_audit['all_pass'] and sha(old / 'readout.json') == old_audit['readout_sha256']
    old_report = read(old / 'readout.json')
    assert sha(old / 'predictions_complete.json') == old_report['predictions_complete_sha256']
    prediction = read(old / 'predictions_complete.json')
    assert prediction['all_pass'] and prediction['sources'] == P.original_sources()
    assert sha(old / 'selection.json') == old_audit['selection_sha256'] == prediction['selection_sha256']
    assert sha(old / 'fit_complete.json') == old_audit['fit_complete_sha256'] == prediction['fit_complete_sha256']
    baseline = {}
    for rule in RULES:
        rel = 'test/' + UNIT + '_' + rule + '.npz'
        assert sha(old / rel) == prediction['files'][rel], 'Audited B116 prediction drift'
        baseline[rel] = prediction['files'][rel]
    return {'inherited': inherited, 'B117_audit_sha256': sha(PRIOR / 'audit.json'),
        'B117_readout_sha256': sha(PRIOR / 'readout.json'),
        'B117_fit_complete_sha256': sha(PRIOR / 'fit_complete.json'),
        'B117_params_sha256': sha(FIT / 'params.json'),
        'B117_best_sha256': sha(FIT / 'best.npz'), 'B117_latest_sha256': sha(FIT / 'latest.pt'),
        'B116_audit_sha256': sha(old / 'audit.json'),
        'B116_selection_sha256': sha(old / 'selection.json'),
        'B116_fit_complete_sha256': sha(old / 'fit_complete.json'),
        'B116_predictions_complete_sha256': sha(old / 'predictions_complete.json'),
        'baseline_test_files': baseline}


def test_grid_indices(idx):
    """Recover chronology privately; anonymous IDs must never define time order."""
    import data116 as D
    lookup, codes = D._row_mapping()
    inverse = np.empty(len(codes), dtype=np.int64)
    inverse[codes] = np.arange(len(codes))
    _, rows = D.raw_sources()
    result = np.asarray(rows['gi'], dtype=np.int64)[inverse[np.asarray(idx, dtype=np.int64)]]
    assert np.array_equal(D.anonymous_indices(result), np.asarray(idx, dtype=np.int64))
    assert len(np.unique(result)) == len(result)
    return result


def test_coordinates(idx, discharge):
    import data116 as D
    gi = test_grid_indices(idx)
    grid, _ = D.raw_sources()
    time_values = np.asarray(grid['time'][gi], dtype=np.float64)
    raw_segments = grid['seg'][gi]
    discharge = np.asarray(discharge)
    segment = np.empty(len(gi), dtype=np.int64)
    for group in np.unique(discharge):
        group_rows = np.flatnonzero(discharge == group)
        ordered = sorted(np.unique(raw_segments[group_rows]),
                         key=lambda s: float(time_values[group_rows[raw_segments[group_rows] == s]].min()))
        for number, raw in enumerate(ordered):
            segment[group_rows[raw_segments[group_rows] == raw]] = number
    assert np.isfinite(time_values).all()
    return {'time': time_values, 'segment': segment}
