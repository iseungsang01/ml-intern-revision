"""Read-only independent preflight of B116 paused scientific artifacts; no fit."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'b116_temporal'))
import common116 as C
import data116 as D
import audit116 as A

ARCHS = ('lstm', 'decay_gru', 'innovation_gru')
PARTIAL = 'fits/V4a/lstm/s1701'


def preflight():
    started = C.read(C.OUT / 'workflow/started.json')
    sources = started['sources']
    for rel, digest in sources.items():
        assert C.sha(ROOT / rel) == digest, 'Frozen source drift: ' + rel
        committed = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT,
                                   capture_output=True, check=True).stdout
        assert hashlib.sha256(committed).hexdigest() == digest, 'Frozen HEAD drift: ' + rel
    status = C.read(C.OUT / 'workflow/status.json')
    assert status['state'] == 'paused_by_author' and status['completed_fits'] == 27
    assert status['owned_processes_stopped'] and not status['automatic_resume']
    assert status['partial_fits'] == [PARTIAL]
    assert status['original_deadlines_preserved']
    assert time.time() > started['CPU_end'], 'Original boundary has not elapsed'
    # No selection or scored B116 TEST material may already exist.
    forbidden = ['fit_complete.json', 'selection.json', 'predict_started.json',
                 'predictions_complete.json', 'readout.json', 'audit.json']
    assert all(not (C.OUT / name).exists() for name in forbidden)
    assert not (C.OUT / 'test').exists() and not (C.OUT / 'val').exists()
    C.verify_inputs()
    snapshot, checked, pending = {}, [], []
    for unit in C.UNITS:
        prepared = D.load_prepared(unit)
        A.verify_normalization(prepared)
        for arch in ARCHS:
            for seed in C.SEEDS:
                folder = C.OUT / 'fits' / unit / arch / ('s' + str(seed))
                if (folder / 'params.json').exists():
                    params = A.verify_history(unit, arch, seed, prepared)
                    assert params['sources'] == sources
                    assert unit != 'V4a', 'Unexpected completed fourth-unit artifact'
                    weights = C.npz(folder / 'best.npz')
                    import numpy as np
                    assert weights and all(np.isfinite(v).all() for v in weights.values())
                    start = C.read(folder / 'started.json')
                    assert start['sources'] == sources
                    for p in sorted(folder.iterdir()):
                        assert p.is_file(), 'Unexpected nested fit artifact'
                        snapshot[p.relative_to(C.OUT).as_posix()] = C.sha(p)
                    checked.append({'unit': unit, 'architecture': arch, 'seed': seed,
                                    'epochs': params['completed_epochs'], 'best_epoch': params['best_epoch']})
                else:
                    pending.append('/'.join((unit, arch, 's' + str(seed))))
                    if folder.exists():
                        assert folder.relative_to(C.OUT).as_posix() == PARTIAL
                        assert {p.name for p in folder.iterdir()} == {'history.json', 'started.json'}
                        history = C.read(folder / 'history.json')
                        assert history and all(e['epoch'] == i for i, e in enumerate(history, 1))
                        assert C.read(folder / 'started.json')['sources'] == sources
    assert len(checked) == 27 and len(pending) == 9
    assert all(p.startswith('V4a/') for p in pending)
    partial = {p.relative_to(C.OUT).as_posix(): C.sha(p) for p in sorted((C.OUT / PARTIAL).iterdir())}
    smoke = C.read(C.OUT / 'smoke_independent.json')
    assert smoke['all_pass'] and smoke['sources'] == sources
    assert smoke['smoke_sha256'] == C.sha(C.OUT / 'smoke.json')
    result = {'all_pass': True, 'read_only': True, 'scientific_fits': 0, 'model_inference': 0,
              'created_unix': time.time(), 'sources': sources, 'completed_count': len(checked),
              'completed': checked, 'preserved_files': snapshot, 'pending': pending,
              'partial': partial, 'original_GPU_end': started['GPU_end'],
              'original_CPU_end': started['CPU_end'], 'old_TEST_accessed': False,
              'scope': 'Raw/prepared provenance, TRAIN normalization, all27 histories/checkpoints/schedules/assignments and no-TEST barrier'}
    return result



def final_verification():
    """Independently bind original27, resumed9 and archived interrupted artifacts."""
    from datetime import datetime
    import numpy as np
    original = C.read(C.OUT / 'workflow/started.json')
    prepath = C.OUT / 'workflow/resume_independent_precheck.json'
    before = C.read(prepath)
    w = C.OUT / 'workflow/resume1'
    resumed = C.read(w / 'started.json')
    assert before['all_pass'] and before['completed_count'] == 27
    pause_path = C.OUT / 'workflow/pause_author_20261126.json'
    pause = C.read(pause_path)
    assert pause['state'] == 'paused_by_author' and pause['completed_fits'] == 27
    assert pause['partial_fits'] == [PARTIAL] and pause['owned_processes_stopped']
    assert before['sources'] == original['sources'] == resumed['sources']
    assert resumed['original_started_sha256'] == C.sha(C.OUT / 'workflow/started.json')
    assert before['created_unix'] <= resumed['started']
    assert resumed['GPU_end'] == resumed['started'] + 6 * 3600
    assert resumed['CPU_end'] == resumed['started'] + 8 * 3600
    assert resumed['started'] > original['CPU_end']
    for rel, expected in before['sources'].items():
        assert C.sha(ROOT / rel) == expected, 'Frozen source drift'
        blob = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT, capture_output=True, check=True).stdout
        assert hashlib.sha256(blob).hexdigest() == expected
    paths = [*sorted(HERE.glob('*.py')), HERE.parent / 'PREREGISTRATION_B116_ADDENDUM1.md']
    new_sources = {}
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        blob = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT, capture_output=True, check=True).stdout
        assert hashlib.sha256(blob).hexdigest() == C.sha(path), 'Uncommitted continuation source'
        new_sources[rel] = C.sha(path)
    assert resumed['resume_sources'] == new_sources
    for rel, expected in before['preserved_files'].items():
        assert C.sha(C.OUT / rel) == expected, 'Original completed artifact changed'
    preserved_folders = {str(Path(rel).parent).replace('\\', '/') for rel in before['preserved_files']}
    for rel in preserved_folders:
        actual_files = {p.relative_to(C.OUT).as_posix() for p in (C.OUT / rel).rglob('*') if p.is_file()}
        expected_files = {p for p in before['preserved_files'] if str(Path(p).parent).replace('\\', '/') == rel}
        assert actual_files == expected_files, 'Original completed file set changed'
    expected_dirs = {f'fits/{u}/{a}/s{s}' for u in C.UNITS for a in ARCHS for s in C.SEEDS}
    actual_dirs = {p.relative_to(C.OUT).as_posix() for p in (C.OUT / 'fits').glob('*/*/s*') if p.is_dir()}
    assert actual_dirs == expected_dirs, 'Unexpected production fit directory'
    archive = w / 'archive/V4a_lstm_s1701_interrupted'
    assert {p.name for p in archive.iterdir()} == {'history.json', 'started.json'}
    for oldrel, expected in before['partial'].items():
        assert C.sha(archive / Path(oldrel).name) == expected, 'Interrupted partial artifact changed'
    checked, new_files = [], {}
    C.verify_inputs()
    for unit in C.UNITS:
        prepared = D.load_prepared(unit)
        for arch in ARCHS:
            for seed in C.SEEDS:
                params = A.verify_history(unit, arch, seed, prepared)
                assert params['sources'] == before['sources']
                folder = C.OUT / 'fits' / unit / arch / ('s' + str(seed))
                assert all(np.isfinite(v).all() for v in C.npz(folder / 'best.npz').values())
                if unit == 'V4a':
                    start = C.read(folder / 'started.json')
                    assert start['sources'] == before['sources']
                    assert datetime.fromisoformat(start['utc']).timestamp() >= resumed['started']
                    assert datetime.fromisoformat(params['completed_utc']).timestamp() <= resumed['GPU_end']
                    checked.append('/'.join((unit, arch, 's' + str(seed))))
                    for p in sorted(folder.iterdir()):
                        assert p.is_file()
                        new_files[p.relative_to(C.OUT).as_posix()] = C.sha(p)
    assert sorted(checked) == sorted(before['pending']) and len(checked) == 9
    fit = C.read(C.OUT / 'fit_complete.json')
    assert fit['all_pass'] and not fit['test_accessed'] and len(fit['fits']) == 36
    expected_keys = {(u, a, s) for u in C.UNITS for a in ARCHS for s in C.SEEDS}
    assert {(r['unit'], r['architecture'], r['seed']) for r in fit['fits']} == expected_keys
    for receipt in fit['fits']:
        assert receipt['normal'] and C.sha(C.OUT / receipt['path']) == receipt['sha256']
    pred = C.read(C.OUT / 'predict_started.json')
    done = C.read(C.OUT / 'predictions_complete.json')
    sel = C.read(C.OUT / 'selection.json')
    stamp = lambda value: datetime.fromisoformat(value).timestamp()
    assert stamp(fit['utc']) <= sel['created_unix'] <= stamp(pred['utc']) <= stamp(done['utc']) <= resumed['GPU_end']
    assert not sel['TEST_used']
    assert pred['fit_complete_sha256'] == C.sha(C.OUT / 'fit_complete.json')
    assert pred['selection_sha256'] == done['selection_sha256'] == C.sha(C.OUT / 'selection.json')
    numerical = C.read(C.OUT / 'audit.json')
    assert numerical['all_pass'] and numerical['sources'] == before['sources']
    assert numerical['selection_sha256'] == C.sha(C.OUT / 'selection.json')
    for name in ('fit', 'select', 'predict', 'readout', 'audit'):
        stage = C.read(w / ('stage_' + name + '_started.json'))
        assert resumed['started'] <= stage['started'] <= resumed['CPU_end']
        assert stage['sources'] == new_sources
    return {'all_pass': True, 'created_unix': time.time(), 'independent_precheck_sha256': C.sha(prepath),
            'scientific_sources': before['sources'], 'resume_sources': new_sources,
            'preserved27_files': before['preserved_files'], 'resumed9_files': new_files,
            'archived_partial_files': before['partial'], 'resumed_groups': checked,
            'numerical_audit_sha256_before_resume_attachment': C.sha(C.OUT / 'audit.json'),
            'resume_started_sha256': C.sha(w / 'started.json'), 'original_started_sha256': C.sha(C.OUT / 'workflow/started.json'), 'pause_sha256': C.sha(pause_path),
            'scope': 'Independent byte preservation, exact resumed fit set, source identities, original data guards, checkpoint histories and chronology'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    result = final_verification() if args.final else preflight()
    target = C.OUT / 'resume_independent_final.json' if args.final else C.OUT / 'workflow/resume_independent_precheck.json'
    C.save(target, result)
    print('B116 independent continuation ' + ('final verification' if args.final else 'preflight') + ' passed')
