"""Registered one-time B116 continuation; original scientific sources stay frozen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'b116_temporal'))
import common116 as C

W = C.OUT / 'workflow/resume1'
ORIGINAL = C.OUT / 'workflow/started.json'
ADDENDUM = HERE.parent / 'PREREGISTRATION_B116_ADDENDUM1.md'
PARTIAL = 'fits/V4a/lstm/s1701'
ARCHS = ('lstm', 'decay_gru', 'innovation_gru')
PY = Path(sys.executable).with_name('python.exe')
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def new_sources():
    paths = [*sorted(HERE.glob('*.py')), ADDENDUM]
    result = {}
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        data = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT,
                              capture_output=True, check=True).stdout
        assert hashlib.sha256(data).hexdigest() == C.sha(path), 'Uncommitted adapter: ' + rel
        result[rel] = C.sha(path)
    return result


def guard():
    original = C.read(ORIGINAL)['sources']
    assert original and not any('ADDENDUM1' in k for k in original)
    for rel, expected in original.items():
        assert C.sha(ROOT / rel) == expected, 'Frozen scientific source drift: ' + rel
    adapter = new_sources()
    if (W / 'started.json').exists():
        started = C.read(W / 'started.json')
        assert started['resume_sources'] == adapter
        assert started['original_started_sha256'] == C.sha(ORIGINAL)
    return original


# All frozen importers share this module object. Preserve scientific receipt identity.
C.source_guard = guard


def file_hashes(folder):
    assert folder.is_dir()
    return {p.relative_to(folder).as_posix(): C.sha(p)
            for p in sorted(folder.rglob('*')) if p.is_file()}


def check_files(folder, expected):
    assert file_hashes(folder) == expected, 'Artifact changed: ' + str(folder)


def smoke_guard():
    smoke = C.read(C.OUT / 'smoke.json')
    independent = C.read(C.OUT / 'smoke_independent.json')
    assert smoke['all_pass'] and independent['all_pass']
    assert smoke['sources'] == independent['sources'] == guard()
    assert independent['smoke_sha256'] == C.sha(C.OUT / 'smoke.json')
    for arch in ARCHS:
        for rel, expected in smoke['architectures'][arch]['files'].items():
            assert C.sha(C.OUT / rel) == expected
    return {'smoke_sha256': C.sha(C.OUT / 'smoke.json'),
            'independent_sha256': C.sha(C.OUT / 'smoke_independent.json')}


def precheck():
    import data116 as D
    import audit116 as A
    sources = guard()
    C.verify_inputs()
    assert not any((C.OUT / p).exists() for p in
                   ('fit_complete.json', 'selection.json', 'predict_started.json', 'audit.json'))
    pause = C.read(C.OUT / 'workflow/pause_author_20261126.json')
    assert pause['state'] == 'paused_by_author' and pause['completed_fits'] == 27
    assert pause['partial_fits'] == [PARTIAL] and pause['owned_processes_stopped']
    complete, missing = [], []
    expected_dirs = set()
    for unit in C.UNITS:
        prepared = D.load_prepared(unit)
        for arch in ARCHS:
            for seed in C.SEEDS:
                rel = f'fits/{unit}/{arch}/s{seed}'
                expected_dirs.add(rel)
                folder = C.OUT / rel
                if (folder / 'params.json').exists():
                    assert unit != 'V4a', 'Unexpected completed continuation fit'
                    params = A.verify_history(unit, arch, seed, prepared)
                    assert params['sources'] == C.read(folder / 'started.json')['sources'] == sources
                    complete.append({'unit': unit, 'architecture': arch, 'seed': seed,
                                     'folder': rel, 'files': file_hashes(folder)})
                else:
                    assert unit == 'V4a'
                    if rel == PARTIAL:
                        assert folder.is_dir() and (folder / 'history.json').is_file()
                        assert not (folder / 'best.npz').exists()
                        assert set(file_hashes(folder)) == {'started.json', 'history.json'}
                    else:
                        assert not folder.exists(), 'Unregistered incomplete fit: ' + rel
                    missing.append({'unit': unit, 'architecture': arch, 'seed': seed, 'folder': rel})
    actual = {p.relative_to(C.OUT).as_posix() for p in (C.OUT / 'fits').glob('*/*/s*') if p.is_dir()}
    assert actual <= expected_dirs and len(complete) == 27 and len(missing) == 9
    return {'all_pass': True, 'sources': sources, 'resume_sources': new_sources(),
            'completed': complete, 'missing': missing, 'smoke': smoke_guard(),
            'partial': {'folder': PARTIAL, 'files': file_hashes(C.OUT / PARTIAL)},
            'pause_sha256': C.sha(C.OUT / 'workflow/pause_author_20261126.json')}


def preserve_originals(pre):
    for item in pre['completed']:
        check_files(C.OUT / item['folder'], item['files'])


def fit():
    import data116 as D
    import train116 as T
    pre = C.read(W / 'precheck.json')
    assert pre['all_pass'] and pre['sources'] == guard()
    preserve_originals(pre)
    smoke_guard()
    assert len(pre['missing']) == 9
    for item in pre['missing']:
        path = C.OUT / item['folder']
        assert item['unit'] == 'V4a'
        assert path.is_dir() if item['folder'] == PARTIAL else not path.exists()
    source = (C.OUT / PARTIAL).resolve()
    archive = (W / 'archive/V4a_lstm_s1701_interrupted').resolve()
    assert source.is_relative_to(C.OUT.resolve()) and archive.is_relative_to(W.resolve())
    check_files(source, pre['partial']['files'])
    assert not archive.exists()
    archive.parent.mkdir(parents=True, exist_ok=True)
    source.rename(archive)
    check_files(archive, pre['partial']['files'])
    C.save(W / 'partial_archive.json', {'source': PARTIAL,
        'archive': archive.relative_to(C.OUT).as_posix(), 'files': file_hashes(archive)})
    device = T.configure()
    completed = {item['folder']: item for item in pre['completed']}
    allowed = {item['folder'] for item in pre['missing']}
    receipts, new = [], []
    for unit in C.UNITS:
        prepared = D.load_prepared(unit)
        val_batches = None
        for arch in ARCHS:
            for seed in C.SEEDS:
                folder = T.fit_dir(unit, arch, seed)
                rel = folder.relative_to(C.OUT).as_posix()
                if rel in completed:
                    check_files(folder, completed[rel]['files'])
                    # Independently verified in the separate CPU precheck process;
                    # exact-byte preservation binds this receipt without importing
                    # audit116 (which intentionally disables CUDA) into this process.
                    params = C.read(folder / 'params.json')
                    assert params['normal'] and params['sources'] == guard()
                else:
                    assert rel in allowed and not folder.exists()
                    if val_batches is None:
                        val_batches = T.validation_batches(prepared, unit)
                    params = T.fit_one(unit, arch, seed, prepared, val_batches, device, guard())
                    new.append({'folder': rel, 'files': file_hashes(folder)})
                    C.save(W / 'new_fits.json', new)
                path = folder / 'params.json'
                receipts.append({'unit': unit, 'architecture': arch, 'seed': seed,
                    'path': path.relative_to(C.OUT).as_posix(), 'sha256': C.sha(path), 'normal': params['normal']})
        del val_batches, prepared
    assert len(new) == 9 and len(receipts) == 36
    preserve_originals(pre)
    files = T.export_split('val', device)
    assert not (C.OUT / 'fit_complete.json').exists()
    C.save(C.OUT / 'fit_complete.json', {'all_pass': True, 'fits': receipts, 'files': files,
        'sources': guard(), 'test_accessed': False, 'utc': T.utc()})


def provenance():
    pre = C.read(W / 'precheck.json')
    preserve_originals(pre)
    archived = C.read(W / 'partial_archive.json')
    check_files(C.OUT / archived['archive'], archived['files'])
    new = C.read(W / 'new_fits.json')
    assert len(new) == 9
    for item in new:
        check_files(C.OUT / item['folder'], item['files'])
    return {'sources': new_sources(), 'started': C.read(W / 'started.json'),
        'precheck_sha256': C.sha(W / 'precheck.json'), 'precheck': pre,
        'partial_archive': archived, 'new_fits': new,
        'original_scientific_sources': guard()}


def run_stage(name):
    guard()
    if name != 'precheck':
        assert (W / 'precheck.json').is_file()
    start = C.read(W / 'started.json')
    deadline = start['GPU_end'] if name in ('precheck', 'fit', 'select', 'predict') else start['CPU_end']
    assert time.time() < deadline
    with (W / ('stage_' + name + '_started.json')).open('x', encoding='utf8') as stream:
        json.dump({'started': time.time(), 'sources': new_sources()}, stream)
    if name == 'precheck':
        C.save(W / 'precheck.json', precheck())
    elif name == 'fit':
        fit()
    elif name == 'predict':
        import train116 as T
        T.predict(T.configure())
    else:
        if name == 'closeout':
            audit = C.read(C.OUT / 'audit.json')
            resume = audit['resume_provenance']
            assert resume['sources'] == new_sources()
            assert resume['precheck_sha256'] == C.sha(W / 'precheck.json')
            assert resume['independent_final_sha256'] == C.sha(C.OUT / 'resume_independent_final.json')
            assert resume['independent_final']['all_pass']
            assert provenance() == {k: v for k, v in resume.items()
                                    if k not in ('independent_final', 'independent_final_sha256')}
        module = __import__({'select': 'select116', 'readout': 'readout116',
                             'audit': 'audit116', 'closeout': 'close116'}[name])
        module.main()
        if name == 'audit':
            subprocess.run([str(PY), str(HERE / 'review116.py'), '--final'],
                           cwd=ROOT, check=True, creationflags=FLAGS)
            audit = C.read(C.OUT / 'audit.json')
            assert audit['all_pass'] and 'resume_provenance' not in audit
            audit['resume_provenance'] = provenance()
            audit['resume_provenance']['independent_final'] = C.read(C.OUT / 'resume_independent_final.json')
            assert audit['resume_provenance']['independent_final']['all_pass']
            audit['resume_provenance']['independent_final_sha256'] = C.sha(C.OUT / 'resume_independent_final.json')
            C.save(C.OUT / 'audit.json', audit)


def status(state, **extra):
    result = {'state': state, 'updated': time.time(), 'resume': 1, **extra}
    C.save(W / 'status.json', result)
    C.save(C.OUT / 'workflow/status.json', result)
    if extra.get('phase') != 'closeout':
        (ROOT / 'docs/paper/b116_status.md').write_text(
            '# B116 registered continuation\n\n' + state + '\n\n' + json.dumps(extra, indent=2)
            + '\n\nLive: data/.b116/workflow/resume1/status.json. Original completed fits preserved.\n', encoding='utf8')


def queue():
    W.mkdir(parents=True, exist_ok=True)
    sources = guard()
    now = time.time()
    with (W / 'started.json').open('x', encoding='utf8') as stream:
        json.dump({'pid': os.getpid(), 'started': now, 'GPU_end': now + 6 * 3600,
            'CPU_end': now + 8 * 3600, 'sources': sources, 'resume_sources': new_sources(),
            'original_started_sha256': C.sha(ORIGINAL)}, stream)
    import queue116 as Q
    Q.W = W
    Q.status = status
    start = C.read(W / 'started.json')
    env = {k: v for k, v in os.environ.items() if not k.startswith('CES_')}
    env.update(C.ENV_PINS)
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    Q.stage('precheck', [HERE / 'resume116.py', '--stage', 'precheck'], env, start['GPU_end'])
    Q.wait_idle(start['GPU_end'])
    for name in ('fit', 'select', 'predict', 'readout', 'audit', 'closeout'):
        deadline = start['GPU_end'] if name in ('fit', 'select', 'predict') else start['CPU_end']
        Q.stage(name, [HERE / 'resume116.py', '--stage', name], env, deadline)
    C.save(W / 'status.json', {'state': 'complete', 'updated': time.time(), 'resume': 1,
        'commit': C.read(C.OUT / 'workflow/closeout.json')['commit']})
    C.save(C.OUT / 'workflow/status.json', C.read(W / 'status.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('precheck', 'fit', 'select', 'predict', 'readout', 'audit', 'closeout'))
    args = parser.parse_args()
    try:
        run_stage(args.stage) if args.stage else queue()
    except BaseException as error:
        if not args.stage:
            current = C.read(W / 'status.json') if (W / 'status.json').exists() else {}
            status('failed', phase=current.get('phase'), error=repr(error),
                   traceback=traceback.format_exc(), no_retry=True)
        raise
