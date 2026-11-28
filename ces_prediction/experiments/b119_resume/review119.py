"""Independent read-only checkpoint14 and resumed-continuity verification."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'b119_anchor_lstm'))
import common119 as C
import audit119 as A
D = A.D
W = C.OUT / 'workflow/resume1'
PRECHECK = C.OUT / 'workflow/resume_independent_precheck.json'
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def hashes(folder):
    return {p.relative_to(folder).as_posix(): C.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def original_guard():
    started = C.read(C.OUT / 'workflow/started.json')
    for rel, digest in started['sources'].items():
        assert C.sha(ROOT / rel) == digest, 'Original source drift: ' + rel
        blob = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT, capture_output=True,
                              check=True, creationflags=FLAGS).stdout
        assert hashlib.sha256(blob).hexdigest() == digest, 'Original HEAD drift: ' + rel
    return started


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def same_tree(a, b, path='state'):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()), path
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            same_tree(a[k], b[k], path + '/' + str(k))
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same_tree(x, y, path + '/' + str(i))
    else:
        assert a == b, path


def check14(checkpoint, history):
    assert checkpoint['epoch'] == len(history) == 14 and checkpoint['next_epoch'] == 15
    assert checkpoint['validated'] and checkpoint['history'] == history
    assert checkpoint['global_updates'] == sum(h['updates'] for h in history)
    assert checkpoint['best_epoch'] == int(np.argmin([h['seven_rule_mean_mse'] for h in history])) + 1
    assert checkpoint['best_val_mse'] == min(h['seven_rule_mean_mse'] for h in history)
    prepared = D.load_prepared(C.UNIT)
    A.A.verify_normalization(prepared)
    keys = sorted((s['fidx'], s['segment']) for s in prepared['train'][C.TRAIN_RULES[0]])
    import zlib
    lr, best, bad = .001, float('inf'), 0
    for epoch, h in enumerate(history, 1):
        assert h['epoch'] == epoch and h['lr_used'] == lr and set(h['val']) == set(C.TRAIN_RULES)
        for rule, value in h['val'].items():
            assert value['count'] == len(D.common_row_support(C.UNIT, rule, 'val')['idx'])
            assert np.isclose(value['mse'], value['sse'] / value['count'], rtol=1e-12)
        score = float(np.mean([h['val'][r]['mse'] for r in C.TRAIN_RULES]))
        assert np.isclose(score, h['seven_rule_mean_mse'], rtol=1e-14)
        if score < best * .9999:
            best, bad = score, 0
        else:
            bad += 1
        if bad > 3:
            candidate = max(lr * .5, 1e-5)
            if lr - candidate > 1e-8:
                lr = candidate
            bad = 0
        assert h['lr_next'] == lr
        rng = np.random.default_rng([C.SEED, epoch, zlib.crc32(C.UNIT.encode())])
        choices = rng.integers(0, 7, len(keys))
        assignment = [(f, s, C.TRAIN_RULES[int(i)]) for (f, s), i in zip(keys, choices)]
        assert h['assignment_sha256'] == A.F.digest(assignment)
    tracking = A.F.stop_tracking(history)
    assert not any(h['converged'] for h in tracking)
    assert checkpoint['significant_best'] == tracking[-1]['meaningful_reference_mse']
    assert checkpoint['last_significant_epoch'] == tracking[-1]['last_meaningful_epoch']
    assert checkpoint['low_lr_streak'] == tracking[-1]['low_lr_streak']
    virtual_params = {k: checkpoint[k] for k in ('sources', 'inputs', 'best_epoch', 'best_val_mse')}
    recovery = A.verify_recovery(history, virtual_params)
    return recovery


def preflight():
    started = original_guard()
    assert len(started['sources']) == 33
    assert time.time() > started['CPU_end']
    status = C.read(C.OUT / 'workflow/author_stop_20261128.json')
    assert status['state'] == 'paused_by_author' and status['completed_epochs'] == 14
    assert status['owned_processes_stopped'] and not status['automatic_resume'] and status['original_deadlines_preserved']
    assert not any((C.OUT / p).exists() for p in ('fit_complete.json', 'val_audit.json', 'predict_started.json', 'predictions_complete.json', 'audit.json'))
    assert not (C.OUT / 'test').exists() and not (C.OUT / 'val').exists()
    assert {p.name for p in C.FIT.iterdir()} == {'best.npz', 'history.json', 'latest.pt', 'recovery.json', 'started.json'}
    inputs = C.verify_inputs()
    checkpoint = load_checkpoint(C.FIT / 'latest.pt')
    history = C.read(C.FIT / 'history.json')
    assert checkpoint['sources'] == started['sources'] and checkpoint['inputs'] == started['inputs'] == inputs
    recovery = check14(checkpoint, history)
    from datetime import datetime
    fit_started = C.read(C.FIT / 'started.json')
    prior_active = status['updated'] - datetime.fromisoformat(fit_started['utc']).timestamp()
    assert prior_active >= history[-1]['elapsed_seconds'] and prior_active < 21600
    smoke = C.read(C.OUT / 'smoke_independent.json')
    assert smoke['all_pass'] and smoke['sources'] == started['sources'] and smoke['optimizer_steps'] == 0
    assert smoke['smoke_sha256'] == C.sha(C.OUT / 'smoke.json')
    return {'all_pass': True, 'created_unix': time.time(), 'scientific_fits': 0, 'inference': 0,
        'sources': started['sources'], 'inputs': inputs, 'fit_files': hashes(C.FIT),
        'original_started_sha256': C.sha(C.OUT / 'workflow/started.json'),
        'original_progress_sha256': C.sha(C.OUT / 'progress.json'),
        'original_events_sha256': C.sha(C.OUT / 'events.jsonl'),
        'pause_status': status, 'history': history, 'recovery': recovery,
        'completed_epochs': 14, 'next_epoch': 15, 'validated_updates': checkpoint['global_updates'],
        'old_active_seconds': prior_active, 'old_validated_elapsed_seconds': history[-1]['elapsed_seconds'],
        'remaining_fit_seconds': 21600 - prior_active,
        'author_stop_sha256': C.sha(C.OUT / 'workflow/author_stop_20261128.json'),
        'scope': 'Exact checkpoint14 model/optimizer/scheduler/RNG/history/best recovery, raw provenance and no-TEST barrier'}

def final_verification():
    original = original_guard()
    before = C.read(PRECHECK)
    launch = C.read(W / 'started.json')
    assert before['all_pass'] and before['completed_epochs'] == 14 and before['next_epoch'] == 15
    assert launch['sources'] == original['sources'] == before['sources']
    assert launch['inputs'] == original['inputs'] == before['inputs']
    assert launch['original_started_sha256'] == before['original_started_sha256'] == C.sha(C.OUT / 'workflow/started.json')
    assert launch['GPU_end'] == launch['started'] + 28800 and launch['CPU_end'] == launch['started'] + 36000
    assert launch['started'] > original['CPU_end'] and launch['production_fits'] == 1
    new_sources = {}
    for path in [*sorted(HERE.glob('*.py')), HERE.parent / 'PREREGISTRATION_B119_ADDENDUM1.md']:
        rel = path.relative_to(ROOT).as_posix()
        blob = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT, capture_output=True,
                              check=True, creationflags=FLAGS).stdout
        assert hashlib.sha256(blob).hexdigest() == C.sha(path), 'Uncommitted continuation source'
        new_sources[rel] = C.sha(path)
    assert launch['resume_sources'] == new_sources
    archive = W / 'archive/fit_before_resume'
    assert hashes(archive) == before['fit_files'], 'Original checkpoint14 archive changed'
    preserved = C.read(W / 'archive.json')
    assert preserved['fit_archive'] == archive.relative_to(C.OUT).as_posix()
    assert preserved['fit_files'] == before['fit_files']
    for rel, expected in preserved['snapshot_files'].items():
        assert C.sha(C.OUT / rel) == expected
    expected_live = {before['original_progress_sha256'], before['original_events_sha256'], before['author_stop_sha256']}
    assert set(preserved['snapshot_files'].values()) == expected_live
    original_cp = load_checkpoint(archive / 'latest.pt')
    restored = load_checkpoint(W / 'restore_checkpoint.pt')
    same_tree(original_cp, restored, 'actual_restored_before_first_update')
    assert original_cp['epoch'] == 14 and original_cp['next_epoch'] == 15
    restored_receipt = C.read(W / 'restore_evidence.json')
    assert restored_receipt['all_pass']
    assert restored_receipt['source_latest_sha256'] == C.sha(archive / 'latest.pt')
    assert restored_receipt['restored_checkpoint_sha256'] == C.sha(W / 'restore_checkpoint.pt')
    assert (restored_receipt['epoch'], restored_receipt['next_epoch'], restored_receipt['global_updates']) == (14, 15, before['validated_updates'])
    assert restored_receipt['rng_restored_after_initialization'] and not restored_receipt['optimizer_step_before_evidence']
    assert restored_receipt['prior_active_seconds'] == before['old_active_seconds']
    assert restored_receipt['remaining_active_seconds'] == before['remaining_fit_seconds']
    # Artifact bytes and entire actual model/optimizer/scheduler/RNG state, rather
    # than a producer assertion alone, establish the restored optimizer trajectory.
    assert C.sha(archive / 'latest.pt') == before['fit_files']['latest.pt']
    history = C.read(C.FIT / 'history.json')
    assert history[:14] == before['history'] == original_cp['history']
    assert len(history) >= 15 and [h['epoch'] for h in history] == list(range(1, len(history) + 1))
    assert len(history) <= 200
    from datetime import datetime
    pause = C.read(C.OUT / 'workflow/author_stop_20261128.json')
    assert C.sha(C.OUT / 'workflow/author_stop_20261128.json') == before['author_stop_sha256']
    initial_started = C.read(archive / 'started.json')
    prior_active = pause['updated'] - datetime.fromisoformat(initial_started['utc']).timestamp()
    assert prior_active == before['old_active_seconds'] and prior_active >= history[13]['elapsed_seconds']
    assert before['remaining_fit_seconds'] == 21600 - prior_active
    assert all(h['elapsed_seconds'] >= prior_active for h in history[14:])
    assert all(a['elapsed_seconds'] <= b['elapsed_seconds'] for a, b in zip(history, history[1:]))
    params = C.read(C.FIT / 'params.json')
    assert (params['min_epochs'], params['max_epochs'], params['budget_seconds']) == (50, 200, 21600)
    assert params['total_updates'] == sum(h['updates'] for h in history) + params['discarded_partial_updates']
    assert params['actual_seconds'] >= history[-1]['elapsed_seconds'] >= prior_active
    assert len(list((C.OUT / 'fits').iterdir())) == 1 and C.FIT.is_dir()
    current = load_checkpoint(C.FIT / 'latest.pt')
    assert current['history'] == history and current['global_updates'] == sum(h['updates'] for h in history)
    assert current['sources'] == original['sources'] and current['inputs'] == original['inputs']
    assert current['epoch'] == len(history) and current['next_epoch'] == len(history) + 1
    # Only these five receipt fields may be viewed through the registered clock adapter.
    expected_clock = {**original, 'started': launch['started'], 'GPU_end': launch['GPU_end'],
                      'CPU_end': launch['CPU_end'], 'pid': launch['pid']}
    assert C.read(W / 'clock_view.json') == expected_clock
    numerical = C.read(C.OUT / 'audit.json')
    assert numerical['all_pass'] and numerical['sources'] == original['sources']
    assert numerical['val_gate_reaudit']['converged'] and params['converged']
    assert numerical['val_gate_reaudit']['best_sha256'] == C.sha(C.FIT / 'best.npz')
    pred_start = C.read(C.OUT / 'predict_started.json')
    assert launch['started'] <= pred_start['started'] <= launch['GPU_end']
    assert pred_start['val_audit_sha256'] == C.sha(C.OUT / 'val_audit.json')
    for name in ('precheck', 'resume_fit', 'val_audit', 'predict', 'readout', 'audit'):
        stage = C.read(W / ('stage_' + name + '_started.json'))
        assert stage['sources'] == new_sources
        assert launch['started'] <= stage['started'] <= launch['CPU_end']
    assert C.verify_inputs() == before['inputs']
    return {'all_pass': True, 'created_unix': time.time(), 'scientific_fit_count': 1,
        'continuation': 'Same model/optimizer trajectory from validated epoch14; next epoch15; no new initialization fit',
        'sources': original['sources'], 'resume_sources': new_sources,
        'independent_precheck_sha256': C.sha(PRECHECK), 'archive_fit_files': before['fit_files'],
        'restore_checkpoint_sha256': C.sha(W / 'restore_checkpoint.pt'),
        'restore_evidence_sha256': C.sha(W / 'restore_evidence.json'),
        'restored_state_exact': True, 'history_prefix14_exact': True,
        'prior_active_seconds_including_discarded_partial': prior_active,
        'prior_validated_elapsed_seconds': history[13]['elapsed_seconds'],
        'remaining_fit_seconds_at_resume': 21600 - prior_active,
        'completed_epochs_total': len(history), 'current_validated_updates': current['global_updates'],
        'original_workflow_sha256': C.sha(C.OUT / 'workflow/started.json'),
        'resume_workflow_sha256': C.sha(W / 'started.json'), 'clock_view': expected_clock,
        'clock_view_sha256': C.sha(W / 'clock_view.json'),
        'numerical_audit_sha256_before_resume_attachment': C.sha(C.OUT / 'audit.json'),
        'old_partial_updates': 'Not part of restored optimizer trajectory; archived heartbeat is a lower bound only',
        'restore_evidence': restored_receipt}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    result = final_verification() if args.final else preflight()
    target = C.OUT / 'resume_independent_final.json' if args.final else PRECHECK
    C.save(target, result)
    print('B119 independent continuation ' + ('final audit' if args.final else 'preflight') + ' passed')
