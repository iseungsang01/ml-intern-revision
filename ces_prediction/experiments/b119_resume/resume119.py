"""Registered one-time continuation of the latest validated B119 optimizer state."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
import traceback

# Apply before imports/guards: even nested git commands must remain invisible.
_popen = subprocess.Popen
_hidden = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
class HiddenPopen(_popen):
    def __init__(self, *args, **kwargs):
        kwargs['creationflags'] = kwargs.get('creationflags', 0) | _hidden
        super().__init__(*args, **kwargs)
subprocess.Popen = HiddenPopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'b119_anchor_lstm'))
import common119 as C

W = C.OUT / 'workflow/resume1'
ORIGINAL = C.OUT / 'workflow/started.json'
PAUSE = C.OUT / 'workflow/author_stop_20261128.json'
ARCHIVE = W / 'archive/fit_before_resume'
ADDENDUM = HERE.parent / 'PREREGISTRATION_B119_ADDENDUM1.md'
PY = Path(sys.executable).with_name('python.exe')


def new_sources():
    result = {}
    for path in [*sorted(HERE.glob('*.py')), ADDENDUM]:
        rel = path.relative_to(ROOT).as_posix()
        data = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT, capture_output=True, check=True).stdout
        assert hashlib.sha256(data).hexdigest() == C.sha(path), 'Uncommitted resume source: ' + rel
        result[rel] = C.sha(path)
    return result


def guard():
    original = C.read(ORIGINAL)['sources']
    assert len(original) == 33
    for rel, expected in original.items():
        assert C.sha(ROOT / rel) == expected, 'Frozen scientific source changed: ' + rel
    adapter = new_sources()
    if (W / 'started.json').exists():
        launch = C.read(W / 'started.json')
        assert launch['resume_sources'] == adapter
        assert launch['original_started_sha256'] == C.sha(ORIGINAL)
    return original


C.source_guard = guard


def hashes(folder):
    return {p.relative_to(folder).as_posix(): C.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def unchanged(folder, expected):
    assert folder.is_dir() and hashes(folder) == expected, 'Preserved artifact drift: ' + str(folder)


def smoke_guard():
    smoke, independent = (C.read(C.OUT / p) for p in ('smoke.json', 'smoke_independent.json'))
    assert smoke['all_pass'] and independent['all_pass']
    assert smoke['sources'] == independent['sources'] == guard()
    assert smoke['training_updates'] == independent['training_updates'] == 0
    assert independent['smoke_sha256'] == C.sha(C.OUT / 'smoke.json')
    for rel, expected in smoke['files'].items():
        assert C.sha(C.OUT / rel) == expected
    return {'smoke_sha256': C.sha(C.OUT / 'smoke.json'),
            'independent_sha256': C.sha(C.OUT / 'smoke_independent.json')}


def precheck():
    import torch
    sources, inputs = guard(), C.verify_inputs()
    assert C.read(ORIGINAL)['inputs'] == inputs
    pause = C.read(PAUSE)
    assert pause['state'] == 'paused_by_author' and pause['completed_epochs'] == 14
    assert pause['owned_processes_stopped'] and pause['latest_checkpoint_preserved']
    assert not any((C.OUT / p).exists() for p in ('fit_complete.json', 'val_audit.json', 'predict_started.json', 'test', 'audit.json'))
    checkpoint = torch.load(C.FIT / 'latest.pt', map_location='cpu', weights_only=False)
    assert checkpoint['validated'] and checkpoint['epoch'] == 14 and checkpoint['next_epoch'] == 15
    assert checkpoint['sources'] == sources and checkpoint['inputs'] == inputs and checkpoint['env'] == C.ENV_PINS
    history = checkpoint['history']
    assert len(history) == 14 and C.read(C.FIT / 'history.json') == history
    assert checkpoint['global_updates'] == sum(e['updates'] for e in history)
    assert checkpoint['history_sha256'] == C.sha(C.FIT / 'history.json')
    required = ('model', 'optimizer', 'scheduler', 'rng_cpu', 'rng_cuda', 'rng_numpy', 'rng_python',
                'best_model_state', 'significant_best', 'last_significant_epoch', 'low_lr_streak')
    assert all(k in checkpoint for k in required)
    start = C.read(C.FIT / 'started.json')
    assert start['sources'] == sources and start['inputs'] == inputs
    prior_active = float(pause['updated']) - datetime.fromisoformat(start['utc']).timestamp()
    assert prior_active + 2 >= float(history[-1]['elapsed_seconds'])
    assert 0 < prior_active < 21600
    snapshot = {}
    for path in (PAUSE, C.OUT / 'progress.json', C.OUT / 'events.jsonl'):
        assert path.is_file()
        snapshot[path.relative_to(C.OUT).as_posix()] = C.sha(path)
    return {'all_pass': True, 'sources': sources, 'inputs': inputs, 'resume_sources': new_sources(),
        'fit_files': hashes(C.FIT), 'live_snapshot_files': snapshot, 'completed_epochs': 14,
        'checkpoint_sha256': C.sha(C.FIT / 'latest.pt'), 'next_epoch': 15,
        'prior_active_seconds': prior_active, 'validated_elapsed_seconds': history[-1]['elapsed_seconds'],
        'remaining_active_seconds': 21600 - prior_active, 'smoke': smoke_guard(),
        'discarded_uncheckpointed_updates': 'unknown; volatile epoch excluded from losses and restored optimizer',
        'original_workflow_sha256': C.sha(ORIGINAL), 'pause_sha256': C.sha(PAUSE)}


def preserve(pre):
    unchanged(C.FIT, pre['fit_files'])
    assert C.FIT.resolve().is_relative_to(C.OUT.resolve()) and ARCHIVE.resolve().is_relative_to(W.resolve())
    assert not ARCHIVE.exists()
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(C.FIT, ARCHIVE)
    unchanged(ARCHIVE, pre['fit_files'])
    snapshots = W / 'archive/live_snapshots'
    snapshots.mkdir()
    copied = {}
    for rel, expected in pre['live_snapshot_files'].items():
        source = C.OUT / rel
        assert C.sha(source) == expected
        target = snapshots / rel.replace('/', '__')
        shutil.copy2(source, target)
        assert C.sha(target) == expected
        copied[target.relative_to(C.OUT).as_posix()] = expected
    C.save(W / 'archive.json', {'fit_archive': ARCHIVE.relative_to(C.OUT).as_posix(),
        'fit_files': pre['fit_files'], 'snapshot_files': copied})


def independent(final=False):
    saved = sys.argv
    try:
        sys.argv = [str(HERE / 'review119.py')] + (['--final'] if final else [])
        runpy.run_path(str(HERE / 'review119.py'), run_name='__main__')
    finally:
        sys.argv = saved


def clock_view():
    original, resume = C.read(ORIGINAL), C.read(W / 'started.json')
    return {**original, 'started': resume['started'], 'GPU_end': resume['GPU_end'],
            'CPU_end': resume['CPU_end'], 'pid': resume['pid']}


def provenance():
    pre = C.read(W / 'precheck.json')
    unchanged(ARCHIVE, pre['fit_files'])
    archived = C.read(W / 'archive.json')
    for rel, expected in archived['snapshot_files'].items():
        assert C.sha(C.OUT / rel) == expected
    return {'sources': new_sources(), 'original_scientific_sources': guard(),
        'original_workflow_sha256': C.sha(ORIGINAL), 'original_workflow': C.read(ORIGINAL),
        'resume_workflow': C.read(W / 'started.json'), 'precheck': pre,
        'precheck_sha256': C.sha(W / 'precheck.json'), 'archive': archived,
        'restore_evidence': C.read(W / 'restore_evidence.json'),
        'clock_view': clock_view(), 'clock_view_scope': 'Only audit119.read(original workflow/started.json); raw receipt immutable',
        'continuation_kind': 'same fit, validated optimizer/model/RNG restored; no new initialization fit'}


def run_stage(name):
    guard()
    launch = C.read(W / 'started.json')
    gpu = name in ('precheck', 'resume_fit', 'val_audit', 'predict')
    assert time.time() < launch['GPU_end' if gpu else 'CPU_end']
    with (W / ('stage_' + name + '_started.json')).open('x', encoding='utf8') as stream:
        json.dump({'started': time.time(), 'sources': new_sources()}, stream)
    if name == 'precheck':
        C.save(W / 'precheck.json', precheck())
        independent()
    elif name == 'resume_fit':
        resume_fit()
    elif name == 'independent_final':
        independent(final=True)
        final = C.read(C.OUT / 'resume_independent_final.json')
        assert final['all_pass']
        audit = C.read(C.OUT / 'audit.json')
        assert audit['all_pass'] and 'resume_provenance' not in audit
        audit['resume_provenance'] = {**provenance(), 'independent_final': final,
            'independent_final_sha256': C.sha(C.OUT / 'resume_independent_final.json')}
        C.save(C.OUT / 'audit.json', audit)
    elif name in ('val_audit', 'audit'):
        import audit119 as A
        raw_read = A.read
        view = clock_view()
        C.save(W / 'clock_view.json', view)
        def adapted_read(path):
            return dict(view) if Path(path).resolve() == ORIGINAL.resolve() else raw_read(path)
        A.read = adapted_read
        (A.val_gate if name == 'val_audit' else A.main)()
    elif name == 'predict':
        import predict119
        predict119.main()
    elif name == 'readout':
        import readout119
        readout119.main()
    elif name == 'closeout':
        audit = C.read(C.OUT / 'audit.json')
        binding = audit['resume_provenance']
        assert binding['sources'] == new_sources() and binding['independent_final']['all_pass']
        assert binding['independent_final_sha256'] == C.sha(C.OUT / 'resume_independent_final.json')
        assert provenance() == {k: v for k, v in binding.items() if k not in ('independent_final', 'independent_final_sha256')}
        import close119
        close119.main()
    else:
        raise ValueError(name)


def status(state, **extra):
    result = {'state': state, 'resume': 1, 'updated': time.time(), **extra}
    C.save(W / 'status.json', result)
    C.save(C.OUT / 'workflow/status.json', result)
    if extra.get('phase') != 'closeout':
        (ROOT / 'docs/paper/b119_status.md').write_text('# B119 validated-state continuation\n\n'
            + state + '\n\n' + json.dumps(extra, indent=2)
            + '\n\nLive: data/.b119/workflow/resume1/status.json. Original active fit budget retained.\n', encoding='utf8')


def queue():
    W.mkdir(parents=True, exist_ok=True)
    sources, inputs = guard(), C.verify_inputs()
    now = time.time()
    with (W / 'started.json').open('x', encoding='utf8') as stream:
        json.dump({'pid': os.getpid(), 'started': now, 'GPU_end': now + 8 * 3600,
            'CPU_end': now + 10 * 3600, 'sources': sources, 'inputs': inputs,
            'production_fits': 1, 'resume_sources': new_sources(),
            'original_started_sha256': C.sha(ORIGINAL)}, stream)
    import queue116 as Q
    Q.W, Q.status = W, status
    launch = C.read(W / 'started.json')
    env = {k: v for k, v in os.environ.items() if not k.startswith('CES_')}
    env.update(C.ENV_PINS)
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    def stage(name, gpu=True):
        Q.stage(name, [HERE / 'resume119.py', '--stage', name], env, launch['GPU_end' if gpu else 'CPU_end'])
    stage('precheck')
    Q.wait_idle(launch['GPU_end'])
    stage('resume_fit')
    stage('val_audit')
    fit, gate = (C.read(C.OUT / p) for p in ('fit_complete.json', 'val_audit.json'))
    assert gate['all_pass'] and gate['fit_complete_sha256'] == C.sha(C.OUT / 'fit_complete.json')
    if not fit['converged']:
        assert not (C.OUT / 'predict_started.json').exists() and not (C.OUT / 'test').exists()
        status('safety_capped_not_converged', phase='stopped_after_VAL', TEST_accessed=False)
        return
    stage('predict')
    for name in ('readout', 'audit', 'independent_final', 'closeout'):
        stage(name, gpu=False)
    C.save(W / 'status.json', {'state': 'complete', 'resume': 1, 'updated': time.time(),
        'commit': C.read(C.OUT / 'workflow/closeout.json')['commit']})
    C.save(C.OUT / 'workflow/status.json', C.read(W / 'status.json'))


def resume_fit():
    import random
    import numpy as np
    import torch
    from torch import nn
    import train119 as S
    import train116 as T
    import data116 as D
    import models119 as M
    event, atomic_npz, atomic_checkpoint = S.event, S.atomic_npz, S.atomic_checkpoint
    MIN_EPOCHS, MAX_EPOCHS, FIT_SECONDS = S.MIN_EPOCHS, S.MAX_EPOCHS, S.FIT_SECONDS
    sources, inputs = guard(), C.verify_inputs()
    pre = C.read(W / 'precheck.json')
    assert pre['all_pass'] and pre['sources'] == sources and pre['inputs'] == inputs
    smoke_guard()
    preserve(pre)
    checkpoint = torch.load(ARCHIVE / 'latest.pt', map_location='cpu', weights_only=False)
    assert C.sha(ARCHIVE / 'latest.pt') == pre['checkpoint_sha256']
    device = S.configure()
    prepared = D.load_prepared(C.UNIT)
    event('preparing_validation_for_state_restore')
    val_batches = T.validation_batches(prepared, C.UNIT)
    model = M.make_model(C.ARCH, C.SEED).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5, min_lr=1e-5)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # AdamW's scalar step stays on its original CPU device; moment tensors follow parameters.
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value) and key != 'step':
                state[key] = value.to(parameter.device)
    scheduler.load_state_dict(checkpoint['scheduler'])
    history = list(checkpoint['history'])
    best, best_epoch, best_state = checkpoint['best_val_mse'], checkpoint['best_epoch'], checkpoint['best_model_state']
    significant_best = checkpoint['significant_best']
    last_significant, low_lr_streak = checkpoint['last_significant_epoch'], checkpoint['low_lr_streak']
    total_updates, partial_updates = checkpoint['global_updates'], 0
    assert checkpoint['next_epoch'] == len(history) + 1 == 15
    assert C.read(C.FIT / 'history.json') == history
    current_best = C.npz(C.FIT / 'best.npz')
    assert set(current_best) == set(best_state)
    assert all(np.array_equal(current_best[k], best_state[k]) for k in best_state)
    mu, sd = (float(prepared['identity']['stats'][k]) for k in ('ti_mu', 'ti_sd'))
    # Restore all RNG states after every initializer/loader/VAL preparation and before new batches.
    torch.set_rng_state(checkpoint['rng_cpu'])
    torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
    np.random.set_state(checkpoint['rng_numpy'])
    random.setstate(checkpoint['rng_python'])
    actual_restored = {**checkpoint, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'rng_cpu': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all(), 'rng_numpy': np.random.get_state(),
        'rng_python': random.getstate()}
    restore_path = W / 'restore_checkpoint.pt'
    assert not restore_path.exists()
    torch.save(actual_restored, restore_path)
    C.save(W / 'restore_evidence.json', {'all_pass': True, 'source_latest_sha256': pre['checkpoint_sha256'],
        'restored_checkpoint_sha256': C.sha(restore_path), 'epoch': checkpoint['epoch'],
        'next_epoch': checkpoint['next_epoch'], 'global_updates': total_updates,
        'prior_active_seconds': pre['prior_active_seconds'],
        'remaining_active_seconds': pre['remaining_active_seconds'],
        'rng_restored_after_initialization': True, 'optimizer_step_before_evidence': False})
    prior_active = float(pre['prior_active_seconds'])
    start = time.monotonic() - prior_active
    deadline = start + FIT_SECONDS
    converged, reason = False, 'max_epochs_safety'
    C.save(W / 'fit_resume_started.json', {'sources': sources, 'resume_sources': new_sources(),
        'checkpoint_sha256': pre['checkpoint_sha256'], 'next_epoch': checkpoint['next_epoch'],
        'prior_active_seconds': prior_active, 'remaining_active_seconds': pre['remaining_active_seconds']})
    for epoch in range(checkpoint['next_epoch'], MAX_EPOCHS + 1):
        if time.monotonic() >= deadline:
            reason = 'wall_safety'
            break
        event('train_start', epoch=epoch)
        segments, assignment_sha = T.epoch_segments(prepared, C.UNIT, C.SEED, epoch)
        model.train()
        updates, sse, count = 0, 0.0, 0
        beat = time.monotonic()
        interrupted = False
        for batch_index, batch in enumerate(D.batches(segments, 'cpu', batch_size=16,
                                                     shuffle_seed=C.SEED * 1000 + epoch)):
            if time.monotonic() >= deadline:
                interrupted = True
                break
            x, y, mask = T.to_device(batch, device)
            if not bool(mask.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, batch['lengths'])
            error = prediction[mask] - y[mask]
            loss = error.square().mean()
            assert torch.isfinite(loss).item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total_updates += 1
            sse += float(error.detach().double().square().sum().item())
            count += int(mask.sum().item())
            if time.monotonic() - beat >= 10:
                event('training', epoch=epoch, batch=batch_index, updates=updates,
                      normalized_loss=float(loss.item()), running_normalized_mse=sse / count)
                beat = time.monotonic()
        if interrupted:
            partial_updates, reason = updates, 'wall_safety_incomplete_epoch'
            break
        metrics = {}
        for rule in C.TRAIN_RULES:
            event('validation', epoch=epoch, rule=rule)
            values = T.validate(model, {rule: val_batches[rule]}, mu, sd, device, deadline)
            if values is None:
                interrupted = True
                break
            metrics.update(values)
        if interrupted:
            partial_updates, reason = updates, 'wall_safety_incomplete_validation'
            break
        score = float(np.mean([metrics[r]['mse'] for r in C.TRAIN_RULES]))
        lr_used = float(optimizer.param_groups[0]['lr'])
        scheduler.step(score)
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
            best_state = T.state_arrays(model)
            atomic_npz(C.FIT / 'best.npz', best_state)
        if score < significant_best * .999:
            significant_best, last_significant = score, epoch
        low_lr_streak = low_lr_streak + 1 if lr_used <= 1e-5 else 0
        elapsed_once = time.monotonic() - start
        wall_hit = elapsed_once >= FIT_SECONDS
        entry = {'epoch': epoch, 'train_normalized_sse': sse, 'train_count': count,
            'train_normalized_mse': sse / count, 'train_physical_mse': sse / count * sd * sd,
            'updates': updates, 'assignment_sha256': assignment_sha, 'val': metrics,
            'seven_rule_mean_mse': score, 'lr_used': lr_used,
            'lr_next': float(optimizer.param_groups[0]['lr']),
            'elapsed_seconds': elapsed_once, 'best_epoch': best_epoch,
            'significant_best': significant_best, 'last_significant_epoch': last_significant,
            'low_lr_streak': low_lr_streak, 'epochs_since_significant': epoch - last_significant}
        history.append(entry)
        converged = (epoch >= MIN_EPOCHS and low_lr_streak >= 10
                     and epoch - last_significant >= 20 and not wall_hit)
        C.save(C.FIT / 'history.json', history)
        atomic_checkpoint({'epoch': epoch, 'next_epoch': epoch + 1, 'validated': True, 'global_updates': total_updates,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
            'rng_numpy': np.random.get_state(), 'rng_python': random.getstate(),
            'best_epoch': best_epoch, 'best_val_mse': best,
            'significant_best': significant_best, 'last_significant_epoch': last_significant,
            'low_lr_streak': low_lr_streak, 'sources': sources, 'inputs': inputs,
            'history_sha256': C.sha(C.FIT / 'history.json'), 'env': C.ENV_PINS,
            # This atomic file is the recoverable epoch boundary even if a pause
            # interrupts later sidecar updates during another epoch.
            'history': list(history), 'best_model_state': best_state})
        C.save(C.FIT / 'recovery.json', {'epoch': epoch, 'validated': True,
            'latest_sha256': C.sha(C.FIT / 'latest.pt'), 'best_sha256': C.sha(C.FIT / 'best.npz'),
            'history_sha256': C.sha(C.FIT / 'history.json'), 'sources': sources})
        event('epoch_complete', **entry)
        if converged:
            reason = 'registered_statistical_stop'
            break
        if wall_hit:
            reason = 'wall_safety_after_complete_validation'
            break
    status = 'CONVERGED' if converged else 'SAFETY_CAPPED_NOT_CONVERGED'
    params = {'status': status, 'converged': converged, 'normal': converged, 'stop_reason': reason,
        'unit': C.UNIT, 'architecture': C.ARCH, 'seed': C.SEED, 'sources': sources, 'inputs': inputs,
        'completed_epochs': len(history), 'best_epoch': best_epoch, 'best_val_mse': best if history else None,
        'total_updates': total_updates, 'discarded_partial_updates': partial_updates,
        'actual_seconds': time.monotonic() - start, 'budget_seconds': FIT_SECONDS,
        'min_epochs': MIN_EPOCHS, 'max_epochs': MAX_EPOCHS,
        'env': C.ENV_PINS, 'runtime': {'python': sys.version, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(device)},
        'files': {p.name: C.sha(p) for p in C.FIT.iterdir() if p.is_file()}}
    C.save(C.FIT / 'params.json', params)
    files = {p.relative_to(C.OUT).as_posix(): C.sha(p) for p in C.FIT.iterdir() if p.is_file()}
    if history:
        event('export_best_validation', best_epoch=best_epoch)
        weights = C.npz(C.FIT / 'best.npz')
        model.load_state_dict({k.removeprefix('model.'): torch.as_tensor(v, device=device)
                               for k, v in weights.items()})
        model.eval()
        for rule in C.TRAIN_RULES:
            event('export_best_validation', rule=rule, best_epoch=best_epoch)
            support = D.common_row_support(C.UNIT, rule, 'val')
            prediction = T.predict_rows(model, prepared['val'][rule], support,
                                        prepared['identity']['stats'], device)
            path = C.OUT / 'val' / (C.UNIT + '_' + rule + '.npz')
            assert not path.exists()
            atomic_npz(path, {**support, 'prediction': prediction})
            files[path.relative_to(C.OUT).as_posix()] = C.sha(path)
    C.save(C.OUT / 'fit_complete.json', {'all_pass': True, 'converged': converged, 'status': status,
        'sources': sources, 'params_sha256': C.sha(C.FIT / 'params.json'),
        'production_fits': 1, 'TEST_accessed': False, 'files': files})
    assert C.source_guard() == sources
    event('fit_finished', status=status, completed_epochs=len(history))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('precheck', 'resume_fit', 'val_audit', 'predict', 'readout', 'audit', 'independent_final', 'closeout'))
    args = parser.parse_args()
    try:
        run_stage(args.stage) if args.stage else queue()
    except BaseException as error:
        if not args.stage:
            current = C.read(W / 'status.json') if (W / 'status.json').exists() else {}
            status('failed', phase=current.get('phase'), error=repr(error), traceback=traceback.format_exc(), no_retry=True)
        raise
