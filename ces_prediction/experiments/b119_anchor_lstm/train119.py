"""One visible-CES anchored residual LSTM with the frozen stopping policy."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import common119 as C
import train116 as T
import data116 as D
import models119 as M

MIN_EPOCHS, MAX_EPOCHS, FIT_SECONDS = 50, 200, 6 * 3600


def event(stage, **values):
    z = {'unix': time.time(), 'stage': stage, **values}
    C.save(C.OUT / 'progress.json', z)
    with (C.OUT / 'events.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(z, allow_nan=False) + '\n')


def atomic_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def atomic_checkpoint(value):
    path = C.FIT / 'latest.pt'
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def configure():
    for key, value in C.ENV_PINS.items():
        assert os.environ.get(key) == value, 'Unpinned environment: ' + key
    return T.configure()


def smoke():
    sources = C.source_guard()
    inputs = C.verify_inputs()
    assert not (C.OUT / 'smoke').exists()
    (C.OUT / 'smoke').mkdir(parents=True)
    device = configure()
    prepared = D.load_prepared(C.UNIT)
    candidates = []
    for segment in prepared['train'][C.TRAIN_RULES[0]]:
        if np.asarray(segment['mask'])[:64, 0].any():
            candidates.append({k: v[:64].copy() if isinstance(v, np.ndarray) and v.ndim else v
                               for k, v in segment.items()})
        if len(candidates) == 2:
            break
    assert len(candidates) == 2
    batch = next(D.batches(candidates, 'cpu', batch_size=2))
    x = batch['x']
    model = M.make_model(C.ARCH, C.SEED).to(device).eval()
    initialized = T.state_arrays(model)
    original = C.npz(C.ROOT / 'data/.b117_focused/smoke/weights.npz')
    for key in initialized:
        if key in ('model.readout.3.weight', 'model.readout.3.bias'):
            assert np.count_nonzero(initialized[key]) == 0
        else:
            assert np.array_equal(initialized[key], original[key]), 'Seeded core differs: ' + key
    with torch.no_grad():
        prediction = model(x.to(device), batch['lengths']).cpu().numpy()
        expected = torch.where(x[..., 18] > .5, x[..., 16], torch.zeros_like(x[..., 16])).numpy()
        assert np.array_equal(prediction, expected)
        state, steps = None, []
        for t in range(x.shape[1]):
            value, state = model.forward_step(x[:, t].to(device), state)
            steps.append(value)
        step_error = float(np.max(np.abs(torch.stack(steps, 1).cpu().numpy() - prediction)))
        assert step_error <= 2e-5
        k = min(8, x.shape[1])
        shorter = model(x[:, :k].to(device)).cpu().numpy()
        assert np.array_equal(shorter, prediction[:, :k])
        changed = x.clone()
        changed[:, k:] = torch.flip(changed[:, k:], [1])
        altered = model(changed.to(device)).cpu().numpy()
        assert np.array_equal(altered[:, :k], prediction[:, :k])
    model.zero_grad(set_to_none=True)
    mask = batch['mask'].to(device)
    loss = (model(x.to(device))[mask] - batch['y'].to(device)[mask]).square().mean()
    assert torch.isfinite(loss).item()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    after_backward = T.state_arrays(model)
    assert all(np.array_equal(initialized[key], after_backward[key]) for key in initialized)
    model.zero_grad(set_to_none=True)
    atomic_npz(C.OUT / 'smoke/weights.npz', T.state_arrays(model))
    atomic_npz(C.OUT / 'smoke/prefix.npz', {'x': x.numpy(), 'pred': prediction, 'expected_anchor': expected,
        'gi': batch['gi'].numpy(), 'fidx': np.asarray(batch['fidx']),
        'segment': np.asarray(batch['segment']), 'mask': batch['mask'].numpy(),
        'lengths': batch['lengths'].numpy()})
    C.save(C.OUT / 'smoke.json', {'all_pass': True, 'sources': sources, 'inputs': inputs,
        'training_updates': 0, 'optimizer_steps': 0, 'backward_finite': True,
        'initial_output_exact_anchor': True, 'nonhead_seeded_core_identical': True,
        'step_max_abs': step_error, 'prefix_causal': True, 'future_causal': True, 'TEST_accessed': False,
        'files': {p: C.sha(C.OUT / p) for p in ('smoke/weights.npz', 'smoke/prefix.npz')}})


def fit():
    sources = C.source_guard()
    inputs = C.verify_inputs()
    qualified = C.read(C.OUT / 'smoke_independent.json')
    assert qualified['all_pass'] and qualified['sources'] == sources
    assert qualified['smoke_sha256'] == C.sha(C.OUT / 'smoke.json')
    assert not C.FIT.exists(), 'One production fit only; no automatic recovery/retry'
    C.FIT.mkdir(parents=True)
    device = configure()
    prepared = D.load_prepared(C.UNIT)
    event('preparing_validation')
    val_batches = T.validation_batches(prepared, C.UNIT)
    model = M.make_model(C.ARCH, C.SEED).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5, min_lr=1e-5)
    mu, sd = (float(prepared['identity']['stats'][k]) for k in ('ti_mu', 'ti_sd'))
    start = time.monotonic()
    deadline = start + FIT_SECONDS
    history, best, best_epoch, best_state = [], float('inf'), None, None
    significant_best, last_significant, low_lr_streak = float('inf'), 0, 0
    total_updates, partial_updates = 0, 0
    converged, reason = False, 'max_epochs_safety'
    C.save(C.FIT / 'started.json', {'sources': sources, 'inputs': inputs, 'utc': T.utc(),
        'unit': C.UNIT, 'architecture': C.ARCH, 'seed': C.SEED, 'budget_seconds': FIT_SECONDS,
        'min_epochs': MIN_EPOCHS, 'max_epochs': MAX_EPOCHS})
    for epoch in range(1, MAX_EPOCHS + 1):
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
    parser.add_argument('--smoke', '--qualify', dest='smoke', action='store_true')
    args = parser.parse_args()
    C.OUT.mkdir(parents=True, exist_ok=True)
    (smoke if args.smoke else fit)()
