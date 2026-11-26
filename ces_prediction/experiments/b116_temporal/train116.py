"""B116 full-segment temporal training; TEST is accessible only via --predict."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

import common116 as C
import data116 as D
import models116 as M

ARCHS = ('lstm', 'decay_gru', 'innovation_gru')
MAX_EPOCHS = 60
FIT_SECONDS = 900
BATCH_SIZE = 16


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def configure(device='cuda'):
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    for key, value in C.ENV_PINS.items():
        assert os.environ.get(key) == str(value), 'Unpinned environment: ' + key
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA required; CPU fallback forbidden'
    return torch.device(device)


def fit_dir(unit, arch, seed):
    return C.OUT / 'fits' / unit / arch / ('s' + str(seed))


def save_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), 'Refusing overwrite: ' + str(path)
    np.savez_compressed(path, **values)


def state_arrays(model):
    return {'model.' + k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def load_best(unit, arch, seed, device):
    folder = fit_dir(unit, arch, seed)
    params = C.read(folder / 'params.json')
    assert params['normal'] and params['completed_epochs'] >= 1
    assert C.sha(folder / 'best.npz') == params['best_sha256']
    model = M.make_model(arch, seed).to(device)
    with np.load(folder / 'best.npz', allow_pickle=False) as saved:
        model.load_state_dict({k[6:]: torch.as_tensor(saved[k], device=device) for k in saved.files})
    return model.eval()


def to_device(batch, device):
    return batch['x'].to(device), batch['y'].to(device), batch['mask'].to(device)


def epoch_segments(prepared, unit, seed, epoch):
    lookups = {r: {(int(s['fidx']), int(s['segment'])): s for s in prepared['train'][r]}
               for r in C.TRAIN_RULES}
    keys = sorted(lookups[C.TRAIN_RULES[0]])
    assert all(set(v) == set(keys) for v in lookups.values())
    rng = np.random.default_rng([seed, epoch, zlib.crc32(unit.encode())])
    choices = rng.integers(0, len(C.TRAIN_RULES), len(keys))
    assignment = [(k[0], k[1], C.TRAIN_RULES[int(i)]) for k, i in zip(keys, choices)]
    return [lookups[r][(f, s)] for f, s, r in assignment], digest(assignment)


def validation_batches(prepared, unit):
    result = {}
    for rule in C.TRAIN_RULES:
        support = D.common_row_support(unit, rule, split='val')
        wanted = np.asarray(support['idx'], dtype=np.int64)
        assert len(wanted) and len(np.unique(wanted)) == len(wanted)
        truth = {int(i): float(y) for i, y in zip(wanted, support['truth'])}
        batches = list(D.batches(prepared['val'][rule], 'cpu', batch_size=BATCH_SIZE))
        found = []
        for batch in batches:
            anon = np.asarray(D.anonymous_indices(batch['gi'].numpy()), dtype=np.int64)
            mask = batch['mask'].numpy() & np.isin(anon, wanted)
            batch['score_mask'] = torch.as_tensor(mask)
            batch['score_truth'] = torch.tensor([truth[int(i)] for i in anon[mask]], dtype=torch.float64)
            found.extend(anon[mask].tolist())
        assert np.array_equal(np.sort(found), np.sort(wanted)), 'VAL support mismatch: ' + unit + '/' + rule
        result[rule] = batches
    return result


@torch.no_grad()
def validate(model, batches_by_rule, mu, sd, device, deadline):
    model.eval()
    metrics = {}
    for rule, batches in batches_by_rule.items():
        total, count = 0.0, 0
        for batch in batches:
            if time.monotonic() >= deadline:
                return None
            x, y, _ = to_device(batch, device)
            pred = model(x, batch['lengths'])
            assert pred.shape == y.shape
            mask = batch['score_mask'].to(device)
            error = pred[mask].double() * sd + mu - batch['score_truth'].to(device)
            total += float((error * error).sum().item())
            count += int(mask.sum().item())
        assert count > 0 and np.isfinite(total)
        metrics[rule] = {'sse': total, 'count': count, 'mse': total / count}
    if time.monotonic() >= deadline:
        return None
    return metrics


def fit_one(unit, arch, seed, prepared, val_batches, device, sources):
    folder = fit_dir(unit, arch, seed)
    assert not folder.exists(), 'No fit retries: ' + str(folder)
    folder.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + FIT_SECONDS
    model = M.make_model(arch, seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5, min_lr=1e-5)
    sd = float(prepared['identity']['stats']['ti_sd'])
    mu = float(prepared['identity']['stats']['ti_mu'])
    best, best_epoch, best_state = float('inf'), None, None
    history, total_updates, partial_updates = [], 0, 0
    stop = 'max_epochs'
    C.save(folder / 'started.json', {'unit': unit, 'architecture': arch, 'seed': seed, 'utc': utc(),
                                   'budget_seconds': FIT_SECONDS, 'max_epochs': MAX_EPOCHS, 'sources': sources})
    for epoch in range(1, MAX_EPOCHS + 1):
        if time.monotonic() >= deadline:
            stop = 'deadline_before_epoch'
            break
        segments, assignment_sha = epoch_segments(prepared, unit, seed, epoch)
        batches = D.batches(segments, 'cpu', batch_size=BATCH_SIZE, shuffle_seed=seed * 1000 + epoch)
        model.train()
        updates, train_sse, train_count = 0, 0.0, 0
        for batch in batches:
            if time.monotonic() >= deadline:
                break
            x, y, mask = to_device(batch, device)
            if not bool(mask.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, batch['lengths'])
            assert pred.shape == y.shape
            error = pred[mask] - y[mask]
            loss = error.square().mean()
            assert torch.isfinite(loss).item(), 'Nonfinite TRAIN loss'
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total_updates += 1
            train_sse += float(error.detach().double().square().sum().item())
            train_count += int(mask.sum().item())
        else:
            metrics = validate(model, val_batches, mu, sd, device, deadline)
            if metrics is not None:
                score = float(np.mean([metrics[r]['mse'] for r in C.TRAIN_RULES]))
                lr_used = float(optimizer.param_groups[0]['lr'])
                scheduler.step(score)
                if score < best:
                    best, best_epoch, best_state = score, epoch, state_arrays(model)
                history.append({'epoch': epoch, 'train_normalized_sse': train_sse,
                                'train_count': train_count, 'updates': updates,
                                'assignment_sha256': assignment_sha, 'val': metrics,
                                'seven_rule_mean_mse': score, 'lr_used': lr_used,
                                'lr_next': float(optimizer.param_groups[0]['lr']),
                                'elapsed_seconds': time.monotonic() - started})
                C.save(folder / 'history.json', history)
                C.save(C.OUT / 'progress.json', {'unit': unit, 'architecture': arch,
                       'seed': seed, 'epoch': epoch, 'best_epoch': best_epoch,
                       'elapsed_seconds': time.monotonic() - started, 'best_val_mse': best})
                continue
        partial_updates = updates
        stop = 'deadline_in_incomplete_epoch'
        break
    if best_state is None:
        C.save(folder / 'failure.json', {'normal': False, 'reason': 'no_complete_epoch',
               'total_updates': total_updates, 'discarded_partial_updates': partial_updates,
               'actual_seconds': time.monotonic() - started, 'sources': sources})
        raise RuntimeError('No complete TRAIN+seven-rule VAL epoch within fit budget; no retry')
    save_npz(folder / 'best.npz', best_state)
    params = {'normal': True, 'unit': unit, 'architecture': arch, 'seed': seed,
              'completed_epochs': len(history), 'best_epoch': best_epoch, 'best_val_mse': best,
              'completed_utc': utc(), 'history_sha256': C.sha(folder / 'history.json'),
              'best_sha256': C.sha(folder / 'best.npz'), 'total_updates': total_updates,
              'discarded_partial_updates': partial_updates, 'stop_reason': stop,
              'actual_seconds': time.monotonic() - started, 'budget_seconds': FIT_SECONDS,
              'max_epochs': MAX_EPOCHS, 'batch_size': BATCH_SIZE, 'sources': sources,
              'optimizer': {'name': 'AdamW', 'initial_lr': .001, 'weight_decay': .0001,
                            'clip_grad_norm': 1.0},
              'scheduler': {'name': 'ReduceLROnPlateau', 'patience': 3, 'factor': .5,
                            'min_lr': 1e-5},
              'manifest_sha256': digest(prepared['identity']['manifest']),
              'stats_sha256': digest(prepared['identity']['stats']),
              'prepared_artifact_sha256': prepared.get('artifact_sha256'),
              'target': 'F', 'objective': 'masked_normalized_MSE',
              'selection': 'earliest_strict_minimum_common_support_seven_rule_physical_VAL_MSE',
              'precision': 'float32', 'cudnn': False, 'tf32': False, 'amp': False, 'threads': 2}
    C.save(folder / 'params.json', params)
    del model, optimizer, scheduler, best_state
    torch.cuda.empty_cache()
    return params


@torch.no_grad()
def predict_rows(model, segments, support, stats, device):
    wanted = np.asarray(support['idx'], dtype=np.int64)
    idxs, preds = [], []
    for batch in D.batches(segments, 'cpu', batch_size=BATCH_SIZE):
        x = batch['x'].to(device)
        pred = model(x, batch['lengths']).cpu().numpy()
        anon = np.asarray(D.anonymous_indices(batch['gi'].numpy()), dtype=np.int64)
        keep = batch['mask'].numpy() & np.isin(anon, wanted)
        idxs.extend(anon[keep].tolist())
        preds.extend(pred[keep].tolist())
    idxs, preds = np.asarray(idxs, dtype=np.int64), np.asarray(preds, dtype=np.float64)
    order = np.argsort(idxs)
    assert np.array_equal(idxs[order], np.sort(wanted)), 'Missing/duplicate prediction support'
    physical = preds[order][np.searchsorted(idxs[order], wanted)] * float(stats['ti_sd']) + float(stats['ti_mu'])
    assert np.isfinite(physical).all()
    return physical


def export_split(split, device):
    files = {}
    for unit in C.UNITS:
        prepared = D.load_prepared(unit)
        stats = prepared['identity']['stats']
        rules = C.TRAIN_RULES if split == 'val' else C.RULES
        values = {rule: dict(D.common_row_support(unit, rule, split=split)) for rule in rules}
        for arch in ARCHS:
            for seed in C.SEEDS:
                model = load_best(unit, arch, seed, device)
                for rule in rules:
                    segments = prepared['val'][rule] if split == 'val' else D.test_segments(unit, rule)
                    values[rule]['Seq_' + arch + '_s' + str(seed)] = predict_rows(model, segments, values[rule], stats, device)
                del model
            for rule in rules:
                values[rule]['Seq_' + arch + '_ensemble'] = np.mean(
                    [values[rule]['Seq_' + arch + '_s' + str(seed)] for seed in C.SEEDS], axis=0)
        for rule in rules:
            path = C.OUT / split / (unit + '_' + rule + '.npz')
            save_npz(path, values[rule])
            files[path.relative_to(C.OUT).as_posix()] = C.sha(path)
    return files


def fit(device):
    sources = C.source_guard()
    C.verify_inputs()
    assert not (C.OUT / 'fit_complete.json').exists(), 'No fit reruns'
    receipts = []
    for unit in C.UNITS:
        prepared = D.load_prepared(unit)
        val_batches = validation_batches(prepared, unit)
        for arch in ARCHS:
            for seed in C.SEEDS:
                params = fit_one(unit, arch, seed, prepared, val_batches, device, sources)
                path = fit_dir(unit, arch, seed) / 'params.json'
                receipts.append({'unit': unit, 'architecture': arch, 'seed': seed,
                                 'path': path.relative_to(C.OUT).as_posix(), 'sha256': C.sha(path),
                                 'normal': params['normal']})
        del val_batches, prepared
    assert len(receipts) == 36 and all(r['normal'] for r in receipts)
    files = export_split('val', device)
    assert C.source_guard() == sources
    C.save(C.OUT / 'fit_complete.json', {'all_pass': True, 'fits': receipts,
           'files': files, 'sources': sources, 'test_accessed': False, 'utc': utc()})


def predict(device):
    sources = C.source_guard()
    C.verify_inputs()
    complete = C.read(C.OUT / 'fit_complete.json')
    assert complete['all_pass'] and len(complete['fits']) == 36
    assert complete['sources'] == sources
    for receipt in complete['fits']:
        assert receipt['normal'] and C.sha(C.OUT / receipt['path']) == receipt['sha256']
    for rel, expected in complete['files'].items():
        assert C.sha(C.OUT / rel) == expected
    selection_path = C.OUT / 'selection.json'
    selection_sha = C.sha(selection_path)
    selection = C.read(selection_path)
    assert set(selection['by_unit']) == set(C.UNITS)
    assert all(v['architecture'] in ARCHS for v in selection['by_unit'].values())
    assert not (C.OUT / 'predictions_complete.json').exists(), 'No prediction reruns'
    assert not (C.OUT / 'predict_started.json').exists(), 'No prediction retries'
    C.save(C.OUT / 'predict_started.json', {'utc': utc(), 'selection_sha256': selection_sha,
           'fit_complete_sha256': C.sha(C.OUT / 'fit_complete.json'), 'sources': sources})
    files = export_split('test', device)
    assert len(files) == 32
    assert C.sha(selection_path) == selection_sha and C.source_guard() == sources
    C.save(C.OUT / 'predictions_complete.json', {'all_pass': True, 'files': files,
           'sources': sources, 'selection_sha256': selection_sha, 'utc': utc(),
           'fit_complete_sha256': C.sha(C.OUT / 'fit_complete.json')})


def smoke(device):
    sources = C.source_guard()
    C.verify_inputs()
    assert not (C.OUT / 'smoke').exists(), 'No smoke retries'
    assert not (C.OUT / 'smoke.json').exists(), 'No smoke reruns'
    prepared = D.load_prepared(C.UNITS[0])
    candidates = []
    for segment in prepared['train'][C.TRAIN_RULES[0]]:
        if np.asarray(segment['mask'])[:64, 0].any():
            copied = {k: v[:64].copy() if isinstance(v, np.ndarray) and v.ndim else v
                      for k, v in segment.items()}
            candidates.append(copied)
        if len(candidates) == 2:
            break
    assert len(candidates) == 2, 'Need two real TRAIN segments with labels within first64 rows'
    batch = next(D.batches(candidates, 'cpu', batch_size=2))
    x, y, mask = to_device(batch, device)
    results = {}
    for arch in ARCHS:
        model = M.make_model(arch, C.SEEDS[0]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        model.train()
        loss = (model(x, batch['lengths'])[mask] - y[mask]).square().mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            full = model(x, batch['lengths'])
            prefix_length = min(8, x.shape[1])
            prefix = model(x[:, :prefix_length], batch['lengths'].clamp(max=prefix_length))
            prefix_error = float((prefix - full[:, :prefix_length]).abs().max().item())
            assert prefix_error <= 2e-5, 'Noncausal prefix'
            state, steps = None, []
            for t in range(x.shape[1]):
                step, state = model.forward_step(x[:, t], state)
                steps.append(step)
            replay_error = float((torch.stack(steps, dim=1) - full).abs().max().item())
            assert replay_error <= 2e-5, 'Sequence/step mismatch'
        weight_path = C.OUT / 'smoke' / (arch + '_weights.npz')
        prefix_path = C.OUT / 'smoke' / (arch + '_prefix.npz')
        save_npz(weight_path, state_arrays(model))
        save_npz(prefix_path, {'x': batch['x'].numpy(), 'pred': full.detach().cpu().numpy(),
                 'lengths': batch['lengths'].numpy(), 'y': batch['y'].numpy(),
                 'mask': batch['mask'].numpy(), 'gi': batch['gi'].numpy(),
                 'fidx': np.asarray(batch['fidx'], dtype=np.int64),
                 'segment': np.asarray(batch['segment'], dtype=np.int64)})
        results[arch] = {'normal': True, 'loss': float(loss.item()),
                         'prefix_max_abs': prefix_error, 'step_max_abs': replay_error,
                         'training_updates': 1,
                         'files': {weight_path.relative_to(C.OUT).as_posix(): C.sha(weight_path),
                                   prefix_path.relative_to(C.OUT).as_posix(): C.sha(prefix_path)},
                         'parameter_count': sum(p.numel() for p in model.parameters())}
        del model, optimizer
    assert C.source_guard() == sources
    C.save(C.OUT / 'smoke.json', {'all_pass': True, 'architectures': results,
           'train_segments': 2, 'max_rows': 64, 'test_accessed': False,
           'validation': 'TRAIN_prefix_only', 'sources': sources,
           'unit': C.UNITS[0], 'rule': C.TRAIN_RULES[0], 'seed': C.SEEDS[0], 'utc': utc()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--fit', action='store_true')
    mode.add_argument('--predict', action='store_true')
    args = parser.parse_args()
    target = configure()
    (smoke if args.smoke else fit if args.fit else predict)(target)
