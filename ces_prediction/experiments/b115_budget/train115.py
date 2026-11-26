"""Fixed-architecture B115 longer schedule; original B114 checkpoints are controls.

No fitting is permitted before the parent common115 source/data registration guard.
All new fits complete and reproduce their original training prefixes before TEST.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import gc
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import common115 as C

sys.path.insert(0, str(C.HERE.parent / 'b114_overnight'))
import neural114 as OLD
from models114 import ScalarNet, ARCHITECTURES

OLD_ROOT = C.ROOT / 'data/.b114'


def ah(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def learning_rate(epoch):
    return 0.001 if epoch < 80 else 0.0003 if epoch < 160 else 0.0001 if epoch < 240 else 0.00003


def setup():
    torch = OLD.setup()
    assert all(os.environ.get(k) == v for k, v in C.ENV_PINS.items()), 'Explicit B115 pins required'
    return torch


def selection():
    z = C.read(OLD_ROOT / 'selection.json')
    return z, z['neural']['by_unit']


def fit_one(tag, architecture, form, seed, x, y, gp, weights, mu, sd, coverage,
            val, dest, smoke=False):
    torch = setup()
    # This initialization and shuffle stream intentionally match OLD.fit_one.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    label = form + '_s' + str(seed)
    target = y if form == 'direct' else y - gp
    ym = float(np.average(target, weights=weights))
    ys = float(max(np.sqrt(np.average((target - ym) ** 2, weights=weights)), 1.0))
    knots = OLD.weighted_knots(x, weights, mu, sd) if architecture == 'ple128' else None
    model = ScalarNet(x.shape[1], architecture, knots).cuda()
    tx = torch.as_tensor(((x - mu) / sd).astype(np.float32), device='cuda')
    ty = torch.as_tensor(((target - ym) / ys).astype(np.float32), device='cuda')
    tw = torch.as_tensor(weights.astype(np.float32), device='cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator(device='cuda').manual_seed(seed + 114)
    dest.mkdir(parents=True, exist_ok=True)
    oldbase = OLD_ROOT / 'neural/models' / tag / architecture
    oldhistorypath = oldbase / (label + '_history.json')
    oldweightpath = oldbase / (label + '.npz')
    oldhistory = [] if smoke else C.read(oldhistorypath)
    if not smoke:
        oldreceipt = C.read(oldbase / (label + '_complete.json'))
        assert oldreceipt['history_sha256'] == C.sha(oldhistorypath)
        assert oldreceipt['weights_sha256'] == C.sha(oldweightpath)
        assert oldreceipt['stopping'] == 'patience8' and len(oldhistory) < 80
    params = {'architecture': architecture, 'width': 128, 'residual_blocks': 3,
              'PLE_intervals': 16, 'PLE_quantiles': 'TRAIN_rule_balanced_midpoint_CDF',
              'PLE_width_floor': 1e-6, 'activation': 'GELU_exact', 'optimizer': 'AdamW',
              'weight_decay': 0.0001, 'batch_size': 8192, 'shuffle_seed_offset': 114,
              'max_epochs': 1 if smoke else 300, 'early_stopping': False, 'seconds_budget': 300,
              'learning_rates': [[0, 0.001], [80, 0.0003], [160, 0.0001], [240, 0.00003]],
              'checkpoint': 'earliest_strict_minimum_equal7rule_physical_VAL_MSE',
              'time_budget': 'check_after_each_complete_epoch_including_VAL',
              'dtype': 'float32', 'AMP': False, 'TF32': False}
    receipt = {'unit': tag, 'form': form, 'seed': seed, 'params': params,
               'TRAIN_coverage': coverage, 'TRAIN_rows': len(y),
               'VAL_fit_rows': 0, 'TEST_fit_rows': 0, 'smoke': smoke,
               'torch': torch.__version__, 'GPU': torch.cuda.get_device_name(0),
               'pooled_hashes': {k: ah(v) for k, v in dict(x=x, y=y, gp=gp, weights=weights, target=target).items()}}
    if not smoke:
        receipt['old_control'] = {'weights_sha256': C.sha(oldweightpath),
                                  'history_sha256': C.sha(oldhistorypath),
                                  'receipt_sha256': C.sha(oldbase / (label + '_complete.json'))}
    with (dest / (label + '_started.json')).open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, indent=2)
    history = []; best = float('inf'); best_epoch = None; best_state = None; stale = 0
    prefix_train_error = 0.0; prefix_val_error = 0.0
    started = time.monotonic(); stop = 'max_epochs'
    for epoch in range(params['max_epochs']):
        lr = learning_rate(epoch)
        for group in opt.param_groups:
            group['lr'] = lr
        model.train(); order = torch.randperm(len(tx), device='cuda', generator=generator); total = 0.0
        for start in range(0, len(tx), 8192):
            ids = order[start:start + 8192]; opt.zero_grad(set_to_none=True)
            loss = (tw[ids] * (model(tx[ids]) - ty[ids]).square()).mean()
            assert torch.isfinite(loss)
            loss.backward(); opt.step(); total += float(loss.detach()) * len(ids)
        assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters())
        scores = {rule: float(np.mean((OLD.infer(model, z['x'], mu, sd, ym, ys,
                    z['gp'] if form == 'residual' else None) - z['y']) ** 2)) for rule, z in val.items()}
        score = float(np.mean(list(scores.values()))) if scores else total / len(tx)
        assert np.isfinite(score)
        train_loss = total / len(tx)
        if epoch < len(oldhistory):
            original = oldhistory[epoch]
            assert original['epoch'] == epoch
            assert np.isclose(train_loss, original['train_normalized_weighted_MSE'], rtol=1e-6, atol=1e-5), 'Original TRAIN prefix mismatch'
            prefix_train_error = max(prefix_train_error, abs(train_loss - original['train_normalized_weighted_MSE']))
            assert set(scores) == set(original['VAL_rule_MSE'])
            for rule, value in scores.items():
                previous = original['VAL_rule_MSE'][rule]
                assert np.isclose(value, previous, rtol=1e-6, atol=0.01), 'Original VAL prefix mismatch'
                prefix_val_error = max(prefix_val_error, abs(value - previous))
            assert np.isclose(score, original['objective'], rtol=1e-6, atol=0.01)
        improved = score < best
        if improved:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        elapsed = time.monotonic() - started
        history.append({'epoch': epoch, 'train_normalized_weighted_MSE': train_loss,
                        'VAL_rule_MSE': scores, 'VAL_MSE': score if scores else None,
                        'objective': score, 'improved': improved, 'seconds': elapsed,
                        'learning_rate': lr})
        C.save(dest / (label + '_history.json'), history)
        C.save(C.OUT / 'progress.json', {'state': 'fitting', 'unit': tag, 'architecture': architecture,
               'form': form, 'seed': seed, 'epoch': epoch, 'best_epoch': best_epoch,
               'learning_rate': lr, 'updated': time.time()})
        print('B115 neural', tag, architecture, label, 'epoch', epoch, 'objective', score, flush=True)
        if elapsed >= 300:
            stop = 'seconds300'; break
    assert best_state is not None and best_epoch == int(np.argmin([v['objective'] for v in history]))
    assert smoke or len(history) >= len(oldhistory), 'Training budget did not reproduce old full prefix'
    model.load_state_dict(best_state); model.eval()
    arrays = {'model.' + k: v.numpy() for k, v in best_state.items()}
    arrays.update(x_mu=mu, x_sd=sd, y_mu=np.array(ym), y_sd=np.array(ys), residual=np.array(form == 'residual'))
    path = dest / (label + '.npz'); np.savez_compressed(path, **arrays)
    receipt.update(best_epoch=best_epoch, best_VAL_MSE=best if val else None, epochs=len(history),
                   stopping=stop, seconds=time.monotonic() - started,
                   weights_sha256=C.sha(path), history_sha256=C.sha(dest / (label + '_history.json')))
    receipt['prefix_check'] = {'all_pass': True, 'epochs': len(oldhistory),
                             'max_abs_train_loss': prefix_train_error, 'max_abs_VAL_MSE': prefix_val_error,
                             'rtol': 1e-6, 'train_atol': 1e-5, 'VAL_atol_ev2': 0.01,
                             'smoke_not_comparable': smoke}
    if not smoke:
        receipt['prefix_check'].update(old_history_sha256=C.sha(oldhistorypath), old_weights_sha256=C.sha(oldweightpath))
    C.save(dest / (label + '_complete.json'), receipt)
    predictions = {rule: OLD.infer(model, z['x'], mu, sd, ym, ys, z['gp'] if form == 'residual' else None)
                   for rule, z in val.items()}
    del model, tx, ty, tw, opt; torch.cuda.empty_cache()
    return predictions


def load_model(path, architecture):
    import torch
    z = C.npz(path)
    model = ScalarNet(len(z['x_mu']), architecture, z.get('model.knots')).cuda()
    model.load_state_dict({k[6:]: torch.as_tensor(v) for k, v in z.items() if k.startswith('model.')})
    return model, z


def execute(mode):
    setup(); sources = C.source_guard(); manifest = C.verify_inputs()
    frozen, choices = selection()
    assert set(choices) == set(manifest['units'])
    dest = C.OUT / 'smoke' if mode == 'smoke' else C.OUT
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / (mode + '_started.json')).open('x', encoding='utf8') as stream:
        json.dump({'sources': sources, 'started': time.time(), 'mode': mode,
                   'selection_sha256': C.sha(OLD_ROOT / 'selection.json'), 'choices': choices}, stream, indent=2)
    try:
        if mode == 'smoke':
            tag = next(iter(manifest['units']))
            _, x, y, gp, weights, mu, sd, coverage = C.load_unit(tag)
            for architecture in ARCHITECTURES:
                fit_one(tag, architecture, 'direct', C.SEEDS[0], x[:512], y[:512], gp[:512], weights[:512],
                        mu, sd, coverage, {}, dest / architecture, smoke=True)
            C.save(dest / 'complete.json', {'all_pass': True, 'fits': 2, 'TEST_predictions': 0,
                                           'VAL_predictions': 0, 'sources': sources})
            return
        if mode == 'fit':
            smoke = C.read(C.OUT / 'smoke/complete.json'); assert smoke['all_pass'] and smoke['sources'] == sources
            fits = {}; val_files = {}
            for tag, choice in choices.items():
                architecture, form = choice['architecture'], choice['form']
                data, x, y, gp, weights, mu, sd, coverage = C.load_unit(tag)
                val = {rule: {k: data[rule][k][data[rule]['split'] == 1] for k in ('idx', 'x', 'y', 'gp', 'cluster', 'discharge')}
                       for rule in C.TRAIN_RULES}
                outputs = {rule: {**{k: v[k] for k in ('idx', 'gp', 'cluster', 'discharge')}, 'truth': v['y']}
                           for rule, v in val.items()}
                for seed in C.SEEDS:
                    model_dest = C.OUT / 'models' / tag / architecture
                    predictions = fit_one(tag, architecture, form, seed, x, y, gp, weights, mu, sd, coverage, val, model_dest)
                    for rule in C.TRAIN_RULES:
                        outputs[rule]['NN_long_s' + str(seed)] = predictions[rule]
                    receipt = model_dest / (form + '_s' + str(seed) + '_complete.json')
                    fits[receipt.relative_to(C.OUT).as_posix()] = C.sha(receipt)
                for rule in C.TRAIN_RULES:
                    oldval = C.npz(OLD_ROOT / 'neural/val' / (tag + '_' + rule + '.npz'))
                    for key in ('idx', 'truth', 'gp', 'cluster', 'discharge'):
                        assert np.array_equal(outputs[rule][key], oldval[key])
                    for seed in C.SEEDS:
                        outputs[rule]['NN_base_s' + str(seed)] = oldval['NN_' + architecture + '_' + form + '_s' + str(seed)]
                    for prefix in ('NN_long', 'NN_base'):
                        outputs[rule][prefix + '_ensemble'] = np.mean([outputs[rule][prefix + '_s' + str(s)] for s in C.SEEDS], axis=0)
                    (C.OUT / 'val').mkdir(exist_ok=True)
                    path = C.OUT / 'val' / (tag + '_' + rule + '.npz'); np.savez_compressed(path, **outputs[rule])
                    val_files[path.relative_to(C.OUT).as_posix()] = C.sha(path)
                del data, x, y, gp, weights, val, outputs; gc.collect()
            assert len(fits) == 12 and len(val_files) == 28 and C.source_guard() == sources
            C.save(C.OUT / 'fit_complete.json', {'all_pass': True, 'fits': 12, 'receipts': fits,
                   'val_files': val_files, 'sources': sources, 'TEST_predictions': 0,
                   'all_prefix_checks': True, 'completed': time.time()})
            return
        fit = C.read(C.OUT / 'fit_complete.json')
        assert fit['all_pass'] and fit['fits'] == 12 and fit['all_prefix_checks'] and fit['sources'] == sources
        estimates = C.OUT / 'estimates'; estimates.mkdir(exist_ok=False); files = {}
        for tag, choice in choices.items():
            architecture, form = choice['architecture'], choice['form']
            models = {seed: load_model(C.OUT / 'models' / tag / architecture / (form + '_s' + str(seed) + '.npz'), architecture)
                      for seed in C.SEEDS}
            for rule in C.RULES:
                z = C.npz(C.PRIOR / (tag + '_' + rule + '.npz')); mask = z['split'] == 2
                arr = C.npz(C.PRIOR / (tag + '_' + rule + '_reference.npz'))
                old = C.npz(OLD_ROOT / 'neural/estimates' / (tag + '_' + rule + '.npz'))
                for k, key in [('idx','idx'),('truth','y'),('cluster','cluster'),('discharge','discharge')]:
                    assert np.array_equal(arr[k], z[key][mask]) and np.array_equal(arr[k], old[k])
                for seed, (model, state) in models.items():
                    arr['NN_long_s' + str(seed)] = OLD.infer(model, z['x'][mask], state['x_mu'], state['x_sd'],
                                    float(state['y_mu']), float(state['y_sd']), z['gp'][mask] if form == 'residual' else None)
                    arr['NN_base_s' + str(seed)] = old['NN_selected_s' + str(seed)]
                for prefix in ('NN_long', 'NN_base'):
                    arr[prefix + '_ensemble'] = np.mean([arr[prefix + '_s' + str(s)] for s in C.SEEDS], axis=0)
                assert np.array_equal(arr['NN_base_ensemble'], old['NN_selected_ensemble'])
                path = estimates / (tag + '_' + rule + '.npz'); np.savez_compressed(path, **arr)
                files[path.relative_to(C.OUT).as_posix()] = C.sha(path)
            del models; gc.collect()
        assert len(files) == 32 and C.source_guard() == sources
        C.save(C.OUT / 'prediction_complete.json', {'all_pass': True, 'files': files, 'sources': sources,
               'choices': choices, 'selection_sha256': C.sha(OLD_ROOT / 'selection.json'), 'cells': 32})
    except BaseException as exc:
        C.save(dest / (mode + '_failed.json'), {'error': repr(exc), 'traceback': traceback.format_exc(), 'no_retry': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    for option in ('smoke', 'fit', 'predict'):
        group.add_argument('--' + option, action='store_true')
    args = parser.parse_args(); execute(next(option for option in ('smoke', 'fit', 'predict') if getattr(args, option)))
