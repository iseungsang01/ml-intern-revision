"""B114 fixed GPU scalar networks: TRAIN fitting and VAL-only checkpoint choice."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import gc
import hashlib
import random
import time
import traceback
import numpy as np
import common114 as C
from models114 import ScalarNet, ARCHITECTURES

FORMS = ('direct', 'residual')
BASE = C.OUT / 'neural'


def ah(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def setup():
    import torch
    assert torch.cuda.is_available(), 'Actual GPU required; no CPU training fallback'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    assert all(os.environ.get(k) == v for k, v in C.ENV_PINS.items()), 'Explicit CES pins required'
    return torch


def weighted_knots(x, weights, mu, sd):
    """TRAIN rule-balanced midpoint-CDF quantiles, then fixed TRAIN scaling."""
    result = []
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind='stable')
        values, ww = x[order, j], weights[order]
        pos = (np.cumsum(ww) - 0.5 * ww) / ww.sum()
        q = np.interp(np.linspace(0, 1, 17), pos, values,
                      left=values[0], right=values[-1])
        result.append((q - mu[j]) / sd[j])
    out = np.asarray(result, dtype=np.float32)
    assert np.isfinite(out).all() and (np.diff(out, axis=1) >= 0).all()
    return out


def infer(model, x, mu, sd, ym, ys, gp=None):
    import torch
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), 8192):
            batch = torch.as_tensor(((x[start:start + 8192] - mu) / sd).astype(np.float32), device='cuda')
            values.append(model(batch).cpu().numpy())
    out = np.concatenate(values).astype(np.float64) * ys + ym
    if gp is not None:
        out += gp
    assert np.isfinite(out).all()
    return out


def fit_one(tag, architecture, form, seed, x, y, gp, weights, mu, sd, coverage,
            val, dest, smoke=False):
    torch = setup()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    label = form + '_s' + str(seed)
    target = y if form == 'direct' else y - gp
    ym = float(np.average(target, weights=weights))
    ys = float(max(np.sqrt(np.average((target - ym) ** 2, weights=weights)), 1.0))
    knots = weighted_knots(x, weights, mu, sd) if architecture == 'ple128' else None
    model = ScalarNet(x.shape[1], architecture, knots).cuda()
    tx = torch.as_tensor(((x - mu) / sd).astype(np.float32), device='cuda')
    ty = torch.as_tensor(((target - ym) / ys).astype(np.float32), device='cuda')
    tw = torch.as_tensor(weights.astype(np.float32), device='cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator(device='cuda').manual_seed(seed + 114)
    dest.mkdir(parents=True, exist_ok=True)
    params = {'architecture': architecture, 'width': 128, 'residual_blocks': 3,
              'PLE_intervals': 16, 'PLE_quantiles': 'TRAIN_rule_balanced_midpoint_CDF',
              'PLE_width_floor': 1e-6, 'activation': 'GELU_exact', 'optimizer': 'AdamW',
              'lr': 0.001, 'weight_decay': 0.0001, 'batch_size': 8192,
              'max_epochs': 1 if smoke else 80, 'patience': 8, 'seconds_budget': 180,
              'checkpoint': 'earliest_strict_minimum_equal7rule_physical_VAL_MSE',
              'time_budget': 'check_after_each_complete_epoch_including_VAL',
              'dtype': 'float32', 'AMP': False, 'TF32': False}
    receipt = {'unit': tag, 'form': form, 'seed': seed, 'params': params, 'TRAIN_coverage': coverage,
               'TRAIN_rows': len(y), 'VAL_fit_rows': 0, 'TEST_fit_rows': 0, 'smoke': smoke,
               'torch': torch.__version__, 'GPU': torch.cuda.get_device_name(0),
               'pooled_hashes': {k: ah(v) for k, v in dict(x=x, y=y, gp=gp, weights=weights, target=target).items()}}
    with (dest / (label + '_started.json')).open('x', encoding='utf8') as stream:
        import json
        json.dump(receipt, stream, indent=2)
    history = []; best = float('inf'); best_epoch = None; best_state = None; stale = 0
    started = time.monotonic(); stop = 'max_epochs'
    for epoch in range(params['max_epochs']):
        model.train(); order = torch.randperm(len(tx), device='cuda', generator=generator); total = 0.0
        for start in range(0, len(tx), 8192):
            ids = order[start:start + 8192]; opt.zero_grad(set_to_none=True)
            loss = (tw[ids] * (model(tx[ids]) - ty[ids]).square()).mean()
            assert torch.isfinite(loss)
            loss.backward(); opt.step(); total += float(loss.detach()) * len(ids)
        assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters())
        # Smoke uses a real TRAIN prefix only and never touches VAL or TEST.
        scores = {rule: float(np.mean((infer(model, z['x'], mu, sd, ym, ys,
                    z['gp'] if form == 'residual' else None) - z['y']) ** 2)) for rule, z in val.items()}
        score = float(np.mean(list(scores.values()))) if scores else total / len(tx)
        assert np.isfinite(score)
        improved = score < best
        if improved:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        elapsed = time.monotonic() - started
        history.append({'epoch': epoch, 'train_normalized_weighted_MSE': total / len(tx),
                        'VAL_rule_MSE': scores, 'VAL_MSE': score if scores else None,
                        'objective': score, 'improved': improved, 'seconds': elapsed})
        C.save(dest / (label + '_history.json'), history)
        C.save(BASE / 'progress.json', {'state': 'fitting', 'unit': tag, 'architecture': architecture,
               'form': form, 'seed': seed, 'epoch': epoch, 'best_epoch': best_epoch, 'updated': time.time()})
        print('B114 neural', tag, architecture, label, 'epoch', epoch, 'objective', score, flush=True)
        if stale >= 8:
            stop = 'patience8'; break
        if elapsed >= 180:
            stop = 'seconds180'; break
    assert best_state is not None and best_epoch == int(np.argmin([v['objective'] for v in history]))
    model.load_state_dict(best_state); model.eval()
    arrays = {'model.' + k: v.numpy() for k, v in best_state.items()}
    arrays.update(x_mu=mu, x_sd=sd, y_mu=np.array(ym), y_sd=np.array(ys), residual=np.array(form == 'residual'))
    path = dest / (label + '.npz'); np.savez_compressed(path, **arrays)
    receipt.update(best_epoch=best_epoch, best_VAL_MSE=best if val else None, epochs=len(history),
                   stopping=stop, seconds=time.monotonic() - started, weights_sha256=C.sha(path),
                   history_sha256=C.sha(dest / (label + '_history.json')))
    C.save(dest / (label + '_complete.json'), receipt)
    predictions = {rule: infer(model, z['x'], mu, sd, ym, ys, z['gp'] if form == 'residual' else None)
                   for rule, z in val.items()}
    del model, tx, ty, tw, opt; torch.cuda.empty_cache()
    return predictions


def load_model(path, architecture):
    import torch
    z = C.npz(path)
    dim = len(z['x_mu'])
    model = ScalarNet(dim, architecture, z.get('model.knots')).cuda()
    model.load_state_dict({k[6:]: torch.as_tensor(v) for k, v in z.items() if k.startswith('model.')})
    return model, z


def execute(mode):
    setup(); sources = C.source_guard(); manifest = C.verify_inputs()
    dest = BASE / 'smoke' if mode == 'smoke' else BASE
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / (mode + '_started.json')).open('x', encoding='utf8') as stream:
        import json
        json.dump({'sources': sources, 'started': time.time(), 'mode': mode}, stream, indent=2)
    try:
        if mode == 'smoke':
            tag = next(iter(manifest['units']))
            _, x, y, gp, weights, mu, sd, coverage = C.load_unit(tag)
            for architecture in ARCHITECTURES:
                fit_one(tag, architecture, 'direct', C.SEEDS[0], x[:512], y[:512], gp[:512], weights[:512],
                        mu, sd, coverage, {}, dest / architecture, smoke=True)
            C.save(dest / 'complete.json', {'all_pass': True, 'fits': 2, 'TEST_predictions': 0, 'VAL_predictions': 0, 'sources': sources})
            return
        if mode == 'fit':
            assert C.read(BASE / 'smoke/complete.json')['all_pass']
            assert C.read(BASE / 'smoke/complete.json')['sources'] == sources
            fits = {}; val_files = {}
            for tag in manifest['units']:
                data, x, y, gp, weights, mu, sd, coverage = C.load_unit(tag)
                val = {rule: {k: data[rule][k][data[rule]['split'] == 1] for k in ('idx', 'x', 'y', 'gp', 'cluster', 'discharge')}
                       for rule in C.TRAIN_RULES}
                outputs = {rule: {**{k: v[k] for k in ('idx', 'gp', 'cluster', 'discharge')},
                                  'truth': v['y']} for rule, v in val.items()}
                for architecture in ARCHITECTURES:
                    for form in FORMS:
                        names = []
                        for seed in C.SEEDS:
                            model_dest = BASE / 'models' / tag / architecture
                            predictions = fit_one(tag, architecture, form, seed, x, y, gp, weights, mu, sd, coverage, val, model_dest)
                            name = 'NN_' + architecture + '_' + form + '_s' + str(seed); names.append(name)
                            for rule in C.TRAIN_RULES:
                                outputs[rule][name] = predictions[rule]
                            receipt = model_dest / (form + '_s' + str(seed) + '_complete.json')
                            fits[receipt.relative_to(BASE).as_posix()] = C.sha(receipt)
                        for rule in C.TRAIN_RULES:
                            outputs[rule]['NN_' + architecture + '_' + form + '_ensemble'] = np.mean([outputs[rule][n] for n in names], axis=0)
                (BASE / 'val').mkdir(exist_ok=True)
                for rule, values in outputs.items():
                    path = BASE / 'val' / (tag + '_' + rule + '.npz'); np.savez_compressed(path, **values)
                    val_files[path.relative_to(BASE).as_posix()] = C.sha(path)
                del data, x, y, gp, weights, val, outputs; gc.collect()
            assert len(fits) == 48 and len(val_files) == 28 and C.source_guard() == sources
            C.save(BASE / 'fit_complete.json', {'all_pass': True, 'fits': 48, 'receipts': fits, 'val_files': val_files, 'sources': sources,
                                              'TEST_predictions': 0, 'completed': time.time()})
            return
        assert C.read(C.OUT / 'all_fits_complete.json')['all_pass'], 'Global fit barrier required'
        assert C.read(BASE / 'fit_complete.json')['all_pass']
        selected = C.read(C.OUT / 'selection.json')['neural']
        assert set(selected['by_unit']) == set(manifest['units']), 'One within-unit VAL selection per outer unit required'
        estimates = BASE / 'estimates'; estimates.mkdir(exist_ok=False); files = {}
        for tag in manifest['units']:
            choice = selected['by_unit'][tag]
            architecture, form = choice['architecture'], choice['form']
            assert architecture in ARCHITECTURES and form in FORMS
            models = {seed: load_model(BASE / 'models' / tag / architecture / (form + '_s' + str(seed) + '.npz'), architecture) for seed in C.SEEDS}
            for rule in C.RULES:
                z = C.npz(C.PRIOR / (tag + '_' + rule + '.npz')); mask = z['split'] == 2
                arr = C.npz(C.PRIOR / (tag + '_' + rule + '_reference.npz'))
                for k, key in [('idx','idx'),('truth','y'),('cluster','cluster'),('discharge','discharge')]:
                    assert np.array_equal(arr[k], z[key][mask])
                names = []
                for seed, (model, state) in models.items():
                    name = 'NN_selected_s' + str(seed); names.append(name)
                    arr[name] = infer(model, z['x'][mask], state['x_mu'], state['x_sd'], float(state['y_mu']), float(state['y_sd']), z['gp'][mask] if form == 'residual' else None)
                arr['NN_selected_ensemble'] = np.mean([arr[n] for n in names], axis=0)
                path = estimates / (tag + '_' + rule + '.npz'); np.savez_compressed(path, **arr)
                files[path.relative_to(BASE).as_posix()] = C.sha(path)
            del models; gc.collect()
        assert len(files) == 32 and C.source_guard() == sources
        C.save(BASE / 'prediction_complete.json', {'all_pass': True, 'files': files, 'sources': sources,
               'selection': selected, 'selection_sha256': C.sha(C.OUT / 'selection.json'), 'cells': 32})
    except BaseException as exc:
        C.save(dest / (mode + '_failed.json'), {'error': repr(exc), 'traceback': traceback.format_exc(), 'no_retry': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    for option in ('smoke', 'fit', 'predict'): group.add_argument('--' + option, action='store_true')
    args = parser.parse_args(); execute(next(option for option in ('smoke', 'fit', 'predict') if getattr(args, option)))
