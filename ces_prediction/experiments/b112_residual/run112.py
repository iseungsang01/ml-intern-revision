"""Colab-only matched direct/residual MLP and HGB comparison; no retry."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
from common112 import TRAIN_RULES, RULES, SEEDS, ENV_PINS, sha, read, write, npz

W = Path(__file__).resolve().parent
FORMS = ('direct', 'residual')


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def units(manifest):
    z = manifest['units']
    return z if isinstance(z, dict) else {u['tag']: u for u in z}


def tree_values(model):
    out = {'baseline': np.asarray(model._baseline_prediction).reshape(-1)}
    assert out['baseline'].shape == (1,)
    for k, iteration in enumerate(model._predictors):
        assert len(iteration) == 1
        nodes = iteration[0].nodes
        assert not nodes['is_categorical'].any()
        for field in ('left', 'right', 'feature_idx', 'num_threshold', 'value', 'is_leaf', 'missing_go_to_left'):
            out[f'{k}_{field}'] = nodes[field]
    out['trees'] = np.array(len(model._predictors), np.int64)
    return out


def tree_predict(z, x):
    out = np.full(len(x), float(z['baseline'][0]))
    rows = np.arange(len(x))
    for k in range(int(z['trees'])):
        pre = str(k) + '_'
        node = np.zeros(len(x), np.int64)
        while True:
            active = ~z[pre + 'is_leaf'][node].astype(bool)
            if not active.any():
                break
            ii = rows[active]
            nn = node[active]
            xx = x[ii, z[pre + 'feature_idx'][nn].astype(int)]
            left = np.where(np.isnan(xx), z[pre + 'missing_go_to_left'][nn].astype(bool), xx <= z[pre + 'num_threshold'][nn])
            node[ii] = np.where(left, z[pre + 'left'][nn], z[pre + 'right'][nn])
        out += z[pre + 'value'][node]
    return out


def load_unit(tag):
    data = {}
    train = []
    coverage = {}
    for rule in RULES:
        z = npz(W / (tag + '_' + rule + '.npz'))
        n = len(z['idx'])
        assert n and z['x'].ndim == 2 and len(z['x']) == n
        assert all(len(z[k]) == n for k in ('y', 'gp', 'split', 'cluster', 'discharge'))
        assert np.isfinite(z['x']).all() and np.isfinite(z['y']).all() and np.isfinite(z['gp']).all()
        assert len(np.unique(z['idx'])) == n and np.isin(z['split'], (0, 1, 2)).all()
        if rule == 'K10':
            assert (z['split'] == 2).all(), 'K10 excluded from all training and selection'
        else:
            assert all(np.count_nonzero(z['split'] == k) >= 30 for k in (0, 1, 2))
            fit = z['split'] == 0
            train.append((rule, z['x'][fit], z['y'][fit], z['gp'][fit]))
            coverage[rule] = {name + '_sha256': array_hash(z[name][fit]) for name in ('idx', 'x', 'y', 'gp')}
            coverage[rule]['rows'] = int(fit.sum())
        data[rule] = z
    n = sum(len(v[2]) for v in train)
    x = np.concatenate([v[1] for v in train])
    y = np.concatenate([v[2] for v in train])
    gp = np.concatenate([v[3] for v in train])
    weights = np.concatenate([np.full(len(v[2]), n / (len(TRAIN_RULES) * len(v[2]))) for v in train])
    assert np.isclose(weights.sum(), n)
    mu = np.average(x, axis=0, weights=weights)
    sd = np.maximum(np.sqrt(np.average((x - mu) ** 2, axis=0, weights=weights)), 1e-8)
    assert np.isfinite(mu).all() and np.isfinite(sd).all()
    return data, x, y, gp, weights, mu, sd, coverage


def gpu_setup():
    import torch
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Actual allocated T4 required'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch


def infer_mlp(model, tx, mu_y, sd_y, gp=None):
    import torch
    parts = []
    with torch.no_grad():
        for start in range(0, len(tx), 65536):
            parts.append(model(tx[start:start + 65536]).cpu().numpy())
    out = np.concatenate(parts).astype(np.float64) * sd_y + mu_y
    if gp is not None:
        out += gp
    assert np.isfinite(out).all()
    return out


def fit_mlp(tag, form, seed, data, tx, y, gp, weights, mu_x, sd_x, coverage, dest):
    torch = gpu_setup()
    from model112 import ScalarMLP, portable_weights
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    label = 'MLP_' + form + '_s' + str(seed)
    target = y if form == 'direct' else y - gp
    mu_y = float(np.average(target, weights=weights))
    sd_y = float(max(np.sqrt(np.average((target - mu_y) ** 2, weights=weights)), 1.0))
    target_gpu = torch.as_tensor(((target - mu_y) / sd_y).astype(np.float32), device='cuda')
    weight_gpu = torch.as_tensor(weights.astype(np.float32), device='cuda')
    val = {}
    for rule in TRAIN_RULES:
        z = data[rule]
        ok = z['split'] == 1
        val[rule] = (torch.as_tensor(((z['x'][ok] - mu_x) / sd_x).astype(np.float32), device='cuda'), z['y'][ok], z['gp'][ok])
    model = ScalarMLP(tx.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device='cuda').manual_seed(seed + 112)
    config = {'unit': tag, 'form': form, 'seed': seed, 'architecture': [int(tx.shape[1]), 64, 64, 1],
              'activation': 'GELU_exact', 'dropout': 0, 'optimizer': 'AdamW', 'lr': 1e-3,
              'weight_decay': 1e-4, 'batch_size': 8192, 'max_epochs': 80, 'patience': 8,
              'loss': 'mean(weight * normalized_squared_error)', 'VAL_objective': 'mean_seven_rule_physical_MSE',
              'TRAIN_rule_weights': 'N_total/(7*N_rule)', 'normalization': 'TRAIN_rule_balanced_population',
              'TRAIN_coverage': coverage, 'VAL_fit_rows': 0, 'TEST_fit_rows': 0,
              'target_mu': mu_y, 'target_sd': sd_y, 'dtype': 'float32', 'AMP': False, 'TF32': False,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'env': {k: os.environ[k] for k in ENV_PINS}}
    write(dest / (label + '_config.json'), config)
    history = []
    best = float('inf')
    best_state = None
    best_epoch = None
    stale = 0
    started = time.time()
    for epoch in range(80):
        model.train()
        order = torch.randperm(len(tx), generator=generator, device='cuda')
        train_sum = 0.0
        for start in range(0, len(tx), 8192):
            ids = order[start:start + 8192]
            optimizer.zero_grad(set_to_none=True)
            loss = (weight_gpu[ids] * (model(tx[ids]) - target_gpu[ids]).square()).mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * len(ids)
        assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters())
        model.eval()
        scores = {}
        for rule, (vx, vy, vg) in val.items():
            pred = infer_mlp(model, vx, mu_y, sd_y, vg if form == 'residual' else None)
            scores[rule] = float(np.mean((pred - vy) ** 2))
        score = float(np.mean(list(scores.values())))
        assert np.isfinite(score)
        improved = score < best
        if improved:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_normalized_weighted_MSE': train_sum / len(tx), 'VAL_rule_MSE': scores,
                        'VAL_MSE': score, 'improved': improved, 'seconds': time.time() - started})
        write(dest / (label + '_history.json'), history)
        write(dest / 'progress.json', {'unit': tag, 'model': label, 'epoch': epoch, 'best_epoch': best_epoch,
                                       'best_VAL_MSE': best, 'stale': stale, 'updated': time.time()})
        if epoch % 5 == 0 or stale >= 8:
            print('B112 ' + tag + ' ' + label + ' epoch=' + str(epoch) + ' val=' + str(score), flush=True)
        if stale >= 8:
            break
    assert best_state is not None and best_epoch == int(np.argmin([h['VAL_MSE'] for h in history]))
    model.load_state_dict(best_state)
    model.eval()
    values = portable_weights(model)
    values.update(x_mu=mu_x, x_sd=sd_x, y_mu=np.array(mu_y), y_sd=np.array(sd_y), residual=np.array(form == 'residual'))
    np.savez_compressed(dest / (label + '.npz'), **values)
    write(dest / (label + '_complete.json'), {'best_epoch': best_epoch, 'best_VAL_MSE': best, 'epochs': len(history),
          'stopping': 'patience8' if stale >= 8 else 'max80', 'seconds': time.time() - started,
          'weights_sha256': sha(dest / (label + '.npz')), 'VAL_fit_rows': 0, 'TEST_fit_rows': 0})
    del target_gpu, weight_gpu, val, optimizer
    return model, mu_y, sd_y


def run_unit(tag):
    import joblib
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    torch = gpu_setup()
    manifest = read(W / 'manifest.json')
    assert tag in units(manifest)
    dest = W / 'models' / tag
    dest.mkdir(parents=True, exist_ok=False)
    data, x, y, gp, weights, mu_x, sd_x, coverage = load_unit(tag)
    np.savez_compressed(dest / 'stats.npz', x_mu=mu_x, x_sd=sd_x)
    tx = torch.as_tensor(((x - mu_x) / sd_x).astype(np.float32), device='cuda')
    mlps = {}
    for form in FORMS:
        for seed in SEEDS:
            mlps[(form, seed)] = fit_mlp(tag, form, int(seed), data, tx, y, gp, weights, mu_x, sd_x, coverage, dest)
    del tx
    torch.cuda.empty_cache()
    trees = {}
    for form in FORMS:
        target = y if form == 'direct' else y - gp
        model = HistGradientBoostingRegressor(random_state=20261116, early_stopping=False, max_iter=100).fit(x, target, sample_weight=weights)
        portable = tree_values(model)
        assert np.allclose(model.predict(x[:512]), tree_predict(portable, x[:512]), rtol=1e-12, atol=1e-9)
        label = 'HGB_' + form
        joblib.dump(model, dest / (label + '.joblib'))
        np.savez_compressed(dest / (label + '_trees.npz'), **portable)
        write(dest / (label + '_complete.json'), {'params': model.get_params(), 'iterations': int(model.n_iter_),
              'TRAIN_coverage': coverage, 'VAL_fit_rows': 0, 'TEST_fit_rows': 0, 'portable_equal': True,
              'joblib_sha256': sha(dest / (label + '.joblib')), 'portable_sha256': sha(dest / (label + '_trees.npz')),
              'sklearn': sklearn.__version__})
        trees[form] = (model, portable)
        print('B112 fitted ' + tag + ' ' + label, flush=True)
    # No TEST inference occurs until every neural and tree fit in this unit is complete.
    estimates = W / 'estimates'
    estimates.mkdir(exist_ok=True)
    files = {}
    for rule in RULES:
        z = data[rule]
        ok = z['split'] == 2
        ref = npz(W / (tag + '_' + rule + '_reference.npz'))
        assert np.array_equal(z['idx'][ok], ref['idx']) and np.array_equal(z['y'][ok], ref['truth'])
        assert np.array_equal(z['cluster'][ok], ref['cluster']) and np.array_equal(z['discharge'][ok], ref['discharge'])
        arr = dict(ref)
        xx = z['x'][ok]
        tg = torch.as_tensor(((xx - mu_x) / sd_x).astype(np.float32), device='cuda')
        for form in FORMS:
            for seed in SEEDS:
                model, my, sy = mlps[(form, seed)]
                arr['MLP_' + form + '_s' + str(seed)] = infer_mlp(model, tg, my, sy, z['gp'][ok] if form == 'residual' else None)
            arr['MLP_' + form + '_ensemble'] = np.mean([arr['MLP_' + form + '_s' + str(seed)] for seed in SEEDS], axis=0)
            model, portable = trees[form]
            pred = model.predict(xx)
            assert np.allclose(pred, tree_predict(portable, xx), rtol=1e-12, atol=1e-9)
            arr['HGB_' + form] = pred + (z['gp'][ok] if form == 'residual' else 0)
        assert all(np.isfinite(a).all() for a in arr.values())
        target = estimates / (tag + '_' + rule + '.npz')
        np.savez_compressed(target, **arr)
        files[target.name] = sha(target)
        del tg
    write(dest / 'complete.json', {'unit': tag, 'MLP_fits': 6, 'HGB_fits': 2, 'TEST_fit_rows': 0, 'VAL_fit_rows': 0,
          'TRAIN_coverage': coverage, 'estimates': files, 'sources': manifest['sources'], 'completed': time.time()})


def execute_units(tags):
    torch = gpu_setup()
    logs = W / 'processes'
    logs.mkdir(exist_ok=False)
    pending, active, done, failure = list(tags), [], [], None
    while pending or active:
        for item in list(active):
            timed = time.time() - item['started'] > 43200
            if timed and item['p'].poll() is None:
                item['p'].kill()
            if item['p'].poll() is None:
                continue
            code = item['p'].wait()
            item['log'].close()
            write(logs / (item['tag'] + '_exit.json'), {'returncode': code, 'timeout': timed, 'elapsed': time.time() - item['started']})
            if code or timed:
                failure = failure or item['tag']
            else:
                record = read(W / 'models' / item['tag'] / 'complete.json')
                assert record['MLP_fits'] == 6 and record['HGB_fits'] == 2 and len(record['estimates']) == len(RULES)
                assert record['TEST_fit_rows'] == record['VAL_fit_rows'] == 0
                for name, h in record['estimates'].items():
                    assert sha(W / 'estimates' / name) == h
                done.append(item['tag'])
            active.remove(item)
        if failure:
            pending.clear()
        while pending and len(active) < 4 and not failure:
            if torch.cuda.mem_get_info()[0] < 1630 * 1024 ** 2:
                break
            tag = pending.pop(0)
            env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
            log = (logs / (tag + '.log')).open('wb')
            p = subprocess.Popen([sys.executable, '-u', str(W / 'run112.py'), '--unit', tag], stdout=log, stderr=subprocess.STDOUT, env=env)
            start = time.time()
            write(logs / (tag + '_start.json'), {'pid': p.pid, 'started': start, 'tag': tag})
            active.append({'tag': tag, 'p': p, 'log': log, 'started': start})
        progress = []
        for item in active:
            path = W / 'models' / item['tag'] / 'progress.json'
            if path.exists():
                progress.append(read(path))
        write(W / 'progress.json', {'state': 'training', 'completed': len(done), 'total': len(tags), 'pending': len(pending),
              'active': [{'tag': e['tag'], 'pid': e['p'].pid} for e in active], 'unit_progress': progress,
              'failure': failure, 'updated': time.time()})
        if pending or active:
            time.sleep(2)
    assert failure is None, 'Scientific process failed; no retry: ' + str(failure)
    write(W / 'training_complete.json', {'all_pass': True, 'tags': done, 'completed': time.time()})


def summarize(manifest):
    from stats112 import compute
    cells = {}
    for tag, unit in units(manifest).items():
        name = unit['name'] if isinstance(unit, dict) else unit
        for rule in RULES:
            key = name + '/' + rule + '/F'
            cells[key] = compute(npz(W / 'estimates' / (tag + '_' + rule + '.npz')), key)
            write(W / 'progress.json', {'state': 'readout', 'completed_cells': len(cells), 'total_cells': 32, 'updated': time.time()})
    report = {'registration': 'PREREGISTRATION_B112.md', 'sources': manifest['sources'], 'cells': cells,
              'units': [v['name'] if isinstance(v, dict) else v for v in units(manifest).values()],
              'seeds': list(SEEDS), 'MLP_fits': 24, 'HGB_fits': 8, 'exploratory': True,
              'target': 'current_report_F', 'draws': 20000, 'bootstrap_seed': 20261125,
              'limits': ['Reused TEST exploratory study, not independent confirmation.',
                         'Same scalar inputs and fit rows; direct/residual target parametrizations have different scales and skip baseline.',
                         'One scalar MLP architecture, three initializations, fixed training budget; not an architecture-class conclusion.',
                         'Ensemble prediction intervals are fixed-fit pointwise intervals, omitting refit/selection uncertainty.',
                         'K10 excluded from TRAIN and VAL; training and VAL use seven equally weighted rules.']}
    fresh = [v for k, v in cells.items() if '/D10/' in k]
    assert len(fresh) == 4 and len(cells) == 32
    report['statements'] = {
        'R1_residual_over_direct_all_fresh': all(c['pairs']['MLP_residual_ensemble_over_MLP_direct_ensemble']['cluster']['ci95'][0] > 0 for c in fresh),
        'R2_residual_over_HGB_all_fresh': all(c['pairs']['MLP_residual_ensemble_over_HGB_residual']['cluster']['ci95'][0] > 0 for c in fresh),
        'R3_residual_over_HGB_all_scenarios': all(c['pairs']['MLP_residual_ensemble_over_HGB_residual']['cluster']['ci95'][0] > 0 for c in cells.values())}
    write(W / 'readout.json', report)


def run():
    import torch
    import sklearn
    assert not (W / 'started.json').exists(), 'No scientific restart'
    assert all(os.environ.get(k) == v for k, v in ENV_PINS.items()), 'Explicit CES pins required'
    gpu_setup()
    manifest = read(W / 'manifest.json')
    assert manifest['registration'] == 'PREREGISTRATION_B112.md'
    for name, h in manifest['files'].items():
        assert Path(name).name == name and sha(W / name) == h
    write(W / 'started.json', {'time': time.time(), 'torch': torch.__version__, 'sklearn': sklearn.__version__,
          'gpu': torch.cuda.get_device_name(0), 'env': {k: os.environ[k] for k in ENV_PINS}})
    execute_units(list(units(manifest)))
    summarize(manifest)
    files = [W / 'readout.json', W / 'started.json', W / 'training_complete.json', W / 'manifest.json']
    for folder in ('models', 'estimates', 'processes'):
        files.extend(p for p in (W / folder).rglob('*') if p.is_file())
    write(W / 'result_hashes.json', {p.relative_to(W).as_posix(): sha(p) for p in files})
    with zipfile.ZipFile(W / 'b112_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in [*files, W / 'result_hashes.json']:
            z.write(p, p.relative_to(W).as_posix())
    write(W / 'progress.json', {'state': 'complete', 'completed_cells': 32, 'total_cells': 32,
          'MLP_fits': 24, 'HGB_fits': 8, 'updated': time.time()})
    print('B112 all matched fits and scenario readouts complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit')
    args = parser.parse_args()
    if args.unit:
        run_unit(args.unit)
    else:
        run()
