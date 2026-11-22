"""Independent, CPU-only B93 evidence audit. Never launches workload children.

Default reads the decision and independently rebuilds actual input membership.
An absent decision produces a pending report and exit 2, never qualification.
The audit deliberately does not import profile93 or its decision verifier.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / 'data/.b93'
BASE = ROOT / 'ces_prediction/experiments/b92_reduced'
PATHS = {
    'cuda_packed_native': dict(device='cuda', dense=False, cudnn=False, threads=2, mkldnn=False),
    'cuda_dense_native': dict(device='cuda', dense=True, cudnn=False, threads=2, mkldnn=False),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def near(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)


def current_sources():
    pinpath = BASE / 'frozen_sources92.json'
    pins = read(pinpath)
    for rel, expected in pins.items():
        path = (ROOT / rel).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == expected, rel
    paths = list(BASE.glob('*.py')) + [pinpath]
    paths += [BASE.parent / 'b93_gpu' / name for name in ('profile93.py', 'worker93.py')]
    paths += [BASE.parent / 'PREREGISTRATION_B93.md']
    return {**pins, **{p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}


def inputs(rebuild=True):
    """Hash prepared inputs; independently collate arrays, never forward a model."""
    configpath = ROOT / 'data/.b92/cpu_r2/pilot/V1_r2_k0__1701__C/config.json'
    config = read(configpath)
    key = config['prepared_key']
    path = ROOT / f'data/.b92/cpu_r2/prepared/V1_r2_k0__{key}.pt'
    meta = path.with_suffix('.json')
    assert read(meta)['key'] == key
    assert sha(path) == read(meta)['sha256'] == config['prepared_artifact_sha256']
    binding = dict(unit='V1_r2_k0', seed=1701, code='C', prepared_key=key,
                   prepared_path=path.relative_to(ROOT).as_posix(), prepared_sha256=sha(path),
                   metadata_sha256=sha(meta), config_path=configpath.relative_to(ROOT).as_posix(),
                   config_sha256=sha(configpath))
    ready = read(ROOT / 'data/.b92/cpu_r2/input_checks.json')
    expected = {p.name: sha(p) for p in BASE.glob('*.py')}
    assert ready['all_pass'] and ready['sources'] == expected, 'Stale B92 readiness'
    if not rebuild:
        return binding, None
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    import numpy as np
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(BASE))
    import common92 as M
    from model92 import create_model
    prepared = torch.load(path, map_location='cpu', weights_only=False)
    assert prepared['key'] == key and prepared['identity']['sources'] == expected
    for name, h in prepared['identity']['raw'].items():
        assert sha(ROOT / 'data/.b73' / name) == h, name
    manifest = prepared['identity']['manifest']
    assert not set(manifest['train']) & set(manifest['val'])
    assert not (set(manifest['train']) | set(manifest['val'])) & set(manifest['test'])
    keys = prepared['keys']
    assert keys == sorted(set(keys))
    lookup = {r: {(s['fidx'], s['segment']): s for s in prepared['train'][r]} for r in M.TRAIN_RULES}
    assert all(set(rows) == set(keys) for rows in lookup.values())
    train = [lookup[M.B85.rule_for('V1_r2_k0', 1701, 0, seg)][(fid, seg)] for fid, seg in keys]
    digest = hashlib.sha256()

    def collate(rows, shuffle=False):
        order = list(range(len(rows)))
        if shuffle:
            random.Random(1701).shuffle(order)
        counts = np.zeros(2, np.int64)
        for start in range(0, len(order), 16):
            ss = [rows[i] for i in order[start:start + 16]]
            lengths = np.array([len(s['x']) for s in ss], dtype=np.int64)
            width = int(lengths.max())
            arrays = [np.zeros((len(ss), width, c), dtype=d)
                      for c, d in ((35, np.float32), (2, np.float32), (2, bool))]
            for j, s in enumerate(ss):
                for array, name in zip(arrays, ('x', 'y', 'mask')):
                    array[j, :lengths[j]] = s[name]
            for array in arrays + [lengths]:
                digest.update(array.tobytes())
            counts += arrays[2].sum((0, 1))
        return (len(rows) + 15) // 16, counts.tolist()

    nb, counts = collate(train, True)
    vb, vc = {}, {}
    for rule in M.TRAIN_RULES:
        digest.update(rule.encode())
        vb[rule], vc[rule] = collate(prepared['val'][rule])
    random.seed(1701)
    np.random.seed(1701)
    torch.manual_seed(1701)
    model = create_model('C', dict(device='cpu', dense=False, cudnn=False, threads=2, mkldnn=False), 1701)
    weights = hashlib.sha256()
    for name, value in model.state_dict().items():
        weights.update(name.encode())
        weights.update(value.detach().contiguous().numpy().tobytes())
    membership = dict(sha256=digest.hexdigest(), train_batches=nb, train_segments=len(train),
                      train_rows=sum(len(s['x']) for s in train), train_supervised_counts=counts,
                      validation_batches=vb, validation_supervised_counts=vc,
                      initial_weights_sha256=weights.hexdigest(), seed=1701, epoch=0, shuffle_seed=1701)
    return binding, membership


def audit():
    path = CACHE / 'runtime_decision93.json'
    pins = current_sources()
    if not path.exists():
        return dict(state='pending', all_pass=False, reason='Immutable B93 decision is not available; no workload launched.')
    d = read(path)
    assert read(CACHE / 'runtime_report93.json') == d
    assert d['sources'] == pins
    binding, membership = inputs()
    assert d['input_binding'] == binding
    assert d['actual_batch_binding'] == membership or d['actual_batch_binding'] is None
    assert d['gpu_only'] and d['speed_factor_vs_cpu'] is None
    rows = d['records']
    serial = [r for r in rows if r.get('phase', 'serial') == 'serial']
    assert {(r['path'], r['rep']) for r in serial} == {(p, n) for p in PATHS for n in range(3)}
    assert len(serial) == 6
    assert {p.name for p in (CACHE / 'probes').glob('*.exit.json')} == {
        f"{r['path']}__{r['rep']}.exit.json" for r in rows if r['state'] != 'not_tested_budget'}
    exit_times, starts = [], []
    for r in rows:
        assert r['path'] in PATHS
        out = CACHE / 'probes' / f"{r['path']}__{r['rep']}.json"
        attpath = out.with_suffix('.exit.json')
        if r['state'] == 'not_tested_budget':
            assert r in serial and not out.exists() and not attpath.exists()
            assert d['elapsed_seconds'] >= 899
            continue
        assert r['state'] in ('clean', 'failed', 'timeout')
        assert sha(attpath) == r['exit_sha256']
        att = read(attpath)
        assert att['sources'] == pins and att['path'] == r['path'] and att['rep'] == r['rep']
        assert att['phase'] == r['phase'] and att['returncode'] == r['returncode']
        assert att['timeout'] == (r['returncode'] is None)
        assert att['result_sha256'] == (sha(out) if out.exists() else None)
        exit_times.append(att['time'])
        starts.append(r['contention_at_start']['time'])
        assert att['time'] >= r['contention_at_start']['time']
        if r['state'] != 'clean':
            assert r['returncode'] != 0 or not out.exists()
            assert (r['state'] == 'timeout') == (r['returncode'] is None)
            continue
        assert r['returncode'] == 0 and r['result_sha256'] == sha(out)
        z = read(out)
        assert z == r['result'] and z['all_pass']
        assert z['sources'] == pins and z['input_binding'] == binding
        assert z['actual_batch_binding'] == membership and z['config'] == PATHS[r['path']]
        assert z['path'] == r['path'] and z['rep'] == r['rep']
        c, t, s = z['checks'], z['timing'], z['software']
        assert all(c[k] for k in ('all_training_losses_finite', 'all_validation_sums_finite', 'all_parameters_finite'))
        assert not c['scientific_loss_values_exported'] and not c['test_inputs_read']
        assert c['actual_dropout'] == .1 and c['validation_rule_count'] == 7
        assert c['training_updates'] == membership['train_batches']
        assert not s['cudnn_enabled'] and not s['mkldnn_enabled'] and s['threads'] == 2
        assert t['validation_cached_on_device']
        for k in ('epoch_seconds', 'training_seconds', 'seven_rule_validation_seconds', 'prepare_cache_seconds'):
            assert math.isfinite(t[k]) and t[k] > 0
        near(t['total_prepare_epoch_seconds'], t['prepare_cache_seconds'] + t['epoch_seconds'])
        assert t['epoch_seconds'] >= t['training_seconds'] + t['seven_rule_validation_seconds'] - .01
    medians = {}
    for name in PATHS:
        rr = [r for r in serial if r['path'] == name]
        if all(r['state'] == 'clean' for r in rr):
            medians[name] = statistics.median(r['result']['timing']['epoch_seconds'] for r in rr)
    assert d['eligible_epoch_seconds'] == medians
    selected = min(medians, key=medians.get) if medians else None
    if selected and medians.get('cuda_packed_native', float('inf')) <= medians[selected] * 1.05:
        selected = 'cuda_packed_native'
    assert d['selected_path'] == selected
    assert d['selected_config'] == (PATHS[selected] if selected else None)
    parallel = [r for r in rows if r.get('phase') == 'parallel']
    gain, workers = None, 1
    if parallel:
        assert len(parallel) == 2 and {r['rep'] for r in parallel} == {30, 31}
        assert all(r['path'] == selected for r in parallel)
        wall = d['parallel_two_jobs_wall_seconds']
        assert math.isfinite(wall) and wall > 0
        if all(r['state'] == 'clean' for r in parallel):
            assert wall >= max(r['result']['timing']['total_prepare_epoch_seconds'] for r in parallel)
            gain = 2 * medians[selected] / wall
            near(gain, d['parallel_aggregate_gain'])
            workers = 2 if gain >= 1.10 else 1
    else:
        assert d['parallel_two_jobs_wall_seconds'] is None
    assert d['parallel_workers'] == workers
    assert d['status'] == ('qualified_for_independent_audit' if selected else 'no_qualified_acceleration')
    if gain is None:
        assert d['parallel_aggregate_gain'] is None
    assert d['elapsed_seconds'] >= 0
    if starts:
        # Epoch budget excludes interpreter startup; conservatively check recorded span.
        assert d['elapsed_seconds'] + 1 >= max(exit_times) - min(starts)
    return dict(state='audited', all_pass=True, decision_sha256=sha(path), verifier_sha256=sha(__file__),
                eligible_epoch_seconds=medians, selected_path=selected, parallel_workers=workers,
                recomputed_parallel_gain=gain, independently_rebuilt_membership=membership,
                elapsed_seconds=d['elapsed_seconds'], nominal_budget_seconds=900,
                budget_overrun_seconds=max(0, d['elapsed_seconds'] - 900),
                evidence_limits=['Child launch/end monotonic timestamps are not exported; exact 300-second timeout and 900-second wall enforcement cannot be independently reconstructed.',
                                 'Parallel wall is exported as a scalar; aggregate-gain arithmetic is independently recomputed, wall capture is source-audited.',
                                 'Finite gradient assertions are source-audited via clip_grad_norm_(error_if_nonfinite=True); gradients are not exported.'],
                all_attempted_records_checked=len(rows), serial_slots_checked=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = audit()
    except Exception as exc:
        report = dict(state='failed', all_pass=False, error=type(exc).__name__ + ': ' + str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf8')
        temp.replace(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report['all_pass'] else 2 if report['state'] == 'pending' else 1


if __name__ == '__main__':
    raise SystemExit(main())
