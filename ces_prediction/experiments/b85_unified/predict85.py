# Claude authored the implementation
# ces_prediction/experiments/b85_unified/predict85.py
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import sys
import json
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'ces_prediction' / 'experiments' / 'b85_unified'))

import torch

from common85 import (
    CACHE, DATA, RULES, INITS, VARIANTS,
    load_sources, select_units, stats_hash, manifest_hash
)
from data85 import unit_manifest, fit_stats, build_segments
from model85 import create_model


def predict_all():
    units = select_units()
    from train85 import verify_complete

    # Strict: all 90 complete with passed input checks
    ic_path = CACHE / 'input_checks.json'
    assert ic_path.exists(), 'Input checks not found'
    with open(ic_path) as f:
        assert json.load(f).get('all_pass', False), 'Input checks not passed'

    for u in units:
        for init in INITS:
            for var in VARIANTS:
                rd = CACHE / 'runs' / f"{u['name']}__{init}__{var}"
                assert verify_complete(rd), f'Not complete or stale: {rd}'

    G, R = load_sources()
    r_lookup = {}
    if 'gi' in R:
        for idx in range(len(R['gi'])):
            r_lookup[int(R['gi'][idx])] = idx

    pred_dir = CACHE / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_dir = Path(__file__).resolve().parent
    src_hash = hashlib.md5(
        (src_dir / 'predict85.py').read_bytes() +
        (src_dir / 'common85.py').read_bytes()
    ).hexdigest()

    for u in units:
        name = u['name']
        manifest = unit_manifest(u)
        test_ids = manifest['test']
        train_ids = manifest['train']

        # Load saved stats and verify hash
        for init in INITS:
            for var in VARIANTS:
                run_dir = CACHE / 'runs' / f'{name}__{init}__{var}'
                best_path = run_dir / 'best.pt'
                assert best_path.exists(), f'best.pt missing (no fallback): {run_dir}'

                # Verify saved stats/manifest hashes
                with open(run_dir / 'config.json') as f:
                    cfg = json.load(f)
                with open(run_dir / 'manifest.json') as f:
                    saved_manifest = json.load(f)

                with open(run_dir / 'stats.json') as f:
                    stats = json.load(f)
                assert stats_hash(stats) == cfg['stats_hash'], \
                    f'Stats hash mismatch: {name}__{init}__{var}'
                assert manifest_hash(saved_manifest) == cfg['manifest_hash'], \
                    f'Manifest hash mismatch: {name}__{init}__{var}'

                ti_mu = float(stats['ti_mu'])
                ti_sd = float(stats['ti_sd'])

                state_dict = torch.load(best_path, map_location=device, weights_only=True)
                model = create_model(variant=var, init_seed=init)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()

                for rule in RULES:
                    segs = build_segments(test_ids, stats, rule, G, R)
                    all_idx = []
                    all_est = []

                    with torch.no_grad():
                        for seg in segs:
                            x = torch.tensor(seg['x'], dtype=torch.float32).unsqueeze(0).to(device)
                            lengths = torch.tensor([len(seg['x'])], dtype=torch.long)
                            pred = model(x, lengths)
                            pred_np = pred.cpu().numpy()[0]  # (L, 2) normalized

                            # Denormalize: eV = pred * ti_sd + ti_mu
                            est_eV = pred_np * ti_sd + ti_mu

                            # Map gi to R row indices (B73 row indices)
                            gi_arr = seg['gi']
                            row_idx = []
                            row_est = []
                            for i, gi in enumerate(gi_arr):
                                gi_int = int(gi)
                                if gi_int in r_lookup:
                                    row_idx.append(r_lookup[gi_int])
                                    row_est.append(est_eV[i])

                            if row_idx:
                                all_idx.append(np.array(row_idx, dtype=np.int64))
                                all_est.append(np.array(row_est, dtype=np.float32))

                    if all_idx:
                        idx = np.concatenate(all_idx)
                        est = np.concatenate(all_est)

                        # Sort by idx for consistency
                        order = np.argsort(idx)
                        idx = idx[order]
                        est = est[order]

                        assert np.isfinite(est).all(), \
                            f'Non-finite predictions: {name}__{init}__{var}__{rule}'

                        out_path = pred_dir / f'{name}__{init}__{var}__{rule}.npz'
                        tmp = out_path.with_suffix('.tmp.npz')
                        np.savez(tmp, idx=idx, est=est, source_hash=np.array(src_hash),config_hash=np.array(cfg['config_hash']))
                        tmp.replace(out_path)

                print(f'Predicted: {name}__{init}__{var}')

    # Label mutation invariance check: mutate R labels, predictions unchanged
    print('Label mutation invariance check...')
    u0 = units[0]
    name0 = u0['name']
    manifest0 = unit_manifest(u0)
    stats0 = fit_stats(manifest0['train'], G, R)
    rd0 = CACHE / 'runs' / f'{name0}__{INITS[0]}__X'
    sd0 = torch.load(rd0 / 'best.pt', map_location=device, weights_only=True)
    m0 = create_model(variant='X', init_seed=INITS[0])
    m0.load_state_dict(sd0)
    m0.to(device)
    m0.eval()
    test_fids = manifest0['test'][:2]
    segs_a = build_segments(test_fids, stats0, 'D100', G=G, R=R)
    R_mut = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in R.items()}
    if 'tg_F' in R_mut:
        R_mut['tg_F'] = R_mut['tg_F'] + 500.0
    if 'tg_L5' in R_mut:
        R_mut['tg_L5'] = R_mut['tg_L5'] + 500.0
    segs_b = build_segments(test_fids, stats0, 'D100', G=G, R=R_mut)
    with torch.no_grad():
        for sa, sb in zip(segs_a, segs_b):
            xa = torch.tensor(sa['x']).unsqueeze(0).to(device)
            xb = torch.tensor(sb['x']).unsqueeze(0).to(device)
            la = torch.tensor([len(sa['x'])], dtype=torch.long)
            pa = m0(xa, la)
            pb = m0(xb, la)
            assert torch.allclose(pa, pb, atol=1e-6), 'Label mutation changed predictions'
    print('Label mutation invariance OK')

    # Verification
    print('Verifying predictions...')
    checks = {'all_pass': True, 'source_hash': src_hash}
    for u in units:
        name = u['name']
        for init in INITS:
            for rule in RULES:
                px = pred_dir / f'{name}__{init}__X__{rule}.npz'
                ph = pred_dir / f'{name}__{init}__H__{rule}.npz'
                if not px.exists() or not ph.exists():
                    checks['all_pass'] = False
                    checks[f'{name}__{init}__{rule}'] = 'missing'
                    continue
                dx = np.load(px)
                dh = np.load(ph)
                if not np.array_equal(dx['idx'], dh['idx']):
                    checks['all_pass'] = False
                    checks[f'{name}__{init}__{rule}'] = 'idx_mismatch'
                if not np.isfinite(dx['est']).all() or not np.isfinite(dh['est']).all():
                    checks['all_pass'] = False
                    checks[f'{name}__{init}__{rule}'] = 'non_finite'

    pc_path = CACHE / 'prediction_checks.json'
    with open(pc_path, 'w') as f:
        json.dump(checks, f, indent=2)
    print(f'Prediction checks: all_pass={checks["all_pass"]} -> {pc_path}')
    return checks['all_pass']


if __name__ == '__main__':
    success = predict_all()
    sys.exit(0 if success else 1)
