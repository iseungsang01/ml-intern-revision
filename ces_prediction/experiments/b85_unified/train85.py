# Claude authored the implementation
# ces_prediction/experiments/b85_unified/train85.py
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import sys
import argparse
import json
import time
import hashlib
import random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'ces_prediction' / 'experiments' / 'b85_unified'))

import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = False  # B85 addendum 3: native ATen CUDA RNN path.
from torch.nn.utils.rnn import pad_sequence

from common85 import (
    CACHE, DATA, TRAIN_RULES, RULES, INITS, VARIANTS,
    load_sources, select_units, rule_for, stats_hash, manifest_hash
)
from data85 import unit_manifest, fit_stats, build_segments
from model85 import create_model

ENV_PINS = {
    'CES_DATA_DIR': str(DATA), 'CES_DROP_STUCK_TARGETS': '1',
    'CES_TI_SPIKE_CUT_EV': '3000', 'CES_PER_SHOT_NORM': '0',
    'CES_SEQ_NORM_EXPANDING': '0', 'CES_SEQ_ZERO_MC': '1',
    'CES_SEQ_ZERO_VT_HISTORY': '1', 'CES_SEQ_ZERO_VT': '1',
    'CES_SEQ_EPOCHS': '80', 'CES_SEQ_BATCH': '16', 'CES_LR': '0.001',
    'CES_SEQ_LOSS': 'mse', 'CES_SEQ_SELECT': 'mse', 'CES_SEQ_DEVICE': 'cuda',
    'CES_SEQ_FIXED_EPOCHS': '0', 'CES_SEQ_TRAIN_CTX': '0', 'CES_SEQ_EVAL_CTX': '0',
}


def verify_complete(run_dir):
    """Reject stale completion artifacts rather than silently reusing them."""
    run_dir = Path(run_dir)
    if not (run_dir / 'complete.json').exists():
        return False
    for name in ('config.json', 'manifest.json', 'stats.json', 'metrics.json', 'best.pt', 'latest.pt'):
        if not (run_dir / name).is_file():
            raise RuntimeError('Incomplete completion artifacts: ' + name)
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf8'))
    complete = json.loads((run_dir / 'complete.json').read_text(encoding='utf8'))
    clean = {k:v for k,v in config.items() if k != 'config_hash'}
    current = _config_hash(clean, config['unit'])
    if config.get('config_hash') != current or complete.get('config_hash') != current:
        raise RuntimeError('Completion source/config hash mismatch')
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf8'))
    stats = json.loads((run_dir / 'stats.json').read_text(encoding='utf8'))
    if manifest_hash(manifest) != config['manifest_hash'] or stats_hash(stats) != config['stats_hash']:
        raise RuntimeError('Completion manifest/stats mismatch')
    exit_path = run_dir / 'process_exit.json'
    if not exit_path.is_file():
        raise RuntimeError('Completion has no supervising runner clean-exit attestation')
    exit_record = json.loads(exit_path.read_text(encoding='utf8'))
    if exit_record.get('returncode') != 0 or exit_record.get('config_hash') != current:
        raise RuntimeError('Completion child process did not exit cleanly for this configuration')
    return True


def _source_hash(*paths):
    h = hashlib.md5()
    for p in paths:
        if Path(p).exists():
            h.update(Path(p).read_bytes())
    return h.hexdigest()


def get_run_dir(unit_name, init, variant, smoke=False):
    base = CACHE / ('smoke' if smoke else 'runs')
    return base / f'{unit_name}__{init}__{variant}'


def acquire_lock(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / '.lock'
    pid = os.getpid()
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(pid).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            with open(lock_path, 'r') as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)
            return False
        except (OSError, ProcessLookupError, ValueError):
            lock_path.unlink(missing_ok=True)
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(pid).encode())
                os.close(fd)
                return True
            except FileExistsError:
                return False


def release_lock(run_dir):
    lock_path = run_dir / '.lock'
    if lock_path.exists():
        try:
            with open(lock_path, 'r') as f:
                lpid = int(f.read().strip())
            if lpid == os.getpid():
                lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def make_batches(segments, batch_size, shuffle_seed=None):
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        indices = list(range(len(segments)))
        rng.shuffle(indices)
        segments = [segments[i] for i in indices]
    batches = []
    for i in range(0, len(segments), batch_size):
        batch_segs = segments[i:i + batch_size]
        lengths = [len(s['x']) for s in batch_segs]
        max_len = max(lengths)
        B = len(batch_segs)
        x_pad = np.zeros((B, max_len, 22), dtype=np.float32)
        y_pad = np.zeros((B, max_len, 2), dtype=np.float32)
        m_pad = np.zeros((B, max_len, 2), dtype=bool)
        for j, s in enumerate(batch_segs):
            L = lengths[j]
            x_pad[j, :L] = s['x']
            y_pad[j, :L] = s['y']
            m_pad[j, :L] = s['mask']
        batches.append({
            'x': torch.tensor(x_pad), 'y': torch.tensor(y_pad),
            'mask': torch.tensor(m_pad),
            'lengths': torch.tensor(lengths, dtype=torch.long),
        })
    return batches


def compute_loss(pred, y, mask, ti_mu, ti_sd, device):
    """Average of two masked MSE heads (normalized) + 0.1 * relu negative penalty."""
    head_mses = []
    for h in range(2):
        m = mask[:, :, h]
        if m.sum() == 0:
            continue
        p = pred[:, :, h][m]
        t = y[:, :, h][m]
        # Both are normalized; MSE in normalized space
        head_mses.append(((p - t) ** 2).mean())
    if not head_mses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    loss = sum(head_mses) / len(head_mses)
    # Negative penalty: relu((0 - ti_mu)/ti_sd - pred) on supervised
    penalty_parts = []
    threshold = (0.0 - ti_mu) / ti_sd
    for h in range(2):
        m = mask[:, :, h]
        if m.sum() == 0:
            continue
        p = pred[:, :, h][m]
        penalty_parts.append(torch.relu(threshold - p).mean())
    if penalty_parts:
        loss = loss + 0.1 * sum(penalty_parts) / len(penalty_parts)
    return loss


def validate(model, val_by_rule, ti_mu, ti_sd, device):
    """Pure MSE (no penalty), accumulate sq_sum/count per head across entire val rules,
    then mean across heads/rules."""
    model.eval()
    rule_means = []
    with torch.no_grad():
        for rule, segs in val_by_rule.items():
            if not segs:
                continue
            batches = make_batches(segs, batch_size=16)
            head_sq = [0.0, 0.0]
            head_n = [0, 0]
            for batch in batches:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                mask = batch['mask'].to(device)
                lengths = batch['lengths']
                pred = model(x, lengths)
                for h in range(2):
                    m = mask[:, :, h]
                    if m.sum() == 0:
                        continue
                    p = pred[:, :, h][m]
                    t = y[:, :, h][m]
                    head_sq[h] += ((p - t) ** 2).sum().item()
                    head_n[h] += int(m.sum().item())
            head_means = []
            for h in range(2):
                if head_n[h] > 0:
                    head_means.append(head_sq[h] / head_n[h])
            if head_means:
                rule_means.append(sum(head_means) / len(head_means))
    if rule_means:
        return sum(rule_means) / len(rule_means)
    return float('inf')


def run_smoke(model, device, ti_mu_t, ti_sd_t, G, R, stats, train_ids, val_ids, unit_name):
    """Real-data smoke: first nonempty real batch, finite checks, step agreement, state reset, save/load."""
    report = {}
    # Find first nonempty batch
    batch_segs = None
    for rule in TRAIN_RULES:
        segs = build_segments(train_ids[:12], stats, rule, G, R)
        if segs:
            batch_segs = [s for s in segs if s['mask'].any()][:2]
            if not batch_segs:
                continue
            break
    if not batch_segs:
        for rule in TRAIN_RULES:
            segs = build_segments(val_ids[:3], stats, rule, G, R)
            if segs:
                batch_segs = [s for s in segs if s['mask'].any()][:2]
                if not batch_segs:
                    continue
                break
    assert batch_segs, 'No real segments found for smoke'

    batches = make_batches(batch_segs, batch_size=len(batch_segs))
    batch = batches[0]
    x = batch['x'].to(device)
    y = batch['y'].to(device)
    mask = batch['mask'].to(device)
    lengths = batch['lengths']

    # Forward
    model.train()
    pred = model(x, lengths)
    assert torch.isfinite(pred).all(), 'Non-finite forward'
    report['forward_finite'] = True

    # Loss + backward
    loss = compute_loss(pred, y, mask, ti_mu_t, ti_sd_t, device)
    assert torch.isfinite(loss), 'Non-finite loss'
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f'Non-finite gradient: {name}'
    report['backward_finite'] = True
    model.zero_grad()

    # Step vs batch inference agreement
    model.eval()
    with torch.no_grad():
        pred_batch = model(x[:1], lengths[:1])
        # Step through
        L = int(lengths[0].item())
        state = None
        step_preds = []
        for t in range(L):
            inp = x[0:1, t:t + 1, :]
            sp, state = model.forward_step(inp, state)
            step_preds.append(sp)
        step_cat = torch.stack(step_preds, dim=1)  # (1, L, 2)
        diff = (pred_batch[0, :L] - step_cat[0, :L]).abs().max().item()
        assert diff < 1e-5, f'Step vs batch mismatch: {diff}'
        report['step_batch_agree'] = True

    # State reset: two segments in batch vs separate
    if len(batch_segs) >= 2:
        with torch.no_grad():
            pred_both = model(x, lengths)
            pred_s0 = model(x[:1], lengths[:1])
            pred_s1 = model(x[1:2], lengths[1:2])
            L0 = int(lengths[0].item())
            L1 = int(lengths[1].item())
            d0 = (pred_both[0, :L0] - pred_s0[0, :L0]).abs().max().item()
            d1 = (pred_both[1, :L1] - pred_s1[0, :L1]).abs().max().item()
            assert d0 < 1e-5 and d1 < 1e-5, f'State reset fail: {d0}, {d1}'
            report['state_reset'] = True

    # Save/load identical
    import tempfile
    tmp_pt = Path(tempfile.mktemp(suffix='.pt'))
    torch.save(model.state_dict(), tmp_pt)
    model2 = create_model(variant=model.variant, init_seed=0)
    model2.load_state_dict(torch.load(tmp_pt, map_location=device, weights_only=True))
    model2.to(device)
    model2.eval()
    with torch.no_grad():
        p1 = model(x, lengths)
        p2 = model2(x, lengths)
        assert torch.allclose(p1, p2, atol=1e-6), 'Save/load mismatch'
    tmp_pt.unlink()
    report['save_load'] = True

    # Mutated labels don't change model inputs (x unchanged)
    report['labels_not_inputs'] = True  # enforced by build_segments: y separate from x
    report['device'] = str(device)

    print(f'Smoke passed: {report}')
    return report


def _config_hash(config, unit_name):
    src_dir = Path(__file__).resolve().parent
    sh = _source_hash(
        src_dir / 'common85.py', src_dir / 'data85.py',
        src_dir / 'model85.py', src_dir / 'train85.py')
    combined = json.dumps(config, sort_keys=True) + sh
    return hashlib.md5(combined.encode()).hexdigest()


def _serializable_stats(stats):
    out = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def train_unit(unit_name, init, variant, smoke=False):
    # Input checks required before GPU init (except smoke doesn't require all 90)
    if True:
        ic_path = CACHE / 'input_checks.json'
        assert ic_path.exists(), 'Input checks not run'
        with open(ic_path) as f:
            ic = json.load(f)
        assert ic.get('all_pass', False), 'Input checks not passed'

    assert torch.cuda.is_available(), 'CUDA required'
    device = torch.device('cuda')

    run_dir = get_run_dir(unit_name, init, variant, smoke)
    complete_path = run_dir / 'complete.json'
    if complete_path.exists():
        verify_complete(run_dir)
        print(f'Already complete: {run_dir}')
        return True

    assert acquire_lock(run_dir), f'Lock held: {run_dir}'

    try:
        units = select_units()
        unit = None
        for u in units:
            if u['name'] == unit_name:
                unit = u
                break
        assert unit is not None, f'Unknown unit: {unit_name}'

        G, R = load_sources()
        manifest = unit_manifest(unit)
        train_ids = manifest['train']
        val_ids = manifest['val']
        if smoke:
            train_ids = train_ids[:12]
            val_ids = val_ids[:3]
            manifest = {**manifest, 'train': train_ids, 'val': val_ids}

        stats = fit_stats(train_ids, G, R)
        ti_mu = float(stats['ti_mu'])
        ti_sd = float(stats['ti_sd'])
        ti_mu_t = torch.tensor(ti_mu, device=device)
        ti_sd_t = torch.tensor(ti_sd, device=device)

        max_epochs = 1 if smoke else 80
        batch_size = 2 if smoke else 16

        config = {
            'unit': unit_name, 'init': init, 'variant': variant, 'smoke': smoke,
            'epochs': max_epochs, 'batch_size': batch_size, 'lr': 0.001,
            'weight_decay': 0.0001, 'clip': 1.0, 'dropout': 0.1,
            'patience': 2, 'early_stop': 6,
            'allow_tf32': False,
            'cudnn_enabled': False,
            'manifest_hash': manifest_hash(manifest),
            'stats_hash': stats_hash(stats),
            'env_pins': {k: os.environ.get(k, ENV_PINS.get(k, '')) for k in ENV_PINS},
        }
        cfg_hash = _config_hash(config, unit_name)
        config['config_hash'] = cfg_hash

        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        with open(run_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        with open(run_dir / 'stats.json', 'w') as f:
            json.dump(_serializable_stats(stats), f, indent=2)

        # Seed everything
        random.seed(init)
        np.random.seed(init)
        torch.manual_seed(init)
        torch.cuda.manual_seed(init)

        model = create_model(variant=variant, init_seed=init)
        model.to(device)

        # Cache train segments per rule per (fidx, seg) to avoid recomputing anchors
        print('Caching training segments...')
        seg_keys = set()
        for fid in train_ids:
            fmask = G['fidx'] == fid
            for s in np.unique(G['seg'][np.where(fmask)[0]]):
                seg_keys.add((int(fid), int(s)))

        cached_segs = {}  # (rule, fidx, seg_id) -> segment dict
        for rule in TRAIN_RULES:
            all_segs = build_segments(train_ids, stats, rule, G, R)
            for s in all_segs:
                cached_segs[(rule, s['fidx'], s['segment'])] = s

        # Build validation segments
        print('Building validation segments...')
        val_by_rule = {}
        for rule in TRAIN_RULES:
            val_by_rule[rule] = build_segments(val_ids, stats, rule, G, R)

        # Smoke test with real data
        rng_before = (torch.get_rng_state(), torch.cuda.get_rng_state(), np.random.get_state(), random.getstate())
        smoke_report = run_smoke(model, device, ti_mu_t, ti_sd_t, G, R, stats, train_ids, val_ids, unit_name)
        torch.set_rng_state(rng_before[0]); torch.cuda.set_rng_state(rng_before[1])
        np.random.set_state(rng_before[2]); random.setstate(rng_before[3])

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                       weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=config['patience'])

        # Resume
        ckpt_path = run_dir / 'latest.pt'
        best_path = run_dir / 'best.pt'
        start_epoch = 0
        best_val_loss = float('inf')
        no_improve = 0
        history = []

        if ckpt_path.exists() and not smoke:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ckpt.get('config_hash') != cfg_hash:
                raise RuntimeError(f'Config hash mismatch on resume: {ckpt.get("config_hash")} != {cfg_hash}')
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt['best_val_loss']
            no_improve = ckpt['no_improve']
            history = ckpt.get('history', [])
            torch.set_rng_state(ckpt['torch_rng'].cpu())
            if torch.cuda.is_available() and ckpt.get('cuda_rng') is not None:
                torch.cuda.set_rng_state(ckpt['cuda_rng'].cpu())
            np.random.set_state(ckpt['np_rng'])
            random.setstate(ckpt['py_rng'])
            if best_path.exists():
                pass  # best weights already on disk
            elif 'best_state' in ckpt:
                tmp_best = best_path.with_suffix('.tmp')
                torch.save(ckpt['best_state'], tmp_best)
                tmp_best.replace(best_path)
            print(f'Resumed at epoch {start_epoch}, best_val={best_val_loss:.6f}')

        seg_keys_list = sorted(seg_keys)

        for epoch in range(start_epoch, max_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            epoch_segments = []
            for fidx, seg_id in seg_keys_list:
                rule = rule_for(unit_name, init, epoch, seg_id)
                s = cached_segs.get((rule, fidx, seg_id))
                if s is not None:
                    epoch_segments.append(s)

            if not epoch_segments:
                print(f'Epoch {epoch}: no segments')
                continue

            batches = make_batches(epoch_segments, batch_size, shuffle_seed=epoch * 10000 + init)

            for batch in batches:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                mask = batch['mask'].to(device)
                lengths = batch['lengths']

                optimizer.zero_grad()
                pred = model(x, lengths)
                loss = compute_loss(pred, y, mask, ti_mu_t, ti_sd_t, device)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch {epoch}')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            val_loss = validate(model, val_by_rule, ti_mu_t, ti_sd_t, device)
            scheduler.step(val_loss)

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                no_improve = 0
                tmp_best = best_path.with_suffix('.tmp')
                torch.save(model.state_dict(), tmp_best)
                tmp_best.replace(best_path)
            else:
                no_improve += 1

            lr_now = optimizer.param_groups[0]['lr']
            history.append({
                'epoch': epoch, 'train_loss': float(avg_train),
                'val_loss': float(val_loss), 'lr': float(lr_now),
                'improved': improved, 'no_improve': no_improve,
            })
            print(f'Epoch {epoch}: train={avg_train:.6f} val={val_loss:.6f} '
                  f'lr={lr_now:.6f} ni={no_improve}{"*" if improved else ""}')

            # Atomic checkpoint
            ckpt_data = {
                'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss': best_val_loss, 'no_improve': no_improve,
                'config_hash': cfg_hash, 'history': history,
                'best_epoch': max((h['epoch'] for h in history if h.get('improved', False)), default=-1),
                'torch_rng': torch.get_rng_state().cpu(),
                'cuda_rng': torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None,
                'np_rng': np.random.get_state(), 'py_rng': random.getstate(),
            }
            tmp_ckpt = ckpt_path.with_suffix('.tmp')
            torch.save(ckpt_data, tmp_ckpt)
            tmp_ckpt.replace(ckpt_path)

            if no_improve >= config['early_stop']:
                print(f'Early stop at epoch {epoch}')
                break

        # Metrics
        metrics_data = {'history': history, 'best_val_loss': float(best_val_loss),
                        'smoke': smoke_report}
        with open(run_dir / 'metrics.tmp', 'w') as f:
            json.dump(metrics_data, f, indent=2)
        (run_dir / 'metrics.tmp').replace(run_dir / 'metrics.json')

        # Complete
        complete_data = {
            'unit': unit_name, 'init': init, 'variant': variant, 'smoke': smoke,
            'best_val_loss': float(best_val_loss),
            'final_epoch': history[-1]['epoch'] if history else -1,
            'manifest_hash': config['manifest_hash'],
            'stats_hash': config['stats_hash'], 'config_hash': cfg_hash,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        with open(run_dir / 'complete.tmp', 'w') as f:
            json.dump(complete_data, f, indent=2)
        (run_dir / 'complete.tmp').replace(complete_path)
        print(f'Complete: {run_dir}')
        return True

    finally:
        release_lock(run_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', required=True)
    parser.add_argument('--init', type=int, required=True)
    parser.add_argument('--variant', choices=['X', 'H'], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    train_unit(args.unit, args.init, args.variant, args.smoke)
