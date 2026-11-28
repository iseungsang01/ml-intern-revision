"""Fresh, same-seed B105 TCN; only stopping policy changes. Never constructs TEST."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import random
import time
import numpy as np
import torch
import common120 as C
from model105 import create_model


def main(qualify=False):
    sources = C.sources()
    assert all(os.environ.get(k) == v for k, v in C.ENV.items())
    runtime = C.B.runtime()
    C.T.configure(runtime)
    z = C.prepared()
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    model = create_model('A35', runtime, C.SEED).to('cuda')
    C.T.configure(runtime)
    stats = z['identity']['stats']
    mu, sd = [torch.tensor(stats[k], device='cuda') for k in ('ti_mu', 'ti_sd')]
    if qualify:
        from checks92 import model_contract
        s = next(s for s in z['train']['D10'] if s['mask'].any())
        n = min(len(s['x']), max(64, int(np.flatnonzero(s['mask'].any(1))[0]) + 1))
        small = [{**s, **{k: s[k][:n] for k in ('x', 'y', 'mask', 'gi')}}]
        checks = model_contract(model, small, runtime)
        b = C.T.batches(small, 'cuda')[0]
        model.train()
        loss = C.T.vector_loss(model(b['x'], b['lengths']), b['y'], b['mask'], mu, sd)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        C.save(C.OUT / 'qualification.json', {'all_pass': True, 'sources': sources,
            'checks': checks, 'optimizer_steps': 0, 'TEST_accessed': False})
        return
    assert C.read(C.OUT / 'qualification.json')['sources'] == sources
    C.FIT.mkdir(parents=True, exist_ok=False)
    C.save(C.FIT / 'config.json', {'sources': sources, 'unit': C.UNIT, 'seed': C.SEED,
        'env': C.ENV, 'runtime': runtime, 'prepared_sha256': z['artifact_sha256'],
        'fresh_fit': True, 'reason': 'Old checkpoint has no RNG states; not an exact continuation.'})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=2)
    val = {r: C.T.batches(z['val'][r], 'cpu') for r in C.RULES}
    memory = sum(b[k].numel()*b[k].element_size() for bb in val.values() for b in bb for k in ('x','y','mask'))
    free, _ = torch.cuda.mem_get_info()
    cached = memory < min(free*.35, 512*1024**2)
    class Transfer:
        def __init__(self, bb): self.bb = bb
        def __iter__(self):
            for b in self.bb:
                yield {k: v.to('cuda') if k != 'lengths' else v for k, v in b.items()}
    if cached:
        val = {r: list(Transfer(bb)) for r, bb in val.items()}
    else:
        val = {r: Transfer(bb) for r, bb in val.items()}
    history, best, significant = [], float('inf'), float('inf')
    last_significant, low_streak, updates = -1, 0, 0
    old = C.read(C.B.rd(C.UNIT) / 'metrics.json')['history']
    started = time.perf_counter()
    deadline = started + C.FIT_SECONDS
    converged, reason = False, 'epoch_safety_cap'
    for epoch in range(C.MAX_EPOCHS):
        model.train()
        loss_sum = torch.zeros((), device='cuda')
        nb, interrupted = 0, False
        lr_used = float(optimizer.param_groups[0]['lr'])
        segments = C.T.epoch_segments(z, C.UNIT, C.SEED, epoch)
        for host in C.T.batches(segments, 'cpu', shuffle_seed=epoch*10000+C.SEED):
            if time.perf_counter() >= deadline:
                interrupted = True
                break
            b = {k: v.to('cuda') if k != 'lengths' else v for k, v in host.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(b['x'], b['lengths'])
            loss = C.T.vector_loss(pred, b['y'], b['mask'], mu, sd)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.detach()
            nb += 1
            updates += 1
            C.event('training', epoch=epoch+1, batch=nb, updates=updates)
        if interrupted:
            reason = 'wall_safety_incomplete_epoch'
            break
        C.event('validation', epoch=epoch+1)
        valid = C.T.validate(model, val, deadline)
        if valid is None:
            reason = 'wall_safety_incomplete_validation'
            break
        value, details = valid
        scheduler.step(value)
        improved = value < best
        if improved:
            best = value
            C.atomic_pt(C.FIT / 'best.pt', model.state_dict())
        if value < significant*.999:
            significant, last_significant = value, epoch
        low_streak = low_streak+1 if lr_used <= 1e-5 else 0
        entry = {'epoch': epoch, 'train_loss': float((loss_sum/max(nb,1)).cpu()),
            'val_loss': value, 'validation': details, 'improved': improved, 'updates': updates,
            'lr': float(optimizer.param_groups[0]['lr']), 'lr_used': lr_used,
            'low_lr_streak': low_streak, 'last_significant_epoch': last_significant}
        if epoch < len(old):
            o = old[epoch]
            for key in ('train_loss', 'val_loss'):
                assert np.isclose(entry[key], o[key], rtol=1e-5, atol=1e-5), 'Historical prefix mismatch: '+key
            assert entry['updates'] == o['updates'] and entry['lr'] == o['lr']
        history.append(entry)
        C.save(C.FIT / 'history.json', history)
        C.atomic_pt(C.FIT / 'latest.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'history': history, 'epoch': epoch,
            'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
            'rng_numpy': np.random.get_state(), 'rng_python': random.getstate(),
            'best': best, 'significant_best': significant, 'last_significant_epoch': last_significant,
            'low_lr_streak': low_streak, 'sources': sources, 'validated': True})
        C.event('epoch_complete', **entry)
        if (epoch+1 >= C.MIN_EPOCHS and low_streak >= 10 and epoch-last_significant >= 20
                and time.perf_counter() < deadline):
            converged, reason = True, 'registered_statistical_stop'
            break
        if time.perf_counter() >= deadline:
            reason = 'wall_safety_after_validation'
            break
    assert history
    model.load_state_dict(torch.load(C.FIT/'best.pt', map_location='cuda', weights_only=True))
    model.eval()
    # Preserve masked predictions in the same batch/row order for independent NumPy audit.
    with torch.no_grad():
        for rule, bb in val.items():
            outputs = [[], []]
            for b in bb:
                p = model(b['x'], b['lengths'])
                for h in (0, 1):
                    outputs[h].append(p[..., h][b['mask'][..., h]].cpu().numpy())
            np.savez_compressed(C.FIT/(rule+'.npz'), **{f'p{h}': np.concatenate(outputs[h]) for h in (0,1)})
        # Separate full-prefix CPU replay checks the fitted numerical forward path.
        s = z['val']['D10'][0]
        x = torch.from_numpy(s['x'][:256]).unsqueeze(0).to('cuda')
        np.savez_compressed(C.FIT/'probe.npz', x=x.cpu().numpy(), prediction=model(x).cpu().numpy())
    C.save(C.OUT / 'fit_complete.json', {'sources': sources, 'converged': converged,
        'reason': reason, 'epochs': len(history), 'best': best, 'TEST_accessed': False,
        'seconds': time.perf_counter()-started, 'updates': updates,
        'best_sha256': C.sha(C.FIT/'best.pt'), 'latest_sha256': C.sha(C.FIT/'latest.pt'),
        'history_sha256': C.sha(C.FIT/'history.json'), 'validation_cached_on_device': cached,
        'prediction_files': {p.name: C.sha(p) for p in C.FIT.glob('*.npz')}})
    assert C.sources() == sources


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--qualify', action='store_true')
    main(p.parse_args().qualify)
