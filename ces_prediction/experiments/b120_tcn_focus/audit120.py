"""Separate CPU/NumPy validation arithmetic and model-forward audit; no fitting."""
import math
import numpy as np
import torch
import common120 as C
from model105 import create_model


def main():
    sources = C.sources()
    z = C.prepared()
    done = C.read(C.OUT/'fit_complete.json')
    assert done['sources'] == sources and done['TEST_accessed'] is False
    for name in ('best', 'latest', 'history'):
        suffix = '.json' if name == 'history' else '.pt'
        assert C.sha(C.FIT/(name+suffix)) == done[name+'_sha256']
    hist = C.read(C.FIT/'history.json')
    latest = torch.load(C.FIT/'latest.pt', map_location='cpu', weights_only=False)
    assert latest['history'] == hist and latest['validated'] and latest['sources'] == sources
    assert all(k in latest for k in ('rng_cpu','rng_cuda','rng_numpy','rng_python','optimizer','scheduler'))
    best_index = min(range(len(hist)), key=lambda i: hist[i]['val_loss'])
    best = hist[best_index]
    significant, last, streak = float('inf'), -1, 0
    for i, h in enumerate(hist):
        assert h['epoch'] == i
        score = np.mean([np.mean(np.asarray(h['validation'][r]['squared_sums']) /
                                  np.asarray(h['validation'][r]['counts'])) for r in C.RULES])
        assert math.isclose(score, h['val_loss'], rel_tol=1e-9, abs_tol=1e-10)
        if score < significant*.999:
            significant, last = score, i
        streak = streak+1 if h['lr_used'] <= 1e-5 else 0
        assert (h['last_significant_epoch'], h['low_lr_streak']) == (last, streak)
    expected = len(hist) >= C.MIN_EPOCHS and streak >= 10 and len(hist)-1-last >= 20
    assert bool(done['converged']) == (expected and done['reason'] == 'registered_statistical_stop')
    scores = {}
    for rule in C.RULES:
        pth = C.FIT/(rule+'.npz')
        assert C.sha(pth) == done['prediction_files'][pth.name]
        pred = np.load(pth, allow_pickle=False)
        bb = C.T.batches(z['val'][rule], 'cpu')
        heads = []
        for h in (0,1):
            y = np.concatenate([b['y'][...,h][b['mask'][...,h]].numpy() for b in bb])
            p = pred[f'p{h}']
            assert p.shape == y.shape and np.isfinite(p).all()
            mse = float(np.mean((p.astype('float64')-y.astype('float64'))**2))
            reference = best['validation'][rule]
            assert len(y) == reference['counts'][h]
            assert np.isclose(mse, reference['squared_sums'][h]/len(y), rtol=1e-5, atol=1e-7)
            heads.append(mse)
        scores[rule] = heads
    runtime = {**C.B.runtime(), 'device':'cpu', 'threads':2, 'cudnn':False}
    C.T.configure(runtime)
    model = create_model('A35', runtime, C.SEED)
    model.load_state_dict(torch.load(C.FIT/'best.pt', map_location='cpu', weights_only=True))
    model.eval()
    probe = np.load(C.FIT/'probe.npz', allow_pickle=False)
    assert C.sha(C.FIT/'probe.npz') == done['prediction_files']['probe.npz']
    assert np.array_equal(probe['x'][0], z['val']['D10'][0]['x'][:256])
    with torch.no_grad():
        actual = model(torch.from_numpy(probe['x'])).numpy()
    assert np.allclose(actual, probe['prediction'], rtol=1e-5, atol=1e-5)
    previous = min(h['val_loss'] for h in C.read(C.B.rd(C.UNIT)/'metrics.json')['history'])
    reduction = 1-best['val_loss']/previous
    original_history = C.read(C.B.rd(C.UNIT)/'metrics.json')['history']
    sd2 = float(z['identity']['stats']['ti_sd'])**2
    physical = {r: [v*sd2 for v in scores[r]] for r in C.RULES}
    report = {'all_pass':True, 'sources':sources, 'converged':done['converged'],
        'best_epoch':best_index+1, 'completed_epochs':len(hist), 'old_val':previous,
        'new_val':best['val_loss'], 'relative_VAL_reduction':reduction,
        'meaningful_VAL_gate':bool(done['converged'] and reduction >= .01),
        'head_rule_normalized_MSE':scores, 'head_rule_physical_MSE_eV2':physical,
        'head_order':['F','L5'], 'history':hist, 'original_history':original_history, 'TEST_accessed':False,
        'fit_complete_sha256':C.sha(C.OUT/'fit_complete.json'),
        'limitation':'TRAIN/VAL optimization pilot only; no GP+fast or TEST improvement established.'}
    C.save(C.OUT/'audit.json', report)
    C.save(C.ROOT/'docs/paper/b120_training.json', report)
    text = ('# B120 TCN training diagnostic\n\nOne fixed unit/seed; fresh same-seed fit, not checkpoint continuation. '
        'TRAIN/VAL only; reused validation and TEST-informed exploratory planning. No confidence interval or generalization claim.\n\n'
        'V1_r2_k0: TRAIN/VAL pilot. V2_s1_k0, V3_q3, V4a: NOT RUN. None primary.\n\n'
        + f"Convergence criterion met: {done['converged']}. Completed epochs: {len(hist)}. Best epoch: {best_index+1}.\n\n"
        + f"Original normalized VAL MSE: {previous:.9g}; new: {best['val_loss']:.9g}; relative reduction: {reduction:.6%}.\n\n"
        + 'This is the equal-seven-rule, equal-two-head F/L5 validation selection metric, not F-only TEST performance. '
        'The registered one-percent VAL gate only motivates a separately registered evaluation; no automatic TEST access.\n')
    (C.ROOT/'docs/paper/b120_training.md').write_text(text, encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9,7), constrained_layout=True)
    for history, label in ((original_history,'Original TCN'),(hist,'Focused TCN')):
        axes[0].plot([h['epoch']+1 for h in history], [h['val_loss'] for h in history], label=label+' VAL')
        axes[1].step([h['epoch']+1 for h in history], [h['lr'] for h in history], where='post', label=label)
    axes[0].set(ylabel='Equal-rule/head normalized VAL MSE', title='B120 TRAIN/VAL optimization pilot; no TEST evaluation')
    axes[1].set(xlabel='Completed epoch', ylabel='Next learning rate', yscale='log')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.savefig(C.ROOT/'docs/paper/b120_learning_curves.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
