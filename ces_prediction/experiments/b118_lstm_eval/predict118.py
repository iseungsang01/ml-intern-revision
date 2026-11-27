"""Eight forward-only TEST scenarios from one fixed B117 checkpoint; no fitting."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import time
import numpy as np
import torch
import common118 as C
import data116 as D
import models116 as M
import train116 as T


def main():
    sources, inputs = C.source_guard(), C.verify_inputs()
    C.OUT.mkdir(parents=True, exist_ok=True)
    with (C.OUT / 'predict_started.json').open('x', encoding='utf8') as stream:
        import json
        json.dump({'started': time.time(), 'sources': sources, 'inputs': inputs,
                   'optimization_steps': 0}, stream)
    for key, value in C.ENV_PINS.items():
        assert os.environ.get(key) == value
    device = T.configure()
    weights = C.npz(C.FIT / 'best.npz')
    model = M.make_model(C.ARCH, C.SEED).to(device)
    model.load_state_dict({k.removeprefix('model.'): torch.as_tensor(v, device=device) for k, v in weights.items()})
    model.eval()
    prepared = D.load_prepared(C.UNIT)
    before = {'best': C.sha(C.FIT / 'best.npz'), 'latest': C.sha(C.FIT / 'latest.pt')}
    files = {}
    with torch.inference_mode():
        for rule in C.RULES:
            C.save(C.OUT / 'progress.json', {'stage': 'forward_only', 'rule': rule,
                'updated': time.time(), 'optimization_steps': 0})
            support = D.common_row_support(C.UNIT, rule, 'test')
            old = C.npz(C.ROOT / 'data/.b116/test' / (C.UNIT + '_' + rule + '.npz'))
            for key in ('idx', 'truth', 'cluster', 'discharge', 'GP_fast_boost'):
                assert np.array_equal(support[key], old[key])
            prediction = T.predict_rows(model, D.test_segments(C.UNIT, rule), support,
                                        prepared['identity']['stats'], device)
            values = {key: support[key] for key in ('idx', 'truth', 'cluster', 'discharge', 'GP_fast_boost')}
            values.update(LSTM_focused=prediction, LSTM_short=old['Seq_lstm_s1701'])
            values.update(C.test_coordinates(support['idx'], support['discharge']))
            assert all(np.isfinite(values[k]).all() for k in ('LSTM_focused', 'LSTM_short', 'GP_fast_boost'))
            path = C.OUT / 'test' / (C.UNIT + '_' + rule + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                np.savez_compressed(stream, **values)
            files[path.relative_to(C.OUT).as_posix()] = C.sha(path)
    current = T.state_arrays(model)
    assert set(current) == set(weights) and all(np.array_equal(current[k], weights[k]) for k in weights)
    after = {'best': C.sha(C.FIT / 'best.npz'), 'latest': C.sha(C.FIT / 'latest.pt')}
    assert before == after and C.source_guard() == sources and C.verify_inputs() == inputs
    C.save(C.OUT / 'predictions_complete.json', {'all_pass': True, 'sources': sources,
        'inputs': inputs, 'files': files, 'scenarios': len(files), 'optimization_steps': 0,
        'model_state_unchanged': True, 'checkpoint_before': before, 'checkpoint_after': after,
        'completed': time.time()})


if __name__ == '__main__':
    main()
