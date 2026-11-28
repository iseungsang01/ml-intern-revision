"""Explicit extended stopping policy; frozen numerical audit adapted only at policy literals."""
import inspect
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'data/.b119'
MIN_EPOCHS, MAX_EPOCHS, FIT_SECONDS = 100, 1000, 86400


def prior_active_seconds():
    read = lambda p: json.loads(p.read_text(encoding='utf8'))
    previous = read(OUT/'workflow/resume1/precheck.json')
    segment = read(OUT/'workflow/resume1/stage_resume_fit_started.json')
    pause = read(OUT/'workflow/resume1/author_stop_resume1.json')
    # Conservatively charge stage setup and unsaved partial work as active time.
    return previous['prior_active_seconds'] + pause['updated'] - segment['started']


def stop_tracking(history):
    significant, last, low_streak = float('inf'), 0, 0
    output = []
    for epoch, entry in enumerate(history, 1):
        score = float(entry['seven_rule_mean_mse'])
        if not np.isfinite(significant) or score < significant*.999:
            significant, last = score, epoch
        low_streak = low_streak+1 if entry['lr_used'] <= 1e-5 else 0
        no_significant = epoch-last
        converged = (epoch >= 100 and low_streak >= 20 and no_significant >= 40
                     and entry['elapsed_seconds'] < 86400)
        output.append({'epoch':epoch, 'meaningful_reference_mse':significant,
            'last_meaningful_epoch':last, 'epochs_without_meaningful_improvement':no_significant,
            'low_lr_streak':low_streak, 'converged':converged})
    return output


def install(auditor):
    """Retain all frozen arithmetic/scheduler/data checks; replace registered policy only."""
    text = inspect.getsource(auditor.history_checks)
    replacements = {
        '1 <= len(history) <= 200': '1 <= len(history) <= 1000',
        '(50, 200, 21600)': '(100, 1000, 86400)',
        "len(history) == 200 or params['actual_seconds'] >= 21600":
            "len(history) == 1000 or params['actual_seconds'] >= 86400",
    }
    for old, new in replacements.items():
        assert text.count(old) == 1, 'Frozen auditor policy location changed: '+old
        text = text.replace(old, new)
    auditor.F.stop_tracking = stop_tracking
    exec(compile(text, '<B119_ADDENDUM2_history_policy>', 'exec'), auditor.__dict__)


def same_tree(a, b, path='state'):
    import torch
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()), path
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a,b), path
    elif isinstance(a, dict):
        assert isinstance(b,dict) and set(a)==set(b), path
        for k in a: same_tree(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b), path
        for i,(x,y) in enumerate(zip(a,b)): same_tree(x,y,path+'/'+str(i))
    else:
        assert a==b, path
