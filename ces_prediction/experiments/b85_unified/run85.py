# Claude authored the implementation
# ces_prediction/experiments/b85_unified/run85.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import subprocess
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'ces_prediction' / 'experiments' / 'b85_unified'))

from common85 import (
    CACHE, PAPER, DATA, INITS, VARIANTS,
    select_units
)

ENV_PINS = {
    'CES_DATA_DIR': str(DATA),
    'CES_DROP_STUCK_TARGETS': '1',
    'CES_TI_SPIKE_CUT_EV': '3000',
    'CES_PER_SHOT_NORM': '0',
    'CES_SEQ_NORM_EXPANDING': '0',
    'CES_SEQ_ZERO_MC': '1',
    'CES_SEQ_ZERO_VT_HISTORY': '1',
    'CES_SEQ_ZERO_VT': '1',
    'CES_SEQ_EPOCHS': '80',
    'CES_SEQ_BATCH': '16',
    'CES_LR': '0.001',
    'CES_SEQ_LOSS': 'mse',
    'CES_SEQ_SELECT': 'mse',
    'CES_SEQ_DEVICE': 'cuda',
    'CES_SEQ_FIXED_EPOCHS': '0',
    'CES_SEQ_TRAIN_CTX': '0',
    'CES_SEQ_EVAL_CTX': '0',
    'CUDA_VISIBLE_DEVICES': '0',
}

LOGS_DIR = CACHE / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def all_runs():
    """Generate all 90 run configs."""
    units = select_units()
    runs = []
    for u in units:
        for init in INITS:
            for var in VARIANTS:
                runs.append({'unit': u['name'], 'init': init, 'variant': var})
    return runs


def is_complete(unit_name, init, variant, smoke=False):
    base = CACHE / ('smoke' if smoke else 'runs')
    p = base / f'{unit_name}__{init}__{variant}' / 'complete.json'
    if not p.exists():
        return False
    from train85 import verify_complete
    return verify_complete(p.parent)


def check_vram(min_mb=2500, timeout=30, poll_interval=5):
    """Wait for free VRAM >= min_mb. Returns True if available."""
    for _ in range(timeout // poll_interval + 1):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                free = int(result.stdout.strip().split('\n')[0])
                if free >= min_mb:
                    return True
                print(f'Free VRAM: {free} MiB < {min_mb} MiB, waiting...')
        except Exception as e:
            print(f'nvidia-smi error: {e}')
        time.sleep(poll_interval)
    return False


def check_input_checks():
    p = CACHE / 'input_checks.json'
    if not p.exists():
        return False
    with open(p) as f:
        data = json.load(f)
    return data.get('all_pass', False)


def write_status(runs_done, runs_total, failures, smoke=False, state='running', note=''):
    suffix = 'smoke' if smoke else 'status'
    md_path = PAPER / f'b85_{suffix}.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# B85 {"Smoke" if smoke else "Status"}\n\n')
        f.write(f'Completed: {runs_done}/{runs_total}\n')
        f.write(f'Failures: {len(failures)}\n\n')
        f.write(f'State: {state}\n\n{note}\n\n')
        for fail in failures:
            f.write(f'- {fail}\n')


def run_plan(unit_filter=None, init_filter=None, variant_filter=None):
    """Write plan manifest only, no GPU."""
    runs = all_runs()
    if unit_filter:
        runs = [r for r in runs if r['unit'] == unit_filter]
    if init_filter:
        runs = [r for r in runs if r['init'] == init_filter]
    if variant_filter:
        runs = [r for r in runs if r['variant'] == variant_filter]

    plan = {
        'runs': runs,
        'count': len(runs),
        'env': ENV_PINS,
    }
    plan_path = CACHE / 'plan.json'
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f'Plan: {len(runs)} runs -> {plan_path}')
    return runs


def run_status(unit_filter=None, init_filter=None, variant_filter=None):
    runs = all_runs()
    if unit_filter:
        runs = [r for r in runs if r['unit'] == unit_filter]
    if init_filter:
        runs = [r for r in runs if r['init'] == init_filter]
    if variant_filter:
        runs = [r for r in runs if r['variant'] == variant_filter]

    done = sum(1 for r in runs if is_complete(r['unit'], r['init'], r['variant']))
    print(f'Status: {done}/{len(runs)} complete')
    write_status(done, len(runs), [])
    return done, len(runs)


def run_batch(unit_filter=None, init_filter=None, variant_filter=None, smoke=False):
    """Run all specified training jobs sequentially."""
    # Batch lock
    lock_path = CACHE / '.batch_lock'
    pid = os.getpid()
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(pid).encode())
        os.close(fd)
    except FileExistsError:
        try:
            with open(lock_path) as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, 0)
                print(f'Batch already running (PID {old_pid})')
                sys.exit(1)
            except (OSError, ProcessLookupError):
                lock_path.unlink(missing_ok=True)
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(pid).encode())
                os.close(fd)
        except Exception:
            print('Cannot acquire batch lock')
            sys.exit(1)

    try:
        # Input checks
        if not check_input_checks():
            print('Running input checks...')
            result = subprocess.run(
                [sys.executable, str(ROOT / 'ces_prediction' / 'experiments' / 'b85_unified' / 'checks85.py')],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f'Input checks failed:\n{result.stderr}')
                sys.exit(1)
            if not check_input_checks():
                print('Input checks did not pass')
                sys.exit(1)

        runs = all_runs()
        if unit_filter:
            runs = [r for r in runs if r['unit'] == unit_filter]
        if init_filter:
            runs = [r for r in runs if r['init'] == init_filter]
        if variant_filter:
            runs = [r for r in runs if r['variant'] == variant_filter]

        if smoke:
            # First unit, both variants, first init
            units = select_units()
            first = units[0]['name']
            runs = [r for r in runs if r['unit'] == first and r['init'] == INITS[0]]

        failures = []
        done = 0

        for i, r in enumerate(runs):
            name = f"{r['unit']}__{r['init']}__{r['variant']}"

            if is_complete(r['unit'], r['init'], r['variant'], smoke):
                print(f'[{i+1}/{len(runs)}] Skip (complete): {name}')
                done += 1
                continue

            # Check VRAM
            while not check_vram():
                print('VRAM unavailable; waiting without stopping other processes', flush=True)
                write_status(done, len(runs), failures, smoke, 'waiting_for_gpu', 'Preserving other GPU processes; waiting for at least 2500 MiB free.')

            print(f'[{i+1}/{len(runs)}] Training: {name}')
            write_status(done, len(runs), failures, smoke, 'running', 'Active run: ' + name)
            log_file = LOGS_DIR / f'{name}.log'

            cmd = [
                sys.executable,
                '-X', 'faulthandler',
                str(ROOT / 'ces_prediction' / 'experiments' / 'b85_unified' / 'train85.py'),
                '--unit', r['unit'],
                '--init', str(r['init']),
                '--variant', r['variant'],
            ]
            if smoke:
                cmd.append('--smoke')

            env = {k: v for k, v in os.environ.items() if not k.startswith('CES_')}
            env.update(ENV_PINS)
            env['PYTHONUNBUFFERED'] = '1'

            with open(log_file, 'w') as lf:
                result = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            exit_record = LOGS_DIR / (name + ('__smoke' if smoke else '') + '.exit.json')
            run_dir = CACHE / ('smoke' if smoke else 'runs') / name
            complete_path = run_dir / 'complete.json'
            complete = json.loads(complete_path.read_text(encoding='utf8')) if complete_path.is_file() else {}
            attestation = {'returncode':result.returncode,'smoke':smoke,'run':name,
                           'config_hash':complete.get('config_hash'),
                           'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S')}
            exit_record.write_text(json.dumps(attestation),encoding='utf8')
            if run_dir.is_dir():
                tmp_exit = run_dir / 'process_exit.tmp'
                tmp_exit.write_text(json.dumps(attestation,indent=2),encoding='utf8')
                tmp_exit.replace(run_dir / 'process_exit.json')

            if result.returncode != 0:
                print(f'FAILED: {name}, child returncode={result.returncode} (see {log_file})')
                failures.append(name)
                break  # Stop on failure
            else:
                if not is_complete(r['unit'], r['init'], r['variant'], smoke):
                    raise RuntimeError('Child exited zero without verified completion: ' + name)
                done += 1

            write_status(done, len(runs), failures, smoke)

        write_status(done, len(runs), failures, smoke, 'failed' if failures else 'complete' if done==len(runs) else 'running')

        if failures:
            raise RuntimeError('Batch stopped after a failed run: ' + ', '.join(failures))
        if not smoke and not failures and all(is_complete(r['unit'], r['init'], r['variant']) for r in all_runs()):
            print('All runs complete, running predict and readout...')
            subprocess.run([
                sys.executable,
                str(ROOT / 'ces_prediction' / 'experiments' / 'b85_unified' / 'predict85.py')
            ], check=True)
            subprocess.run([
                sys.executable,
                str(ROOT / 'ces_prediction' / 'experiments' / 'b85_unified' / 'readout85.py')
            ], check=True)

    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='B85 Runner')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--unit', default=None)
    parser.add_argument('--init', type=int, default=None)
    parser.add_argument('--variant', default=None)
    args = parser.parse_args()

    if args.plan:
        run_plan(args.unit, args.init, args.variant)
    elif args.status:
        run_status(args.unit, args.init, args.variant)
    elif args.smoke:
        run_batch(args.unit, args.init, args.variant, smoke=True)
    else:
        run_batch(args.unit, args.init, args.variant)
