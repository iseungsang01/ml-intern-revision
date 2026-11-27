"""Notify the author's current cvmux session when B119 optimization stops.

Read-only with respect to scientific artifacts. This does not wait for analysis
or claim convergence; workflow failures and author pauses are also reported.
"""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/.b119'
STATE = OUT / 'notification'
FIT = OUT / 'fits/U0_anchor_lstm_s1701'
APP = Path(os.environ['LOCALAPPDATA']) / 'Programs/cvmux'
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(name, value):
    path = STATE / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf8')
    temporary.replace(path)


def notify(message):
    assert os.environ.get('CVMUX') == '1'
    assert os.environ.get('CVMUX_SESSION_ID'), 'Original session identity required'
    env = {**os.environ, 'ELECTRON_RUN_AS_NODE': '1'}
    result = subprocess.run([str(APP / 'cvmux.exe'), str(APP / 'resources/cli/cvmux.mjs'),
        '--json', 'notify', message, '--title', 'B119 LSTM',
        '--session', os.environ['CVMUX_SESSION_ID']],
        capture_output=True, text=True, encoding='utf8', env=env,
        timeout=30, creationflags=FLAGS)
    assert result.returncode == 0, 'cvmux notification failed: ' + result.stderr[:500]
    return result.stdout.strip()


def process_alive(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        okay = kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        return bool(okay) and code.value == 259
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def stopped():
    params = FIT / 'params.json'
    if params.exists():
        p = read(params)
        reason = ('검증 정체 종료 기준 충족' if p['converged']
                  else '안전 상한 도달, 미수렴')
        return {'kind': 'training_stopped', 'epochs': p['completed_epochs'],
                'reason': p['stop_reason'], 'converged': p['converged'],
                'message': f"LSTM 학습 종료: {p['completed_epochs']} epoch. {reason}. 결과 분석·검산은 별도로 진행됩니다."}
    status = read(OUT / 'workflow/status.json')
    if status['state'] in ('failed', 'paused_by_author', 'paused', 'complete'):
        return {'kind': status['state'],
                'message': 'LSTM 학습이 멈췄습니다. 상태: ' + status['state'] + '. 종료 기록을 확인해 주세요.'}
    started = read(OUT / 'workflow/started.json')
    if time.time() - status.get('updated', 0) > 60 and not process_alive(started['pid']):
        return {'kind': 'workflow_process_exited',
                'message': 'LSTM 실행 프로세스가 종료됐습니다. 정상 종료 여부는 아직 확인되지 않았습니다.'}
    if time.time() > started['CPU_end'] + 900:
        return {'kind': 'watch_boundary',
                'message': 'LSTM 감시 기한에 도달했습니다. 학습 종료 상태를 확인해 주세요.'}
    return None


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'started.json').open('x', encoding='utf8') as stream:
        json.dump({'pid': os.getpid(), 'started': time.time(),
                   'notify_on': 'optimization stops, not audit completion',
                   'session': os.environ.get('CVMUX_SESSION_ID')}, stream)
    acknowledgment = notify('LSTM 학습 종료 알림을 설정했습니다. 수렴 여부와 무관하게 학습이 멈추면 알려드립니다.')
    save('ready.json', {'ready': True, 'pid': os.getpid(), 'updated': time.time(),
                        'start_notification': acknowledgment})
    while True:
        event = stopped()
        if event:
            save('detected.json', {**event, 'detected': time.time()})
            # Retry delivery only; never restart or modify the scientific workflow.
            for attempt in range(10):
                try:
                    response = notify(event['message'])
                    save('delivered.json', {**event, 'delivered': time.time(), 'response': response})
                    return
                except Exception as error:
                    save('delivery_error.json', {'attempt': attempt + 1, 'error': str(error), 'updated': time.time()})
                    time.sleep(30)
            raise RuntimeError('Notification delivery failed after bounded retries')
        save('heartbeat.json', {'pid': os.getpid(), 'updated': time.time(), 'state': 'watching'})
        time.sleep(15)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        STATE.mkdir(parents=True, exist_ok=True)
        save('failed.json', {'error': traceback.format_exc(), 'updated': time.time()})
        raise
