"""Atomic JSON publication with bounded transient Windows permission handling.

Only an artifact's replacement is retried. No model, optimizer, process launch or
scientific trajectory is retried. Failed temporary files remain as evidence.
"""
import json,os,tempfile,threading,time
from pathlib import Path

def save_json(path,obj,*,timeout_seconds=10.0):
    if not 0<=timeout_seconds<=10:raise ValueError('Artifact replacement retry budget must be within 0..10 seconds')
    path=Path(path);payload=json.dumps(obj,indent=2,sort_keys=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    prefix=f'.{path.name}.{os.getpid()}.{threading.get_ident()}.'
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf8',dir=path.parent,prefix=prefix,suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);stream.write(payload);stream.flush();os.fsync(stream.fileno())
        deadline=time.monotonic()+timeout_seconds;delay=.05;attempts=0
        while True:
            attempts+=1
            try:os.replace(temporary,path);return
            except PermissionError as error:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    error.add_note(f'Atomic JSON replacement exhausted {timeout_seconds:g}s after {attempts} attempts; preserved temporary file: {temporary}')
                    raise
                time.sleep(min(delay,remaining));delay=min(.25,delay*2)
    except BaseException as error:
        if temporary is not None:
            error.add_note(f'Unpublished JSON temporary path (not removed on failure): {temporary}')
        raise
