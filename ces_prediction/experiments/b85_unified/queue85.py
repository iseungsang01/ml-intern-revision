"""Supervise clean real-data smoke exits before launching the registered batch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CACHE=ROOT/'data/.b85'


def status(state,**details):
    path=CACHE/'queue_status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'state':state,'pid':os.getpid(),'updated':time.strftime('%Y-%m-%dT%H:%M:%S'),**details},indent=2),encoding='utf8')
    tmp.replace(path)


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    lock=CACHE/'queue.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Queue lock exists; inspect the recorded process before resuming')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
    try:
        for phase,args in [('smoke',['--smoke']),('registered_batch',[])]:
            status(phase,note='Runner may be waiting for free GPU memory; see b85_smoke/status reports')
            print('Starting '+phase,flush=True)
            proc=subprocess.run([sys.executable,'-u',str(HERE/'run85.py'),*args],cwd=ROOT,env=env)
            if proc.returncode:
                status('failed',phase=phase,returncode=proc.returncode)
                raise RuntimeError(f'{phase} exited {proc.returncode}; no blind retry')
        status('complete')
    finally:
        if lock.is_file() and lock.read_text()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
