"""Narrow monitor-only adapter; frozen durable/numeric sources are untouched."""
import contextlib,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
CACHE=ROOT/'data/.b99'
sys.path.insert(0,str(HERE.parent/'b93_gpu'))
import durable93 as D
from atomic93 import save_json

def note(kind,**details):
    record={'kind':kind,'time':time.time(),**details}
    try:print(json.dumps(record),file=sys.stderr,flush=True)
    except OSError:pass
    try:save_json(CACHE/'monitor_fallback_status.json',record,timeout_seconds=1)
    except OSError:pass

class MonitorPath(type(Path())):
    read_budget=10.0
    def read_text(self,*args,**kwargs):
        if self.name!='queue_status.json':return super().read_text(*args,**kwargs)
        start=time.monotonic()
        while True:
            try:return super().read_text(*args,**kwargs)
            except PermissionError as error:
                if time.monotonic()-start>=self.read_budget:
                    note('phase_read_unavailable',path=str(self),error=repr(error),elapsed=time.monotonic()-start)
                    return json.dumps({'state':'metadata_unavailable','monitor_note':'Bounded read permission retries exhausted; no scientific process stopped'})
                time.sleep(.05)

def save(path,obj):
    try:return save_json(path,obj)
    except PermissionError as error:
        note('heartbeat_write_unavailable',path=str(path),error=repr(error),attempted_status=obj)
        return None

def install():
    original=D.safe_path
    D.safe_path=lambda value:MonitorPath(original(value))
    D.save=save

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    with (CACHE/'monitor99.stdout.log').open('a',encoding='utf8') as out,(CACHE/'monitor99.stderr.log').open('a',encoding='utf8') as err:
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:
                install();return D.main()
            except BaseException:
                traceback.print_exc();raise

if __name__=='__main__':raise SystemExit(main())
