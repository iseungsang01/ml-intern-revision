"""Receive only the registered B112 download and run independent CPU audit."""
import os,shutil,subprocess,sys,time,traceback,zipfile
from common112 import *
CON=ROOT/'data/.colab_connection'
def notify(message):
    exe=shutil.which('cvmux')
    if exe:
        result=subprocess.run([exe,'notify',message],capture_output=True)
        write(OUT/'notification.json',{'message':message,'returncode':result.returncode,'updated':time.time()})
def main():
    a=read(OUT/'standalone_attempt.json');expected=Path(a['expected_download']);started=time.time()
    assert not (OUT/'b112_results.zip').exists(),'Existing result; inspect instead of importing twice'
    while time.time()-started<43200:
        write(OUT/'watch_status.json',{'state':'waiting_download','pid':os.getpid(),'updated':time.time()})
        if not expected.exists() or expected.stat().st_mtime<a['prepared']:
            time.sleep(10);continue
        with zipfile.ZipFile(expected) as z:
            assert z.testzip() is None
            info=z.getinfo('standalone_delivery.json');assert info.file_size<65536
            receipt=json.loads(z.read(info))
        assert receipt['attempt']==a['attempt'] and receipt['archive_sha256']==a['archive_sha256']
        assert receipt['manifest_sha256']==a['manifest_sha256'] and receipt['archive_verified']
        assert receipt['gpu']=='Tesla T4' and receipt['returncode']==0
        target=OUT/'b112_results.zip';shutil.copyfile(expected,target);h=sha(target);assert h==sha(expected)
        write(CON/'b112_remote.json',{'downloaded':True,'results_sha256':h,'launch':receipt,'updated':time.time()})
        write(OUT/'watch_status.json',{'state':'independent_audit','updated':time.time()})
        with (OUT/'audit.stdout.log').open('a',encoding='utf8') as out,(OUT/'audit.stderr.log').open('a',encoding='utf8') as err:
            result=subprocess.run([sys.executable,str(HERE/'audit112.py')],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'},stdout=out,stderr=err)
        assert result.returncode==0,'Independent audit failed; no fit retry'
        write(OUT/'watch_status.json',{'state':'complete','updated':time.time(),'results_sha256':h})
        notify('B112 Colab fits and all scenario comparisons independently audited.');return
    raise RuntimeError('Observer expired; no fits restarted')
if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'watch_status.json',{'state':'failed','updated':time.time(),'traceback':traceback.format_exc()})
        notify('B112 download/audit stopped; preserved evidence needs inspection.');raise
