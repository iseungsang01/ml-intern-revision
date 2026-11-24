"""Watch one expected browser download, bind its receipt, run CPU audit only."""
import hashlib,json,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CON=ROOT/'data/.colab_connection';OUT=ROOT/'data/colab_b111'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b93_gpu'));from atomic93 import save_json
import durable93 as D
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b105_merit'))
from ui105 import unprotect

def note(message):
    config=json.loads((ROOT/'data/.b106/watch_config.json').read_text());os.environ.update(unprotect(config['ui_protected']))
    result=D.powershell('& '+D.quote_ps(config['notify_bin'])+' notify '+D.quote_ps(message))
    save_json(CON/'b111_download_notification.json',{'message':message,'returncode':result.returncode,'updated':time.time()})

def main():
    a=json.loads((CON/'b111_standalone_attempt.json').read_text());expected=Path(a['expected_download']);start=time.time()
    assert not (OUT/'b111_results.zip').exists(),'Existing results; inspect rather than overwrite'
    note('B111 standalone result observer ready; no scientific process is started by this observer.')
    while time.time()-start<14400:
        save_json(CON/'b111_download_watch.json',{'state':'waiting_for_registered_download','pid':os.getpid(),'updated':time.time(),'expected':str(expected)})
        if not expected.exists() or expected.stat().st_mtime<a['prepared']:time.sleep(10);continue
        with zipfile.ZipFile(expected) as z:
            assert z.testzip() is None;info=z.getinfo('standalone_delivery.json');assert info.file_size<65536
            receipt=json.loads(z.read(info))
        assert receipt['attempt']==a['attempt'] and receipt['archive_sha256']==a['archive_sha256'] and receipt['manifest_sha256']==a['manifest_sha256'] and receipt['gpu']=='Tesla T4' and receipt['returncode']==0 and receipt['archive_verified']
        assert receipt['standalone_helper_sha256']==a['helper_sha256'] and receipt['operational_addendum_sha256']==a['addendum_sha256']
        dest=OUT/'b111_results.zip';shutil.copyfile(expected,dest);h=hashlib.sha256(dest.read_bytes()).hexdigest();assert h==hashlib.sha256(expected.read_bytes()).hexdigest()
        state=json.loads((CON/'b111_remote.json').read_text());save_json(CON/'b111_before_download_import.json',state)
        state.update(status='results_downloaded',downloaded=True,results_sha256=h,delivery='standalone Colab notebook/browser download',launch=receipt);save_json(CON/'b111_remote.json',state)
        save_json(CON/'b111_download_watch.json',{'state':'independent_audit','updated':time.time(),'results_sha256':h})
        note('B111 registered result download received; independent CPU audit started.')
        r=subprocess.run([sys.executable,str(ROOT/'ces_prediction/experiments/b111_full_compare/audit111.py')],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
        assert r.returncode==0,'Independent audit failed; preserve evidence, no fit retry'
        save_json(CON/'b111_download_watch.json',{'state':'complete','updated':time.time(),'results_sha256':h});note('B111 complete: eight correction fits and all32 scenario cells independently audited.');return
    raise RuntimeError('Expected download not received within observer lifetime; no scientific work launched here')

if __name__=='__main__':
    try:
        with (CON/'b111_download_watch.stdout.log').open('a',encoding='utf8',buffering=1) as out,(CON/'b111_download_watch.stderr.log').open('a',encoding='utf8',buffering=1) as err:
            from contextlib import redirect_stdout,redirect_stderr
            with redirect_stdout(out),redirect_stderr(err):main()
    except BaseException:
        (CON/'b111_download_watch_failure.txt').write_text(traceback.format_exc(),encoding='utf8');note('B111 result observer/audit stopped; inspect preserved evidence. No fits restarted.');raise
