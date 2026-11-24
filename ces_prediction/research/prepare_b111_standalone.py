"""Make the registered single-cell notebook independent of the MCP proxy."""
import hashlib,json,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/colab_b111';CON=ROOT/'data/.colab_connection'

def main():
    source=OUT/'COLAB_B111_CELL.py';original=source.read_text(encoding='utf8');assert 'files.upload()' in original
    backup=OUT/'COLAB_B111_CELL_before_standalone.py';assert not backup.exists(),'Preserve prior standalone preparation';backup.write_text(original,encoding='utf8')
    token=re.search(r'B111_MANUAL_OWNED_([a-f0-9]{32})',original).group(1);archive=OUT/'ces_b111_bundle.zip';expected=hashlib.sha256(archive.read_bytes()).hexdigest()
    v=json.loads((OUT/'export_verification.json').read_text());assert expected==v['archive_sha256']
    helper_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();ad_hash=hashlib.sha256((ROOT/'ces_prediction/experiments/PREREGISTRATION_B111_ADDENDUM5.md').read_bytes()).hexdigest()
    tail=f'''
# Follow the real child; no synthetic keepalive and no fit retries.
last_progress=None
while proc.poll() is None:
    status_path=work/'progress.json'
    if status_path.exists():
        current=json.loads(status_path.read_text())
        key=(current.get('state'),current.get('completed'),current.get('completed_cells'))
        if key!=last_progress:
            print(json.dumps(current),flush=True)
            last_progress=key
    time.sleep(10)
assert proc.returncode==0, 'Scientific process failed. Preserve run.log/processes; do not re-run this notebook.'
result=work/'b111_results.zip'
assert result.exists()
receipt={{**info,'returncode':proc.returncode,'manifest_sha256':hashlib.sha256((work/'manifest.json').read_bytes()).hexdigest(),'standalone_helper_sha256':{helper_hash!r},'operational_addendum_sha256':{ad_hash!r}}}
with zipfile.ZipFile(result,'a',zipfile.ZIP_DEFLATED) as z:
    assert 'standalone_delivery.json' not in z.namelist()
    z.writestr('standalone_delivery.json',json.dumps(receipt,indent=2))
print('B111_DOWNLOAD_READY bytes='+str(result.stat().st_size)+' sha256='+hashlib.sha256(result.read_bytes()).hexdigest(),flush=True)
files.download(str(result))
'''
    code=original+tail;compile(code,'COLAB_B111_CELL.py','exec')
    source.write_text(code,encoding='utf8');nb={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','colab':{'gpuType':'T4'},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(True)}]}
    notebook=OUT/'COLAB_B111.ipynb';notebook.write_text(json.dumps(nb,indent=2),encoding='utf8')
    manifest_hash=hashlib.sha256((OUT/'bundle/manifest.json').read_bytes()).hexdigest()
    meta={'attempt':token,'prepared':time.time(),'archive_sha256':expected,'manifest_sha256':manifest_hash,'notebook_sha256':hashlib.sha256(notebook.read_bytes()).hexdigest(),'cell_sha256':hashlib.sha256(code.encode()).hexdigest(),'expected_download':str(Path.home()/'Downloads/b111_results.zip'),'helper_sha256':helper_hash,'addendum_sha256':ad_hash,'scientific_process_started':False}
    (CON/'b111_standalone_attempt.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print('Standalone B111 notebook prepared; no inference or fit started.',flush=True)
if __name__=='__main__':main()
