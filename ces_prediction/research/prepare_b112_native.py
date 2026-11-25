"""Prepare native upload delivery, retaining the exact B112 payload."""
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b112_residual'))
from common112 import *
import prepare112
def main():
    guard();old=read(OUT/'export_verification.json');assert old['all_pass']
    archive=OUT/'ces_b112_bundle.zip';assert sha(archive)==old['archive_sha256']
    backup=OUT/'ces_b112_widget_bundle.zip';assert not backup.exists();archive.rename(backup)
    write(OUT/'export_verification_before_native.json',old)
    m=read(OUT/'bundle/manifest.json')
    for name,h in m['files'].items():assert sha(OUT/'bundle'/name)==h
    write(OUT/'manifest_before_native.json',m);m['sources']=sources();write(OUT/'bundle/manifest.json',m)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((OUT/'bundle').iterdir()):z.write(p,p.name)
    old.update(archive_sha256=sha(archive),bytes=archive.stat().st_size,sources=sources(),delivery='native_Files_UI');write(OUT/'export_verification.json',old)
    prepare112.main()
    p=OUT/'COLAB_B112.ipynb';n=read(p);code=''.join(n['cells'][0]['source'])
    before="uploaded=files.upload()\nassert set(uploaded)=={'ces_b112_bundle.zip'}"
    assert code.count(before)==1
    code=code.replace(before,"print('Using the completed native Files upload.',flush=True)")
    helper=sha(__file__)
    code=code.replace("receipt={**info,", "receipt={'delivery':'native_Files_UI','native_helper_sha256':'"+helper+"',**info,")
    n['cells'][0]['source']=code.splitlines(True);write(p,n)
    (OUT/'COLAB_B112_NATIVE_CELL.py').write_text(code,encoding='utf8')
    a=read(OUT/'standalone_attempt.json');a.update(notebook_sha256=sha(p),cell_sha256=hashlib.sha256(code.encode()).hexdigest(),delivery='native_Files_UI',native_helper_sha256=helper);write(OUT/'standalone_attempt.json',a)
    print(json.dumps({'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'attempt':a['attempt']}))
if __name__=='__main__':main()
