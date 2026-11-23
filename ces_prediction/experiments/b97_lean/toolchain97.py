"""Local pinned official Windows Lean archive, outside tracked sources."""
import hashlib,json,urllib.request,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'data/.lean_toolchain'
NAME='lean-4.36.0-windows'
DIGEST='5af5f955df6cfe37082d71c56c7d47fe620faa9f59e14b917b43976c08dc676b'
URL=f'https://github.com/leanprover/lean4/releases/download/v4.36.0/{NAME}.zip'

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    archive=BASE/(NAME+'.zip')
    if not archive.exists():urllib.request.urlretrieve(URL,archive)
    actual=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert actual==DIGEST,'Official archive SHA-256 mismatch; no extraction'
    binary=BASE/NAME/'bin/lean.exe'
    if not binary.exists():
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                assert (BASE/member.filename).resolve().is_relative_to(BASE.resolve())
            z.extractall(BASE)
    assert binary.is_file()
    print(json.dumps({'archive_sha256':actual,'source':URL,'lean':str(binary.relative_to(ROOT))}))

if __name__=='__main__':main()
