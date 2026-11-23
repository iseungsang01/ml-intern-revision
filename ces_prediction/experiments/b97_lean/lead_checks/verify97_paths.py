"""B97 addendum1: path-key comparison adapter; original verifier untouched."""
import json
from pathlib import Path
import verify97 as V

def main():
    original=V.reload_inputs
    def load():
        grid,frames,selected,hashes=original()
        return grid,frames,selected,{str(Path(k)):h for k,h in hashes.items()}
    V.reload_inputs=load
    V.main()
    path=V.R.CACHE/'independent.json'
    record=json.loads(path.read_text())
    record['path_adapter']={'registration':'PREREGISTRATION_B97_ADDENDUM1.md',
                            'source_sha256':V.R.sha(Path(__file__)),
                            'preserved_failure_sha256':V.R.sha(V.R.CACHE/'independent_first_failure.json'),
                            'effect':'Relative hash-dictionary key separators only; original verifier and producer unchanged'}
    path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    z=json.loads((V.R.CACHE/'result.json').read_text())
    (V.R.ROOT/'docs/paper/b97_lean.json').write_text(json.dumps({**z,'independent':record},indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':record['status'],'path_normalization_disclosed':True}))

if __name__=='__main__':main()
