"""Read-only recovery gates; source/attempt/exit/deadline identity checks."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import recover99 as R

def main():
    z=json.loads((R.M.CACHE/'archive99.json').read_text());checks={}
    checks['frozen_unchanged']=R.G.guard()==z['frozen']
    checks['new_sources_bound']=R.sources()==z['sources99']
    checks['clock_unchanged']=R.G.clock()['deadline']==z['deadline']
    checks['metadata_probe_pass']=json.loads((R.M.CACHE/'probe99.json').read_text())['all_pass']
    checks['verified_completes_unchanged']=R.verify_existing()==z['completed']
    for run,files in z['archived'].items():
        rd=R.G.CACHE/'interrupted_permission99'/run
        actual={str(p.relative_to(rd)):R.G.sha(p) for p in rd.rglob('*') if p.is_file()}
        checks['archive_'+run]=actual==files and not (R.G.CACHE/'confirm'/run).exists()
    checks['old_processes_absent']=all(R.M.D.process_identity(a['pid'])['alive'] is not True for a in z['original_status']['queue']['active'])
    assert all(checks.values()),checks
    result={'all_pass':True,'checks':checks,'archive_sha256':R.G.sha(R.M.CACHE/'archive99.json')}
    R.G.save(R.M.CACHE/'lead_checks99.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
