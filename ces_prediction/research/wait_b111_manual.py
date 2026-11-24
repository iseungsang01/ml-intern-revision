"""Read-only prelaunch attachment, followed by the existing results observer."""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
from colab_t4_operator import call,source,new_cell
ROOT=Path(__file__).resolve().parents[2];CON=ROOT/'data/.colab_connection';STATE=CON/'b111_remote.json'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b93_gpu'))
from atomic93 import save_json

def main():
    attempt=json.loads((CON/'b111_manual_attempt.json').read_text());start=time.time();failures=0
    while time.time()-start<14400:
        state=json.loads(STATE.read_text());assert 'launch' not in state and 'poll_cell' not in state,'Existing attachment; do not duplicate observer'
        try:cc=call('get_cells',{'includeOutputs':True})['cells'];failures=0
        except Exception as error:
            failures+=1;save_json(CON/'b111_attach_status.json',{'state':'connector_error','error':repr(error),'updated':time.time()})
            if failures>=3:raise
            time.sleep(15);continue
        matching=[c for c in cc if c['id']==attempt['cell']];assert len(matching)==1,'Owned bootstrap cell missing; preserve state'
        cell=matching[0];assert hashlib.sha256(source(cell).encode()).hexdigest()==attempt['source_sha256'],'Bootstrap source changed; inspect'
        text=''.join(''.join(o.get('text',[])) if isinstance(o.get('text'),list) else o.get('text','') for o in cell.get('outputs',[]) if o.get('output_type')=='stream')
        found=[line[len('B111_LAUNCHED='):] for line in text.splitlines() if line.startswith('B111_LAUNCHED=')]
        save_json(CON/'b111_attach_status.json',{'state':'launch_acknowledged' if found else 'waiting_for_owned_cell','updated':time.time(),'cell':attempt['cell']})
        if not found:
            errors=[o.get('ename') for o in cell.get('outputs',[]) if o.get('output_type')=='error']
            if errors:save_json(CON/'b111_manual_cell_error.json',{'errors':errors,'updated':time.time()})
            time.sleep(10);continue
        assert len(found)==1;launch=json.loads(found[0]);assert launch['attempt']==attempt['attempt'] and launch['root']==attempt['root'] and launch['archive_verified'] and launch['archive_sha256']==attempt['archive_sha256'] and isinstance(launch['pid'],int) and launch['pid']>0 and launch['gpu']=='Tesla T4'
        backup=CON/'b111_before_manual_attach.json';assert not backup.exists(),'Prior attachment attempt; inspect without repeating';save_json(backup,state)
        root=attempt['root'];poll=f"from pathlib import Path\nimport json\nroot=Path({root!r})\np=root/'bundle/progress.json'\ns=json.loads(p.read_text()) if p.exists() else {{'state':'starting'}}\ne=root/'bundle/process_exit.json'\ns['exit']=json.loads(e.read_text()) if e.exists() else None\ns['zip_ready']=(root/'bundle/b111_results.zip').exists()\ns['log_tail']=(root/'run.log').read_text(errors='replace')[-1600:]\nprint(json.dumps(s))"
        poll_cell=new_cell(poll);state.update(status='running',root=root,launch=launch,poll_cell=poll_cell,delivery='Colab files.upload owned one-cell bootstrap',uploaded=149990972,manual_attempt=attempt['attempt'],attachment_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        save_json(STATE,state);save_json(CON/'b111_attach_status.json',{'state':'observer_attached','updated':time.time(),'pid':launch['pid']})
        result=subprocess.run([sys.executable,'-u',str(ROOT/'ces_prediction/research/watch_colab_b111.py')],cwd=ROOT)
        assert result.returncode==0,'Observer/audit failed; no scientific retry';return
    raise RuntimeError('Attachment lifetime reached; no scientific process launched by this observer')

if __name__=='__main__':
    try:main()
    except BaseException:
        (CON/'b111_attach_failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise
