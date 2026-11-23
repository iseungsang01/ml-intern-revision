"""Fail-closed original-audit gate and whitelist-only anonymous evidence."""
import hashlib,json,statistics,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CACHE=ROOT/'data/.claude_posteval_20261123';UP=ROOT/'data/.b94'
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b85_unified'))
import claude_review85 as API

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    deadline=time.monotonic()+2
    while True:
        try:return json.loads(Path(p).read_text(encoding='utf8'))
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.05)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b93_gpu'))
    from atomic93 import save_json
    save_json(p,x)

def gate():
    sup=read(UP/'supervisor_status.json');queue=read(UP/'queue_status.json')
    if sup['state'] in ['failed','interrupted','deadline_exceeded'] or queue['state']=='failed':
        raise RuntimeError('Upstream evaluation stopped; no partial-result research review')
    if sup['state']!='complete' or queue['state']!='complete':
        sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b93_gpu'))
        from status93 import process_identity
        for record,pidkey,birthkey in [(sup,'pid','birth'),(sup,'queue_pid','queue_birth')]:
            if record.get(pidkey):
                identity=process_identity(record[pidkey])
                if identity['alive'] is False or identity['alive'] is True and record.get(birthkey) and identity['birth']!=record[birthkey]:
                    raise RuntimeError('Upstream process interrupted; original final audit not complete')
        return None
    assert sup['returncode']==0
    audit=read(UP/'lead_checks/post.json');result=read(UP/'readout.json');selection=read(UP/'selection.json')
    assert audit['passed'] and audit['phase']=='post' and audit['confirmation_fits']==46
    assert audit['readout_sha256']==sha(UP/'readout.json') and audit['selection_sha256']==sha(UP/'selection.json')
    assert result==read(UP/'reports/b94_methods.json')==read(ROOT/'docs/paper/b94_methods.json')
    assert result['selection']==selection and result['registration']=='PREREGISTRATION_B94.md'
    assert set(result['scopes'])=={'FULL15_one_seed','FOUR_units_three_seeds'}
    return result,{'post_audit_sha256':sha(UP/'lead_checks/post.json'),'readout_sha256':sha(UP/'readout.json'),
                   'selection_sha256':sha(UP/'selection.json'),'post_cells':audit['cells_independently_recomputed']}

def interval(z):
    return {k:z[k] for k in ['point','cluster','per_init','excludes_zero'] if k in z}

def packet(result,binding):
    evidence={'context':{
        'purpose':'Pedestal-top chord causal CES report nowcasting, best useful estimator vs hold/GP/interpolation, 10ms compute; preserve all five paper questions.',
        'privacy':'Only anonymous aggregate evidence; no personal identity, shots, raw observations, absolute paths or secrets.',
        'models':'Selected architecture from validation; for TCN TX=fast-enabled TCN, TH=history-only TCN. These TX labels are NOT the manuscript GP/tree estimator.',
        'selected_architecture':result['selection']['chosen'],
        'original_evaluation_gate':{'queue_complete':True,'supervisor_exit0':True,'post_audit_passed':True,'selection_readout_hashes_equal':True},
        'limits':['F/L5 are report-derived targets, not independent bulk-ion truth. L5 is a future median, not an instantaneous temperature.',
                  'Timestamp-causal inputs do not establish instrument arrival timing. No channel-specific pedestal mechanism established.',
                  'Used campaign; TEST-informed future proposals are exploratory. Reused TRAIN/VAL selection and finite budgets; no fitting/selection/multiplicity uncertainty.',
                  'FULL15 one-seed and fixed FOUR-unit three-seed scopes cannot be pooled as global initialization stability.',
                  'Supplementary baseline row supports/training budgets may differ. Missing baseline is not evidence of inferiority.',
                  'B98 is internal VAL/F only, has cluster overlap/few clusters and different support. Do not compare its MSE directly to B94 TEST.',
                  'M-H in B98 adds ECEI/BES AND NBI, not fast beyond beam. No physical transport time is identified.',
                  'No acceptance grade, no automatic model adoption, no automatic training, source edits or paper claims.'],
        'known_physics_directions':'Forced first-order grey-box response implemented; geometry-based displacement and CES linewidth budget need extra calibration/profiles/spectra. Do not invent these measurements.',
        'lean':'Six generic discrete-selector theorems compiled, finite80-real-query bridge passed; not complete Python/CUDA/physical verification.'},
        'b94':{'scopes':{},'latency':result['latency'],'limitations':result['limitations'],
               'overturning_measurement':result['overturning_measurement'],'budget_summary':{},'paired_by_design':{}},'b98':{},'reviews':{}}
    for scope,record in result['scopes'].items():
        target={'seed_count':len(record['seeds']),'unit_count':len(record['units']),'designs':{}}
        for design,rec in record['designs'].items():
            target['designs'][design]={'statements':rec['statements'],'cells':{}}
            for name,cell in rec['cells'].items():
                target['designs'][design]['cells'][name]={
                    **{k:cell[k] for k in ['rows','shots','clusters','excluded_from_prediction_support']},
                    'skill':interval(cell['skill']),
                    'metrics':{c:{m:interval(ms[m]) for m in ['RMSE','MAE','medianAE','share_within_10pct','share_outside_0_3000'] if m in ms} for c,ms in cell['metrics'].items()},
                    'supplementary_pairs':{c:{**interval(v),**{k:v[k] for k in ['rows','shots','clusters','excluded_from_cell']}} for c,v in cell['supplementary_tree_pairs'].items()},
                    'supplementary_baseline_support':{c:{'missing_unit_count':len(v['missing_units']),'available_rows':v['available_rows'],'different_model_budget':v['different_model_budget']} for c,v in cell['supplementary_tree_notes'].items()}}
        evidence['b94']['scopes'][scope]=target
        evidence['b94']['paired_by_design'][scope]={design:{name:{'rows':v['rows'],'clusters':v['clusters'],**v['skill']} for name,v in rec['cells'].items()} for design,rec in target['designs'].items()}
    budgets={}
    for name,v in result['fit_budgets'].items():
        parts=Path(name).parts;stage=parts[-2];code=parts[-1].split('__')[-1]
        budgets.setdefault(stage+'/'+code,[]).append({'epochs':len(v['history']),'updates':v['updates'],'validated_updates':v['validated_updates'],'discarded_updates':v['discarded_incomplete_epoch_updates'],'wall_seconds':v['wall_seconds'],'stop':v['stop']})
    for key,rows in budgets.items():
        evidence['b94']['budget_summary'][key]={'fits':len(rows),'stop_counts':{v:sum(r['stop']==v for r in rows) for v in set(r['stop'] for r in rows)},**{field:{'min':min(r[field] for r in rows),'median':statistics.median(r[field] for r in rows),'max':max(r[field] for r in rows)} for field in rows[0] if field!='stop'}}
    b=read(ROOT/'docs/paper/b98_physics.json');assert b['independent_readout_pass'] and not b['outer_test_evaluated']
    assert read(ROOT/'data/.b98/readout_complete.json')['result_sha256']==sha(ROOT/'docs/paper/b98_physics.json')
    evidence['b98']={'scope':b['scope'],'target':b['target'],'units':{u:{k:v[k] for k in ['metrics','clusters','shots','train_val_cluster_overlap','tau']} for u,v in b['units'].items()}}
    return evidence

def pointer(evidence,path):
    assert isinstance(path,str) and path.startswith('/') and '..' not in path
    item=evidence
    for name in path[1:].split('/'):
        key=name.replace('~1','/').replace('~0','~')
        item=item[int(key)] if isinstance(item,list) else item[key]
    return item
