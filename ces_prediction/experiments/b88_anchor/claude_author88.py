"""Bounded, author-authorized Claude implementation calls for B88 only.

Prompts contain protocol and source contracts, never data arrays or credentials.
Completed responses are cached; no automatic paid retries. Run with py from root.
"""
import argparse,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'b85_unified'))
import claude_review85 as API
CACHE=ROOT/'data/.b88/claude'
MODEL='claude-opus-4-6'
CONTEXT="""Implement a registered causal single-chord temperature nowcast experiment.
No author identity, private arrays or performance values are supplied. Existing training is
running in another process and its source files MUST NOT be modified. Write only the new
files requested. No fabricated observations, no random-tensor tests. All runs use existing
real caches. Return ONLY JSON {"files":{"filename.py":"complete source"}}, compact code,
not an essay. Reuse imported code rather than copying an entire trainer. Do not omit guards.

Frozen baseline helpers in sibling b85_unified/:
common85 ROOT,DATA,CACHE(data/.b85),PAPER; C is b73 common module with save_json(path,obj),
save_text(path,text),load_npz(path); load_sources()->(G,R), select_units()->15 dicts with
name,fit,test; INITS=[1701,1702,1703],TRAIN_RULES seven D10,D50,D100,D300,D650,D1000,S;
RULES includes held-out K10. data85.unit_manifest(u)->train,val,test file-id lists;
fit_stats(train_ids,G=None,R=None)->fast_mu/sd13,ti_mu/sd,train_files; anchors(rule,G=None)
->global source indices (-1 absent); build_segments(ids,stats,rule,G=None,R=None)->list of
dict x(n,22),y(n,2)NORMALIZED,mask(n,2),gi(n)globalgrid,fidx,segment. G keys fast(N,13),
time,ti,seg,fidx,off,shot,names,grow. R keys gi,fidx,shot,clus,tg_F,tg_L5.
model85.B85Model(variant='X',n_in=22,n_fast=15) has TiLSTM160x2,LayerNorm,two64GELUheads;
forward(x,lengths=None)->(B,L,2),forward_step(x,state=None)->(B,2),state.
Native ATen CUDA (cudnn disabled),TF32off. It imports actual inherited Ti modules.

B88 registered model: original22 inputs plus13 standardized fast at latest visible
CES MEASUREMENT index, same segment, strictlyprior; absentanchor appendedzero. Shared
phi Linear13,32 GELU Linear32,16 GELU for current x[:13] and anchor x[22:35]. A uses
phi(anchor)*x[18:19]; matched C zeros entire anchor embedding. Both fuse
[phi(current)16,anchor16,x[15:22]7]39 into B85Model n_in39/n_fast32. Direct F/L5heads,
no correctionstage. A/C identicalparameters/initialweights/tensorwidth35, seedpaired.
SameB85data/stats/units/mixture/optimizer/loss/batch16/cap80/earlystop6. NeedGPUafterB85
complete only; CPUinput audits/untrainedforward allowed. Cache/data/.b88 and b88_ docs
only. Parent supplies frozen_b85_sources.json mapping rootrelativepaths->SHA256;
never generate/update those pins. Checksha on launch/reuse. Missing pins fail loudly.
"""

PROMPTS={
'readout':CONTEXT+"""
Write predict88.py/readout88.py compact complete. common88 pathlib constants,C=common85.C. adapter88 activate()->T adaptedtrainer; guarded_ready() validates inputchecks+ALL90 A/C exacthash/cleanexit; dependency_ready() validates B85complete+90clean. Call both and assert_frozen BEFORE ANY scientificerrorcalculation orprediction. Sourceinventory hash binds execution files (claude_author excluded).
predict88: import torch; ownA/C loops15x3x2x8rules, savedstats/manifesthash verified byT.verify_complete, savedbest.pt weights_onlyTrue maplocationcuda ifavailable elsecpu, create_model(variant,init_seed). Build REALfulltestsegments35, noctxtruncation, eval/no_grad normalizedoutputsdenorm usingstats, mapglobalgi toRrowidx sorted unique. Atomic .b88/predictions/unit__init__variant__rule.npz idx,est(n2),config_hash,source_hash(ownprediction/common/data/model). No partialresultreuse. Finite+all variants/init/rules sameidxperunit checks. Labelmutation predictioninvariance onreal firstunit stats andfirst2testids D100: mutate R tg_F/tg_L5 by+500; originalinput35exact andpredsunchanged1e-5. Writeprediction_checks.json all_pass onlyafterall predictions+labelcheck; no errorscores.
readout88 reuse importreadout85 as B onlypuremetrics/bootstrap/median_bootstrap; set B.rng_for=lambda name:np.random.default_rng([20261188,zlib.crc32(name.encode())]) localprocess. DO NOTpatchB.M(common85). Implementown guarded_ready:adapter.guard+dependency +predictionchecks +verifyALLpred idxunique,estshape,finite,confighashandsourcehashcurrent before scoring. ownpooled_predictions(group,rule,nrows) ->A/C andB85X arrays3,n,2 + exactcommon support; B85X cached .b85/predictions sameunit/init/rule, validatedconfig_hash viafrozenuniquetrain85 dependencyverification+configs. Assertnonsplitoverlap, supportidenticalA/C eachseed; B85Xuseexplicitintersection withseparatecounts ifsupportdiff.
Read G,R. sets .b75/sets.npz common_F/common_L5, S_F/S_L5; .b78/sets.npz common_F/L5forK10. FourgroupsV1,V2,V3,V4a. All8rules×2heads+ D10/F_full np.ones support. Rtruth tg_F/tg_L5 andfinite. For eachcell B.metrics(name,estshape3rows,truthrows,labels{'cluster':R.clus,'discharge':R.fidx}); A/C exactsame rows; pairs A over C B.bootstrap(name,Aerror²,Cerror²,labels), A over B85X onexplicitfiniteintersection rows; all metrics/count/exclusions/perinit noensemble. Optionaltreesnotrequired.
Statement specs perdesign A1 D10/F_full AoverC lower>-.05 expectationholds; A2 D650/L5 AoverC lower>0 uncertain; A3 S/L5 same; A4 K10/L5 same; A5 D10/F_full AoverB85X lower>-.05 uncertain. 20000defaultdraws inheritedB funcs,excludeszero property. Save completeJSON .b88/readout.json plusdocs/paper/b88_anchor.json and b88_anchor.md (parent permits scientificoutputs ONLYafterguards; docs owned runtimeoutput okay). Resultlimitations useddataexploratory,F/L5notindependentphysicaltruth,3fixedinits,no refituncertainty/multiplicity,smallclusters,constantpersegmentavailability. Adoption allA1/A2/A4/A5 all4designs AND CPU p99<=10ms; Sclaim onlyifA3holds all4; noautomaticsmanuscriptedit. Negative overturningmeasurement untouchedcampaign acquisition+arrivaltimestamps incrementalagainstmatchedC.
cpu_latency own:guardalready;torch.set_num_threads1; firstunitfirstseedA savedmodelcpu, REALheldoutD10 segments; timepreprocessingtotal separatelyperrow; foreachsegmentstateNone; forward_step actualrow(1,1,35),skipfirst20warmup realrows, collectmax2000realsteps; returnmedian,p99ms,threads,platform,samples,amortizedpreprocessms. No synthetic/randomdata. JSONscientificreadoutonlyafterguard. Markdown allfourstatements tables andallcellmetricpoint/cluster95% withdischargeinJSON, completecounts/ranges. no wins/novel/outperforms language. Exactfunctions reusable standalone nofakereadouttest. Return JSONfilesonly.
""",
'runtime':CONTEXT+"""
Write adapter88.py,train88.py,run88.py,queue88.py only, compact complete code. New common88 uses pathlib Paths ROOT(repo),HERE,CACHE,DATA,PAPER, C=common85.C; assert_frozen returns pin mapping. model88.create_model(variant,init_seed), model.variant A/C. data88 signature matches data85.
adapter88.activate()->import train85 T and patch ONLY T.CACHE=.b88,T.build_segments,T.create_model,T.make_batches,T._config_hash,T.acquire_lock,T.release_lock. NEVER common85.CACHE. Dynamic make_batches identical baseline shuffle using random.Random(seed).shuffle(indices), padding dtype and lengths, infer35 width. T._config_hash(config,unit) idempotently inject B88 metadata (input35/fused39/phi13,32,16/backbone160x2/heads64/direct/AorC) and sha256 canonical config plus ALL local execution .py hashes excluding claude_author88.py and lead_checks, frozen map. T.verify_complete calls _config_hash(clean,config['unit']) and requires process_exit0+matchinghash. T.train_unit(unit,init,variant,smoke=False) uses CACHE/input_checks.json all_pass, CUDA mandatory, get_run_dir CACHE/{smoke|runs}/unit__init__variant, builds fixed protocol config itself and calls patchedhash, runs actual original trainingloop. No own trainer implementation.
Safe new locks: do not use os.kill(pid,0) anywhere, Windows may terminate. Use ctypes OpenProcess QUERY_LIMITED_INFORMATION/GetExitCodeProcess STILL_ACTIVE259/CloseHandle; error accessdenied must treatalive. New run locks exclusive O_EXCL, stale only if provablydead, release owneronly.
adapter dependency_ready(): assert frozen; read .b85/queue_status.json, requirestatecomplete, then import train85.py under UNIQUE alias via importlib.util WITHOUT activate to preserve original callback/hash/CACHE and verify ALL15x3xX/H existingfits complete. Explicit assert common85.CACHE == DATA/'.b85'. Function must work even after activate (uniquealias own globals). Missing/running -> returnFalse; statefailed -> raiseRuntimeError. statecomplete butfitmissing/stale -> raiseRuntimeError. Implement guarded_ready() for ALL90 own A/Cfits+inputchecks, and source_inventory() hashes. All sourcepins validated launch/checkpointreuse.
train88 CLI --unit --init --variant A/C --smoke; dependency_ready must true; activate and direct T.train_unit, never train85main choicesX/H.
run88 CLI --smoke,--status,--plan; status/plan readonly noGPU. Runactual requiresdependencyready+inputchecks. SequentialA/Csmokes(firstunit/firstseed) or90full fits. Before launch wait freeVRAM>=2500MiB via nvidia-smi polling10s (do not kill anything); verifydependency. Explicit env clears CES_* then sets T.ENV_PINS and PYTHONUNBUFFERED1. Launch subprocess sys.executable -Xfaulthandler -u train88.py args, logto run_dir/process.log. Onexit atomicallywrite process_exit.json returncode,config_hash fromsavedconfig, timestamp; nonzerofailfast; zero requireT.verify_complete. Reusecompleteonly after verification. New batchlock ownisolatedcache. No readout inrun88.
queue88 sequential dependency wait30s, dependencyfailed->statusblocked_dependency exit1, no spin. Newexclusivequeuelock. status JSON stateswaiting_dependency,smoke,registered_batch,predict,readout,complete,failed; phasefail stops noautoretry. Onready callrun88 --smoke,run88,predict88.py,readout88.py subprocesses. NoGPU starts beforedependency. launchfromroot only. Return JSONcomplete sources only.
""",
'core':CONTEXT+"""
Write common88.py,data88.py,model88.py ONLY. common88 assert_frozen() reads pins and
checks each safe root-relativefile; constantsROOT,DATA,CACHE,PAPER,HERE,INITS,VARIANTS
['A','C'],RULES,TRAIN_RULES and load_sources/select_units forwardedread-only. Creating
data/.b88 okay; do NOT mutate common85.CACHE. data88 forwards fit_stats/unit_manifest/
anchors then build_segments uses data85 list and appends anchorfeatures using SAMEstats,
first22/y/mask/gi preserved exactly; support explicit G/R mutation audit args, normalize
lists via np.asarray; enforceprior/samefile/segment, finitefeatures. Do not cache mutable
audit dictionaries. model88 subclassB85Model and overrideforward/forward_step to fuse,
avoid doublefusing. constructorvariantA/C maps no H inputmask; create_model paired
torchmanualseed; preserveactualbackbonearch, <1millionparams. torch CPU construction
allowed and CUDA seed onlyifavailable. assertinput35. No statistics fitting inmodel.
""",
}

ALLOWED={
 'core':{'common88.py','data88.py','model88.py'},
 'runtime':{'adapter88.py','train88.py','run88.py','queue88.py'},
 'readout':{'predict88.py','readout88.py'},
}

def generate(stage,prompt):
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/(stage+'.json')
    if path.exists():record=json.loads(path.read_text(encoding='utf8'))
    else:
        r=API.request('/messages',API.credential(),{'model':MODEL,'max_tokens':{'core':6000,'runtime':9500,'readout':8500}[stage],
            'thinking':{'type':'disabled'},'messages':[{'role':'user','content':prompt}]})
        record={'prompt':prompt,'model':r.get('model'),'usage':r.get('usage'),
                'stop_reason':r.get('stop_reason'),'content':r.get('content',[])}
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    text='\n'.join(b['text'] for b in record['content'] if b.get('type')=='text').strip()
    if text.startswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0]
    obj=json.loads(text)
    files=obj['files']
    assert set(files)==ALLOWED[stage],set(files)
    for name,source in files.items():
        compile(source,name,'exec')
        assert (HERE/name).resolve().parent==HERE
    # Candidates never overwrite reviewed execution sources on later replay.
    candidates=CACHE/'candidates'/stage;candidates.mkdir(parents=True,exist_ok=True)
    for name,source in files.items():(candidates/name).write_text(source.rstrip()+'\n',encoding='utf8')
    print(json.dumps({'stage':stage,'model':record['model'],'usage':record['usage'],'stop_reason':record['stop_reason'],'files':list(files)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=ALLOWED);a=p.parse_args()
    if a.stage not in PROMPTS:raise RuntimeError('Prompt not finalized for stage')
    generate(a.stage,PROMPTS[a.stage])
