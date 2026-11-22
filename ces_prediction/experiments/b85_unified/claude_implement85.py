"""Claude authors allowlisted new B85 files from a redacted methodological spec."""
from pathlib import Path
import json
import urllib.request
import claude_review85 as REVIEW

HERE = Path(__file__).resolve().parent
ROOT = REVIEW.ROOT
ALLOW = {'common85.py','data85.py','model85.py','checks85.py','train85.py','run85.py','predict85.py','readout85.py'}
CONTEXT = """Author complete executable Python implementation for the registered experiment below. Return ONLY valid JSON {"files":{"common85.py":"source",...}} with exactly eight files: common85.py,data85.py,model85.py,checks85.py,train85.py,run85.py,predict85.py,readout85.py. No fences/TODO/stubs. All new files reside ces_prediction/experiments/b85_unified; ROOT=Path(__file__).resolve().parents[3]. No existing module changes. No synthetic data. Standard numpy/torch/scipy available, Python3.14 Windows. ASCII print strings, utf8 reports, atomic files. All files start a comment saying Claude authored the implementation.

Existing interfaces: b73_targets/common.py C has load_npz,save_npz,save_json,save_text,DATA,PAPER,grid(),rows(). WARNING importing C sets CUDA_VISIBLE_DEVICES empty: common85 must save/restore prior variable AFTER imports and BEFORE torch import; runner explicitly sets CUDA_VISIBLE_DEVICES=0. b73_targets/designs.py DS.units(R) returns(list,tab,fifth). Unit dict name,design,sub,fit int-file-ids,test int-file-ids. Select V1_r2_k0..4,V2_s1_k0..4,V3_q1..4,V4a, exactly15. INITS1701,1702,1703 variantsX/H exactly90runs. Common85 exposes select_units(),load_sources()->(G,R), ROOT,DATA,CACHE(data/.b85),PAPER,TRAIN_RULES,RULES,INITS,VARIANTS. Avoid package-name import collisions via sys.path careful explicit paths.

Mandatory caches: .b73/grid.npz G keys names(strarray),off(file offsets),time,ti,fast(N,13 BES9/ECEI4),seg(global unique segments gap>=.5s),fidx(fileids),grow(filerowindex). .b73/rows.npz R keys gi(uniqueglobalgridindices),fidx,grow,shot,clus,blk,tstep,tg_F,tg_L5,y,r2_k,r3_k,r4_k. Labels only at R.gi; other gridrows maskfalse. Targets from R exactly, no recomputing. data85 public unit_manifest(u) ->train,val,test integerlists; fit_stats(train_file_ids) serializabledict fast_mu/sd(13),ti_mu/sd,train_files; build_segments(file_ids,stats,rule) ->listdict x(n,22)float32,y(n,2)float32,mask(n,2)bool,gi(n)int,fidx scalar,segment scalar. Expose anchors(rule,G=None) globalindexvector -1 whennone and a feature helper allowing modified G/frozenstats for fullprefixaudit. Rule_for(unit,init,epoch,segment) uniform7 deterministic pairedX/H. G/R processcachedokay. Stats perunit ALL finite G.ti andG.fast trainingfiles regardlessrule, floorstd1e-6. L5 >=3 in sameoriginalsegment already inR, undefinedmasked; report reduced-count future label rows separately wherepossible.

Input x slots0:13 standardizedfast;13:15zeroMC;15log1p(dtsec)firstsegmentrow0;16normalizedvisibleTi;17log1p(agesec);18hasvisible;19:22zeroVThistory. No history value/age ->zero. Every anchor strictlyearlier globalindexsame segment. D limit t-delay+.003. S: alreadyobservedgap .015<dt<.045 cutoff atindexjustbeforegap; subsequentcutoffsstrictlyadvance, nohistorybeforefirstgap. K10 can use mandatory existing .b78/frames.npz keyfr nominalindex,scored. Beforecurrentrow,finiteTi,fr>=0,fr%10==0. No perrule normalization, contextonlypast/currentfast, targetsgluedloss only. Oneconstant ruleperoriginalsegment so recurrent memory not retroactively hidden.

Model85 imports existing seq/model_seq_v2.py SeqCESLSTMv2(n_in22,n_fast15,hidden_ti160,layers_ti2,head64,dropout.1,n_extra_vt0,n_extra_ti0). Extract only lstm_ti,norm_ti,head_ti; discardVTmodules. Add second separate160->64GELU->1head, bothreadsameTistate. forward(x,lengths=None)->B,L,2 F/L5; forward_step(x,state=None)->pred,state forlatency. VariantH zerosfirst15 inforward, sameparamcount<1million. Packedseqpadding andresetstatepersample. No GP/residualcorrection. CPU inferenceokaybuttrainingCUDAmandatory.

Checks85 command beforeGPUsmoke: compare anchors/ages/hasagainst .b75/inputs/{D10..S}.npz anc_g,age(ms),nvis onRrows;K10from.b78/inputs/K10.npz samekeys. Mutate currentlyinvisibleGti byNaN/+1000 atsampledtime, FROZENstats: entireprefixxthroughtimeunchanged;futurefastmutationssame;labels-onlymutationsxunchanged. Allmanifestsdisjoint, statsunchangedwhenheldoutmutated,labelsRexact,rulesdeterministicnoK10,resetboundaries. Exposeaudithelpers. Use realdata only 5sampletimes/rule reasonable. Write data/.b85/input_checks.json passestrue onlyonsuccess, docs/paper/b85_readiness.md. No testerrors. Parentindependentverifierlater.

Train85 CLI --unit NAME --init INT --variant X/H [--smoke]. Use manifesttrain/valonly, stats excludeval/test. RealCUDArequiredfailotherwise. Smoke up to12train/3valfiles,1epoch,batch2,separateCACHE/smoke no productionmarker. Fullcap80batch16 AdamW.001decay.0001clip1 dropout.1 schedulerfactor.5patience2 earlystop6. Eachtrainsegment chooses rule deterministically perepoch; twin samechoices/shuffle. Wholeoriginalsegments paddedpacked, no carryacrosssamples. Training lossmeanheads ofmaskedMSE +.1meannegativephysicalTireluinnormalizedscale supervisedonly. Validation each7rules fixed equalrules/headMSE, bestcheckpointselection. Cache rule-specificdatasets toavoid expensive rebuilding everyepoch whenpossible. Atomic latestcheckpoint includes optimizer/scheduler/epoch/bestweights and RNGstates inclCUDA/np/random toexactresume; bestweights separate. Config+manifest/statshash checkedonresume and complete. Exclusiveperrunlock O_EXCL andPIDguard; rejectactivelock, recoverclearlydeadlockonly. Don'tinterpret testperformance. Save metrics.json,complete.json onlysuccessful. CPUfallbackforbidden. Finiteforward/backward/gradientchecksmoke; report load/savepredictionequality realbatch andlabelmutationinvariance.

Run85 CLI --plan,--smoke,--status, optional --unit/--init/--variant, defaultall90 sequentialsubprocesses. Plan writesmanifest/config/count only; noGPU. Batchlockpreventduplicatewriter. Alwayspin CES vars data/dropheld/cut3000/per_shot0/expand0/zeroMC1/zeroVT1/epochs80/batch16/lr.001/lossmse/selectmse/devicecuda/fixed0/train_ctx0/eval_ctx0 and CUDA_VISIBLE_DEVICES0. Waitnvidia-smifreeVRAM>=2500MiB in30spolls (neverkillotherprocess). Inputcheckpassrequired. Per-runlogsCACHE/logs, globaldocs/paper/b85_status.md updatedstate/runprogress. Stopbatchnonzeroonfailure. Resumeonlymatchingcomplete. Smoke command firstunit/initbothvariants, docs/paper/b85_smoke.md. Afterall90 invokepredict85/readout85 automatically. Don'tselfdaemon; callerwillStartProcessHidden.

Predict85 requireall90complete,inputchecks; evalnografulltestsegments each8rules, sameweights. SaveCACHE/predictions/{unit}__{init}__{variant}__{rule}.npz idx(Rindices),est(n,2)eV, finite. AllRtestrowsincludingmissingL5labels; no imputepredictions. Verifyindicesidenticalandfinite acrossarms, targetlabelmutationinvariance, metadatahashes. data/.b85/prediction_checks.json trueonlyallfilescomplete.

Readout85 guardedall90andinput/predictionchecks. BaselineinventoryfixedBEFOREerrors. .b75/sets.npz common_F/common_L5 D rules,S_F/S_L5 S; .b78/sets.npz common_F/common_L5 K10. BaselineT0/TX .b75/fits/{unit}__{learner}__{rule}__{F/L5}.npz idx,estscalar;K10.b78/fits samepattern. .b84/fits/{unit}__TXall__{rule}.npz L5only,V1r2/V3/V4a; V2unavailableexplicit. .b73/trees/{unit}__TX__F.npz D10/Ffullcomparison. Exactrowintersections withlabelsfinite/baselinefinite, reportexclusions no silentrows. AggregateV1/V2folds,V3fifths perinit, V4asingle. MetricsRMSE,MAE,medianAE,sharewithin10%target>100eV,range,shareoutside0to3000,eachinit+mean. PairskillsX/H,X/T0,TX,TXall. Bootstrap20000initpickedfirstpairedthencluster, dischargebeside, deterministic[20261185,crc32(cell/form)], vectorizeperunitsums/batches500. RMSE/MAE/skillintervalsessential; descriptive medAE/share points explicitlylabelif bootstrapnotimplemented. Allregisteredstatementsperdesign exactly prereg. Neverreadpartialscientificscores. Writeb85_unified.json/md tablescountsallinitsverdictslimitationsandbaselinemissing. Agebins/coldstartdescriptive; benchmarkcompletedmodelCPUforward_stepbatch1realtrace thread1,warmup20,max2000steps, preprocessingseparatemedian/p99. NoGP/treefit. Keepoutputcompactbutcomplete.

Reviewer disposition: originalreviewmistakenlycalledSnonmonotonic and strictpriorindexwithdelaytolerancefutureleakage; do NOT change these definitions. No timestamps sortedindexfuture leakage allowed. Crosscorrelationdoesnotestablishacquisitionalignment; no newcrosscorrrequired. L5 surrogateonlynottruth, labelsminimum3perexistingdefinition, fewerthan3masked. StatsrecomputedperunitALLfinitetrainingvalues ruleindependent. Statezeropersample. Comparatortrain/testlistsprovenanceoriginalDSdisjoint. Noextraablation/hyperparametersearch. Allcodecomplete; no external API calls in training/source otherthanthisgenerator. Limitverbositytofit24000outputtokens.

Registration follows (methodological protocol, no identity/raw data or experimental results):
"""


def main():
    reg = (ROOT/'ces_prediction/experiments/PREREGISTRATION_B85.md').read_text(encoding='utf-8')
    prompt = CONTEXT + reg
    payload = {'model':'claude-opus-4-6','max_tokens':24000,'thinking':{'type':'disabled'},
        'messages':[{'role':'user','content':prompt}]}
    req = urllib.request.Request(REVIEW.BASE+'/messages',
        headers={'x-api-key':REVIEW.credential(),'anthropic-version':REVIEW.VERSION,'content-type':'application/json'},
        data=json.dumps(payload).encode('utf-8'),method='POST')
    with urllib.request.urlopen(req,timeout=900) as r:
        response=json.load(r)
    text='\n'.join(b['text'] for b in response.get('content',[]) if b.get('type')=='text')
    record={'prompt':prompt,'model':response.get('model'),'usage':response.get('usage'),
        'stop_reason':response.get('stop_reason'),'response':text}
    cache=ROOT/'data/.b85';cache.mkdir(parents=True,exist_ok=True)
    (cache/'claude_implementation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    if not text: raise RuntimeError('No Claude text returned')
    clean=text.strip()
    if clean.startswith(chr(96)*3): clean=clean.split('\n',1)[1].rsplit(chr(96)*3,1)[0]
    files=json.loads(clean)['files']
    if set(files)!=ALLOW: raise RuntimeError('Returned file allowlist mismatch')
    for name,source in files.items():
        compile(source,name,'exec')
        if (HERE/name).exists(): raise RuntimeError('Refusing overwrite '+name)
    for name,source in files.items(): (HERE/name).write_text(source.rstrip()+'\n',encoding='utf-8')
    print('Claude authored '+str(len(files))+' allowlisted files using '+record['model'])
    print('Usage '+json.dumps(record['usage']))


if __name__=='__main__': main()
