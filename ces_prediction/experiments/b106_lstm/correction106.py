"""Original15-feature post-correction principle; strict file-OOF residual fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,sys,time
import numpy as np,joblib
import common106 as C
sys.path.insert(0,str(C.HERE.parent/'b75_availability'));sys.path.insert(0,str(C.HERE.parent/'b78_sparse'))
import stage_fits as SF
import common78
from sklearn.ensemble import HistGradientBoostingRegressor
from predict106 import npz

def features(ctx,fast8,rule,idx,c):
    I=common78.load_inputs(rule) if rule=='K10' else SF.K.load_inputs(rule)
    h=I['H'][idx];gp=I['G'][idx]
    x=np.column_stack([fast8[idx],ctx.R['tstep'][idx],I['age'][idx],h,c-h,c-gp,np.abs(c-h),np.abs(c-gp)])
    assert x.shape==(len(idx),15)
    return x
def main(part):
    C.guard();C.inputs_guard();ctx=SF.Ctx();R=ctx.R;reports={}
    for u in C.UNITS:
        outer=C.prepared(u+'__outer');man=outer['identity']['manifest'];fast8,_,_=ctx.unit_arrays({'fit':man['train']})
        rd=C.CACHE/'correction'/u;rd.mkdir(parents=True,exist_ok=True)
        if part=='fit':
            assert not list(rd.glob('*.joblib')),'No silent fit repeat'
            allrules={};coverage={};provenance={}
            for rule in C.TRAIN_RULES:
                xx=[];yy=[];ids=[]
                for k in range(4):
                    case=u+'__inner'+str(k);meta=C.load(C.rd(case)/'manifest.json');p=C.CACHE/'predictions'/f'{case}__held__{rule}.npz';z=npz(p)
                    assert not set(meta['held'])&set(meta['train']+meta['val']+meta['test'])
                    assert set(meta['held'])<=set(man['train']) and str(z['manifest_hash'])==C.digest(meta)
                    assert str(z['source_hash'])==C.digest(C.sources()) and str(z['model_sha256'])==C.load(C.rd(case)/'complete.json')['best_sha256']
                    idx=z['idx'];assert np.isin(R['fidx'][idx],meta['held']).all()
                    X=features(ctx,fast8,rule,idx,z['est']);y=R['tg_F'][idx]-z['est'];keep=np.isfinite(X).all(1)&np.isfinite(y)
                    xx.append(X[keep]);yy.append(y[keep]);ids.append(idx[keep]);provenance[str(p.relative_to(C.ROOT))]=C.sha(p)
                allidx=np.concatenate(ids);assert len(np.unique(allidx))==len(allidx),'OOF overlap'
                X=np.concatenate(xx);y=np.concatenate(yy);assert len(y)>=30
                allrules[rule]=(X,y);coverage[rule]={'calibration_rows':len(y),'files':len(np.unique(R['fidx'][allidx]))}
            for label,rules in [('frozen_D10',('D10',)),('mixed_post',C.TRAIN_RULES)]:
                X=np.concatenate([allrules[r][0] for r in rules]);y=np.concatenate([allrules[r][1] for r in rules]);n=len(y)
                weights=np.concatenate([np.full(len(allrules[r][1]),n/(len(rules)*len(allrules[r][1]))) for r in rules])
                mod=HistGradientBoostingRegressor(random_state=20261116,early_stopping=False,max_iter=100).fit(X,y,sample_weight=weights)
                joblib.dump(mod,rd/(label+'.joblib'))
            C.save(rd/'complete.json',{'sources':C.sources(),'outer_manifest_hash':C.digest(man),'OOF_sources':provenance,'coverage':coverage,
                'models':{p.name:C.sha(p) for p in rd.glob('*.joblib')},'formula':'first_stage + HGB15 prediction; complete-feature fallback to first stage',
                'differences_from_historical':['F-only active Ti branch on B105 manifests','correction uses internal TRAIN OOF only, not earlystop VAL','early_stoppingFalse,max_iter100','mixed_post is a declared scenario extension, not historical reproduction']})
        else:
            done=C.load(rd/'complete.json');assert done['sources']==C.sources() and done['outer_manifest_hash']==C.digest(man)
            models={name[:-7]:joblib.load(rd/name) for name,h in done['models'].items() if C.sha(rd/name)==h};assert len(models)==2
            for rule in C.RULES:
                z=npz(C.CACHE/'predictions'/f'{u}__outer__test__{rule}.npz');idx=z['idx'];c=z['est'];X=features(ctx,fast8,rule,idx,c);keep=np.isfinite(X).all(1)
                arrays={'idx':idx,'LSTM':c,'fallback':~keep}
                for name,mod in models.items():
                    est=c.copy();est[keep]+=mod.predict(X[keep]);arrays['LSTM_'+name]=est
                out=C.CACHE/'estimates'/f'{u}__{rule}.npz';out.parent.mkdir(exist_ok=True);np.savez(out,**arrays)
        reports[u]={'part':part,'complete':True}
        print('Correction '+part+' '+u,flush=True)
    if part=='fit':C.save(C.CACHE/'correction_checks.json',{'all_pass':True,'sources':C.sources(),'units':reports,'calibration_files_are_outer_internal_TRAIN':True})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=('fit','test'));a=p.parse_args();main(a.part)
