"""Fair internal-TRAIN-only specialized and rule-balanced mixture GP residual trees."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,time
import numpy as np,joblib
import common105 as C
import sys
sys.path.insert(0,str(C.HERE.parent/'b75_availability'))
sys.path.insert(0,str(C.HERE.parent/'b78_sparse'))
import stage_fits as SF
from sklearn.ensemble import HistGradientBoostingRegressor

def model():return HistGradientBoostingRegressor(random_state=20261116,early_stopping=False,max_iter=100)
def npz(path,**z):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**z);tmp.replace(path)
def run(unit,part='fit'):
    C.guard();C.inputs_guard();prep=C.prepared(unit);m=prep['identity']['manifest'];stats=prep['identity']['stats'];rd=C.CACHE/'trees'/unit
    rd.mkdir(parents=True,exist_ok=True);start=time.time()
    if part=='fit':assert not (rd/'complete.json').exists() and not list(rd.glob('*.joblib')),'No blind fit retry'
    else:
        assert C.load(C.CACHE/'gate.json')['all_pass'],'TEST forbidden after failed gate'
        done=C.load(rd/'complete.json');assert done['sources']==C.sources() and done['manifest_hash']==C.digest(m)
        for name,h in done['files'].items():assert C.sha(rd/name)==h
    ctx=SF.Ctx();R=ctx.R;f8,mb,me=ctx.unit_arrays({'fit':m['train']})
    istrain=np.isin(R['fidx'],m['train']);isval=np.isin(R['fidx'],m['val']);istest=np.isin(R['fidx'],m['test'])
    assert not (istrain&isval).any() and not ((istrain|isval)&istest).any()
    files={};counts={};mixture={h:[] for h in ('F','L5')}
    masks78=SF.K.load_npz(C.DATA/'.b78/sets.npz') if part=='test' else None
    for rule in C.TRAIN_RULES if part=='fit' else C.RULES:
        if rule=='K10':
            import common78
            I=common78.load_inputs('K10');H=SF.K.history_matrix(R['tstep'],I)
            A=SF.K.anchor_scalars(R['gi'],I['anc_g'],mb,me,ctx.mr_b,ctx.mr_e);XS=SF.Lazy(H,f8,A);gp=I['G'];sets=masks78
        else:XS,gp=ctx.matrices(rule,f8,mb,me);sets=ctx.S
        for head in ('F','L5'):
            mask=sets[('S_' if rule=='S' else 'common_')+head]
            fit=np.flatnonzero(istrain&mask);val=np.flatnonzero(isval&mask);test=np.flatnonzero(istest&mask)
            counts[rule+'/'+head]={'train':len(fit),'val':len(val),'test':len(test)}
            if part=='fit':
                assert len(fit)>=30 and len(val)>=30 and len(np.unique(R['fidx'][val]))>=2 and len(np.unique(R['clus'][val]))>=2,'Insufficient VAL support'
                x=XS['TX'][fit];y=(R['tg_'+head]-gp)[fit];assert np.isfinite(x).all() and np.isfinite(y).all()
                mod=model().fit(x,y);name=f'{rule}__{head}.joblib';joblib.dump(mod,rd/name);files[name]=C.sha(rd/name)
                npz(rd/f'{rule}__{head}__val.npz',idx=val,est=gp[val]+mod.predict(XS['TX'][val]))
                mixture[head].append((rule,x,y,len(fit)))
            else:
                xx=XS['TX'][test]
                for label in ('specialized','mixture'):
                    if label=='specialized' and rule=='K10':continue
                    name=f'{rule}__{head}.joblib' if label=='specialized' else f'mixture__{head}.joblib'
                    mod=joblib.load(rd/name)
                    npz(rd/f'{rule}__{head}__test__{label}.npz',idx=test,est=gp[test]+mod.predict(xx))
        print('Tree '+part+' '+unit+' '+rule,flush=True)
    if part=='fit':
        for head,parts in mixture.items():
            X=np.concatenate([p[1] for p in parts]);y=np.concatenate([p[2] for p in parts])
            weights=np.concatenate([np.full(p[3],len(y)/(7*p[3])) for p in parts])
            assert np.isclose(weights.sum(),len(y))
            mod=model().fit(X,y,sample_weight=weights);name=f'mixture__{head}.joblib';joblib.dump(mod,rd/name);files[name]=C.sha(rd/name)
            for rule in C.TRAIN_RULES:
                XS,gp=ctx.matrices(rule,f8,mb,me);idx=np.flatnonzero(isval&ctx.S[('S_' if rule=='S' else 'common_')+head])
                npz(rd/f'{rule}__{head}__val__mixture.npz',idx=idx,est=gp[idx]+mod.predict(XS['TX'][idx]))
        for p in rd.glob('*.npz'):files[p.name]=C.sha(p)
        C.save(rd/'complete.json',{'sources':C.sources(),'manifest_hash':C.digest(m),'stats_hash':C.digest(stats),'fit_files_hash':C.digest(m['train']),
               'normalization_files_hash':C.digest(m['train']),'files':files,'counts':counts,'fits':16,'seconds':time.time()-start,
               'tree':model().get_params(),'execution_device':'cpu','threads':2,'env':{k:v for k,v in os.environ.items() if k.startswith('CES_')},'mixture_weight':'N_total/(7*n_rule), no rule ID, no early stopping',
               'GP':'per-row <=16 visible past CES only; no pooled fit/VAL hyperparameters'})
    else:C.save(rd/'test_complete.json',{'sources':C.sources(),'files':{p.name:C.sha(p) for p in rd.glob('*__test__*.npz')}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--unit',choices=C.UNITS,required=True);p.add_argument('--part',choices=('fit','test'),default='fit');a=p.parse_args();run(a.unit,a.part)
