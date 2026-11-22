"""Independent post-fit validation selection and paired NPZ score points."""
import sys,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import common90 as M


def main():
    from adapter90 import guarded_ready,dependency_ready
    assert dependency_ready();guarded_ready()
    lock=M.promotion();scores={}
    for method in M.METHODS:
        ratios=[]
        for u in M.select_units():
            name=f"{u['name']}__1701__X"
            a=json.loads((M.CACHE/'runs'/method/name/'metrics.json').read_text())['best_val_loss']
            b=json.loads((M.DATA/'.b85/runs'/name/'metrics.json').read_text())['best_val_loss'];ratios.append(a/b)
        scores[method]=sum(ratios)/len(ratios)
    assert scores==lock['scores'] and sorted(scores,key=lambda m:(scores[m],m))[:2]==lock['promoted']
    _,r=M.load_sources();n=len(r['gi']);checked=0
    for stage in ('screen','methods'):
        record=json.loads((M.CACHE/f'b90_{stage}.json').read_text());seeds=record['initializations']
        for method,rec in record['methods'].items():
            for design,d in rec['designs'].items():
                units=[u for u in M.select_units() if u['name'].startswith(design)]
                for key,cell in d['cells'].items():
                    rule,target=key.split('/');head=0 if target.startswith('F') else 1;y=r['tg_F' if head==0 else 'tg_L5']
                    if target=='F_full':mask=np.isfinite(y)
                    else:
                        with np.load(M.DATA/('.b78' if rule=='K10' else '.b75')/'sets.npz') as z:mask=z[('S_' if rule=='S' else 'common_')+target]&np.isfinite(y)
                    pred={}
                    for var in ('X','H','B85X'):
                        p=np.full((len(seeds),n),np.nan)
                        for i,seed in enumerate(seeds):
                            for u in units:
                                folder=M.DATA/'.b85/predictions' if var=='B85X' else M.CACHE/'predictions'/method
                                v='X' if var=='B85X' else var
                                with np.load(folder/f"{u['name']}__{seed}__{v}__{rule}.npz") as z:
                                    idx=z['idx'];assert np.isnan(p[i,idx]).all();p[i,idx]=z['est'][:,head]
                        pred[var]=p
                    for title,ref in (('X over H','H'),('X over B85X','B85X')):
                        keep=mask&np.isfinite(pred['X']).all(0)&np.isfinite(pred[ref]).all(0)
                        num=((pred['X'][:,keep]-y[keep])**2).sum(1);den=((pred[ref][:,keep]-y[keep])**2).sum(1)
                        point=float(np.mean(1-num/den));assert np.isclose(point,cell['pairs'][title]['point'],rtol=1e-10,atol=1e-12)
                        checked+=1
    report={'passes':True,'paired_cell_points':checked,'validation_only_promotion_recomputed':True,'scope':'independent artifact pooling and arithmetic; selection remains used-data exploration'}
    M.save(M.CACHE/'output_lead_checks.json',report);M.C.save_text(M.PAPER/'b90_output_lead_checks.md','# B.90 independent output audit\n\n'+json.dumps(report,indent=2)+'\n')
    print('PASS: B90 independent promotion and output points')


if __name__=='__main__':main()
