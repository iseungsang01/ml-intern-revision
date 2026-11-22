"""Independent locked validation allocation and completed paired score points."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common92 as M


def main():
    from predict92 import guarded_ready
    locked,_,_=guarded_ready();values={}
    for stage,u,s,c in M.pilot_jobs():values[u,c]=json.loads((M.run_dir(stage,u,s,c)/'complete.json').read_text())['best_val_loss']
    means={c:sum(values[u,c]/values[u,'C'] for u in M.PILOT_UNITS)/4 for c in ('A','T')}
    eligible=[c for c in ('A','T') if means[c]<.98 and all(values[u,c]/values[u,'C']<=1.05 for u in ('V3_q3','V4a'))]
    chosen=None if not eligible else eligible[0] if len(eligible)==1 else 'A' if abs(means['A']-means['T'])<=.005 else min(eligible,key=means.get)
    assert chosen==locked['chosen'] and chosen is not None
    _,r=M.load_sources();n=len(r['gi']);record=json.loads((M.CACHE/'readout.json').read_text());codes=M.pair(chosen);checked=0
    for scope,sr in record['scopes'].items():
        seeds=sr['seeds']
        for design,dr in sr['designs'].items():
            units=[u for u in sr['units'] if u.startswith(design)]
            for key,cell in dr['cells'].items():
                rule,target=key.split('/');h=0 if target.startswith('F') else 1;y=r['tg_F' if h==0 else 'tg_L5']
                if target=='F_full':mask=np.isfinite(y)
                else:
                    with np.load(M.DATA/('.b78' if rule=='K10' else '.b75')/'sets.npz') as z:mask=z[('S_' if rule=='S' else 'common_')+target]&np.isfinite(y)
                pp=[]
                for c in codes:
                    p=np.full((len(seeds),n),np.nan)
                    for i,seed in enumerate(seeds):
                        for u in units:
                            with np.load(M.CACHE/'predictions'/f'{u}__{seed}__{c}__{rule}.npz') as z:
                                idx=z['idx'];assert np.isnan(p[i,idx]).all();p[i,idx]=z['est'][:,h]
                    pp.append(p)
                keep=mask&np.isfinite(pp[0]).all(0)&np.isfinite(pp[1]).all(0)
                num=((pp[0][:,keep]-y[keep])**2).sum(1);den=((pp[1][:,keep]-y[keep])**2).sum(1)
                assert np.isclose(float(np.mean(1-num/den)),cell['skill']['point'],rtol=1e-10,atol=1e-12)
                assert int(keep.sum())==cell['rows'];checked+=1
    report={'passes':True,'paired_cells':checked,'validation_selection_independently_recomputed':True,'scopes_separate':True}
    M.save(M.CACHE/'output_lead_checks.json',report);M.C.save_text(M.PAPER/'b92_output_lead_checks.md','# B.92 independent output audit\n\n'+json.dumps(report,indent=2)+'\n')
    print('PASS: B92 independent validation allocation and output points')


if __name__=='__main__':main()
