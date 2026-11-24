"""Actual-cache and optimizer/scheduler restoration checks without new fits."""
import torch
import common107 as C
from model106 import create_model
from runtime92 import configure
def main():
    C.guard();C.inputs_guard();rt={**C.runtime(),'device':'cpu','threads':1,'mkldnn':False};configure(rt);checked=[]
    for case in C.CASES:
        z=C.prepared(case);m=z['identity']['manifest'];assert not set(m['train'])&set(m['val']+m['test']+m['held'])
        p=C.OLD.rd(case)
        if not (p/'process_exit.json').exists():continue
        C.OLD.verify_run(case);metrics=C.load(p/'metrics.json')
        if metrics['stop']=='converged':continue
        w=torch.load(p/'latest.pt',map_location='cpu',weights_only=False);model=create_model('LSTM_F',rt,C.SEED);model.load_state_dict(w['model'])
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);opt.load_state_dict(w['optimizer'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2);scheduler.load_state_dict(w['scheduler'])
        assert scheduler.state_dict()==w['scheduler'] and opt.param_groups[0]['lr']==w['optimizer']['param_groups'][0]['lr']
        for k,ss in w['optimizer']['state'].items():
            for n,v in ss.items():
                if torch.is_tensor(v):assert torch.equal(v,opt.state_dict()['state'][k][n])
        seg=next(s for s in z['val']['D10'] if s['mask'][:,0].any());model.eval()
        with torch.no_grad():
            pred=model(torch.from_numpy(seg['x'][:64]).unsqueeze(0));assert torch.isfinite(pred).all() and torch.count_nonzero(pred[...,1])==0
        checked.append(case)
    assert 'V1_r2_k0__outer' in checked and 'V4a__outer' in checked
    C.save(C.CACHE/'readiness.json',{'all_pass':True,'sources':C.sources(),'cases':list(C.CASES),'optimizer_restore_checked':checked,'new_fits':0,'TEST_scored':False})
    print('B107 real-cache/optimizer/scheduler restoration checks passed',flush=True)
if __name__=='__main__':main()
