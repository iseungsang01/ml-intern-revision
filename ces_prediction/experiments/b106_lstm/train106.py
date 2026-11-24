"""F-only D10 adapter to the original frozen optimizer/loss loop."""
import argparse
import common106 as C
import common92 as M
import train92 as T
from model106 import create_model
def main(case,stage):
    C.guard();C.inputs_guard();rt=C.runtime()
    M.CACHE=C.CACHE;M.sources=C.sources;M.guard=C.guard;M.save=C.save;M.TRAIN_RULES=('D10',)
    M.run_dir=lambda st,u,s,co:C.rd(u,st)
    T.prepare_unit=C.prepared;T.decision=lambda:rt;T.checks_ready=lambda:C.guard();T.create_model=create_model
    T.effective_env=lambda st,co,r:C.env(case,st);T.epoch_segments=lambda prep,u,s,e:prep['train']['D10']
    original=T.config
    def config(st,u,s,co,r,prep):
        c=original(st,u,s,co,r,prep);c.pop('config_hash');c.update(experiment='B106',case=case,target='F',rule='D10',epochs=1 if st=='smoke' else 80,seconds=120 if st=='smoke' else 3600)
        c['config_hash']=C.digest(c);return c
    T.config=config;T.verify_complete=lambda p:C.verify_run(case,stage);T.train(stage,case,C.SEED,'LSTM_F')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=C.CASES,required=True);p.add_argument('--stage',choices=('smoke','confirm'),default='confirm');a=p.parse_args();main(a.case,a.stage)
