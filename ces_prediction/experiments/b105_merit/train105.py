"""Namespace/config/model adapter to frozen train92; no loop changes."""
import argparse,os
import common105 as C
import common92 as M
import train92 as T
from model105 import create_model

def main(unit,stage):
    C.guard();C.inputs_guard();assert unit in C.UNITS
    M.CACHE=C.CACHE;M.sources=C.sources;M.guard=C.guard;M.save=C.save
    M.run_dir=lambda st,u,s,co:C.rd(u,st)
    T.prepare_unit=C.prepared;T.decision=C.runtime;T.checks_ready=lambda:C.guard()
    T.create_model=create_model;T.effective_env=lambda st,co,rt:C.env(unit,st)
    original=T.config
    def config(st,u,se,co,rt,prep):
        c=original(st,u,se,co,rt,prep);c.pop('config_hash')
        c.update(experiment='B105',registration='PREREGISTRATION_B105.md',epochs=1 if st=='smoke' else 40,
                 seconds=120 if st=='smoke' else 3600,channels=35,initial_anchor_columns='zero_trainable',
                 loop_sha256=C.sha(C.HERE.parent/'b92_reduced/train92.py'))
        c['config_hash']=C.digest(c);return c
    T.config=config;T.verify_complete=lambda rd:C.verify_run(unit,stage)
    T.train(stage,unit,C.SEED,'A35')
    import torch
    C.save(C.rd(unit,stage)/'hardware.json',{'device':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--unit',required=True,choices=C.UNITS);p.add_argument('--stage',choices=('smoke','confirm'),default='confirm');a=p.parse_args();main(a.unit,a.stage)
