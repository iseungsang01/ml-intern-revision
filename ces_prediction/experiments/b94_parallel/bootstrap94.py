"""Fresh-process entry points; import adapters before any inherited callback."""
import argparse,importlib,json
import common94 as G
from adapter94 import activate
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('checks','train','select','predict','readout'))
    p.add_argument('--stage',choices=('smoke','smoke_confirm','pilot','confirm'));p.add_argument('--unit');p.add_argument('--seed',type=int);p.add_argument('--code');a=p.parse_args()
    T=activate()
    if a.action=='train':
        assert a.stage and a.unit and a.seed is not None and a.code
        T.train(a.stage,a.unit,a.seed,a.code)
    elif a.action=='select':print(json.dumps(importlib.import_module('select92').lock_selection()))
    else:importlib.import_module({'checks':'checks92','predict':'predict92','readout':'readout92'}[a.action]).main()
if __name__=='__main__':main()
