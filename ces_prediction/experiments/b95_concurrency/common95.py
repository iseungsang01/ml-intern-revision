"""Bind new dispatch provenance without modifying frozen B94 fit sources."""
import subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'b94_parallel'))
import common94 as G
from audit95 import qualification,dispatch_config

def sources():
    paths=[HERE/n for n in ('profile95.py','audit95.py','common95.py','bootstrap95.py','production95.py')]
    paths += [HERE.parent/'PREREGISTRATION_B95.md',HERE.parent/'PREREGISTRATION_B94_ADDENDUM2.md',HERE.parent/'PREREGISTRATION_B94_ADDENDUM3.md']
    for p in paths:G.committed(p)
    return {p.relative_to(ROOT).as_posix():G.sha(p) for p in paths}

def policy():
    G.guard();s=sources();q=qualification()
    return s,q
