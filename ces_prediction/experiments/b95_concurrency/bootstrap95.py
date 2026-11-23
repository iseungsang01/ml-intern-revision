"""Frozen numeric training, explicit global dispatch and bounded local groups."""
import os,sys
from pathlib import Path
import common95 as C

def main():
    sources,decision=C.policy()
    from adapter94 import activate
    T=activate()
    if len(sys.argv)>1 and sys.argv[1]=='train':
        n=int(os.environ['B94_CONCURRENCY']);i=int(os.environ['B94_WORKER_SLOT']);group=i//2
        assert n in (1,decision['selected_workers']) and 0<=i<n
        os.environ['B94_CONCURRENCY']=str(min(2,n-2*group));os.environ['B94_WORKER_SLOT']=str(i%2)
        original=T.config
        def config(*args,**kwargs):
            c=original(*args,**kwargs);c.pop('config_hash')
            c.update(global_gpu_worker_limit=n,global_worker_slot=i,local_worker_group=group,
                     scheduled_concurrency_scope='local group; global capacity recorded separately',
                     dispatch_sources95=sources,dispatch_policy_sha256=decision['decision_sha256'],
                     dispatch_registration='PREREGISTRATION_B94_ADDENDUM3.md')
            c['config_hash']=C.G.digest(c);return c
        T.config=config
    from bootstrap94 import main as inherited
    inherited()
    assert C.sources()==sources

if __name__=='__main__':main()
