"""Original numerical loop with explicit four-worker dispatch provenance."""
import os,sys
import common96 as C

def main():
    sources,decision=C.policy()
    from adapter94 import activate
    T=activate()
    if len(sys.argv)>1 and sys.argv[1]=='train':
        n=int(os.environ['B94_CONCURRENCY']);i=int(os.environ['B94_WORKER_SLOT']);g=i//2
        assert n in (1,4) and 0<=i<n
        os.environ['B94_CONCURRENCY']=str(min(2,n-2*g));os.environ['B94_WORKER_SLOT']=str(i%2)
        original=T.config
        def config(*args,**kwargs):
            c=original(*args,**kwargs);c.pop('config_hash')
            c.update(global_gpu_worker_limit=n,global_worker_slot=i,local_worker_group=g,
                     scheduled_concurrency_scope='local group; global capacity recorded separately',
                     dispatch_sources96=sources,dispatch_policy96_sha256=decision['decision_sha256'],
                     dispatch_registration='PREREGISTRATION_B96.md')
            c['config_hash']=C.G.digest(c);return c
        T.config=config
    from bootstrap94 import main as inherited
    inherited();assert C.sources()==sources

if __name__=='__main__':main()
