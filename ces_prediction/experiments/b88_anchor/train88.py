"""New CLI; the actual optimizer, mixture and full training loop remain frozen."""
import argparse
import common88 as M
from adapter88 import activate,dependency_ready,input_ready
def main():
    p=argparse.ArgumentParser();p.add_argument('--unit',required=True);p.add_argument('--init',required=True,type=int,choices=M.INITS)
    p.add_argument('--variant',required=True,choices=M.VARIANTS);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    M.assert_frozen();assert dependency_ready(),'B85 dependency not complete';input_ready()
    assert a.unit in {u['name'] for u in M.select_units()}
    activate().train_unit(a.unit,a.init,a.variant,smoke=a.smoke)
if __name__=='__main__':main()
