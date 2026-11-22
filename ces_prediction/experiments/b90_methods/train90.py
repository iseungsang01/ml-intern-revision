"""One supervised B90 training child."""
import argparse
import common90 as M
from adapter90 import activate,dependency_ready,input_ready
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=M.METHODS);p.add_argument('--unit',required=True)
    p.add_argument('--init',required=True,type=int,choices=M.INITS);p.add_argument('--variant',required=True,choices=M.VARIANTS);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert dependency_ready();input_ready()
    assert a.unit in {u['name'] for u in M.select_units()}
    if a.init!=1701:assert a.method in M.promotion()['promoted'],'Unpromoted additional seed'
    activate(a.method).train_unit(a.unit,a.init,a.variant,smoke=a.smoke)
if __name__=='__main__':main()
