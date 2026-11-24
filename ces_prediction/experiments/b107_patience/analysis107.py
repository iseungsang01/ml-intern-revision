"""Reuse frozen B106 inference/calibration and separate statistical auditor."""
import argparse,importlib
import common107 as C
def main(action):
    C.guard();C.inputs_guard()
    if action=='fits':
        fits={case:C.verify_run(case) for case in C.CASES};assert all(v['patience_met'] for v in fits.values())
        C.save(C.CACHE/'fit_checks.json',{'all_pass':True,'sources':C.sources(),'fits':fits});return
    module,part={'held':('predict106','held'),'test':('predict106','test'),'correction_fit':('correction106','fit'),
                 'correction_test':('correction106','test'),'readout':('readout106',None),'audit':('audit106',None)}[action]
    x=importlib.import_module(module);x.C=C
    x.main() if part is None else x.main(part)
    if action=='readout':
        p=C.CACHE/'readout.json';r=C.load(p);r['registration']='PREREGISTRATION_B107.md'
        r['limits'].extend(['All selected paths met six-epoch validation patience; this is not proof of global optimization.',
             'Continued paths restore optimizer/scheduler/history; original RNG was not saved, so the declared resumed dropout stream differs from uninterrupted execution.'])
        r['selected_paths']=C.load(C.CACHE/'selected.json');C.save(p,r)
        md=C.CACHE/'readout.md';text=md.read_text(encoding='utf8').replace('# B106:','# B107:')
        md.write_text(text+'\n'+ '\n'.join(r['limits'][-2:])+'\n',encoding='utf8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('fits','held','test','correction_fit','correction_test','readout','audit'));main(p.parse_args().action)
