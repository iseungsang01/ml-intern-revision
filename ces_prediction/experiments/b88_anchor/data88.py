"""Claude-origin data adapter; corrected direct visible-measurement gathering."""
import numpy as np
import common88 as M
import data85 as _d85
unit_manifest=_d85.unit_manifest
fit_stats=_d85.fit_stats
anchors=_d85.anchors
def build_segments(file_ids,stats,rule,G=None,R=None):
    if G is None or R is None:
        g,r=M.load_sources();G=g if G is None else G;R=r if R is None else R
    base=_d85.build_segments(file_ids,stats,rule,G=G,R=R);anc=anchors(rule,G=G)
    # B85 subtracts Python-list statistics, yielding float64 intermediates before
    # assigning its float32 input. Preserve that exact arithmetic for anchors.
    mu=np.asarray(stats['fast_mu']);sd=np.asarray(stats['fast_sd'])
    out=[]
    for s in base:
        gi=np.asarray(s['gi'],dtype=np.int64);a=anc[gi];has=a>=0
        assert np.array_equal(has,s['x'][:,18].astype(bool))
        assert (a[has]<gi[has]).all()
        assert np.array_equal(G['seg'][a[has]],G['seg'][gi[has]])
        assert np.array_equal(G['fidx'][a[has]],G['fidx'][gi[has]])
        feat=np.zeros((len(gi),13),dtype=np.float32)
        feat[has]=(G['fast'][a[has]].astype(np.float32)-mu)/sd
        source_rows=np.searchsorted(gi,a[has])
        assert np.array_equal(gi[source_rows],a[has])
        assert np.array_equal(feat[has],s['x'][source_rows,:13]),'Anchor/current normalization differs'
        assert np.isfinite(feat).all(),'Nonfinite anchor input'
        row=dict(s);row['x']=np.concatenate((s['x'],feat),axis=1)
        assert row['x'].shape==(len(gi),35);out.append(row)
    return out
