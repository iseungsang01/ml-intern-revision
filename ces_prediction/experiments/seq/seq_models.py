"""The single registry of named seq-family architectures.

`train_seq.py` and `eval_seq.py` both resolve `CES_SEQ_MODEL` through this table. They used
to carry their own copies, which is exactly the failure mode it looks like: a variant
registered for training trains fine and then dies at eval with "must be one of [...]".
Add a variant here and both sides see it.

Variants are named rather than env-configured on purpose -- a checkpoint must always reload
under the architecture it was trained with.
"""

import functools

from model_seq import SeqCESLSTM
from model_seq_v2 import SeqCESLSTMv2
from model_seq_v3 import SeqCESLSTMv3
from model_seq_b3 import SeqCESB3
from model_seq_tcn import SeqCESTCN
from model_seq_xfmr import SeqCESXfmr

SEQ_MODELS = {
    "v1": SeqCESLSTM,
    "v2": SeqCESLSTMv2,
    "v3": SeqCESLSTMv3,

    # B.3 interpretable rung: the ONE explored variable is the T_i latent width
    # (V_rot latent fixed at 4).
    "b3k4": functools.partial(SeqCESB3, latent_ti=4),
    "b3k6": functools.partial(SeqCESB3, latent_ti=6),
    "b3k8": functools.partial(SeqCESB3, latent_ti=8),

    # B.4 width ladder: seq_v2 with ONLY the T_i encoder width varied (V_rot branch
    # and heads fixed). "v2" itself is the 160-unit point.
    "v2w24": functools.partial(SeqCESLSTMv2, hidden_ti=24),
    "v2w40": functools.partial(SeqCESLSTMv2, hidden_ti=40),
    "v2w80": functools.partial(SeqCESLSTMv2, hidden_ti=80),
    "v2w260": functools.partial(SeqCESLSTMv2, hidden_ti=260),

    # B.8 minimal ladder: shrink BOTH branches and the heads, not just hidden_ti.
    # B.4 bottomed out near 34k params because the V_rot branch (18,688) and the heads
    # were held fixed -- that floor was set by the parts never varied, not by the problem.
    "v2m12k": functools.partial(SeqCESLSTMv2, hidden_ti=24, layers_ti=1, hidden_vt=32, head=32),
    "v2m7k": functools.partial(SeqCESLSTMv2, hidden_ti=16, layers_ti=1, hidden_vt=24, head=24),
    "v2m4k": functools.partial(SeqCESLSTMv2, hidden_ti=12, layers_ti=1, hidden_vt=16, head=16),
    "v2m2k": functools.partial(SeqCESLSTMv2, hidden_ti=8, layers_ti=1, hidden_vt=12, head=12),

    # Same for the interpretable rung: the latent count is NOT what costs parameters
    # (k8 -> k4 saves 264 of 21,498); the GRU hidden sizes are.
    "b3m7k": functools.partial(SeqCESB3, hidden_ti=32, hidden_vt=16, latent_ti=4, latent_vt=2),
    "b3m2k": functools.partial(SeqCESB3, hidden_ti=16, hidden_vt=8, latent_ti=3, latent_vt=2),
    "b3m1k": functools.partial(SeqCESB3, hidden_ti=8, hidden_vt=8, latent_ti=2, latent_vt=1),

    # B.9 axis B family comparison. Receptive fields land on the reach-ladder rungs
    # (2^(L+1)-1 for the TCN, the attention band for the transformer) so each arm
    # pairs against a v2 rung trained at the same reach, not against an interpolation.
    "tcn15": functools.partial(SeqCESTCN, layers=3),
    "tcn63": functools.partial(SeqCESTCN, layers=5),
    "xfmr63": functools.partial(SeqCESXfmr, reach=63),
    # Reach 15 is where axis A says skill saturates, and attention pays O(band) per
    # step -- so this is the arm that asks whether the transformer misses 1 ms because
    # of attention or because it was only ever built at reach 63.
    "xfmr15": functools.partial(SeqCESXfmr, reach=15),

    # B.9 axis D: the 1k-10k band, which b8_minimal swept with recurrent arms only.
    # Reach is fixed at 15 (>= the 7-step saturation measured in axis A) so the only
    # variable against the LSTM rungs is the operator, and the question is whether the
    # family tie of axis B survives where capacity is scarce enough to bite.
    "tcn8k": functools.partial(SeqCESTCN, layers=3, hidden_ti=24, hidden_vt=12, head=16),
    "tcn3k": functools.partial(SeqCESTCN, layers=3, hidden_ti=14, hidden_vt=8, head=12),
    "tcn2k": functools.partial(SeqCESTCN, layers=3, hidden_ti=10, hidden_vt=6, head=8),
}


def resolve(variant):
    """Name -> model factory, with the error message both callers used to duplicate."""
    if variant not in SEQ_MODELS:
        raise SystemExit(
            "CES_SEQ_MODEL must be one of " + str(sorted(SEQ_MODELS)) + ", got " + repr(variant))
    return SEQ_MODELS[variant]
