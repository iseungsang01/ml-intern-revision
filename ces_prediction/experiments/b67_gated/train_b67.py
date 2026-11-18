"""B.67 training: seq/train_seq.py with the variants v2s and v2a registered at run time.

Launched by run_b67.py instead of seq/train_seq.py.  model_seq_v2gate.register() inserts the two variants into
seq_models.SEQ_MODELS; train_seq imports that dict by name (`from seq_models import SEQ_MODELS`) and resolves
CES_SEQ_MODEL through it inside main(), so the insertion is seen.  Nothing else of train_seq is touched.
No shared file is edited.
"""
from __future__ import annotations

import os

import model_seq_v2gate as M   # puts seq/ and ces_prediction/ on sys.path


def main():
    variant = os.environ.get("CES_SEQ_MODEL", "")
    if variant not in M.VARIANTS:
        raise SystemExit(f"FATAL: train_b67.py trains only {sorted(M.VARIANTS)}, CES_SEQ_MODEL={variant!r}")
    table = M.register()
    import train_seq
    if train_seq.SEQ_MODELS is not table or variant not in train_seq.SEQ_MODELS:
        raise SystemExit("FATAL: train_seq does not see the registry the B.67 variants were inserted into")
    print(f"[b67] train_seq with {variant} registered (seq_v2 + per-row source gates, mode "
          f"{M.VARIANTS[variant].keywords['mode']})", flush=True)
    train_seq.main()


if __name__ == "__main__":
    main()
    os._exit(0)   # as train_seq: skip the Windows CUDA teardown so the return code reflects the run
