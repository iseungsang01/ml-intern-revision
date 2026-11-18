"""B.68 training: seq/train_seq.py with the auxiliary head and loss installed at run time (aux_b68.install_train).

Launched by run_b68.py instead of seq/train_seq.py, under the env of run_b39.pooled_env("v2", r, k, out).  After
train_seq.main() has saved its checkpoint, the checkpoint is loaded into plain seq_v2 (seq_models.SEQ_MODELS["v2"])
with strict loading; then b68_aux.json is written next to metrics.json (the auxiliary weight, the cache it used, the
share of training and validation rows with an auxiliary target, the per-epoch main and auxiliary losses).  The
runner counts the training as done only when b68_aux.json exists.  No shared file is edited.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import aux_b68 as A   # puts seq/ and ces_prediction/ on sys.path
import target_b68 as T


def coverage(counts, names):
    c = [sum(counts.get(n, [0, 0, 0, 0])[j] for n in names) for j in range(4)]
    return {"discharges": sum(n in counts for n in names), "rows": c[0], "rows_with_aux_target": c[1],
            "share_rows_with_aux_target": c[1] / max(c[0], 1), "ti_measured_rows": c[2],
            "ti_measured_rows_with_aux_target": c[3], "share_ti_measured_rows_with_aux_target": c[3] / max(c[2], 1)}


def main():
    A.check_env()
    train_seq, state = A.install_train(A.AUX_WEIGHT)
    print(f"[b68] train_seq with the auxiliary head (weight {A.AUX_WEIGHT}) on the offline-GP target "
          f"({T.CACHE_NPZ.name}, built {state['cache_meta']['built']})", flush=True)
    train_seq.main()
    out = Path(os.environ["CES_OUTPUT_DIR"])
    import torch
    import seq_models
    sd = torch.load(out / "weights" / "seq_lstm.pth", map_location="cpu")
    plain = seq_models.SEQ_MODELS["v2"]()
    plain.load_state_dict(sd, strict=True)            # raises unless the checkpoint is exactly seq_v2
    if any(k.startswith(A.AUX_PREFIX) for k in sd):
        raise SystemExit("FATAL: the saved checkpoint carries the auxiliary head")
    man = json.loads((Path(os.environ["CES_SPLIT_DIR"]) / "split_manifest.json").read_text(encoding="utf-8"))
    tr, va = list(man["train_files"]), list(man["val_files"])
    mf = int(os.getenv("CES_SEQ_MAX_FILES", "0"))
    if mf:
        tr, va = tr[:mf], va[:max(2, mf // 4)]
    probe = A.SeqCESLSTMv2Aux()
    rec = {"registration": "ces_prediction/experiments/PREREGISTRATION_B68.md", "aux_weight": A.AUX_WEIGHT,
           "aux_head": "Linear(160->64), GELU, Linear(64->1) on the T_i LSTM's LayerNorm output (same form as head_ti)",
           "aux_head_parameters": probe.n_params_aux, "seq_v2_parameters": probe.n_params,
           "checkpoint": "plain seq_v2 (auxiliary head not saved); loaded into SEQ_MODELS['v2'] with strict=True after saving",
           "cache": str(T.CACHE_NPZ.relative_to(T.REPO)), "cache_built": state["cache_meta"]["built"],
           "cache_source_sha256": state["cache_meta"]["source_sha256"], "estimator": state["cache_meta"]["estimator"],
           "grid_calls": state["grid_calls"], "block_calls": state["block_calls"],
           "train": coverage(state["counts"], tr), "val": coverage(state["counts"], va), "passes": state["epochs"]}
    (out / "b68_aux.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"[b68] checkpoint loads as plain seq_v2 (strict); training rows with an auxiliary target "
          f"{rec['train']['share_rows_with_aux_target']:.4f}; wrote b68_aux.json", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)   # as train_seq: skip the Windows CUDA teardown so the return code reflects the run
