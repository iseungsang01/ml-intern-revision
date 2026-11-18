"""B.67 scoring: seq/eval_seq.py with the variants v2s and v2a registered at run time, then (arm S) the gate
readout on the scored rows.

eval_seq resolves CES_SEQ_MODEL through seq_models.SEQ_MODELS inside build_pred_lookup, the dict that
model_seq_v2gate.register() inserted the variants into, so scoring is eval_seq.compare() unchanged.  Two module
globals of eval_seq (load_grid_files, build_clean_val_subset) are wrapped to keep their return values, which the
gate readout reuses; the wrappers return them unchanged.  For v2s, gates_b67.readout() then writes
b67_gates_test.npz / .json next to comparison_errors_test.npz and stops with an error if the alignment check fails.
Refuses a run whose metrics.json names another architecture.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import model_seq_v2gate as M   # puts seq/ and ces_prediction/ on sys.path
import gates_b67 as G


def main():
    variant = os.environ.get("CES_SEQ_MODEL", "")
    if variant not in M.VARIANTS:
        raise SystemExit(f"FATAL: eval_b67.py scores only {sorted(M.VARIANTS)}, CES_SEQ_MODEL={variant!r}")
    out = Path(os.environ["CES_OUTPUT_DIR"])
    trained = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["seq_model"]
    if trained != variant:
        raise SystemExit(f"FATAL: {out.name} was trained as {trained!r}, CES_SEQ_MODEL={variant!r}")
    table = M.register()
    import eval_seq
    if eval_seq.SEQ_MODELS is not table or variant not in eval_seq.SEQ_MODELS:
        raise SystemExit("FATAL: eval_seq does not see the registry the B.67 variants were inserted into")
    cap = G.install_captures(eval_seq)
    print(f"[b67] eval_seq with {variant} registered", flush=True)
    eval_seq.compare()
    if variant == "v2s":
        summary = G.readout(out, variant, G.device_of_env(), captured=cap, dest=out, strict=True)
        print(f"[b67] gate readout: {summary['rows']} rows, alignment check passes={summary['passes']} "
              f"(identical squared errors {summary['identical_rows']}, p99 |diff| {summary['p99_abs_err_diff_eV']:.3g} eV)",
              flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
