# B112 addendum1: correct exporter membership and reuse allocated runtime

2026-11-25, no B112 scientific fit/inference/readout started. Initial export
failed before writing any rule table: np.isin received a Python set, which
NumPy treats as one object rather than a collection of file IDs. Explicitly
convert each set to its sorted list for membership, including V2 verification.
Preserve the partial export directory as attempt1, never use it as input.
This changes no files, masks, rows, model, target, normalization or judgement
of the scientific plan; the original lists are exactly the intended manifests.

Author specifies the already allocated B111 T4 runtime. Instead of opening
a fresh notebook/runtime, insert the prepared B112 single cell into that
existing B111 notebook and run it there. Keep the completed B111 cell and
/content/ces_b111 outputs intact; B112 uses /content/ces_b112, verifies the
new archive, refuses duplicate B112 launches, and downloads its own result.
No reset,disconnect or new GPU allocation. This changes delivery only.
Record failed export and corrected sources; commit/push before re-export.
