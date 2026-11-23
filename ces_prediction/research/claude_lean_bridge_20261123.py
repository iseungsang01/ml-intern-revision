"""Additional targeted official-API audit of a newly found TorchLean bridge."""
import json
import claude_lean_scout_20261123 as S

PROMPT=S.COMMON+'''
This is an additional operator/refinement specialist review, not a fourth general theory essay.
Primary sources newly found by the working researcher:
https://arxiv.org/abs/2602.22631 (TorchLean v2, May2026)
https://github.com/lean-dojo/TorchLean
https://lean-dojo.github.io/TorchLean/blueprint/Verification-and-Certificates/Neural-Network-Verification/
https://arxiv.org/abs/2610.24548 (Lean/Velvet + VerCors for generated GPU kernels)
https://github.com/lecopivo/SciLean
https://proceedings.mlr.press/v70/selsam17a/selsam17a.pdf

Use your two searches primarily to inspect TorchLean's ACTUAL documented operator/export/trust boundaries; do not infer support for our graph from a broad abstract. Does it cover a PyTorch trained float32 causal Conv1d(kernel3,dilations1..32), explicit left pad, GELU, per-time channel LayerNorm, residual add, frozen Linear heads, buffer-based streaming and train-only preprocessing? Classify each support claim as verified, unsupported or not checked. Clarify that an accepted external certificate can prove a stated real-network property without automatically certifying the LibTorch/CUDA run; identify explicit FFI/rounding/GELU transcendental assumptions.

Compare three concrete paths: a pure Lean latest-visible selector/noninterference specification; a small supported exported model certificate; and porting the full TCN. Give a bounded go/no-go operator checklist and actual artifact needed to justify a claim. No GPU benchmark/training changes; no theorem or source API should be called checked without compilation. Is the existing research project a practical proof-checking target today or an expensive rewrite? Distinguish a formal-methods contribution from the existing plasma prediction contribution. End in Korean under900words with your single preferred path, a stop condition, and exact verified URLs. If the sources cannot be checked, say so.
'''

if __name__=='__main__':
    catalog=json.loads((S.CACHE/'model_catalog.json').read_text());assert 'claude-fable-5-2' in catalog['available_relevant']
    r=S.call('torchlean_bridge',PROMPT,'claude-fable-5-2',S.API.credential())
    print(json.dumps({'review':'torchlean_bridge','returned_model':r['model'],'stop_reason':r['stop_reason']}))
