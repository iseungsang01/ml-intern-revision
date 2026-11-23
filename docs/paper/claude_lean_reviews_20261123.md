# Claude API Lean review audit

Official Anthropic API, authorized by the author. Only redacted methodology and public sources were transmitted. No raw measurements, shot IDs, identity, credentials or unpublished predictive scores were sent. Exact responses, including private reasoning blocks, remain in the ignored local cache and are not reproduced here. These records establish API use, not correctness or originality.

Adjudicated findings, source checks, mathematical corrections and the bounded recommendation: [Lean feasibility](lean_feasibility_20261123.md). No Lean proof was compiled and no training method or TEST allocation was changed.

| Review | Returned model | Stop | Input tokens | Output tokens | Seconds | Local record SHA-256 |
|---|---|---|---:|---:|---:|---|
| adversarial_synthesis attempt 1 | `claude-opus-5-5` | `end_turn` | 12636 | 5002 | 55.12 | `c93b2948370ba2aaf146b9eb9ec81a3446cbbd42f1df7e1bd79859d49d085dd7` |
| formal_invariants attempt 1 | `claude-fable-5-2` | `end_turn` | 27570 | 5417 | 60.71 | `3528fccb4b58c87fa6af46dbc41bb089478b643de615b89b9b7eb7a471584247` |
| mathematical_audit attempt 1 | `claude-opus-5-5` | `end_turn` | 5971 | 6975 | 68.74 | `d1375319915677c500042bb792bd205f15c953a659f79714b941f0c08fac548c` |
| physics_value attempt 1 | `claude-opus-5-5` | `end_turn` | 25347 | 4286 | 47.56 | `bdf3b73188706955382be9a8e7260e4fa68f071390ed91fa7e64f3f8ecd4ff98` |
| staleness_theory attempt 1 | `claude-fable-5-2` | `end_turn` | 26926 | 7322 | 79.40 | `6c9608c2d9a3de79cd83769de1e00a2d5fa59844fbe62163bb1a61cf600dffc0` |
| torchlean_bridge attempt 1 | `claude-fable-5-2` | `end_turn` | 26313 | 6919 | 74.25 | `337a6dd47a03d5119fc6106cfdc885ad60efd94d84a6b87d234c9fe539c3e061` |

Total: 6 paid attempts; 124763 input tokens; 35921 output tokens; 8 server web searches. Output includes billed thinking tokens where reported. No price estimate is inferred.

Three initial reviews covered formal invariants, staleness theory and physical/control value. A fresh adversarial synthesis challenged those proposals. Two focused follow-ups checked the TorchLean implementation bridge and disputed mathematics. The final audit explicitly corrected earlier reviews; agreement among API responses was never treated as proof.

Prompts and credential-safe API orchestration are in `ces_prediction/research/claude_lean_scout_20261123.py`, `claude_lean_bridge_20261123.py` and `claude_lean_math_audit_20261123.py`. Regenerate this metadata without an API request by importing the scout module and calling `render()`. Its `main()` performs the paid research workflow and reuses exact completed prompt/model caches.
