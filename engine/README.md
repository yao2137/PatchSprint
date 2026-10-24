# Inference Engine

This directory currently defines responsibilities only. No engine implementation exists.

- `reference/`: ordinary greedy generation, KV tracing, minimal speculative verification, and a correctness reference.
- `proposers/`: source matching, alignment tracking, and proposal-length control.
- `integrations/`: integration with pinned inference backends, such as a validated vLLM interface.

Establish token, position, stopping-rule, and cache correctness before optimization. Report reference-path and production-backend performance separately.
