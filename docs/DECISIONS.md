# Decision Log

Record the date, status, evidence, and reconsideration conditions for each decision. Keep proposals separate from user-confirmed goals.

## D001: Project Direction

- Date: 2026-10-16.
- Status: confirmed by the user.
- Decision: build a PatchSprint backend and a lightweight VS Code editing extension, focused on code editing and AI Infra inference.
- Evidence: the user explicitly selected this project and requested technical depth, practical value, and an effective interview demonstration.
- Constraints: solo development; quality first; learn PyTorch and CUDA through implementation without lowering the final standard because of current experience.

## D002: Separate Context, Guidance, and Progress

- Date: 2026-10-16.
- Status: organization authorized by the user.
- Decision: keep full background in `PROJECT_CONTEXT.md`, concise working guidance in root `AGENTS.md`, and actual progress in `STATUS.md`.
- Rationale: avoid mixing long-term goals with changing implementation status or loading the full context for every task.
- Maintenance: record important tradeoffs here and experiments in `EXPERIMENTS.md`.

## D003: Initial Implementation Boundary

- Date: 2026-10-16.
- Status: design proposal; not validated through implementation.
- Proposal: Python single-file editing, explicit user requests, full-file output, and greedy decoding. Establish ordinary decoding and verification correctness first.
- Pending: model editing capability, backend versions, resources, output format, and further user direction.
- Evaluation requirement: retain diff generation as an alternative. Product design must not exclude a strong baseline in advance.

## D004: Configurable Model Scale

- Date: 2026-10-16.
- Status: design requirement; specific checkpoints and configurations remain unverified.
- Decision: support existing checkpoints and expand from 7B to 14B/32B or other sizes as resources allow.
- Constraints: changing a parameter-count setting does not turn an existing checkpoint into a larger model. Lock weights for each request and recheck state and quality after a switch.
- Scope: pretraining a foundation model from scratch is not the default project direction.

## D005: English-Only Repository

- Date: 2026-10-16.
- Status: explicitly required by the user.
- Decision: all repository text must be in English, including future documentation, comments, examples, and product messages.
- Sole exception: preserve the existing `docs/PROJECT_CONTEXT.md` background prompt unchanged.
- Maintenance: enforce this rule through `AGENTS.md` and check edited files for untranslated content.

## New Decision Template

```text
ID and title:
Date:
Status: proposed / accepted / superseded
Problem:
Options considered:
Decision and rationale:
Evidence (code, experiment, or user instruction):
Costs and limitations:
Reconsideration conditions:
```
