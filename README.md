# PatchSprint

An edit-aware speculative decoding backend with a lightweight VS Code editing extension.

PatchSprint helps developers modify existing code using natural-language instructions. It proposes candidate tokens from the original source, verifies them with the target model, and presents an accept-or-reject diff in VS Code.

**Current stage: the repository scaffold and design documentation are in place. The inference engine, service, extension, GPU kernels, and experimental results have not been implemented or validated.**

## Start Here

- [Project status](docs/STATUS.md): completed work, open questions, and next steps.
- [Full project context](docs/PROJECT_CONTEXT.md): goals, algorithms, architecture, quality standards, and AI collaboration guidance. This background prompt is intentionally preserved in its original language.
- [Decision log](docs/DECISIONS.md): confirmed decisions, rationale, and pending proposals.
- [Experiment log](docs/EXPERIMENTS.md): experiment design, baselines, and reproducibility requirements.
- [AI usage log](docs/AI_USAGE.md): original prompt records and the course documentation requirement.
- [Repository guidance](AGENTS.md): reading order and working agreements for AI contributors.

## Product and Technical Focus

User workflow: open a Python file -> enter an editing instruction -> generate a result -> preview the diff -> accept or reject.

Technical progression: ordinary decoding and KV state -> speculative verification and rollback -> an edit-aware proposer -> profiling and relevant GPU optimization -> integration with an established backend -> reproducible evaluation.

Model scale is configurable through existing checkpoints, such as 7B, 14B, or 32B. Deploying a larger inference model generally does not require retraining. Memory requirements, compatibility, and editing quality must be validated again after a change.

## Repository Layout

```text
engine/                Reference decoding, proposal strategies, and backend integrations
kernels/               GPU implementations targeting measured bottlenecks
server/                Editing API, request lifecycle, and cancellation
vscode-extension/      Instructions, diffs, acceptance, and document version checks
benchmarks/            Editing tasks and evaluation harness
configs/               Model, runtime, and decoding configuration design
scripts/               Development utilities and SLURM job entry points
results/               Real experiment summaries and figures suitable for sharing
tests/                 Unit, decoding correctness, and integration checks
docs/                  Project context, current status, and supporting records
```

## Working on the Project

Use the following instruction when starting a new chat:

> Read AGENTS.md and docs/STATUS.md first. Read the sections of docs/PROJECT_CONTEXT.md relevant to this task. Distinguish implemented work, verified results, and plans. The goal of this session is: ...

There are no executable installation, inference, test, or extension launch commands yet. Add dependencies and commands after a minimal implementation has been verified. A directory is not evidence that its functionality exists.

## Development and Reproducibility

- Establish correctness with a single GPU, a fixed model, and greedy decoding first.
- Make performance comparisons under explicit, comparable hardware, weights, precision, length limits, and cache conditions.
- Compare against existing prompt lookup and alternatives such as generating a diff. Report local and end-to-end improvements separately.
- Do not present plans, simulated data, or experiments that have not run as results.
- Keep model weights, environments, credentials, and large raw profiler artifacts out of Git.
- Use English for repository text. The only language exception is the preserved `docs/PROJECT_CONTEXT.md` prompt.

A code license has not been selected. Choose a license and document third-party sources before public distribution.
