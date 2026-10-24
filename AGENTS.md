# PatchSprint Working Agreements

## Reading Order

- Read `docs/STATUS.md` at the start of a task and inspect the actual implementation state.
- When first learning the project or discussing goals, architecture, models, or evaluation, read the relevant sections of `docs/PROJECT_CONTEXT.md`.
- Consult `docs/DECISIONS.md` for previous tradeoffs and `docs/EXPERIMENTS.md` for experiment work.
- Read code relevant to the task. Do not reload the entire long context document for every small change.

## Project Goals

- This is a solo AI Infra portfolio project focused on code-editing inference engines and serving.
- Prioritize technical depth, correctness, interpretable performance evidence, and useful product behavior. Learn through verifiable milestones.
- Model scale and HPC access require verification. Do not assume that B200 GPUs, multiple GPUs, or any speedup are available.

## Language

- Write all repository text in English, including documentation, comments, examples, templates, and user-facing messages.
- The sole exception is `docs/PROJECT_CONTEXT.md`, which must remain unchanged unless the user explicitly requests edits to it.
- Apply the English requirement to future additions as well as existing files.

## Implementation and Verification

- Distinguish implemented work, verified results, design proposals, hypotheses, and experiments that have not run.
- When changing generation, verify token alignment, position/cache state, acceptance and rejection, EOS, and length boundaries.
- Token equivalence and functional editing correctness are separate checks. Report them separately.
- A proposer only supplies candidates. Commit final tokens according to the target decoding rules.
- Isolate request state by request_id rather than unstable batch row indices.
- Profile before selecting GPU optimizations. Report kernel-level and end-to-end effects separately.
- Compare against strong baselines. Do not use a slow reference implementation as the only performance baseline or silently exclude failures.
- Distinguish inference deployment, fine-tuning, and pretraining. Rebuild incompatible state after checkpoint changes.
- Installation, runtime, and test commands do not exist yet. Do not fabricate executed commands, GPU results, or passing tests. Add verified commands to README when implemented.

## Documentation Maintenance

- Update `docs/STATUS.md` after meaningful milestones using actual evidence.
- Record important tradeoffs in `docs/DECISIONS.md`. Mark new approaches as proposals until adopted.
- Register experiment configurations, logs, and results in `docs/EXPERIMENTS.md` with traceable evidence.
- Follow `docs/AI_USAGE.md` when preserving original AI prompts. Summaries do not replace complete records.
- Keep the preserved project context separate from changing implementation status.
- Do not create remote repositories, publish, push, or perform external writes unless requested by the user.
