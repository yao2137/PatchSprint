# Experiment Log and Reproducibility

No model or GPU experiments have run. This file defines the recording process; it is not a results report.

## Suggested Initial Experiments

1. Ordinary greedy decoding: can the model complete representative single-file edits?
2. Ordinary decoding versus fixed-candidate verification: token equivalence and cache state.
3. Generic prompt lookup versus source-aware proposals: quality and runtime costs.
4. Fixed versus adaptive proposal length: improvements, regressions, and ablations.
5. Full-file versus diff generation: final task quality and end-to-end latency.

## Required Experiment Metadata

- Experiment ID, date, code commit, and uncommitted changes that affect results.
- GPU/CPU, memory, interconnects, driver, CUDA, PyTorch, backend, and dependency versions.
- Model and tokenizer IDs/revisions, precision, quantization, template, and stopping rules.
- Task-set version, split, task IDs, input/output lengths, parameters, and randomization conditions.
- Configuration, actual commands, warmup, compilation, repetitions, and cold/warm cache policy.
- Baselines, resource budgets, result locations, and failure/cancellation/timeout counts.
- Conclusions, evidence, limitations, and next steps rather than only the best speedup.

## Result Storage

Use `results/<experiment-id>/` for shareable configuration snapshots, metric tables, analysis, and small figures. Store models, caches, and large raw profiler artifacts in ignored `raw/` directories or HPC storage. Record how to retrieve or reproduce them.

Do not link nonexistent result files or cite them as evidence. Separate profiling overhead from formal measurement and kernel microbenchmarks from end-to-end results.

## Registry

| Experiment ID | Status | Code/task version | Configuration and result location | Conclusion |
| --- | --- | --- | --- | --- |

## Experiment Record Template

```text
Experiment ID:
Goal/hypothesis:
Code and environment:
Model/data/configuration:
Execution command:
Warmup and measurement method:
Baselines and fairness:
Correctness results:
Performance results:
Failures and anomalies:
Raw evidence location:
Conclusions and limitations:
```
