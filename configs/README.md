# Configuration Design

`runtime.example.yaml` illustrates the intended PatchSprint configuration format. No code consumes it yet, and it is not a configuration that can be passed directly to vLLM.

Manage the model, tokenizer, revisions, precision, length budgets, concurrency, and parallelism separately. Pin revisions and dependencies for formal experiments and validate against actual GPU resources. After switching to a 14B/32B checkpoint, recheck capabilities, memory, stopping rules, and cache state.
