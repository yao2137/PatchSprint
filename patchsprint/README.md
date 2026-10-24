# PatchSprint

Edit-aware speculative decoding for fast LLM code editing (VS Code extension + self-hosted backend).

## Preliminary analysis: how much of an edited file is copied?

`analysis/copy_upper_bound.py` replays greedy speculative decoding on CanItEdit, treating each
reference solution as the model output, and compares proposers by decode-step count.

Assumptions: (1) teacher forcing — the model emits exactly the reference solution; (2) speedup is
counted in forward passes, which matches wall-clock only when verification is memory-bound.

### Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/download_data.sh
.venv/bin/python analysis/copy_upper_bound.py \
  --data data/raw/canitedit/test-00000-of-00001.parquet \
  --tokenizer data/raw/qwen2.5-coder-tokenizer/tokenizer.json \
  --out results/upper_bound
.venv/bin/python analysis/plot_upper_bound.py --results results/upper_bound
.venv/bin/python -m pytest -q tests
```

Outputs: `results/upper_bound/per_problem.csv`, `summary.json`, `fig_copy_fraction.png`,
`fig_speedup_vs_cost.png`.

### Proposers compared

| name | description |
|---|---|
| `ctx_oracle` | Drafts the longest true continuation found anywhere in input + output. Upper bound for copy-based proposers. |
| `align_oracle` | Follows only the monotonic diff alignment. Not an upper bound (misses non-monotonic reuse). |
| `ngram` | vLLM-style prompt lookup: stateless, longest suffix match (n = 2..5), earliest position wins. |
| `pointer` | PatchSprint alignment pointer (COPY / EDIT / RESYNC), source only. |
| `hybrid` | Pointer with n-gram fallback over input + output while resync fails. |

Note: the pointer parameters were chosen while looking at CanItEdit, so final claims must be
measured on held-out tasks.

## Data sources

- CanItEdit (Cassano et al., arXiv:2312.12450), MIT license — https://huggingface.co/datasets/nuprl/CanItEdit
- Qwen2.5-Coder-7B-Instruct tokenizer (Hui et al., arXiv:2409.12186) — https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
