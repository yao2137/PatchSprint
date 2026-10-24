# GPU Implementations

No CUDA/Triton kernels exist yet. Profile first, then select a measured bottleneck in the main execution path.

Each production case needs explicit inputs and outputs, a PyTorch or existing-implementation baseline, correctness checks, microbenchmarks with real shapes, and an end-to-end impact analysis. Label learning exercises separately from production contributions.
