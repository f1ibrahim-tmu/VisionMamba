# CPU / GPU parallelism and Option 2B (OMP_NUM_THREADS)

## How CPU cores, threads, and workers fit together

### Hierarchy on one node (4 GPUs)

```
SLURM allocates:  N CPU cores  (e.g. 32 or 64) for the whole job
                  +
                  4 GPUs

Your training runs:  4 processes  (one per GPU, via --num-gpus 4)
                     |
                     +-- Each process has:
                         • 1 main thread (runs Python, drives training)
                         • OMP_NUM_THREADS threads (OpenMP: linear algebra, etc.)
                         • num_workers child processes (DataLoader workers)
```

- **CPU cores** = physical (or hyperthread) cores SLURM gives you. Everything that runs (main process, OpenMP, DataLoader workers) competes for these.
- **Workers** = separate processes that load and augment data; each typically uses ~1 core when busy.
- **OMP_NUM_THREADS** = how many threads each of the 4 training processes may use for parallel CPU ops (e.g. in PyTorch/NumPy). Total OpenMP threads = 4 × OMP_NUM_THREADS.

### The math that causes trouble

Rough balance (no oversubscription):

  **4 × OMP_NUM_THREADS  +  4 × num_workers  ≤  N (allocated cores)**

Example with **32 cores** and your previous defaults:

- OMP_NUM_THREADS = 32/4 = **8**  →  4×8 = **32** threads for OpenMP
- num_workers = 32/4 = **8** per GPU  →  4×8 = **32** worker processes

So 32 + 32 = **64** “threads/processes” on **32** cores → heavy oversubscription. The OS keeps switching between them; the DataLoader can’t feed the GPU in time → **GPU at 0%** while waiting for the next batch (Option 2B addresses this).

### What OMP_NUM_THREADS does

- **OMP_NUM_THREADS** limits how many threads **each** of the 4 training processes can use for **parallel CPU work** (OpenMP): matrix ops, some NumPy/PyTorch CPU kernels, etc.
- It is **not** the number of DataLoader workers. Workers are separate processes and are set by `dataloader.train.num_workers`.

**How it’s often set:**

- `OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))`  →  one “share” of cores per GPU process.
  - With 32 CPUs: 8 threads per process (4×8 = 32).
  - With 64 CPUs: 16 threads per process (4×16 = 64).

**Option 2B (when you stay on 32 CPUs):**

- Set **OMP_NUM_THREADS=2** (or 3) in the SLURM script.
- Then: 4×2 = **8** cores for OpenMP, so 32 − 8 = **24** cores effectively free for workers.
- With 6 workers per GPU: 4×6 = 24 worker processes → fits in 24 cores, so no oversubscription and GPUs are fed better.

### Summary

| Item              | What it is                         | Who sets it                          |
|-------------------|------------------------------------|--------------------------------------|
| CPU cores         | Total cores for the job            | SLURM `--cpus-per-task`              |
| OMP_NUM_THREADS   | Threads per process for CPU math   | You in the SLURM script               |
| num_workers       | DataLoader processes per GPU       | Training script (WORKERS_PER_GPU)     |
| Total contention  | 4×OMP + 4×workers vs cores         | Keep ≤ allocated cores to avoid stalls|

Option 2B = **lower OMP_NUM_THREADS** so that **workers** get enough cores and the GPU is not starved for data.
