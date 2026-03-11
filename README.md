# FAISS Benchmark

Config-driven benchmarking for dense retrieval with:
- **Torch exact chunked top-k** (single or multi-GPU).
- **FAISS GPU FlatIP** (exact, single or multi-GPU).
- **FAISS GPU IVFPQ** (approximate, single GPU in this script).

It writes a JSON report that includes per-run metrics plus **time and FLOPs matrices**.

## 1) Conda environment (fair benchmarking setup)

```bash
conda create -n faiss-bench python=3.10 -y
conda activate faiss-bench

# Install PyTorch that matches your CUDA version from pytorch.org instructions.
# Example (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# FAISS GPU build (recommended via conda-forge)
conda install -c conda-forge faiss-gpu -y

pip install numpy
```

### Fair benchmarking checklist
- Keep driver/CUDA/PyTorch/FAISS versions fixed across runs.
- Use the same `seed`, `warmup`, and `iters`.
- Avoid other GPU workloads during benchmarks.
- For multi-GPU, pin the same visible devices each run:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

## 2) Configure benchmarks

Edit `benchmark_config.json` to define global defaults and one or more experiments.

```json
{
  "global": {
    "dtype": "fp16",
    "k": 10,
    "chunk": 250000,
    "seed": 123,
    "normalize": true,
    "warmup": 1,
    "iters": 3,
    "gpu_mode": "single",
    "faiss": { "enabled": true, "index": "flat" }
  },
  "experiments": [
    {"name": "single", "n": 500000, "d": 768, "q": 64, "gpu_mode": "single"},
    {"name": "multi", "n": 1000000, "d": 1024, "q": 128, "gpu_mode": "multi"}
  ]
}
```

In each experiment:
- `n`: number of database vectors to index/search against.
- `d`: embedding dimensionality (number of features per vector).
- `q`: number of query vectors issued in that benchmark run (query batch size).

## 3) Run

### Config mode (recommended)
```bash
python bench_retrieval_gpu.py --config benchmark_config.json --output_json benchmark_results.json
```

### CLI mode (single experiment)
```bash
python bench_retrieval_gpu.py --name cli_run --n 1000000 --d 1024 --q 64 --k 10 --gpu_mode single --faiss --faiss_index flat
```

## 4) Output JSON

`benchmark_results.json` contains:
- `results`: detailed metrics for each experiment.
- `matrices.time_ms_matrix`: runtime matrix by backend.
- `matrices.flops_matrix`: computed/estimated FLOPs matrix by backend.

Notes:
- Torch and FAISS Flat use exact inner-product search FLOPs: `2 * N * D * Q`.
- IVFPQ FLOPs are an estimate based on scanned candidates using `nlist`/`nprobe`.
