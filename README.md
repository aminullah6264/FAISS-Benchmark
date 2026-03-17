# FAISS Benchmark

This repository provides scripts to:
1. Generate synthetic embedding datasets on disk (`.dat` + `meta.json`).
2. Run nearest-neighbor benchmark experiments (Torch / FAISS modes supported by the benchmark script).
3. Save benchmark outputs to a JSON report.

## 1) Generate a dataset

Use `generate_embedding_dataset.py` to create a memmap-friendly dataset directory.

### Example commands

```bash
python generate_embedding_dataset.py --n 5000000 --d 768  --output /mnt/projects/tool-control/results/embeddings_5M_768
python generate_embedding_dataset.py --n 5000000 --d 2048 --output /mnt/projects/tool-control/results/embeddings_5M_2048
python generate_embedding_dataset.py --n 5000000 --d 2304 --output /mnt/projects/tool-control/results/embeddings_5M_2304
python generate_embedding_dataset.py --n 5000000 --d 4096 --output /mnt/projects/tool-control/results/embeddings_5M_4096
```

### Common options

- `--n`: number of bank vectors.
- `--d`: embedding dimension.
- `--q`: number of query vectors (default `64`).
- `--output`: output dataset directory.
- `--seed`: random seed.
- `--dtype`: `fp32` (default) or `fp16`.
- `--chunk_size`: rows generated per chunk.
- `--no_normalize`: disable L2 normalization.

### Generated files

Each dataset directory contains:

- `bank.dat` — embedding matrix `(N, D)`.
- `queries.dat` — query matrix `(Q, D)`.
- `ids.dat` — int64 IDs `(N,)`.
- `meta.json` — metadata for loading shapes/dtypes.

---

## 2) Run benchmark experiments

Benchmarks are configured by JSON (see `benchmark_config.json`).

### Run command

```bash
python benchmark_nearest_search.py --config benchmark_config.json
```

This writes the output report to the `output_json` path in config (currently `benchmark_results.json`).

### Config structure (summary)

- Global settings: `metric`, `top_k`, `precisions`, `gpu_modes`, `warmup_runs`, `timed_runs`, batch sizes, `frameworks`.
- `experiments`: list of datasets (name + `dataset_path`) with optional per-experiment overrides.

---

## 3) Results table (from `benchmark_results.json`)

The table below is generated from the current checked-in results JSON.

| Experiment | Framework | Precision | GPU Mode | Batch Size | Avg ms/query | Total timed s | Peak GPU Δ MiB (per GPU) |
|---|---|---|---|---|---|---|---|
| dataset_1_768d | torch | fp32 | single | 64 | 0.541331 | 0.346452 | 15887 |
| dataset_1_768d | torch | fp32 | multi | 64 | 0.143428 | 0.091794 | 3970, 3978, 3978, 3978 |
| dataset_1_768d | torch | fp16 | single | 64 | 0.156565 | 0.100202 | 7944 |
| dataset_1_768d | torch | fp16 | multi | 64 | 0.050178 | 0.032114 | 1986, 1986, 1986, 1986 |
| dataset_2_2048d | torch | fp32 | single | 64 | 1.205408 | 0.771461 | 40293 |
| dataset_2_2048d | torch | fp32 | multi | 64 | 0.323039 | 0.206745 | 10073, 10075, 10075, 10075 |
| dataset_2_2048d | torch | fp16 | single | 64 | 0.284677 | 0.182194 | 20151 |
| dataset_2_2048d | torch | fp16 | multi | 64 | 0.082708 | 0.052933 | 5038, 5038, 5038, 5038 |
| dataset_2_2304d | torch | fp32 | single | 64 | 1.332354 | 0.852706 | 45177 |
| dataset_2_2304d | torch | fp32 | multi | 64 | 0.356888 | 0.228408 | 11294, 11295, 11295, 11295 |
| dataset_2_2304d | torch | fp16 | single | 64 | 0.309122 | 0.197838 | 22593 |
| dataset_2_2304d | torch | fp16 | multi | 64 | 0.088 | 0.05632 | 5648, 5648, 5648, 5648 |
| dataset_3_4096d | torch | fp32 | single | 64 | 2.278027 | 2.915874 | 79357 |
| dataset_3_4096d | torch | fp32 | multi | 64 | 0.59307 | 0.75913 | 19840, 19841, 19841, 19841 |
| dataset_3_4096d | torch | fp16 | single | 64 | 0.478078 | 0.61194 | 39683 |
| dataset_3_4096d | torch | fp16 | multi | 64 | 0.131619 | 0.168472 | 9921, 9921, 9921, 9921 |

If you regenerate benchmarks, re-run the benchmark script and update this table from the new JSON.
