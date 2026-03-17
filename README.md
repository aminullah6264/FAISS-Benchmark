# FAISS Benchmark

This repository provides scripts to:
1. Generate synthetic embedding datasets on disk (`.dat` + `meta.json`).
2. Create a FAISS-native dataset with a prebuilt serialized index (`index.faiss` + queries).
3. Run nearest-neighbor benchmark experiments (Torch / FAISS modes supported by the benchmark script).
4. Save benchmark outputs to a JSON report.

## 1) Generate a memmap dataset

Use `generate_embedding_dataset.py` to create a memmap-friendly dataset directory.

```bash
python generate_embedding_dataset.py --n 5000000 --d 768  --output /mnt/projects/tool-control/results/embeddings_5M_768
```

## 2) Generate a FAISS-native dataset (search-only benchmark input)

Use `create_faiss_dataset.py` when you want benchmark runs that only measure search latency on an already-built index.

```bash
python create_faiss_dataset.py \
  --n 5000000 \
  --d 768 \
  --q 64 \
  --metric ip \
  --index_type ivf_flat \
  --nlist 4096 \
  --output /mnt/projects/tool-control/results/faiss_5M_768_ivf
```

This writes:
- `index.faiss` (serialized FAISS index)
- `queries.dat`
- `ids.dat`
- `meta.json`

## 3) Run benchmark experiments

Benchmarks are configured by JSON (see `benchmark_config.json`).

```bash
python benchmark_nearest_search.py --config benchmark_config.json
```

### Notes on FAISS GPU failures like `cublas failed (13)`

If your Torch benchmark works but FAISS `IndexFlat*` on GPU crashes with:
- `Faiss assertion 'err == CUBLAS_STATUS_SUCCESS' failed`
- `cublas failed (13)`

that is usually caused by the FAISS GPU flat-search path in a specific FAISS/CUDA/driver combination for very large exhaustive GEMM tiles. A practical workaround is:
- benchmark a prebuilt IVF index (`index_type=ivf_flat`) instead of exhaustive flat search
- and run search-only benchmarks from serialized index files (`index.faiss`).

The benchmark script now supports both dataset styles:
- classic memmap dataset: uses `bank.dat` + builds FAISS index in the run
- FAISS-native dataset: uses `index.faiss` + benchmarks search directly
