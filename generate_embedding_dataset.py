#!/usr/bin/env python3
"""
Generate a synthetic embedding dataset and save to disk for benchmarking.

Usage:
    python generate_embedding_dataset.py --n 5000000 --d 768 --output /mnt/projects/tool-control/results/embeddings_5M_768
    python generate_embedding_dataset.py --n 5000000 --d 2048 --output /mnt/projects/tool-control/results/embeddings_5M_2048
    python generate_embedding_dataset.py --n 5000000 --d 2304 --output /mnt/projects/tool-control/results/embeddings_5M_2304
    python generate_embedding_dataset.py --n 5000000 --d 4096 --output /mnt/projects/tool-control/results/embeddings_5M_4096

This creates a directory with:
    embeddings_5M_768/
    ├── bank.dat          # memory-mappable fp32 embedding matrix (N x D)
    ├── queries.dat       # memory-mappable fp32 query matrix (Q x D)
    ├── meta.json         # metadata (N, D, Q, dtype, seed, normalize, etc.)
    └── ids.dat           # int64 ID array (N,) for simulating real doc IDs

The .dat files are raw binary (numpy memmap-compatible) so they can be
loaded without ever pulling the full array into RAM:

    import numpy as np, json
    meta = json.load(open("embeddings_5M_768/meta.json"))
    bank = np.memmap("embeddings_5M_768/bank.dat", dtype="float32", mode="r",
                     shape=(meta["n"], meta["d"]))
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a synthetic embedding dataset for FAISS / Torch benchmarking."
    )
    p.add_argument("--n", type=int, required=True, help="Number of bank embeddings (e.g. 24000000)")
    p.add_argument("--d", type=int, required=True, help="Embedding dimension (e.g. 768, 2048, 4096)")
    p.add_argument("--q", type=int, default=64, help="Number of query embeddings (default: 64)")
    p.add_argument("--output", type=str, required=True, help="Output directory name")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--normalize", action="store_true", default=True, help="L2-normalize embeddings (default: True)")
    p.add_argument("--no_normalize", action="store_true", help="Disable L2 normalization")
    p.add_argument("--chunk_size", type=int, default=500_000,
                   help="Rows generated per chunk to limit peak RAM (default: 500000)")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "fp32"],
                   help="Storage dtype on disk (default: fp32, recommended for FAISS compatibility)")
    p.add_argument("--id_offset", type=int, default=0,
                   help="Starting document ID (IDs will be id_offset .. id_offset+n-1)")
    return p.parse_args()


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def generate_dataset(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalize = args.normalize and not args.no_normalize
    np_dtype = np.float16 if args.dtype == "fp16" else np.float32

    print(f"=== Generating Embedding Dataset ===")
    print(f"  Bank:       {args.n:,} x {args.d}")
    print(f"  Queries:    {args.q} x {args.d}")
    print(f"  Dtype:      {args.dtype}")
    print(f"  Normalize:  {normalize}")
    print(f"  Seed:       {args.seed}")
    print(f"  Chunk size: {args.chunk_size:,}")
    print(f"  Output dir: {out_dir}")

    bank_bytes = args.n * args.d * np.dtype(np_dtype).itemsize
    print(f"  Est. bank file size: {bank_bytes / (1024**3):.2f} GB")

    # ── Queries (small, generate in one shot) ──────────────────────────
    rng = np.random.RandomState(args.seed + 999)
    queries = rng.randn(args.q, args.d).astype(np_dtype)
    if normalize:
        queries = l2_normalize_rows(queries.astype(np.float32)).astype(np_dtype)

    queries_path = out_dir / "queries.dat"
    queries.tofile(str(queries_path))
    print(f"  [✓] Queries saved: {queries_path}  ({os.path.getsize(queries_path) / (1024**2):.1f} MB)")

    # ── Bank (chunked to limit peak RAM) ───────────────────────────────
    bank_path = out_dir / "bank.dat"
    fp = np.memmap(str(bank_path), dtype=np_dtype, mode="w+", shape=(args.n, args.d))

    rng_bank = np.random.RandomState(args.seed)
    written = 0
    t0 = time.perf_counter()

    while written < args.n:
        chunk_n = min(args.chunk_size, args.n - written)
        chunk = rng_bank.randn(chunk_n, args.d).astype(np_dtype)
        if normalize:
            chunk = l2_normalize_rows(chunk.astype(np.float32)).astype(np_dtype)
        fp[written : written + chunk_n] = chunk
        written += chunk_n

        elapsed = time.perf_counter() - t0
        pct = written / args.n * 100
        rate = written / elapsed if elapsed > 0 else 0
        eta = (args.n - written) / rate if rate > 0 else 0
        print(f"\r  Bank progress: {written:>12,}/{args.n:,}  ({pct:5.1f}%)  "
              f"{rate:,.0f} rows/s  ETA {eta:.0f}s", end="", flush=True)

    fp.flush()
    del fp  # close memmap
    elapsed_total = time.perf_counter() - t0
    print(f"\n  [✓] Bank saved: {bank_path}  ({os.path.getsize(bank_path) / (1024**3):.2f} GB)  in {elapsed_total:.1f}s")

    # ── Document IDs ───────────────────────────────────────────────────
    ids_path = out_dir / "ids.dat"
    ids = np.arange(args.id_offset, args.id_offset + args.n, dtype=np.int64)
    ids.tofile(str(ids_path))
    print(f"  [✓] IDs saved: {ids_path}  ({os.path.getsize(ids_path) / (1024**2):.1f} MB)")

    # ── Metadata ───────────────────────────────────────────────────────
    meta = {
        "n": args.n,
        "d": args.d,
        "q": args.q,
        "dtype": args.dtype,
        "np_dtype": str(np_dtype),
        "seed": args.seed,
        "normalize": normalize,
        "id_offset": args.id_offset,
        "files": {
            "bank": "bank.dat",
            "queries": "queries.dat",
            "ids": "ids.dat",
        },
        "shapes": {
            "bank": [args.n, args.d],
            "queries": [args.q, args.d],
            "ids": [args.n],
        },
        "generation_time_s": round(elapsed_total, 2),
    }

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  [✓] Metadata saved: {meta_path}")
    print(f"\n=== Done ===")
    print(f"\nTo load in your benchmark script:")
    print(f"  import numpy as np, json")
    print(f'  meta = json.load(open("{meta_path}"))')
    print(f'  bank = np.memmap("{bank_path}", dtype="{np_dtype}", mode="r", shape=({args.n}, {args.d}))')
    print(f'  queries = np.memmap("{queries_path}", dtype="{np_dtype}", mode="r", shape=({args.q}, {args.d}))')


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)