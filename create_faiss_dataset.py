#!/usr/bin/env python3
"""Create a FAISS-native benchmark dataset (serialized index + queries).

This is useful when you want benchmark runs to measure search latency only,
without rebuilding and re-adding vectors to the index every run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a FAISS dataset with a prebuilt index")
    p.add_argument("--n", type=int, required=True, help="Number of database vectors")
    p.add_argument("--d", type=int, required=True, help="Vector dimension")
    p.add_argument("--q", type=int, default=64, help="Number of query vectors")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--metric", choices=["ip", "l2"], default="ip", help="Distance metric")
    p.add_argument(
        "--index_type",
        choices=["flat", "ivf_flat"],
        default="ivf_flat",
        help="Index type. ivf_flat is recommended for large GPU benchmarks.",
    )
    p.add_argument("--nlist", type=int, default=4096, help="Number of IVF lists (ivf_flat only)")
    p.add_argument("--normalize", action="store_true", default=True, help="L2-normalize vectors")
    p.add_argument("--no_normalize", action="store_true", help="Disable normalization")
    p.add_argument("--train_size", type=int, default=200000, help="Training vectors for IVF")
    p.add_argument("--id_offset", type=int, default=0, help="Starting ID value")
    return p.parse_args()


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def make_index(d: int, metric: str, index_type: str, nlist: int):
    import faiss

    metric_type = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    if index_type == "flat":
        if metric == "ip":
            return faiss.IndexIDMap2(faiss.IndexFlatIP(d))
        return faiss.IndexIDMap2(faiss.IndexFlatL2(d))

    quantizer = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
    return faiss.IndexIDMap2(faiss.IndexIVFFlat(quantizer, d, nlist, metric_type))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalize = args.normalize and not args.no_normalize
    rng_bank = np.random.RandomState(args.seed)
    rng_q = np.random.RandomState(args.seed + 999)

    print("=== Creating FAISS dataset ===")
    print(f"bank={args.n:,}x{args.d}, queries={args.q}x{args.d}, index_type={args.index_type}")

    t0 = time.perf_counter()
    bank = rng_bank.randn(args.n, args.d).astype(np.float32)
    queries = rng_q.randn(args.q, args.d).astype(np.float32)
    if normalize:
        bank = l2_normalize_rows(bank)
        queries = l2_normalize_rows(queries)

    ids = np.arange(args.id_offset, args.id_offset + args.n, dtype=np.int64)

    index = make_index(args.d, args.metric, args.index_type, args.nlist)

    if args.index_type == "ivf_flat":
        train_n = min(args.n, max(10000, args.train_size))
        print(f"Training IVF index with {train_n:,} vectors...")
        index.train(bank[:train_n])

    print("Adding vectors...")
    index.add_with_ids(bank, ids)

    index_path = out_dir / "index.faiss"
    queries_path = out_dir / "queries.dat"
    ids_path = out_dir / "ids.dat"
    meta_path = out_dir / "meta.json"

    import faiss

    faiss.write_index(index, str(index_path))
    queries.tofile(str(queries_path))
    ids.tofile(str(ids_path))

    elapsed = time.perf_counter() - t0
    meta = {
        "dataset_type": "faiss_index",
        "n": args.n,
        "d": args.d,
        "q": args.q,
        "dtype": "fp32",
        "seed": args.seed,
        "normalize": normalize,
        "metric": args.metric,
        "index_type": args.index_type,
        "nlist": args.nlist if args.index_type == "ivf_flat" else None,
        "files": {
            "index": "index.faiss",
            "queries": "queries.dat",
            "ids": "ids.dat",
        },
        "generation_time_s": round(elapsed, 3),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[✓] Wrote index:   {index_path}")
    print(f"[✓] Wrote queries: {queries_path}")
    print(f"[✓] Wrote meta:    {meta_path}")


if __name__ == "__main__":
    main()
