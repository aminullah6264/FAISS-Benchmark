#!/usr/bin/env python3
"""
Benchmark exact top-k retrieval on GPU with:
  (1) Torch chunked matmul (no faiss)
  (2) FAISS GPU IndexFlatIP (exact) OR IVF-PQ (approx, optional)

Designed for very large banks (e.g., 10M x 2048) on A100.
- For cosine similarity: we L2-normalize and use inner product.
- Bank can be stored as float16 to fit.

IMPORTANT:
- 10M x 2048 float16 ~ 40.96 GB just for the bank.
  This fits on A100 80GB; may NOT fit on A100 40GB.
- Torch baseline does NOT materialize full (B x N) similarity matrix; it streams chunks.

Usage examples:
  python bench_retrieval_gpu.py --n 10000000 --d 2048 --k 10 --q 64 --dtype fp16 --chunk 250000
  python bench_retrieval_gpu.py --n 5000000 --d 2048 --k 10 --q 64 --dtype fp16 --chunk 250000 --faiss
  python bench_retrieval_gpu.py --n 10000000 --d 2048 --k 10 --q 64 --dtype fp16 --chunk 250000 --faiss --faiss_index ivfpq
"""

from __future__ import annotations
import argparse
import time
import math
import numpy as np
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000_000, help="bank size")
    p.add_argument("--d", type=int, default=2048, help="embedding dim")
    p.add_argument("--k", type=int, default=10, help="top-k")
    p.add_argument("--q", type=int, default=64, help="number of queries")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--chunk", type=int, default=250_000, help="bank chunk size for torch baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_norm", action="store_true", help="disable L2 normalization (not recommended for cosine)")
    p.add_argument("--faiss", action="store_true", help="run FAISS GPU benchmark")
    p.add_argument("--faiss_index", type=str, default="flat", choices=["flat", "ivfpq"])
    p.add_argument("--nlist", type=int, default=65536, help="IVF: number of coarse clusters")
    p.add_argument("--nprobe", type=int, default=32, help="IVF: number of probes")
    p.add_argument("--pq_m", type=int, default=64, help="IVFPQ: number of subquantizers (code bytes if 8-bit)")
    p.add_argument("--train_ivf", type=int, default=500_000, help="IVF training sample size (subset of bank)")
    return p.parse_args()

def now():
    return time.perf_counter()

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def torch_topk_chunked_ip(queries: torch.Tensor, bank: torch.Tensor, k: int, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact top-k inner product search: queries [Q,D], bank [N,D]
    Streams bank in chunks to avoid (Q x N) score matrix.
    Returns: (topk_scores [Q,k], topk_indices [Q,k]) on GPU
    """
    device = queries.device
    Q, D = queries.shape
    N = bank.shape[0]
    k = int(k)
    chunk = int(chunk)

    # initialize with -inf
    top_scores = torch.full((Q, k), -float("inf"), device=device, dtype=torch.float32)
    top_indices = torch.full((Q, k), -1, device=device, dtype=torch.int64)

    # iterate over bank chunks
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        b = bank[start:end]                      # [C,D]
        # compute scores [Q,C] via GEMM
        s = queries @ b.t()                      # dtype: fp16/fp32 -> accumulate to fp16/32 depending; we cast below
        s = s.float()

        # top-k within this chunk
        cs, ci = torch.topk(s, k=min(k, s.shape[1]), dim=1, largest=True, sorted=True)  # [Q,k']

        # merge with running top-k
        merged_scores = torch.cat([top_scores, cs], dim=1)                     # [Q, 2k]
        merged_indices = torch.cat([top_indices, ci + start], dim=1)           # [Q, 2k]

        new_scores, new_pos = torch.topk(merged_scores, k=k, dim=1, largest=True, sorted=True)
        new_indices = torch.gather(merged_indices, dim=1, index=new_pos)

        top_scores = new_scores
        top_indices = new_indices

    return top_scores, top_indices

# def make_random_embeddings(n: int, d: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
#     g = torch.Generator(device="cpu")
#     g.manual_seed(seed)
#     # Generate on CPU then move to GPU to avoid huge GPU rand overhead (either is fine)
#     x = torch.randn((n, d), generator=g, dtype=torch.float32)  # CPU fp32
#     x = x.to(device=device, dtype=dtype, non_blocking=True)
#     return x


def make_random_embeddings(n: int, d: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    # Generate directly on GPU in the target dtype (no giant CPU fp32 buffer, no huge transfer)
    return torch.randn((n, d), device=device, dtype=dtype)

def bytes_human(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(nbytes)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}PB"

def bench_torch(bank: torch.Tensor, queries: torch.Tensor, k: int, chunk: int, warmup: int = 2, iters: int = 5):
    # warmup
    for _ in range(warmup):
        _ = torch_topk_chunked_ip(queries, bank, k=k, chunk=chunk)
        torch.cuda.synchronize()

    ts = []
    for _ in range(iters):
        t0 = now()
        scores, idx = torch_topk_chunked_ip(queries, bank, k=k, chunk=chunk)
        torch.cuda.synchronize()
        t1 = now()
        ts.append(t1 - t0)

    return float(np.mean(ts)), float(np.std(ts)), scores, idx

def bench_faiss_flat(bank: torch.Tensor, queries: torch.Tensor, k: int):
    import faiss

    # FAISS expects float32 on GPU by default; you can store fp16 in torch but pass float32 to faiss,
    # or use faiss with float16 support. The simplest reliable path: float32 input.
    xb = bank.detach().float().cpu().numpy()
    xq = queries.detach().float().cpu().numpy()

    res = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatIP(xb.shape[1])
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    t0 = now()
    index_gpu.add(xb)   # copies to GPU
    t1 = now()

    # warmup
    _ = index_gpu.search(xq, k)

    t2 = now()
    D, I = index_gpu.search(xq, k)
    t3 = now()

    return (t1 - t0), (t3 - t2), D, I

def bench_faiss_ivfpq(bank: torch.Tensor, queries: torch.Tensor, k: int, nlist: int, nprobe: int, pq_m: int, train_ivf: int, seed: int):
    import faiss

    xb = bank.detach().float().cpu().numpy()
    xq = queries.detach().float().cpu().numpy()

    d = xb.shape[1]
    n = xb.shape[0]
    train_ivf = min(int(train_ivf), n)

    # training subset
    rng = np.random.RandomState(seed)
    tr_idx = rng.choice(n, size=train_ivf, replace=False)
    xt = xb[tr_idx]

    res = faiss.StandardGpuResources()

    quantizer = faiss.IndexFlatIP(d)
    index_cpu = faiss.IndexIVFPQ(quantizer, d, int(nlist), int(pq_m), 8)  # 8 bits per subquantizer
    index_cpu.metric_type = faiss.METRIC_INNER_PRODUCT
    index_cpu.nprobe = int(nprobe)

    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    t0 = now()
    index_gpu.train(xt)
    t1 = now()

    t2 = now()
    index_gpu.add(xb)
    t3 = now()

    # warmup
    _ = index_gpu.search(xq, k)

    t4 = now()
    D, I = index_gpu.search(xq, k)
    t5 = now()

    return (t1 - t0), (t3 - t2), (t5 - t4), D, I

def recall_at_k(ref_I: np.ndarray, test_I: np.ndarray, k: int) -> float:
    # ref_I and test_I are [Q,k]
    Q = ref_I.shape[0]
    s = 0
    ref_sets = [set(ref_I[i, :k].tolist()) for i in range(Q)]
    for i in range(Q):
        s += len(ref_sets[i].intersection(test_I[i, :k].tolist())) / float(k)
    return s / float(Q)

def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda:0")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Create bank + queries ----
    # Note: generating 10M x 2048 may take time and memory.
    # If you already have embeddings on disk, replace this with loading/memmap->GPU.
    print(f"[Setup] Creating bank: N={args.n:,} D={args.d} dtype={args.dtype} on {device} ...")
    bank = make_random_embeddings(args.n, args.d, dtype=dtype, device=device, seed=args.seed)

    print(f"[Setup] Creating queries: Q={args.q} D={args.d} ...")
    queries = make_random_embeddings(args.q, args.d, dtype=dtype, device=device, seed=args.seed + 999)

    if not args.no_norm:
        bank = l2_normalize(bank)
        queries = l2_normalize(queries)

    torch.cuda.synchronize()

    # Rough memory
    bank_bytes = args.n * args.d * (2 if args.dtype == "fp16" else 4)
    print(f"[Info] Bank raw size (approx): {bytes_human(bank_bytes)}")

    # ---- Torch baseline ----
    print(f"\n[Torch] Exact chunked IP top-{args.k} (chunk={args.chunk:,})")
    mean_s, std_s, torch_scores, torch_idx = bench_torch(bank, queries, k=args.k, chunk=args.chunk)
    print(f"[Torch] mean={mean_s*1000:.2f} ms  std={std_s*1000:.2f} ms  per_query={(mean_s/args.q)*1000:.4f} ms")

    # ---- FAISS ----
    if args.faiss:
        if args.faiss_index == "flat":
            print(f"\n[FAISS] GPU IndexFlatIP (exact)")
            build_t, search_t, D, I = bench_faiss_flat(bank, queries, k=args.k)
            print(f"[FAISS] build(add)={build_t:.2f} s  search={search_t*1000:.2f} ms  per_query={(search_t/args.q)*1000:.4f} ms")
            # Compare FAISS to torch indices (both exact) on small Q
            rec = recall_at_k(torch_idx.detach().cpu().numpy(), I, k=args.k)
            print(f"[FAISS] recall@{args.k} vs torch baseline: {rec:.4f}")

        else:
            print(f"\n[FAISS] GPU IndexIVFPQ (approx) nlist={args.nlist} nprobe={args.nprobe} pq_m={args.pq_m} train_ivf={args.train_ivf}")
            train_t, add_t, search_t, D, I = bench_faiss_ivfpq(
                bank, queries, k=args.k,
                nlist=args.nlist, nprobe=args.nprobe,
                pq_m=args.pq_m, train_ivf=args.train_ivf, seed=args.seed
            )
            print(f"[FAISS] train={train_t:.2f} s  add={add_t:.2f} s  search={search_t*1000:.2f} ms  per_query={(search_t/args.q)*1000:.4f} ms")
            rec = recall_at_k(torch_idx.detach().cpu().numpy(), I, k=args.k)
            print(f"[FAISS] recall@{args.k} vs exact torch baseline: {rec:.4f}")

if __name__ == "__main__":
    main()