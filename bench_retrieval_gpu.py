#!/usr/bin/env python3
"""
Configurable retrieval benchmarking for Torch and FAISS on single or multi-GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    p.add_argument("--output_json", type=str, default="benchmark_results.json", help="Output JSON report path.")

    # legacy single-run flags still supported
    p.add_argument("--name", type=str, default="cli_run", help="Experiment name for CLI mode")
    p.add_argument("--n", type=int, default=1_000_000, help="bank size")
    p.add_argument("--d", type=int, default=1024, help="embedding dim")
    p.add_argument("--k", type=int, default=10, help="top-k")
    p.add_argument("--q", type=int, default=64, help="number of queries")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--chunk", type=int, default=250_000, help="bank chunk size for torch baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_norm", action="store_true", help="disable L2 normalization")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)

    p.add_argument("--gpu_mode", type=str, default="single", choices=["single", "multi"])

    p.add_argument("--faiss", action="store_true", help="run FAISS GPU benchmark")
    p.add_argument("--faiss_index", type=str, default="flat", choices=["flat", "ivfpq"])
    p.add_argument("--nlist", type=int, default=8192)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument("--pq_m", type=int, default=32)
    p.add_argument("--train_ivf", type=int, default=200_000)

    p.add_argument("--embedding_cache_path", type=str, default=None, help="Optional .npz path to save/load bank+query embeddings")
    p.add_argument("--reuse_embedding_cache", action="store_true", help="Load embeddings from --embedding_cache_path if present")
    p.add_argument("--faiss_cache_path", type=str, default=None, help="Optional FAISS index file path")
    p.add_argument("--faiss_load_cache", action="store_true", help="Load FAISS index from --faiss_cache_path if present")
    p.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--stream_torch_bank", action="store_true", help="Stream random bank chunks for Torch so full bank is never materialized in GPU memory")
    return p.parse_args()


def progress(iterable, *, desc: str, total: int | None = None, enabled: bool = True):
    return tqdm(iterable, desc=desc, total=total, leave=False) if enabled else iterable


def now() -> float:
    return time.perf_counter()


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def make_random_embeddings(n: int, d: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn((n, d), device=device, dtype=dtype)


def bytes_human(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(nbytes)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}PB"


def available_gpu_bytes(gpu_mode: str) -> int:
    if gpu_mode == "multi":
        total = 0
        for i in range(torch.cuda.device_count()):
            free_b, _ = torch.cuda.mem_get_info(i)
            total += int(free_b)
        return total
    free_b, _ = torch.cuda.mem_get_info(0)
    return int(free_b)


@torch.no_grad()
def torch_topk_chunked_ip(queries: torch.Tensor, bank: torch.Tensor, k: int, chunk: int, show_progress: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    q_count = queries.shape[0]
    n = bank.shape[0]
    top_scores = torch.full((q_count, k), -float("inf"), device=queries.device, dtype=torch.float32)
    top_indices = torch.full((q_count, k), -1, device=queries.device, dtype=torch.int64)

    starts = range(0, n, chunk)
    for start in progress(starts, desc="Torch chunked top-k over bank chunks", total=len(range(0, n, chunk)), enabled=show_progress):
        end = min(start + chunk, n)
        s = (queries @ bank[start:end].t()).float()
        cs, ci = torch.topk(s, k=min(k, s.shape[1]), dim=1, largest=True, sorted=True)

        merged_scores = torch.cat([top_scores, cs], dim=1)
        merged_indices = torch.cat([top_indices, ci + start], dim=1)
        top_scores, new_pos = torch.topk(merged_scores, k=k, dim=1, largest=True, sorted=True)
        top_indices = torch.gather(merged_indices, dim=1, index=new_pos)

    return top_scores, top_indices


@torch.no_grad()
def torch_topk_chunked_ip_multi_gpu(
    queries: torch.Tensor,
    bank_shards: list[torch.Tensor],
    shard_offsets: list[int],
    k: int,
    chunk: int,
    show_progress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_cpu = queries.float().cpu()
    all_scores_cpu: list[torch.Tensor] = []
    all_indices_cpu: list[torch.Tensor] = []

    shard_iter = list(zip(bank_shards, shard_offsets))
    for shard, offset in progress(shard_iter, desc="Multi-GPU shard top-k", total=len(shard_iter), enabled=show_progress):
        q_dev = q_cpu.to(shard.device, non_blocking=True)
        scores, idx = torch_topk_chunked_ip(q_dev, shard, k=k, chunk=chunk, show_progress=False)
        all_scores_cpu.append(scores.cpu())
        all_indices_cpu.append((idx + offset).cpu())

    merged_scores = torch.cat(all_scores_cpu, dim=1)
    merged_indices = torch.cat(all_indices_cpu, dim=1)
    top_scores, top_pos = torch.topk(merged_scores, k=k, dim=1, largest=True, sorted=True)
    top_indices = torch.gather(merged_indices, dim=1, index=top_pos)
    return top_scores.cuda(), top_indices.cuda()


def bench_torch_single(bank: torch.Tensor, queries: torch.Tensor, k: int, chunk: int, warmup: int, iters: int, show_progress: bool = False):
    for _ in progress(range(warmup), desc="Torch warmup iterations", total=warmup, enabled=show_progress):
        _ = torch_topk_chunked_ip(queries, bank, k=k, chunk=chunk, show_progress=False)
        torch.cuda.synchronize()

    ts = []
    for _ in progress(range(iters), desc="Torch timed iterations", total=iters, enabled=show_progress):
        t0 = now()
        scores, idx = torch_topk_chunked_ip(queries, bank, k=k, chunk=chunk, show_progress=False)
        torch.cuda.synchronize()
        ts.append(now() - t0)
    return float(np.mean(ts)), float(np.std(ts)), scores, idx


def bench_torch_multi(bank: torch.Tensor, queries: torch.Tensor, k: int, chunk: int, warmup: int, iters: int, show_progress: bool = False):
    ngpu = torch.cuda.device_count()
    shard_sizes = [bank.shape[0] // ngpu + (1 if i < bank.shape[0] % ngpu else 0) for i in range(ngpu)]
    starts = [0]
    for s in shard_sizes[:-1]:
        starts.append(starts[-1] + s)

    bank_shards = []
    enum_shards = list(enumerate(zip(starts, shard_sizes)))
    for i, (start, size) in progress(enum_shards, desc="Sharding bank to GPUs", total=len(enum_shards), enabled=show_progress):
        bank_shards.append(bank[start:start + size].to(f"cuda:{i}", non_blocking=True))

    for _ in progress(range(warmup), desc="Multi-GPU warmup iterations", total=warmup, enabled=show_progress):
        _ = torch_topk_chunked_ip_multi_gpu(queries, bank_shards, starts, k=k, chunk=chunk, show_progress=False)
        torch.cuda.synchronize()

    ts = []
    for _ in progress(range(iters), desc="Multi-GPU timed iterations", total=iters, enabled=show_progress):
        t0 = now()
        scores, idx = torch_topk_chunked_ip_multi_gpu(queries, bank_shards, starts, k=k, chunk=chunk, show_progress=False)
        torch.cuda.synchronize()
        ts.append(now() - t0)

    return float(np.mean(ts)), float(np.std(ts)), scores, idx


@torch.no_grad()
def torch_topk_chunked_ip_streaming(
    queries: torch.Tensor,
    n: int,
    d: int,
    k: int,
    chunk: int,
    dtype: torch.dtype,
    seed: int,
    normalize: bool,
    show_progress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_count = queries.shape[0]
    top_scores = torch.full((q_count, k), -float("inf"), device=queries.device, dtype=torch.float32)
    top_indices = torch.full((q_count, k), -1, device=queries.device, dtype=torch.int64)

    gen = torch.Generator(device=queries.device)
    gen.manual_seed(int(seed))
    starts = range(0, n, chunk)
    for start in progress(starts, desc="Streaming random bank chunks", total=len(range(0, n, chunk)), enabled=show_progress):
        size = min(chunk, n - start)
        xb_chunk = torch.randn((size, d), device=queries.device, dtype=dtype, generator=gen)
        if normalize:
            xb_chunk = l2_normalize(xb_chunk)
        s = (queries @ xb_chunk.t()).float()
        cs, ci = torch.topk(s, k=min(k, s.shape[1]), dim=1, largest=True, sorted=True)
        merged_scores = torch.cat([top_scores, cs], dim=1)
        merged_indices = torch.cat([top_indices, ci + start], dim=1)
        top_scores, new_pos = torch.topk(merged_scores, k=k, dim=1, largest=True, sorted=True)
        top_indices = torch.gather(merged_indices, dim=1, index=new_pos)
    return top_scores, top_indices


def bench_torch_single_streaming(
    queries: torch.Tensor,
    n: int,
    d: int,
    k: int,
    chunk: int,
    warmup: int,
    iters: int,
    dtype: torch.dtype,
    seed: int,
    normalize: bool,
    show_progress: bool = False,
):
    for _ in progress(range(warmup), desc="Torch warmup iterations", total=warmup, enabled=show_progress):
        _ = torch_topk_chunked_ip_streaming(queries, n=n, d=d, k=k, chunk=chunk, dtype=dtype, seed=seed, normalize=normalize, show_progress=False)
        torch.cuda.synchronize()

    ts = []
    for _ in progress(range(iters), desc="Torch timed iterations", total=iters, enabled=show_progress):
        t0 = now()
        scores, idx = torch_topk_chunked_ip_streaming(queries, n=n, d=d, k=k, chunk=chunk, dtype=dtype, seed=seed, normalize=normalize, show_progress=False)
        torch.cuda.synchronize()
        ts.append(now() - t0)
    return float(np.mean(ts)), float(np.std(ts)), scores, idx


def bench_faiss_flat(bank: torch.Tensor, queries: torch.Tensor, k: int, gpu_mode: str, cache_path: str | None = None, load_cache: bool = False):
    import faiss

    xb = bank.detach().float().cpu().numpy()
    xq = queries.detach().float().cpu().numpy()

    index_cpu = None
    build_t = 0.0
    if cache_path and load_cache and Path(cache_path).exists():
        index_cpu = faiss.read_index(cache_path)
    else:
        index_cpu = faiss.IndexFlatIP(xb.shape[1])
        if cache_path and gpu_mode == "multi":
            t0 = now(); index_cpu.add(xb); t1 = now(); build_t = t1 - t0
        else:
            if gpu_mode == "multi":
                index_gpu_build = faiss.index_cpu_to_all_gpus(index_cpu)
            else:
                res_build = faiss.StandardGpuResources()
                index_gpu_build = faiss.index_cpu_to_gpu(res_build, 0, index_cpu)
            t0 = now(); index_gpu_build.add(xb); t1 = now(); build_t = t1 - t0
            index_cpu = faiss.index_gpu_to_cpu(index_gpu_build)
        if cache_path:
            faiss.write_index(index_cpu, cache_path)

    if gpu_mode == "multi":
        index_gpu = faiss.index_cpu_to_all_gpus(index_cpu)
    else:
        res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    _ = index_gpu.search(xq, k)
    t2 = now(); D, I = index_gpu.search(xq, k); t3 = now()
    return build_t, (t3 - t2), D, I


def bench_faiss_ivfpq(
    bank: torch.Tensor,
    queries: torch.Tensor,
    k: int,
    nlist: int,
    nprobe: int,
    pq_m: int,
    train_ivf: int,
    seed: int,
    cache_path: str | None = None,
    load_cache: bool = False,
):
    import faiss

    xb = bank.detach().float().cpu().numpy()
    xq = queries.detach().float().cpu().numpy()

    res = faiss.StandardGpuResources()
    if cache_path and load_cache and Path(cache_path).exists():
        index_cpu = faiss.read_index(cache_path)
        train_t = 0.0
        add_t = 0.0
    else:
        d = xb.shape[1]
        n = xb.shape[0]
        train_ivf = min(int(train_ivf), n)
        rng = np.random.RandomState(seed)
        tr_idx = rng.choice(n, size=train_ivf, replace=False)
        xt = xb[tr_idx]

        quantizer = faiss.IndexFlatIP(d)
        index_cpu = faiss.IndexIVFPQ(quantizer, d, int(nlist), int(pq_m), 8)
        index_cpu.metric_type = faiss.METRIC_INNER_PRODUCT
        index_cpu.nprobe = int(nprobe)
        index_gpu_build = faiss.index_cpu_to_gpu(res, 0, index_cpu)

        t0 = now(); index_gpu_build.train(xt); t1 = now(); train_t = t1 - t0
        t2 = now(); index_gpu_build.add(xb); t3 = now(); add_t = t3 - t2
        index_cpu = faiss.index_gpu_to_cpu(index_gpu_build)
        if cache_path:
            faiss.write_index(index_cpu, cache_path)

    index_cpu.nprobe = int(nprobe)
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    _ = index_gpu.search(xq, k)
    t4 = now(); D, I = index_gpu.search(xq, k); t5 = now()
    return train_t, add_t, (t5 - t4), D, I


def recall_at_k(ref_I: np.ndarray, test_I: np.ndarray, k: int, show_progress: bool = False) -> float:
    q_count = ref_I.shape[0]
    total = 0.0
    for i in progress(range(q_count), desc="Recall@k aggregation", total=q_count, enabled=show_progress):
        total += len(set(ref_I[i, :k].tolist()).intersection(test_I[i, :k].tolist())) / float(k)
    return total / float(q_count)


def torch_flops(n: int, d: int, q: int) -> int:
    return int(2 * n * d * q)


def ivfpq_estimated_flops(n: int, d: int, q: int, nlist: int, nprobe: int) -> int:
    candidates = max(1, int(n * min(nprobe, nlist) / max(1, nlist)))
    return int(2 * candidates * d * q)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def config_to_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.config:
        data = json.loads(Path(args.config).read_text())
        global_cfg = data.get("global", {})
        experiments = data.get("experiments", [])
        merged = [deep_update(global_cfg, exp) for exp in experiments]
        if not merged:
            raise ValueError("Config file must define at least one experiment in 'experiments'.")
        return merged

    return [{
        "name": args.name,
        "n": args.n,
        "d": args.d,
        "k": args.k,
        "q": args.q,
        "dtype": args.dtype,
        "chunk": args.chunk,
        "seed": args.seed,
        "normalize": not args.no_norm,
        "warmup": args.warmup,
        "iters": args.iters,
        "gpu_mode": args.gpu_mode,
        "faiss": {
            "enabled": args.faiss,
            "index": args.faiss_index,
            "nlist": args.nlist,
            "nprobe": args.nprobe,
            "pq_m": args.pq_m,
            "train_ivf": args.train_ivf,
            "cache_path": args.faiss_cache_path,
            "load_cache": args.faiss_load_cache,
        },
        "embedding_cache": {
            "path": args.embedding_cache_path,
            "reuse": args.reuse_embedding_cache,
        },
        "tqdm": {"enabled": not args.no_tqdm},
        "stream_torch_bank": args.stream_torch_bank,
    }]


def run_experiment(exp: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    show_progress = bool(exp.get("tqdm", {}).get("enabled", True))
    gpu_mode = exp.get("gpu_mode", "single")
    ngpu = torch.cuda.device_count()
    if gpu_mode == "multi" and ngpu < 2:
        raise RuntimeError("gpu_mode=multi requires at least 2 visible CUDA devices.")

    dtype = torch.float16 if exp["dtype"] == "fp16" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(exp["seed"])
    np.random.seed(exp["seed"])

    print(f"\n=== Experiment: {exp['name']} ===")
    print(f"[Setup] N={exp['n']:,} D={exp['d']} Q={exp['q']} k={exp['k']} dtype={exp['dtype']} gpu_mode={gpu_mode}")

    cache_cfg = exp.get("embedding_cache", {})
    emb_path = cache_cfg.get("path")
    emb_reuse = bool(cache_cfg.get("reuse", False))
    bank_bytes = exp["n"] * exp["d"] * (2 if exp["dtype"] == "fp16" else 4)
    free_bytes = available_gpu_bytes(gpu_mode)
    auto_stream = bank_bytes > int(0.8 * free_bytes)
    stream_torch_bank = bool(exp.get("stream_torch_bank", False) or auto_stream)
    if stream_torch_bank and emb_path:
        print("[Cache] embedding_cache is ignored in stream_torch_bank mode.")

    if (not stream_torch_bank) and emb_path and emb_reuse and Path(emb_path).exists():
        cached = np.load(emb_path)
        bank = torch.from_numpy(cached["bank"]).to(torch.device("cuda:0"), dtype=dtype)
        queries = torch.from_numpy(cached["queries"]).to(torch.device("cuda:0"), dtype=dtype)
        print(f"[Cache] Loaded embeddings from {emb_path}")
    else:
        queries = make_random_embeddings(exp["q"], exp["d"], dtype=dtype, device=torch.device("cuda:0"), seed=exp["seed"] + 999)
        if not stream_torch_bank:
            bank = make_random_embeddings(exp["n"], exp["d"], dtype=dtype, device=torch.device("cuda:0"), seed=exp["seed"])
        if (not stream_torch_bank) and emb_path:
            np.savez(emb_path, bank=bank.detach().float().cpu().numpy(), queries=queries.detach().float().cpu().numpy())
            print(f"[Cache] Saved embeddings to {emb_path}")

    if exp.get("normalize", True):
        if not stream_torch_bank:
            bank = l2_normalize(bank)
        queries = l2_normalize(queries)

    print(f"[Info] Bank size ≈ {bytes_human(bank_bytes)}")
    if auto_stream:
        print(f"[Info] Auto-enabled stream_torch_bank because estimated bank size exceeds ~80% of available GPU memory ({bytes_human(free_bytes)}).")

    if stream_torch_bank and gpu_mode == "multi":
        raise RuntimeError("stream_torch_bank currently supports gpu_mode=single only. Reduce N/D, use fp16, or run single GPU streaming.")

    if stream_torch_bank:
        t_mean, t_std, _, torch_idx = bench_torch_single_streaming(
            queries, n=exp["n"], d=exp["d"], k=exp["k"], chunk=exp["chunk"], warmup=exp["warmup"], iters=exp["iters"],
            dtype=dtype, seed=exp["seed"], normalize=bool(exp.get("normalize", True)), show_progress=show_progress
        )
    elif gpu_mode == "single":
        t_mean, t_std, _, torch_idx = bench_torch_single(bank, queries, k=exp["k"], chunk=exp["chunk"], warmup=exp["warmup"], iters=exp["iters"], show_progress=show_progress)
    else:
        t_mean, t_std, _, torch_idx = bench_torch_multi(bank, queries, k=exp["k"], chunk=exp["chunk"], warmup=exp["warmup"], iters=exp["iters"], show_progress=show_progress)

    t_flops = torch_flops(exp["n"], exp["d"], exp["q"])
    result: dict[str, Any] = {
        "name": exp["name"],
        "setup": {
            "n": exp["n"], "d": exp["d"], "q": exp["q"], "k": exp["k"], "dtype": exp["dtype"],
            "chunk": exp["chunk"], "seed": exp["seed"], "gpu_mode": gpu_mode, "ngpu": ngpu,
            "normalize": bool(exp.get("normalize", True)),
        },
        "torch": {
            "mean_s": t_mean,
            "std_s": t_std,
            "per_query_ms": (t_mean / exp["q"]) * 1000.0,
            "flops": t_flops,
            "tflops_per_s": t_flops / max(t_mean, 1e-12) / 1e12,
        },
    }

    print(f"[Torch] mean={t_mean*1000:.2f}ms std={t_std*1000:.2f}ms per_query={(t_mean/exp['q'])*1000:.4f}ms TFLOP/s={result['torch']['tflops_per_s']:.2f}")

    faiss_cfg = exp.get("faiss", {})
    if faiss_cfg.get("enabled", False):
        if stream_torch_bank:
            raise RuntimeError("FAISS benchmark requires materialized bank embeddings; disable stream_torch_bank or disable FAISS.")
        index_type = faiss_cfg.get("index", "flat")
        if index_type == "flat":
            build_t, search_t, _, I = bench_faiss_flat(
                bank, queries, exp["k"], gpu_mode=gpu_mode,
                cache_path=faiss_cfg.get("cache_path"), load_cache=bool(faiss_cfg.get("load_cache", False))
            )
            rec = recall_at_k(torch_idx.detach().cpu().numpy(), I, k=exp["k"], show_progress=show_progress)
            f_flops = torch_flops(exp["n"], exp["d"], exp["q"])
            result["faiss"] = {
                "index": "flat",
                "build_s": build_t,
                "search_s": search_t,
                "per_query_ms": (search_t / exp["q"]) * 1000.0,
                "recall_at_k_vs_torch": rec,
                "flops_estimate": f_flops,
                "tflops_per_s_estimate": f_flops / max(search_t, 1e-12) / 1e12,
            }
        else:
            if gpu_mode == "multi":
                print("[FAISS] IVFPQ multi-GPU not enabled in this script; running on cuda:0.")
            train_t, add_t, search_t, _, I = bench_faiss_ivfpq(
                bank, queries, exp["k"], faiss_cfg["nlist"], faiss_cfg["nprobe"], faiss_cfg["pq_m"], faiss_cfg["train_ivf"], exp["seed"],
                cache_path=faiss_cfg.get("cache_path"), load_cache=bool(faiss_cfg.get("load_cache", False))
            )
            rec = recall_at_k(torch_idx.detach().cpu().numpy(), I, k=exp["k"], show_progress=show_progress)
            approx_flops = ivfpq_estimated_flops(exp["n"], exp["d"], exp["q"], faiss_cfg["nlist"], faiss_cfg["nprobe"])
            result["faiss"] = {
                "index": "ivfpq",
                "train_s": train_t,
                "add_s": add_t,
                "search_s": search_t,
                "per_query_ms": (search_t / exp["q"]) * 1000.0,
                "recall_at_k_vs_torch": rec,
                "flops_estimate": approx_flops,
                "tflops_per_s_estimate": approx_flops / max(search_t, 1e-12) / 1e12,
                "nlist": faiss_cfg["nlist"],
                "nprobe": faiss_cfg["nprobe"],
                "pq_m": faiss_cfg["pq_m"],
            }

    return result


def summarize_matrices(results: list[dict[str, Any]]) -> dict[str, Any]:
    time_matrix = []
    flops_matrix = []
    for r in results:
        row_time = {"name": r["name"], "torch_ms": r["torch"]["mean_s"] * 1000.0}
        row_flops = {"name": r["name"], "torch_flops": r["torch"]["flops"]}
        if "faiss" in r:
            row_time["faiss_ms"] = r["faiss"]["search_s"] * 1000.0
            row_flops["faiss_flops_estimate"] = r["faiss"].get("flops_estimate")
        time_matrix.append(row_time)
        flops_matrix.append(row_flops)
    return {"time_ms_matrix": time_matrix, "flops_matrix": flops_matrix}


def main() -> None:
    args = parse_args()
    experiments = config_to_experiments(args)

    results = []
    for exp in progress(experiments, desc="Running experiments", total=len(experiments), enabled=not args.no_tqdm):
        results.append(run_experiment(exp))
    matrices = summarize_matrices(results)
    payload = {"results": results, "matrices": matrices}

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[Done] Wrote JSON report to: {output_path}")


if __name__ == "__main__":
    main()
