#!/usr/bin/env python3
"""Benchmark nearest-neighbor search with Torch vs FAISS on generated datasets.

This script reads a JSON config and produces a benchmark JSON report.
It supports:
- fp32 + fp16 runs
- single-GPU + multi-GPU modes
- Torch brute-force top-k search
- FAISS Flat index search (GPU)
- multiple datasets/experiments in one run

Dataset format expected is the output from `generate_embedding_dataset.py`:
  <dataset_path>/meta.json
  <dataset_path>/bank.dat
  <dataset_path>/queries.dat
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


DEFAULTS: Dict[str, Any] = {
    "metric": "ip",
    "top_k": 10,
    "precisions": ["fp32", "fp16"],
    "gpu_modes": ["single", "multi"],
    "warmup_runs": 2,
    "timed_runs": 10,
    "torch_batch_size": 64,
    "faiss_batch_size": 256,
    "nprobe": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Torch vs FAISS nearest-neighbor search")
    parser.add_argument("--config", required=True, help="Path to benchmark JSON config file")
    return parser.parse_args()


def _apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    merged.update(config)
    return merged


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if "output_json" not in config:
        raise ValueError("Missing required config field: output_json")

    # New format: experiments list.
    if "experiments" in config:
        if not isinstance(config["experiments"], list) or not config["experiments"]:
            raise ValueError("config.experiments must be a non-empty list")

        normalized_experiments: List[Dict[str, Any]] = []
        for i, exp in enumerate(config["experiments"]):
            if "dataset_path" not in exp:
                raise ValueError(f"Missing required experiments[{i}].dataset_path")
            exp_cfg = _apply_defaults({**config, **exp})
            exp_cfg["name"] = exp.get("name", f"experiment_{i + 1}")
            exp_cfg["dataset_path"] = exp["dataset_path"]
            normalized_experiments.append(exp_cfg)

        return {
            "output_json": config["output_json"],
            "experiments": normalized_experiments,
        }

    # Backward compatibility: single dataset config.
    if "dataset_path" not in config:
        raise ValueError("Missing required config field: dataset_path (or use experiments list)")

    exp_cfg = _apply_defaults(config)
    exp_cfg["name"] = config.get("name", "experiment_1")

    return {
        "output_json": config["output_json"],
        "experiments": [exp_cfg],
    }


def load_dataset(dataset_path: Path) -> Dict[str, Any]:
    meta_path = dataset_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found at: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n, d, q = meta["n"], meta["d"], meta["q"]

    file_dtype = np.float16 if meta.get("dtype") == "fp16" else np.float32

    bank = np.memmap(dataset_path / meta["files"]["bank"], dtype=file_dtype, mode="r", shape=(n, d))
    queries = np.memmap(dataset_path / meta["files"]["queries"], dtype=file_dtype, mode="r", shape=(q, d))

    return {
        "meta": meta,
        "bank": bank,
        "queries": queries,
    }


def chunk_iter(total: int, batch_size: int) -> Sequence[Tuple[int, int]]:
    return [(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]


def bench_torch(
    bank_np: np.ndarray,
    queries_np: np.ndarray,
    precision: str,
    gpu_mode: str,
    metric: str,
    top_k: int,
    warmup_runs: int,
    timed_runs: int,
    batch_size: int,
) -> Dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for Torch benchmarking")

    num_gpus = torch.cuda.device_count()
    if gpu_mode == "multi" and num_gpus < 2:
        raise RuntimeError("Multi-GPU mode requested but fewer than 2 GPUs detected")

    dtype = torch.float16 if precision == "fp16" else torch.float32
    bank_cast = bank_np.astype(np.float32 if precision == "fp32" else np.float16, copy=False)
    queries_cast = queries_np.astype(np.float32 if precision == "fp32" else np.float16, copy=False)

    setup_t0 = time.perf_counter()

    if gpu_mode == "single":
        bank_t = torch.from_numpy(bank_cast).to(device="cuda:0", dtype=dtype, non_blocking=True)
        shard_tensors = [bank_t]
    else:
        shard_tensors = []
        per_gpu = int(np.ceil(bank_cast.shape[0] / num_gpus))
        for gpu_id in range(num_gpus):
            s = gpu_id * per_gpu
            e = min((gpu_id + 1) * per_gpu, bank_cast.shape[0])
            if s >= e:
                break
            shard = torch.from_numpy(bank_cast[s:e]).to(device=f"cuda:{gpu_id}", dtype=dtype, non_blocking=True)
            shard_tensors.append(shard)

    setup_t1 = time.perf_counter()

    def run_once() -> None:
        for qs, qe in chunk_iter(queries_cast.shape[0], batch_size):
            q_np = queries_cast[qs:qe]
            if gpu_mode == "single":
                q = torch.from_numpy(q_np).to(device="cuda:0", dtype=dtype, non_blocking=True)
                sims = q @ shard_tensors[0].T
                if metric == "l2":
                    sims = -torch.cdist(q, shard_tensors[0])
                torch.topk(sims, k=top_k, dim=1)
            else:
                per_gpu_vals = []
                per_gpu_idx = []
                offset = 0
                for gpu_id, shard in enumerate(shard_tensors):
                    q = torch.from_numpy(q_np).to(device=f"cuda:{gpu_id}", dtype=dtype, non_blocking=True)
                    sims = q @ shard.T
                    if metric == "l2":
                        sims = -torch.cdist(q, shard)
                    vals, idx = torch.topk(sims, k=top_k, dim=1)
                    per_gpu_vals.append(vals.to(device="cuda:0"))
                    per_gpu_idx.append((idx + offset).to(device="cuda:0"))
                    offset += shard.shape[0]

                merged_vals = torch.cat(per_gpu_vals, dim=1)
                merged_idx = torch.cat(per_gpu_idx, dim=1)
                _, final_pos = torch.topk(merged_vals, k=top_k, dim=1)
                torch.gather(merged_idx, 1, final_pos)

        torch.cuda.synchronize()

    for _ in range(warmup_runs):
        run_once()

    t0 = time.perf_counter()
    for _ in range(timed_runs):
        run_once()
    t1 = time.perf_counter()

    total_queries = queries_cast.shape[0] * timed_runs
    total_ms = (t1 - t0) * 1000.0

    return {
        "framework": "torch",
        "precision": precision,
        "gpu_mode": gpu_mode,
        "metric": metric,
        "setup_time_s": round(setup_t1 - setup_t0, 6),
        "total_timed_time_s": round(t1 - t0, 6),
        "queries_timed": total_queries,
        "avg_time_per_query_ms": round(total_ms / total_queries, 6),
        "batch_size": batch_size,
    }


def bench_faiss(
    bank_np: np.ndarray,
    queries_np: np.ndarray,
    precision: str,
    gpu_mode: str,
    metric: str,
    top_k: int,
    warmup_runs: int,
    timed_runs: int,
    batch_size: int,
    nprobe: int,
) -> Dict[str, Any]:
    import faiss

    bank_f32 = np.asarray(bank_np, dtype=np.float32)
    queries_f32 = np.asarray(queries_np, dtype=np.float32)

    cpu_index = faiss.IndexFlatL2(bank_f32.shape[1]) if metric == "l2" else faiss.IndexFlatIP(bank_f32.shape[1])

    setup_t0 = time.perf_counter()

    use_float16 = precision == "fp16"
    if gpu_mode == "single":
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = use_float16
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
    else:
        ngpu = faiss.get_num_gpus()
        if ngpu < 2:
            raise RuntimeError("Multi-GPU mode requested but fewer than 2 GPUs detected")
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = use_float16
        co.shard = True
        index = faiss.index_cpu_to_gpus_list(cpu_index, co, list(range(ngpu)))

    index.add(bank_f32)

    if hasattr(index, "nprobe"):
        index.nprobe = nprobe

    setup_t1 = time.perf_counter()

    def run_once() -> None:
        for qs, qe in chunk_iter(queries_f32.shape[0], batch_size):
            index.search(queries_f32[qs:qe], top_k)

    for _ in range(warmup_runs):
        run_once()

    t0 = time.perf_counter()
    for _ in range(timed_runs):
        run_once()
    t1 = time.perf_counter()

    total_queries = queries_f32.shape[0] * timed_runs
    total_ms = (t1 - t0) * 1000.0

    return {
        "framework": "faiss",
        "precision": precision,
        "gpu_mode": gpu_mode,
        "metric": metric,
        "setup_time_s": round(setup_t1 - setup_t0, 6),
        "total_timed_time_s": round(t1 - t0, 6),
        "queries_timed": total_queries,
        "avg_time_per_query_ms": round(total_ms / total_queries, 6),
        "batch_size": batch_size,
        "nprobe": nprobe,
    }


def _bench_faiss_worker(args: Dict[str, Any], result_queue: "mp.Queue[Dict[str, Any]]") -> None:
    try:
        result = bench_faiss(**args)
        result_queue.put({"ok": True, "result": result})
    except Exception as e:
        result_queue.put({"ok": False, "error": str(e)})


def bench_faiss_isolated(**kwargs: Any) -> Dict[str, Any]:
    """Run FAISS benchmark in a subprocess so native crashes don't abort the full run."""
    ctx = mp.get_context("spawn")
    result_queue: "mp.Queue[Dict[str, Any]]" = ctx.Queue()
    proc = ctx.Process(target=_bench_faiss_worker, args=(kwargs, result_queue))
    proc.start()
    proc.join()

    if proc.exitcode == 0:
        if result_queue.empty():
            raise RuntimeError("FAISS subprocess exited successfully but returned no result")
        msg = result_queue.get()
        if msg.get("ok"):
            return msg["result"]
        raise RuntimeError(msg.get("error", "Unknown FAISS subprocess error"))

    if proc.exitcode is None:
        raise RuntimeError("FAISS subprocess did not terminate cleanly")

    if proc.exitcode < 0:
        raise RuntimeError(f"FAISS subprocess terminated by signal {-proc.exitcode}")

    raise RuntimeError(f"FAISS subprocess exited with code {proc.exitcode}")


def run_experiment(exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_path = Path(exp_cfg["dataset_path"])
    data = load_dataset(dataset_path)
    bank, queries = data["bank"], data["queries"]

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for precision in exp_cfg["precisions"]:
        if precision not in {"fp32", "fp16"}:
            failures.append({"precision": precision, "error": "Unsupported precision"})
            continue
        for gpu_mode in exp_cfg["gpu_modes"]:
            if gpu_mode not in {"single", "multi"}:
                failures.append({"precision": precision, "gpu_mode": gpu_mode, "error": "Unsupported gpu_mode"})
                continue

            for framework in ("torch", "faiss"):
                try:
                    if framework == "torch":
                        result = bench_torch(
                            bank_np=bank,
                            queries_np=queries,
                            precision=precision,
                            gpu_mode=gpu_mode,
                            metric=exp_cfg["metric"],
                            top_k=exp_cfg["top_k"],
                            warmup_runs=exp_cfg["warmup_runs"],
                            timed_runs=exp_cfg["timed_runs"],
                            batch_size=exp_cfg["torch_batch_size"],
                        )
                    else:
                        result = bench_faiss_isolated(
                            bank_np=bank,
                            queries_np=queries,
                            precision=precision,
                            gpu_mode=gpu_mode,
                            metric=exp_cfg["metric"],
                            top_k=exp_cfg["top_k"],
                            warmup_runs=exp_cfg["warmup_runs"],
                            timed_runs=exp_cfg["timed_runs"],
                            batch_size=exp_cfg["faiss_batch_size"],
                            nprobe=exp_cfg["nprobe"],
                        )
                    result["experiment_name"] = exp_cfg["name"]
                    result["dataset_path"] = str(dataset_path)
                    print(
                        f"[OK] exp={exp_cfg['name']} {framework:5s} "
                        f"precision={precision} gpu_mode={gpu_mode} "
                        f"avg={result['avg_time_per_query_ms']:.6f} ms/query"
                    )
                    results.append(result)
                except Exception as e:
                    err = {
                        "experiment_name": exp_cfg["name"],
                        "dataset_path": str(dataset_path),
                        "framework": framework,
                        "precision": precision,
                        "gpu_mode": gpu_mode,
                        "error": str(e),
                    }
                    print(
                        f"[FAIL] exp={exp_cfg['name']} {framework} "
                        f"precision={precision} gpu_mode={gpu_mode}: {e}"
                    )
                    failures.append(err)

    return {
        "experiment_name": exp_cfg["name"],
        "dataset_path": str(dataset_path),
        "dataset_meta": data["meta"],
        "settings": {
            "metric": exp_cfg["metric"],
            "top_k": exp_cfg["top_k"],
            "precisions": exp_cfg["precisions"],
            "gpu_modes": exp_cfg["gpu_modes"],
            "warmup_runs": exp_cfg["warmup_runs"],
            "timed_runs": exp_cfg["timed_runs"],
            "torch_batch_size": exp_cfg["torch_batch_size"],
            "faiss_batch_size": exp_cfg["faiss_batch_size"],
            "nprobe": exp_cfg["nprobe"],
        },
        "results": results,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    experiment_reports = []
    all_results: List[Dict[str, Any]] = []
    all_failures: List[Dict[str, Any]] = []

    for exp_cfg in config["experiments"]:
        print(f"\n=== Running {exp_cfg['name']} ({exp_cfg['dataset_path']}) ===")
        report = run_experiment(exp_cfg)
        experiment_reports.append(report)
        all_results.extend(report["results"])
        all_failures.extend(report["failures"])

    output_json = Path(config["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "config_path": str(config_path),
        "timestamp_epoch_s": time.time(),
        "num_experiments": len(experiment_reports),
        "experiments": experiment_reports,
        "results": all_results,
        "failures": all_failures,
    }

    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nBenchmark JSON written to: {output_json}")


if __name__ == "__main__":
    main()
