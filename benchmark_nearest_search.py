#!/usr/bin/env python3
"""Benchmark nearest-neighbor search with Torch vs FAISS on generated datasets.

This script reads a JSON config and produces a benchmark JSON report.
It supports:
- fp32 + fp16 runs
- single-GPU + multi-GPU modes
- Torch brute-force top-k search
- FAISS Flat index search (GPU)
- configurable frameworks per run: torch, faiss, or both
- per-run GPU memory usage reporting (per GPU in multi-GPU mode)
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
import subprocess
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
    "frameworks": ["torch", "faiss"],
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


def _query_gpu_memory_used_mib() -> List[int] | None:
    """Return per-GPU used memory (MiB) from nvidia-smi, or None if unavailable."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    values: List[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            return None
    return values


def _build_gpu_memory_report(
    before_mib: Sequence[int] | None,
    after_setup_mib: Sequence[int] | None,
    after_timed_mib: Sequence[int] | None,
    peak_mib: Sequence[int] | None,
) -> Dict[str, Any] | None:
    if not before_mib:
        return None

    n = len(before_mib)
    if n == 0:
        return None

    def deltas(values: Sequence[int] | None) -> List[int] | None:
        if values is None or len(values) < n:
            return None
        return [max(0, values[i] - before_mib[i]) for i in range(n)]

    setup_delta = deltas(after_setup_mib)
    timed_end_delta = deltas(after_timed_mib)
    peak_delta = deltas(peak_mib)

    return {
        "unit": "MiB",
        "before_used_mib": list(before_mib),
        "after_setup_used_mib": list(after_setup_mib) if after_setup_mib is not None else None,
        "after_timed_used_mib": list(after_timed_mib) if after_timed_mib is not None else None,
        "peak_used_mib": list(peak_mib) if peak_mib is not None else None,
        "setup_delta_mib": setup_delta,
        "timed_end_delta_mib": timed_end_delta,
        "peak_delta_mib": peak_delta,
    }


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
    if num_gpus < 1:
        raise RuntimeError("No CUDA devices detected for Torch benchmarking")

    # Probe CUDA devices up-front because some environments expose a CUDA runtime
    # but reject specific device ordinals at first allocation with
    # "Invalid device argument". We only benchmark devices that pass a small
    # allocation test.
    available_device_ids = []
    for gpu_id in range(num_gpus):
        try:
            torch.empty((1,), device=torch.device("cuda", gpu_id))
            available_device_ids.append(gpu_id)
        except Exception:
            continue

    if not available_device_ids:
        raise RuntimeError("CUDA is available but no allocatable GPU device was found")

    if gpu_mode == "multi" and len(available_device_ids) < 2:
        raise RuntimeError("Multi-GPU mode requested but fewer than 2 allocatable GPUs detected")

    dtype = torch.float16 if precision == "fp16" else torch.float32
    bank_cast = bank_np.astype(np.float32 if precision == "fp32" else np.float16, copy=False)
    queries_cast = queries_np.astype(np.float32 if precision == "fp32" else np.float16, copy=False)

    tracked_gpu_ids = [available_device_ids[0]] if gpu_mode == "single" else available_device_ids
    for gpu_id in tracked_gpu_ids:
        torch.cuda.reset_peak_memory_stats(gpu_id)

    mem_before = [int(torch.cuda.memory_allocated(gpu_id) / (1024 * 1024)) for gpu_id in tracked_gpu_ids]

    setup_t0 = time.perf_counter()

    if gpu_mode == "single":
        main_device = torch.device("cuda", tracked_gpu_ids[0])
        bank_t = torch.from_numpy(bank_cast).to(device=main_device, dtype=dtype, non_blocking=True)
        shard_tensors = [bank_t]
    else:
        shard_tensors = []
        per_gpu = int(np.ceil(bank_cast.shape[0] / len(tracked_gpu_ids)))
        for gpu_idx, gpu_id in enumerate(tracked_gpu_ids):
            s = gpu_idx * per_gpu
            e = min((gpu_idx + 1) * per_gpu, bank_cast.shape[0])
            if s >= e:
                break
            shard = torch.from_numpy(bank_cast[s:e]).to(
                device=torch.device("cuda", gpu_id), dtype=dtype, non_blocking=True
            )
            shard_tensors.append(shard)

    setup_t1 = time.perf_counter()
    mem_after_setup = [int(torch.cuda.memory_allocated(gpu_id) / (1024 * 1024)) for gpu_id in tracked_gpu_ids]

    def run_once() -> None:
        for qs, qe in chunk_iter(queries_cast.shape[0], batch_size):
            q_np = queries_cast[qs:qe]
            if gpu_mode == "single":
                q = torch.from_numpy(q_np).to(device=shard_tensors[0].device, dtype=dtype, non_blocking=True)
                sims = q @ shard_tensors[0].T
                if metric == "l2":
                    sims = -torch.cdist(q, shard_tensors[0])
                torch.topk(sims, k=top_k, dim=1)
            else:
                per_gpu_vals = []
                per_gpu_idx = []
                offset = 0
                gather_device = shard_tensors[0].device
                for shard in shard_tensors:
                    q = torch.from_numpy(q_np).to(device=shard.device, dtype=dtype, non_blocking=True)
                    sims = q @ shard.T
                    if metric == "l2":
                        sims = -torch.cdist(q, shard)
                    vals, idx = torch.topk(sims, k=top_k, dim=1)
                    per_gpu_vals.append(vals.to(device=gather_device))
                    per_gpu_idx.append((idx + offset).to(device=gather_device))
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

    mem_after_timed = [int(torch.cuda.memory_allocated(gpu_id) / (1024 * 1024)) for gpu_id in tracked_gpu_ids]
    mem_peak = [int(torch.cuda.max_memory_allocated(gpu_id) / (1024 * 1024)) for gpu_id in tracked_gpu_ids]

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
        "gpu_memory": {
            "unit": "MiB",
            "gpu_ids": tracked_gpu_ids,
            "before_allocated_mib": mem_before,
            "after_setup_allocated_mib": mem_after_setup,
            "after_timed_allocated_mib": mem_after_timed,
            "peak_allocated_mib": mem_peak,
            "setup_delta_mib": [max(0, mem_after_setup[i] - mem_before[i]) for i in range(len(tracked_gpu_ids))],
            "timed_end_delta_mib": [max(0, mem_after_timed[i] - mem_before[i]) for i in range(len(tracked_gpu_ids))],
            "peak_delta_mib": [max(0, mem_peak[i] - mem_before[i]) for i in range(len(tracked_gpu_ids))],
        },
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

    mem_before = _query_gpu_memory_used_mib()
    mem_peak = list(mem_before) if mem_before is not None else None

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
    mem_after_setup = _query_gpu_memory_used_mib()
    if mem_peak is not None and mem_after_setup is not None and len(mem_after_setup) == len(mem_peak):
        mem_peak = [max(mem_peak[i], mem_after_setup[i]) for i in range(len(mem_peak))]

    def run_once() -> None:
        for qs, qe in chunk_iter(queries_f32.shape[0], batch_size):
            index.search(queries_f32[qs:qe], top_k)

    for _ in range(warmup_runs):
        run_once()
        snap = _query_gpu_memory_used_mib()
        if mem_peak is not None and snap is not None and len(snap) == len(mem_peak):
            mem_peak = [max(mem_peak[i], snap[i]) for i in range(len(mem_peak))]

    t0 = time.perf_counter()
    for _ in range(timed_runs):
        run_once()
        snap = _query_gpu_memory_used_mib()
        if mem_peak is not None and snap is not None and len(snap) == len(mem_peak):
            mem_peak = [max(mem_peak[i], snap[i]) for i in range(len(mem_peak))]
    t1 = time.perf_counter()

    mem_after_timed = _query_gpu_memory_used_mib()
    if mem_peak is not None and mem_after_timed is not None and len(mem_after_timed) == len(mem_peak):
        mem_peak = [max(mem_peak[i], mem_after_timed[i]) for i in range(len(mem_peak))]

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
        "gpu_memory": _build_gpu_memory_report(mem_before, mem_after_setup, mem_after_timed, mem_peak),
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


def _normalize_frameworks(frameworks: Sequence[str]) -> List[str]:
    if isinstance(frameworks, str):
        raise ValueError("frameworks must be a list, e.g. [\"torch\", \"faiss\"]")

    normalized: List[str] = []
    for framework in frameworks:
        if framework not in {"torch", "faiss"}:
            raise ValueError(f"Unsupported framework: {framework}")
        if framework not in normalized:
            normalized.append(framework)
    if not normalized:
        raise ValueError("At least one framework must be configured")
    return normalized


def run_experiment(exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_path = Path(exp_cfg["dataset_path"])
    data = load_dataset(dataset_path)
    bank, queries = data["bank"], data["queries"]

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    raw_frameworks = exp_cfg["frameworks"]
    try:
        frameworks = _normalize_frameworks(raw_frameworks)
    except ValueError as e:
        failures.append(
            {
                "experiment_name": exp_cfg["name"],
                "dataset_path": str(dataset_path),
                "error": str(e),
            }
        )
        frameworks = []

    for precision in exp_cfg["precisions"]:
        if precision not in {"fp32", "fp16"}:
            failures.append({"precision": precision, "error": "Unsupported precision"})
            continue
        for gpu_mode in exp_cfg["gpu_modes"]:
            if gpu_mode not in {"single", "multi"}:
                failures.append({"precision": precision, "gpu_mode": gpu_mode, "error": "Unsupported gpu_mode"})
                continue

            for framework in frameworks:
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
            "frameworks": frameworks if frameworks else raw_frameworks,
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
