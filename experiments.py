"""Experiment runners for depth vs width latency study."""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from benchmark import BenchmarkResult, benchmark_config, quick_decode_latency, measure_total_response_time, create_model, reset_memory_stats, get_device
import numpy as np
from config import (
    ModelConfig,
    count_params,
    create_config_for_search,
    get_depth_sweep_configs,
    get_param_matched_configs,
    get_width_sweep_configs,
)

RESULTS_DIR = Path("results/raw")


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def result_to_dict(result: BenchmarkResult) -> dict:
    """Convert BenchmarkResult to dict for DataFrame."""
    return {
        "config_name": result.config_name,
        "num_layers": result.num_layers,
        "hidden_size": result.hidden_size,
        "param_count": result.param_count,
        "prefill_latency_mean": result.prefill_latency_mean,
        "prefill_latency_std": result.prefill_latency_std,
        "prefill_latency_p50": result.prefill_latency_p50,
        "prefill_latency_p95": result.prefill_latency_p95,
        "prefill_latency_p99": result.prefill_latency_p99,
        "decode_latency_mean": result.decode_latency_mean,
        "decode_latency_std": result.decode_latency_std,
        "decode_latency_p50": result.decode_latency_p50,
        "decode_latency_p95": result.decode_latency_p95,
        "decode_latency_p99": result.decode_latency_p99,
        "total_time_50tok": result.total_time_50tok,
        "total_time_100tok": result.total_time_100tok,
        "total_time_200tok": result.total_time_200tok,
        "throughput_batch1": result.throughput_batch1,
        "throughput_batch8": result.throughput_batch8,
        "throughput_batch16": result.throughput_batch16,
        "throughput_batch32": result.throughput_batch32,
        "peak_memory_gb": result.peak_memory_gb,
    }


def save_results(results: List[BenchmarkResult], filename: str):
    """Save results to CSV."""
    ensure_results_dir()
    df = pd.DataFrame([result_to_dict(r) for r in results])
    filepath = RESULTS_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def run_experiment(
    configs: List[ModelConfig],
    experiment_name: str,
    resume_from: Optional[str] = None,
) -> List[BenchmarkResult]:
    """
    Run benchmark on a list of configs, saving incrementally.

    Args:
        configs: List of model configurations to benchmark
        experiment_name: Name for the experiment (used in filename)
        resume_from: If provided, skip configs already in this CSV file
    """
    ensure_results_dir()

    # Check for existing results to resume from
    completed_configs = set()
    existing_results = []
    output_file = RESULTS_DIR / f"{experiment_name}.csv"

    if resume_from and Path(resume_from).exists():
        existing_df = pd.read_csv(resume_from)
        completed_configs = set(existing_df["config_name"].tolist())
        existing_results = existing_df.to_dict("records")
        print(f"Resuming from {resume_from}, {len(completed_configs)} configs already completed")
    elif output_file.exists():
        existing_df = pd.read_csv(output_file)
        completed_configs = set(existing_df["config_name"].tolist())
        existing_results = existing_df.to_dict("records")
        print(f"Found existing results, {len(completed_configs)} configs already completed")

    results = []
    for config in tqdm(configs, desc=experiment_name):
        if config.name in completed_configs:
            print(f"Skipping {config.name} (already completed)")
            continue

        try:
            result = benchmark_config(config)
            results.append(result)

            # Save incrementally
            all_results_dicts = existing_results + [result_to_dict(r) for r in results]
            df = pd.DataFrame(all_results_dicts)
            df.to_csv(output_file, index=False)

        except Exception as e:
            print(f"Error benchmarking {config.name}: {e}")
            continue

    return results


def run_param_matched_sweep() -> List[BenchmarkResult]:
    """
    Experiment 1: Parameter-matched sweep (~300M params each).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Parameter-Matched Sweep (~300M params)")
    print("=" * 70)

    configs = get_param_matched_configs()
    return run_experiment(configs, "param_matched_sweep")


def run_depth_sweep() -> List[BenchmarkResult]:
    """
    Experiment 2: Depth sweep at fixed width (576).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Depth Sweep (fixed width=576)")
    print("=" * 70)

    configs = get_depth_sweep_configs()
    return run_experiment(configs, "depth_sweep")


def run_width_sweep() -> List[BenchmarkResult]:
    """
    Experiment 3: Width sweep at fixed depth (40 layers).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Width Sweep (fixed depth=40)")
    print("=" * 70)

    configs = get_width_sweep_configs()
    return run_experiment(configs, "width_sweep")


def quick_total_time(config: ModelConfig, gen_tokens: int = 100) -> float:
    """
    Quick total response time measurement for search.
    Returns mean total time in ms for prefill + gen_tokens.
    """
    reset_memory_stats()
    model = create_model(config)

    times = measure_total_response_time(model, prompt_len=512, gen_tokens=gen_tokens, warmup_runs=1, num_runs=2)

    del model
    reset_memory_stats()

    return float(np.mean(times)) if times else float('inf')


def find_width_for_target_time(
    num_layers: int,
    target_time_ms: float,
    tolerance: float = 0.10,
    min_hidden: int = 256,
    max_hidden: int = 8192,
) -> tuple[int, float, int]:
    """
    Binary search for hidden size that achieves target total response time.

    Returns: (hidden_size, actual_time, param_count)
    """
    low, high = min_hidden, max_hidden
    best_hidden = min_hidden
    best_time = float('inf')
    best_params = 0

    print(f"  Searching for width at {num_layers}L to match {target_time_ms:.0f}ms...")

    while high - low > 64:
        mid = ((low + high) // 2 // 64) * 64
        config = create_config_for_search(num_layers, mid)
        params = count_params(config)

        try:
            total_time = quick_total_time(config, gen_tokens=100)
            print(f"    hidden={mid}: time={total_time:.0f}ms, params={params/1e9:.2f}B")

            # Track the closest match
            if abs(total_time - target_time_ms) < abs(best_time - target_time_ms):
                best_hidden = mid
                best_time = total_time
                best_params = params

            # Binary search: if too fast, go wider; if too slow, go narrower
            if total_time < target_time_ms * (1 - tolerance):
                low = mid  # Too fast, try wider
            elif total_time > target_time_ms * (1 + tolerance):
                high = mid  # Too slow, try narrower
            else:
                # Within tolerance
                return mid, total_time, params

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    hidden={mid}: OOM")
                high = mid  # Too big, try smaller
            else:
                raise

    return best_hidden, best_time, best_params


def run_max_width_search() -> List[BenchmarkResult]:
    """
    Experiment 4: Max Width at Each Depth.

    For each depth level, find the maximum width that fits in memory,
    then compare total response time (prefill + 100 tokens) and parameters.

    This shows: shallower models can be BOTH faster AND have more parameters.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Max Width at Each Depth")
    print("=" * 70)

    # Test depths from deep (baseline) to shallow
    test_depths = [80, 60, 40, 20, 10]

    results = []
    summary_data = []

    for depth in test_depths:
        print(f"\n--- Testing {depth} layers ---")
        hidden, total_time, params = find_max_width_for_depth(depth, max_hidden=4096)

        # Run full benchmark
        config = create_config_for_search(depth, hidden)
        result = benchmark_config(config)
        results.append(result)

        summary_data.append({
            "num_layers": depth,
            "hidden_size": hidden,
            "param_count": params,
            "total_time_100tok": result.total_time_100tok,
            "decode_latency_mean": result.decode_latency_mean,
            "prefill_latency_mean": result.prefill_latency_mean,
        })

    # Get baseline for comparison
    baseline = summary_data[0]  # 80 layers
    for item in summary_data:
        item["params_vs_baseline"] = item["param_count"] / baseline["param_count"]
        item["speedup_vs_baseline"] = baseline["total_time_100tok"] / item["total_time_100tok"]

    # Print summary
    print("\n" + "=" * 70)
    print("MAX WIDTH RESULTS: Shallower = Faster AND Bigger")
    print("=" * 70)
    print(f"{'Layers':>8} {'Hidden':>8} {'Params':>10} {'Time(100t)':>12} {'vs 80L Params':>14} {'Speedup':>10}")
    print("-" * 72)
    for m in summary_data:
        print(f"{m['num_layers']:>8} {m['hidden_size']:>8} {m['param_count']/1e9:>9.2f}B "
              f"{m['total_time_100tok']:>11.0f}ms {m['params_vs_baseline']:>13.1f}x {m['speedup_vs_baseline']:>9.1f}x")

    # Save results
    ensure_results_dir()
    df = pd.DataFrame([result_to_dict(r) for r in results])
    df.to_csv(RESULTS_DIR / "max_width_search.csv", index=False)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(RESULTS_DIR / "max_width_summary.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR}")

    return results


def run_all_experiments():
    """Run all experiments."""
    run_param_matched_sweep()
    run_depth_sweep()
    run_width_sweep()
    run_max_width_search()


if __name__ == "__main__":
    run_all_experiments()
