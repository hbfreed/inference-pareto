"""
Time-Matched Experiment: At equal response time, how many more parameters can shallow-wide hold?

The question: At the latency of Baguettotron (80L, 576h, ~321M params),
how much wider can shallower models be while maintaining the same response time?
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from benchmark import (
    measure_total_response_time,
    measure_decode,
    measure_prefill,
    create_model,
    reset_memory_stats,
)
from config import create_config_for_search, count_params

RESULTS_DIR = Path("results/raw")


def measure_config(config, gen_tokens=100):
    """
    Measure both total response time and decode-only latency.

    Returns: (total_time_ms, decode_latency_ms_per_token, prefill_ms)
    """
    reset_memory_stats()
    model = create_model(config)

    # Total time (prefill + all tokens)
    total_times = measure_total_response_time(model, prompt_len=512, gen_tokens=gen_tokens, warmup_runs=1, num_runs=2)
    total_time = float(np.mean(total_times)) if total_times else float('inf')

    # Decode latency (per token)
    decode_latencies = measure_decode(model, prompt_len=128, gen_len=30, warmup_runs=1, num_runs=2)
    decode_latency = float(np.mean(decode_latencies)) if decode_latencies else float('inf')

    # Prefill latency
    prefill_latencies = measure_prefill(model, seq_len=512, warmup_runs=1, num_runs=3)
    prefill_latency = float(np.mean(prefill_latencies)) if prefill_latencies else float('inf')

    del model
    reset_memory_stats()

    return total_time, decode_latency, prefill_latency


def quick_total_time(config, gen_tokens=100):
    """Fast measurement of just total time for binary search."""
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
) -> tuple[int, float, int]:
    """
    Search for hidden size that achieves target total response time.
    Start small and increase until we match or exceed target time.

    Returns: (hidden_size, actual_time_ms, param_count)
    """
    print(f"\n  Searching for width at {num_layers}L to match {target_time_ms:.0f}ms...", flush=True)

    # Start from a reasonable width and increase
    # For shallow models, we'll need to go wider to match deep model's time
    hidden_sizes = [512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]

    best_hidden = 512
    best_time = 0.0
    best_params = 0

    for hidden in hidden_sizes:
        config = create_config_for_search(num_layers, hidden)
        params = count_params(config)

        try:
            total_time = quick_total_time(config, gen_tokens=100)
            print(f"    hidden={hidden}: time={total_time:.0f}ms ({total_time/1000:.2f}s), params={params/1e9:.2f}B", flush=True)

            # Track best match (closest to target)
            if best_time == 0 or abs(total_time - target_time_ms) < abs(best_time - target_time_ms):
                best_hidden = hidden
                best_time = total_time
                best_params = params

            # If we've reached or exceeded target time, we found it
            if total_time >= target_time_ms * (1 - tolerance):
                print(f"    Found match: {hidden}h = {total_time:.0f}ms", flush=True)
                return hidden, total_time, params

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"    hidden={hidden}: OOM, stopping search", flush=True)
                break
            else:
                raise

    print(f"    Best found: {best_hidden}h = {best_time:.0f}ms (may not match target)", flush=True)
    return best_hidden, best_time, best_params


def run_time_matched_experiment():
    """
    Main experiment: Find width at each depth that matches baseline response time.
    """
    print("=" * 70, flush=True)
    print("TIME-MATCHED EXPERIMENT", flush=True)
    print("At equal response time, how many more params can shallow-wide hold?", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Measure baseline (Baguettotron-like: 80L, 576h)
    print("\n[Step 1] Measuring baseline: 80 layers, 576 hidden...", flush=True)
    baseline_config = create_config_for_search(80, 576)
    baseline_params = count_params(baseline_config)
    baseline_total, baseline_decode, baseline_prefill = measure_config(baseline_config, gen_tokens=100)

    print(f"  Baseline: {baseline_params/1e9:.3f}B params", flush=True)
    print(f"    Total time (512 prompt + 100 gen): {baseline_total:.0f}ms ({baseline_total/1000:.2f}s)", flush=True)
    print(f"    Decode latency: {baseline_decode:.2f}ms/token", flush=True)
    print(f"    Prefill latency: {baseline_prefill:.2f}ms", flush=True)

    # Step 2: For each depth, find width that matches baseline total time
    print(f"\n[Step 2] Finding time-matched configurations (target: {baseline_total:.0f}ms)...", flush=True)

    depths_to_test = [60, 40, 20, 10]
    results = []

    # Add baseline to results
    results.append({
        "num_layers": 80,
        "hidden_size": 576,
        "param_count": baseline_params,
        "total_time_ms": baseline_total,
        "decode_ms_per_token": baseline_decode,
        "prefill_ms": baseline_prefill,
        "params_vs_baseline": 1.0,
    })

    for depth in depths_to_test:
        # First, binary search to find the right width
        hidden, _, params = find_width_for_target_time(depth, baseline_total)

        # Then do a proper measurement with all metrics
        config = create_config_for_search(depth, hidden)
        total_time, decode_latency, prefill_latency = measure_config(config, gen_tokens=100)

        results.append({
            "num_layers": depth,
            "hidden_size": hidden,
            "param_count": params,
            "total_time_ms": total_time,
            "decode_ms_per_token": decode_latency,
            "prefill_ms": prefill_latency,
            "params_vs_baseline": params / baseline_params,
        })

        print(f"\n  {depth}L @ {hidden}h: {params/1e9:.2f}B params")
        print(f"    Total: {total_time:.0f}ms, Decode: {decode_latency:.2f}ms/tok, Prefill: {prefill_latency:.2f}ms")

    # Step 3: Print summary
    print("\n" + "=" * 70)
    print("RESULTS: Same Response Time, Different Capacity")
    print("=" * 70)
    print(f"{'Layers':>8} {'Hidden':>8} {'Total':>10} {'Decode':>12} {'Prefill':>10} {'Params':>10} {'vs Base':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['num_layers']:>8} {r['hidden_size']:>8} {r['total_time_ms']/1000:>9.2f}s "
              f"{r['decode_ms_per_token']:>10.1f}ms {r['prefill_ms']:>9.1f}ms "
              f"{r['param_count']/1e9:>9.2f}B {r['params_vs_baseline']:>7.1f}x")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "time_matched.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'time_matched.csv'}")

    return results


if __name__ == "__main__":
    import torch
    run_time_matched_experiment()
