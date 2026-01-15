"""
Compare Baguettotron vs Gemma 3 12B architectures.

Baguettotron: 80 layers, 576 hidden (~321M params) - extreme deep-narrow
Gemma 3 12B: 48 layers, 3840 hidden (~12B params) - more balanced
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from pathlib import Path

from benchmark import (
    measure_total_response_time,
    measure_decode,
    measure_prefill,
    create_model,
    reset_memory_stats,
)
from config import create_config_for_search, count_params, ModelConfig

RESULTS_DIR = Path("results/raw")


def measure_config(config, gen_tokens=100):
    """Measure total time, decode latency, and prefill latency."""
    reset_memory_stats()
    model = create_model(config)

    total_times = measure_total_response_time(model, prompt_len=512, gen_tokens=gen_tokens, warmup_runs=2, num_runs=3)
    total_time = float(np.mean(total_times)) if total_times else float('inf')

    decode_latencies = measure_decode(model, prompt_len=128, gen_len=30, warmup_runs=2, num_runs=3)
    decode_latency = float(np.mean(decode_latencies)) if decode_latencies else float('inf')

    prefill_latencies = measure_prefill(model, seq_len=512, warmup_runs=2, num_runs=3)
    prefill_latency = float(np.mean(prefill_latencies)) if prefill_latencies else float('inf')

    del model
    reset_memory_stats()

    return total_time, decode_latency, prefill_latency


def run_comparison():
    """Compare Baguettotron vs Gemma 3 12B."""
    print("=" * 70, flush=True)
    print("ARCHITECTURE COMPARISON: Baguettotron vs Gemma 3 12B", flush=True)
    print("=" * 70, flush=True)

    # Define the two architectures
    configs = [
        {
            "name": "Baguettotron",
            "num_layers": 80,
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_attention_heads": 9,
            "num_kv_heads": 3,
        },
        {
            "name": "Gemma-3-12B",
            "num_layers": 48,
            "hidden_size": 3840,
            "intermediate_size": 15360,
            "num_attention_heads": 16,
            "num_kv_heads": 8,
        },
    ]

    results = []

    for cfg in configs:
        print(f"\n[Measuring {cfg['name']}]", flush=True)
        print(f"  {cfg['num_layers']} layers, {cfg['hidden_size']} hidden", flush=True)

        config = ModelConfig(
            name=cfg["name"],
            num_layers=cfg["num_layers"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_attention_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_kv_heads"],
        )

        params = count_params(config)
        print(f"  Parameters: {params/1e9:.2f}B", flush=True)

        total_time, decode_latency, prefill_latency = measure_config(config, gen_tokens=100)

        print(f"  Total time (512 prompt + 100 gen): {total_time:.0f}ms ({total_time/1000:.2f}s)", flush=True)
        print(f"  Decode latency: {decode_latency:.2f}ms/token", flush=True)
        print(f"  Prefill latency: {prefill_latency:.2f}ms", flush=True)

        results.append({
            "name": cfg["name"],
            "num_layers": cfg["num_layers"],
            "hidden_size": cfg["hidden_size"],
            "param_count": params,
            "total_time_ms": total_time,
            "decode_ms_per_token": decode_latency,
            "prefill_ms": prefill_latency,
        })

    # Print comparison
    print("\n" + "=" * 70, flush=True)
    print("COMPARISON RESULTS", flush=True)
    print("=" * 70, flush=True)

    baguette = results[0]
    gemma = results[1]

    print(f"\n{'':20} {'Baguettotron':>15} {'Gemma 3 12B':>15} {'Ratio':>10}", flush=True)
    print("-" * 65, flush=True)
    print(f"{'Layers':20} {baguette['num_layers']:>15} {gemma['num_layers']:>15} {baguette['num_layers']/gemma['num_layers']:>10.2f}x", flush=True)
    print(f"{'Hidden size':20} {baguette['hidden_size']:>15} {gemma['hidden_size']:>15} {gemma['hidden_size']/baguette['hidden_size']:>10.2f}x", flush=True)
    print(f"{'Parameters':20} {baguette['param_count']/1e9:>14.2f}B {gemma['param_count']/1e9:>14.2f}B {gemma['param_count']/baguette['param_count']:>10.1f}x", flush=True)
    print(f"{'Total time':20} {baguette['total_time_ms']/1000:>14.2f}s {gemma['total_time_ms']/1000:>14.2f}s {baguette['total_time_ms']/gemma['total_time_ms']:>10.2f}x", flush=True)
    print(f"{'Decode latency':20} {baguette['decode_ms_per_token']:>13.1f}ms {gemma['decode_ms_per_token']:>13.1f}ms {baguette['decode_ms_per_token']/gemma['decode_ms_per_token']:>10.2f}x", flush=True)
    print(f"{'Prefill latency':20} {baguette['prefill_ms']:>13.1f}ms {gemma['prefill_ms']:>13.1f}ms {gemma['prefill_ms']/baguette['prefill_ms']:>10.2f}x", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("KEY INSIGHT:", flush=True)
    print(f"  Gemma 3 12B has {gemma['param_count']/baguette['param_count']:.0f}x more parameters", flush=True)
    print(f"  But is {baguette['total_time_ms']/gemma['total_time_ms']:.1f}x FASTER for a full response!", flush=True)
    print("=" * 70, flush=True)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "architecture_comparison.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'architecture_comparison.csv'}", flush=True)

    return results


if __name__ == "__main__":
    run_comparison()
