"""Compare actual Baguettotron vs Gemma 3 12B models from HuggingFace."""

import sys
import time
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

RESULTS_DIR = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_TOKENS = 512
GEN_TOKENS = 100
WARMUP_RUNS = 2
BENCHMARK_RUNS = 3


def measure_model(model_id: str, name: str) -> dict:
    """Load and benchmark a real model from HuggingFace."""
    print(f"\n{'='*60}", flush=True)
    print(f"Loading {name} ({model_id})...", flush=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e9:.2f}B", flush=True)

    # Create dummy input (use padding token or eos token)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((1, PROMPT_TOKENS), pad_token_id, dtype=torch.long, device=DEVICE)

    # Warmup
    print(f"Warming up ({WARMUP_RUNS} runs)...", flush=True)
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

    # Benchmark
    print(f"Benchmarking ({BENCHMARK_RUNS} runs, {GEN_TOKENS} tokens each)...", flush=True)
    times = []

    with torch.no_grad():
        for run in range(BENCHMARK_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()

            output = model.generate(
                input_ids,
                max_new_tokens=GEN_TOKENS,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {run+1}: {elapsed:.2f}s", flush=True)

    avg_time = sum(times) / len(times)
    decode_latency = (avg_time * 1000) / GEN_TOKENS  # Approximate (includes prefill)

    print(f"Average: {avg_time:.2f}s ({decode_latency:.1f}ms/token)", flush=True)

    # Clean up to free memory for next model
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return {
        "name": name,
        "model_id": model_id,
        "param_count": param_count,
        "total_time_ms": avg_time * 1000,
        "decode_ms_per_token": decode_latency,
    }


def main():
    print("Real Model Comparison: Baguettotron vs Gemma 3 12B", flush=True)
    print(f"Config: {PROMPT_TOKENS} prompt tokens, {GEN_TOKENS} generated tokens", flush=True)

    models = [
        ("PleIAs/Baguettotron", "Baguettotron"),
        ("google/gemma-3-12b-it", "Gemma-3-12B"),
    ]

    results = []
    for model_id, name in models:
        try:
            result = measure_model(model_id, name)
            results.append(result)
        except Exception as e:
            print(f"ERROR loading {name}: {e}", flush=True)
            continue

    if len(results) < 2:
        print("\nCouldn't load both models for comparison.", flush=True)
        return

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "real_model_comparison.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'real_model_comparison.csv'}", flush=True)

    # Print comparison
    print("\n" + "="*60, flush=True)
    print("RESULTS", flush=True)
    print("="*60, flush=True)

    baguette = results[0]
    gemma = results[1]

    param_ratio = gemma["param_count"] / baguette["param_count"]
    speed_ratio = baguette["total_time_ms"] / gemma["total_time_ms"]

    print(f"\n{baguette['name']}:", flush=True)
    print(f"  Params: {baguette['param_count']/1e9:.2f}B", flush=True)
    print(f"  Total time: {baguette['total_time_ms']/1000:.2f}s", flush=True)

    print(f"\n{gemma['name']}:", flush=True)
    print(f"  Params: {gemma['param_count']/1e9:.2f}B", flush=True)
    print(f"  Total time: {gemma['total_time_ms']/1000:.2f}s", flush=True)

    print(f"\n{gemma['name']} has {param_ratio:.0f}x more params", flush=True)
    if speed_ratio > 1:
        print(f"{gemma['name']} is {speed_ratio:.1f}x FASTER", flush=True)
    else:
        print(f"{baguette['name']} is {1/speed_ratio:.1f}x faster", flush=True)


if __name__ == "__main__":
    main()
