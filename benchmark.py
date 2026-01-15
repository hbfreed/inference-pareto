"""Benchmarking utilities for measuring inference latency and throughput."""

import gc
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from transformers import LlamaForCausalLM

from config import ModelConfig, make_llama_config


# ============================================================================
# Performance optimizations for torch 2.9+
# ============================================================================

def setup_torch_optimizations():
    """Configure torch for maximum inference performance."""
    if torch.cuda.is_available():
        # Enable TF32 for faster matmuls on Ampere+ GPUs (RTX 30xx, 40xx, A100, etc.)
        # TF32 uses 19 bits instead of 32, ~2x faster with minimal precision loss
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Flash attention is enabled by default in transformers for supported models

        print("Torch optimizations enabled: TF32, cuDNN benchmark")


# Run optimizations at import time
setup_torch_optimizations()


def get_device() -> torch.device:
    """Get the device to use (cuda:0 if available, else cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        print("WARNING: CUDA not available, using CPU (benchmarks will be slow)")
        return torch.device("cpu")


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config_name: str
    num_layers: int
    hidden_size: int
    param_count: int

    # Prefill metrics (ms)
    prefill_latency_mean: float = 0.0
    prefill_latency_std: float = 0.0
    prefill_latency_p50: float = 0.0
    prefill_latency_p95: float = 0.0
    prefill_latency_p99: float = 0.0

    # Decode metrics (ms per token)
    decode_latency_mean: float = 0.0
    decode_latency_std: float = 0.0
    decode_latency_p50: float = 0.0
    decode_latency_p95: float = 0.0
    decode_latency_p99: float = 0.0

    # Total response time (prefill + N tokens generation) in ms
    total_time_50tok: float = 0.0   # Time for prefill + 50 tokens
    total_time_100tok: float = 0.0  # Time for prefill + 100 tokens
    total_time_200tok: float = 0.0  # Time for prefill + 200 tokens

    # Throughput (tokens/sec) at different batch sizes
    throughput_batch1: float = 0.0
    throughput_batch8: float = 0.0
    throughput_batch16: float = 0.0
    throughput_batch32: float = 0.0

    # Memory (GB)
    peak_memory_gb: float = 0.0

    # Raw latencies for analysis
    prefill_latencies: List[float] = field(default_factory=list)
    decode_latencies: List[float] = field(default_factory=list)


def get_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(get_device()) / (1024 ** 3)
    return 0.0


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        device = get_device()
        # Initialize CUDA context if needed (required before calling memory stats functions)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        # Only reset if CUDA context is initialized
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats(device)
    gc.collect()


def create_model(config: ModelConfig) -> LlamaForCausalLM:
    """Create a model from config on the available device."""
    device = get_device()
    llama_config = make_llama_config(config)
    model = LlamaForCausalLM(llama_config)
    # Use bfloat16 on CUDA, float32 on CPU
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(dtype=dtype, device=device)
    model.eval()
    # Note: torch.compile disabled due to recompilation overhead with dynamic shapes
    # (prefill vs decode have different sequence lengths). TF32 still enabled.
    return model


@torch.inference_mode()
def measure_prefill(
    model: LlamaForCausalLM,
    seq_len: int = 512,
    warmup_runs: int = 3,
    num_runs: int = 10,
) -> List[float]:
    """
    Measure prefill latency (time to process input prompt).

    Returns list of latencies in milliseconds.
    """
    # Create input
    input_ids = torch.randint(1, 1000, (1, seq_len), device=get_device())

    # Warmup
    for _ in range(warmup_runs):
        _ = model(input_ids, use_cache=True)
        if torch.cuda.is_available(): torch.cuda.synchronize(get_device())

    # Measure
    latencies = []
    for _ in range(num_runs):
        if torch.cuda.is_available(): torch.cuda.synchronize(get_device())
        start = time.perf_counter()
        _ = model(input_ids, use_cache=True)
        if torch.cuda.is_available(): torch.cuda.synchronize(get_device())
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    return latencies


@torch.inference_mode()
def measure_decode(
    model: LlamaForCausalLM,
    prompt_len: int = 128,
    gen_len: int = 50,
    warmup_runs: int = 2,
    num_runs: int = 5,
) -> List[float]:
    """
    Measure per-token decode latency (autoregressive generation).

    Returns list of per-token latencies in milliseconds.
    """
    input_ids = torch.randint(1, 1000, (1, prompt_len), device=get_device())

    all_latencies = []

    for run_idx in range(warmup_runs + num_runs):
        # Initial forward pass to get KV cache
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        run_latencies = []
        for _ in range(gen_len):
            if torch.cuda.is_available(): torch.cuda.synchronize(get_device())
            start = time.perf_counter()
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            if torch.cuda.is_available(): torch.cuda.synchronize(get_device())
            end = time.perf_counter()

            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

            if run_idx >= warmup_runs:
                run_latencies.append((end - start) * 1000)  # ms

        if run_idx >= warmup_runs:
            all_latencies.extend(run_latencies)

    return all_latencies


@torch.inference_mode()
def measure_total_response_time(
    model: LlamaForCausalLM,
    prompt_len: int = 512,
    gen_tokens: int = 100,
    warmup_runs: int = 2,
    num_runs: int = 5,
) -> List[float]:
    """
    Measure total wall-clock time for a complete response (prefill + all tokens).

    This is what users actually experience - the time from sending a prompt
    to receiving the full response.

    Returns list of total times in milliseconds.
    """
    input_ids = torch.randint(1, 1000, (1, prompt_len), device=get_device())

    all_times = []

    for run_idx in range(warmup_runs + num_runs):
        if torch.cuda.is_available(): torch.cuda.synchronize(get_device())
        start = time.perf_counter()

        # Prefill
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        # Generate all tokens
        for _ in range(gen_tokens):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        if torch.cuda.is_available(): torch.cuda.synchronize(get_device())
        end = time.perf_counter()

        if run_idx >= warmup_runs:
            all_times.append((end - start) * 1000)  # ms

    return all_times


@torch.inference_mode()
def measure_throughput(
    model: LlamaForCausalLM,
    batch_size: int,
    seq_len: int = 128,
    gen_tokens: int = 32,
    warmup_runs: int = 2,
    num_runs: int = 3,
) -> float:
    """
    Measure throughput in tokens/second at a given batch size.

    Returns tokens per second.
    """
    input_ids = torch.randint(1, 1000, (batch_size, seq_len), device=get_device())

    total_tokens = 0
    total_time = 0.0

    for run_idx in range(warmup_runs + num_runs):
        # Initial forward pass
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_tokens = outputs.logits[:, -1:, :].argmax(dim=-1)

        if torch.cuda.is_available(): torch.cuda.synchronize(get_device())
        start = time.perf_counter()

        for _ in range(gen_tokens):
            outputs = model(next_tokens, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_tokens = outputs.logits[:, -1:, :].argmax(dim=-1)

        if torch.cuda.is_available(): torch.cuda.synchronize(get_device())
        end = time.perf_counter()

        if run_idx >= warmup_runs:
            total_tokens += batch_size * gen_tokens
            total_time += (end - start)

    return total_tokens / total_time if total_time > 0 else 0.0


def compute_stats(latencies: List[float]) -> dict:
    """Compute statistics from a list of latencies."""
    if not latencies:
        return {"mean": 0, "std": 0, "p50": 0, "p95": 0, "p99": 0}
    arr = np.array(latencies)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def benchmark_config(
    config: ModelConfig,
    prefill_seq_lens: List[int] = [512],
    decode_prompt_len: int = 128,
    decode_gen_len: int = 50,
    throughput_batch_sizes: List[int] = [1, 8, 16, 32],
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run full benchmark suite for a single configuration.
    """
    from config import count_params

    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {config.name}")
        print(f"  Layers: {config.num_layers}, Hidden: {config.hidden_size}")
        print(f"  Parameters: {count_params(config):,}")
        print(f"{'='*60}")

    reset_memory_stats()

    # Create model
    if verbose:
        print("Creating model...")
    model = create_model(config)

    param_count = count_params(config)
    result = BenchmarkResult(
        config_name=config.name,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        param_count=param_count,
    )

    # Measure prefill (using first seq_len in list)
    if verbose:
        print(f"Measuring prefill latency (seq_len={prefill_seq_lens[0]})...")
    prefill_latencies = measure_prefill(model, seq_len=prefill_seq_lens[0])
    result.prefill_latencies = prefill_latencies
    stats = compute_stats(prefill_latencies)
    result.prefill_latency_mean = stats["mean"]
    result.prefill_latency_std = stats["std"]
    result.prefill_latency_p50 = stats["p50"]
    result.prefill_latency_p95 = stats["p95"]
    result.prefill_latency_p99 = stats["p99"]
    if verbose:
        print(f"  Prefill: {stats['mean']:.2f}ms (±{stats['std']:.2f}ms)")

    # Measure decode
    if verbose:
        print(f"Measuring decode latency (prompt={decode_prompt_len}, gen={decode_gen_len})...")
    decode_latencies = measure_decode(model, prompt_len=decode_prompt_len, gen_len=decode_gen_len)
    result.decode_latencies = decode_latencies
    stats = compute_stats(decode_latencies)
    result.decode_latency_mean = stats["mean"]
    result.decode_latency_std = stats["std"]
    result.decode_latency_p50 = stats["p50"]
    result.decode_latency_p95 = stats["p95"]
    result.decode_latency_p99 = stats["p99"]
    if verbose:
        print(f"  Decode: {stats['mean']:.2f}ms/token (±{stats['std']:.2f}ms)")

    # Measure total response time (prefill + generation)
    if verbose:
        print("Measuring total response time (prefill + generation)...")
    for gen_tokens in [50, 100, 200]:
        times = measure_total_response_time(model, prompt_len=512, gen_tokens=gen_tokens)
        mean_time = float(np.mean(times)) if times else 0.0
        if gen_tokens == 50:
            result.total_time_50tok = mean_time
        elif gen_tokens == 100:
            result.total_time_100tok = mean_time
        elif gen_tokens == 200:
            result.total_time_200tok = mean_time
        if verbose:
            print(f"  Total time ({gen_tokens} tokens): {mean_time:.0f}ms ({mean_time/1000:.2f}s)")

    # Measure throughput at different batch sizes
    for batch_size in throughput_batch_sizes:
        if verbose:
            print(f"Measuring throughput (batch_size={batch_size})...")
        try:
            throughput = measure_throughput(model, batch_size=batch_size)
            if batch_size == 1:
                result.throughput_batch1 = throughput
            elif batch_size == 8:
                result.throughput_batch8 = throughput
            elif batch_size == 16:
                result.throughput_batch16 = throughput
            elif batch_size == 32:
                result.throughput_batch32 = throughput
            if verbose:
                print(f"  Throughput (batch={batch_size}): {throughput:.1f} tokens/sec")
        except torch.cuda.OutOfMemoryError:
            if verbose:
                print(f"  Throughput (batch={batch_size}): OOM")

    # Record peak memory
    result.peak_memory_gb = get_memory_gb()
    if verbose:
        print(f"  Peak memory: {result.peak_memory_gb:.2f} GB")

    # Cleanup
    del model
    reset_memory_stats()

    return result


def quick_decode_latency(config: ModelConfig, num_tokens: int = 20) -> float:
    """
    Quick decode latency measurement for binary search.
    Returns mean decode latency in ms.
    """
    reset_memory_stats()
    model = create_model(config)

    latencies = measure_decode(model, prompt_len=64, gen_len=num_tokens, warmup_runs=1, num_runs=2)

    del model
    reset_memory_stats()

    return float(np.mean(latencies)) if latencies else float('inf')


if __name__ == "__main__":
    # Quick test
    from config import ModelConfig, count_params

    test_config = ModelConfig(
        name="test-small",
        num_layers=4,
        hidden_size=256,
        intermediate_size=683,
        num_attention_heads=4,
        num_kv_heads=2,
    )
    print(f"Test config params: {count_params(test_config):,}")

    result = benchmark_config(test_config, throughput_batch_sizes=[1])
    print(f"\nResult: decode_latency={result.decode_latency_mean:.2f}ms")
