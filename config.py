"""Model configuration utilities for depth vs width experiments."""

from dataclasses import dataclass
from typing import List
from transformers import LlamaConfig


@dataclass
class ModelConfig:
    """Configuration for a transformer model."""
    name: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int = 65536
    max_position_embeddings: int = 4096

    def __post_init__(self):
        # Validate head dimensions
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        self.head_dim = self.hidden_size // self.num_attention_heads


def count_params(config: ModelConfig) -> int:
    """
    Calculate total parameter count for a Llama-style model with GQA and tied embeddings.

    Components:
    - Embedding: vocab_size * hidden_size (tied with output, count once)
    - Per layer:
      - Q projection: hidden_size * hidden_size
      - K projection: hidden_size * (kv_heads * head_dim)
      - V projection: hidden_size * (kv_heads * head_dim)
      - O projection: hidden_size * hidden_size
      - gate_proj: hidden_size * intermediate_size
      - up_proj: hidden_size * intermediate_size
      - down_proj: intermediate_size * hidden_size
      - input_layernorm: hidden_size
      - post_attention_layernorm: hidden_size
    - Final norm: hidden_size
    """
    head_dim = config.hidden_size // config.num_attention_heads
    kv_dim = config.num_kv_heads * head_dim

    # Embedding (tied with lm_head)
    embed_params = config.vocab_size * config.hidden_size

    # Per-layer parameters
    # Attention
    q_params = config.hidden_size * config.hidden_size  # Q projection
    k_params = config.hidden_size * kv_dim  # K projection (GQA)
    v_params = config.hidden_size * kv_dim  # V projection (GQA)
    o_params = config.hidden_size * config.hidden_size  # O projection
    attn_params = q_params + k_params + v_params + o_params

    # MLP (SwiGLU style: gate, up, down)
    mlp_params = 3 * config.hidden_size * config.intermediate_size

    # LayerNorms (2 per layer)
    norm_params = 2 * config.hidden_size

    per_layer = attn_params + mlp_params + norm_params

    # Final layer norm
    final_norm = config.hidden_size

    total = embed_params + config.num_layers * per_layer + final_norm
    return total


def make_llama_config(config: ModelConfig) -> LlamaConfig:
    """Convert ModelConfig to HuggingFace LlamaConfig."""
    return LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_kv_heads,
        max_position_embeddings=config.max_position_embeddings,
        # Fixed settings from Baguettotron
        hidden_act="silu",
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        use_cache=True,
    )


# Pre-defined configurations for experiments

def get_param_matched_configs() -> List[ModelConfig]:
    """
    Experiment 1: Parameter-matched sweep (~300M params each).
    Vary depth/width ratio while keeping total params roughly constant.
    """
    return [
        ModelConfig("deep-100L", num_layers=100, hidden_size=512, intermediate_size=1365,
                   num_attention_heads=8, num_kv_heads=4),
        ModelConfig("baguette-80L", num_layers=80, hidden_size=576, intermediate_size=1536,
                   num_attention_heads=9, num_kv_heads=3),
        ModelConfig("deep-60L", num_layers=60, hidden_size=672, intermediate_size=1792,
                   num_attention_heads=12, num_kv_heads=4),
        ModelConfig("medium-48L", num_layers=48, hidden_size=768, intermediate_size=2048,
                   num_attention_heads=12, num_kv_heads=4),
        ModelConfig("medium-40L", num_layers=40, hidden_size=832, intermediate_size=2218,
                   num_attention_heads=16, num_kv_heads=4),
        ModelConfig("medium-32L", num_layers=32, hidden_size=960, intermediate_size=2560,
                   num_attention_heads=16, num_kv_heads=4),
        ModelConfig("shallow-24L", num_layers=24, hidden_size=1088, intermediate_size=2901,
                   num_attention_heads=16, num_kv_heads=4),
        ModelConfig("shallow-20L", num_layers=20, hidden_size=1216, intermediate_size=3242,
                   num_attention_heads=16, num_kv_heads=4),
        ModelConfig("shallow-16L", num_layers=16, hidden_size=1344, intermediate_size=3584,
                   num_attention_heads=16, num_kv_heads=4),
        ModelConfig("wide-12L", num_layers=12, hidden_size=1536, intermediate_size=4096,
                   num_attention_heads=16, num_kv_heads=4),
        ModelConfig("wide-10L", num_layers=10, hidden_size=1792, intermediate_size=4778,
                   num_attention_heads=16, num_kv_heads=4),
        ModelConfig("wide-8L", num_layers=8, hidden_size=2048, intermediate_size=5461,
                   num_attention_heads=16, num_kv_heads=4),
    ]


def get_depth_sweep_configs() -> List[ModelConfig]:
    """
    Experiment 2: Depth sweep at fixed width (576).
    Isolate depth's effect on latency.
    """
    return [
        ModelConfig("depth-20L", num_layers=20, hidden_size=576, intermediate_size=1536,
                   num_attention_heads=9, num_kv_heads=3),
        ModelConfig("depth-40L", num_layers=40, hidden_size=576, intermediate_size=1536,
                   num_attention_heads=9, num_kv_heads=3),
        ModelConfig("depth-60L", num_layers=60, hidden_size=576, intermediate_size=1536,
                   num_attention_heads=9, num_kv_heads=3),
        ModelConfig("depth-80L", num_layers=80, hidden_size=576, intermediate_size=1536,
                   num_attention_heads=9, num_kv_heads=3),
        ModelConfig("depth-100L", num_layers=100, hidden_size=576, intermediate_size=1536,
                   num_attention_heads=9, num_kv_heads=3),
    ]


def get_width_sweep_configs() -> List[ModelConfig]:
    """
    Experiment 3: Width sweep at fixed depth (40 layers).
    Isolate width's effect on latency.
    """
    return [
        ModelConfig("width-512", num_layers=40, hidden_size=512, intermediate_size=1365,
                   num_attention_heads=8, num_kv_heads=4),
        ModelConfig("width-768", num_layers=40, hidden_size=768, intermediate_size=2048,
                   num_attention_heads=12, num_kv_heads=4),
        ModelConfig("width-1024", num_layers=40, hidden_size=1024, intermediate_size=2730,
                   num_attention_heads=16, num_kv_heads=4),
        ModelConfig("width-1280", num_layers=40, hidden_size=1280, intermediate_size=3413,
                   num_attention_heads=16, num_kv_heads=4),
        ModelConfig("width-1536", num_layers=40, hidden_size=1536, intermediate_size=4096,
                   num_attention_heads=16, num_kv_heads=4),
    ]


def create_config_for_search(num_layers: int, hidden_size: int) -> ModelConfig:
    """
    Create a config for latency-matched binary search.
    Automatically determines appropriate head counts.
    """
    # Round hidden_size to nearest 64 for clean head dimensions
    hidden_size = (hidden_size // 64) * 64
    if hidden_size < 256:
        hidden_size = 256

    # Determine number of heads (head_dim = 64)
    num_heads = hidden_size // 64

    # KV heads: find a divisor of num_heads that's roughly 1/3 to 1/4
    # Must satisfy: num_heads % num_kv_heads == 0
    num_kv_heads = 1
    for candidate in [num_heads // 4, num_heads // 3, num_heads // 2, 1]:
        if candidate >= 1 and num_heads % candidate == 0:
            num_kv_heads = candidate
            break

    # FFN size: approximately 2.67x hidden (like Baguettotron)
    intermediate_size = int(hidden_size * 2.67)
    # Round to nearest 64
    intermediate_size = (intermediate_size // 64) * 64

    name = f"search-{num_layers}L-{hidden_size}h"
    return ModelConfig(
        name=name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )


def print_config_table(configs: List[ModelConfig]) -> None:
    """Print a table of configurations with parameter counts."""
    print(f"{'Name':<20} {'Layers':>6} {'Hidden':>6} {'FFN':>6} {'Heads':>6} {'KV':>4} {'Params':>10}")
    print("-" * 70)
    for cfg in configs:
        params = count_params(cfg)
        print(f"{cfg.name:<20} {cfg.num_layers:>6} {cfg.hidden_size:>6} {cfg.intermediate_size:>6} "
              f"{cfg.num_attention_heads:>6} {cfg.num_kv_heads:>4} {params:>10,}")


if __name__ == "__main__":
    # Verify Baguettotron parameter count
    baguette = ModelConfig(
        "baguettotron",
        num_layers=80,
        hidden_size=576,
        intermediate_size=1536,
        num_attention_heads=9,
        num_kv_heads=3,
    )
    params = count_params(baguette)
    print(f"Baguettotron parameters: {params:,} (expected ~321M)")
    print()

    print("=== Experiment 1: Parameter-Matched Sweep ===")
    print_config_table(get_param_matched_configs())
    print()

    print("=== Experiment 2: Depth Sweep (fixed width=576) ===")
    print_config_table(get_depth_sweep_configs())
    print()

    print("=== Experiment 3: Width Sweep (fixed depth=40) ===")
    print_config_table(get_width_sweep_configs())
