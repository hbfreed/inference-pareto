"""Visualization utilities for depth vs width latency study."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/raw")
PLOTS_DIR = Path("results/plots")


def ensure_plots_dir():
    """Create plots directory if it doesn't exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_results(filename: str) -> Optional[pd.DataFrame]:
    """Load results CSV file."""
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        return pd.read_csv(filepath)
    print(f"Warning: {filepath} not found")
    return None


def plot_pareto_depth_vs_latency():
    """
    Plot depth vs decode latency, with points colored by parameter count.
    Uses param-matched sweep data.
    """
    df = load_results("param_matched_sweep.csv")
    if df is None:
        return

    ensure_plots_dir()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by parameter count
    scatter = ax.scatter(
        df["num_layers"],
        df["decode_latency_mean"],
        c=df["param_count"] / 1e9,
        cmap="viridis",
        s=100,
        alpha=0.8,
    )

    # Add labels for each point
    for _, row in df.iterrows():
        ax.annotate(
            f'{row["hidden_size"]}h',
            (row["num_layers"], row["decode_latency_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Number of Layers", fontsize=12)
    ax.set_ylabel("Decode Latency (ms/token)", fontsize=12)
    ax.set_title("Depth vs Decode Latency (Parameter-Matched ~0.3B)", fontsize=14)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Parameters (B)", fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pareto_depth_latency.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'pareto_depth_latency.png'}")


def plot_pareto_width_vs_latency():
    """
    Plot width vs decode latency using width sweep data.
    """
    df = load_results("width_sweep.csv")
    if df is None:
        return

    ensure_plots_dir()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by parameter count
    scatter = ax.scatter(
        df["hidden_size"],
        df["decode_latency_mean"],
        c=df["param_count"] / 1e9,
        cmap="plasma",
        s=100,
        alpha=0.8,
    )

    ax.set_xlabel("Hidden Size", fontsize=12)
    ax.set_ylabel("Decode Latency (ms/token)", fontsize=12)
    ax.set_title("Width vs Decode Latency (Fixed Depth = 40 Layers)", fontsize=14)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Parameters (B)", fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pareto_width_latency.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'pareto_width_latency.png'}")


def plot_depth_scaling():
    """
    Plot how latency scales with depth at fixed width.
    """
    df = load_results("depth_sweep.csv")
    if df is None:
        return

    ensure_plots_dir()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Latency vs depth
    ax1.plot(df["num_layers"], df["decode_latency_mean"], 'o-', linewidth=2, markersize=8)
    ax1.fill_between(
        df["num_layers"],
        df["decode_latency_mean"] - df["decode_latency_std"],
        df["decode_latency_mean"] + df["decode_latency_std"],
        alpha=0.3,
    )
    ax1.set_xlabel("Number of Layers", fontsize=12)
    ax1.set_ylabel("Decode Latency (ms/token)", fontsize=12)
    ax1.set_title("Latency Scaling with Depth (Hidden=576)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Latency per layer
    latency_per_layer = df["decode_latency_mean"] / df["num_layers"]
    ax2.bar(df["num_layers"].astype(str), latency_per_layer, color='steelblue')
    ax2.set_xlabel("Number of Layers", fontsize=12)
    ax2.set_ylabel("Latency per Layer (ms)", fontsize=12)
    ax2.set_title("Per-Layer Overhead", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "depth_scaling.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'depth_scaling.png'}")


def plot_params_vs_latency():
    """
    Plot parameter count vs latency across all param-matched configs.
    This shows that same params can have very different latencies.
    """
    df = load_results("param_matched_sweep.csv")
    if df is None:
        return

    ensure_plots_dir()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by depth
    scatter = ax.scatter(
        df["param_count"] / 1e9,
        df["decode_latency_mean"],
        c=df["num_layers"],
        cmap="coolwarm",
        s=100,
        alpha=0.8,
    )

    # Add labels
    for _, row in df.iterrows():
        ax.annotate(
            f'{row["num_layers"]}L',
            (row["param_count"] / 1e9, row["decode_latency_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Parameters (Billions)", fontsize=12)
    ax.set_ylabel("Decode Latency (ms/token)", fontsize=12)
    ax.set_title("Parameters vs Latency: Same Params, Different Latency!", fontsize=14)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Number of Layers", fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "params_vs_latency.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'params_vs_latency.png'}")


def plot_max_width():
    """
    The key visualization: shallower models can be BOTH faster AND have more parameters.
    Shows total response time (prefill + 100 tokens) vs parameters.
    """
    df = load_results("max_width_summary.csv")
    if df is None:
        return

    ensure_plots_dir()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axes

    # Sort by depth for consistent ordering (deep to shallow)
    df = df.sort_values("num_layers", ascending=False)

    x = range(len(df))
    labels = [f"{int(row['num_layers'])}L\n{int(row['hidden_size'])}h" for _, row in df.iterrows()]

    # Color gradient from red (deep/slow) to green (shallow/fast)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df)))

    # Plot 1: Parameters (more = better)
    bars1 = ax1.bar(x, df["param_count"] / 1e9, color=colors)
    for bar, (_, row) in zip(bars1, df.iterrows()):
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}B', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Parameters (Billions)", fontsize=12)
    ax1.set_title("Model Capacity", fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Total response time (less = better)
    bars2 = ax2.bar(x, df["total_time_100tok"] / 1000, color=colors)  # Convert to seconds
    for bar, (_, row) in zip(bars2, df.iterrows()):
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Total Time (seconds)", fontsize=12)
    ax2.set_title("Response Time (512 prompt + 100 tokens)", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Speedup vs baseline
    if "speedup_vs_baseline" in df.columns:
        bars3 = ax3.bar(x, df["speedup_vs_baseline"], color=colors)
        for bar, (_, row) in zip(bars3, df.iterrows()):
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}x', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.set_ylabel("Speedup vs 80L Baseline", fontsize=12)
        ax3.set_title("Speed Improvement", fontsize=14)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax3.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Shallow-Wide Wins: Faster AND More Parameters", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "max_width.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'max_width.png'}")


def plot_batch_scaling():
    """
    Plot throughput scaling with batch size across different architectures.
    """
    df = load_results("param_matched_sweep.csv")
    if df is None:
        return

    ensure_plots_dir()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Select a subset of configs for clarity
    selected = df[df["num_layers"].isin([8, 20, 40, 80, 100])]

    batch_sizes = [1, 8, 16, 32]
    width = 0.15
    x = np.arange(len(selected))

    for i, bs in enumerate(batch_sizes):
        col = f"throughput_batch{bs}"
        if col in selected.columns:
            bars = ax.bar(x + i * width, selected[col], width, label=f'Batch {bs}')

    ax.set_xlabel("Model Configuration", fontsize=12)
    ax.set_ylabel("Throughput (tokens/sec)", fontsize=12)
    ax.set_title("Throughput Scaling with Batch Size", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{row['num_layers']}L\n{row['hidden_size']}h"
                       for _, row in selected.iterrows()], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "batch_scaling.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'batch_scaling.png'}")


def plot_memory_usage():
    """
    Plot memory usage across different configurations.
    """
    df = load_results("param_matched_sweep.csv")
    if df is None:
        return

    ensure_plots_dir()

    fig, ax = plt.subplots(figsize=(10, 6))

    df_sorted = df.sort_values("num_layers")

    ax.scatter(df_sorted["param_count"] / 1e9, df_sorted["peak_memory_gb"],
               c=df_sorted["num_layers"], cmap="coolwarm", s=100, alpha=0.8)

    ax.set_xlabel("Parameters (Billions)", fontsize=12)
    ax.set_ylabel("Peak Memory (GB)", fontsize=12)
    ax.set_title("Memory Usage vs Model Size", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "memory_usage.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'memory_usage.png'}")


def generate_all_plots():
    """Generate all visualization plots."""
    print("\nGenerating plots...")
    ensure_plots_dir()

    plot_pareto_depth_vs_latency()
    plot_pareto_width_vs_latency()
    plot_depth_scaling()
    plot_params_vs_latency()
    plot_max_width()
    plot_batch_scaling()
    plot_memory_usage()

    print(f"\nAll plots saved to {PLOTS_DIR}")


def print_summary_report():
    """Print a text summary of key findings."""
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    # Param-matched findings
    df = load_results("param_matched_sweep.csv")
    if df is not None:
        # Use total_time_100tok if available, fall back to decode_latency
        time_col = "total_time_100tok" if "total_time_100tok" in df.columns else "decode_latency_mean"
        fastest = df.loc[df[time_col].idxmin()]
        slowest = df.loc[df[time_col].idxmax()]
        print("\n[Parameter-Matched Sweep (~0.3-0.5B params)]")
        if time_col == "total_time_100tok":
            print(f"  Fastest: {fastest['config_name']} - {fastest[time_col]/1000:.1f}s total (512 prompt + 100 tokens)")
            print(f"  Slowest: {slowest['config_name']} - {slowest[time_col]/1000:.1f}s total")
        else:
            print(f"  Fastest: {fastest['config_name']} - {fastest[time_col]:.2f}ms/token")
            print(f"  Slowest: {slowest['config_name']} - {slowest[time_col]:.2f}ms/token")
        print(f"  Speed difference: {slowest[time_col] / fastest[time_col]:.1f}x!")

    # Depth sweep findings
    df = load_results("depth_sweep.csv")
    if df is not None:
        df = df.sort_values("num_layers")
        time_col = "total_time_100tok" if "total_time_100tok" in df.columns else "decode_latency_mean"
        print("\n[Depth Sweep (fixed width=576)]")
        print(f"  Scaling: {df[time_col].iloc[-1] / df[time_col].iloc[0]:.1f}x slower "
              f"from {int(df['num_layers'].iloc[0])}L to {int(df['num_layers'].iloc[-1])}L")

    # Max width findings
    df = load_results("max_width_summary.csv")
    if df is not None:
        deep = df[df["num_layers"] == 80].iloc[0]
        shallow = df[df["num_layers"] == 10].iloc[0]
        print("\n[Max Width at Each Depth - At maximum width for each depth:]")
        print(f"  80 layers: {deep['param_count']/1e9:.1f}B params, {deep['total_time_100tok']/1000:.1f}s response")
        print(f"  10 layers: {shallow['param_count']/1e9:.1f}B params, {shallow['total_time_100tok']/1000:.1f}s response")
        print(f"  --> 10L is {deep['total_time_100tok']/shallow['total_time_100tok']:.1f}x FASTER (but {deep['param_count']/shallow['param_count']:.1f}x fewer params)")

    # The key finding
    df_param = load_results("param_matched_sweep.csv")
    if df_param is not None and "total_time_100tok" in df_param.columns:
        wide = df_param[df_param["config_name"] == "wide-8L"].iloc[0]
        deep = df_param[df_param["config_name"] == "deep-100L"].iloc[0]
        print("\n[KEY FINDING - Similar params, vastly different speed:]")
        print(f"  wide-8L:   {wide['param_count']/1e9:.2f}B params, {wide['total_time_100tok']/1000:.1f}s response")
        print(f"  deep-100L: {deep['param_count']/1e9:.2f}B params, {deep['total_time_100tok']/1000:.1f}s response")
        print(f"  --> 8L has {wide['param_count']/deep['param_count']:.1f}x MORE params AND is {deep['total_time_100tok']/wide['total_time_100tok']:.1f}x FASTER!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    generate_all_plots()
    print_summary_report()
