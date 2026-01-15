"""Plot Baguettotron vs Gemma 3 12B comparison."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results/raw")
PLOTS_DIR = Path("results/plots")


def plot_architecture_comparison():
    """
    Create side-by-side comparison of Baguettotron vs Gemma 3 12B.
    """
    df = pd.read_csv(RESULTS_DIR / "architecture_comparison.csv")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = df["name"].tolist()
    x = np.arange(len(models))

    # Colors: Baguettotron (deep-narrow) = red, Gemma (shallow-wide) = green
    colors = ["#e74c3c", "#27ae60"]

    # Plot 1: Parameters
    ax1 = axes[0]
    bars1 = ax1.bar(x, df["param_count"] / 1e9, color=colors, edgecolor="black", linewidth=1)
    for bar, params in zip(bars1, df["param_count"]):
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}B",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.set_ylabel("Parameters (Billions)", fontsize=12)
    ax1.set_title("Model Capacity", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, max(df["param_count"] / 1e9) * 1.25)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Total Response Time
    ax2 = axes[1]
    bars2 = ax2.bar(x, df["total_time_ms"] / 1000, color=colors, edgecolor="black", linewidth=1)
    for bar, time in zip(bars2, df["total_time_ms"]):
        height = bar.get_height()
        ax2.annotate(f"{height:.2f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=12)
    ax2.set_ylabel("Total Response Time (seconds)", fontsize=12)
    ax2.set_title("Response Time\n(512 prompt + 100 tokens)", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, max(df["total_time_ms"] / 1000) * 1.25)
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Decode Latency
    ax3 = axes[2]
    bars3 = ax3.bar(x, df["decode_ms_per_token"], color=colors, edgecolor="black", linewidth=1)
    for bar, latency in zip(bars3, df["decode_ms_per_token"]):
        height = bar.get_height()
        ax3.annotate(f"{height:.1f}ms",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=12)
    ax3.set_ylabel("Decode Latency (ms/token)", fontsize=12)
    ax3.set_title("Per-Token Latency", fontsize=14, fontweight="bold")
    ax3.set_ylim(0, max(df["decode_ms_per_token"]) * 1.25)
    ax3.grid(True, alpha=0.3, axis="y")

    # Calculate ratios for suptitle
    baguette = df[df["name"] == "Baguettotron"].iloc[0]
    gemma = df[df["name"] == "Gemma-3-12B"].iloc[0]
    param_ratio = gemma["param_count"] / baguette["param_count"]
    speed_ratio = baguette["total_time_ms"] / gemma["total_time_ms"]

    plt.suptitle(f"Gemma 3 12B: {param_ratio:.0f}x more params, {speed_ratio:.1f}x faster",
                fontsize=16, fontweight="bold", y=1.02)

    # Add architecture labels
    fig.text(0.5, -0.02,
             f"Baguettotron: 80 layers, 576 hidden  |  Gemma 3 12B: 48 layers, 3840 hidden",
             ha="center", fontsize=11, style="italic")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "architecture_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'architecture_comparison.png'}")


if __name__ == "__main__":
    plot_architecture_comparison()
