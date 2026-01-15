"""Plot the time-matched experiment results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results/raw")
PLOTS_DIR = Path("results/plots")


def plot_time_matched():
    """
    Create visualization showing: shallow-wide = MORE params AND FASTER.
    """
    df = pd.read_csv(RESULTS_DIR / "time_matched.csv")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by depth (deep to shallow)
    df = df.sort_values("num_layers", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(df))
    labels = [f"{int(row['num_layers'])}L\n{int(row['hidden_size'])}h" for _, row in df.iterrows()]

    # Colors: red (deep/slow) to green (shallow/fast)
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(df)))

    # Left plot: Parameters (Billions)
    ax1 = axes[0]
    bars1 = ax1.bar(x, df["param_count"] / 1e9, color=colors, edgecolor='black', linewidth=0.5)
    for bar, (_, row) in zip(bars1, df.iterrows()):
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}B',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Parameters (Billions)", fontsize=12)
    ax1.set_title("Model Capacity", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(df["param_count"] / 1e9) * 1.15)

    # Right plot: Total Response Time
    ax2 = axes[1]
    bars2 = ax2.bar(x, df["total_time_ms"] / 1000, color=colors, edgecolor='black', linewidth=0.5)
    for bar, (_, row) in zip(bars2, df.iterrows()):
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Total Response Time (seconds)", fontsize=12)
    ax2.set_title("Response Time (512 prompt + 100 tokens)", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(df["total_time_ms"] / 1000) * 1.15)

    # Add baseline reference line
    baseline_time = df[df["num_layers"] == 80]["total_time_ms"].values[0] / 1000
    ax2.axhline(y=baseline_time, color='red', linestyle='--', alpha=0.7, linewidth=2)

    plt.suptitle("Shallow-Wide Wins: More Capacity AND Faster Response",
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "time_matched.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'time_matched.png'}")


if __name__ == "__main__":
    plot_time_matched()
