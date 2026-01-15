#!/usr/bin/env python3
"""
Inference Pareto: Depth vs Width Latency Trade-off Study

This tool benchmarks transformer models with varying depth/width ratios
to demonstrate that for on-device inference, latency matters more than
raw parameter count.
"""

import argparse
import sys


def cmd_run(args):
    """Run benchmark experiments."""
    from experiments import (
        run_all_experiments,
        run_depth_sweep,
        run_max_width_search,
        run_param_matched_sweep,
        run_width_sweep,
    )

    experiment = args.experiment

    if experiment == "all":
        run_all_experiments()
    elif experiment == "param-matched":
        run_param_matched_sweep()
    elif experiment == "depth-sweep":
        run_depth_sweep()
    elif experiment == "width-sweep":
        run_width_sweep()
    elif experiment == "max-width":
        run_max_width_search()
    else:
        print(f"Unknown experiment: {experiment}")
        sys.exit(1)


def cmd_plot(args):
    """Generate visualization plots."""
    from visualize import generate_all_plots
    generate_all_plots()


def cmd_report(args):
    """Print summary report."""
    from visualize import print_summary_report
    print_summary_report()


def cmd_configs(args):
    """List all experiment configurations."""
    from config import (
        get_depth_sweep_configs,
        get_param_matched_configs,
        get_width_sweep_configs,
        print_config_table,
    )

    print("\n=== Experiment 1: Parameter-Matched Sweep ===")
    print_config_table(get_param_matched_configs())

    print("\n=== Experiment 2: Depth Sweep (fixed width=576) ===")
    print_config_table(get_depth_sweep_configs())

    print("\n=== Experiment 3: Width Sweep (fixed depth=40) ===")
    print_config_table(get_width_sweep_configs())


def cmd_test(args):
    """Run a quick test benchmark."""
    from benchmark import benchmark_config
    from config import ModelConfig

    print("Running quick test benchmark...")
    test_config = ModelConfig(
        name="test-tiny",
        num_layers=4,
        hidden_size=256,
        intermediate_size=683,
        num_attention_heads=4,
        num_kv_heads=2,
    )
    result = benchmark_config(test_config, throughput_batch_sizes=[1])
    print(f"\nTest passed! Decode latency: {result.decode_latency_mean:.2f}ms/token")


def main():
    parser = argparse.ArgumentParser(
        description="Inference Pareto: Depth vs Width Latency Trade-off Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python main.py run --experiment all          # Run all experiments
  uv run python main.py run --experiment param-matched # Parameter-matched sweep
  uv run python main.py run --experiment max-width    # Max width at each depth
  uv run python main.py plot                          # Generate plots
  uv run python main.py report                        # Print summary
  uv run python main.py configs                       # List configurations
  uv run python main.py test                          # Quick test
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark experiments")
    run_parser.add_argument(
        "--experiment",
        choices=["all", "param-matched", "depth-sweep", "width-sweep", "max-width"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    run_parser.set_defaults(func=cmd_run)

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate visualization plots")
    plot_parser.set_defaults(func=cmd_plot)

    # Report command
    report_parser = subparsers.add_parser("report", help="Print summary report")
    report_parser.set_defaults(func=cmd_report)

    # Configs command
    configs_parser = subparsers.add_parser("configs", help="List experiment configurations")
    configs_parser.set_defaults(func=cmd_configs)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run quick test benchmark")
    test_parser.set_defaults(func=cmd_test)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
