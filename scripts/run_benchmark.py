#!/usr/bin/env python3
"""
CLI script to run complete benchmark evaluation and generate reports.

Usage:
    python scripts/run_benchmark.py [--output REPORT_PATH] [--plots-dir PLOTS_DIR]

Example:
    python scripts/run_benchmark.py --output reports/benchmark_summary.md --plots-dir reports/figures
"""

import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.benchmark import run_benchmark, generate_summary_table


def create_qc_violin_plots(df: pd.DataFrame, output_path: Path):
    """
    Generate violin plots showing QC metric distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results containing condition and metrics
    output_path : Path
        Where to save the plot
    """
    # We don't have direct access to per-cell QC distributions from the benchmark df
    # Instead, we'll plot the aggregate metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Auditor Performance Metrics by Condition", fontsize=16, fontweight="bold")

    # Precision
    sns.violinplot(data=df, x="condition", y="precision", hue="auditor", ax=axes[0, 0], split=True)
    axes[0, 0].set_title("Precision")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend(title="Auditor")

    # Recall
    sns.violinplot(data=df, x="condition", y="recall", hue="auditor", ax=axes[0, 1], split=True)
    axes[0, 1].set_title("Recall")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend(title="Auditor")

    # Groundedness
    sns.violinplot(data=df, x="condition", y="groundedness", hue="auditor", ax=axes[1, 0], split=True)
    axes[1, 0].set_title("Groundedness Ratio")
    axes[1, 0].set_xlabel("")
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend(title="Auditor")

    # Speculation Score
    sns.violinplot(data=df, x="condition", y="speculation_score", hue="auditor", ax=axes[1, 1], split=True)
    axes[1, 1].set_title("Speculation Score (1=Pass, 0=Fail)")
    axes[1, 1].set_xlabel("")
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title="Auditor")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved QC metrics plot: {output_path}")


def create_benchmark_comparison_plot(df: pd.DataFrame, output_path: Path):
    """
    Generate bar chart comparing baseline vs Claude performance.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results
    output_path : Path
        Where to save the plot
    """
    # Calculate aggregate metrics by auditor
    summary = df.groupby("auditor").agg({
        "precision": "mean",
        "recall": "mean",
        "f1": "mean",
        "groundedness": "mean",
        "speculation_score": "mean",
    }).reset_index()

    # Reshape for plotting
    metrics = ["precision", "recall", "f1", "groundedness", "speculation_score"]
    metric_labels = ["Precision", "Recall", "F1 Score", "Groundedness", "Speculation\nPass Rate"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(metrics))
    width = 0.35

    # Plot bars for each auditor
    auditors = summary["auditor"].unique()
    colors = ["#3498db", "#e74c3c"]  # Blue for baseline, red for claude

    for i, auditor in enumerate(auditors):
        auditor_data = summary[summary["auditor"] == auditor]
        values = [auditor_data[metric].values[0] for metric in metrics]
        offset = width * (i - 0.5)
        ax.bar([xi + offset for xi in x], values, width, label=auditor.capitalize(), color=colors[i % len(colors)])

    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Baseline vs Claude Auditor Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved comparison plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run sc-audit benchmark evaluation and generate reports"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/benchmark_summary.md",
        help="Output path for benchmark summary (markdown)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="reports/figures",
        help="Directory for saving diagnostic plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )

    args = parser.parse_args()

    # Create output directories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(args.plots_dir)
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("sc-audit Benchmark Evaluation")
    print("=" * 80 + "\n")

    # Run benchmark
    try:
        df = run_benchmark()
    except Exception as e:
        print(f"\n❌ ERROR: Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate plots
    if not args.no_plots:
        print(f"\nGenerating diagnostic plots in {plots_dir}/...")

        try:
            create_qc_violin_plots(df, plots_dir / "qc_metrics.png")
            create_benchmark_comparison_plot(df, plots_dir / "benchmark_comparison.png")
        except Exception as e:
            print(f"\n⚠️  WARNING: Plot generation failed: {e}")
            import traceback
            traceback.print_exc()

    # Generate markdown report
    print(f"\nGenerating benchmark summary...")
    try:
        summary_md = generate_summary_table(df)

        # Add plot references if plots were generated
        if not args.no_plots:
            summary_md += "\n## Diagnostic Plots\n\n"
            summary_md += f"![QC Metrics by Condition]({plots_dir / 'qc_metrics.png'})\n\n"
            summary_md += f"![Baseline vs Claude Comparison]({plots_dir / 'benchmark_comparison.png'})\n\n"

        # Save to file
        with open(output_path, "w") as f:
            f.write(summary_md)

        print(f"  ✓ Saved benchmark summary: {output_path}")

        # Display summary to console
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(summary_md)

    except Exception as e:
        print(f"\n❌ ERROR: Failed to generate summary: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ Benchmark complete!")
    print(f"   Report: {output_path}")
    if not args.no_plots:
        print(f"   Plots:  {plots_dir}/")
    print()


if __name__ == "__main__":
    main()
