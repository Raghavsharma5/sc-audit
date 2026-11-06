#!/usr/bin/env python3
"""
Generate comprehensive comparative diagnostics report.

Runs sensitivity analysis, generates publication-grade figures, and updates
the benchmark summary with empirical boundary analysis.

Usage:
    python scripts/generate_limitations_report.py [--output-dir reports]
"""

import argparse
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.ingest import load_pbmc3k
from src.pipeline.qc_engine import run_qc_pipeline
from src.pipeline.extractor import extract_qc_summary
from src.perturbation.inject import (
    generate_perturbed_dataset,
    inject_lysis,
    inject_doublets,
)
from src.auditor.baseline import (
    baseline_audit,
    MT_PERCENT_ABSOLUTE_THRESHOLD,
    DOUBLET_COUNT_MULTIPLIER,
)
from src.auditor.claude_client import claude_audit
from eval.sensitivity import run_full_sensitivity_analysis
from eval.comparative_plots import (
    plot_sensitivity_curves,
    plot_distribution_divergence,
    plot_biological_hallucination_heatmap,
)


def generate_limitations_analysis_markdown(
    lysis_df, doublets_df, has_claude_results: bool
) -> str:
    """
    Generate markdown section for limitations analysis.

    Parameters
    ----------
    lysis_df : pd.DataFrame
        Lysis sensitivity results
    doublets_df : pd.DataFrame
        Doublet sensitivity results
    has_claude_results : bool
        Whether Claude results are available

    Returns
    -------
    str
        Markdown formatted analysis section
    """
    md = "\n\n---\n\n"
    md += "# Empirical Boundary & AI Limitations Analysis\n\n"

    md += "## Sensitivity Thresholds\n\n"

    # Lysis analysis
    md += "### Lysis Detection (Mitochondrial Read %)\n\n"

    baseline_lysis_threshold = None
    for _, row in lysis_df.iterrows():
        if row["baseline_detected"] and baseline_lysis_threshold is None:
            baseline_lysis_threshold = row["median_pct_mt_target"]
            md += f"- **Baseline Detection Threshold**: First detected at {row['median_pct_mt_target']:.2f}% "
            md += f"(MT scale {row['mt_scale']:.2f}×)\n"
            break

    if has_claude_results and "claude_detected" in lysis_df.columns:
        claude_lysis_threshold = None
        for _, row in lysis_df.iterrows():
            if row["claude_detected"] and claude_lysis_threshold is None:
                claude_lysis_threshold = row["median_pct_mt_target"]
                md += f"- **Claude Detection Threshold**: First detected at {row['median_pct_mt_target']:.2f}% "
                md += f"(MT scale {row['mt_scale']:.2f}×)\n"
                break

    # Doublet analysis
    md += "\n### Doublet Detection (Elevated UMI Counts)\n\n"

    baseline_doublet_threshold = None
    for _, row in doublets_df.iterrows():
        if row["baseline_detected"] and baseline_doublet_threshold is None:
            baseline_doublet_threshold = row["doublet_rate"]
            md += f"- **Baseline Detection Threshold**: First detected at {row['doublet_rate']:.1%} doublet rate "
            md += f"({row['counts_ratio']:.2f}× counts ratio)\n"
            break

    if has_claude_results and "claude_detected" in doublets_df.columns:
        claude_doublet_threshold = None
        for _, row in doublets_df.iterrows():
            if row["claude_detected"] and claude_doublet_threshold is None:
                claude_doublet_threshold = row["doublet_rate"]
                md += f"- **Claude Detection Threshold**: First detected at {row['doublet_rate']:.1%} doublet rate "
                md += f"({row['counts_ratio']:.2f}× counts ratio)\n"
                break

    # Comparative analysis
    md += "\n## Baseline vs Claude: Strengths & Limitations\n\n"

    md += "| Aspect | Baseline (Rule-based) | Claude (LLM) |\n"
    md += "|--------|----------------------|-------------|\n"

    md += "| **Consistency** | ✓ Deterministic thresholds | ⚠️ May vary across runs |\n"
    md += "| **Calibration** | ✓ Pre-calibrated to dataset | ⚠️ Requires prompt engineering |\n"
    md += "| **Edge Cases** | ✗ Binary cutoffs miss borderline | ✓ Can reason about context |\n"
    md += "| **Groundedness** | ✓ Direct numeric comparison | ⚠️ May hallucinate values |\n"
    md += "| **Biological Speculation** | ✓ No interpretation risk | ⚠️ Requires explicit warnings |\n"
    md += "| **Speed** | ✓ Milliseconds | ✗ Seconds (API latency) |\n"
    md += "| **Cost** | ✓ Free | ✗ API token costs |\n"

    if not has_claude_results:
        md += "\n\n*Note: Claude results not available (no API key provided). "
        md += "Only baseline sensitivity analysis is shown.*\n"

    md += "\n## Key Findings\n\n"

    md += "1. **Lysis Detection**: "
    if baseline_lysis_threshold:
        md += f"Baseline reliably detects elevated MT% above {baseline_lysis_threshold:.2f}%. "
    md += "The deterministic threshold works well for clear pathological cases.\n\n"

    md += "2. **Doublet Detection**: "
    md += "Post-clustering dispersion causes both auditors to miss subtle doublet signatures. "
    md += "Pre-clustering doublet-specific tools (Scrublet, DoubletFinder) are recommended.\n\n"

    md += "3. **Threshold Calibration**: "
    md += "Baseline thresholds are dataset-specific and may need adjustment for different "
    md += "tissue types, protocols, or species.\n\n"

    if has_claude_results:
        md += "4. **Claude Adaptive Reasoning**: "
        md += "Claude can incorporate relative context (e.g., \"2.5× higher than dataset median\"), "
        md += "but requires strict schema enforcement to prevent biological speculation.\n\n"

    return md


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive comparative diagnostics report"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports and figures",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="Skip sensitivity analysis (use cached results if available)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("COMPARATIVE DIAGNOSTICS & LIMITATIONS ANALYSIS")
    print("=" * 80 + "\n")

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    has_claude = api_key is not None

    if not has_claude:
        print("⚠️  WARNING: ANTHROPIC_API_KEY not found.")
        print("    Sensitivity analysis will only test baseline auditor.\n")

    # Step 1: Run sensitivity analysis
    print("[1/5] Running sensitivity analysis...")
    if not args.no_sensitivity:
        sensitivity_results = run_full_sensitivity_analysis(api_key=api_key)
        lysis_df = sensitivity_results["lysis"]
        doublets_df = sensitivity_results["doublets"]

        # Save results
        lysis_df.to_csv(output_dir / "lysis_sensitivity.csv", index=False)
        doublets_df.to_csv(output_dir / "doublet_sensitivity.csv", index=False)
        print(f"  ✓ Saved sensitivity results to {output_dir}/")
    else:
        print("  ⚠️  Skipping sensitivity analysis (--no-sensitivity)")
        import pandas as pd

        lysis_df = pd.read_csv(output_dir / "lysis_sensitivity.csv")
        doublets_df = pd.read_csv(output_dir / "doublet_sensitivity.csv")

    # Step 2: Generate Figure 1 (sensitivity curves)
    print("\n[2/5] Generating Figure 1: Sensitivity curves...")
    plot_sensitivity_curves(
        lysis_df, doublets_df, figures_dir / "fig1_sensitivity_curves.png"
    )

    # Step 3: Generate Figure 2 (distribution divergence)
    print("\n[3/5] Generating Figure 2: Distribution divergence...")

    # Load datasets for visualization
    print("  Loading datasets...")
    adata = load_pbmc3k(cache_dir=project_root / "data")
    adata_processed = run_qc_pipeline(adata, species="human")

    # Generate clean control
    adata_clean, _ = generate_perturbed_dataset(
        adata_processed.copy(), "clean_control", species="human"
    )

    # Generate lysis condition
    adata_lysis, _ = inject_lysis(
        adata_processed.copy(), target_cluster="1", mt_scale_factor=2.0, species="human"
    )
    adata_lysis = run_qc_pipeline(adata_lysis, species="human")

    # Generate doublets
    adata_doublets, _ = inject_doublets(
        adata_processed.copy(), doublet_rate=0.08, random_seed=42
    )
    adata_doublets = run_qc_pipeline(adata_doublets, species="human")

    # Get thresholds from baseline
    summary_clean = extract_qc_summary(
        adata_clean, dataset_name="clean", species="human"
    )
    baseline_mt_threshold = MT_PERCENT_ABSOLUTE_THRESHOLD
    baseline_doublet_threshold = (
        summary_clean.total_counts_distribution.p50 * DOUBLET_COUNT_MULTIPLIER
    )

    plot_distribution_divergence(
        adata_clean,
        adata_lysis,
        adata_doublets,
        baseline_mt_threshold,
        baseline_doublet_threshold,
        figures_dir / "fig2_distribution_divergence.png",
    )

    # Step 4: Generate Figure 3 (biological hallucination heatmap)
    print("\n[4/5] Generating Figure 3: Biological hallucination heatmap...")

    # Extract summaries and run audits
    summary_doublets = extract_qc_summary(
        adata_doublets, dataset_name="doublets", species="human"
    )
    baseline_report_doublets = baseline_audit(summary_doublets)

    claude_report_doublets = None
    if has_claude:
        try:
            claude_report_doublets = claude_audit(summary_doublets, api_key=api_key)
        except Exception as e:
            print(f"  ⚠️  Claude audit failed: {e}")

    plot_biological_hallucination_heatmap(
        adata_clean,
        adata_doublets,
        baseline_report_doublets,
        claude_report_doublets,
        figures_dir / "fig3_biological_hallucination_heatmap.png",
    )

    # Step 5: Update benchmark summary
    print("\n[5/5] Updating benchmark summary with limitations analysis...")

    # Generate limitations section
    limitations_md = generate_limitations_analysis_markdown(
        lysis_df, doublets_df, has_claude
    )

    # Read existing summary
    summary_file = output_dir / "benchmark_summary.md"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            existing_content = f.read()

        # Append limitations analysis
        updated_content = existing_content + limitations_md

        # Add figure references if not already present
        if "fig1_sensitivity_curves.png" not in updated_content:
            updated_content += "\n## Comparative Diagnostic Figures\n\n"
            updated_content += "### Figure 1: Detection Sensitivity Curves\n\n"
            updated_content += (
                f"![Sensitivity Curves]({figures_dir / 'fig1_sensitivity_curves.png'})\n\n"
            )
            updated_content += "### Figure 2: Distribution Divergence Analysis\n\n"
            updated_content += f"![Distribution Divergence]({figures_dir / 'fig2_distribution_divergence.png'})\n\n"
            updated_content += "### Figure 3: Biological Hallucination Risk\n\n"
            updated_content += f"![Biological Hallucination]({figures_dir / 'fig3_biological_hallucination_heatmap.png'})\n\n"

        # Save updated summary
        with open(summary_file, "w") as f:
            f.write(updated_content)

        print(f"  ✓ Updated {summary_file}")
    else:
        print(f"  ⚠️  Benchmark summary not found at {summary_file}, creating new one...")
        with open(summary_file, "w") as f:
            f.write("# sc-audit Comparative Analysis\n")
            f.write(limitations_md)

    print("\n" + "=" * 80)
    print("✅ COMPARATIVE DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - Sensitivity data: {output_dir}/lysis_sensitivity.csv")
    print(f"  - Sensitivity data: {output_dir}/doublet_sensitivity.csv")
    print(f"  - Figure 1: {figures_dir}/fig1_sensitivity_curves.png")
    print(f"  - Figure 2: {figures_dir}/fig2_distribution_divergence.png")
    print(f"  - Figure 3: {figures_dir}/fig3_biological_hallucination_heatmap.png")
    print(f"  - Report: {summary_file}\n")


if __name__ == "__main__":
    main()
