#!/usr/bin/env python3
"""
CLI script to run QC audits on arbitrary AnnData files.

Usage:
    python scripts/run_audit.py --input data.h5ad --mode baseline --output report.json
    python scripts/run_audit.py --input data.h5ad --mode claude --output report.json
    python scripts/run_audit.py --input data.h5ad --mode both --output reports/

Examples:
    # Run baseline audit only
    python scripts/run_audit.py --input data/processed.h5ad --mode baseline

    # Run Claude audit with custom API key
    python scripts/run_audit.py --input data/processed.h5ad --mode claude --api-key sk-...

    # Run both auditors and save to directory
    python scripts/run_audit.py --input data/processed.h5ad --mode both --output reports/
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Literal
import scanpy as sc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.extractor import extract_qc_summary
from src.auditor.baseline import baseline_audit
from src.auditor.claude_client import claude_audit
from src.auditor.schemas import AuditReport


def save_report(report: AuditReport, output_path: Path):
    """
    Save audit report to JSON file.

    Parameters
    ----------
    report : AuditReport
        Audit report to save
    output_path : Path
        Output file path (.json)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    print(f"  ✓ Saved report: {output_path}")


def print_report_summary(report: AuditReport, auditor_name: str):
    """
    Print human-readable summary of audit report.

    Parameters
    ----------
    report : AuditReport
        Audit report to display
    auditor_name : str
        Name of auditor ("baseline" or "claude")
    """
    print(f"\n{'=' * 80}")
    print(f"{auditor_name.upper()} AUDIT REPORT")
    print(f"{'=' * 80}")

    print(f"\nSummary Verdict: {report.summary_verdict}")
    print(f"Total Flags: {len(report.flags)}")

    if len(report.flags) > 0:
        print("\nFlags Detected:")
        for i, flag in enumerate(report.flags, 1):
            print(f"\n  [{i}] {flag.flag_type}")
            print(f"      Severity: {flag.severity}")
            print(f"      Affected Clusters: {', '.join(flag.affected_clusters)}")
            print(f"      Evidence:")
            for evidence in flag.numerical_evidence[:3]:  # Show first 3 evidence items
                print(f"        - {evidence}")
            if len(flag.numerical_evidence) > 3:
                print(f"        ... and {len(flag.numerical_evidence) - 3} more")

    print("\nRecommended Actions:")
    for i, action in enumerate(report.recommended_actions, 1):
        print(f"  {i}. {action}")

    print(f"\n{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run QC audits on scRNA-seq AnnData files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input AnnData file (.h5ad) with QC metrics and clustering",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "claude", "both"],
        default="baseline",
        help="Auditor to run: baseline (deterministic), claude (LLM), or both",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for audit report (JSON). If --mode=both, should be a directory.",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (for Claude mode). Reads from ANTHROPIC_API_KEY env var if not provided.",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name for QC summary (defaults to input filename)",
    )

    parser.add_argument(
        "--species",
        type=str,
        choices=["human", "mouse"],
        default="human",
        help="Species for QC analysis (default: human)",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ ERROR: Input file not found: {input_path}")
        sys.exit(1)

    if not input_path.suffix == ".h5ad":
        print(f"⚠️  WARNING: Input file does not have .h5ad extension")

    # Determine dataset name
    dataset_name = args.dataset_name or input_path.stem

    # Load AnnData
    print(f"\nLoading AnnData from {input_path}...")
    try:
        adata = sc.read_h5ad(input_path)
        print(f"  ✓ Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    except Exception as e:
        print(f"❌ ERROR: Failed to load AnnData: {e}")
        sys.exit(1)

    # Verify required fields
    required_obs_cols = ["total_counts", "n_genes_by_counts", "pct_counts_mt", "leiden"]
    missing_cols = [col for col in required_obs_cols if col not in adata.obs.columns]

    if missing_cols:
        print(f"❌ ERROR: Missing required columns in adata.obs: {missing_cols}")
        print("   Make sure the dataset has been processed with run_qc_pipeline()")
        sys.exit(1)

    # Extract QC summary
    print(f"\nExtracting QC summary...")
    try:
        summary = extract_qc_summary(adata, dataset_name=dataset_name, species=args.species)
        print(f"  ✓ Summary extracted: {summary.total_cells} cells, {len(summary.cluster_profiles)} clusters")
    except Exception as e:
        print(f"❌ ERROR: Failed to extract QC summary: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run audits based on mode
    if args.mode in ["baseline", "both"]:
        print(f"\nRunning baseline audit...")
        try:
            baseline_report = baseline_audit(summary)
            print_report_summary(baseline_report, "baseline")

            # Save if output specified
            if args.output:
                if args.mode == "both":
                    output_dir = Path(args.output)
                    save_report(baseline_report, output_dir / "baseline_report.json")
                else:
                    save_report(baseline_report, Path(args.output))

        except Exception as e:
            print(f"❌ ERROR: Baseline audit failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    if args.mode in ["claude", "both"]:
        print(f"\nRunning Claude audit...")
        try:
            claude_report = claude_audit(summary, api_key=args.api_key)
            print_report_summary(claude_report, "claude")

            # Save if output specified
            if args.output:
                if args.mode == "both":
                    output_dir = Path(args.output)
                    save_report(claude_report, output_dir / "claude_report.json")
                else:
                    save_report(claude_report, Path(args.output))

        except Exception as e:
            print(f"❌ ERROR: Claude audit failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("✅ Audit complete!\n")


if __name__ == "__main__":
    main()
