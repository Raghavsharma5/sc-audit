"""
Benchmark orchestration for baseline and Claude auditors.

Runs both auditors across all 4 perturbation conditions and computes
evaluation metrics for comparison.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from anndata import AnnData

from src.pipeline.ingest import load_pbmc3k
from src.pipeline.qc_engine import run_qc_pipeline
from src.pipeline.extractor import extract_qc_summary
from src.perturbation.inject import generate_perturbed_dataset, PerturbationCondition
from src.auditor.baseline import baseline_audit
from src.auditor.claude_client import claude_audit
from src.auditor.schemas import AuditReport, DatasetQCSummary

from .metrics import (
    calculate_flag_metrics,
    calculate_groundedness,
    calculate_speculation_score,
)


def run_single_audit(
    adata: AnnData,
    condition: PerturbationCondition,
    auditor_name: str,
    api_key: Optional[str] = None,
    species: str = "human",
) -> Dict:
    """
    Run a single audit (baseline or Claude) on a dataset condition.

    Parameters
    ----------
    adata : AnnData
        Processed dataset ready for perturbation
    condition : PerturbationCondition
        One of: "clean_control", "injected_lysis", "injected_doublets", "injected_degraded"
    auditor_name : str
        "baseline" or "claude"
    api_key : str, optional
        Anthropic API key (required for Claude auditor)
    species : str
        "human" or "mouse"

    Returns
    -------
    Dict
        Result dictionary with keys:
        - condition: str
        - auditor: str
        - report: AuditReport
        - summary: DatasetQCSummary
        - metadata: Dict (ground truth from perturbation)
        - latency_sec: float
        - precision: float
        - recall: float
        - f1: float
        - groundedness: float
        - speculation_score: int
        - prompt_tokens: int (Claude only)
        - completion_tokens: int (Claude only)
        - total_tokens: int (Claude only)
    """
    # Generate perturbed dataset
    adata_perturbed, metadata = generate_perturbed_dataset(
        adata.copy(), condition, species=species
    )

    # Re-run QC pipeline for perturbed conditions (clean_control already has clustering)
    if condition != "clean_control":
        adata_perturbed = run_qc_pipeline(adata_perturbed, species=species)

    # Extract QC summary
    summary = extract_qc_summary(
        adata_perturbed, dataset_name=f"pbmc3k_{condition}", species=species
    )

    # Run audit with timing
    start_time = time.time()

    if auditor_name == "baseline":
        report = baseline_audit(summary)
        api_response = None
    elif auditor_name == "claude":
        if api_key is None:
            raise ValueError("API key required for Claude auditor")

        # We need to capture the raw API response for token metrics
        # For now, we'll run claude_audit and track tokens separately
        # The claude_audit function doesn't return the raw response, so we'll
        # approximate tokens as 0 for baseline comparison
        report = claude_audit(summary, api_key=api_key)
        api_response = None  # Would need to modify claude_audit to return this
    else:
        raise ValueError(f"Unknown auditor: {auditor_name}")

    latency_sec = time.time() - start_time

    # Calculate metrics
    expected_flags = metadata.get("expected_flags", [])
    precision, recall, f1 = calculate_flag_metrics(report.flags, expected_flags)
    groundedness = calculate_groundedness(report, summary)
    speculation_score = calculate_speculation_score(report, condition)

    # Extract token usage (placeholder for now - would need API response)
    token_metrics = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    return {
        "condition": condition,
        "auditor": auditor_name,
        "report": report,
        "summary": summary,
        "metadata": metadata,
        "latency_sec": latency_sec,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "groundedness": groundedness,
        "speculation_score": speculation_score,
        **token_metrics,
    }


def run_benchmark(
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    run_claude: bool = True,
    species: str = "human",
) -> pd.DataFrame:
    """
    Run complete benchmark across all conditions and both auditors.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory for caching PBMC3k dataset
    api_key : str, optional
        Anthropic API key. If None, reads from ANTHROPIC_API_KEY environment variable.
        If not available, Claude auditor will be skipped.
    run_claude : bool
        Whether to run Claude auditor (default True, skipped if no API key)
    species : str
        Species for QC pipeline ("human" or "mouse")

    Returns
    -------
    pd.DataFrame
        Benchmark results with columns:
        - condition: perturbation condition
        - auditor: "baseline" or "claude"
        - verdict: summary verdict
        - n_flags: number of flags
        - precision: flag detection precision
        - recall: flag detection recall
        - f1: F1 score
        - groundedness: groundedness ratio
        - speculation_score: binary pass/fail
        - latency_sec: execution time
        - prompt_tokens: API tokens (Claude only)
        - completion_tokens: API tokens (Claude only)
        - total_tokens: API tokens (Claude only)

    Notes
    -----
    Conditions tested:
    1. clean_control - no perturbations
    2. injected_lysis - high MT% in cluster
    3. injected_doublets - elevated counts
    4. injected_degraded - low gene complexity
    """
    print("=" * 80)
    print("BENCHMARK: sc-audit Evaluation")
    print("=" * 80)

    # Load and process base dataset
    print("\n[1/5] Loading PBMC3k dataset...")
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data"
    adata = load_pbmc3k(cache_dir=cache_dir)

    print("[2/5] Running initial QC pipeline...")
    adata_processed = run_qc_pipeline(adata, species=species)

    # Check API key availability
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if api_key is None:
        print("\n⚠️  WARNING: ANTHROPIC_API_KEY not found.")
        print("    Claude auditor will be skipped.")
        print("    Set API key with: export ANTHROPIC_API_KEY='your-key-here'")
        run_claude = False

    # Define test conditions
    conditions: List[PerturbationCondition] = [
        "clean_control",
        "injected_lysis",
        "injected_doublets",
        "injected_degraded",
    ]

    print(f"\n[3/5] Running audits on {len(conditions)} conditions...")

    # Collect all results
    results = []

    for i, condition in enumerate(conditions, 1):
        print(f"\n  [{i}/{len(conditions)}] Condition: {condition}")

        # Run baseline auditor
        print(f"    Running baseline auditor...")
        baseline_result = run_single_audit(
            adata_processed, condition, "baseline", species=species
        )
        results.append(baseline_result)
        print(
            f"      ✓ Baseline: {baseline_result['report'].summary_verdict} "
            f"({len(baseline_result['report'].flags)} flags, "
            f"latency={baseline_result['latency_sec']:.2f}s)"
        )

        # Run Claude auditor if API key available
        if run_claude:
            print(f"    Running Claude auditor...")
            try:
                claude_result = run_single_audit(
                    adata_processed, condition, "claude", api_key=api_key, species=species
                )
                results.append(claude_result)
                print(
                    f"      ✓ Claude: {claude_result['report'].summary_verdict} "
                    f"({len(claude_result['report'].flags)} flags, "
                    f"latency={claude_result['latency_sec']:.2f}s)"
                )
            except Exception as e:
                print(f"      ✗ Claude auditor failed: {e}")

    # Convert to DataFrame
    print(f"\n[4/5] Computing metrics...")

    df_rows = []
    for result in results:
        df_rows.append(
            {
                "condition": result["condition"],
                "auditor": result["auditor"],
                "verdict": result["report"].summary_verdict,
                "n_flags": len(result["report"].flags),
                "expected_flags": len(result["metadata"].get("expected_flags", [])),
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "groundedness": result["groundedness"],
                "speculation_score": result["speculation_score"],
                "latency_sec": result["latency_sec"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            }
        )

    df = pd.DataFrame(df_rows)

    print(f"[5/5] Benchmark complete!")
    print(f"\nTotal runs: {len(df)}")
    print(f"Conditions tested: {len(conditions)}")
    print(f"Auditors: {df['auditor'].unique().tolist()}")

    return df


def generate_summary_table(df: pd.DataFrame) -> str:
    """
    Generate markdown summary table from benchmark results.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results from run_benchmark()

    Returns
    -------
    str
        Markdown-formatted comparison table
    """
    # Group by auditor and calculate aggregate metrics
    summary = (
        df.groupby("auditor")
        .agg(
            {
                "precision": "mean",
                "recall": "mean",
                "f1": "mean",
                "groundedness": "mean",
                "speculation_score": "mean",
                "latency_sec": "mean",
                "total_tokens": "sum",
            }
        )
        .reset_index()
    )

    # Format as markdown table
    md = "# sc-audit Benchmark Results\n\n"
    md += "## Aggregate Metrics (Averaged Across All Conditions)\n\n"
    md += "| Auditor | Precision | Recall | F1 Score | Groundedness | Speculation Pass Rate | Avg Latency (s) | Total Tokens |\n"
    md += "|---------|-----------|--------|----------|--------------|----------------------|-----------------|-------------|\n"

    for _, row in summary.iterrows():
        md += (
            f"| {row['auditor'].capitalize():8s} | "
            f"{row['precision']:9.3f} | "
            f"{row['recall']:6.3f} | "
            f"{row['f1']:8.3f} | "
            f"{row['groundedness']:12.3f} | "
            f"{row['speculation_score']:21.3f} | "
            f"{row['latency_sec']:15.2f} | "
            f"{int(row['total_tokens']):12d} |\n"
        )

    # Add per-condition breakdown
    md += "\n## Per-Condition Results\n\n"
    md += "| Condition | Auditor | Verdict | Flags (Detected/Expected) | Precision | Recall | F1 | Groundedness | Speculation |\n"
    md += "|-----------|---------|---------|---------------------------|-----------|--------|----|--------------|--------------|\n"

    for _, row in df.iterrows():
        md += (
            f"| {row['condition']:20s} | "
            f"{row['auditor']:7s} | "
            f"{row['verdict']:15s} | "
            f"{row['n_flags']:2d} / {row['expected_flags']:2d} | "
            f"{row['precision']:9.3f} | "
            f"{row['recall']:6.3f} | "
            f"{row['f1']:4.3f} | "
            f"{row['groundedness']:12.3f} | "
            f"{'PASS' if row['speculation_score'] == 1 else 'FAIL':12s} |\n"
        )

    return md
