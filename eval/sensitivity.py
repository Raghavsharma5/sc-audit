"""
Sensitivity analysis for auditor detection thresholds.

Performs parameter sweeps to identify the boundary conditions where
baseline rules vs. Claude LLM transition from MISS to DETECT.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from anndata import AnnData

from src.pipeline.ingest import load_pbmc3k
from src.pipeline.qc_engine import run_qc_pipeline
from src.pipeline.extractor import extract_qc_summary
from src.perturbation.inject import inject_lysis, inject_doublets
from src.auditor.baseline import baseline_audit
from src.auditor.claude_client import claude_audit
from src.auditor.schemas import AuditReport


def run_lysis_sensitivity_sweep(
    adata: AnnData,
    mt_scale_factors: List[float],
    target_cluster: str = "1",
    api_key: Optional[str] = None,
    species: str = "human",
) -> pd.DataFrame:
    """
    Sweep mitochondrial scaling factor and test detection sensitivity.

    Parameters
    ----------
    adata : AnnData
        Processed dataset with clustering
    mt_scale_factors : List[float]
        List of MT% scaling factors to test (e.g., [1.1, 1.25, 1.5, 2.0, 3.0])
    target_cluster : str
        Cluster to inject lysis into
    api_key : str, optional
        Anthropic API key for Claude auditor
    species : str
        Species for QC pipeline

    Returns
    -------
    pd.DataFrame
        Results with columns: mt_scale, baseline_detected, claude_detected,
        baseline_verdict, claude_verdict, median_pct_mt_target, median_pct_mt_dataset
    """
    results = []

    for mt_scale in mt_scale_factors:
        print(f"  Testing MT scale factor: {mt_scale:.2f}x")

        # Generate lysis with specific scaling
        adata_lysis, metadata = inject_lysis(
            adata.copy(),
            target_cluster=target_cluster,
            mt_scale_factor=mt_scale,
            species=species,
        )

        # Re-run QC pipeline
        adata_lysis_processed = run_qc_pipeline(adata_lysis, species=species)

        # Extract summary
        summary = extract_qc_summary(
            adata_lysis_processed,
            dataset_name=f"lysis_mt{mt_scale:.2f}x",
            species=species,
        )

        # Get actual median MT% in target cluster
        target_profile = next(
            (c for c in summary.cluster_profiles if c.cluster_id == target_cluster),
            None,
        )
        median_pct_mt_target = (
            target_profile.median_pct_mt if target_profile else 0.0
        )
        median_pct_mt_dataset = summary.mito_pct_distribution.p50

        # Run baseline auditor
        baseline_report = baseline_audit(summary)
        baseline_detected = any(
            f.flag_type == "HIGH_MITOCHONDRIAL_READS"
            and target_cluster in f.affected_clusters
            for f in baseline_report.flags
        )

        # Run Claude auditor if API key available
        claude_detected = False
        claude_verdict = "SKIPPED"
        if api_key:
            try:
                claude_report = claude_audit(summary, api_key=api_key)
                claude_detected = any(
                    f.flag_type == "HIGH_MITOCHONDRIAL_READS"
                    for f in claude_report.flags
                )
                claude_verdict = claude_report.summary_verdict
            except Exception as e:
                print(f"    ⚠️  Claude audit failed: {e}")
                claude_verdict = "ERROR"

        results.append(
            {
                "mt_scale": mt_scale,
                "baseline_detected": baseline_detected,
                "claude_detected": claude_detected,
                "baseline_verdict": baseline_report.summary_verdict,
                "claude_verdict": claude_verdict,
                "median_pct_mt_target": median_pct_mt_target,
                "median_pct_mt_dataset": median_pct_mt_dataset,
                "n_baseline_flags": len(baseline_report.flags),
                "n_claude_flags": (
                    len(claude_report.flags)
                    if api_key and claude_verdict != "ERROR"
                    else 0
                ),
            }
        )

        print(
            f"    Baseline: {'DETECTED' if baseline_detected else 'MISSED'} "
            f"(median MT% = {median_pct_mt_target:.2f}%)"
        )
        if api_key and claude_verdict != "ERROR":
            print(f"    Claude:   {'DETECTED' if claude_detected else 'MISSED'}")

    return pd.DataFrame(results)


def run_doublet_sensitivity_sweep(
    adata: AnnData,
    doublet_rates: List[float],
    api_key: Optional[str] = None,
    species: str = "human",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Sweep doublet rate and test detection sensitivity.

    Parameters
    ----------
    adata : AnnData
        Processed dataset with clustering
    doublet_rates : List[float]
        List of doublet rates to test (e.g., [0.02, 0.05, 0.08, 0.12, 0.20])
    api_key : str, optional
        Anthropic API key for Claude auditor
    species : str
        Species for QC pipeline
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Results with columns: doublet_rate, baseline_detected, claude_detected,
        n_doublets_created, median_counts_doublets, median_counts_original
    """
    results = []

    for doublet_rate in doublet_rates:
        print(f"  Testing doublet rate: {doublet_rate:.1%}")

        # Generate doublets with specific rate
        adata_doublets, metadata = inject_doublets(
            adata.copy(), doublet_rate=doublet_rate, random_seed=random_seed
        )

        # Re-run QC pipeline
        adata_doublets_processed = run_qc_pipeline(adata_doublets, species=species)

        # Extract summary
        summary = extract_qc_summary(
            adata_doublets_processed,
            dataset_name=f"doublets_{doublet_rate:.0%}",
            species=species,
        )

        # Calculate actual doublet metrics
        doublet_mask = adata_doublets.obs.get("is_synthetic_doublet", False) == True
        n_doublets = doublet_mask.sum() if doublet_mask.any() else 0
        median_counts_doublets = (
            adata_doublets.obs.loc[doublet_mask, "total_counts"].median()
            if doublet_mask.any()
            else 0
        )
        median_counts_original = (
            adata_doublets.obs.loc[~doublet_mask, "total_counts"].median()
            if (~doublet_mask).any()
            else 0
        )

        # Run baseline auditor
        baseline_report = baseline_audit(summary)
        baseline_detected = any(
            f.flag_type == "POTENTIAL_DOUBLETS" for f in baseline_report.flags
        )

        # Run Claude auditor if API key available
        claude_detected = False
        claude_verdict = "SKIPPED"
        if api_key:
            try:
                claude_report = claude_audit(summary, api_key=api_key)
                claude_detected = any(
                    f.flag_type == "POTENTIAL_DOUBLETS" for f in claude_report.flags
                )
                claude_verdict = claude_report.summary_verdict
            except Exception as e:
                print(f"    ⚠️  Claude audit failed: {e}")
                claude_verdict = "ERROR"

        results.append(
            {
                "doublet_rate": doublet_rate,
                "baseline_detected": baseline_detected,
                "claude_detected": claude_detected,
                "baseline_verdict": baseline_report.summary_verdict,
                "claude_verdict": claude_verdict,
                "n_doublets_created": n_doublets,
                "median_counts_doublets": median_counts_doublets,
                "median_counts_original": median_counts_original,
                "counts_ratio": (
                    median_counts_doublets / median_counts_original
                    if median_counts_original > 0
                    else 0
                ),
                "n_baseline_flags": len(baseline_report.flags),
                "n_claude_flags": (
                    len(claude_report.flags)
                    if api_key and claude_verdict != "ERROR"
                    else 0
                ),
            }
        )

        print(
            f"    Baseline: {'DETECTED' if baseline_detected else 'MISSED'} "
            f"({n_doublets} doublets, {median_counts_doublets / median_counts_original:.2f}x counts)"
        )
        if api_key and claude_verdict != "ERROR":
            print(f"    Claude:   {'DETECTED' if claude_detected else 'MISSED'}")

    return pd.DataFrame(results)


def run_full_sensitivity_analysis(
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    species: str = "human",
) -> Dict[str, pd.DataFrame]:
    """
    Run complete sensitivity analysis across all perturbation types.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory for caching PBMC3k dataset
    api_key : str, optional
        Anthropic API key. If None, reads from ANTHROPIC_API_KEY environment variable.
    species : str
        Species for QC pipeline

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys "lysis" and "doublets" containing sweep results
    """
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: Boundary Detection Thresholds")
    print("=" * 80)

    # Load and process base dataset
    print("\n[1/4] Loading PBMC3k dataset...")
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data"
    adata = load_pbmc3k(cache_dir=cache_dir)

    print("[2/4] Running initial QC pipeline...")
    adata_processed = run_qc_pipeline(adata, species=species)

    # Check API key
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key is None:
        print(
            "\n⚠️  WARNING: ANTHROPIC_API_KEY not found. Claude sensitivity analysis will be skipped."
        )

    # Run lysis sensitivity sweep
    print("\n[3/4] Running lysis sensitivity sweep...")
    mt_scale_factors = [1.1, 1.25, 1.5, 2.0, 3.0, 4.0]  # +10% to +300%
    lysis_results = run_lysis_sensitivity_sweep(
        adata_processed,
        mt_scale_factors=mt_scale_factors,
        api_key=api_key,
        species=species,
    )

    # Run doublet sensitivity sweep
    print("\n[4/4] Running doublet sensitivity sweep...")
    doublet_rates = [0.02, 0.05, 0.08, 0.12, 0.20]  # 2% to 20%
    doublet_results = run_doublet_sensitivity_sweep(
        adata_processed, doublet_rates=doublet_rates, api_key=api_key, species=species
    )

    print("\n✅ Sensitivity analysis complete!")

    return {"lysis": lysis_results, "doublets": doublet_results}


if __name__ == "__main__":
    """Quick test of sensitivity analysis."""
    results = run_full_sensitivity_analysis()

    print("\n" + "=" * 80)
    print("LYSIS SENSITIVITY RESULTS")
    print("=" * 80)
    print(results["lysis"].to_string())

    print("\n" + "=" * 80)
    print("DOUBLET SENSITIVITY RESULTS")
    print("=" * 80)
    print(results["doublets"].to_string())
