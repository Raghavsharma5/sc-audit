"""
Extractor module for converting AnnData objects to Pydantic schemas.

This module serializes processed AnnData objects into structured DatasetQCSummary
instances that can be sent to auditors without exposing raw expression matrices.
"""

from typing import List, Literal
import numpy as np
import pandas as pd
from anndata import AnnData

from ..auditor.schemas import (
    DistributionStats,
    ClusterQCProfile,
    DatasetQCSummary,
)


def compute_distribution_stats(values: np.ndarray) -> DistributionStats:
    """
    Compute summary statistics for a distribution.

    Parameters
    ----------
    values : np.ndarray
        Array of numerical values.

    Returns
    -------
    DistributionStats
        Pydantic model with percentiles and summary statistics.
    """
    # Remove NaN values if present
    values = values[~np.isnan(values)]

    return DistributionStats(
        min=float(np.min(values)),
        p5=float(np.percentile(values, 5)),
        p25=float(np.percentile(values, 25)),
        p50=float(np.percentile(values, 50)),
        p75=float(np.percentile(values, 75)),
        p95=float(np.percentile(values, 95)),
        max=float(np.max(values)),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
    )


def extract_top_marker_genes(
    adata: AnnData,
    cluster_id: str,
    n_genes: int = 5,
) -> List[str]:
    """
    Extract top marker genes for a specific cluster.

    Parameters
    ----------
    adata : AnnData
        Processed AnnData object with rank_genes_groups results.
    cluster_id : str
        Cluster identifier (e.g., "0", "1", "2").
    n_genes : int, default 5
        Number of top marker genes to extract.

    Returns
    -------
    List[str]
        List of gene names (up to n_genes).
    """
    # Check if marker genes have been computed
    if "rank_genes_groups" not in adata.uns:
        return []

    # Extract gene names for this cluster
    try:
        gene_names = adata.uns["rank_genes_groups"]["names"][cluster_id]
        # Convert to list and take top n_genes
        if isinstance(gene_names, np.ndarray):
            return gene_names[:n_genes].tolist()
        else:
            return list(gene_names[:n_genes])
    except (KeyError, IndexError):
        return []


def extract_cluster_profile(
    adata: AnnData,
    cluster_id: str,
    total_cells: int,
) -> ClusterQCProfile:
    """
    Extract QC profile for a single cluster.

    Parameters
    ----------
    adata : AnnData
        Processed AnnData object with QC metrics and clustering.
    cluster_id : str
        Cluster identifier.
    total_cells : int
        Total number of cells in the dataset (for percentage calculation).

    Returns
    -------
    ClusterQCProfile
        Pydantic model with cluster-specific QC metrics.
    """
    # Get cells in this cluster
    cluster_mask = adata.obs["leiden"] == cluster_id
    cluster_obs = adata.obs[cluster_mask]

    # Count cells in cluster
    cell_count = cluster_mask.sum()

    # Calculate percentage
    pct_of_total = (cell_count / total_cells) * 100.0

    # Extract median QC metrics
    median_total_counts = float(cluster_obs["total_counts"].median())
    median_genes_detected = float(cluster_obs["n_genes_by_counts"].median())
    median_pct_mt = float(cluster_obs["pct_counts_mt"].median())

    # Extract top marker genes
    top_marker_genes = extract_top_marker_genes(adata, cluster_id, n_genes=5)

    return ClusterQCProfile(
        cluster_id=cluster_id,
        cell_count=int(cell_count),
        pct_of_total=float(pct_of_total),
        median_total_counts=median_total_counts,
        median_genes_detected=median_genes_detected,
        median_pct_mt=median_pct_mt,
        top_marker_genes=top_marker_genes,
    )


def extract_qc_summary(
    adata: AnnData,
    dataset_name: str,
    species: Literal["human", "mouse"] = "human",
) -> DatasetQCSummary:
    """
    Extract complete QC summary from processed AnnData.

    Parameters
    ----------
    adata : AnnData
        Processed AnnData object with QC metrics and clustering.
    dataset_name : str
        Human-readable dataset name (e.g., "pbmc3k").
    species : str, default "human"
        Species identifier.

    Returns
    -------
    DatasetQCSummary
        Complete QC summary ready for audit.

    Raises
    ------
    ValueError
        If required QC metrics or clustering are missing.
    """
    # Validate required columns
    required_cols = ["total_counts", "n_genes_by_counts", "pct_counts_mt", "leiden"]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required QC metrics: {missing_cols}. "
            "Run qc_engine.standard_pipeline() first."
        )

    # Extract basic dataset info
    total_cells = adata.n_obs
    total_genes = adata.n_vars

    # Compute distribution statistics
    total_counts_distribution = compute_distribution_stats(adata.obs["total_counts"].values)
    detected_genes_distribution = compute_distribution_stats(
        adata.obs["n_genes_by_counts"].values
    )
    mito_pct_distribution = compute_distribution_stats(adata.obs["pct_counts_mt"].values)

    # Extract cluster profiles
    cluster_ids = sorted(adata.obs["leiden"].unique())
    cluster_profiles = [
        extract_cluster_profile(adata, cluster_id, total_cells) for cluster_id in cluster_ids
    ]

    return DatasetQCSummary(
        dataset_name=dataset_name,
        total_cells=total_cells,
        total_genes=total_genes,
        species=species,
        total_counts_distribution=total_counts_distribution,
        detected_genes_distribution=detected_genes_distribution,
        mito_pct_distribution=mito_pct_distribution,
        cluster_profiles=cluster_profiles,
    )


if __name__ == "__main__":
    """Quick test of the extractor."""
    from pathlib import Path
    import json
    from .ingest import load_pbmc3k
    from .qc_engine import run_qc_pipeline

    print("Loading and processing PBMC3k dataset...")
    project_root = Path(__file__).parent.parent.parent
    adata = load_pbmc3k(cache_dir=project_root / "data")
    adata_processed = run_qc_pipeline(adata, species="human")

    print("\nExtracting QC summary...")
    summary = extract_qc_summary(adata_processed, dataset_name="pbmc3k", species="human")

    print(f"\nDataset: {summary.dataset_name}")
    print(f"Cells: {summary.total_cells}")
    print(f"Genes: {summary.total_genes}")
    print(f"Clusters: {len(summary.cluster_profiles)}")

    print(f"\nTotal counts distribution:")
    print(f"  min={summary.total_counts_distribution.min:.1f}")
    print(f"  p50={summary.total_counts_distribution.p50:.1f}")
    print(f"  max={summary.total_counts_distribution.max:.1f}")

    print(f"\n% MT distribution:")
    print(f"  min={summary.mito_pct_distribution.min:.2f}%")
    print(f"  p50={summary.mito_pct_distribution.p50:.2f}%")
    print(f"  max={summary.mito_pct_distribution.max:.2f}%")

    print(f"\nCluster profiles:")
    for profile in summary.cluster_profiles[:3]:
        print(f"  Cluster {profile.cluster_id}: {profile.cell_count} cells")
        print(f"    Top markers: {', '.join(profile.top_marker_genes[:3])}")

    # Export as JSON to verify serialization
    print("\nSerializing to JSON...")
    summary_json = summary.model_dump()
    print(f"JSON keys: {list(summary_json.keys())}")
    print("\nSerialization successful!")
