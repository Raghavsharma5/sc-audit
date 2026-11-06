"""
Perturbation injection module for generating ground-truth test cases.

This module creates controlled pathological conditions in scRNA-seq data
to test whether auditors can correctly identify technical artifacts.
"""

from typing import Tuple, Dict, List, Literal
import numpy as np
import scanpy as sc
from anndata import AnnData, concat
from scipy.sparse import issparse, csr_matrix
import warnings


PerturbationCondition = Literal["clean_control", "injected_lysis", "injected_doublets", "injected_degraded"]


def _ensure_raw_counts(adata: AnnData) -> AnnData:
    """
    Ensure we have raw counts in .X before perturbation.

    Parameters
    ----------
    adata : AnnData
        Input data (may be normalized/log-transformed)

    Returns
    -------
    AnnData
        AnnData with raw counts in .X
    """
    if adata.raw is not None:
        # Use raw counts from .raw
        return AnnData(
            X=adata.raw.X.copy(),
            obs=adata.obs.copy(),
            var=adata.raw.var.copy(),
            uns=adata.uns.copy(),
        )
    else:
        # Assume .X contains raw counts
        return adata.copy()


def inject_lysis(
    adata: AnnData,
    target_cluster: str = "1",
    mt_scale_factor: float = 5.0,
    species: Literal["human", "mouse"] = "human",
) -> Tuple[AnnData, Dict]:
    """
    Inject apoptotic lysis by inflating mitochondrial gene counts in a target cluster.

    This simulates membrane damage during dissociation that causes cytoplasmic
    mRNA to leak out while mitochondrial transcripts remain trapped.

    Parameters
    ----------
    adata : AnnData
        Input AnnData with raw counts and 'leiden' clustering
    target_cluster : str
        Leiden cluster ID to target
    mt_scale_factor : float
        Multiplicative factor for MT gene counts (5.0 = +400%)
    species : str
        Species for MT gene identification

    Returns
    -------
    adata_perturbed : AnnData
        Perturbed dataset
    metadata : dict
        Perturbation metadata
    """
    # Ensure we have raw counts
    adata = _ensure_raw_counts(adata)

    # Ensure leiden clustering exists
    if "leiden" not in adata.obs.columns:
        raise ValueError("Input AnnData must have 'leiden' cluster assignments")

    # Identify mitochondrial genes
    if species == "human":
        mt_pattern = "^MT-"
    elif species == "mouse":
        mt_pattern = "^mt-"
    else:
        raise ValueError(f"Unsupported species: {species}")

    mt_mask = adata.var_names.str.match(mt_pattern)
    mt_gene_indices = np.where(mt_mask)[0]

    if len(mt_gene_indices) == 0:
        warnings.warn(f"No mitochondrial genes found with pattern {mt_pattern}")
        return adata, {"condition": "injected_lysis", "warning": "no_mt_genes", "expected_flags": []}

    # Identify target cells
    target_mask = adata.obs["leiden"] == target_cluster
    target_cell_indices = np.where(target_mask)[0]

    if len(target_cell_indices) == 0:
        warnings.warn(f"Target cluster {target_cluster} not found")
        return adata, {"condition": "injected_lysis", "warning": "cluster_not_found", "expected_flags": []}

    # Scale mitochondrial gene counts in target cells
    X = adata.X
    if issparse(X):
        # For sparse matrices, extract the relevant submatrix, modify, and update
        X = X.tocsr()  # Ensure CSR format for efficient row slicing
        for cell_idx in target_cell_indices:
            for gene_idx in mt_gene_indices:
                current_value = X[cell_idx, gene_idx]
                X[cell_idx, gene_idx] = current_value * mt_scale_factor
        adata.X = X
    else:
        # For dense matrices
        X[np.ix_(target_cell_indices, mt_gene_indices)] *= mt_scale_factor
        adata.X = X

    # Recompute QC metrics
    adata.var["mt"] = mt_mask
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    metadata = {
        "condition": "injected_lysis",
        "target_cluster": target_cluster,
        "mt_scale_factor": mt_scale_factor,
        "n_cells_affected": len(target_cell_indices),
        "n_mt_genes": len(mt_gene_indices),
        "expected_flags": ["HIGH_MITOCHONDRIAL_READS"],
        "expected_affected_clusters": [target_cluster],
    }

    return adata, metadata


def inject_doublets(
    adata: AnnData,
    doublet_rate: float = 0.08,
    random_seed: int = 42,
) -> Tuple[AnnData, Dict]:
    """
    Inject synthetic doublets by summing expression profiles of random cell pairs.

    This simulates multi-cell co-encapsulation in droplets, which produces
    cells with elevated total counts and mixed cell-type marker expression.

    Parameters
    ----------
    adata : AnnData
        Input AnnData with raw counts
    doublet_rate : float
        Fraction of cells to generate as doublets (0.08 = 8%)
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    adata_perturbed : AnnData
        Dataset with synthetic doublets appended
    metadata : dict
        Perturbation metadata
    """
    # Ensure we have raw counts
    adata = _ensure_raw_counts(adata)

    np.random.seed(random_seed)

    n_cells = adata.n_obs
    n_doublets = int(n_cells * doublet_rate)

    if n_doublets < 1:
        warnings.warn("doublet_rate too low, no doublets generated")
        return adata, {"condition": "injected_doublets", "n_doublets": 0, "expected_flags": []}

    # Randomly select pairs of cells to merge
    cell_indices_a = np.random.choice(n_cells, size=n_doublets, replace=True)
    cell_indices_b = np.random.choice(n_cells, size=n_doublets, replace=True)

    # Sum expression profiles
    X = adata.X
    if issparse(X):
        X_doublets = X[cell_indices_a, :] + X[cell_indices_b, :]
    else:
        X_doublets = X[cell_indices_a, :] + X[cell_indices_b, :]

    # Create obs dataframe for doublets
    obs_doublets = adata.obs.iloc[cell_indices_a].copy()
    obs_doublets.index = [f"DOUBLET_{i}" for i in range(n_doublets)]
    obs_doublets["is_synthetic_doublet"] = True

    # Mark original cells as non-doublets
    adata.obs["is_synthetic_doublet"] = False

    # Concatenate original and doublet data
    adata_doublets = AnnData(
        X=X_doublets,
        obs=obs_doublets,
        var=adata.var.copy(),
    )

    # Combine using concat
    adata_combined = concat(
        [adata, adata_doublets],
        axis=0,
        join="outer",
        label="batch",
        keys=["original", "doublets"],
        index_unique=None,
    )

    # Ensure 'mt' column is preserved in var after concat
    if "mt" in adata.var.columns and "mt" not in adata_combined.var.columns:
        adata_combined.var["mt"] = adata.var["mt"]

    # Recompute QC metrics
    if "mt" in adata_combined.var.columns:
        sc.pp.calculate_qc_metrics(
            adata_combined,
            qc_vars=["mt"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )
    else:
        sc.pp.calculate_qc_metrics(
            adata_combined,
            percent_top=None,
            log1p=False,
            inplace=True,
        )

    metadata = {
        "condition": "injected_doublets",
        "doublet_rate": doublet_rate,
        "n_doublets": n_doublets,
        "n_original_cells": n_cells,
        "n_total_cells": adata_combined.n_obs,
        "expected_flags": ["POTENTIAL_DOUBLETS"],
    }

    return adata_combined, metadata


def inject_degraded(
    adata: AnnData,
    target_cluster: str = "2",
    target_genes_median: int = 250,
    random_seed: int = 42,
) -> Tuple[AnnData, Dict]:
    """
    Inject library degradation by downsampling detected genes in a target cluster.

    This simulates degraded libraries or empty droplet contamination where
    library complexity is reduced but some counts remain.

    Parameters
    ----------
    adata : AnnData
        Input AnnData with raw counts and 'leiden' clustering
    target_cluster : str
        Leiden cluster ID to target
    target_genes_median : int
        Target median number of detected genes (should be < 300)
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    adata_perturbed : AnnData
        Perturbed dataset
    metadata : dict
        Perturbation metadata
    """
    # Ensure we have raw counts
    adata = _ensure_raw_counts(adata)

    # Ensure leiden clustering exists
    if "leiden" not in adata.obs.columns:
        raise ValueError("Input AnnData must have 'leiden' cluster assignments")

    np.random.seed(random_seed)

    # Identify target cells
    target_mask = adata.obs["leiden"] == target_cluster
    target_cell_indices = np.where(target_mask)[0]

    if len(target_cell_indices) == 0:
        warnings.warn(f"Target cluster {target_cluster} not found")
        return adata, {"condition": "injected_degraded", "warning": "cluster_not_found", "expected_flags": []}

    # For each target cell, randomly drop genes to reduce complexity
    X = adata.X
    if issparse(X):
        X = X.tocsr()

    for cell_idx in target_cell_indices:
        # Get current gene counts for this cell
        if issparse(X):
            cell_counts = X[cell_idx, :].toarray().flatten()
        else:
            cell_counts = X[cell_idx, :].copy()

        # Find genes that are expressed
        expressed_genes = np.where(cell_counts > 0)[0]
        n_expressed = len(expressed_genes)

        if n_expressed <= target_genes_median:
            continue  # Already degraded enough

        # Randomly select genes to keep (target_genes_median genes)
        n_to_keep = max(target_genes_median, int(n_expressed * 0.3))  # Keep at least 30% or target
        genes_to_keep = np.random.choice(expressed_genes, size=n_to_keep, replace=False)

        # Create mask for genes to zero out
        genes_to_zero = np.setdiff1d(expressed_genes, genes_to_keep)

        # Zero out the dropped genes
        if issparse(X):
            for gene_idx in genes_to_zero:
                X[cell_idx, gene_idx] = 0
        else:
            X[cell_idx, genes_to_zero] = 0

    # Update the data
    if issparse(X):
        X.eliminate_zeros()  # Remove explicit zeros from sparse matrix
        adata.X = X
    else:
        adata.X = X

    # Recompute QC metrics
    if "mt" in adata.var.columns:
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )
    else:
        sc.pp.calculate_qc_metrics(
            adata,
            percent_top=None,
            log1p=False,
            inplace=True,
        )

    metadata = {
        "condition": "injected_degraded",
        "target_cluster": target_cluster,
        "target_genes_median": target_genes_median,
        "n_cells_affected": len(target_cell_indices),
        "expected_flags": ["LOW_LIBRARY_COMPLEXITY"],
        "expected_affected_clusters": [target_cluster],
    }

    return adata, metadata


def generate_perturbed_dataset(
    adata: AnnData,
    condition: PerturbationCondition,
    species: Literal["human", "mouse"] = "human",
    **kwargs,
) -> Tuple[AnnData, Dict]:
    """
    Factory function to generate perturbed datasets for testing auditors.

    Parameters
    ----------
    adata : AnnData
        Input AnnData with QC metrics and clustering
    condition : str
        Perturbation condition: "clean_control", "injected_lysis",
        "injected_doublets", or "injected_degraded"
    species : str
        Species for mitochondrial gene identification
    **kwargs
        Additional arguments passed to specific perturbation functions

    Returns
    -------
    adata_perturbed : AnnData
        Perturbed dataset
    metadata : dict
        Perturbation metadata including expected flags

    Examples
    --------
    >>> adata_lysis, meta = generate_perturbed_dataset(adata, "injected_lysis")
    >>> print(meta["expected_flags"])
    ['HIGH_MITOCHONDRIAL_READS']
    """
    if condition == "clean_control":
        # Return unperturbed copy
        adata_clean = _ensure_raw_counts(adata)
        metadata = {
            "condition": "clean_control",
            "expected_flags": [],
        }
        return adata_clean, metadata

    elif condition == "injected_lysis":
        return inject_lysis(adata, species=species, **kwargs)

    elif condition == "injected_doublets":
        return inject_doublets(adata, **kwargs)

    elif condition == "injected_degraded":
        return inject_degraded(adata, **kwargs)

    else:
        raise ValueError(
            f"Unknown condition: {condition}. "
            f"Must be one of: clean_control, injected_lysis, injected_doublets, injected_degraded"
        )


if __name__ == "__main__":
    """Quick test of perturbation injection."""
    from pathlib import Path
    from ..pipeline.ingest import load_pbmc3k
    from ..pipeline.qc_engine import run_qc_pipeline
    from ..pipeline.extractor import extract_qc_summary

    print("Loading and processing PBMC3k dataset...")
    project_root = Path(__file__).parent.parent.parent
    adata = load_pbmc3k(cache_dir=project_root / "data")
    adata_processed = run_qc_pipeline(adata, species="human")

    print(f"\nOriginal data: {adata_processed.n_obs} cells")
    print(f"Clusters: {adata_processed.obs['leiden'].nunique()}")

    # Test each perturbation
    for condition in ["clean_control", "injected_lysis", "injected_doublets", "injected_degraded"]:
        print(f"\n{'='*60}")
        print(f"Testing: {condition}")
        print(f"{'='*60}")

        adata_perturbed, metadata = generate_perturbed_dataset(
            adata_processed, condition, species="human"
        )

        print(f"Metadata: {metadata}")

        if condition == "injected_lysis" and "target_cluster" in metadata:
            cluster_id = metadata["target_cluster"]
            cluster_mask = adata_perturbed.obs["leiden"] == cluster_id
            pct_mt = adata_perturbed.obs.loc[cluster_mask, "pct_counts_mt"].median()
            print(f"Cluster {cluster_id} median pct_mt: {pct_mt:.2f}%")

        elif condition == "injected_doublets":
            if "is_synthetic_doublet" in adata_perturbed.obs.columns:
                doublet_mask = adata_perturbed.obs["is_synthetic_doublet"]
                doublet_counts = adata_perturbed.obs.loc[doublet_mask, "total_counts"].median()
                original_counts = adata_perturbed.obs.loc[~doublet_mask, "total_counts"].median()
                print(f"Doublet median counts: {doublet_counts:.0f}")
                print(f"Original median counts: {original_counts:.0f}")
                print(f"Ratio: {doublet_counts / original_counts:.2f}x")

        elif condition == "injected_degraded" and "target_cluster" in metadata:
            cluster_id = metadata["target_cluster"]
            cluster_mask = adata_perturbed.obs["leiden"] == cluster_id
            genes_detected = adata_perturbed.obs.loc[cluster_mask, "n_genes_by_counts"].median()
            print(f"Cluster {cluster_id} median genes detected: {genes_detected:.0f}")
