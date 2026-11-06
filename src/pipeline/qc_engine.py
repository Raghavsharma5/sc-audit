"""
QC engine for single-cell RNA-seq data using Scanpy.

Implements standard quality control metrics computation and Leiden clustering
following the Scanpy tutorial workflow.
"""

from typing import Optional, Literal
import numpy as np
import scanpy as sc
from anndata import AnnData


def compute_qc_metrics(
    adata: AnnData,
    species: Literal["human", "mouse"] = "human",
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Calculate QC metrics including mitochondrial gene percentages.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with raw counts.
    species : str, default "human"
        Species identifier to determine mitochondrial gene prefix.
        - "human": uses "^MT-" pattern
        - "mouse": uses "^mt-" pattern
    inplace : bool, default True
        If True, modifies adata in place. If False, returns a copy.

    Returns
    -------
    AnnData or None
        If inplace=False, returns the modified AnnData object.
        If inplace=True, returns None and modifies adata in place.

    Notes
    -----
    Adds the following columns to adata.obs:
    - n_genes_by_counts: Number of genes with positive counts
    - total_counts: Total UMI counts per cell
    - total_counts_mt: Total counts from mitochondrial genes
    - pct_counts_mt: Percentage of counts from mitochondrial genes
    """
    if not inplace:
        adata = adata.copy()

    # Determine mitochondrial gene prefix based on species
    if species == "human":
        mt_pattern = "^MT-"
    elif species == "mouse":
        mt_pattern = "^mt-"
    else:
        raise ValueError(f"Unsupported species: {species}. Use 'human' or 'mouse'.")

    # Identify mitochondrial genes
    adata.var["mt"] = adata.var_names.str.match(mt_pattern)

    # Calculate QC metrics
    # Note: percent_top=None avoids computing top gene percentages (saves memory)
    # log1p=False keeps raw counts for downstream analysis
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    if not inplace:
        return adata


def standard_pipeline(
    adata: AnnData,
    species: Literal["human", "mouse"] = "human",
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mt: float = 10.0,
    n_top_genes: int = 2000,
    n_pcs: int = 40,
    n_neighbors: int = 10,
    resolution: float = 0.5,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run standard Scanpy preprocessing pipeline with QC filtering and clustering.

    Pipeline steps:
    1. Compute QC metrics (including % mitochondrial reads)
    2. Filter cells and genes
    3. Normalize to 10,000 counts per cell
    4. Log1p transform
    5. Select highly variable genes
    6. Scale and regress out total_counts
    7. PCA
    8. Neighborhood graph construction
    9. Leiden clustering
    10. UMAP embedding
    11. Marker gene identification per cluster

    Parameters
    ----------
    adata : AnnData
        Annotated data object with raw counts.
    species : str, default "human"
        Species for mitochondrial gene identification.
    min_genes : int, default 200
        Minimum number of genes per cell for filtering.
    min_cells : int, default 3
        Minimum number of cells per gene for filtering.
    max_pct_mt : float, default 10.0
        Maximum percentage of mitochondrial reads per cell.
    n_top_genes : int, default 2000
        Number of highly variable genes to select.
    n_pcs : int, default 40
        Number of principal components to compute.
    n_neighbors : int, default 10
        Number of neighbors for graph construction.
    resolution : float, default 0.5
        Leiden clustering resolution.
    inplace : bool, default True
        If True, modifies adata in place. If False, returns a copy.

    Returns
    -------
    AnnData or None
        If inplace=False, returns the processed AnnData object.
        If inplace=True, returns None and modifies adata in place.

    Notes
    -----
    The processed adata will contain:
    - adata.obs['leiden']: Cluster assignments
    - adata.obsm['X_pca']: PCA coordinates
    - adata.obsm['X_umap']: UMAP coordinates
    - adata.uns['rank_genes_groups']: Top marker genes per cluster
    """
    if not inplace:
        adata = adata.copy()

    # Step 1: Compute QC metrics
    compute_qc_metrics(adata, species=species, inplace=True)

    # Step 2: Filter cells and genes
    # Keep cells with at least min_genes expressed genes
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # Keep genes expressed in at least min_cells cells
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Filter cells with high mitochondrial content (likely apoptotic/lysed)
    adata._inplace_subset_obs(adata.obs["pct_counts_mt"] < max_pct_mt)

    # Step 3: Normalize to 10,000 counts per cell
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Step 4: Log1p transform
    sc.pp.log1p(adata)

    # Step 5: Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    # Store raw counts for later use
    adata.raw = adata

    # Subset to highly variable genes for clustering
    # Use _inplace_subset_var to modify the original object when inplace=True
    if inplace:
        adata._inplace_subset_var(adata.var.highly_variable)
        adata_subset = adata
    else:
        adata_subset = adata[:, adata.var.highly_variable]

    # Step 6: Regress out unwanted sources of variation and scale
    sc.pp.regress_out(adata_subset, ["total_counts"])
    sc.pp.scale(adata_subset, max_value=10)

    # Step 7: PCA
    sc.tl.pca(adata_subset, svd_solver="arpack", n_comps=n_pcs)

    # Step 8: Compute neighborhood graph
    sc.pp.neighbors(adata_subset, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # Step 9: Leiden clustering
    sc.tl.leiden(adata_subset, resolution=resolution)

    # Step 10: UMAP for visualization
    sc.tl.umap(adata_subset)

    # Step 11: Identify marker genes per cluster
    sc.tl.rank_genes_groups(adata_subset, "leiden", method="t-test")

    if not inplace:
        return adata_subset


def run_qc_pipeline(
    adata: AnnData,
    species: Literal["human", "mouse"] = "human",
    **kwargs,
) -> AnnData:
    """
    Convenience function to run the complete QC pipeline on a fresh dataset.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with raw counts.
    species : str, default "human"
        Species for mitochondrial gene identification.
    **kwargs
        Additional arguments passed to standard_pipeline().

    Returns
    -------
    AnnData
        Processed AnnData object with clustering and QC metrics.
    """
    # Work on a copy to avoid modifying the input
    adata = adata.copy()

    # Run the standard pipeline
    standard_pipeline(adata, species=species, inplace=True, **kwargs)

    return adata


if __name__ == "__main__":
    """Quick test of the QC pipeline."""
    from pathlib import Path
    from .ingest import load_pbmc3k

    print("Loading PBMC3k dataset...")
    project_root = Path(__file__).parent.parent.parent
    adata = load_pbmc3k(cache_dir=project_root / "data")

    print(f"Raw data: {adata.n_obs} cells, {adata.n_vars} genes")

    print("\nRunning QC pipeline...")
    adata_processed = run_qc_pipeline(adata, species="human")

    print(f"\nProcessed data: {adata_processed.n_obs} cells, {adata_processed.n_vars} genes")
    print(f"Clusters found: {adata_processed.obs['leiden'].nunique()}")
    print(f"Cluster distribution:\n{adata_processed.obs['leiden'].value_counts().sort_index()}")

    # Show QC metrics
    print(f"\nQC metrics summary:")
    print(f"  Total counts: {adata_processed.obs['total_counts'].median():.0f} (median)")
    print(
        f"  Detected genes: {adata_processed.obs['n_genes_by_counts'].median():.0f} (median)"
    )
    print(f"  % MT reads: {adata_processed.obs['pct_counts_mt'].median():.2f}% (median)")
