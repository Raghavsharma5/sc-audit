"""
Data ingestion module for loading and caching single-cell datasets.

Currently supports the standard PBMC3k dataset from Scanpy.
"""

from pathlib import Path
from typing import Literal, Optional
import scanpy as sc
from anndata import AnnData


def load_pbmc3k(
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> AnnData:
    """
    Load the standard PBMC 3k dataset from 10X Genomics.

    This dataset contains ~2,700 peripheral blood mononuclear cells (PBMCs)
    sequenced on the Illumina NextSeq 500 using 10X Genomics Chromium.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory to cache the downloaded dataset. If None, uses Scanpy's default cache.
    force_download : bool, default False
        If True, re-download the dataset even if cached.

    Returns
    -------
    AnnData
        Annotated data object with gene expression counts and metadata.
        Gene names are deduplicated using .var_names_make_unique()
    """
    # Set cache directory if provided
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.cachedir = cache_dir

    # Download or load from cache
    adata = sc.datasets.pbmc3k()

    # Ensure gene symbols are deduplicated
    adata.var_names_make_unique()

    return adata


def load_dataset(
    dataset_name: Literal["pbmc3k"] = "pbmc3k",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> AnnData:
    """
    Generic dataset loader with support for multiple datasets.

    Parameters
    ----------
    dataset_name : str, default "pbmc3k"
        Name of the dataset to load. Currently only "pbmc3k" is supported.
    cache_dir : Path, optional
        Directory to cache downloaded datasets.
    force_download : bool, default False
        If True, re-download the dataset even if cached.

    Returns
    -------
    AnnData
        Annotated data object with gene expression counts and metadata.

    Raises
    ------
    ValueError
        If dataset_name is not supported.
    """
    if dataset_name == "pbmc3k":
        return load_pbmc3k(cache_dir=cache_dir, force_download=force_download)
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Currently only 'pbmc3k' is supported."
        )


if __name__ == "__main__":
    # Quick test: load and display basic info
    import sys
    from pathlib import Path

    # Use data/ directory in project root
    project_root = Path(__file__).parent.parent.parent
    cache_dir = project_root / "data"

    print(f"Loading PBMC3k dataset...")
    print(f"Cache directory: {cache_dir}")

    adata = load_pbmc3k(cache_dir=cache_dir)

    print(f"\nDataset loaded successfully!")
    print(f"  Cells: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Unique gene names: {adata.var_names.is_unique}")
    print(f"\nFirst 5 genes: {adata.var_names[:5].tolist()}")
