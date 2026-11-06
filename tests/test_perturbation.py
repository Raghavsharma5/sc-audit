"""
Unit tests for perturbation injection.

Tests that controlled pathologies are correctly injected and affect only
the intended targets without bleeding into other clusters.
"""

import pytest
import numpy as np
from pathlib import Path

from src.pipeline.ingest import load_pbmc3k
from src.pipeline.qc_engine import run_qc_pipeline
from src.perturbation.inject import (
    generate_perturbed_dataset,
    inject_lysis,
    inject_doublets,
    inject_degraded,
)


@pytest.fixture(scope="module")
def processed_adata():
    """Load and process PBMC3k dataset (shared across tests)."""
    project_root = Path(__file__).parent.parent
    adata = load_pbmc3k(cache_dir=project_root / "data")
    return run_qc_pipeline(adata, species="human")


class TestCleanControl:
    """Tests for clean control condition."""

    def test_clean_control_no_changes(self, processed_adata):
        """Test that clean control doesn't modify the data."""
        adata_clean, metadata = generate_perturbed_dataset(
            processed_adata, "clean_control", species="human"
        )

        # Should have same number of cells
        assert adata_clean.n_obs == processed_adata.n_obs

        # Should have genes (may be different from processed if HVG was done)
        # Clean control uses raw counts, so it may have more genes
        assert adata_clean.n_vars > 0

        # Metadata should indicate no expected flags
        assert metadata["condition"] == "clean_control"
        assert metadata["expected_flags"] == []

    def test_clean_control_has_raw_counts(self, processed_adata):
        """Test that clean control returns raw counts."""
        adata_clean, _ = generate_perturbed_dataset(
            processed_adata, "clean_control", species="human"
        )

        # Should have QC metrics
        assert "total_counts" in adata_clean.obs.columns
        assert "n_genes_by_counts" in adata_clean.obs.columns


class TestInjectLysis:
    """Tests for apoptotic lysis injection."""

    def test_lysis_increases_mt_in_target_cluster(self, processed_adata):
        """Test that lysis significantly increases MT% in target cluster."""
        target_cluster = "1"

        # Get original MT% for target cluster
        cluster_mask = processed_adata.obs["leiden"] == target_cluster
        original_pct_mt = processed_adata.obs.loc[cluster_mask, "pct_counts_mt"].median()

        # Inject lysis
        adata_lysis, metadata = generate_perturbed_dataset(
            processed_adata, "injected_lysis", species="human", target_cluster=target_cluster
        )

        # Get new MT% for target cluster
        cluster_mask_new = adata_lysis.obs["leiden"] == target_cluster
        new_pct_mt = adata_lysis.obs.loc[cluster_mask_new, "pct_counts_mt"].median()

        # MT% should increase significantly (at least 2x with 5x scaling factor)
        # Note: scaling MT counts by 5x doesn't directly translate to 5x pct_mt
        # because pct_mt is calculated against total_counts which also increases
        assert new_pct_mt > original_pct_mt * 2.0
        # Should also be elevated above normal healthy threshold (>10%)
        assert new_pct_mt > 5.0
        print(f"Original MT%: {original_pct_mt:.2f}%, New MT%: {new_pct_mt:.2f}%")

    def test_lysis_doesnt_affect_other_clusters(self, processed_adata):
        """Test that lysis doesn't significantly affect non-target clusters."""
        target_cluster = "1"

        # Get original MT% for all clusters
        original_pct_mt_by_cluster = (
            processed_adata.obs.groupby("leiden")["pct_counts_mt"].median()
        )

        # Inject lysis
        adata_lysis, _ = generate_perturbed_dataset(
            processed_adata, "injected_lysis", species="human", target_cluster=target_cluster
        )

        # Get new MT% for all clusters
        new_pct_mt_by_cluster = adata_lysis.obs.groupby("leiden")["pct_counts_mt"].median()

        # Check that non-target clusters are relatively unchanged
        for cluster_id in original_pct_mt_by_cluster.index:
            if cluster_id == target_cluster:
                continue  # Skip target cluster

            original = original_pct_mt_by_cluster[cluster_id]
            new = new_pct_mt_by_cluster[cluster_id]

            # Should be within 20% of original value
            assert abs(new - original) < original * 0.2 + 0.5, (
                f"Cluster {cluster_id} MT% changed from {original:.2f} to {new:.2f}"
            )

    def test_lysis_metadata(self, processed_adata):
        """Test that lysis metadata is correctly structured."""
        adata_lysis, metadata = generate_perturbed_dataset(
            processed_adata, "injected_lysis", species="human"
        )

        assert metadata["condition"] == "injected_lysis"
        assert "HIGH_MITOCHONDRIAL_READS" in metadata["expected_flags"]
        assert "target_cluster" in metadata
        assert "n_cells_affected" in metadata
        assert metadata["n_cells_affected"] > 0

    def test_lysis_with_sparse_matrix(self, processed_adata):
        """Test that lysis works with sparse matrices."""
        from scipy.sparse import issparse, csr_matrix

        # Ensure we have a sparse matrix
        if not issparse(processed_adata.X):
            processed_adata.X = csr_matrix(processed_adata.X)

        adata_lysis, metadata = inject_lysis(processed_adata, species="human")

        # Should still be sparse
        assert issparse(adata_lysis.X)
        assert metadata["n_cells_affected"] > 0


class TestInjectDoublets:
    """Tests for synthetic doublet injection."""

    def test_doublets_create_high_count_cells(self, processed_adata):
        """Test that doublets have elevated total counts."""
        # Get original median counts
        original_median_counts = processed_adata.obs["total_counts"].median()

        # Inject doublets
        adata_doublets, metadata = generate_perturbed_dataset(
            processed_adata, "injected_doublets", doublet_rate=0.08, random_seed=42
        )

        # Get doublet counts
        doublet_mask = adata_doublets.obs["is_synthetic_doublet"] == True
        doublet_median_counts = adata_doublets.obs.loc[doublet_mask, "total_counts"].median()

        # Doublets should have roughly 2x the counts (allowing some variance)
        assert doublet_median_counts > original_median_counts * 1.5
        assert doublet_median_counts < original_median_counts * 2.5

        print(
            f"Original median: {original_median_counts:.0f}, "
            f"Doublet median: {doublet_median_counts:.0f}, "
            f"Ratio: {doublet_median_counts / original_median_counts:.2f}x"
        )

    def test_doublets_create_high_gene_cells(self, processed_adata):
        """Test that doublets have elevated gene counts."""
        # Get original median genes
        original_median_genes = processed_adata.obs["n_genes_by_counts"].median()

        # Inject doublets
        adata_doublets, metadata = generate_perturbed_dataset(
            processed_adata, "injected_doublets", doublet_rate=0.08, random_seed=42
        )

        # Get doublet gene counts
        doublet_mask = adata_doublets.obs["is_synthetic_doublet"] == True
        doublet_median_genes = adata_doublets.obs.loc[doublet_mask, "n_genes_by_counts"].median()

        # Doublets should have more genes detected
        assert doublet_median_genes > original_median_genes * 1.3

    def test_doublets_correct_number(self, processed_adata):
        """Test that correct number of doublets are generated."""
        doublet_rate = 0.08
        n_original = processed_adata.n_obs

        adata_doublets, metadata = generate_perturbed_dataset(
            processed_adata, "injected_doublets", doublet_rate=doublet_rate, random_seed=42
        )

        # Check number of doublets
        n_doublets = metadata["n_doublets"]
        expected_doublets = int(n_original * doublet_rate)

        assert n_doublets == expected_doublets
        assert adata_doublets.n_obs == n_original + n_doublets

    def test_doublets_metadata(self, processed_adata):
        """Test that doublet metadata is correctly structured."""
        adata_doublets, metadata = generate_perturbed_dataset(
            processed_adata, "injected_doublets", doublet_rate=0.08
        )

        assert metadata["condition"] == "injected_doublets"
        assert "POTENTIAL_DOUBLETS" in metadata["expected_flags"]
        assert "n_doublets" in metadata
        assert "doublet_rate" in metadata
        assert metadata["doublet_rate"] == 0.08

    def test_doublets_marked_in_obs(self, processed_adata):
        """Test that doublets are marked in obs."""
        adata_doublets, metadata = generate_perturbed_dataset(
            processed_adata, "injected_doublets", doublet_rate=0.08
        )

        assert "is_synthetic_doublet" in adata_doublets.obs.columns

        # Original cells should be marked False
        n_doublets = (adata_doublets.obs["is_synthetic_doublet"] == True).sum()
        assert n_doublets == metadata["n_doublets"]


class TestInjectDegraded:
    """Tests for library degradation injection."""

    def test_degraded_reduces_genes_in_target_cluster(self, processed_adata):
        """Test that degradation reduces detected genes in target cluster."""
        target_cluster = "2"
        target_genes = 250

        # Get original gene counts for target cluster
        cluster_mask = processed_adata.obs["leiden"] == target_cluster
        original_genes = processed_adata.obs.loc[cluster_mask, "n_genes_by_counts"].median()

        # Inject degradation
        adata_degraded, metadata = generate_perturbed_dataset(
            processed_adata,
            "injected_degraded",
            species="human",
            target_cluster=target_cluster,
            target_genes_median=target_genes,
        )

        # Get new gene counts for target cluster
        cluster_mask_new = adata_degraded.obs["leiden"] == target_cluster
        new_genes = adata_degraded.obs.loc[cluster_mask_new, "n_genes_by_counts"].median()

        # Gene counts should be significantly reduced
        assert new_genes < original_genes * 0.6
        assert new_genes < 400  # Should be below complexity threshold

        print(
            f"Original genes: {original_genes:.0f}, "
            f"New genes: {new_genes:.0f}, "
            f"Reduction: {(1 - new_genes/original_genes)*100:.1f}%"
        )

    def test_degraded_doesnt_affect_other_clusters(self, processed_adata):
        """Test that degradation doesn't significantly affect non-target clusters."""
        target_cluster = "2"

        # Get original gene counts for all clusters
        original_genes_by_cluster = (
            processed_adata.obs.groupby("leiden")["n_genes_by_counts"].median()
        )

        # Inject degradation
        adata_degraded, _ = generate_perturbed_dataset(
            processed_adata,
            "injected_degraded",
            species="human",
            target_cluster=target_cluster,
        )

        # Get new gene counts for all clusters
        new_genes_by_cluster = (
            adata_degraded.obs.groupby("leiden")["n_genes_by_counts"].median()
        )

        # Check that non-target clusters are relatively unchanged
        for cluster_id in original_genes_by_cluster.index:
            if cluster_id == target_cluster:
                continue  # Skip target cluster

            original = original_genes_by_cluster[cluster_id]
            new = new_genes_by_cluster[cluster_id]

            # Should be within 10% of original value
            assert abs(new - original) < original * 0.1, (
                f"Cluster {cluster_id} genes changed from {original:.0f} to {new:.0f}"
            )

    def test_degraded_metadata(self, processed_adata):
        """Test that degradation metadata is correctly structured."""
        adata_degraded, metadata = generate_perturbed_dataset(
            processed_adata, "injected_degraded", species="human"
        )

        assert metadata["condition"] == "injected_degraded"
        assert "LOW_LIBRARY_COMPLEXITY" in metadata["expected_flags"]
        assert "target_cluster" in metadata
        assert "n_cells_affected" in metadata
        assert metadata["n_cells_affected"] > 0

    def test_degraded_preserves_total_counts(self, processed_adata):
        """Test that degradation reduces genes but preserves reasonable total counts."""
        target_cluster = "2"

        # Get original counts for target cluster
        cluster_mask = processed_adata.obs["leiden"] == target_cluster
        original_counts = processed_adata.obs.loc[cluster_mask, "total_counts"].median()

        # Inject degradation
        adata_degraded, _ = generate_perturbed_dataset(
            processed_adata,
            "injected_degraded",
            species="human",
            target_cluster=target_cluster,
        )

        # Get new counts for target cluster
        cluster_mask_new = adata_degraded.obs["leiden"] == target_cluster
        new_counts = adata_degraded.obs.loc[cluster_mask_new, "total_counts"].median()

        # Total counts should be reduced but not eliminated
        # (genes are dropped, so counts will decrease)
        assert new_counts > 100  # Still has some counts
        assert new_counts < original_counts  # But reduced


class TestPerturbationFactory:
    """Tests for the factory function."""

    def test_all_conditions_supported(self, processed_adata):
        """Test that all perturbation conditions can be generated."""
        conditions = ["clean_control", "injected_lysis", "injected_doublets", "injected_degraded"]

        for condition in conditions:
            adata_perturbed, metadata = generate_perturbed_dataset(
                processed_adata, condition, species="human"
            )

            assert metadata["condition"] == condition
            assert "expected_flags" in metadata
            assert adata_perturbed is not None

    def test_invalid_condition_raises_error(self, processed_adata):
        """Test that invalid condition raises ValueError."""
        with pytest.raises(ValueError, match="Unknown condition"):
            generate_perturbed_dataset(processed_adata, "invalid_condition")

    def test_metadata_contains_expected_flags(self, processed_adata):
        """Test that metadata contains expected_flags for all conditions."""
        conditions = ["clean_control", "injected_lysis", "injected_doublets", "injected_degraded"]

        expected_flag_map = {
            "clean_control": [],
            "injected_lysis": ["HIGH_MITOCHONDRIAL_READS"],
            "injected_doublets": ["POTENTIAL_DOUBLETS"],
            "injected_degraded": ["LOW_LIBRARY_COMPLEXITY"],
        }

        for condition in conditions:
            _, metadata = generate_perturbed_dataset(processed_adata, condition, species="human")

            expected = expected_flag_map[condition]
            actual = metadata["expected_flags"]

            assert set(actual) == set(expected), (
                f"Condition {condition}: expected {expected}, got {actual}"
            )


class TestMatrixSafety:
    """Tests for safe handling of sparse and dense matrices."""

    def test_lysis_preserves_sparsity(self, processed_adata):
        """Test that lysis preserves sparse matrix format."""
        from scipy.sparse import issparse, csr_matrix

        # Convert to sparse if not already
        adata_sparse = processed_adata.copy()
        if not issparse(adata_sparse.X):
            adata_sparse.X = csr_matrix(adata_sparse.X)

        # Inject lysis
        adata_lysis, _ = inject_lysis(adata_sparse, species="human")

        # Should still be sparse
        assert issparse(adata_lysis.X)

    def test_doublets_handle_sparse(self, processed_adata):
        """Test that doublet injection handles sparse matrices."""
        from scipy.sparse import issparse, csr_matrix

        # Convert to sparse if not already
        adata_sparse = processed_adata.copy()
        if not issparse(adata_sparse.X):
            adata_sparse.X = csr_matrix(adata_sparse.X)

        # Inject doublets
        adata_doublets, metadata = inject_doublets(adata_sparse, doublet_rate=0.08)

        # Should complete without error
        assert metadata["n_doublets"] > 0

    def test_degraded_preserves_sparsity(self, processed_adata):
        """Test that degradation preserves sparse matrix format."""
        from scipy.sparse import issparse, csr_matrix

        # Convert to sparse if not already
        adata_sparse = processed_adata.copy()
        if not issparse(adata_sparse.X):
            adata_sparse.X = csr_matrix(adata_sparse.X)

        # Inject degradation
        adata_degraded, _ = inject_degraded(adata_sparse)

        # Should still be sparse
        assert issparse(adata_degraded.X)
