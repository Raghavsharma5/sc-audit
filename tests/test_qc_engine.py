"""
Unit tests for QC engine and extractor.

Tests QC metric computation, clustering, and serialization to schemas.
"""

import pytest
import numpy as np
from pathlib import Path

from src.pipeline.ingest import load_pbmc3k
from src.pipeline.qc_engine import compute_qc_metrics, run_qc_pipeline
from src.pipeline.extractor import (
    compute_distribution_stats,
    extract_qc_summary,
    extract_cluster_profile,
)
from src.auditor.schemas import DatasetQCSummary


@pytest.fixture(scope="module")
def raw_adata():
    """Load raw PBMC3k dataset (shared across tests)."""
    project_root = Path(__file__).parent.parent
    return load_pbmc3k(cache_dir=project_root / "data")


@pytest.fixture(scope="module")
def processed_adata(raw_adata):
    """Run QC pipeline on PBMC3k (shared across tests)."""
    return run_qc_pipeline(raw_adata.copy(), species="human")


class TestDataIngestion:
    """Tests for data loading."""

    def test_load_pbmc3k(self, raw_adata):
        """Test that PBMC3k dataset loads correctly."""
        assert raw_adata is not None
        assert raw_adata.n_obs > 0
        assert raw_adata.n_vars > 0

    def test_gene_names_unique(self, raw_adata):
        """Test that gene names are deduplicated."""
        assert raw_adata.var_names.is_unique


class TestQCMetrics:
    """Tests for QC metric computation."""

    def test_compute_qc_metrics(self, raw_adata):
        """Test that QC metrics are computed correctly."""
        adata = raw_adata.copy()
        compute_qc_metrics(adata, species="human", inplace=True)

        # Check that required columns were added
        assert "pct_counts_mt" in adata.obs.columns
        assert "total_counts" in adata.obs.columns
        assert "n_genes_by_counts" in adata.obs.columns

        # Check that mitochondrial genes were identified
        assert "mt" in adata.var.columns
        assert adata.var["mt"].sum() > 0  # Should have some MT genes

    def test_qc_metrics_non_negative(self, processed_adata):
        """Test that all QC metrics are non-negative."""
        assert (processed_adata.obs["total_counts"] >= 0).all()
        assert (processed_adata.obs["n_genes_by_counts"] >= 0).all()
        assert (processed_adata.obs["pct_counts_mt"] >= 0).all()

    def test_pct_mt_bounded(self, processed_adata):
        """Test that percent mitochondrial is between 0 and 100."""
        # Should be filtered to < 10% by default
        assert (processed_adata.obs["pct_counts_mt"] < 10.0).all()
        assert (processed_adata.obs["pct_counts_mt"] >= 0).all()


class TestClustering:
    """Tests for clustering results."""

    def test_leiden_clustering(self, processed_adata):
        """Test that Leiden clustering was performed."""
        assert "leiden" in processed_adata.obs.columns
        assert processed_adata.obs["leiden"].nunique() > 1  # Should have multiple clusters

    def test_all_cells_clustered(self, processed_adata):
        """Test that all cells are assigned to a cluster."""
        assert processed_adata.obs["leiden"].notna().all()

    def test_marker_genes_computed(self, processed_adata):
        """Test that marker genes were computed."""
        assert "rank_genes_groups" in processed_adata.uns
        assert "names" in processed_adata.uns["rank_genes_groups"]

    def test_pca_and_umap(self, processed_adata):
        """Test that dimensionality reduction was performed."""
        assert "X_pca" in processed_adata.obsm
        assert "X_umap" in processed_adata.obsm


class TestDistributionStats:
    """Tests for distribution statistics computation."""

    def test_percentiles_monotonic(self, processed_adata):
        """Test that percentiles are monotonically non-decreasing."""
        # Test on total_counts
        stats = compute_distribution_stats(processed_adata.obs["total_counts"].values)

        # Check monotonicity: min <= p5 <= p25 <= p50 <= p75 <= p95 <= max
        assert stats.min <= stats.p5
        assert stats.p5 <= stats.p25
        assert stats.p25 <= stats.p50
        assert stats.p50 <= stats.p75
        assert stats.p75 <= stats.p95
        assert stats.p95 <= stats.max

    def test_mean_and_std(self, processed_adata):
        """Test that mean and std are computed correctly."""
        values = processed_adata.obs["total_counts"].values
        stats = compute_distribution_stats(values)

        # Compare with numpy
        assert np.isclose(stats.mean, np.mean(values), rtol=1e-5)
        assert np.isclose(stats.std, np.std(values), rtol=1e-5)

    def test_min_max_bounds(self, processed_adata):
        """Test that min and max are actual bounds."""
        values = processed_adata.obs["n_genes_by_counts"].values
        stats = compute_distribution_stats(values)

        assert stats.min == np.min(values)
        assert stats.max == np.max(values)


class TestClusterProfiles:
    """Tests for cluster profile extraction."""

    def test_cluster_profile_extraction(self, processed_adata):
        """Test that cluster profiles can be extracted."""
        cluster_id = str(processed_adata.obs["leiden"].iloc[0])
        profile = extract_cluster_profile(
            processed_adata,
            cluster_id,
            total_cells=processed_adata.n_obs,
        )

        assert profile.cluster_id == cluster_id
        assert profile.cell_count > 0
        assert 0 < profile.pct_of_total <= 100
        assert profile.median_total_counts > 0
        assert profile.median_genes_detected > 0
        assert profile.median_pct_mt >= 0

    def test_top_marker_genes(self, processed_adata):
        """Test that top marker genes are extracted."""
        cluster_id = str(processed_adata.obs["leiden"].iloc[0])
        profile = extract_cluster_profile(
            processed_adata,
            cluster_id,
            total_cells=processed_adata.n_obs,
        )

        # Should have marker genes (up to 5)
        assert len(profile.top_marker_genes) <= 5
        assert len(profile.top_marker_genes) > 0

        # All marker genes should be strings
        assert all(isinstance(gene, str) for gene in profile.top_marker_genes)

    def test_cluster_percentages_sum(self, processed_adata):
        """Test that cluster percentages approximately sum to 100%."""
        cluster_ids = processed_adata.obs["leiden"].unique()
        profiles = [
            extract_cluster_profile(
                processed_adata,
                str(cid),
                total_cells=processed_adata.n_obs,
            )
            for cid in cluster_ids
        ]

        total_pct = sum(p.pct_of_total for p in profiles)
        assert np.isclose(total_pct, 100.0, rtol=1e-5)


class TestQCSummaryExtraction:
    """Tests for complete QC summary extraction."""

    def test_extract_qc_summary(self, processed_adata):
        """Test that complete QC summary can be extracted."""
        summary = extract_qc_summary(processed_adata, dataset_name="pbmc3k", species="human")

        assert isinstance(summary, DatasetQCSummary)
        assert summary.dataset_name == "pbmc3k"
        assert summary.species == "human"
        assert summary.total_cells > 0
        assert summary.total_genes > 0

    def test_summary_has_all_distributions(self, processed_adata):
        """Test that all distribution stats are present."""
        summary = extract_qc_summary(processed_adata, dataset_name="pbmc3k", species="human")

        # Check all distributions exist
        assert summary.total_counts_distribution is not None
        assert summary.detected_genes_distribution is not None
        assert summary.mito_pct_distribution is not None

        # Check monotonicity for each distribution
        for dist_name in ["total_counts", "detected_genes", "mito_pct"]:
            dist = getattr(summary, f"{dist_name}_distribution")
            assert dist.min <= dist.p5 <= dist.p25 <= dist.p50 <= dist.p75 <= dist.p95 <= dist.max

    def test_summary_has_clusters(self, processed_adata):
        """Test that cluster profiles are extracted."""
        summary = extract_qc_summary(processed_adata, dataset_name="pbmc3k", species="human")

        # Should have at least one cluster
        assert len(summary.cluster_profiles) > 0

        # All clusters should have valid data
        for profile in summary.cluster_profiles:
            assert profile.cell_count > 0
            assert profile.pct_of_total > 0
            assert len(profile.top_marker_genes) <= 5

    def test_summary_serialization(self, processed_adata):
        """Test that summary can be serialized to JSON."""
        summary = extract_qc_summary(processed_adata, dataset_name="pbmc3k", species="human")

        # Serialize to dict
        data_dict = summary.model_dump()
        assert isinstance(data_dict, dict)
        assert "dataset_name" in data_dict
        assert "cluster_profiles" in data_dict

        # Serialize to JSON string
        json_str = summary.model_dump_json()
        assert isinstance(json_str, str)
        assert "pbmc3k" in json_str

        # Deserialize and verify
        reconstructed = DatasetQCSummary.model_validate_json(json_str)
        assert reconstructed.dataset_name == summary.dataset_name
        assert reconstructed.total_cells == summary.total_cells

    def test_missing_qc_metrics_raises_error(self, raw_adata):
        """Test that extraction fails if QC metrics are missing."""
        # Try to extract from raw data without running QC pipeline
        with pytest.raises(ValueError, match="Missing required QC metrics"):
            extract_qc_summary(raw_adata, dataset_name="test", species="human")


class TestQCPipelineIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from loading to extraction."""
        # Load raw data
        project_root = Path(__file__).parent.parent
        adata = load_pbmc3k(cache_dir=project_root / "data")

        # Run QC pipeline
        adata_processed = run_qc_pipeline(adata, species="human")

        # Extract summary
        summary = extract_qc_summary(adata_processed, dataset_name="pbmc3k", species="human")

        # Verify complete summary
        assert summary.total_cells == adata_processed.n_obs
        assert summary.total_genes == adata_processed.n_vars
        assert len(summary.cluster_profiles) == adata_processed.obs["leiden"].nunique()

        # Verify all percentiles are monotonic for all distributions
        for dist_attr in [
            "total_counts_distribution",
            "detected_genes_distribution",
            "mito_pct_distribution",
        ]:
            dist = getattr(summary, dist_attr)
            assert dist.min <= dist.p5 <= dist.p25 <= dist.p50 <= dist.p75 <= dist.p95 <= dist.max

    def test_pipeline_reproducibility(self, raw_adata):
        """Test that pipeline produces consistent results."""
        # Run pipeline twice on copies
        adata1 = run_qc_pipeline(raw_adata.copy(), species="human", resolution=0.5)
        adata2 = run_qc_pipeline(raw_adata.copy(), species="human", resolution=0.5)

        # Should produce same number of cells (after filtering)
        assert adata1.n_obs == adata2.n_obs
        assert adata1.n_vars == adata2.n_vars

        # Should have same QC metric distributions
        assert np.isclose(
            adata1.obs["total_counts"].median(),
            adata2.obs["total_counts"].median(),
            rtol=1e-5,
        )
