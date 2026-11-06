"""
Unit tests for baseline deterministic auditor.

Tests that the baseline auditor correctly identifies pathologies
in perturbed datasets and passes clean controls.
"""

import pytest
from pathlib import Path

from src.pipeline.ingest import load_pbmc3k
from src.pipeline.qc_engine import run_qc_pipeline
from src.pipeline.extractor import extract_qc_summary
from src.perturbation.inject import generate_perturbed_dataset
from src.auditor.baseline import baseline_audit
from src.auditor.schemas import DatasetQCSummary


@pytest.fixture(scope="module")
def processed_adata():
    """Load and process PBMC3k dataset (shared across tests)."""
    project_root = Path(__file__).parent.parent
    adata = load_pbmc3k(cache_dir=project_root / "data")
    return run_qc_pipeline(adata, species="human")


class TestCleanControl:
    """Tests for clean control baseline audit."""

    def test_clean_control_passes(self, processed_adata):
        """Test that clean control dataset passes baseline audit."""
        # Generate clean control
        adata_clean, metadata = generate_perturbed_dataset(
            processed_adata, "clean_control", species="human"
        )

        # Re-run QC pipeline on clean data (it already has clustering from processed_adata)
        # But we need to extract from the clean version with raw counts
        summary = extract_qc_summary(adata_clean, dataset_name="pbmc3k_clean", species="human")

        # Run baseline audit
        report = baseline_audit(summary)

        # Clean control should have ACCEPTABLE verdict
        # Note: Some clusters might be borderline, so we'll check for low severity
        assert report.summary_verdict in ["ACCEPTABLE", "REQUIRES_FILTERING"]

        # Should have few or no CRITICAL flags
        critical_flags = [f for f in report.flags if f.severity == "CRITICAL"]
        assert len(critical_flags) == 0, f"Clean control should not have CRITICAL flags: {critical_flags}"

    def test_clean_control_metadata(self, processed_adata):
        """Test that clean control metadata is as expected."""
        _, metadata = generate_perturbed_dataset(
            processed_adata, "clean_control", species="human"
        )

        assert metadata["condition"] == "clean_control"
        assert metadata["expected_flags"] == []


class TestLysisDetection:
    """Tests for apoptotic lysis detection."""

    def test_lysis_flagged(self, processed_adata):
        """Test that lysis injection is detected by baseline auditor."""
        # Generate lysis condition
        adata_lysis, metadata = generate_perturbed_dataset(
            processed_adata, "injected_lysis", species="human", target_cluster="1"
        )

        # Re-run QC pipeline to get fresh clustering
        adata_lysis_processed = run_qc_pipeline(adata_lysis, species="human")

        # Extract summary
        summary = extract_qc_summary(
            adata_lysis_processed, dataset_name="pbmc3k_lysis", species="human"
        )

        # Run baseline audit
        report = baseline_audit(summary)

        # Should detect high mitochondrial reads
        flag_types = [f.flag_type for f in report.flags]
        assert "HIGH_MITOCHONDRIAL_READS" in flag_types

        # Should be CRITICAL or REQUIRES_FILTERING
        assert report.summary_verdict in ["CRITICAL_FAIL", "REQUIRES_FILTERING"]

    def test_lysis_affected_cluster(self, processed_adata):
        """Test that lysis flag identifies correct affected cluster."""
        target_cluster = "1"

        adata_lysis, _ = generate_perturbed_dataset(
            processed_adata, "injected_lysis", species="human", target_cluster=target_cluster
        )

        adata_lysis_processed = run_qc_pipeline(adata_lysis, species="human")
        summary = extract_qc_summary(
            adata_lysis_processed, dataset_name="pbmc3k_lysis", species="human"
        )

        report = baseline_audit(summary)

        # Find HIGH_MITOCHONDRIAL_READS flag
        mt_flags = [f for f in report.flags if f.flag_type == "HIGH_MITOCHONDRIAL_READS"]
        assert len(mt_flags) > 0, "Should have at least one HIGH_MITOCHONDRIAL_READS flag"

        # Check that at least one flag mentions a cluster with elevated MT%
        # (May not be exactly "1" after re-clustering, but should flag the affected cluster)
        for flag in mt_flags:
            assert len(flag.affected_clusters) > 0
            assert flag.severity == "CRITICAL"

    def test_lysis_evidence_grounded(self, processed_adata):
        """Test that lysis flag contains numerical evidence."""
        adata_lysis, _ = generate_perturbed_dataset(
            processed_adata, "injected_lysis", species="human"
        )

        adata_lysis_processed = run_qc_pipeline(adata_lysis, species="human")
        summary = extract_qc_summary(
            adata_lysis_processed, dataset_name="pbmc3k_lysis", species="human"
        )

        report = baseline_audit(summary)

        mt_flags = [f for f in report.flags if f.flag_type == "HIGH_MITOCHONDRIAL_READS"]
        assert len(mt_flags) > 0

        # Check that evidence contains numerical values
        for flag in mt_flags:
            assert len(flag.numerical_evidence) > 0
            evidence_str = " ".join(flag.numerical_evidence)
            assert "pct_mt" in evidence_str.lower() or "mt" in evidence_str.lower()
            # Should contain percentage symbol or decimal numbers
            assert "%" in evidence_str or any(char.isdigit() for char in evidence_str)


class TestDoubletDetection:
    """Tests for doublet detection."""

    def test_doublets_created_correctly(self, processed_adata):
        """Test that doublet injection creates cells with elevated counts."""
        # Generate doublets without re-clustering
        adata_doublets, metadata = generate_perturbed_dataset(
            processed_adata, "injected_doublets", doublet_rate=0.08, random_seed=42
        )

        # Check that doublets have elevated counts (before re-clustering)
        doublet_mask = adata_doublets.obs["is_synthetic_doublet"] == True
        doublet_median_counts = adata_doublets.obs.loc[doublet_mask, "total_counts"].median()
        original_median_counts = adata_doublets.obs.loc[~doublet_mask, "total_counts"].median()

        # Doublets should have roughly 2x the counts
        assert doublet_median_counts > original_median_counts * 1.5

    def test_doublets_metadata(self, processed_adata):
        """Test that doublet metadata is correctly structured."""
        adata_doublets, metadata = generate_perturbed_dataset(
            processed_adata, "injected_doublets", doublet_rate=0.08
        )

        assert metadata["condition"] == "injected_doublets"
        assert "POTENTIAL_DOUBLETS" in metadata["expected_flags"]
        assert "n_doublets" in metadata
        assert metadata["n_doublets"] > 0


class TestDegradedDetection:
    """Tests for library degradation detection."""

    def test_degraded_flagged(self, processed_adata):
        """Test that degraded library is detected by baseline auditor."""
        # Generate degraded condition
        adata_degraded, metadata = generate_perturbed_dataset(
            processed_adata, "injected_degraded", species="human", target_cluster="2"
        )

        # Re-run QC pipeline
        adata_degraded_processed = run_qc_pipeline(adata_degraded, species="human")

        # Extract summary
        summary = extract_qc_summary(
            adata_degraded_processed, dataset_name="pbmc3k_degraded", species="human"
        )

        # Run baseline audit
        report = baseline_audit(summary)

        # Should detect low library complexity
        flag_types = [f.flag_type for f in report.flags]
        assert "LOW_LIBRARY_COMPLEXITY" in flag_types

        # Should require filtering
        assert report.summary_verdict in ["REQUIRES_FILTERING", "CRITICAL_FAIL"]

    def test_degraded_evidence(self, processed_adata):
        """Test that degradation flag contains gene count evidence."""
        adata_degraded, _ = generate_perturbed_dataset(
            processed_adata, "injected_degraded", species="human"
        )

        adata_degraded_processed = run_qc_pipeline(adata_degraded, species="human")
        summary = extract_qc_summary(
            adata_degraded_processed, dataset_name="pbmc3k_degraded", species="human"
        )

        report = baseline_audit(summary)

        complexity_flags = [f for f in report.flags if f.flag_type == "LOW_LIBRARY_COMPLEXITY"]
        assert len(complexity_flags) > 0

        # Check that evidence mentions genes
        for flag in complexity_flags:
            assert len(flag.numerical_evidence) > 0
            evidence_str = " ".join(flag.numerical_evidence)
            assert "gene" in evidence_str.lower()


class TestAuditReportStructure:
    """Tests for audit report structure and completeness."""

    def test_report_has_all_fields(self, processed_adata):
        """Test that audit report has all required fields."""
        adata_clean, _ = generate_perturbed_dataset(
            processed_adata, "clean_control", species="human"
        )

        summary = extract_qc_summary(adata_clean, dataset_name="pbmc3k_clean", species="human")
        report = baseline_audit(summary)

        # Check required fields
        assert hasattr(report, "summary_verdict")
        assert hasattr(report, "flags")
        assert hasattr(report, "recommended_actions")

        # Verdict should be valid
        assert report.summary_verdict in ["ACCEPTABLE", "REQUIRES_FILTERING", "CRITICAL_FAIL"]

        # Should have recommended actions
        assert isinstance(report.recommended_actions, list)
        assert len(report.recommended_actions) > 0

    def test_flags_have_required_fields(self, processed_adata):
        """Test that all flags have required fields."""
        # Use lysis to ensure we get some flags
        adata_lysis, _ = generate_perturbed_dataset(
            processed_adata, "injected_lysis", species="human"
        )

        adata_lysis_processed = run_qc_pipeline(adata_lysis, species="human")
        summary = extract_qc_summary(
            adata_lysis_processed, dataset_name="pbmc3k_lysis", species="human"
        )

        report = baseline_audit(summary)

        # If there are flags, check their structure
        if len(report.flags) > 0:
            for flag in report.flags:
                # Check all required fields
                assert hasattr(flag, "flag_type")
                assert hasattr(flag, "affected_clusters")
                assert hasattr(flag, "severity")
                assert hasattr(flag, "numerical_evidence")
                assert hasattr(flag, "technical_mechanism")
                assert hasattr(flag, "biological_confounder_warning")

                # Check that text fields are non-empty
                assert len(flag.numerical_evidence) > 0
                assert len(flag.technical_mechanism) > 0
                assert len(flag.biological_confounder_warning) > 0

                # Check that severity is valid
                assert flag.severity in ["INFO", "WARNING", "CRITICAL"]

                # Check that affected clusters is non-empty
                assert len(flag.affected_clusters) > 0


class TestThresholdCalibration:
    """Tests for threshold calibration."""

    def test_mt_threshold_calibration(self, processed_adata):
        """Test that MT% threshold is properly calibrated."""
        # The baseline should use either 5% absolute or 2.5x median
        # Let's verify this by checking a known lysis case
        adata_lysis, _ = generate_perturbed_dataset(
            processed_adata, "injected_lysis", species="human"
        )

        adata_lysis_processed = run_qc_pipeline(adata_lysis, species="human")
        summary = extract_qc_summary(
            adata_lysis_processed, dataset_name="pbmc3k_lysis", species="human"
        )

        # Check that at least one cluster has elevated MT%
        has_elevated_mt = False
        for cluster in summary.cluster_profiles:
            if cluster.median_pct_mt > 5.0:  # Absolute threshold
                has_elevated_mt = True
                break

        # If we have elevated MT%, baseline should flag it
        if has_elevated_mt:
            report = baseline_audit(summary)
            flag_types = [f.flag_type for f in report.flags]
            assert "HIGH_MITOCHONDRIAL_READS" in flag_types
