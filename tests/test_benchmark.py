"""
Integration tests for benchmark evaluation system.

Tests metrics calculation, benchmark orchestration, and report generation.
"""

import pytest
from pathlib import Path

from src.auditor.schemas import AuditReport, QCFlag, DatasetQCSummary, DistributionStats, ClusterQCProfile
from eval.metrics import (
    calculate_flag_metrics,
    calculate_groundedness,
    calculate_speculation_score,
    extract_numeric_values_from_summary,
)


class TestFlagMetrics:
    """Tests for flag precision/recall/F1 calculation."""

    def test_perfect_detection(self):
        """Test metrics when all expected flags are detected correctly."""
        detected = [
            QCFlag(
                flag_type="HIGH_MITOCHONDRIAL_READS",
                affected_clusters=["1"],
                severity="CRITICAL",
                numerical_evidence=["test"],
                technical_mechanism="test",
                biological_confounder_warning="test",
            )
        ]
        expected = ["HIGH_MITOCHONDRIAL_READS"]

        precision, recall, f1 = calculate_flag_metrics(detected, expected)

        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_partial_detection(self):
        """Test metrics when only some expected flags are detected."""
        detected = [
            QCFlag(
                flag_type="HIGH_MITOCHONDRIAL_READS",
                affected_clusters=["1"],
                severity="CRITICAL",
                numerical_evidence=["test"],
                technical_mechanism="test",
                biological_confounder_warning="test",
            )
        ]
        expected = ["HIGH_MITOCHONDRIAL_READS", "POTENTIAL_DOUBLETS"]

        precision, recall, f1 = calculate_flag_metrics(detected, expected)

        assert precision == 1.0  # No false positives
        assert recall == 0.5  # Detected 1 out of 2 expected
        assert f1 == pytest.approx(2 * (1.0 * 0.5) / (1.0 + 0.5))

    def test_false_positives(self):
        """Test metrics when auditor detects unexpected flags."""
        detected = [
            QCFlag(
                flag_type="HIGH_MITOCHONDRIAL_READS",
                affected_clusters=["1"],
                severity="CRITICAL",
                numerical_evidence=["test"],
                technical_mechanism="test",
                biological_confounder_warning="test",
            ),
            QCFlag(
                flag_type="POTENTIAL_DOUBLETS",
                affected_clusters=["2"],
                severity="WARNING",
                numerical_evidence=["test"],
                technical_mechanism="test",
                biological_confounder_warning="test",
            ),
        ]
        expected = ["HIGH_MITOCHONDRIAL_READS"]

        precision, recall, f1 = calculate_flag_metrics(detected, expected)

        assert precision == 0.5  # 1 true positive, 1 false positive
        assert recall == 1.0  # All expected flags detected
        assert f1 == pytest.approx(2 * (0.5 * 1.0) / (0.5 + 1.0))

    def test_no_flags_detected_or_expected(self):
        """Test metrics when clean data has no flags."""
        detected = []
        expected = []

        precision, recall, f1 = calculate_flag_metrics(detected, expected)

        # Perfect score when no flags expected and none detected
        assert precision == 0.0  # No TP, no FP
        assert recall == 1.0  # Vacuously true
        assert f1 == 0.0


class TestGroundedness:
    """Tests for groundedness ratio calculation."""

    def test_perfectly_grounded_evidence(self):
        """Test that exact numerical matches yield 100% groundedness."""
        # Create mock summary
        summary = DatasetQCSummary(
            dataset_name="test",
            total_cells=100,
            total_genes=500,
            species="human",
            total_counts_distribution=DistributionStats(
                min=100.0, p5=200.0, p25=300.0, p50=400.0, p75=500.0, p95=600.0, max=700.0, mean=400.0, std=100.0
            ),
            detected_genes_distribution=DistributionStats(
                min=50.0, p5=100.0, p25=150.0, p50=200.0, p75=250.0, p95=300.0, max=350.0, mean=200.0, std=50.0
            ),
            mito_pct_distribution=DistributionStats(
                min=1.0, p5=2.0, p25=3.0, p50=4.0, p75=5.0, p95=6.0, max=7.0, mean=4.0, std=1.5
            ),
            cluster_profiles=[
                ClusterQCProfile(
                    cluster_id="1",
                    cell_count=50,
                    pct_of_total=50.0,
                    median_total_counts=400.0,
                    median_genes_detected=200.0,
                    median_pct_mt=4.0,
                    top_marker_genes=["GENE1", "GENE2"],
                )
            ],
        )

        # Create report citing exact values
        report = AuditReport(
            summary_verdict="ACCEPTABLE",
            flags=[
                QCFlag(
                    flag_type="HIGH_MITOCHONDRIAL_READS",
                    affected_clusters=["1"],
                    severity="CRITICAL",
                    numerical_evidence=[
                        "Cluster 1: median pct_mt = 4.0%",
                        "Dataset median = 400.0",
                    ],
                    technical_mechanism="test",
                    biological_confounder_warning="test",
                )
            ],
            recommended_actions=["test"],
        )

        groundedness = calculate_groundedness(report, summary)
        assert groundedness == 1.0  # All cited numbers match

    def test_ungrounded_evidence(self):
        """Test that fabricated numbers yield low groundedness."""
        summary = DatasetQCSummary(
            dataset_name="test",
            total_cells=100,
            total_genes=500,
            species="human",
            total_counts_distribution=DistributionStats(
                min=100.0, p5=200.0, p25=300.0, p50=400.0, p75=500.0, p95=600.0, max=700.0, mean=400.0, std=100.0
            ),
            detected_genes_distribution=DistributionStats(
                min=50.0, p5=100.0, p25=150.0, p50=200.0, p75=250.0, p95=300.0, max=350.0, mean=200.0, std=50.0
            ),
            mito_pct_distribution=DistributionStats(
                min=1.0, p5=2.0, p25=3.0, p50=4.0, p75=5.0, p95=6.0, max=7.0, mean=4.0, std=1.5
            ),
            cluster_profiles=[],
        )

        # Create report with fabricated numbers (using unique values not in reference)
        report = AuditReport(
            summary_verdict="ACCEPTABLE",
            flags=[
                QCFlag(
                    flag_type="HIGH_MITOCHONDRIAL_READS",
                    affected_clusters=["1"],
                    severity="CRITICAL",
                    numerical_evidence=[
                        "Cluster: median pct_mt = 87.65%",  # Fabricated
                        "Dataset median = 9876.54",  # Fabricated
                    ],
                    technical_mechanism="test",
                    biological_confounder_warning="test",
                )
            ],
            recommended_actions=["test"],
        )

        groundedness = calculate_groundedness(report, summary)
        assert groundedness == 0.0  # No cited numbers match

    def test_no_numerical_evidence(self):
        """Test that reports with no numbers are vacuously grounded."""
        summary = DatasetQCSummary(
            dataset_name="test",
            total_cells=100,
            total_genes=500,
            species="human",
            total_counts_distribution=DistributionStats(
                min=100.0, p5=200.0, p25=300.0, p50=400.0, p75=500.0, p95=600.0, max=700.0, mean=400.0, std=100.0
            ),
            detected_genes_distribution=DistributionStats(
                min=50.0, p5=100.0, p25=150.0, p50=200.0, p75=250.0, p95=300.0, max=350.0, mean=200.0, std=50.0
            ),
            mito_pct_distribution=DistributionStats(
                min=1.0, p5=2.0, p25=3.0, p50=4.0, p75=5.0, p95=6.0, max=7.0, mean=4.0, std=1.5
            ),
            cluster_profiles=[],
        )

        report = AuditReport(
            summary_verdict="ACCEPTABLE",
            flags=[],
            recommended_actions=["Everything looks good"],
        )

        groundedness = calculate_groundedness(report, summary)
        assert groundedness == 1.0  # Vacuously true


class TestSpeculationScore:
    """Tests for speculation score (biological confounder warnings)."""

    def test_proper_warnings_on_perturbed_data(self):
        """Test that proper warnings yield PASS on perturbed data."""
        report = AuditReport(
            summary_verdict="CRITICAL_FAIL",
            flags=[
                QCFlag(
                    flag_type="HIGH_MITOCHONDRIAL_READS",
                    affected_clusters=["1"],
                    severity="CRITICAL",
                    numerical_evidence=["test"],
                    technical_mechanism="Membrane lysis during dissociation",
                    biological_confounder_warning="This must NOT be interpreted as a metabolically active cell type",
                )
            ],
            recommended_actions=["Filter cluster"],
        )

        score = calculate_speculation_score(report, "injected_lysis")
        assert score == 1  # PASS

    def test_missing_warnings_on_perturbed_data(self):
        """Test that missing warnings yield FAIL on perturbed data."""
        report = AuditReport(
            summary_verdict="CRITICAL_FAIL",
            flags=[
                QCFlag(
                    flag_type="HIGH_MITOCHONDRIAL_READS",
                    affected_clusters=["1"],
                    severity="CRITICAL",
                    numerical_evidence=["test"],
                    technical_mechanism="test",
                    biological_confounder_warning="",  # Missing warning
                )
            ],
            recommended_actions=["test"],
        )

        score = calculate_speculation_score(report, "injected_lysis")
        assert score == 0  # FAIL

    def test_biological_speculation_without_disclaimer(self):
        """Test that biological claims without disclaimers yield FAIL."""
        report = AuditReport(
            summary_verdict="ACCEPTABLE",
            flags=[
                QCFlag(
                    flag_type="HIGH_MITOCHONDRIAL_READS",
                    affected_clusters=["1"],
                    severity="CRITICAL",
                    numerical_evidence=["test"],
                    technical_mechanism="test",
                    biological_confounder_warning="This represents a novel cell type with unique metabolic properties",
                    # Speculation without disclaimer
                )
            ],
            recommended_actions=["test"],
        )

        score = calculate_speculation_score(report, "injected_lysis")
        assert score == 0  # FAIL - biological claim without negation

    def test_clean_control_no_critical_flags(self):
        """Test that clean control with no CRITICAL flags passes."""
        report = AuditReport(
            summary_verdict="ACCEPTABLE",
            flags=[],
            recommended_actions=["Dataset passes QC"],
        )

        score = calculate_speculation_score(report, "clean_control")
        assert score == 1  # PASS

    def test_clean_control_with_critical_flags(self):
        """Test that clean control with CRITICAL flags fails."""
        report = AuditReport(
            summary_verdict="CRITICAL_FAIL",
            flags=[
                QCFlag(
                    flag_type="HIGH_MITOCHONDRIAL_READS",
                    affected_clusters=["1"],
                    severity="CRITICAL",
                    numerical_evidence=["test"],
                    technical_mechanism="test",
                    biological_confounder_warning="test",
                )
            ],
            recommended_actions=["Filter"],
        )

        score = calculate_speculation_score(report, "clean_control")
        assert score == 0  # FAIL - false positive on clean data


class TestNumericExtraction:
    """Tests for numeric value extraction from summaries."""

    def test_extract_all_numeric_values(self):
        """Test that all numeric values are extracted from summary."""
        summary = DatasetQCSummary(
            dataset_name="test",
            total_cells=100,
            total_genes=500,
            species="human",
            total_counts_distribution=DistributionStats(
                min=100.0, p5=200.0, p25=300.0, p50=400.0, p75=500.0, p95=600.0, max=700.0, mean=400.0, std=100.0
            ),
            detected_genes_distribution=DistributionStats(
                min=50.0, p5=100.0, p25=150.0, p50=200.0, p75=250.0, p95=300.0, max=350.0, mean=200.0, std=50.0
            ),
            mito_pct_distribution=DistributionStats(
                min=1.0, p5=2.0, p25=3.0, p50=4.0, p75=5.0, p95=6.0, max=7.0, mean=4.0, std=1.5
            ),
            cluster_profiles=[
                ClusterQCProfile(
                    cluster_id="1",
                    cell_count=50,
                    pct_of_total=50.0,
                    median_total_counts=400.0,
                    median_genes_detected=200.0,
                    median_pct_mt=4.0,
                    top_marker_genes=["GENE1"],
                )
            ],
        )

        values = extract_numeric_values_from_summary(summary)

        # Should include total_cells, total_genes, all distribution stats, and cluster stats
        assert 100.0 in values  # total_cells and min from distribution
        assert 500.0 in values  # total_genes and p75 from distribution
        assert 400.0 in values  # p50 and mean
        assert 4.0 in values  # median_pct_mt
        assert 50.0 in values  # cell_count and pct_of_total

        # Should have extracted many unique values
        # 3 distributions × 9 stats each = 27, but with overlap (e.g., 100.0, 200.0, 50.0, etc.)
        # Total unique values will be less due to duplicates across distributions
        assert len(values) >= 15  # Reasonable lower bound for unique values
