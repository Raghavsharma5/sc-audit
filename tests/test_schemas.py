"""
Unit tests for Pydantic schemas.

Tests serialization, deserialization, and validation of all schema models.
"""

import pytest
from pydantic import ValidationError

from src.auditor.schemas import (
    DistributionStats,
    ClusterQCProfile,
    DatasetQCSummary,
    QCFlag,
    AuditReport,
)


class TestDistributionStats:
    """Tests for DistributionStats schema."""

    def test_valid_distribution(self):
        """Test creation with valid data."""
        stats = DistributionStats(
            min=100.0,
            p5=150.0,
            p25=200.0,
            p50=250.0,
            p75=300.0,
            p95=400.0,
            max=500.0,
            mean=260.0,
            std=80.0,
        )
        assert stats.min == 100.0
        assert stats.p50 == 250.0
        assert stats.max == 500.0

    def test_serialization_roundtrip(self):
        """Test that serialization and deserialization preserve data."""
        original = DistributionStats(
            min=10.0,
            p5=15.0,
            p25=20.0,
            p50=25.0,
            p75=30.0,
            p95=40.0,
            max=50.0,
            mean=26.0,
            std=8.0,
        )

        # Serialize to dict
        data_dict = original.model_dump()

        # Deserialize from dict
        reconstructed = DistributionStats.model_validate(data_dict)

        # Verify all fields match
        assert reconstructed == original

    def test_json_serialization(self):
        """Test JSON serialization."""
        stats = DistributionStats(
            min=1.0,
            p5=2.0,
            p25=3.0,
            p50=4.0,
            p75=5.0,
            p95=6.0,
            max=7.0,
            mean=4.2,
            std=1.5,
        )

        # Serialize to JSON
        json_str = stats.model_dump_json()

        # Deserialize from JSON
        reconstructed = DistributionStats.model_validate_json(json_str)

        assert reconstructed == stats


class TestClusterQCProfile:
    """Tests for ClusterQCProfile schema."""

    def test_valid_cluster_profile(self):
        """Test creation with valid data."""
        profile = ClusterQCProfile(
            cluster_id="0",
            cell_count=500,
            pct_of_total=25.0,
            median_total_counts=2500.0,
            median_genes_detected=800.0,
            median_pct_mt=2.5,
            top_marker_genes=["IL7R", "CCR7", "CD3D", "LTB", "LDHB"],
        )
        assert profile.cluster_id == "0"
        assert profile.cell_count == 500
        assert len(profile.top_marker_genes) == 5

    def test_max_length_marker_genes(self):
        """Test that top_marker_genes is limited to 5 genes."""
        # Valid: exactly 5 genes
        profile = ClusterQCProfile(
            cluster_id="1",
            cell_count=100,
            pct_of_total=10.0,
            median_total_counts=1000.0,
            median_genes_detected=500.0,
            median_pct_mt=3.0,
            top_marker_genes=["A", "B", "C", "D", "E"],
        )
        assert len(profile.top_marker_genes) == 5

        # Valid: fewer than 5 genes
        profile = ClusterQCProfile(
            cluster_id="2",
            cell_count=50,
            pct_of_total=5.0,
            median_total_counts=900.0,
            median_genes_detected=450.0,
            median_pct_mt=2.0,
            top_marker_genes=["X", "Y", "Z"],
        )
        assert len(profile.top_marker_genes) == 3

        # Invalid: more than 5 genes should fail
        with pytest.raises(ValidationError):
            ClusterQCProfile(
                cluster_id="3",
                cell_count=200,
                pct_of_total=20.0,
                median_total_counts=2000.0,
                median_genes_detected=700.0,
                median_pct_mt=4.0,
                top_marker_genes=["A", "B", "C", "D", "E", "F"],
            )


class TestDatasetQCSummary:
    """Tests for DatasetQCSummary schema."""

    def test_valid_summary(self):
        """Test creation with valid complete dataset summary."""
        summary = DatasetQCSummary(
            dataset_name="pbmc3k",
            total_cells=2000,
            total_genes=15000,
            species="human",
            total_counts_distribution=DistributionStats(
                min=500.0,
                p5=1000.0,
                p25=1500.0,
                p50=2000.0,
                p75=2500.0,
                p95=3500.0,
                max=5000.0,
                mean=2100.0,
                std=600.0,
            ),
            detected_genes_distribution=DistributionStats(
                min=200.0,
                p5=400.0,
                p25=600.0,
                p50=800.0,
                p75=1000.0,
                p95=1400.0,
                max=2000.0,
                mean=850.0,
                std=250.0,
            ),
            mito_pct_distribution=DistributionStats(
                min=0.5,
                p5=1.0,
                p25=2.0,
                p50=3.0,
                p75=4.5,
                p95=7.0,
                max=9.5,
                mean=3.2,
                std=1.8,
            ),
            cluster_profiles=[
                ClusterQCProfile(
                    cluster_id="0",
                    cell_count=800,
                    pct_of_total=40.0,
                    median_total_counts=2100.0,
                    median_genes_detected=850.0,
                    median_pct_mt=2.8,
                    top_marker_genes=["IL7R", "CCR7", "CD3D"],
                )
            ],
        )

        assert summary.dataset_name == "pbmc3k"
        assert summary.species == "human"
        assert len(summary.cluster_profiles) == 1

    def test_species_validation(self):
        """Test that only 'human' or 'mouse' are accepted."""
        # Valid: human
        summary = DatasetQCSummary(
            dataset_name="test",
            total_cells=100,
            total_genes=1000,
            species="human",
            total_counts_distribution=DistributionStats(
                min=1.0, p5=2.0, p25=3.0, p50=4.0, p75=5.0, p95=6.0, max=7.0, mean=4.0, std=1.0
            ),
            detected_genes_distribution=DistributionStats(
                min=1.0, p5=2.0, p25=3.0, p50=4.0, p75=5.0, p95=6.0, max=7.0, mean=4.0, std=1.0
            ),
            mito_pct_distribution=DistributionStats(
                min=1.0, p5=2.0, p25=3.0, p50=4.0, p75=5.0, p95=6.0, max=7.0, mean=4.0, std=1.0
            ),
            cluster_profiles=[],
        )
        assert summary.species == "human"

        # Valid: mouse
        summary.species = "mouse"
        assert summary.species == "mouse"

        # Invalid: other species should fail
        with pytest.raises(ValidationError):
            DatasetQCSummary(
                dataset_name="test",
                total_cells=100,
                total_genes=1000,
                species="rat",  # Invalid
                total_counts_distribution=DistributionStats(
                    min=1.0, p5=2.0, p25=3.0, p50=4.0, p75=5.0, p95=6.0, max=7.0, mean=4.0, std=1.0
                ),
                detected_genes_distribution=DistributionStats(
                    min=1.0, p5=2.0, p25=3.0, p50=4.0, p75=5.0, p95=6.0, max=7.0, mean=4.0, std=1.0
                ),
                mito_pct_distribution=DistributionStats(
                    min=1.0, p5=2.0, p25=3.0, p50=4.0, p75=5.0, p95=6.0, max=7.0, mean=4.0, std=1.0
                ),
                cluster_profiles=[],
            )


class TestQCFlag:
    """Tests for QCFlag schema."""

    def test_valid_flag(self):
        """Test creation with valid flag data."""
        flag = QCFlag(
            flag_type="HIGH_MITOCHONDRIAL_READS",
            affected_clusters=["1", "3"],
            severity="CRITICAL",
            numerical_evidence=[
                "Cluster 1 median pct_mt = 34.2%",
                "Cluster 3 median pct_mt = 28.5%",
            ],
            technical_mechanism="Membrane lysis during dissociation leading to cytoplasmic mRNA leakage and enrichment of mitochondrial transcripts.",
            biological_confounder_warning="This pattern is consistent with apoptotic cell damage during sample preparation and must not be interpreted as a metabolically hyperactive cell state.",
        )

        assert flag.flag_type == "HIGH_MITOCHONDRIAL_READS"
        assert flag.severity == "CRITICAL"
        assert len(flag.affected_clusters) == 2

    def test_flag_type_validation(self):
        """Test that only valid flag types are accepted."""
        valid_types = [
            "HIGH_MITOCHONDRIAL_READS",
            "POTENTIAL_DOUBLETS",
            "LOW_LIBRARY_COMPLEXITY",
            "UNEVEN_CLUSTER_DISTRIBUTION",
            "NOVEL_SUBTYPE_CLAIM_RISK",
        ]

        for flag_type in valid_types:
            flag = QCFlag(
                flag_type=flag_type,
                affected_clusters=["0"],
                severity="WARNING",
                numerical_evidence=["Test evidence"],
                technical_mechanism="Test mechanism",
                biological_confounder_warning="Test warning",
            )
            assert flag.flag_type == flag_type

        # Invalid type should fail
        with pytest.raises(ValidationError):
            QCFlag(
                flag_type="INVALID_TYPE",
                affected_clusters=["0"],
                severity="WARNING",
                numerical_evidence=["Test"],
                technical_mechanism="Test",
                biological_confounder_warning="Test",
            )


class TestAuditReport:
    """Tests for AuditReport schema."""

    def test_valid_report(self):
        """Test creation with valid audit report."""
        report = AuditReport(
            summary_verdict="REQUIRES_FILTERING",
            flags=[
                QCFlag(
                    flag_type="HIGH_MITOCHONDRIAL_READS",
                    affected_clusters=["2"],
                    severity="WARNING",
                    numerical_evidence=["Cluster 2 median pct_mt = 18.5%"],
                    technical_mechanism="Possible membrane damage during dissociation",
                    biological_confounder_warning="Not a biological phenotype",
                )
            ],
            recommended_actions=[
                "Filter cells with pct_counts_mt > 15%",
                "Re-cluster after filtering",
            ],
        )

        assert report.summary_verdict == "REQUIRES_FILTERING"
        assert len(report.flags) == 1
        assert len(report.recommended_actions) == 2

    def test_verdict_validation(self):
        """Test that only valid verdicts are accepted."""
        valid_verdicts = ["ACCEPTABLE", "REQUIRES_FILTERING", "CRITICAL_FAIL"]

        for verdict in valid_verdicts:
            report = AuditReport(
                summary_verdict=verdict,
                flags=[],
                recommended_actions=[],
            )
            assert report.summary_verdict == verdict

        # Invalid verdict should fail
        with pytest.raises(ValidationError):
            AuditReport(
                summary_verdict="INVALID_VERDICT",
                flags=[],
                recommended_actions=[],
            )

    def test_complete_serialization_roundtrip(self):
        """Test full serialization roundtrip of a complex audit report."""
        original = AuditReport(
            summary_verdict="CRITICAL_FAIL",
            flags=[
                QCFlag(
                    flag_type="HIGH_MITOCHONDRIAL_READS",
                    affected_clusters=["1"],
                    severity="CRITICAL",
                    numerical_evidence=["Cluster 1: median pct_mt = 45.2%"],
                    technical_mechanism="Severe membrane lysis",
                    biological_confounder_warning="Not a biological state",
                ),
                QCFlag(
                    flag_type="POTENTIAL_DOUBLETS",
                    affected_clusters=["3"],
                    severity="WARNING",
                    numerical_evidence=["Cluster 3: median total_counts = 12000"],
                    technical_mechanism="Multi-cell co-encapsulation",
                    biological_confounder_warning="Not a cell fusion event",
                ),
            ],
            recommended_actions=[
                "Remove cluster 1",
                "Screen cluster 3 with doublet detection tools",
                "Re-run clustering after filtering",
            ],
        )

        # Serialize to dict
        data_dict = original.model_dump()

        # Deserialize from dict
        reconstructed = AuditReport.model_validate(data_dict)

        # Verify structure
        assert reconstructed.summary_verdict == original.summary_verdict
        assert len(reconstructed.flags) == len(original.flags)
        assert reconstructed.flags[0].flag_type == original.flags[0].flag_type
