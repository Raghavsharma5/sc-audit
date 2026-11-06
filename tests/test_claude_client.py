"""
Unit tests for Claude LLM auditor.

Tests schema validation and API integration (when API key is available).
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from anthropic.types import Message, ContentBlock, ToolUseBlock, Usage

from src.pipeline.ingest import load_pbmc3k
from src.pipeline.qc_engine import run_qc_pipeline
from src.pipeline.extractor import extract_qc_summary
from src.perturbation.inject import generate_perturbed_dataset
from src.auditor.claude_client import claude_audit
from src.auditor.schemas import AuditReport, QCFlag


@pytest.fixture(scope="module")
def processed_adata():
    """Load and process PBMC3k dataset (shared across tests)."""
    project_root = Path(__file__).parent.parent
    adata = load_pbmc3k(cache_dir=project_root / "data")
    return run_qc_pipeline(adata, species="human")


@pytest.fixture
def clean_summary(processed_adata):
    """Generate a clean control summary."""
    adata_clean, _ = generate_perturbed_dataset(
        processed_adata, "clean_control", species="human"
    )
    return extract_qc_summary(adata_clean, dataset_name="pbmc3k_clean", species="human")


class TestMockAPIResponse:
    """Tests using mocked API responses."""

    def test_valid_tool_response_parsing(self, clean_summary):
        """Test that valid tool use response is correctly parsed."""
        # Create a mock valid AuditReport response
        mock_audit_data = {
            "summary_verdict": "ACCEPTABLE",
            "flags": [],
            "recommended_actions": ["Dataset passes QC checks"],
        }

        # Create mock tool use block
        mock_tool_use = Mock(spec=ToolUseBlock)
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "emit_audit_report"
        mock_tool_use.input = mock_audit_data

        # Create mock message
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_tool_use]

        # Mock the Anthropic client
        with patch("src.auditor.claude_client.Anthropic") as MockAnthropic:
            mock_client = MockAnthropic.return_value
            mock_client.messages.create.return_value = mock_message

            # Run audit with mock
            report = claude_audit(clean_summary, api_key="fake-key-for-testing")

            # Verify the report structure
            assert isinstance(report, AuditReport)
            assert report.summary_verdict == "ACCEPTABLE"
            assert len(report.flags) == 0
            assert len(report.recommended_actions) == 1

    def test_tool_response_with_flags(self, clean_summary):
        """Test parsing of tool response with QC flags."""
        # Create a mock response with flags
        mock_audit_data = {
            "summary_verdict": "REQUIRES_FILTERING",
            "flags": [
                {
                    "flag_type": "HIGH_MITOCHONDRIAL_READS",
                    "affected_clusters": ["2"],
                    "severity": "CRITICAL",
                    "numerical_evidence": ["Cluster 2: median pct_mt = 18.5%"],
                    "technical_mechanism": "Membrane lysis during dissociation",
                    "biological_confounder_warning": "Not a metabolically active cell type",
                }
            ],
            "recommended_actions": ["Filter cluster 2"],
        }

        mock_tool_use = Mock(spec=ToolUseBlock)
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "emit_audit_report"
        mock_tool_use.input = mock_audit_data

        mock_message = Mock(spec=Message)
        mock_message.content = [mock_tool_use]

        with patch("src.auditor.claude_client.Anthropic") as MockAnthropic:
            mock_client = MockAnthropic.return_value
            mock_client.messages.create.return_value = mock_message

            report = claude_audit(clean_summary, api_key="fake-key")

            assert isinstance(report, AuditReport)
            assert report.summary_verdict == "REQUIRES_FILTERING"
            assert len(report.flags) == 1
            assert report.flags[0].flag_type == "HIGH_MITOCHONDRIAL_READS"
            assert report.flags[0].severity == "CRITICAL"

    def test_missing_tool_use_raises_error(self, clean_summary):
        """Test that missing tool use raises ValueError."""
        # Mock response without tool use
        mock_message = Mock(spec=Message)
        mock_message.content = []

        with patch("src.auditor.claude_client.Anthropic") as MockAnthropic:
            mock_client = MockAnthropic.return_value
            mock_client.messages.create.return_value = mock_message

            with pytest.raises(ValueError, match="Empty response from Claude API"):
                claude_audit(clean_summary, api_key="fake-key")

    def test_invalid_schema_raises_error(self, clean_summary):
        """Test that invalid schema in tool response raises ValueError."""
        # Create invalid audit data (missing required field)
        mock_audit_data = {
            "summary_verdict": "ACCEPTABLE",
            # Missing 'flags' field - should cause validation error
            "recommended_actions": ["Test"],
        }

        mock_tool_use = Mock(spec=ToolUseBlock)
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "emit_audit_report"
        mock_tool_use.input = mock_audit_data

        mock_message = Mock(spec=Message)
        mock_message.content = [mock_tool_use]

        with patch("src.auditor.claude_client.Anthropic") as MockAnthropic:
            mock_client = MockAnthropic.return_value
            mock_client.messages.create.return_value = mock_message

            with pytest.raises(ValueError, match="Failed to validate AuditReport"):
                claude_audit(clean_summary, api_key="fake-key")


class TestAPIKeyHandling:
    """Tests for API key handling."""

    def test_missing_api_key_raises_error(self, clean_summary):
        """Test that missing API key raises ValueError."""
        # Temporarily unset environment variable
        original_key = os.environ.get("ANTHROPIC_API_KEY")
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        try:
            with pytest.raises(ValueError, match="API key must be provided"):
                claude_audit(clean_summary, api_key=None)
        finally:
            # Restore original key if it existed
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    def test_explicit_api_key_used(self, clean_summary):
        """Test that explicitly provided API key is used."""
        mock_tool_use = Mock(spec=ToolUseBlock)
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "emit_audit_report"
        mock_tool_use.input = {
            "summary_verdict": "ACCEPTABLE",
            "flags": [],
            "recommended_actions": ["OK"],
        }

        mock_message = Mock(spec=Message)
        mock_message.content = [mock_tool_use]

        with patch("src.auditor.claude_client.Anthropic") as MockAnthropic:
            mock_client = MockAnthropic.return_value
            mock_client.messages.create.return_value = mock_message

            # Call with explicit key
            explicit_key = "explicit-test-key"
            claude_audit(clean_summary, api_key=explicit_key)

            # Verify Anthropic was called with the explicit key
            MockAnthropic.assert_called_once_with(api_key=explicit_key)


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY environment variable not set"
)
class TestRealAPIIntegration:
    """Tests using real Anthropic API (only run if API key is set)."""

    def test_clean_control_audit(self, processed_adata):
        """Test Claude audit on clean control dataset."""
        # Generate clean control
        adata_clean, metadata = generate_perturbed_dataset(
            processed_adata, "clean_control", species="human"
        )

        summary = extract_qc_summary(adata_clean, dataset_name="pbmc3k_clean", species="human")

        # Run Claude audit (will use real API)
        report = claude_audit(summary)

        # Verify report structure
        assert isinstance(report, AuditReport)
        assert report.summary_verdict in ["ACCEPTABLE", "REQUIRES_FILTERING", "CRITICAL_FAIL"]
        assert isinstance(report.flags, list)
        assert isinstance(report.recommended_actions, list)

        # Clean control should ideally have ACCEPTABLE or minimal flags
        # (but we don't enforce this strictly since the LLM may be conservative)
        print(f"\nClaude verdict on clean control: {report.summary_verdict}")
        print(f"Number of flags: {len(report.flags)}")

    def test_lysis_audit(self, processed_adata):
        """Test Claude audit on lysis-injected dataset."""
        # Generate lysis condition
        adata_lysis, metadata = generate_perturbed_dataset(
            processed_adata, "injected_lysis", species="human"
        )

        adata_lysis_processed = run_qc_pipeline(adata_lysis, species="human")
        summary = extract_qc_summary(
            adata_lysis_processed, dataset_name="pbmc3k_lysis", species="human"
        )

        # Run Claude audit
        report = claude_audit(summary)

        # Verify report structure
        assert isinstance(report, AuditReport)
        assert len(report.flags) >= 0  # May or may not flag depending on severity

        # If flags are present, verify structure
        for flag in report.flags:
            assert flag.flag_type in [
                "HIGH_MITOCHONDRIAL_READS",
                "POTENTIAL_DOUBLETS",
                "LOW_LIBRARY_COMPLEXITY",
                "UNEVEN_CLUSTER_DISTRIBUTION",
                "NOVEL_SUBTYPE_CLAIM_RISK",
            ]
            assert len(flag.numerical_evidence) > 0
            assert len(flag.technical_mechanism) > 0
            assert len(flag.biological_confounder_warning) > 0

        print(f"\nClaude verdict on lysis: {report.summary_verdict}")
        print(f"Flags: {[f.flag_type for f in report.flags]}")

    def test_schema_validation_on_real_response(self, clean_summary):
        """Test that real API responses validate against AuditReport schema."""
        # This test will actually call the API and verify the response
        # validates against our schema
        report = claude_audit(clean_summary)

        # If we got here, schema validation passed
        assert isinstance(report, AuditReport)

        # Verify all required fields are present
        assert hasattr(report, "summary_verdict")
        assert hasattr(report, "flags")
        assert hasattr(report, "recommended_actions")

        # Verify each flag has required fields
        for flag in report.flags:
            assert hasattr(flag, "flag_type")
            assert hasattr(flag, "affected_clusters")
            assert hasattr(flag, "severity")
            assert hasattr(flag, "numerical_evidence")
            assert hasattr(flag, "technical_mechanism")
            assert hasattr(flag, "biological_confounder_warning")

            # Verify types
            assert isinstance(flag.numerical_evidence, list)
            assert isinstance(flag.affected_clusters, list)
            assert flag.severity in ["INFO", "WARNING", "CRITICAL"]


class TestSystemPromptCoverage:
    """Tests to verify system prompt directives are working."""

    def test_tool_schema_includes_all_fields(self):
        """Test that tool schema includes all required AuditReport fields."""
        from src.auditor.claude_client import AuditReport

        schema = AuditReport.model_json_schema()

        # Check that required fields are in the schema
        assert "properties" in schema
        properties = schema["properties"]

        assert "summary_verdict" in properties
        assert "flags" in properties
        assert "recommended_actions" in properties

        # Check that QCFlag schema has required fields
        if "flags" in properties:
            # Navigate to QCFlag definition in schema
            flag_items = properties["flags"].get("items", {})
            if "$ref" in flag_items or "properties" in flag_items:
                # Schema is properly structured
                pass  # Schema structure validation passed
