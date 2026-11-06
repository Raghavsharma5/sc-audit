"""
Claude-based LLM auditor using Anthropic API with tool use for structured output.

This auditor uses Claude 3.5 Sonnet to identify QC pathologies from structured
summaries, enforcing strict schema compliance via tool use.
"""

import os
import json
from typing import Optional
from anthropic import Anthropic

from .schemas import DatasetQCSummary, AuditReport


# System prompt emphasizing groundedness and technical rigor
SYSTEM_PROMPT = """You are a single-cell computational biology quality control auditor specializing in identifying technical artifacts in scRNA-seq data.

Your role is to analyze QC summaries and flag experimental pathologies that compromise data quality. Focus exclusively on:
1. Technical dissociation damage (apoptotic lysis, membrane compromise)
2. Doublet/multiplet contamination from co-encapsulation
3. Library complexity issues (degradation, ambient RNA, empty droplets)

CRITICAL REQUIREMENTS:

**Grounding in Evidence:**
- Every claim in `numerical_evidence` must cite EXACT values from the input JSON
- Use precise numbers with appropriate decimal places (e.g., "Cluster 1 median pct_mt = 6.54%")
- Reference dataset-level statistics when making comparisons (e.g., "Dataset median = 2.31%")
- Never invent, approximate, or estimate values not present in the input

**Technical Mechanism:**
- Explain the biophysical or molecular cause of each artifact
- Examples:
  * High MT%: "Membrane lysis during dissociation allows cytoplasmic mRNA to escape while mitochondrial transcripts remain sequestered"
  * Doublets: "Multi-cell co-encapsulation in droplets produces merged transcriptomes with additive UMI counts"
  * Low complexity: "RNA degradation or ambient contamination reduces detected gene diversity"

**Biological Confounder Warning:**
- Explicitly warn against misinterpreting technical artifacts as biological phenomena
- Examples of FORBIDDEN interpretations:
  * High MT% ≠ "metabolically hyperactive cells" or "mitochondria-rich cell type"
  * Doublets ≠ "trans-differentiation" or "hybrid cell state"
  * Low complexity ≠ "quiescent stem cells" or "dormant progenitors"
- State clearly: "This is a technical artifact, not a biological phenotype"

**Conservative Flagging:**
- Only flag clear outliers with strong numerical support
- Use severity levels appropriately:
  * CRITICAL: Unambiguous technical failure requiring immediate filtering
  * WARNING: Suspicious patterns warranting investigation
  * INFO: Minor observations for awareness
- If a cluster appears normal, do not flag it

**Prohibited Actions:**
- Do NOT speculate about novel cell types or states
- Do NOT claim biological discoveries based on QC anomalies
- Do NOT ignore technical explanations in favor of biological narratives
- Do NOT fabricate numerical values or metrics not in the input

Your output will be validated against the ground truth. Precision and groundedness are paramount."""


def claude_audit(
    summary: DatasetQCSummary,
    api_key: Optional[str] = None,
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 4096,
) -> AuditReport:
    """
    Perform LLM-based QC audit using Claude with tool use for structured output.

    Parameters
    ----------
    summary : DatasetQCSummary
        QC summary extracted from processed AnnData
    api_key : str, optional
        Anthropic API key. If None, reads from ANTHROPIC_API_KEY environment variable.
    model : str
        Claude model identifier
    max_tokens : int
        Maximum tokens for response

    Returns
    -------
    AuditReport
        Structured audit report with flags and recommendations

    Raises
    ------
    ValueError
        If API key is not provided and not found in environment
    anthropic.APIError
        If API call fails

    Notes
    -----
    Uses Anthropic's tool use feature to enforce strict schema compliance.
    The model must call the "emit_audit_report" tool with a valid AuditReport payload.
    """
    # Initialize Anthropic client
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

    client = Anthropic(api_key=api_key)

    # Convert summary to JSON for the prompt
    summary_json = summary.model_dump_json(indent=2)

    # Define tool schema from AuditReport
    tool_schema = {
        "name": "emit_audit_report",
        "description": "Emit a structured quality control audit report for the scRNA-seq dataset",
        "input_schema": AuditReport.model_json_schema(),
    }

    # Construct user message with the QC summary
    user_message = f"""Analyze the following single-cell RNA-seq QC summary and emit an audit report.

Dataset QC Summary (JSON):
```json
{summary_json}
```

Instructions:
1. Identify technical QC issues (high MT%, doublets, low complexity)
2. For each issue, provide:
   - Exact numerical evidence from the JSON above
   - Technical mechanism explaining the biophysical cause
   - Explicit warning against biological misinterpretation
3. Emit your findings using the emit_audit_report tool

Be precise, conservative, and grounded in the numerical evidence."""

    # Make API call with tool use
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": "emit_audit_report"},
        messages=[{"role": "user", "content": user_message}],
    )

    # Extract tool use from response
    if not response.content or len(response.content) == 0:
        raise ValueError("Empty response from Claude API")

    # Find the tool use block
    tool_use_block = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "emit_audit_report":
            tool_use_block = block
            break

    if tool_use_block is None:
        raise ValueError("No tool use found in Claude response")

    # Parse and validate the tool input as AuditReport
    try:
        audit_report = AuditReport.model_validate(tool_use_block.input)
    except Exception as e:
        raise ValueError(f"Failed to validate AuditReport from Claude response: {e}") from e

    return audit_report


if __name__ == "__main__":
    """Quick test of Claude auditor (requires API key)."""
    import sys
    from pathlib import Path
    from ..pipeline.ingest import load_pbmc3k
    from ..pipeline.qc_engine import run_qc_pipeline
    from ..pipeline.extractor import extract_qc_summary
    from ..perturbation.inject import generate_perturbed_dataset

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    print("Testing Claude auditor on different conditions...")
    print(f"Using model: claude-3-5-sonnet-20241022")
    project_root = Path(__file__).parent.parent.parent

    # Load and process data
    adata = load_pbmc3k(cache_dir=project_root / "data")
    adata_processed = run_qc_pipeline(adata, species="human")

    # Test on clean control first
    condition = "clean_control"
    print(f"\n{'='*60}")
    print(f"Condition: {condition}")
    print(f"{'='*60}")

    adata_perturbed, metadata = generate_perturbed_dataset(
        adata_processed, condition, species="human"
    )

    summary = extract_qc_summary(adata_perturbed, dataset_name=f"pbmc3k_{condition}")

    try:
        report = claude_audit(summary, api_key=api_key)

        print(f"Verdict: {report.summary_verdict}")
        print(f"Flags: {len(report.flags)}")
        for flag in report.flags:
            print(f"  - {flag.flag_type} (Severity: {flag.severity})")
            print(f"    Affected clusters: {flag.affected_clusters}")
            print(f"    Evidence: {flag.numerical_evidence[0] if flag.numerical_evidence else 'N/A'}")

        print(f"\nExpected flags: {metadata.get('expected_flags', [])}")
        print("\nClaude audit completed successfully!")

    except Exception as e:
        print(f"ERROR: Claude audit failed: {e}")
        import traceback

        traceback.print_exc()
