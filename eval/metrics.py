"""
Evaluation metrics for auditor performance.

Implements:
- Flag Precision/Recall/F1 (comparing detected vs expected flags)
- Groundedness Ratio (verifying numerical evidence matches input data)
- Speculation Score (detecting biological narratives for technical artifacts)
- Latency and token cost tracking
"""

import re
from typing import List, Tuple, Set, Dict, Any
import numpy as np

from src.auditor.schemas import AuditReport, QCFlag, DatasetQCSummary


def calculate_flag_metrics(
    detected_flags: List[QCFlag], expected_flags: List[str]
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 for flag detection.

    Parameters
    ----------
    detected_flags : List[QCFlag]
        Flags returned by the auditor
    expected_flags : List[str]
        Ground truth flag types (e.g., ["HIGH_MITOCHONDRIAL_READS"])

    Returns
    -------
    Tuple[float, float, float]
        (precision, recall, f1_score)

    Notes
    -----
    - Precision = TP / (TP + FP) = fraction of detected flags that are correct
    - Recall = TP / (TP + FN) = fraction of expected flags that were detected
    - F1 = 2 * (precision * recall) / (precision + recall)
    - Matching is based on exact flag_type string equality
    """
    detected_types = {flag.flag_type for flag in detected_flags}
    expected_types = set(expected_flags)

    # True Positives: flags that were expected and detected
    true_positives = len(detected_types & expected_types)

    # False Positives: flags detected but not expected
    false_positives = len(detected_types - expected_types)

    # False Negatives: flags expected but not detected
    false_negatives = len(expected_types - detected_types)

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 1.0  # If no flags expected and none detected, perfect recall
    )

    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1_score


def extract_numeric_values_from_summary(summary: DatasetQCSummary) -> Set[float]:
    """
    Extract all numeric values from a DatasetQCSummary for groundedness checking.

    Parameters
    ----------
    summary : DatasetQCSummary
        QC summary containing all numeric metrics

    Returns
    -------
    Set[float]
        All numeric values present in the summary (with rounding tolerance)

    Notes
    -----
    Extracts values from:
    - Distribution stats (min, p5, p25, p50, p75, p95, max, mean, std)
    - Cluster profiles (cell_count, pct_of_total, medians)
    - Dataset-level counts (total_cells, total_genes)
    """
    values = set()

    # Dataset-level counts
    values.add(float(summary.total_cells))
    values.add(float(summary.total_genes))

    # Distribution statistics
    for dist in [
        summary.total_counts_distribution,
        summary.detected_genes_distribution,
        summary.mito_pct_distribution,
    ]:
        values.add(dist.min)
        values.add(dist.p5)
        values.add(dist.p25)
        values.add(dist.p50)
        values.add(dist.p75)
        values.add(dist.p95)
        values.add(dist.max)
        values.add(dist.mean)
        values.add(dist.std)

    # Cluster profiles
    for cluster in summary.cluster_profiles:
        values.add(float(cluster.cell_count))
        values.add(cluster.pct_of_total)
        values.add(cluster.median_total_counts)
        values.add(cluster.median_genes_detected)
        values.add(cluster.median_pct_mt)

    return values


def is_value_grounded(
    cited_value: float, reference_values: Set[float], tolerance_pct: float = 1.0
) -> bool:
    """
    Check if a cited numerical value matches any reference value within tolerance.

    Parameters
    ----------
    cited_value : float
        Value cited in numerical_evidence
    reference_values : Set[float]
        Ground truth values from input summary
    tolerance_pct : float
        Percentage tolerance for matching (default 1.0%)

    Returns
    -------
    bool
        True if cited_value matches any reference value within tolerance
    """
    for ref_val in reference_values:
        # Calculate absolute tolerance
        tolerance = abs(ref_val * tolerance_pct / 100.0)
        if abs(cited_value - ref_val) <= tolerance:
            return True
    return False


def calculate_groundedness(
    report: AuditReport, summary: DatasetQCSummary, tolerance_pct: float = 1.0
) -> float:
    """
    Calculate groundedness ratio: fraction of cited numbers that match input data.

    Parameters
    ----------
    report : AuditReport
        Audit report containing flags with numerical_evidence
    summary : DatasetQCSummary
        Original input summary containing ground truth values
    tolerance_pct : float
        Percentage tolerance for numeric matching (default 1.0%)

    Returns
    -------
    float
        Groundedness ratio in [0.0, 1.0]
        - 1.0 = all cited numbers match input data
        - 0.0 = no cited numbers match input data
        - 1.0 if no numbers cited (vacuously true)

    Notes
    -----
    Uses regex ``\\b\\d+(?:\\.\\d+)?\\b`` to extract all integers and floats from
    numerical_evidence strings. Matches against all numeric values in the
    summary with tolerance for rounding.
    """
    # Extract all reference values from summary
    reference_values = extract_numeric_values_from_summary(summary)

    # Extract all cited numbers from numerical_evidence
    cited_numbers = []
    number_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")

    for flag in report.flags:
        for evidence_string in flag.numerical_evidence:
            # Extract all numeric tokens
            matches = number_pattern.findall(evidence_string)
            cited_numbers.extend([float(m) for m in matches])

    # If no numbers cited, return 1.0 (vacuously grounded)
    if len(cited_numbers) == 0:
        return 1.0

    # Count how many cited numbers are grounded
    grounded_count = sum(
        1
        for cited in cited_numbers
        if is_value_grounded(cited, reference_values, tolerance_pct)
    )

    groundedness_ratio = grounded_count / len(cited_numbers)
    return groundedness_ratio


def calculate_speculation_score(
    report: AuditReport, ground_truth_condition: str
) -> int:
    """
    Binary pass/fail for detecting biological speculation on technical artifacts.

    Parameters
    ----------
    report : AuditReport
        Audit report to evaluate
    ground_truth_condition : str
        Ground truth condition name (e.g., "injected_lysis", "clean_control")

    Returns
    -------
    int
        1 = PASS (proper technical interpretation)
        0 = FAIL (biological speculation or missing warnings)

    Scoring Logic
    -------------
    For perturbation conditions ("injected_lysis", "injected_doublets", "injected_degraded"):
    - PASS (1): Every flag has non-empty `biological_confounder_warning`
    - FAIL (0): Any flag missing warning or contains naive phenotypic claims

    For "clean_control":
    - PASS (1): No spurious high-severity (CRITICAL) flags
    - FAIL (0): Contains CRITICAL flags on clean data

    Notes
    -----
    This metric penalizes:
    - Missing biological confounder warnings
    - Hallucinating novel cell types from technical artifacts
    - Over-flagging clean control data
    """
    perturbation_conditions = [
        "injected_lysis",
        "injected_doublets",
        "injected_degraded",
    ]

    if ground_truth_condition in perturbation_conditions:
        # For perturbed data: every flag must have biological_confounder_warning
        for flag in report.flags:
            if not flag.biological_confounder_warning.strip():
                # Missing or empty warning
                return 0

            # Check for naive phenotypic claims without disclaimers
            warning_lower = flag.biological_confounder_warning.lower()

            # Red flags: claiming biological phenomena without "not", "don't", "do not"
            biological_claims = [
                "novel cell type",
                "new cell type",
                "unique population",
                "metabolically active",
                "hyperactive",
                "trans-differentiation",
                "cell fusion",
                "hybrid state",
                "quiescent",
                "dormant",
                "stem cell",
            ]

            for claim in biological_claims:
                if claim in warning_lower:
                    # Check if there's a negation nearby (within 50 chars before)
                    claim_idx = warning_lower.find(claim)
                    context = warning_lower[max(0, claim_idx - 50) : claim_idx]

                    negations = ["not", "don't", "do not", "must not", "cannot"]
                    has_negation = any(neg in context for neg in negations)

                    if not has_negation:
                        # Biological claim without disclaimer = speculation
                        return 0

        # All flags have proper warnings
        return 1

    elif ground_truth_condition == "clean_control":
        # For clean data: no spurious CRITICAL flags
        critical_flags = [f for f in report.flags if f.severity == "CRITICAL"]
        if len(critical_flags) > 0:
            return 0  # False positives on clean data
        return 1

    else:
        # Unknown condition, default to pass
        return 1


def extract_api_metrics(api_response: Any) -> Dict[str, int]:
    """
    Extract token usage from Anthropic API response.

    Parameters
    ----------
    api_response : Any
        Response object from Anthropic API (anthropic.types.Message)

    Returns
    -------
    Dict[str, int]
        Dictionary with keys: "prompt_tokens", "completion_tokens", "total_tokens"

    Notes
    -----
    Anthropic API response has a `.usage` attribute with:
    - input_tokens (prompt tokens)
    - output_tokens (completion tokens)
    """
    if hasattr(api_response, "usage"):
        usage = api_response.usage
        return {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        }
    else:
        # Baseline auditor or mock responses don't have usage
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
