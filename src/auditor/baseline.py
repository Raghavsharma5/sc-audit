"""
Baseline deterministic auditor using hard-coded QC thresholds.

This auditor serves as the ground-truth reference against which the LLM
auditor's performance will be benchmarked.
"""

from typing import List, Literal
from .schemas import (
    DatasetQCSummary,
    QCFlag,
    AuditReport,
)


# Threshold constants
MT_PERCENT_ABSOLUTE_THRESHOLD = 5.0  # Absolute pct_mt threshold
MT_PERCENT_RELATIVE_MULTIPLIER = 2.5  # Relative to dataset median
DOUBLET_COUNT_MULTIPLIER = 1.5  # Relative to dataset median total counts (lowered to catch dispersed doublets)
LOW_COMPLEXITY_GENE_THRESHOLD = 400  # Minimum genes for acceptable complexity


def baseline_audit(summary: DatasetQCSummary) -> AuditReport:
    """
    Perform deterministic rule-based QC audit.

    Parameters
    ----------
    summary : DatasetQCSummary
        QC summary extracted from processed AnnData

    Returns
    -------
    AuditReport
        Structured audit report with flags and recommendations

    Notes
    -----
    Thresholds:
    - HIGH_MITOCHONDRIAL_READS: cluster median pct_mt > 5.0% OR > 2.5x dataset median
    - POTENTIAL_DOUBLETS: cluster median total_counts > 1.5x dataset p50
    - LOW_LIBRARY_COMPLEXITY: cluster median genes_detected < 400
    """
    flags: List[QCFlag] = []
    recommended_actions: List[str] = []

    # Calculate dataset-level reference values
    dataset_median_pct_mt = summary.mito_pct_distribution.p50
    dataset_median_counts = summary.total_counts_distribution.p50

    # Check each cluster for pathologies
    for cluster in summary.cluster_profiles:
        cluster_id = cluster.cluster_id

        # 1. Check for high mitochondrial reads (apoptotic lysis)
        mt_threshold = max(
            MT_PERCENT_ABSOLUTE_THRESHOLD,
            dataset_median_pct_mt * MT_PERCENT_RELATIVE_MULTIPLIER,
        )
        if cluster.median_pct_mt > mt_threshold:
            flags.append(
                QCFlag(
                    flag_type="HIGH_MITOCHONDRIAL_READS",
                    affected_clusters=[cluster_id],
                    severity="CRITICAL",
                    numerical_evidence=[
                        f"Cluster {cluster_id}: median pct_mt = {cluster.median_pct_mt:.2f}%",
                        f"Dataset median pct_mt = {dataset_median_pct_mt:.2f}%",
                        f"Threshold = {mt_threshold:.2f}%",
                    ],
                    technical_mechanism=(
                        "Elevated mitochondrial read percentage is consistent with membrane damage "
                        "during tissue dissociation. Apoptotic or necrotic cells exhibit compromised "
                        "plasma membranes that allow cytoplasmic mRNA to leak out while mitochondrial "
                        "transcripts remain sequestered, artificially enriching the MT fraction."
                    ),
                    biological_confounder_warning=(
                        "This pattern must NOT be interpreted as a metabolically hyperactive cell state "
                        "or novel mitochondria-rich cell type. Elevated MT% is a technical artifact "
                        "from sample preparation and indicates poor cell quality requiring filtration."
                    ),
                )
            )
            recommended_actions.append(
                f"Filter or remove cluster {cluster_id} due to high mitochondrial content (likely lysed cells)"
            )

        # 2. Check for potential doublets (multi-cell encapsulation)
        doublet_threshold = dataset_median_counts * DOUBLET_COUNT_MULTIPLIER
        if cluster.median_total_counts > doublet_threshold:
            flags.append(
                QCFlag(
                    flag_type="POTENTIAL_DOUBLETS",
                    affected_clusters=[cluster_id],
                    severity="WARNING",
                    numerical_evidence=[
                        f"Cluster {cluster_id}: median total_counts = {cluster.median_total_counts:.0f}",
                        f"Dataset median total_counts = {dataset_median_counts:.0f}",
                        f"Doublet threshold = {doublet_threshold:.0f} ({DOUBLET_COUNT_MULTIPLIER}x median)",
                    ],
                    technical_mechanism=(
                        "Abnormally high total UMI counts and gene detection rates suggest multi-cell "
                        "co-encapsulation in droplets (doublets/multiplets). This occurs when two or more "
                        "cells are captured in the same reaction volume, producing merged transcriptomes "
                        "with additive count profiles and mixed cell-type marker expression."
                    ),
                    biological_confounder_warning=(
                        "Do NOT classify this as a trans-differentiation event, cell fusion, or hybrid "
                        "cell state without first ruling out doublet contamination using computational "
                        "doublet detection tools (e.g., Scrublet, DoubletFinder). Elevated counts are "
                        "a hallmark technical signature of multiplet artifacts."
                    ),
                )
            )
            recommended_actions.append(
                f"Screen cluster {cluster_id} with doublet detection tools (e.g., Scrublet) due to elevated UMI counts"
            )

        # 3. Check for low library complexity (degradation or empty drops)
        if cluster.median_genes_detected < LOW_COMPLEXITY_GENE_THRESHOLD:
            flags.append(
                QCFlag(
                    flag_type="LOW_LIBRARY_COMPLEXITY",
                    affected_clusters=[cluster_id],
                    severity="WARNING",
                    numerical_evidence=[
                        f"Cluster {cluster_id}: median genes_detected = {cluster.median_genes_detected:.0f}",
                        f"Complexity threshold = {LOW_COMPLEXITY_GENE_THRESHOLD} genes",
                    ],
                    technical_mechanism=(
                        "Low gene detection rates indicate degraded RNA libraries, ambient RNA contamination, "
                        "or empty droplet capture. This can result from extended sample processing times, "
                        "improper storage conditions, or inefficient cell lysis and reverse transcription. "
                        "Cells with <400 detected genes typically lack sufficient transcriptomic information "
                        "for robust cell-type annotation."
                    ),
                    biological_confounder_warning=(
                        "Do NOT interpret this as a quiescent stem cell population, dormant progenitor state, "
                        "or transcriptionally silent cell type without first verifying technical quality. "
                        "Low complexity is most often a technical artifact requiring removal rather than "
                        "a meaningful biological signal."
                    ),
                )
            )
            recommended_actions.append(
                f"Filter cluster {cluster_id} due to low library complexity (<{LOW_COMPLEXITY_GENE_THRESHOLD} genes detected)"
            )

    # Determine overall verdict based on severity
    if any(flag.severity == "CRITICAL" for flag in flags):
        summary_verdict: Literal["ACCEPTABLE", "REQUIRES_FILTERING", "CRITICAL_FAIL"] = (
            "CRITICAL_FAIL"
        )
        recommended_actions.insert(
            0, "CRITICAL ISSUES DETECTED: Dataset requires immediate filtering and re-analysis"
        )
    elif any(flag.severity == "WARNING" for flag in flags):
        summary_verdict = "REQUIRES_FILTERING"
        recommended_actions.insert(
            0, "Dataset contains quality issues that should be addressed through filtering"
        )
    else:
        summary_verdict = "ACCEPTABLE"
        recommended_actions.append("Dataset passes baseline QC checks")

    return AuditReport(
        summary_verdict=summary_verdict,
        flags=flags,
        recommended_actions=recommended_actions,
    )


if __name__ == "__main__":
    """Quick test of baseline auditor."""
    from pathlib import Path
    from ..pipeline.ingest import load_pbmc3k
    from ..pipeline.qc_engine import run_qc_pipeline
    from ..pipeline.extractor import extract_qc_summary
    from ..perturbation.inject import generate_perturbed_dataset

    print("Testing baseline auditor on different conditions...")
    project_root = Path(__file__).parent.parent.parent

    # Load and process data
    adata = load_pbmc3k(cache_dir=project_root / "data")
    adata_processed = run_qc_pipeline(adata, species="human")

    # Test on different conditions
    conditions = ["clean_control", "injected_lysis", "injected_doublets", "injected_degraded"]

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        # Generate perturbed dataset
        adata_perturbed, metadata = generate_perturbed_dataset(
            adata_processed, condition, species="human"
        )

        # Re-run QC pipeline to get clustering on perturbed data
        if condition != "clean_control":
            adata_perturbed = run_qc_pipeline(adata_perturbed, species="human")

        # Extract summary
        summary = extract_qc_summary(adata_perturbed, dataset_name=f"pbmc3k_{condition}")

        # Run baseline audit
        report = baseline_audit(summary)

        print(f"Verdict: {report.summary_verdict}")
        print(f"Flags: {len(report.flags)}")
        for flag in report.flags:
            print(f"  - {flag.flag_type} (Severity: {flag.severity})")
            print(f"    Affected clusters: {flag.affected_clusters}")
            print(f"    Evidence: {flag.numerical_evidence[0]}")

        print(f"\nExpected flags (from metadata): {metadata.get('expected_flags', [])}")
