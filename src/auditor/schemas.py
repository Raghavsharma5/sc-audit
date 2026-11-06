"""
Pydantic v2 schemas for sc-audit data contracts.

These schemas define strict contracts between the Scanpy pipeline and the LLM auditors.
All communication must conform to these schemas to ensure deterministic, verifiable audits.
"""

from typing import List, Literal
from pydantic import BaseModel, Field


class DistributionStats(BaseModel):
    """Statistical summary of a distribution (e.g., UMI counts, genes detected, % MT)."""

    min: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    max: float
    mean: float
    std: float


class ClusterQCProfile(BaseModel):
    """Quality control profile for a single Leiden cluster."""

    cluster_id: str
    cell_count: int
    pct_of_total: float
    median_total_counts: float
    median_genes_detected: float
    median_pct_mt: float
    top_marker_genes: List[str] = Field(..., max_length=5)


class DatasetQCSummary(BaseModel):
    """Complete QC summary of a single-cell dataset for audit purposes.

    This is the INPUT schema sent to both baseline and Claude auditors.
    """

    dataset_name: str
    total_cells: int
    total_genes: int
    species: Literal["human", "mouse"]
    total_counts_distribution: DistributionStats
    detected_genes_distribution: DistributionStats
    mito_pct_distribution: DistributionStats
    cluster_profiles: List[ClusterQCProfile]


class QCFlag(BaseModel):
    """A single quality control flag raised by an auditor.

    Each flag must provide:
    - Numerical evidence citing exact values from the input JSON
    - Technical mechanism explaining the biophysical cause
    - Biological confounder warning preventing false biological narratives
    """

    flag_type: Literal[
        "HIGH_MITOCHONDRIAL_READS",
        "POTENTIAL_DOUBLETS",
        "LOW_LIBRARY_COMPLEXITY",
        "UNEVEN_CLUSTER_DISTRIBUTION",
        "NOVEL_SUBTYPE_CLAIM_RISK",
    ]
    affected_clusters: List[str]
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    numerical_evidence: List[str] = Field(
        ...,
        description="Exact numerical figures from the input JSON (e.g. 'Cluster 1 median pct_mt = 34.2%').",
    )
    technical_mechanism: str = Field(
        ...,
        description="Biophysical etiology (e.g. membrane lysis leaking cytoplasmic mRNA; multi-cell co-encapsulation).",
    )
    biological_confounder_warning: str = Field(
        ...,
        description="Warning explicitly identifying why this must not be classified as a novel biological phenotype.",
    )


class AuditReport(BaseModel):
    """Complete audit report from either baseline or Claude auditor.

    This is the OUTPUT schema that both auditors must conform to.
    """

    summary_verdict: Literal["ACCEPTABLE", "REQUIRES_FILTERING", "CRITICAL_FAIL"]
    flags: List[QCFlag]
    recommended_actions: List[str]
