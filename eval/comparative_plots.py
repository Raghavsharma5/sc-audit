"""
Comparative visualization suite for baseline vs. Claude auditor analysis.

Generates publication-grade figures highlighting empirical boundaries and
AI reasoning limitations.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData

# Publication-quality styling
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})


def plot_sensitivity_curves(
    lysis_df: pd.DataFrame,
    doublets_df: pd.DataFrame,
    output_path: Path,
):
    """
    Generate dual sensitivity curve plots comparing baseline vs Claude.

    Parameters
    ----------
    lysis_df : pd.DataFrame
        Lysis sensitivity sweep results
    doublets_df : pd.DataFrame
        Doublet sensitivity sweep results
    output_path : Path
        Where to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Lysis sensitivity
    ax = axes[0]

    # Plot detection curves
    ax.plot(
        lysis_df["mt_scale"],
        lysis_df["baseline_detected"].astype(int),
        marker="o",
        linewidth=2,
        markersize=8,
        label="Baseline (Rule-based)",
        color="#2E86AB",
    )

    if "claude_detected" in lysis_df.columns and lysis_df["claude_detected"].any():
        ax.plot(
            lysis_df["mt_scale"],
            lysis_df["claude_detected"].astype(int),
            marker="s",
            linewidth=2,
            markersize=8,
            label="Claude (LLM)",
            color="#A23B72",
        )

    # Overlay actual median MT% on secondary axis
    ax2 = ax.twinx()
    ax2.plot(
        lysis_df["mt_scale"],
        lysis_df["median_pct_mt_target"],
        linestyle="--",
        color="#F18F01",
        alpha=0.7,
        linewidth=1.5,
        label="Actual median MT% (target cluster)",
    )
    ax2.set_ylabel("Median MT% in Target Cluster", color="#F18F01")
    ax2.tick_params(axis="y", labelcolor="#F18F01")
    ax2.spines["right"].set_color("#F18F01")

    ax.set_xlabel("Mitochondrial Scaling Factor (× original)")
    ax.set_ylabel("Detection Status (0=Miss, 1=Detect)")
    ax.set_title("A. Lysis Detection Sensitivity", fontweight="bold", loc="left")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["MISS", "DETECT"])
    ax.legend(loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    sns.despine(ax=ax, right=False)

    # Panel B: Doublet sensitivity
    ax = axes[1]

    ax.plot(
        doublets_df["doublet_rate"] * 100,
        doublets_df["baseline_detected"].astype(int),
        marker="o",
        linewidth=2,
        markersize=8,
        label="Baseline (Rule-based)",
        color="#2E86AB",
    )

    if (
        "claude_detected" in doublets_df.columns
        and doublets_df["claude_detected"].any()
    ):
        ax.plot(
            doublets_df["doublet_rate"] * 100,
            doublets_df["claude_detected"].astype(int),
            marker="s",
            linewidth=2,
            markersize=8,
            label="Claude (LLM)",
            color="#A23B72",
        )

    # Overlay counts ratio on secondary axis
    ax2 = ax.twinx()
    ax2.plot(
        doublets_df["doublet_rate"] * 100,
        doublets_df["counts_ratio"],
        linestyle="--",
        color="#F18F01",
        alpha=0.7,
        linewidth=1.5,
        label="Counts ratio (doublets/singles)",
    )
    ax2.set_ylabel("Median UMI Counts Ratio", color="#F18F01")
    ax2.tick_params(axis="y", labelcolor="#F18F01")
    ax2.spines["right"].set_color("#F18F01")
    ax2.axhline(y=1.5, color="#F18F01", linestyle=":", alpha=0.5, linewidth=1)

    ax.set_xlabel("Doublet Rate (%)")
    ax.set_ylabel("Detection Status (0=Miss, 1=Detect)")
    ax.set_title("B. Doublet Detection Sensitivity", fontweight="bold", loc="left")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["MISS", "DETECT"])
    ax.legend(loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    sns.despine(ax=ax, right=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved sensitivity curves: {output_path}")


def plot_distribution_divergence(
    adata_clean: AnnData,
    adata_lysis: AnnData,
    adata_doublets: AnnData,
    baseline_mt_threshold: float,
    baseline_doublet_threshold: float,
    output_path: Path,
):
    """
    Generate distribution divergence plots with rule overlays.

    Parameters
    ----------
    adata_clean : AnnData
        Clean control dataset
    adata_lysis : AnnData
        Lysis-injected dataset
    adata_doublets : AnnData
        Doublet-injected dataset
    baseline_mt_threshold : float
        Baseline MT% threshold
    baseline_doublet_threshold : float
        Baseline doublet count threshold
    output_path : Path
        Where to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: MT% distribution (violin + KDE)
    ax = axes[0, 0]

    # Prepare data for violin plot
    clean_mt = adata_clean.obs["pct_counts_mt"].values
    lysis_mt = adata_lysis.obs["pct_counts_mt"].values

    data = pd.DataFrame(
        {
            "pct_mt": np.concatenate([clean_mt, lysis_mt]),
            "Condition": ["Clean"] * len(clean_mt) + ["Lysis"] * len(lysis_mt),
        }
    )

    sns.violinplot(
        data=data, x="Condition", y="pct_mt", palette=["#95B8D1", "#E07A5F"], ax=ax
    )

    # Overlay threshold
    ax.axhline(
        y=baseline_mt_threshold,
        color="#C73E1D",
        linestyle="--",
        linewidth=2,
        label=f"Baseline threshold ({baseline_mt_threshold:.1f}%)",
    )

    # Overlay percentiles
    clean_p95 = np.percentile(clean_mt, 95)
    lysis_p95 = np.percentile(lysis_mt, 95)
    ax.axhline(
        y=clean_p95, color="#2E86AB", linestyle=":", alpha=0.6, linewidth=1.5
    )
    ax.axhline(
        y=lysis_p95, color="#A23B72", linestyle=":", alpha=0.6, linewidth=1.5
    )

    ax.set_ylabel("Mitochondrial Read %")
    ax.set_title(
        "A. MT% Distribution: Clean vs Lysis", fontweight="bold", loc="left"
    )
    ax.legend(loc="upper right", fontsize=8)
    sns.despine(ax=ax)

    # Panel B: Total counts distribution (violin + KDE)
    ax = axes[0, 1]

    # Prepare data
    clean_counts = adata_clean.obs["total_counts"].values
    doublets_counts = adata_doublets.obs["total_counts"].values

    data = pd.DataFrame(
        {
            "total_counts": np.concatenate([clean_counts, doublets_counts]),
            "Condition": ["Clean"] * len(clean_counts)
            + ["Doublets"] * len(doublets_counts),
        }
    )

    sns.violinplot(
        data=data,
        x="Condition",
        y="total_counts",
        palette=["#95B8D1", "#F18F01"],
        ax=ax,
    )

    # Overlay threshold
    ax.axhline(
        y=baseline_doublet_threshold,
        color="#C73E1D",
        linestyle="--",
        linewidth=2,
        label=f"Baseline threshold ({baseline_doublet_threshold:.0f} counts)",
    )

    ax.set_ylabel("Total UMI Counts")
    ax.set_title(
        "B. UMI Counts: Clean vs Doublets", fontweight="bold", loc="left"
    )
    ax.legend(loc="upper right", fontsize=8)
    sns.despine(ax=ax)

    # Panel C: MT% KDE comparison
    ax = axes[1, 0]

    sns.kdeplot(clean_mt, shade=True, color="#95B8D1", label="Clean", ax=ax, alpha=0.6)
    sns.kdeplot(
        lysis_mt, shade=True, color="#E07A5F", label="Lysis", ax=ax, alpha=0.6
    )

    ax.axvline(
        x=baseline_mt_threshold,
        color="#C73E1D",
        linestyle="--",
        linewidth=2,
        label="Threshold",
    )
    ax.axvline(x=clean_p95, color="#2E86AB", linestyle=":", alpha=0.6, linewidth=1.5)
    ax.axvline(x=lysis_p95, color="#A23B72", linestyle=":", alpha=0.6, linewidth=1.5)

    ax.set_xlabel("Mitochondrial Read %")
    ax.set_ylabel("Density")
    ax.set_title("C. MT% Kernel Density", fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    sns.despine(ax=ax)

    # Panel D: Counts KDE comparison
    ax = axes[1, 1]

    sns.kdeplot(
        clean_counts, shade=True, color="#95B8D1", label="Clean", ax=ax, alpha=0.6
    )
    sns.kdeplot(
        doublets_counts,
        shade=True,
        color="#F18F01",
        label="Doublets",
        ax=ax,
        alpha=0.6,
    )

    ax.axvline(
        x=baseline_doublet_threshold,
        color="#C73E1D",
        linestyle="--",
        linewidth=2,
        label="Threshold",
    )

    ax.set_xlabel("Total UMI Counts")
    ax.set_ylabel("Density")
    ax.set_title("D. UMI Counts Kernel Density", fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved distribution divergence: {output_path}")


def plot_biological_hallucination_heatmap(
    adata_clean: AnnData,
    adata_doublets: AnnData,
    baseline_report_doublets,
    claude_report_doublets,
    output_path: Path,
):
    """
    Generate marker gene heatmap highlighting synthetic doublet artifacts.

    Parameters
    ----------
    adata_clean : AnnData
        Clean control with marker genes
    adata_doublets : AnnData
        Doublet-injected dataset
    baseline_report_doublets : AuditReport
        Baseline audit report on doublets
    claude_report_doublets : AuditReport or None
        Claude audit report on doublets (or None if unavailable)
    output_path : Path
        Where to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Marker gene expression heatmap
    ax = axes[0]

    # Get top marker genes per cluster from clean data
    if "rank_genes_groups" not in adata_clean.uns:
        print("    ⚠️  No marker genes found in clean data, skipping heatmap")
        plt.close()
        return

    # Extract top 5 markers per cluster
    marker_genes = []
    n_clusters = len(adata_clean.obs["leiden"].unique())
    for cluster_id in range(min(3, n_clusters)):  # Limit to 3 clusters for clarity
        cluster_markers = adata_clean.uns["rank_genes_groups"]["names"][str(cluster_id)][
            :5
        ]
        marker_genes.extend(cluster_markers)

    # Remove duplicates while preserving order
    marker_genes = list(dict.fromkeys(marker_genes))[:15]  # Max 15 genes

    # Compute mean expression per cluster
    cluster_ids = sorted(adata_clean.obs["leiden"].unique())[:5]  # Max 5 clusters
    expression_matrix = np.zeros((len(marker_genes), len(cluster_ids)))

    for i, gene in enumerate(marker_genes):
        if gene in adata_clean.var_names:
            gene_idx = adata_clean.var_names.get_loc(gene)
            for j, cluster in enumerate(cluster_ids):
                cluster_mask = (adata_clean.obs["leiden"] == cluster).values  # Convert to numpy array
                gene_expr = adata_clean.X[cluster_mask, gene_idx]
                expression_matrix[i, j] = np.mean(
                    gene_expr.toarray() if hasattr(gene_expr, "toarray") else gene_expr
                )

    # Plot heatmap
    sns.heatmap(
        expression_matrix,
        xticklabels=[f"C{c}" for c in cluster_ids],
        yticklabels=marker_genes,
        cmap="RdYlBu_r",
        center=0,
        cbar_kws={"label": "Mean Expression"},
        ax=ax,
    )

    ax.set_title(
        "A. Top Marker Genes Across Clusters", fontweight="bold", loc="left"
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Marker Gene")

    # Panel B: Interpretation summary (text annotations)
    ax = axes[1]
    ax.axis("off")

    # Ground truth
    doublet_mask = adata_doublets.obs.get("is_synthetic_doublet", False) == True
    n_doublets = doublet_mask.sum() if doublet_mask.any() else 0

    # Baseline interpretation
    baseline_detected = any(
        f.flag_type == "POTENTIAL_DOUBLETS" for f in baseline_report_doublets.flags
    )
    baseline_verdict = baseline_report_doublets.summary_verdict

    # Claude interpretation
    claude_detected = False
    claude_verdict = "N/A"
    claude_speculation = "N/A"
    if claude_report_doublets:
        claude_detected = any(
            f.flag_type == "POTENTIAL_DOUBLETS" for f in claude_report_doublets.flags
        )
        claude_verdict = claude_report_doublets.summary_verdict

        # Check for biological speculation
        doublet_flags = [
            f
            for f in claude_report_doublets.flags
            if f.flag_type == "POTENTIAL_DOUBLETS"
        ]
        if doublet_flags:
            warning = doublet_flags[0].biological_confounder_warning
            # Check if proper disclaimer exists
            has_disclaimer = any(
                phrase in warning.lower()
                for phrase in ["not", "don't", "do not", "must not", "technical"]
            )
            claude_speculation = "PROPER DISCLAIMER" if has_disclaimer else "⚠️ SPECULATION RISK"

    # Create annotation text box
    annotation_text = f"""
GROUND TRUTH:
  • Synthetic doublets: {n_doublets} cells
  • Mechanism: In-silico co-encapsulation artifact
  • Expected: Additive transcriptomes (CD3D + CD19)

BASELINE AUDITOR (Rule-based):
  • Detection: {'✓ DETECTED' if baseline_detected else '✗ MISSED'}
  • Verdict: {baseline_verdict}
  • Method: UMI count threshold (1.5× median)

CLAUDE AUDITOR (LLM):
  • Detection: {'✓ DETECTED' if claude_detected else '✗ MISSED'}
  • Verdict: {claude_verdict}
  • Speculation check: {claude_speculation}
"""

    ax.text(
        0.1,
        0.5,
        annotation_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round,pad=1", facecolor="#F5F5F5", edgecolor="#333333"),
    )

    ax.set_title(
        "B. Doublet Interpretation Summary", fontweight="bold", loc="left"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved biological hallucination heatmap: {output_path}")
