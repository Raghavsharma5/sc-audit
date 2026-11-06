#!/usr/bin/env python3
"""
Export processed PBMC3k cells to JavaScript format for 3D visualizer.

Usage: python export_cells_json.py
Output: cells_data.js
"""

import scanpy as sc
from pathlib import Path
import json

def load_processed_pbmc3k():
    """Load processed PBMC3k dataset with clustering and QC metrics."""
    print("Loading PBMC3k dataset...")

    # Try loading from local cache first, otherwise download
    cache_path = Path("data/pbmc3k_processed.h5ad")

    if cache_path.exists():
        print(f"  Loading from {cache_path}")
        adata = sc.read_h5ad(cache_path)
    else:
        print("  Downloading and processing from Scanpy...")
        # Load raw data
        adata = sc.datasets.pbmc3k()

        # Run standard QC pipeline
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        # Filter
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

        # Normalize and log
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # HVG and PCA
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
        sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

        # Clustering
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.leiden(adata, resolution=0.5)
        sc.tl.umap(adata)

        # Marker genes
        sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')

        # Save for future use
        cache_path.parent.mkdir(exist_ok=True)
        adata.write(cache_path)
        print(f"  Saved to {cache_path}")

    print(f"  Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    return adata

def compute_3d_coordinates(adata):
    """Get or compute 3D spatial coordinates for visualization."""
    print("Computing 3D coordinates...")

    # Check if 3D UMAP exists
    if 'X_umap' in adata.obsm and adata.obsm['X_umap'].shape[1] == 3:
        print("  Using existing 3D UMAP")
        coords = adata.obsm['X_umap']

    # Check if 2D UMAP exists, compute 3D
    elif 'X_umap' in adata.obsm and adata.obsm['X_umap'].shape[1] == 2:
        print("  Computing 3D UMAP...")
        sc.tl.umap(adata, n_components=3)
        coords = adata.obsm['X_umap']

    # Use top 3 PCA dimensions
    elif 'X_pca' in adata.obsm:
        print("  Using top 3 PCA dimensions")
        coords = adata.obsm['X_pca'][:, :3]

    else:
        raise ValueError("No suitable coordinates found. Run PCA first.")

    # Scale coordinates to reasonable range for Three.js (-30 to 30)
    coords_scaled = coords.copy()
    for i in range(3):
        min_val = coords[:, i].min()
        max_val = coords[:, i].max()
        coords_scaled[:, i] = (coords[:, i] - min_val) / (max_val - min_val) * 60 - 30

    print(f"  Coordinates scaled to range: [{coords_scaled.min():.1f}, {coords_scaled.max():.1f}]")
    return coords_scaled

def get_marker_expression(adata, gene, cell_idx):
    """Get expression level for a specific gene and cell."""
    if gene not in adata.raw.var_names:
        return 0.0

    gene_idx = adata.raw.var_names.get_loc(gene)

    # Get raw counts
    value = adata.raw.X[cell_idx, gene_idx]

    # Handle sparse matrix
    if hasattr(value, 'toarray'):
        expr = value.toarray()[0, 0]
    elif hasattr(value, 'item'):
        expr = value.item()
    else:
        expr = value

    return float(expr)

def classify_expression_level(value):
    """Convert expression value to categorical level."""
    if value == 0:
        return "None"
    elif value < 1.0:
        return "Low"
    elif value < 3.0:
        return "Medium"
    else:
        return "High"

def get_cluster_annotation(cluster_id):
    """Map Leiden cluster to cell type based on standard PBMC3k annotations."""
    # Standard PBMC3k cluster annotations
    annotations = {
        '0': 'CD4+ T-cells',
        '1': 'CD14+ Monocytes',
        '2': 'CD8+ T-cells',
        '3': 'B-cells',
        '4': 'NK cells',
        '5': 'FCGR3A+ Monocytes',
        '6': 'Dendritic cells',
        '7': 'Megakaryocytes'
    }
    return annotations.get(str(cluster_id), f'Cluster {cluster_id}')

def determine_qc_status(mt_pct, genes_detected, umi_counts):
    """Determine QC status and artifact type based on metrics."""
    artifact_type = None
    qc_status = 'clean'

    # Lysis detection (high MT%)
    if mt_pct > 5.0:
        artifact_type = 'lysis'
        qc_status = 'critical'

    # Degradation detection (low complexity)
    elif genes_detected < 400:
        artifact_type = 'degraded'
        qc_status = 'warning'

    # Doublet detection (very high UMI)
    elif umi_counts > 10000:  # 2.5× typical median of ~4000
        artifact_type = 'doublet'
        qc_status = 'warning'

    return qc_status, artifact_type

def generate_qc_note(cell_data):
    """Generate plain-English QC explanation."""
    mt = cell_data['mtPct']
    genes = cell_data['genesDetected']
    umi = cell_data['umiCounts']
    artifact = cell_data['artifactType']

    if artifact == 'lysis':
        return f"HIGH MT%: {mt:.1f}% indicates membrane damage during sample prep. Cytoplasm leaked while mitochondria remain trapped. Technical artifact, not a 'hyperactive metabolic state.' Should be filtered."

    elif artifact == 'degraded':
        return f"LOW COMPLEXITY: Only {genes} genes detected (threshold: 400). Indicates RNA degradation, ambient contamination, or empty droplet. Not a 'quiescent stem cell' - likely technical failure."

    elif artifact == 'doublet':
        return f"POTENTIAL DOUBLET: Very high UMI count ({umi:,}) suggests multi-cell co-encapsulation. Two cells captured in one droplet, creating merged transcriptome. Technical artifact, not 'cell fusion.'"

    else:
        return f"Cell passes QC checks. Normal UMI ({umi:,}), low MT% ({mt:.1f}%), good complexity ({genes} genes). High-quality cell suitable for downstream analysis."

def export_cells_data(adata, output_path="cells_data.js"):
    """Export cell data to JavaScript-embeddable format."""
    print(f"\nExporting cell data to {output_path}...")

    # Get 3D coordinates
    coords_3d = compute_3d_coordinates(adata)

    # Get cluster assignments
    if 'leiden' in adata.obs.columns:
        cluster_key = 'leiden'
    elif 'louvain' in adata.obs.columns:
        cluster_key = 'louvain'
    else:
        raise ValueError("No cluster assignments found (leiden or louvain)")

    # Marker genes to extract
    marker_genes = ['CD3D', 'FTL', 'RPS12', 'MT-ND1']

    cells_data = []

    print("Processing cells...")
    for i in range(adata.n_obs):
        if i % 500 == 0:
            print(f"  {i}/{adata.n_obs}...")

        barcode = adata.obs_names[i]
        cluster_id = str(adata.obs[cluster_key].iloc[i])
        cluster_name = get_cluster_annotation(cluster_id)

        # Get QC metrics
        umi_counts = int(adata.obs['total_counts'].iloc[i])
        genes_detected = int(adata.obs['n_genes_by_counts'].iloc[i])
        mt_pct = float(adata.obs['pct_counts_mt'].iloc[i])

        # Get marker expression
        markers = {}
        for gene in marker_genes:
            expr = get_marker_expression(adata, gene, i)
            markers[gene] = {
                'value': float(round(expr, 2)),
                'level': str(classify_expression_level(expr))
            }

        # Determine QC status
        qc_status, artifact_type = determine_qc_status(mt_pct, genes_detected, umi_counts)

        cell_obj = {
            'id': str(barcode),
            'clusterId': int(cluster_id),
            'clusterName': str(cluster_name),
            'x': float(round(coords_3d[i, 0], 2)),
            'y': float(round(coords_3d[i, 1], 2)),
            'z': float(round(coords_3d[i, 2], 2)),
            'umiCounts': int(umi_counts),
            'genesDetected': int(genes_detected),
            'mtPct': float(round(mt_pct, 2)),
            'markers': markers,
            'qcStatus': str(qc_status),
            'artifactType': str(artifact_type) if artifact_type else None
        }

        # Add QC note
        cell_obj['qcNote'] = generate_qc_note(cell_obj)

        cells_data.append(cell_obj)

    # Write as JavaScript variable
    with open(output_path, 'w') as f:
        f.write("// PBMC3k cell data\n")
        f.write("// Generated by export_cells_json.py\n\n")
        f.write("window.REAL_CELL_DATA = ")
        json.dump(cells_data, f, indent=2)
        f.write(";\n\n")
        f.write(f"console.log('Loaded {len(cells_data)} PBMC3k cells');\n")

    print(f"✓ Exported {len(cells_data)} cells to {output_path}")

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"  Total cells: {len(cells_data)}")
    print(f"  Clusters: {len(set(c['clusterId'] for c in cells_data))}")
    print(f"  Clean cells: {sum(1 for c in cells_data if c['qcStatus'] == 'clean')}")
    print(f"  Lysis artifacts: {sum(1 for c in cells_data if c['artifactType'] == 'lysis')}")
    print(f"  Doublet candidates: {sum(1 for c in cells_data if c['artifactType'] == 'doublet')}")
    print(f"  Degraded cells: {sum(1 for c in cells_data if c['artifactType'] == 'degraded')}")
    print(f"\n  UMI range: {min(c['umiCounts'] for c in cells_data):,} - {max(c['umiCounts'] for c in cells_data):,}")
    print(f"  Genes range: {min(c['genesDetected'] for c in cells_data)} - {max(c['genesDetected'] for c in cells_data)}")
    print(f"  MT% range: {min(c['mtPct'] for c in cells_data):.1f}% - {max(c['mtPct'] for c in cells_data):.1f}%")

    return cells_data

def main():
    """Main export pipeline."""
    print("=" * 60)
    print("PBMC3k Cell Data Export")
    print("=" * 60)

    adata = load_processed_pbmc3k()
    export_cells_data(adata, "cells_data.js")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print("\nNext: open sc_audit_3d_visualizer.html in your browser")

if __name__ == "__main__":
    main()
