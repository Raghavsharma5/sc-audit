# sc-audit 3D Visualizer

Interactive 3D visualization of single-cell RNA-seq data with real-time QC analysis.

## Overview

This tool provides an interactive WebGL-based visualizer for exploring single-cell RNA-seq datasets in 3D space. It displays authentic cell measurements from Scanpy analysis including UMI counts, gene detection rates, mitochondrial percentages, and marker expression.

## Quick Start

### 1. Export Cell Data

```bash
python export_cells_json.py
```

This processes the PBMC3k dataset and generates `cells_data.js` containing:
- Real 10X cell barcodes
- 3D spatial coordinates (UMAP or PCA)
- QC metrics (UMI, genes, MT%)
- Marker gene expression
- Automated QC classification

### 2. Open Visualizer

```bash
open sc_audit_3d_visualizer.html
```

**Requirements:**
- `cells_data.js` must be in the same directory
- Modern browser with WebGL support
- No web server required (works on `file://` protocol)

## Features

### Interactive 3D Visualization

- **2,600+ cells** rendered as spheres in 3D space
- **Smooth 60fps** rotation and zooming
- **Hover tooltips** showing cluster and key metrics
- **Click to inspect** detailed cell information

### View Modes

**1. Cell Types (Default)**
- Colors by Leiden cluster assignment
- Shows standard PBMC cell types (T-cells, Monocytes, B-cells, NK cells, etc.)

**2. Mitochondrial % Heatmap**
- Colors cells by MT% (blue = low, red = high)
- **Interactive threshold slider** (2-15%)
- Demonstrates sensitivity of hard threshold cutoffs

**3. Doublet Detection**
- Highlights high-UMI cells
- Shows cluster dispersion patterns

### Cell Diagnostic Panel

Click any cell to view:
- Cell barcode (e.g., `AAACATACAACCAC-1`)
- Cluster assignment and cell type
- Total UMI counts
- Genes detected
- Mitochondrial percentage
- Marker expression (CD3D, FTL, RPS12, MT-ND1)
- QC status with technical explanation

## Data Pipeline

### Export Process

```
PBMC3k Raw Data
    ↓
Scanpy QC (filter < 200 genes, > 5% MT)
    ↓
Normalize (10,000 counts/cell, log transform)
    ↓
PCA (50 components)
    ↓
Leiden clustering (resolution 0.5)
    ↓
3D UMAP (n_components=3)
    ↓
Extract cell metrics
    ↓
cells_data.js
```

### Cell Data Structure

Each cell object contains:

```javascript
{
  "id": "AAACATACAACCAC-1",           // 10X barcode
  "clusterId": 0,                      // Leiden cluster
  "clusterName": "CD4+ T-cells",       // Cell type
  "x": -12.34, "y": 5.67, "z": -8.90,  // 3D coordinates
  "umiCounts": 2419,                   // Total UMI
  "genesDetected": 779,                // Number of genes
  "mtPct": 3.02,                       // Mitochondrial %
  "markers": {
    "CD3D": { "value": 4.52, "level": "High" },
    "FTL": { "value": 0.23, "level": "Low" },
    "RPS12": { "value": 1.45, "level": "Medium" },
    "MT-ND1": { "value": 0.89, "level": "Low" }
  },
  "qcStatus": "clean",                 // clean/warning/critical
  "artifactType": null,                // lysis/doublet/degraded
  "qcNote": "Cell passes all QC checks..."
}
```

## QC Classification

Cells are automatically classified based on metrics:

### Lysis (Critical)
- **Criteria**: Mitochondrial % > 5.0%
- **Visual**: Burnt orange color
- **Mechanism**: Membrane damage during dissociation

### Doublet (Warning)
- **Criteria**: UMI counts > 10,000 (≈2.5× median)
- **Visual**: Purple color, larger size
- **Mechanism**: Multi-cell co-encapsulation

### Degraded (Warning)
- **Criteria**: Genes detected < 400
- **Visual**: Grey color, smaller size
- **Mechanism**: RNA degradation or ambient contamination

## Controls

### Mouse/Trackpad
- **Rotate**: Left-click + drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click + drag
- **Select**: Click cell

### Buttons
- **Pause/Resume Rotation**: Toggle auto-rotation
- **Reset View**: Return to default camera position
- **View Mode Toggle**: Switch between Cell Types, MT%, Doublets

### Threshold Slider
- Available in Mitochondrial % view
- Adjust detection threshold in real-time
- Range: 2% - 15%

## Troubleshooting

### "Cell data not found" Error

**Cause**: Missing `cells_data.js` file

**Solution**:
```bash
# Check files exist together
ls sc_audit_3d_visualizer.html cells_data.js

# Re-run export if missing
python export_cells_json.py
```

### Blank/Black Screen

**Cause**: Three.js failed to load or JavaScript error

**Solution**: Open browser console (F12 or Cmd+Option+J) and check for errors

### Poor Performance

**Solutions**:
- Close unnecessary browser tabs
- Use Chrome for best WebGL performance
- Reduce browser zoom level
- Disable auto-rotation

## Technical Details

### Dependencies

**Python (export script)**:
- scanpy >= 1.9.0
- anndata >= 0.9.0
- numpy >= 1.24.0
- pandas >= 2.0.0

**Browser (visualizer)**:
- Three.js r128 (loaded from CDN)
- Modern browser with WebGL 1.0 support

### File Sizes
- `cells_data.js`: ~1.5-2.0 MB (2,600+ cells)
- `sc_audit_3d_visualizer.html`: ~36 KB

### Performance
- **Load Time**: 1-2 seconds
- **Frame Rate**: 60 fps (on modern hardware)
- **Memory Usage**: ~200 MB browser memory

## Dataset

**Source**: 10X Genomics PBMC 3k Dataset
- 2,700 peripheral blood mononuclear cells
- Sequenced on Illumina NextSeq 500
- Standard reference dataset for single-cell analysis

**Citation**: Zheng et al. (2017) "Massively parallel digital transcriptional profiling of single cells" Nature Communications 8, 14049

## License

MIT License

## Files

```
sc_audit_3d_visualizer.html    # Interactive visualizer
export_cells_json.py            # Data export script
cells_data.js                   # Generated cell data
data/pbmc3k_processed.h5ad      # Cached processed data
```
