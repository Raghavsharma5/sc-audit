# Quick Start Guide

## Setup (One-Time)

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Why?** Avoids system-wide package conflicts (PEP 668 on macOS/Homebrew Python).

### 2. Install Dependencies

```bash
pip install scanpy anndata numpy pandas
```

### 3. Export Cell Data

```bash
python export_cells_json.py
```

**Or run all steps at once:**
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install scanpy anndata numpy pandas && ./test_visualizer.sh
```

**Expected output:**
```
Loading PBMC3k dataset...
  Loaded 2,643 cells, 13,714 genes
Computing 3D coordinates...
  Using top 3 PCA dimensions
Processing cells...
✓ Exported 2,643 cells to cells_data.js

Dataset Summary:
  Total cells: 2,643
  Clusters: 6-8
  Clean cells: ~2,500
  Artifacts: ~100
```

### 4. Open Visualizer

```bash
open sc_audit_3d_visualizer.html
```

## What You'll See

### Real Dataset
- **2,600+ cells** from 10X PBMC3k
- **Real barcodes**: `AAACATACAACCAC-1`
- **Authentic metrics**: UMI counts, genes detected, MT%
- **6-8 clusters**: T-cells, Monocytes, B-cells, NK cells

### Interactive Features

**Hover**: Tooltip with cluster and metrics
**Click**: Detailed diagnostic panel
**Drag**: Rotate 3D view
**Scroll**: Zoom in/out
**Right-drag**: Pan camera

## View Modes

### 1. Cell Types (Default)
Colors by cell type cluster

### 2. Mitochondrial %
- Gradient: blue (low) → red (high)
- **Slider**: Adjust threshold 2-15%
- **Use**: Demonstrates threshold sensitivity

### 3. Doublets
- Highlights high-UMI cells
- Shows dispersion patterns

## Example Cell Data

Click any cell to see:

```
Cell ID: AAACATACAACCAC-1
Cluster: CD4+ T-cells
UMI: 2,419
Genes: 779
MT%: 3.02%

Markers:
  CD3D: High (4.52)
  FTL: Low (0.23)
  RPS12: Medium (1.45)
  MT-ND1: Low (0.89)

QC Status: Clean
```

## Controls Summary

| Action | Control |
|--------|---------|
| Rotate | Left-click + drag |
| Zoom | Scroll wheel |
| Pan | Right-click + drag |
| Select cell | Click |
| Pause rotation | Button in left panel |
| Reset camera | Button in left panel |
| Change view | View mode buttons |
| Adjust threshold | Slider (MT% view only) |

## Troubleshooting

### Issue: "externally-managed-environment" (PEP 668)
```bash
# Activate virtual environment first
source .venv/bin/activate

# Then install packages
pip install scanpy anndata numpy pandas
```

### Issue: "Cell data not found"
```bash
# Check file exists
ls cells_data.js

# Regenerate if missing
python export_cells_json.py
```

### Issue: Black screen
- Open browser console (F12)
- Check for JavaScript errors
- Verify Three.js CDN loaded

### Issue: Slow performance
- Close other tabs
- Use Chrome browser
- Pause auto-rotation

## File Checklist

Both files must be in the same directory:

- [ ] `sc_audit_3d_visualizer.html` (36 KB)
- [ ] `cells_data.js` (1.5-2 MB)

## Validation

Verify real data loaded:

1. **Barcodes**: Format `XXXXXXXXXXXXXXXX-1`
2. **UMI range**: 500-15,000 (varies)
3. **Cluster sizes**: Unequal distribution
4. **Legend**: Shows actual cell counts

## Next Steps

- Explore different clusters
- Test threshold slider in MT% view
- Compare clean vs. artifact cells
- Export data for your own datasets (modify `export_cells_json.py`)

## Support

For issues or questions, check:
- Browser console for errors
- Python script output for export problems
- README.md for detailed documentation
