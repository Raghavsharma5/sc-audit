# sc-audit 3D Visualizer

Interactive WebGL visualization of single-cell RNA-seq QC data.

## Files

- `sc_audit_3d_visualizer.html` - Main visualizer (36 KB)
- `export_cells_json.py` - Data export script
- `cells_data.js` - Generated cell data (created by export script)
- `VISUALIZER_README.md` - Complete documentation
- `VISUALIZER_QUICKSTART.md` - Quick start guide

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install scanpy anndata numpy pandas

# 3. Export data
python export_cells_json.py

# 4. Open visualizer
open sc_audit_3d_visualizer.html
```

**Quick setup (one command):**
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install scanpy anndata numpy pandas && ./test_visualizer.sh
```

## Usage

### Controls
- **Rotate**: Drag mouse
- **Zoom**: Scroll wheel
- **Select cell**: Click

### View Modes
1. **Cell Types** - Color by cluster
2. **Mitochondrial %** - QC threshold visualization
3. **Doublets** - High-UMI cell detection

### Features
- Real 10X PBMC3k data (2,600+ cells)
- Interactive threshold adjustment
- Detailed cell diagnostics
- Real-time QC metrics

## Documentation

See `VISUALIZER_README.md` for complete documentation.

See `VISUALIZER_QUICKSTART.md` for step-by-step guide.

## Requirements

- Python 3.9+
- Modern browser with WebGL support
- No web server required

**Note**: Uses virtual environment (`.venv`) to avoid system-wide package conflicts (PEP 668).

## License

MIT
