#!/bin/bash
# Quick test script for sc-audit 3D visualizer

echo "=================================="
echo "sc-audit Visualizer Setup Test"
echo "=================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "[0/4] Activating virtual environment..."
    source .venv/bin/activate
    echo "  ✓ Virtual environment activated"
    echo ""
fi

# Check Python dependencies
echo "[1/4] Checking Python dependencies..."
python3 -c "import scanpy, anndata, numpy, pandas" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ All Python packages installed"
else
    echo "  ✗ Missing Python packages. Install with:"
    echo "    pip install scanpy anndata numpy pandas"
    exit 1
fi

# Run export script
echo ""
echo "[2/4] Running export script..."
python3 export_cells_json.py
if [ $? -eq 0 ]; then
    echo "  ✓ Export completed"
else
    echo "  ✗ Export failed"
    exit 1
fi

# Check output file
echo ""
echo "[3/4] Checking output file..."
if [ -f "cells_data.js" ]; then
    SIZE=$(du -h cells_data.js | cut -f1)
    LINES=$(wc -l < cells_data.js)
    echo "  ✓ cells_data.js created ($SIZE, $LINES lines)"
else
    echo "  ✗ cells_data.js not found"
    exit 1
fi

# Check HTML file
echo ""
echo "[4/4] Checking HTML file..."
if [ -f "sc_audit_3d_visualizer.html" ]; then
    echo "  ✓ Visualizer HTML found"
else
    echo "  ✗ sc_audit_3d_visualizer.html not found"
    exit 1
fi

echo ""
echo "=================================="
echo "✓ All checks passed!"
echo "=================================="
echo ""
echo "To view the visualizer:"
echo "  open sc_audit_3d_visualizer.html"
echo ""
echo "Or in Chrome:"
echo "  /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome sc_audit_3d_visualizer.html"
