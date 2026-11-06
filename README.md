# sc-audit

A deterministic, schema-constrained audit harness for single-cell RNA-sequencing (scRNA-seq) workflows.

## Overview

`sc-audit` tests whether an LLM (Claude 3.5 Sonnet) can accurately identify technical quality-control pathologies (e.g., apoptotic lysis, synthetic doublets, degraded libraries) from structured AnnData summaries without inventing unsupported biological phenotypes.

## Core Principles

1. **Zero Autonomous Execution**: The LLM does not execute arbitrary shell/Python code and does not receive raw expression arrays
2. **Contract-First**: All communication between Scanpy and Claude is strictly governed by Pydantic v2 schemas
3. **Benchmarked Evaluation**: Performance is measured against a deterministic rule-based baseline over an in-silico perturbed ground-truth test suite

## Installation

### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate sc-audit
pip install -e .
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Configuration

### API Key (Optional)

An Anthropic API key is **only required** if you want to run the LLM auditor:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

**What requires an API key:**
- `--mode claude` (LLM auditor only)
- `--mode both` (baseline + LLM comparison)

**What does NOT require an API key:**
- `--mode baseline` (deterministic rule-based auditor) - **runs 100% locally**
- 3D visualizer (`./test_visualizer.sh` and `sc_audit_3d_visualizer.html`) - **runs 100% locally**

## Usage

### Audit a single dataset

```bash
python scripts/run_audit.py --input data/pbmc3k.h5ad --mode both
```

Modes:
- `baseline`: Run only the deterministic rule-based auditor
- `claude`: Run only the Claude LLM auditor
- `both`: Run both auditors and compare results

### Run full benchmark suite

```bash
python scripts/run_benchmark.py --output reports/benchmark_summary.md
```

This will:
1. Generate four test conditions (clean, lysis, doublets, degraded)
2. Run both auditors on all conditions
3. Calculate precision, recall, groundedness ratio, and speculation scores
4. Generate a comprehensive benchmark report

## Project Structure

```
sc-audit/
├── src/
│   ├── pipeline/        # Data ingestion and QC computation
│   ├── auditor/         # Baseline and Claude auditors
│   └── perturbation/    # Ground-truth pathology injection
├── eval/                # Benchmark harness and metrics
├── scripts/             # CLI entry points
├── tests/               # Unit and integration tests
├── data/                # Cached datasets
└── reports/             # Benchmark results and figures
```

## Testing

```bash
pytest tests/ -v
```

## Development

The project uses:
- `black` for code formatting
- `ruff` for linting
- `pytest` for testing

Run formatting and linting:

```bash
black src/ eval/ scripts/ tests/
ruff check src/ eval/ scripts/ tests/
```

## License

MIT
