# PathogenScope

FlashPPI-based microbial target discovery pipeline for anti-infective drug development.

## Pipeline

1. **FlashPPI** — Proteome-scale PPI prediction using [TattaBio/flashppi](https://github.com/TattaBio/flashppi)
2. **Community Detection** — Louvain clustering to identify functional protein modules
3. **Annotation & Scoring** — UniProt annotations + target prioritization scoring

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Download E. coli K12 proteome
python scripts/download_ecoli.py

# Run full pipeline
python scripts/run_pipeline.py --fasta data/ecoli_k12/ecoli_k12_proteome.fasta

# Skip FlashPPI if you already have predictions
python scripts/run_pipeline.py --fasta data/ecoli_k12/ecoli_k12_proteome.fasta --skip_flashppi

# Skip UniProt annotations (offline mode)
python scripts/run_pipeline.py --fasta data/ecoli_k12/ecoli_k12_proteome.fasta --no_annotations
```

## Output

| File | Contents |
|---|---|
| `results/ppi_predictions.csv` | Predicted protein pairs + contact scores |
| `results/community_stats.csv` | Per-protein community assignment, degree, centrality |
| `results/target_scores.csv` | Ranked target candidates with annotations |

## Project Structure

```
pathogenscope/
├── flashppi/        # FlashPPI inference wrapper
├── community/       # Louvain community detection
├── annotation/      # UniProt annotation + target scoring
└── agent/           # Downstream analysis agents (WIP)
scripts/
├── download_ecoli.py
└── run_pipeline.py
data/
└── ecoli_k12/
```
