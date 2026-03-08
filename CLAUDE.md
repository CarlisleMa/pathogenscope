# PathogenScope

## Project Overview
FlashPPI-based microbial target discovery pipeline for anti-infective drug development.
Uses TattaBio's FlashPPI model to predict proteome-scale protein-protein interactions,
then runs community detection and target scoring for drug target prioritization.

## Current State
- **Branch**: `carlagent` (tracks `claude/carlagent-Yyxyx` on remote)
- **Working pipeline**: Tested on 20 M. genitalium proteins — RuvA-RuvB interaction correctly detected (score 0.73)
- **Full 483-protein M. genitalium run**: Not yet completed (needs GPU — too slow on CPU)
- **Agent module**: Stub exists at `pathogenscope/agent/` — not yet implemented

## Architecture

```
pathogenscope/
├── flashppi/predict.py     # 3-stage FlashPPI wrapper (embed → FAISS retrieve → contact score)
├── community/detect.py     # Louvain clustering (networkx built-in) + community stats
├── annotation/annotate.py  # UniProt annotation fetch + target prioritization scoring
├── agent/                  # Empty — downstream analysis agents (TODO)
scripts/
├── download_ecoli.py       # Downloads E. coli K12 proteome from UniProt
└── run_pipeline.py         # End-to-end runner script
data/                       # gitignored — download locally
└── m_genitalium/
    ├── m_genitalium_proteome.fasta   # Full 483-protein proteome
    └── m_genitalium_test20.fasta     # 20-protein test subset
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Download M. genitalium proteome (483 proteins, human STI pathogen)
python -c "
import requests, os
os.makedirs('data/m_genitalium', exist_ok=True)
url = 'https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=(organism_id:243273)+AND+(reviewed:true)'
resp = requests.get(url, timeout=120)
with open('data/m_genitalium/m_genitalium_proteome.fasta', 'w') as f:
    f.write(resp.text)
print(f'Downloaded {resp.text.count(chr(62))} proteins')
"

# Run full pipeline (needs GPU for reasonable speed)
PYTHONPATH=. python scripts/run_pipeline.py \
    --fasta data/m_genitalium/m_genitalium_proteome.fasta \
    --output_dir results/m_genitalium_full \
    --threshold 0.5

# Or skip annotations for faster run
PYTHONPATH=. python scripts/run_pipeline.py \
    --fasta data/m_genitalium/m_genitalium_proteome.fasta \
    --output_dir results/m_genitalium_full \
    --threshold 0.5 --no_annotations
```

## FlashPPI Model Details
- **Model**: `tattabio/flashppi` on HuggingFace (0.7B params, CC BY-NC 4.0)
- **API**: Custom — NOT standard HuggingFace. Key methods:
  - `model.encode_protein(input_ids, attention_mask)` → residue embeddings (B, L, 1280)
  - `model.head_q(residue_embs, mask)` → CLIP query embedding (B, 1024)
  - `model.head_k(residue_embs, mask)` → CLIP key embedding (B, 1024)
  - `model.predict_contacts(emb1, emb2, mask1, mask2)` → (contact_map, contact_mask)
    - contact_map needs sigmoid applied; contact_mask is boolean
    - contact_score = max of sigmoid(contact_map) masked by contact_mask
  - `model.forward(ids1, ids2, mask1, mask2)` → FlashPPIOutput with contact_map, contact_score, clip_embed1/2, clip_score
- **Tokenizer**: Per-residue tokenizer (each amino acid = 1 token)
- **Contact score > 0.61** ≈ 70% precision (E. coli benchmark from paper)
- **Threshold default**: 0.5 for intra-proteome, 0.4 for cross-proteome

## Pipeline Output
- `results/ppi_predictions.csv`: query_id, match_id, contact_score
- `results/community_stats.csv`: protein_id, ppi_community, ppi_edges, betweenness_centrality, etc.
- `results/target_scores.csv`: ranked targets with combined score (centrality + hub + confidence)

## Key Technical Decisions
- Using `networkx.algorithms.community.louvain_communities` (not python-louvain — it has build issues)
- FAISS IndexFlatIP for nearest-neighbor retrieval in stage 2
- Single-pass embedding: encode_protein → head_q gives both residue and CLIP embeddings
- Contact maps are NOT saved to disk by default (only max score in CSV)
- Sequences truncated to 1024 residues (model max_len)

## Next Steps (TODO)
1. **Run full M. genitalium on GPU** — 483 proteins, ~22K candidate pairs
2. **Compare results with SeqHub** — user is uploading same FASTA to seqhub.org for validation
3. **Build agent module** — downstream analysis agents for:
   - Target nomination from PPI communities
   - Druggability/tractability scoring
   - Host-pathogen cross-proteome analysis (teammate's task)
   - Literature/MoA synthesis
4. **Add cross-proteome support** — wrapper around predict_cross_proteome pattern for host-pathogen PPI
5. **Teammate task**: Human-bacteria PPI (separate from this pipeline, uses cross-proteome mode)

## Relevant External Resources
- FlashPPI paper: https://www.biorxiv.org/content/10.64898/2026.03.01.708874v1
- FlashPPI GitHub: https://github.com/TattaBio/flashppi
- FlashPPI HuggingFace: https://huggingface.co/tattabio/flashppi
- SeqHub: https://seqhub.org
- M. tuberculosis on SeqHub: https://seqhub.org/tattabio/mycobacterium_tb?ppi=true

## Test Commands
```bash
# Quick test with 20 proteins (works on CPU, ~2 min)
PYTHONPATH=. python scripts/run_pipeline.py \
    --fasta data/m_genitalium/m_genitalium_test20.fasta \
    --output_dir results/test_run \
    --threshold 0.3 --no_annotations
```
