# PathogenScope — Session Handoff

## What This Project Is
A **FlashPPI-based microbial drug target discovery pipeline**. It takes a bacterial proteome (FASTA), predicts all pairwise protein-protein interactions using TattaBio's FlashPPI model, clusters them into functional communities, and scores proteins as drug target candidates.

**Goal**: Identify high-value anti-infective drug targets in pathogen proteomes by combining PPI network topology with functional annotations.

## Current State (as of 2026-03-08)

### What's Done and Working
- **Full 3-stage FlashPPI pipeline** (`pathogenscope/flashppi/predict.py`) — embed → FAISS retrieve → contact score
- **Louvain community detection** (`pathogenscope/community/detect.py`) — builds PPI graph, clusters, computes centrality stats
- **UniProt annotation + target scoring** (`pathogenscope/annotation/annotate.py`) — fetches annotations, produces ranked target list
- **End-to-end runner script** (`scripts/run_pipeline.py`) — orchestrates all 3 steps
- **Validated on 20-protein M. genitalium test set** — RuvA-RuvB interaction correctly detected at score 0.73 (known biological interaction)

### Test Run Results (20 proteins, threshold 0.3)
```
Interactions found: 3
  RuvB ↔ RuvA:   0.73 (true positive — known Holliday junction complex)
  RuvB ↔ DPO3X:  0.32
  PcrA ↔ DPO3X:  0.30
Communities: 2 across 4 proteins
Top target: RuvB (target_score 0.92)
```

### What's NOT Done Yet
1. **Full 483-protein M. genitalium run** — started but m_genitalium_full/ is empty. Needs GPU (too slow on CPU, ~22K candidate pairs)
2. **Agent module** (`pathogenscope/agent/`) — stub only, empty `__init__.py`. Intended for downstream analysis agents (target nomination, druggability scoring, literature synthesis)
3. **Cross-proteome support** — host-pathogen PPI prediction (uses FlashPPI cross-proteome mode). A teammate's task.
4. **SeqHub comparison** — user is uploading same FASTA to seqhub.org for validation

## Architecture

```
pathogenscope/
├── __init__.py
├── flashppi/
│   ├── __init__.py
│   └── predict.py          # FlashPPIPredictor class — 3-stage pipeline
├── community/
│   ├── __init__.py
│   └── detect.py           # build_graph, detect_communities, community_stats, label_communities
├── annotation/
│   ├── __init__.py
│   └── annotate.py         # fetch_uniprot_batch, extract_accessions, score_targets
├── agent/
│   └── __init__.py          # EMPTY — TODO
scripts/
├── download_ecoli.py        # Downloads E. coli K12 proteome from UniProt
└── run_pipeline.py          # End-to-end CLI runner
data/                        # gitignored
└── m_genitalium/
    ├── m_genitalium_proteome.fasta   # Full 483-protein proteome
    └── m_genitalium_test20.fasta     # 20-protein test subset
results/                     # gitignored
├── test_run/                # 20-protein run (has data)
│   ├── ppi_predictions.csv
│   ├── community_stats.csv
│   └── target_scores.csv
└── m_genitalium_full/       # EMPTY — full run not completed yet
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Download M. genitalium proteome if not present (483 proteins, human STI pathogen)
python -c "
import requests, os
os.makedirs('data/m_genitalium', exist_ok=True)
url = 'https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=(organism_id:243273)+AND+(reviewed:true)'
resp = requests.get(url, timeout=120)
with open('data/m_genitalium/m_genitalium_proteome.fasta', 'w') as f:
    f.write(resp.text)
print(f'Downloaded {resp.text.count(chr(62))} proteins')
"

# Quick test (20 proteins, CPU OK, ~2 min)
PYTHONPATH=. python scripts/run_pipeline.py \
    --fasta data/m_genitalium/m_genitalium_test20.fasta \
    --output_dir results/test_run \
    --threshold 0.3 --no_annotations

# Full pipeline (needs GPU)
PYTHONPATH=. python scripts/run_pipeline.py \
    --fasta data/m_genitalium/m_genitalium_proteome.fasta \
    --output_dir results/m_genitalium_full \
    --threshold 0.5

# Skip annotations for faster run
PYTHONPATH=. python scripts/run_pipeline.py \
    --fasta data/m_genitalium/m_genitalium_proteome.fasta \
    --output_dir results/m_genitalium_full \
    --threshold 0.5 --no_annotations
```

## FlashPPI Model — Critical API Details

**Model**: `tattabio/flashppi` on HuggingFace (0.7B params, CC BY-NC 4.0)

**WARNING: This is NOT a standard HuggingFace model.** It has a custom API:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tattabio/flashppi", trust_remote_code=True)
model = AutoModel.from_pretrained("tattabio/flashppi", trust_remote_code=True)

# Per-residue tokenizer (each amino acid = 1 token)
tokens = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)

# Key methods:
residue_emb = model.encode_protein(input_ids, attention_mask)  # → (B, L, 1280)
clip_q = model.head_q(residue_emb, mask)                       # → (B, 1024) query embedding
clip_k = model.head_k(residue_emb, mask)                       # → (B, 1024) key embedding
contact_map, contact_mask = model.predict_contacts(emb1, emb2, mask1, mask2)
# contact_map needs sigmoid! contact_mask is boolean
# contact_score = max of sigmoid(contact_map) masked by contact_mask

# Or all-in-one:
output = model.forward(ids1, ids2, mask1, mask2)  # → FlashPPIOutput
# output.contact_map, output.contact_score, output.clip_embed1/2, output.clip_score
```

**Thresholds**:
- Contact score > 0.61 ≈ 70% precision (from E. coli benchmark in paper)
- Default: 0.5 for intra-proteome, 0.4 for cross-proteome
- Test run used 0.3 (lower to see more interactions with small protein set)

## Key Technical Decisions Already Made
- **networkx.algorithms.community.louvain_communities** — NOT python-louvain (has C build issues)
- **FAISS IndexFlatIP** for stage 2 nearest-neighbor retrieval
- **Single-pass embedding**: `encode_protein` → `head_q` gives both residue and CLIP embeddings in one forward pass
- **Contact maps NOT saved to disk** — only max score stored in CSV (saves disk space)
- **Sequences truncated to 1024 residues** (model max_len)
- **Deduplication**: pairs are deduplicated keeping max contact score

## Pipeline Output Format
- `ppi_predictions.csv`: columns = `query_id, match_id, contact_score`
- `community_stats.csv`: columns = `protein_id, ppi_community, ppi_edges, ppi_degree, ppi_min_contact_score, ppi_max_contact_score, ppi_mean_contact_score, betweenness_centrality`
- `target_scores.csv`: above + `centrality_score, hub_score, confidence_score, target_score` (+ UniProt annotation columns if fetched)

## Target Scoring Formula
```
target_score = 0.4 * centrality_score + 0.3 * hub_score + 0.3 * confidence_score
```
Where:
- `centrality_score` = betweenness_centrality / max(betweenness_centrality)
- `hub_score` = ppi_edges / max(ppi_edges)
- `confidence_score` = ppi_max_contact_score

## Next Steps (Priority Order)
1. **Run full M. genitalium on GPU** — 483 proteins, ~22K candidate pairs after FAISS retrieval
2. **Compare results with SeqHub** — user uploading same FASTA to seqhub.org for cross-validation
3. **Build agent module** — downstream analysis agents:
   - Target nomination from PPI communities
   - Druggability/tractability scoring
   - Host-pathogen cross-proteome analysis (teammate's task)
   - Literature/MoA synthesis
4. **Add cross-proteome support** — host-pathogen PPI using FlashPPI's cross-proteome mode
5. **Teammate's task**: Human-bacteria PPI (separate pipeline, uses cross-proteome mode)

## External Resources
- FlashPPI paper: https://www.biorxiv.org/content/10.64898/2026.03.01.708874v1
- FlashPPI GitHub: https://github.com/TattaBio/flashppi
- FlashPPI HuggingFace: https://huggingface.co/tattabio/flashppi
- SeqHub: https://seqhub.org
- M. tuberculosis on SeqHub: https://seqhub.org/tattabio/mycobacterium_tb?ppi=true

## Git Info
- **Branch**: `carlagent` (tracks `claude/carlagent-Yyxyx` on remote)
- **Repo**: `CarlisleMa/pathogenscope`
