# PathogenScope — Project Reference

## What This Is

Pathogen genome → predicted host-pathogen protein interactions → druggable targets → AI-generated validation plans.

**Organism**: *Acinetobacter baumannii* (WHO #1 critical priority, ~3,800 proteins, reference proteome UP000006737).

## Team

| Person | Role |
|--------|------|
| **Joe** | Training cross-kingdom FlashPPI (NTv3) on Modal GPUs — predicts bacterial↔human PPIs directly |
| **Gabriel (me)** | Foldseek mimicry screen (Tamarind Bio) + FlashPPI intra-pathogen network (Modal) + all data infra + merging + eval + exports |
| **Joseph** | React dashboard + network visualization |
| **Zijian** | Claude (Anthropic) AI scientist agent |

## Sponsors — Must Demonstrate Usage

| Sponsor | What we use it for |
|---------|-------------------|
| **Tamarind Bio** | AlphaFold structure prediction (XDR novel proteins), Foldseek structural mimicry screen |
| **Modal** | FlashPPI intra-pathogen network (GPU), Joe's cross-kingdom model training (GPU), ESM2 embeddings |
| **Anthropic (Claude)** | AI scientist agent — PubMed search, drug repurposing, experiment proposals |

## How Outputs Combine

```
Joe's cross-kingdom FlashPPI (Modal) ──→ ┐
My Foldseek mimicry (Tamarind) ────────→ ├─→ MERGED RANKING → Claude Agent (Anthropic) → Dashboard
My FlashPPI pathogen hubs (Modal) ─────→ ┘
```

## Ground Truth for Validation

OmpA → TLR2/TLR4/fibronectin/DRP1, Ata → collagen IV/fibronectin. Blind rediscovery of these known interactions = demo money shot.

## Expected Outputs

| # | File | Description |
|---|------|-------------|
| 1 | `results/mimicry_hits.csv` | Foldseek mimicry hits (TM>0.5, seqid<30%) |
| 2 | `results/mimicry_with_partners.csv` | Mimicry hits + hijacked STRING interaction partners |
| 3 | `results/pathogen_network.csv` | FlashPPI intra-pathogen PPI network |
| 4 | `results/pathogen_hubs.csv` | Hub proteins ranked by degree + betweenness |
| 5 | `results/merged_targets.json` | Combined ranking (after Joe delivers his predictions) |
| 6 | `results/network_data.json` | Export for Joseph's React dashboard |

## Environment

- **Platform**: HPC cluster (Linux), not everything needs Modal
- **Python**: 3.12 with biopython, pandas, networkx, requests pre-installed
- **Tamarind API key**: Set in environment (`TAMARIND_API_KEY`)
- **Modal**: Needs `pip install modal && modal setup`
- **FlashPPI**: Need to clone https://github.com/TattaBio/FlashPPI

## Data Sources

- **AlphaFold DB structures**: `https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000006737_470.tar`
- **UniProt proteome FASTA**: `https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=(proteome:UP000006737)`
- **STRING DB**: REST API for interaction partners (species 9606 = human)
- **UniProt ID mapping**: REST API for gene names + organism filtering

## Pipeline Steps

### Step 1: Data Acquisition
- Download A. baumannii AlphaFold structures (reference strain)
- Download proteome FASTA
- Send FASTA to Joe immediately

### Step 2: Foldseek Mimicry Screen (Tamarind Bio)
Search A. baumannii structures against human proteome for structural mimicry.
- **Preferred**: Tamarind Bio API (`POST /submit-job` with type `foldseek`)
- **Fallback A**: Tamarind web UI at `https://app.tamarind.bio/tools/foldseek`
- **Fallback B**: Local Foldseek binary (less sponsor credit but works)

Tamarind API pattern:
1. `GET /tools` — discover available tools + settings
2. `PUT /upload/{filename}` — upload structure files
3. `POST /submit-job` — submit Foldseek job
4. `GET /jobs` — poll for completion
5. `POST /result` — download results

### Step 3: AlphaFold on Tamarind (if needed)
For XDR clinical isolate proteins not in AlphaFold DB — predict structures via Tamarind.

### Step 4: FlashPPI Intra-Pathogen Network (Modal)
Run FlashPPI on Modal GPU to get A. baumannii's internal PPI network. Identifies hub proteins — hubs that also interact with host are best drug targets.

### Step 5: Process Mimicry Results
1. Filter: TM-score > 0.5, sequence identity < 30% = structural mimicry
2. Map UniProt IDs → gene names via UniProt API
3. Filter to human-only targets
4. Look up STRING interaction partners (score > 700)
5. Validate against known interactions (OmpA→TLR2, etc.)

### Step 6: Build Pathogen Hub Scores
From FlashPPI output, build networkx graph, compute degree + betweenness centrality, rank hub proteins.

### Step 7: Merge with Joe's Cross-Kingdom Predictions (at 2-hour mark)
Composite score = mimicry_TM * 0.35 + flashppi_score * 0.35 + hub_degree * 0.30. Flag convergent hits (both mimicry + FlashPPI signals).

### Step 8: Export for Dashboard + Agent
Package everything into `results/network_data.json` for Joseph's React dashboard and `results/merged_targets.json` for Zijian's Claude agent.

## Handoffs

| To | File | When |
|----|------|------|
| **Joe** | `data/ab_proteome.fasta` | ASAP |
| **Joe** | `results/mimicry_hits.csv` | When ready (overlap check) |
| **Joseph** | `results/network_data.json` | After mimicry done (don't wait for Joe) |
| **Joseph** | Updated `network_data.json` | After merge |
| **Zijian** | `results/merged_targets.json` + `results/mimicry_with_partners.csv` | After merge |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Tamarind API auth fails | Check key at `https://app.tamarind.bio/settings`. Fallback: use their web UI manually. |
| Tamarind Foldseek settings unclear | Run `GET /tools` to list tools + settings. Or use web UI and download results. |
| Tamarind queue slow | Switch to local Foldseek. Still mention "we explored Tamarind for structure prediction." |
| Modal GPU quota | Use smaller GPU (`T4` or `A10G`). A. baumannii is small enough. |
| FlashPPI repo confusing | Read their README. If stuck, skip — mimicry + Joe's model is enough. Use STRING degree for hub scores. |
| Joe's output columns don't match | Rename to pathogen_id / human_protein / score — merge script handles the rest. |

## Sponsor Demo Talking Points

- "We used **Tamarind Bio** to run Foldseek structural similarity search and AlphaFold structure prediction at proteome scale"
- "**Modal** provided GPU compute for FlashPPI pathogen network analysis and our cross-kingdom interaction model training"
- "**Claude** acts as the AI scientist agent, searching PubMed, identifying druggable pathways, and proposing validation experiments"
