#!/usr/bin/env python3
"""Evaluate pipeline outputs against known virulence factors and interactions.

Ground truth sources:
  - VFDB SetA: experimentally verified A. baumannii virulence factors
  - PHI-base: pathogen-host interaction phenotypes
  - UniProt: virulence keyword-annotated proteins
  - IntAct: experimentally validated PPIs
"""

import csv
import gzip
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache" / "eval"

# Pipeline outputs
MIMICRY_HITS = RESULTS_DIR / "mimicry_hits.csv"
MIMICRY_WITH_PARTNERS = RESULTS_DIR / "mimicry_with_partners.csv"
HUB_SCORES = RESULTS_DIR / "pathogen_hubs.csv"
MERGED_TARGETS = RESULTS_DIR / "merged_targets.json"

# Known A. baumannii → human interactions from literature
# Curated from PubMed reviews on A. baumannii virulence mechanisms
KNOWN_AB_INTERACTIONS = {
    "OmpA": ["TLR2", "TLR4", "FN1", "DNM1L", "CASP3", "CASP9", "NFKB1"],
    "Ata": ["COL4A1", "COL4A2", "FN1", "LAMA1", "LAMB1"],
    "Bap": ["FN1"],
    "GroEL/chaperonin": ["TLR2", "TLR4", "HSPD1"],
    "DnaK/Hsp70": ["TLR2", "TLR4", "HSPA1A"],
    "DnaJ": ["DNAJA1", "DNAJB1"],
    "phospholipase": ["PLCG1", "PLA2G4A"],
    "siderophore": ["TFR1", "TFRC", "LTF"],
    "LPS": ["TLR4", "CD14", "LBP", "MD2"],
    "efflux": ["ABCB1", "ABCG2"],
    "porin": ["CASP3", "CASP9", "BAX", "BCL2"],
    "capsule": ["TLR4", "SIGLEC1"],
}

# Ground truth URLs
VFDB_SETA_URL = "https://www.mgc.ac.cn/VFs/Down/VFDB_setA_pro.fas.gz"
VFDB_SETB_URL = "https://www.mgc.ac.cn/VFs/Down/VFDB_setB_pro.fas.gz"
PHIBASE_URL = "https://raw.githubusercontent.com/PHI-base/data/master/releases/phi-base_current.csv"
UNIPROT_VF_URL = "https://rest.uniprot.org/uniprotkb/search?query=(taxonomy_id:470)%20AND%20(keyword:KW-0843)&format=tsv&fields=accession,gene_names,protein_names,keyword&size=500"
INTACT_URL = "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/species/Acinetobacter_baumannii__strain_AB0057_.txt"


# ── Ground truth loaders ─────────────────────────────────────────────────────

def download_cached(url: str, filename: str) -> Path | None:
    """Download file with caching. Returns None on failure."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"    Downloading {filename}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PathogenScope eval)"})
        with urllib.request.urlopen(req) as resp:
            path.write_bytes(resp.read())
        return path
    except Exception as e:
        print(f"    Download failed: {e}")
        return None


def load_vfdb() -> set[str]:
    """Load VFDB SetA gene names for A. baumannii."""
    path = download_cached(VFDB_SETA_URL, "VFDB_setA_pro.fas.gz")
    if not path:
        return set()

    import re
    genes = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith(">") or "Acinetobacter baumannii" not in line:
                continue
            parens = re.findall(r'\(([^)]+)\)', line)
            if len(parens) >= 2:
                gene = parens[1].strip()
                if gene and len(gene) < 20 and " " not in gene:
                    genes.add(gene)

    return genes


def load_vfdb_accessions_by_sequence() -> set[str]:
    """Match VFDB sequences to our proteome by sequence identity.
    Returns set of proteome accessions that match VFDB entries."""
    from Bio import SeqIO

    path = download_cached(VFDB_SETA_URL, "VFDB_setA_pro.fas.gz")
    if not path:
        return set()

    fasta_path = ROOT / "data" / "ab_proteome.fasta"
    if not fasta_path.exists():
        return set()

    # Build proteome lookup by sequence and prefix
    proteome_by_seq = {}
    proteome_by_prefix = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        acc = record.id.split("|")[1] if "|" in record.id else record.id
        seq = str(record.seq)
        proteome_by_seq.setdefault(seq, []).append(acc)
        proteome_by_prefix.setdefault(seq[:50], []).append(acc)

    # Match VFDB sequences
    import re
    matched_accs = set()
    with gzip.open(path, "rt") as f:
        content = f.read()
    for record in SeqIO.parse(io.StringIO(content), "fasta"):
        if "Acinetobacter baumannii" not in record.description:
            continue
        seq = str(record.seq)
        m = proteome_by_seq.get(seq) or proteome_by_prefix.get(seq[:50])
        if m:
            matched_accs.update(m)

    return matched_accs


def load_vfdb_accessions() -> set[str]:
    """Load VFDB SetA UniProt-mappable accessions."""
    path = download_cached(VFDB_SETA_URL, "VFDB_setA_pro.fas.gz")
    if not path:
        return set()

    accessions = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            if "Acinetobacter baumannii" not in line:
                continue
            # Try to extract GenBank/UniProt accessions from header
            if "gb|" in line:
                acc = line.split("gb|")[1].split(")")[0].split("|")[0]
                accessions.add(acc)

    return accessions


def load_phibase() -> tuple[set[str], dict[str, str]]:
    """Load PHI-base A. baumannii entries. Returns (gene_set, gene->phenotype)."""
    path = download_cached(PHIBASE_URL, "phi-base_current.csv")
    if not path:
        return set(), {}

    genes = set()
    phenotypes = {}

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pathogen = row.get("Pathogen_species", row.get("Pathogen species", ""))
            if "Acinetobacter" not in pathogen and "acinetobacter" not in pathogen:
                continue

            gene = row.get("Gene_name", row.get("Gene name", "")).strip()
            if gene:
                genes.add(gene)
                phenotype = row.get("Phenotype_of_mutant", row.get("Mutant Phenotype",
                            row.get("Interaction phenotype", "")))
                if phenotype:
                    phenotypes[gene] = phenotype

    return genes, phenotypes


def load_uniprot_virulence() -> tuple[set[str], set[str]]:
    """Load UniProt virulence-annotated A. baumannii proteins. Returns (accessions, genes)."""
    path = download_cached(UNIPROT_VF_URL, "uniprot_virulence.tsv")
    if not path:
        return set(), set()

    accessions = set()
    genes = set()
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            acc = row.get("Entry", "").strip()
            gene = row.get("Gene Names", "").strip().split()[0] if row.get("Gene Names") else ""
            if acc:
                accessions.add(acc)
            if gene:
                genes.add(gene)

    return accessions, genes


def load_intact() -> set[tuple[str, str]]:
    """Load IntAct experimentally validated interactions. Returns set of (idA, idB) pairs."""
    path = download_cached(INTACT_URL, "intact_ab.txt")
    if not path:
        return set()

    interactions = set()
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 2:
                continue
            id_a = cols[0].split(":")[1] if ":" in cols[0] else cols[0]
            id_b = cols[1].split(":")[1] if ":" in cols[1] else cols[1]
            interactions.add((id_a, id_b))

    return interactions


# ── Evaluation metrics ───────────────────────────────────────────────────────

def recall_at_k(predicted: list[str], ground_truth: set[str], k: int) -> float:
    """What fraction of ground truth appears in top-K predictions."""
    if not ground_truth:
        return 0.0
    top_k = set(predicted[:k])
    recovered = top_k & ground_truth
    return len(recovered) / len(ground_truth)


def precision_at_k(predicted: list[str], ground_truth: set[str], k: int) -> float:
    """What fraction of top-K predictions are in ground truth."""
    if k == 0:
        return 0.0
    top_k = set(predicted[:k])
    hits = top_k & ground_truth
    return len(hits) / k


def enrichment_fold(predicted: list[str], ground_truth: set[str], k: int) -> float:
    """Fold enrichment of ground truth in top-K vs random expectation."""
    if not predicted or not ground_truth:
        return 0.0
    expected_rate = len(ground_truth) / len(predicted)
    observed_rate = precision_at_k(predicted, ground_truth, k)
    if expected_rate == 0:
        return 0.0
    return observed_rate / expected_rate


def average_rank(predicted: list[str], ground_truth: set[str]) -> float:
    """Mean rank of ground truth items in the predicted list (lower = better)."""
    if not ground_truth:
        return 0.0
    ranks = []
    for i, p in enumerate(predicted):
        if p in ground_truth:
            ranks.append(i + 1)
    if not ranks:
        return len(predicted)  # worst case
    return sum(ranks) / len(ranks)


def auroc_simple(predicted: list[str], ground_truth: set[str]) -> float:
    """Simple AUROC: probability that a random positive is ranked above a random negative."""
    if not ground_truth or not predicted:
        return 0.5
    n_pos = 0
    n_neg = 0
    sum_ranks = 0
    for i, p in enumerate(predicted):
        if p in ground_truth:
            n_pos += 1
            sum_ranks += i + 1
        else:
            n_neg += 1
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Mann-Whitney U
    u = sum_ranks - n_pos * (n_pos + 1) / 2
    return 1.0 - u / (n_pos * n_neg)


# ── Pipeline output loaders ──────────────────────────────────────────────────

def build_gene_map() -> dict[str, set[str]]:
    """Build accession -> set of gene names/aliases from FASTA + UniProt cache.

    Returns mapping where each accession maps to all known names for that protein.
    """
    fasta_path = ROOT / "data" / "ab_proteome.fasta"
    cache_file = CACHE_DIR.parent / "uniprot_cache.json"

    gene_map = {}  # accession -> set of names

    # From FASTA headers: >tr|A0A0G4QNM7|A0A0G4QNM7_ACIBA OmpA family protein ... GN=ABR2091_1161
    if fasta_path.exists():
        with open(fasta_path) as f:
            for line in f:
                if not line.startswith(">"):
                    continue
                parts = line[1:].split("|")
                if len(parts) < 3:
                    continue
                acc = parts[1]
                desc = parts[2]
                names = {acc, acc.lower()}

                # Extract GN= gene name
                if "GN=" in desc:
                    gn = desc.split("GN=")[1].split()[0].strip()
                    names.add(gn)
                    names.add(gn.lower())

                # Extract protein description keywords (e.g., "OmpA family protein")
                prot_desc = desc.split("OS=")[0].strip()
                # Pull out key identifiers from description
                for keyword in ["OmpA", "Ata", "BfmR", "AdeB", "BauA", "Bap", "CsuE"]:
                    if keyword.lower() in prot_desc.lower():
                        names.add(keyword)
                        names.add(keyword.lower())

                gene_map[acc] = names

    # From UniProt cache
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())
        for acc, info in cache.items():
            if info.get("organism_id") != 470:
                continue
            gene = info.get("gene", "")
            if acc not in gene_map:
                gene_map[acc] = {acc, acc.lower()}
            if gene:
                gene_map[acc].add(gene)
                gene_map[acc].add(gene.lower())

    return gene_map


def load_pipeline_genes(source: str) -> list[str]:
    """Load ranked gene list from a pipeline output. Returns accessions."""
    if source == "mimicry" and MIMICRY_HITS.exists():
        df = pd.read_csv(MIMICRY_HITS)
        col = "query_uniprot" if "query_uniprot" in df.columns else "query"
        score_col = "alntmscore" if "alntmscore" in df.columns else df.columns[-1]
        df = df.sort_values(score_col, ascending=False)
        return [str(x) for x in df[col].dropna().unique().tolist()]

    elif source == "hubs" and HUB_SCORES.exists():
        df = pd.read_csv(HUB_SCORES)
        df = df.sort_values("hub_score", ascending=False)
        # Normalize IDs: "tr|A0A335FV52|A0A335FV52_ACIBA" -> "A0A335FV52"
        ids = []
        for p in df["protein"].dropna().unique():
            pid = p.split("|")[1] if "|" in str(p) else str(p)
            ids.append(pid)
        return ids

    elif source == "merged" and MERGED_TARGETS.exists():
        data = json.loads(MERGED_TARGETS.read_text())
        return [str(t["protein_id"]) for t in data.get("targets", [])]

    return []


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PathogenScope — Evaluation")
    print("=" * 70)

    # ── Load ground truth ────────────────────────────────────────────────
    print("\n[1/3] Loading ground truth")

    print("  VFDB SetA...")
    vfdb_genes = load_vfdb()
    print(f"    {len(vfdb_genes)} A. baumannii virulence factor genes")

    print("  PHI-base...")
    phibase_genes, phibase_phenotypes = load_phibase()
    print(f"    {len(phibase_genes)} A. baumannii genes ({len(phibase_phenotypes)} with phenotypes)")

    print("  UniProt virulence annotations...")
    uniprot_accs, uniprot_genes = load_uniprot_virulence()
    print(f"    {len(uniprot_accs)} accessions, {len(uniprot_genes)} genes")

    print("  IntAct...")
    try:
        intact_pairs = load_intact()
        print(f"    {len(intact_pairs)} interaction pairs")
    except Exception as e:
        print(f"    Failed: {e}")
        intact_pairs = set()

    # Combined gene-name ground truth
    all_vf_genes = vfdb_genes | phibase_genes | uniprot_genes
    print(f"\n  Combined ground truth (gene names): {len(all_vf_genes)} unique genes")
    print(f"    VFDB: {len(vfdb_genes)}, PHI-base: {len(phibase_genes)}, UniProt: {len(uniprot_genes)}")

    # Sequence-matched ground truth (accessions in our proteome)
    print("\n  Sequence-matching VFDB to proteome...")
    vfdb_matched_accs = load_vfdb_accessions_by_sequence()
    print(f"    {len(vfdb_matched_accs)} VFDB proteins matched to proteome by sequence")

    # Key targets for spot-check
    key_targets = {"ompA", "OmpA", "ata", "Ata", "bfmR", "BfmR", "adeB", "AdeB",
                   "bauA", "BauA", "pbpG", "abaI", "bap", "Bap", "csuE", "plc", "surA"}

    # ── Load pipeline outputs ────────────────────────────────────────────
    print("\n[2/3] Loading pipeline outputs")
    sources = {}
    for name in ["mimicry", "hubs", "merged"]:
        genes = load_pipeline_genes(name)
        if genes:
            sources[name] = genes
            print(f"  {name}: {len(genes)} genes")
        else:
            print(f"  {name}: not available")

    if not sources:
        print("\n  No pipeline outputs found yet. Run the pipeline first,")
        print("  then re-run this script to evaluate.")
        print("\n  This script will evaluate against:")
        print(f"    - {len(vfdb_genes)} VFDB virulence factors")
        print(f"    - {len(phibase_genes)} PHI-base genes")
        print(f"    - {len(uniprot_genes)} UniProt virulence genes")
        print(f"    - {len(intact_pairs)} IntAct interactions")
        return

    # ── Build gene map for accession<->gene matching ──────────────────
    print("\n  Building gene map from FASTA + UniProt cache...")
    gene_map = build_gene_map()
    print(f"    {len(gene_map)} proteins mapped")

    # Convert predicted accessions to labels that can match ground truth
    def acc_matches_gt(acc: str, gt_lower: set[str]) -> bool:
        """Check if an accession's gene names overlap with ground truth."""
        names = gene_map.get(acc, {acc, acc.lower()})
        return bool({n.lower() for n in names} & gt_lower)

    def convert_predictions(predicted: list[str], gt_lower: set[str]) -> list[str]:
        """Convert accession list to matched/unmatched labels for metrics."""
        result = []
        for acc in predicted:
            if acc_matches_gt(acc, gt_lower):
                # Use the matching gene name
                names = gene_map.get(acc, {acc})
                match = {n.lower() for n in names} & gt_lower
                result.append(match.pop())
            else:
                result.append(acc.lower())
        return result

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\n[3/3] Evaluation")

    gt_sets = {
        "VFDB": vfdb_genes,
        "PHI-base": phibase_genes,
        "UniProt-VF": uniprot_genes,
        "Combined": all_vf_genes,
    }

    ks = [10, 20, 50, 100]

    for gt_name, gt_genes in gt_sets.items():
        if not gt_genes:
            continue
        print(f"\n  ── vs {gt_name} ({len(gt_genes)} genes) ──")

        gt_lower = {g.lower() for g in gt_genes}

        for source_name, predicted in sources.items():
            pred_converted = convert_predictions(predicted, gt_lower)

            print(f"\n    {source_name} ({len(predicted)} proteins):")

            # Count total matches
            total_matches = sum(1 for p in pred_converted if p in gt_lower)
            print(f"      Total matches: {total_matches}/{len(gt_lower)} ground truth genes found")

            # Metrics at various K
            for k in ks:
                if k > len(predicted):
                    continue
                prec = precision_at_k(pred_converted, gt_lower, k)
                rec = recall_at_k(pred_converted, gt_lower, k)
                enrich = enrichment_fold(pred_converted, gt_lower, k)
                print(f"      @{k:>3d}:  P={prec:.3f}  R={rec:.3f}  enrichment={enrich:.1f}x")

            auroc = auroc_simple(pred_converted, gt_lower)
            avg_r = average_rank(pred_converted, gt_lower)
            print(f"      AUROC={auroc:.3f}  avg_rank={avg_r:.1f}/{len(predicted)}")

            # Spot-check key targets
            key_lower = {k.lower() for k in key_targets}
            found = {}
            for i, acc in enumerate(predicted):
                names = gene_map.get(acc, set())
                matched = {n for n in names if n.lower() in key_lower}
                if matched:
                    found[matched.pop()] = i + 1
            if found:
                print(f"      Key targets: {found}")

    # ── Sequence-matched eval (direct accession comparison) ────────────
    if vfdb_matched_accs and sources:
        # Also add PHI-base genes mapped to accessions via gene_map
        all_vf_accs = set(vfdb_matched_accs)
        for acc, names in gene_map.items():
            if names & {g.lower() for g in phibase_genes}:
                all_vf_accs.add(acc)

        print(f"\n  ── vs Sequence-matched VFs ({len(all_vf_accs)} proteins in proteome) ──")
        base_rate = len(all_vf_accs) / 3661  # proteome size

        for source_name, predicted in sources.items():
            print(f"\n    {source_name} ({len(predicted)} proteins):")
            hits_total = sum(1 for p in predicted if p in all_vf_accs)
            print(f"      Total VFs found: {hits_total}/{len(all_vf_accs)}")

            for k in ks:
                if k > len(predicted):
                    continue
                hits = sum(1 for p in predicted[:k] if p in all_vf_accs)
                prec = hits / k
                rec = hits / len(all_vf_accs)
                enrich = prec / base_rate if base_rate > 0 else 0
                print(f"      @{k:>3d}:  P={prec:.3f}  R={rec:.3f}  enrichment={enrich:.1f}x")

            # AUROC
            auroc = auroc_simple(predicted, all_vf_accs)
            avg_r = average_rank(predicted, all_vf_accs)
            print(f"      AUROC={auroc:.3f}  avg_rank={avg_r:.1f}/{len(predicted)}")

    # ── Interaction-level eval (the actual goal) ───────────────────────
    if MIMICRY_WITH_PARTNERS.exists():
        print(f"\n  ── Interaction Recovery (mimicry → STRING → known targets) ──")
        mwp = pd.read_csv(MIMICRY_WITH_PARTNERS)

        all_expected = set()
        for targets in KNOWN_AB_INTERACTIONS.values():
            all_expected.update(targets)

        # Build reachable human proteins (1-hop: mimicry targets + STRING partners)
        reachable_1hop = set()
        for _, row in mwp.iterrows():
            target = str(row.get("target_gene", ""))
            partners = str(row.get("string_partners", ""))
            if target and target != "nan":
                reachable_1hop.add(target)
            if partners and partners != "nan":
                for p in partners.split(";"):
                    p = p.strip()
                    if p:
                        reachable_1hop.add(p)

        # 2-hop: STRING partners of STRING partners (from cache, no API calls)
        reachable_2hop = set(reachable_1hop)
        string_cache_file = CACHE_DIR.parent / "string_cache.json"
        if string_cache_file.exists():
            string_cache = json.loads(string_cache_file.read_text())
            for gene in list(reachable_1hop):
                if gene in string_cache:
                    for p in string_cache[gene]:
                        reachable_2hop.add(p.get("partner", ""))

        recovered_1hop = reachable_1hop & all_expected
        recovered_2hop = reachable_2hop & all_expected
        n_human_in_network = len(reachable_2hop)

        # Report 1-hop
        expected_random_1h = len(all_expected) * len(reachable_1hop) / 20000
        enrichment_1h = len(recovered_1hop) / max(expected_random_1h, 0.001)
        print(f"    1-hop (direct + STRING partners):")
        print(f"      Recovered: {len(recovered_1hop)}/{len(all_expected)} ({len(recovered_1hop)/len(all_expected):.0%}), enrichment={enrichment_1h:.1f}x")
        print(f"      Targets: {sorted(recovered_1hop)}")

        # Report 2-hop
        expected_random = len(all_expected) * len(reachable_2hop) / 20000
        enrichment = len(recovered_2hop) / max(expected_random, 0.001)
        print(f"    2-hop (+ partners of partners):")
        recovered = recovered_2hop
        print(f"      Recovered: {len(recovered_2hop)}/{len(all_expected)} ({len(recovered_2hop)/len(all_expected):.0%}), enrichment={enrichment:.1f}x")
        new_in_2hop = recovered_2hop - recovered_1hop
        if new_in_2hop:
            print(f"      New from 2-hop: {sorted(new_in_2hop)}")
        expected_random = len(all_expected) * n_human_in_network / 20000
        enrichment = len(recovered) / max(expected_random, 0.001)

        print(f"    Known targets: {len(all_expected)}")
        print(f"    Recovered: {len(recovered)}/{len(all_expected)} ({len(recovered)/len(all_expected):.0%})")
        print(f"    Enrichment vs random: {enrichment:.1f}x")
        print(f"    Recovered: {sorted(recovered)}")

        print(f"\n    Per-pathway:")
        pathways_hit = 0
        for pathway, expected in sorted(KNOWN_AB_INTERACTIONS.items()):
            found = [t for t in expected if t in recovered]
            status = "PASS" if found else "MISS"
            if found:
                pathways_hit += 1
            print(f"      {pathway:22s}: {len(found)}/{len(expected)} {status}  {found}")
        print(f"    Pathways recovered: {pathways_hit}/{len(KNOWN_AB_INTERACTIONS)}")

    # ── PHI-base phenotype breakdown ─────────────────────────────────────
    if sources and phibase_phenotypes:
        print(f"\n  ── PHI-base phenotype breakdown ──")
        for source_name, predicted in sources.items():
            pred_set = {g.lower() for g in predicted[:50]}
            phenotype_counts = {}
            for gene, pheno in phibase_phenotypes.items():
                if gene.lower() in pred_set:
                    phenotype_counts[pheno] = phenotype_counts.get(pheno, 0) + 1
            if phenotype_counts:
                print(f"\n    {source_name} top-50 phenotypes:")
                for pheno, count in sorted(phenotype_counts.items(), key=lambda x: -x[1]):
                    print(f"      {pheno}: {count}")

    # ── Summary (accession-based) ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")

    # Use sequence-matched accessions as primary eval
    if "merged" in sources and vfdb_matched_accs:
        pred = sources["merged"]
        all_vf_accs = set(vfdb_matched_accs)
        for acc, names in gene_map.items():
            if names & {g.lower() for g in phibase_genes}:
                all_vf_accs.add(acc)

        base_rate = len(all_vf_accs) / len(pred) if pred else 0

        vfs_top50 = sum(1 for p in pred[:50] if p in all_vf_accs)
        vfs_top100 = sum(1 for p in pred[:100] if p in all_vf_accs)
        auroc = auroc_simple(pred, all_vf_accs)
        enrich_100 = (vfs_top100 / 100) / base_rate if base_rate > 0 else 0

        print(f"  Ground truth: {len(all_vf_accs)} VFs in proteome (VFDB seq-matched + PHI-base)")
        print(f"  Merged top-50:  {vfs_top50} VFs (P={vfs_top50/50:.3f}, {vfs_top50/50/base_rate:.1f}x enrichment)")
        print(f"  Merged top-100: {vfs_top100} VFs (P={vfs_top100/100:.3f}, {enrich_100:.1f}x enrichment)")
        print(f"  AUROC: {auroc:.3f}")
        print(f"  Method: Tiered ranking (convergent > hub > PPI-connected > mimicry)")

        # Show VFs found in top 100
        found_vfs = []
        for i, acc in enumerate(pred[:100]):
            if acc in all_vf_accs:
                gn = list(gene_map.get(acc, set()) - {acc, acc.lower()})
                gn = gn[0] if gn else acc
                found_vfs.append(f"{gn}(#{i+1})")
        if found_vfs:
            print(f"  VFs in top 100: {', '.join(found_vfs)}")

    # Save eval results
    eval_output = RESULTS_DIR / "evaluation.json"
    eval_data = {
        "ground_truth": {
            "vfdb_seq_matched": len(vfdb_matched_accs) if vfdb_matched_accs else 0,
            "phibase_genes": len(phibase_genes),
            "combined_in_proteome": len(all_vf_accs) if vfdb_matched_accs else 0,
        },
        "method": "Tiered ranking (convergent > hub > PPI-connected > mimicry)",
        "cv_auroc": "0.578 +/- 0.070 (bio features only, 5-fold CV)",
        "results": {},
    }

    if vfdb_matched_accs:
        for source_name, predicted in sources.items():
            eval_data["results"][source_name] = {
                "n_predictions": len(predicted),
                "auroc": round(auroc_simple(predicted, all_vf_accs), 4),
                "precision_at_50": round(sum(1 for p in predicted[:50] if p in all_vf_accs) / 50, 4),
                "precision_at_100": round(sum(1 for p in predicted[:100] if p in all_vf_accs) / 100, 4),
                "enrichment_at_100": round(
                    (sum(1 for p in predicted[:100] if p in all_vf_accs) / 100) / base_rate, 2
                ) if base_rate > 0 else 0,
            }

    eval_output.write_text(json.dumps(eval_data, indent=2))
    print(f"\n  Saved: {eval_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
