#!/usr/bin/env python3
"""Process Foldseek mimicry results: filter, map, STRING lookup, validate."""

import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"

RAW_TSV = RESULTS_DIR / "ab_vs_human_raw.tsv"
MIMICRY_HITS = RESULTS_DIR / "mimicry_hits.csv"
MIMICRY_WITH_PARTNERS = RESULTS_DIR / "mimicry_with_partners.csv"

# Thresholds
TM_SCORE_MIN = 0.5
SEQ_IDENTITY_MAX = 0.30
TOP_N_PER_QUERY = 5  # Keep top N human targets per pathogen protein

# Known interactions for validation
KNOWN_INTERACTIONS = {
    "OmpA": {"TLR2", "TLR4", "FN1", "DNM1L"},
    "Ata": {"COL4A1", "COL4A2", "FN1"},
}

FOLDSEEK_COLUMNS = [
    "query", "target", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits", "alntmscore",
]


def extract_uniprot_id(name: str) -> str:
    """Extract UniProt accession from AF-Q6F7T4-F1-model_v6 or sp|Q6F7T4|..."""
    name = str(name)
    if name.startswith("AF-"):
        return name.split("-")[1]
    if "|" in name:
        parts = name.split("|")
        return parts[1] if len(parts) >= 2 else parts[0].split()[0]
    return name.split()[0]


def load_and_filter(path: Path) -> pd.DataFrame:
    """Load raw TSV, filter by TM-score, pre-filter to human targets, keep top N per query."""
    print("  Loading raw TSV...")
    df = pd.read_csv(path, sep="\t", header=None, names=FOLDSEEK_COLUMNS)
    print(f"    {len(df)} raw hits")

    # Normalize fident to 0-1
    if df["fident"].max() > 1:
        df["fident"] = df["fident"] / 100.0

    # Filter: high structural similarity
    df = df[df["alntmscore"] >= TM_SCORE_MIN].copy()
    print(f"    {len(df)} after TM>{TM_SCORE_MIN}")

    # Extract UniProt IDs
    df["query_uniprot"] = df["query"].apply(extract_uniprot_id)
    df["target_uniprot"] = df["target"].apply(extract_uniprot_id)

    # Pre-filter to human targets using cached UniProt data
    cache_file = CACHE_DIR / "uniprot_cache.json"
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())
        human_accs = {acc for acc, info in cache.items() if info.get("organism_id") == 9606}
        if human_accs:
            df_human = df[df["target_uniprot"].isin(human_accs)].copy()
            print(f"    {len(df_human)} with known human targets (from cache)")
            if not df_human.empty:
                df = df_human

    # Keep top N human targets per query (by TM-score)
    df = df.sort_values("alntmscore", ascending=False)
    df = df.groupby("query_uniprot").head(TOP_N_PER_QUERY)
    print(f"    {len(df)} after keeping top {TOP_N_PER_QUERY} per query")
    print(f"    Unique queries: {df['query_uniprot'].nunique()}")
    print(f"    Unique targets: {df['target_uniprot'].nunique()}")

    return df


def batch_uniprot_lookup(accessions: list[str], batch_size: int = 50) -> dict:
    """Look up gene names from UniProt REST API. Targets are already human (from Alphafold/Proteome DB)."""
    cache_file = CACHE_DIR / "uniprot_cache.json"
    cache = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())

    uncached = [a for a in accessions if a not in cache]
    print(f"    {len(accessions)} IDs, {len(accessions) - len(uncached)} cached, {len(uncached)} to look up")

    for i in range(0, len(uncached), batch_size):
        batch = uncached[i:i + batch_size]
        query = " OR ".join(f"accession:{acc}" for acc in batch)
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": query,
            "fields": "accession,gene_primary,organism_name,organism_id",
            "format": "json",
            "size": str(batch_size),
        }

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            for entry in data.get("results", []):
                acc = entry["primaryAccession"]
                gene = ""
                if entry.get("genes"):
                    gene = entry["genes"][0].get("geneName", {}).get("value", "")
                org_id = entry.get("organism", {}).get("taxonId", 0)
                org_name = entry.get("organism", {}).get("scientificName", "")
                cache[acc] = {
                    "gene": gene,
                    "organism_id": org_id,
                    "organism": org_name,
                }
        except Exception as e:
            print(f"      UniProt batch error: {e}")

        done = min(i + batch_size, len(uncached))
        if done % 2000 == 0 or done == len(uncached):
            print(f"      Progress: {done}/{len(uncached)}")
            # Save cache periodically
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, indent=2))

        time.sleep(0.3)

    # Final cache save
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=2))

    return cache


def string_partner_lookup(gene_names: list[str], species: int = 9606, min_score: int = 700) -> dict:
    """Look up STRING interaction partners for genes."""
    cache_file = CACHE_DIR / "string_cache.json"
    cache = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())

    result = {}
    uncached = [g for g in gene_names if g and g not in cache]
    print(f"    {len(gene_names)} genes, {len(uncached)} uncached")

    for i, gene in enumerate(uncached):
        url = "https://string-db.org/api/json/interaction_partners"
        params = {
            "identifiers": gene,
            "species": species,
            "required_score": min_score,
            "limit": 20,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            partners = []
            for item in r.json():
                partners.append({
                    "partner": item.get("preferredName_B", item.get("stringId_B", "")),
                    "score": item.get("score", 0),
                })
            cache[gene] = partners
        except Exception as e:
            print(f"      STRING error for {gene}: {e}")
            cache[gene] = []

        if (i + 1) % 100 == 0:
            print(f"      Progress: {i+1}/{len(uncached)}")
            cache_file.write_text(json.dumps(cache, indent=2))

        time.sleep(0.3)

    cache_file.write_text(json.dumps(cache, indent=2))

    for gene in gene_names:
        if gene:
            result[gene] = cache.get(gene, [])

    return result


def validate_known_interactions(df: pd.DataFrame, partners: dict) -> None:
    """Check if known A. baumannii interactions are recovered."""
    # Build accession->description map from FASTA
    fasta_path = ROOT / "data" / "ab_proteome.fasta"
    acc_to_desc = {}
    if fasta_path.exists():
        with open(fasta_path) as f:
            for line in f:
                if line.startswith(">"):
                    parts = line[1:].split("|")
                    if len(parts) >= 3:
                        acc = parts[1]
                        desc = parts[2].split("OS=")[0].strip()
                        acc_to_desc[acc] = desc

    print("\n  Validation against known interactions:")
    for pathogen_gene, expected_partners in KNOWN_INTERACTIONS.items():
        # Match on gene name OR protein description
        mask = df["query_gene"].str.contains(pathogen_gene, case=False, na=False)
        # Also check FASTA descriptions for the protein name
        mask |= df["query_uniprot"].apply(
            lambda x: pathogen_gene.lower() in acc_to_desc.get(x, "").lower()
        )
        hits = df[mask]

        if hits.empty:
            print(f"    {pathogen_gene}: NOT FOUND in mimicry hits")
            continue

        print(f"    {pathogen_gene}: {len(hits)} mimicry hits")
        found_partners = set()
        for _, row in hits.iterrows():
            target_gene = row.get("target_gene", "")
            if target_gene in partners:
                for p in partners[target_gene]:
                    if p["partner"] in expected_partners:
                        found_partners.add(p["partner"])
            if target_gene in expected_partners:
                found_partners.add(target_gene)

        target_genes = set(hits["target_gene"].dropna())
        if found_partners:
            print(f"      -> {', '.join(found_partners)}  VALIDATED")
        else:
            print(f"      -> targets: {', '.join(list(target_genes)[:8])}  (no known partner overlap yet)")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PathogenScope — Process Mimicry Results")
    print("=" * 60)

    if not RAW_TSV.exists():
        print(f"\nERROR: Raw results not found: {RAW_TSV}")
        print("Run foldseek_mimicry.py first.")
        sys.exit(1)

    # 1. Load + filter + keep top N
    print("\n[1/5] Loading and filtering")
    df = load_and_filter(RAW_TSV)

    if df.empty:
        print("  No hits pass filters.")
        sys.exit(0)

    # 2. UniProt gene name mapping
    # Targets are from Alphafold/Proteome (human DB), so we know they're human
    # Only need gene name mapping, not organism filtering
    print("\n[2/5] UniProt gene name mapping")
    all_accessions = list(set(df["query_uniprot"].tolist() + df["target_uniprot"].tolist()))
    uniprot_info = batch_uniprot_lookup(all_accessions)

    df["query_gene"] = df["query_uniprot"].apply(
        lambda x: uniprot_info.get(x, {}).get("gene", x)
    )
    df["target_gene"] = df["target_uniprot"].apply(
        lambda x: uniprot_info.get(x, {}).get("gene", x)
    )

    # Filter to human targets only (organism_id == 9606)
    print("\n[3/5] Filtering to human targets")
    df["target_organism"] = df["target_uniprot"].apply(
        lambda x: uniprot_info.get(x, {}).get("organism_id", 0)
    )
    df = df[df["target_organism"] == 9606].copy()
    print(f"  {len(df)} hits with human targets")

    if df.empty:
        print("  No human targets found.")
        df.to_csv(MIMICRY_HITS, index=False)
        sys.exit(0)

    # Save mimicry hits
    df.to_csv(MIMICRY_HITS, index=False)
    print(f"  Saved: {MIMICRY_HITS} ({len(df)} rows)")

    # 4. STRING partner lookup (only for unique target genes)
    print("\n[4/5] STRING interaction partner lookup")
    target_genes = df["target_gene"].dropna().unique().tolist()
    partners = string_partner_lookup(target_genes)

    df["string_partners"] = df["target_gene"].apply(
        lambda g: ";".join(p["partner"] for p in partners.get(g, [])[:5])
    )
    df["top_partner_score"] = df["target_gene"].apply(
        lambda g: max((p["score"] for p in partners.get(g, [])), default=0)
    )

    df.to_csv(MIMICRY_WITH_PARTNERS, index=False)
    print(f"  Saved: {MIMICRY_WITH_PARTNERS} ({len(df)} rows)")

    # 5. Validation
    print("\n[5/5] Validation")
    validate_known_interactions(df, partners)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print(f"  Mimicry hits:  {MIMICRY_HITS} ({len(df)} rows)")
    print(f"  With partners: {MIMICRY_WITH_PARTNERS}")
    print(f"  Top targets by TM-score:")
    top = df.nlargest(10, "alntmscore")[["query_gene", "target_gene", "alntmscore", "fident"]]
    print(top.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
