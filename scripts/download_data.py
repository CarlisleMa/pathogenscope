#!/usr/bin/env python3
"""Download A. baumannii AlphaFold structures and proteome FASTA."""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STRUCTURES_DIR = DATA_DIR / "structures"

# UP000006737 is redundant — UP000072389 is the active A. baumannii proteome
PROTEOME_ID = "UP000072389"

FASTA_URL = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=(proteome:{PROTEOME_ID})"
FASTA_PATH = DATA_DIR / "ab_proteome.fasta"

# AlphaFold has no bulk tar for A. baumannii — download per-protein
ALPHAFOLD_CIF_URL = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.cif"


def download_file(url: str, dest: Path, desc: str) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {desc} already exists: {dest}")
        return
    print(f"  Downloading {desc}...")
    print(f"    URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"    Done ({size_mb:.1f} MB)")


def parse_fasta_ids(fasta_path: Path) -> list[str]:
    """Extract UniProt accessions from FASTA headers."""
    accessions = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                # >sp|Q6F7T4|... or >tr|A0A0D5Y8Z2|...
                parts = line[1:].split("|")
                if len(parts) >= 2:
                    accessions.append(parts[1])
                else:
                    accessions.append(parts[0].split()[0])
    return accessions


def count_fasta_sequences(fasta_path: Path) -> int:
    count = 0
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def download_structure(accession: str) -> tuple[str, bool]:
    """Download a single AlphaFold structure. Returns (accession, success)."""
    dest = STRUCTURES_DIR / f"AF-{accession}-F1-model_v4.cif"
    if dest.exists() and dest.stat().st_size > 0:
        return accession, True

    url = ALPHAFOLD_CIF_URL.format(accession=accession)
    try:
        urllib.request.urlretrieve(url, dest)
        return accession, True
    except Exception:
        # Try older versions as fallback
        for v in [4, 3, 2]:
            url_old = url.replace("model_v6", f"model_v{v}")
            try:
                urllib.request.urlretrieve(url_old, dest)
                return accession, True
            except Exception:
                continue
        return accession, False


def download_all_structures(accessions: list[str], max_workers: int = 8) -> tuple[int, int]:
    """Download AlphaFold structures in parallel. Returns (success, failed) counts."""
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check how many already exist
    existing = sum(
        1 for a in accessions
        if (STRUCTURES_DIR / f"AF-{a}-F1-model_v6.cif").exists()
    )
    if existing == len(accessions):
        print(f"  [skip] All {existing} structures already downloaded")
        return existing, 0

    if existing > 0:
        print(f"  {existing}/{len(accessions)} already downloaded, fetching rest...")

    success = existing
    failed = 0
    to_download = [
        a for a in accessions
        if not (STRUCTURES_DIR / f"AF-{a}-F1-model_v6.cif").exists()
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_structure, acc): acc for acc in to_download}
        for i, future in enumerate(as_completed(futures), 1):
            acc, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            if i % 200 == 0 or i == len(to_download):
                print(f"    Progress: {i}/{len(to_download)}  (ok={success}, failed={failed})")

    return success, failed


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PathogenScope — Data Download")
    print(f"Organism: A. baumannii ({PROTEOME_ID})")
    print("=" * 60)

    # 1. Download FASTA
    print("\n[1/3] Proteome FASTA")
    download_file(FASTA_URL, FASTA_PATH, "proteome FASTA")
    n_seq = count_fasta_sequences(FASTA_PATH)
    print(f"  Sequences in FASTA: {n_seq}")

    if n_seq == 0:
        print("  ERROR: FASTA is empty. Check the proteome ID.")
        # Clean up empty file so it re-downloads next time
        FASTA_PATH.unlink(missing_ok=True)
        sys.exit(1)

    # 2. Parse accessions for structure download
    print("\n[2/3] Parsing accessions")
    accessions = parse_fasta_ids(FASTA_PATH)
    print(f"  Found {len(accessions)} accessions")

    # 3. Download AlphaFold structures per-protein
    print("\n[3/3] Downloading AlphaFold structures (per-protein)")
    print(f"  No bulk tar available for A. baumannii — downloading individually")
    success, failed = download_all_structures(accessions)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print(f"  FASTA:      {FASTA_PATH}  ({n_seq} sequences)")
    print(f"  Structures: {STRUCTURES_DIR}  ({success} downloaded, {failed} failed)")
    if failed > 0:
        print(f"  NOTE: {failed} proteins had no AlphaFold structure (expected for some)")
    print(f"\n  Send {FASTA_PATH} to Joe for cross-kingdom FlashPPI training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
