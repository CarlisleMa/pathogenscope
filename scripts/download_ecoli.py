"""Download E. coli K12 reference proteome from UniProt."""

import os
import requests

UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?format=fasta"
    "&query=(organism_id:83333)+AND+(reviewed:true)"
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ecoli_k12")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ecoli_k12_proteome.fasta")


def download():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        print(f"Already exists: {OUTPUT_FILE}")
        return OUTPUT_FILE

    print("Downloading E. coli K12 proteome from UniProt (reviewed/Swiss-Prot)...")
    resp = requests.get(UNIPROT_URL, timeout=120)
    resp.raise_for_status()

    with open(OUTPUT_FILE, "w") as f:
        f.write(resp.text)

    n_seqs = resp.text.count(">")
    print(f"Downloaded {n_seqs} proteins to {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    download()
