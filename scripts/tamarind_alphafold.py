#!/usr/bin/env python3
"""(Optional) Run AlphaFold on Tamarind Bio for proteins missing from AlphaFold DB.

Use case: XDR clinical isolate with novel/mutated proteins not in AlphaFold DB.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
STRUCTURES_DIR = DATA_DIR / "structures"

TAMARIND_BASE = "https://api.tamarind.bio"


def tamarind_headers() -> dict:
    key = os.environ.get("TAMARIND_API_KEY")
    if not key:
        print("ERROR: TAMARIND_API_KEY not set")
        sys.exit(1)
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def parse_fasta(fasta_path: Path) -> dict[str, str]:
    """Parse FASTA into {id: sequence}."""
    sequences = {}
    current_id = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                parts = line[1:].split("|")
                current_id = parts[1] if len(parts) >= 2 else parts[0].split()[0]
                current_seq = []
            elif current_id:
                current_seq.append(line)
    if current_id:
        sequences[current_id] = "".join(current_seq)
    return sequences


def find_missing_structures(sequences: dict[str, str]) -> list[str]:
    """Find proteins without AlphaFold structures."""
    existing = set()
    for f in STRUCTURES_DIR.glob("*"):
        # Extract UniProt ID from filename: AF-Q6F7T4-F1-model_v4.cif.gz
        name = f.stem.replace(".cif", "").replace(".pdb", "")
        if name.startswith("AF-"):
            parts = name.split("-")
            if len(parts) >= 2:
                existing.add(parts[1])

    missing = [pid for pid in sequences if pid not in existing]
    return missing


def submit_alphafold_job(sequence: str, protein_id: str) -> str:
    """Submit AlphaFold prediction via Tamarind."""
    payload = {
        "type": "alphafold",
        "settings": {
            "name": protein_id,
        },
        "sequences": [sequence],
    }
    r = requests.post(
        f"{TAMARIND_BASE}/submit-job",
        headers=tamarind_headers(),
        json=payload,
    )
    r.raise_for_status()
    return r.json().get("job_id", r.json().get("id"))


def poll_job(job_id: str, timeout: int = 7200, interval: int = 60) -> dict:
    """Poll until job completes."""
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{TAMARIND_BASE}/jobs/{job_id}", headers=tamarind_headers())
        r.raise_for_status()
        status = r.json()
        state = status.get("status", status.get("state", "unknown"))
        print(f"    Job {job_id}: {state} ({int(time.time()-start)}s)")
        if state in ("completed", "done", "finished"):
            return status
        if state in ("failed", "error"):
            raise RuntimeError(f"Job failed: {status}")
        time.sleep(interval)
    raise TimeoutError(f"Job {job_id} timed out")


def download_structure(job_id: str, protein_id: str) -> Path:
    """Download predicted structure."""
    r = requests.post(
        f"{TAMARIND_BASE}/result",
        headers=tamarind_headers(),
        json={"job_id": job_id, "format": "pdb"},
    )
    r.raise_for_status()
    out_path = STRUCTURES_DIR / f"tamarind-{protein_id}.pdb"
    out_path.write_bytes(r.content)
    return out_path


def main():
    print("=" * 60)
    print("PathogenScope — Tamarind AlphaFold (Optional)")
    print("=" * 60)

    fasta_path = DATA_DIR / "ab_proteome.fasta"
    if not fasta_path.exists():
        print(f"\nERROR: FASTA not found: {fasta_path}")
        print("Run download_data.py first.")
        sys.exit(1)

    sequences = parse_fasta(fasta_path)
    print(f"\nTotal proteins: {len(sequences)}")

    missing = find_missing_structures(sequences)
    print(f"Missing structures: {len(missing)}")

    if not missing:
        print("\nAll proteins have AlphaFold structures. Nothing to do.")
        return

    print(f"\nSubmitting {len(missing)} proteins for AlphaFold prediction...")

    # Track jobs
    jobs = {}
    job_cache = CACHE_DIR / "alphafold_jobs.json"
    if job_cache.exists():
        jobs = json.loads(job_cache.read_text())

    for pid in missing:
        if pid in jobs:
            print(f"  [skip] {pid} already submitted (job {jobs[pid]})")
            continue

        seq = sequences[pid]
        print(f"  Submitting {pid} ({len(seq)} aa)...")
        try:
            job_id = submit_alphafold_job(seq, pid)
            jobs[pid] = job_id
            print(f"    Job ID: {job_id}")
        except Exception as e:
            print(f"    Failed: {e}")

    # Save job IDs
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    job_cache.write_text(json.dumps(jobs, indent=2))

    # Poll and download
    print("\nWaiting for results...")
    for pid, job_id in jobs.items():
        out_path = STRUCTURES_DIR / f"tamarind-{pid}.pdb"
        if out_path.exists():
            print(f"  [skip] {pid} already downloaded")
            continue
        try:
            poll_job(job_id)
            path = download_structure(job_id, pid)
            print(f"  Downloaded: {path}")
        except Exception as e:
            print(f"  {pid} failed: {e}")

    print("\nAlphaFold predictions complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
