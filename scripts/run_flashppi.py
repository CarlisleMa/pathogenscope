#!/usr/bin/env python3
"""Run FlashPPI all-vs-all on A. baumannii proteome (local H100 GPUs)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FASTA_PATH = DATA_DIR / "ab_proteome.fasta"
FLASHPPI_DIR = ROOT / "FlashPPI"
OUTPUT_PATH = RESULTS_DIR / "pathogen_ppis.tsv"


def clone_flashppi():
    if FLASHPPI_DIR.exists():
        print(f"  [skip] FlashPPI already cloned at {FLASHPPI_DIR}")
        return
    print("  Cloning FlashPPI...")
    subprocess.run(
        ["git", "clone", "https://github.com/TattaBio/FlashPPI.git", str(FLASHPPI_DIR)],
        check=True,
    )
    print("  Clone complete")


def install_deps():
    """Install FlashPPI dependencies."""
    req_file = FLASHPPI_DIR / "requirements.txt"
    print("  Installing FlashPPI dependencies...")
    subprocess.run(
        ["uv", "pip", "install", "-r", str(req_file)],
        check=True,
    )
    # Try installing flash-attn for H100 speedup
    print("  Installing flash-attn (optional, for GPU speedup)...")
    result = subprocess.run(
        ["uv", "pip", "install", "flash-attn", "--no-build-isolation"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("    flash-attn installed")
    else:
        print("    flash-attn install failed (non-critical, continuing without it)")


def run_prediction():
    """Run predict_proteome.py from FlashPPI."""
    script = FLASHPPI_DIR / "predict_proteome.py"
    # FlashPPI outputs CSV, we'll convert to TSV after
    csv_output = RESULTS_DIR / "pathogen_ppis.csv"

    cmd = [
        sys.executable, str(script),
        "--fasta", str(FASTA_PATH),
        "--output", str(csv_output),
        "--threshold", "0.5",
        "--batch_size", "64",
    ]

    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(FLASHPPI_DIR))

    # Convert CSV to TSV for consistency with rest of pipeline
    if csv_output.exists():
        import pandas as pd
        df = pd.read_csv(csv_output)
        # Rename columns to match pipeline expectations
        col_map = {
            "query_id": "protein_a",
            "match_id": "protein_b",
            "contact_score": "score",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df.to_csv(OUTPUT_PATH, sep="\t", index=False)
        print(f"  Converted to TSV: {OUTPUT_PATH}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PathogenScope — FlashPPI Intra-Pathogen Network")
    print("=" * 60)

    if not FASTA_PATH.exists():
        print(f"\nERROR: FASTA not found: {FASTA_PATH}")
        print("Run download_data.py first.")
        sys.exit(1)

    if OUTPUT_PATH.exists():
        n_lines = sum(1 for _ in open(OUTPUT_PATH)) - 1
        print(f"\n[skip] Output already exists: {OUTPUT_PATH} ({n_lines} pairs)")
        print("Delete it to re-run.")
        return

    # Setup
    print("\n[1/3] FlashPPI setup")
    clone_flashppi()

    print("\n[2/3] Installing dependencies")
    install_deps()

    # Count sequences
    n_seq = sum(1 for line in open(FASTA_PATH) if line.startswith(">"))
    print(f"\n[3/3] Running all-vs-all prediction ({n_seq} proteins)")
    run_prediction()

    # Verify
    if OUTPUT_PATH.exists():
        n_lines = sum(1 for _ in open(OUTPUT_PATH)) - 1
        print(f"\nOutput: {OUTPUT_PATH} ({n_lines} predicted pairs)")
    else:
        print(f"\nWARNING: Output not created at {OUTPUT_PATH}")

    print("\nFlashPPI intra-pathogen network complete.")


if __name__ == "__main__":
    main()
