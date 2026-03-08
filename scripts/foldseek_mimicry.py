#!/usr/bin/env python3
"""Foldseek structural mimicry screen: A. baumannii vs human proteome.

Primary: Local foldseek binary (auto-installed if missing).
Fallback: Tamarind Bio API.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STRUCTURES_DIR = DATA_DIR / "structures"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"

RAW_OUTPUT = RESULTS_DIR / "ab_vs_human_raw.tsv"

FOLDSEEK_URL = "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"
FOLDSEEK_LOCAL_DIR = ROOT / "foldseek"

TAMARIND_BASE = "https://api.tamarind.bio"


# ── Local Foldseek ───────────────────────────────────────────────────────────

def foldseek_binary() -> str | None:
    """Find foldseek binary."""
    path = shutil.which("foldseek")
    if path:
        return path
    local = FOLDSEEK_LOCAL_DIR / "bin" / "foldseek"
    if local.exists():
        return str(local)
    return None


def install_foldseek() -> str:
    """Download and extract foldseek static binary. Returns path to binary."""
    local = FOLDSEEK_LOCAL_DIR / "bin" / "foldseek"
    if local.exists():
        return str(local)

    print("  Installing foldseek...")
    tar_path = CACHE_DIR / "foldseek-linux-avx2.tar.gz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not tar_path.exists():
        print(f"    Downloading from {FOLDSEEK_URL}")
        urllib.request.urlretrieve(FOLDSEEK_URL, tar_path)

    print("    Extracting...")
    FOLDSEEK_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=ROOT)

    binary = foldseek_binary()
    if binary:
        print(f"    Installed: {binary}")
        return binary

    raise RuntimeError(f"Foldseek extraction failed — expected binary at {local}")


def run_local_foldseek(structure_files: list[Path]) -> bool:
    """Run Foldseek locally against AlphaFold human proteome DB."""
    # Auto-install if needed
    binary = foldseek_binary()
    if not binary:
        try:
            binary = install_foldseek()
        except Exception as e:
            print(f"  Foldseek install failed: {e}")
            return False

    print(f"\n[Local] Using foldseek at {binary}")

    query_dir = STRUCTURES_DIR
    db_dir = CACHE_DIR / "foldseek_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = db_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Download human proteome DB if not cached
    human_db = db_dir / "h-sapiens"
    if not (db_dir / "h-sapiens.dbtype").exists():
        print("  Downloading AlphaFold human proteome database...")
        subprocess.run(
            [binary, "databases", "Alphafold/Proteome", str(human_db), str(tmp_dir)],
            check=True,
        )
    else:
        print("  [skip] Human proteome DB already cached")

    # Run search
    print("  Running easy-search (this may take a while)...")
    tmp_search = db_dir / "tmp_search"
    tmp_search.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            binary, "easy-search",
            str(query_dir),
            str(human_db),
            str(RAW_OUTPUT),
            str(tmp_search),
            "--format-output", "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore",
            "-e", "10",
        ],
        check=True,
    )

    print(f"  Results saved to {RAW_OUTPUT}")
    return True


# ── Tamarind Bio API (fallback) ──────────────────────────────────────────────

def tamarind_headers() -> dict:
    key = os.environ.get("TAMARIND_API_KEY")
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def tamarind_available() -> bool:
    key = os.environ.get("TAMARIND_API_KEY")
    if not key:
        print("  TAMARIND_API_KEY not set — skipping Tamarind")
        return False
    try:
        r = requests.get(f"{TAMARIND_BASE}/tools", headers=tamarind_headers(), timeout=15)
        r.raise_for_status()
        tools = r.json()
        print(f"  Tamarind API reachable — {len(tools)} tools available")
        for t in tools:
            name = t.get("name", t.get("id", ""))
            if "foldseek" in str(name).lower():
                print(f"    Found: {name}")
        return True
    except Exception as e:
        print(f"  Tamarind API check failed: {e}")
        return False


def upload_to_tamarind(filepath: Path) -> str:
    headers = tamarind_headers()
    headers.pop("Content-Type", None)
    with open(filepath, "rb") as f:
        r = requests.put(
            f"{TAMARIND_BASE}/upload/{filepath.name}",
            headers=headers,
            data=f.read(),
        )
    r.raise_for_status()
    resp = r.json()
    return resp.get("file_id", resp.get("url", filepath.name))


def submit_foldseek_job(file_ids: list[str], db: str = "afdb50") -> str:
    payload = {
        "type": "foldseek",
        "settings": {"database": db, "mode": "3diaa"},
        "files": file_ids,
    }
    r = requests.post(f"{TAMARIND_BASE}/submit-job", headers=tamarind_headers(), json=payload)
    r.raise_for_status()
    resp = r.json()
    return resp.get("job_id", resp.get("id"))


def poll_job(job_id: str, timeout: int = 3600, interval: int = 30) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{TAMARIND_BASE}/jobs/{job_id}", headers=tamarind_headers())
        r.raise_for_status()
        status = r.json()
        state = status.get("status", status.get("state", "unknown"))
        print(f"    Job {job_id}: {state} ({int(time.time()-start)}s elapsed)")
        if state in ("completed", "done", "finished"):
            return status
        if state in ("failed", "error"):
            raise RuntimeError(f"Job failed: {status}")
        time.sleep(interval)
    raise TimeoutError(f"Job {job_id} timed out after {timeout}s")


def download_result(job_id: str, out_path: Path) -> None:
    r = requests.post(
        f"{TAMARIND_BASE}/result",
        headers=tamarind_headers(),
        json={"job_id": job_id, "format": "tsv"},
    )
    r.raise_for_status()
    out_path.write_bytes(r.content)


def run_tamarind(structure_files: list[Path]) -> bool:
    print("\n[Tamarind] Uploading structures...")
    file_ids = []
    for i, sf in enumerate(structure_files):
        try:
            fid = upload_to_tamarind(sf)
            file_ids.append(fid)
            if (i + 1) % 100 == 0:
                print(f"    Uploaded {i+1}/{len(structure_files)}")
        except Exception as e:
            print(f"    Upload failed for {sf.name}: {e}")

    if not file_ids:
        return False

    print(f"  Uploaded {len(file_ids)} structures")
    print("\n[Tamarind] Submitting Foldseek job...")
    job_id = submit_foldseek_job(file_ids, db="afdb50")
    print(f"  Job ID: {job_id}")

    job_cache = CACHE_DIR / "foldseek_job_id.txt"
    job_cache.parent.mkdir(parents=True, exist_ok=True)
    job_cache.write_text(job_id)

    print("\n[Tamarind] Waiting for results...")
    poll_job(job_id)

    print("\n[Tamarind] Downloading results...")
    download_result(job_id, RAW_OUTPUT)
    print(f"  Results saved to {RAW_OUTPUT}")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PathogenScope — Foldseek Mimicry Screen")
    print("=" * 60)

    # Check for existing output
    if RAW_OUTPUT.exists():
        n = sum(1 for _ in open(RAW_OUTPUT))
        print(f"\n[skip] Output already exists: {RAW_OUTPUT} ({n} lines)")
        print("Delete it to re-run.")
        return

    # Gather structure files
    exts = ("*.cif.gz", "*.pdb.gz", "*.cif", "*.pdb")
    structure_files = []
    for ext in exts:
        structure_files.extend(STRUCTURES_DIR.glob(ext))
    structure_files = sorted(set(structure_files))

    if not structure_files:
        print(f"\nERROR: No structure files found in {STRUCTURES_DIR}")
        print("Run download_data.py first.")
        sys.exit(1)

    print(f"\nFound {len(structure_files)} structure files")

    # Try local foldseek first
    print("\n[1] Attempting local foldseek...")
    if run_local_foldseek(structure_files):
        print("\nFoldseek mimicry screen complete (local).")
        return

    # Fallback: Tamarind API
    print("\n[2] Local foldseek failed. Trying Tamarind Bio API...")
    if tamarind_available():
        try:
            if run_tamarind(structure_files):
                print("\nFoldseek mimicry screen complete (Tamarind).")
                return
        except Exception as e:
            print(f"\nTamarind failed: {e}")

    print("\nERROR: Both local foldseek and Tamarind failed.")
    print("Check the errors above and retry.")
    sys.exit(1)


if __name__ == "__main__":
    main()
