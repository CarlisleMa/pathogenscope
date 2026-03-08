#!/usr/bin/env python3
"""Analyze FlashPPI output: build network, compute hub scores."""

import csv
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

PPI_INPUT = RESULTS_DIR / "pathogen_ppis.tsv"
NETWORK_OUTPUT = RESULTS_DIR / "pathogen_network.csv"
HUB_OUTPUT = RESULTS_DIR / "pathogen_hubs.csv"

SCORE_THRESHOLD = 0.7


def load_ppis(path: Path) -> pd.DataFrame:
    """Load FlashPPI output TSV."""
    # FlashPPI output format varies — try common column sets
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None)

    # Normalize column names
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    # Identify protein A, protein B, and score columns
    col_map = {}
    for c in cols:
        if c in ("protein_a", "protein1", "query", "id_a", "source"):
            col_map["protein_a"] = c
        elif c in ("protein_b", "protein2", "target", "id_b", "dest"):
            col_map["protein_b"] = c
        elif c in ("score", "probability", "confidence", "ppi_score"):
            col_map["score"] = c

    # Fallback: first three columns
    if len(col_map) < 3 and len(df.columns) >= 3:
        col_map.setdefault("protein_a", df.columns[0])
        col_map.setdefault("protein_b", df.columns[1])
        col_map.setdefault("score", df.columns[2])

    df = df.rename(columns={v: k for k, v in col_map.items()})

    # Ensure score is numeric
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    print(f"  Loaded {len(df)} PPI pairs")
    return df


def build_network(df: pd.DataFrame, threshold: float) -> nx.Graph:
    """Build networkx graph from filtered PPIs."""
    filtered = df[df["score"] >= threshold].copy()
    print(f"  Edges above threshold ({threshold}): {len(filtered)}")

    G = nx.Graph()
    for _, row in filtered.iterrows():
        G.add_edge(
            row["protein_a"],
            row["protein_b"],
            score=row["score"],
        )

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_hub_scores(G: nx.Graph) -> pd.DataFrame:
    """Compute degree and betweenness centrality for all nodes."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["protein", "degree", "betweenness", "hub_score"])

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)

    # Normalize degree to 0-1
    max_degree = max(degree.values()) if degree else 1

    rows = []
    for node in G.nodes():
        d = degree[node]
        b = betweenness[node]
        # Composite hub score: weighted combo of normalized degree + betweenness
        norm_degree = d / max_degree
        hub_score = 0.6 * norm_degree + 0.4 * b
        rows.append({
            "protein": node,
            "degree": d,
            "betweenness": round(b, 6),
            "norm_degree": round(norm_degree, 4),
            "hub_score": round(hub_score, 6),
        })

    df = pd.DataFrame(rows).sort_values("hub_score", ascending=False)
    return df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PathogenScope — Pathogen Hub Analysis")
    print("=" * 60)

    # Check input
    if not PPI_INPUT.exists():
        print(f"\nERROR: FlashPPI output not found: {PPI_INPUT}")
        print("Run run_flashppi.py first.")
        sys.exit(1)

    # 1. Load PPIs
    print("\n[1/3] Loading FlashPPI results")
    df = load_ppis(PPI_INPUT)

    # Save full network (filtered)
    filtered = df[df["score"] >= SCORE_THRESHOLD].copy()
    filtered.to_csv(NETWORK_OUTPUT, index=False)
    print(f"  Saved network: {NETWORK_OUTPUT} ({len(filtered)} edges)")

    # 2. Build network
    print("\n[2/3] Building network")
    G = build_network(df, SCORE_THRESHOLD)

    # 3. Hub scores
    print("\n[3/3] Computing hub scores")
    hubs = compute_hub_scores(G)
    hubs.to_csv(HUB_OUTPUT, index=False)
    print(f"  Saved hub ranking: {HUB_OUTPUT} ({len(hubs)} proteins)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    if not hubs.empty:
        print(f"  Top 15 hub proteins:")
        print(hubs.head(15).to_string(index=False))
    else:
        print("  No hub proteins found (empty network)")
    print("=" * 60)


if __name__ == "__main__":
    main()
