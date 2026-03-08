#!/usr/bin/env python3
"""Export network data for Joseph's React dashboard."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Inputs
MERGED = RESULTS_DIR / "merged_targets.json"
MIMICRY = RESULTS_DIR / "mimicry_with_partners.csv"
NETWORK = RESULTS_DIR / "pathogen_network.csv"

# Output
DASHBOARD_JSON = RESULTS_DIR / "network_data.json"


def build_dashboard_data() -> dict:
    """Build network_data.json for the React dashboard."""

    nodes = []
    edges = []
    node_ids = set()

    # Load merged targets
    targets = []
    if MERGED.exists():
        merged = json.loads(MERGED.read_text())
        targets = merged.get("targets", [])
        metadata = merged.get("metadata", {})
    else:
        metadata = {"organism": "Acinetobacter baumannii", "proteome": "UP000006737"}

    # Add pathogen protein nodes from merged targets
    for t in targets:
        pid = t["protein_id"]
        if pid not in node_ids:
            nodes.append({
                "id": pid,
                "label": t.get("gene", pid),
                "type": "pathogen",
                "organism": "Acinetobacter baumannii",
                "composite_score": t.get("composite_score", 0),
                "hub_score": t.get("hub_score", 0),
                "degree": t.get("degree", 0),
                "mimicry_tm": t.get("mimicry_tm", 0),
                "signals": t.get("signals", []),
                "convergent": t.get("convergent", False),
            })
            node_ids.add(pid)

        # Add human target nodes + mimicry edges
        for human_target in t.get("mimicry_targets", []):
            if human_target and human_target not in node_ids:
                nodes.append({
                    "id": human_target,
                    "label": human_target,
                    "type": "human",
                    "organism": "Homo sapiens",
                })
                node_ids.add(human_target)

            if human_target:
                edges.append({
                    "source": pid,
                    "target": human_target,
                    "type": "mimicry",
                    "score": t.get("mimicry_tm", 0),
                })

    # Add pathogen-pathogen PPI edges
    if NETWORK.exists():
        ppi_df = pd.read_csv(NETWORK)
        for _, row in ppi_df.iterrows():
            pa = str(row.get("protein_a", row.iloc[0]))
            pb = str(row.get("protein_b", row.iloc[1]))
            score = float(row.get("score", row.iloc[2]) if len(row) > 2 else 0)

            # Add nodes if not already present
            for p in [pa, pb]:
                if p not in node_ids:
                    nodes.append({
                        "id": p,
                        "label": p,
                        "type": "pathogen",
                        "organism": "Acinetobacter baumannii",
                    })
                    node_ids.add(p)

            edges.append({
                "source": pa,
                "target": pb,
                "type": "ppi",
                "score": score,
            })

    # Add STRING partner edges from mimicry_with_partners
    if MIMICRY.exists():
        mdf = pd.read_csv(MIMICRY)
        if "string_partners" in mdf.columns:
            for _, row in mdf.iterrows():
                target_gene = row.get("target_gene", "")
                partners_str = row.get("string_partners", "")
                if pd.isna(partners_str) or not partners_str:
                    continue
                for partner in str(partners_str).split(";"):
                    partner = partner.strip()
                    if not partner:
                        continue
                    if partner not in node_ids:
                        nodes.append({
                            "id": partner,
                            "label": partner,
                            "type": "human",
                            "organism": "Homo sapiens",
                        })
                        node_ids.add(partner)
                    edges.append({
                        "source": target_gene,
                        "target": partner,
                        "type": "string",
                        "score": row.get("top_partner_score", 0),
                    })

    return {
        "metadata": {
            **metadata,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "edge_types": list({e["type"] for e in edges}),
        },
        "nodes": nodes,
        "edges": edges,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PathogenScope — Dashboard Export")
    print("=" * 60)

    if not any(p.exists() for p in [MERGED, MIMICRY, NETWORK]):
        print("\nERROR: No result files found. Run upstream scripts first.")
        sys.exit(1)

    print("\nBuilding dashboard data...")
    data = build_dashboard_data()

    DASHBOARD_JSON.write_text(json.dumps(data, indent=2))
    print(f"\nSaved: {DASHBOARD_JSON}")
    print(f"  Nodes: {data['metadata']['node_count']}")
    print(f"  Edges: {data['metadata']['edge_count']}")
    print(f"  Edge types: {data['metadata']['edge_types']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
