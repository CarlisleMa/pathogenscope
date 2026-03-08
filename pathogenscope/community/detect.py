"""
Louvain community detection on FlashPPI interaction networks.

Takes FlashPPI predictions CSV and builds a weighted graph,
then runs Louvain clustering to identify functional protein modules.
"""

import community as community_louvain
import networkx as nx
import pandas as pd


def build_graph(predictions: pd.DataFrame) -> nx.Graph:
    """Build a weighted NetworkX graph from FlashPPI predictions."""
    G = nx.Graph()
    for _, row in predictions.iterrows():
        G.add_edge(
            row["query_id"],
            row["match_id"],
            weight=row["contact_score"],
        )
    return G


def detect_communities(G: nx.Graph, resolution: float = 1.0) -> dict[str, int]:
    """Run Louvain community detection. Returns {protein_id: community_id}."""
    if len(G) == 0:
        return {}
    partition = community_louvain.best_partition(
        G, weight="weight", resolution=resolution
    )
    return partition


def community_stats(G: nx.Graph, partition: dict[str, int]) -> pd.DataFrame:
    """Compute per-protein stats: community, degree, edges, betweenness."""
    if len(G) == 0:
        return pd.DataFrame()

    betweenness = nx.betweenness_centrality(G, weight="weight")

    rows = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        edge_weights = [G[node][n]["weight"] for n in neighbors]
        rows.append(
            {
                "protein_id": node,
                "ppi_community": partition.get(node, -1),
                "ppi_edges": len(neighbors),
                "ppi_degree": G.degree(node),
                "ppi_min_contact_score": min(edge_weights) if edge_weights else 0,
                "ppi_max_contact_score": max(edge_weights) if edge_weights else 0,
                "ppi_mean_contact_score": (
                    sum(edge_weights) / len(edge_weights) if edge_weights else 0
                ),
                "betweenness_centrality": betweenness.get(node, 0),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "betweenness_centrality", ascending=False
    ).reset_index(drop=True)


def label_communities(
    community_df: pd.DataFrame,
    annotations: pd.DataFrame | None = None,
    annotation_col: str = "annotation",
    protein_col: str = "protein_id",
) -> dict[int, str]:
    """
    Generate consensus labels for each community from annotation keywords.
    Returns {community_id: consensus_label}.
    """
    if annotations is None or annotation_col not in annotations.columns:
        # Return generic labels
        communities = community_df["ppi_community"].unique()
        return {c: f"Module_{c}" for c in communities}

    merged = community_df.merge(annotations, on=protein_col, how="left")
    labels = {}

    for comm_id, group in merged.groupby("ppi_community"):
        ann_values = group[annotation_col].dropna().str.lower()
        if len(ann_values) == 0:
            labels[comm_id] = f"Module_{comm_id}"
            continue

        # Count word frequencies (skip common stopwords)
        stopwords = {
            "protein", "putative", "uncharacterized", "hypothetical",
            "predicted", "probable", "domain", "family", "like",
            "containing", "related", "the", "and", "of", "in", "to",
        }
        word_counts: dict[str, int] = {}
        for ann in ann_values:
            for word in ann.split():
                word = word.strip(".,;:()")
                if len(word) > 2 and word not in stopwords:
                    word_counts[word] = word_counts.get(word, 0) + 1

        if word_counts:
            top_words = sorted(word_counts, key=word_counts.get, reverse=True)[:3]
            labels[comm_id] = " ".join(top_words)
        else:
            labels[comm_id] = f"Module_{comm_id}"

    return labels


def run(
    predictions_csv: str,
    output_path: str | None = None,
    resolution: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Run full community detection pipeline."""
    predictions = pd.read_csv(predictions_csv)
    G = build_graph(predictions)
    partition = detect_communities(G, resolution=resolution)
    stats = community_stats(G, partition)

    print(f"Found {len(set(partition.values()))} communities across {len(G)} proteins")
    print(f"Largest community: {max(pd.Series(partition).value_counts())} proteins")

    if output_path:
        stats.to_csv(output_path, index=False)
        print(f"Saved community stats to {output_path}")

    return stats, partition
