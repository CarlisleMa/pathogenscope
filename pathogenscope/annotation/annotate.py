"""
Downstream annotation pipeline.

Fetches functional annotations from UniProt for proteins in the PPI network,
then combines them with community detection results to produce a
target-prioritization table.
"""

import time

import pandas as pd
import requests


UNIPROT_FIELDS = [
    "accession",
    "gene_names",
    "protein_name",
    "organism_name",
    "go_f",       # GO molecular function
    "go_p",       # GO biological process
    "go_c",       # GO cellular component
    "xref_pfam",  # Pfam domains
    "xref_interpro",
    "cc_subcellular_location",
    "cc_function",
    "keyword",
    "length",
]


def fetch_uniprot_batch(
    accessions: list[str],
    batch_size: int = 100,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Fetch annotations from UniProt REST API for a list of accessions."""
    all_results = []
    fields_str = ",".join(UNIPROT_FIELDS)

    for i in range(0, len(accessions), batch_size):
        batch = accessions[i : i + batch_size]
        query = " OR ".join(f"accession:{acc}" for acc in batch)
        url = (
            f"https://rest.uniprot.org/uniprotkb/search"
            f"?query={query}"
            f"&fields={fields_str}"
            f"&format=tsv"
            f"&size={batch_size}"
        )

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            if len(lines) > 1:
                header = lines[0].split("\t")
                for line in lines[1:]:
                    values = line.split("\t")
                    row = dict(zip(header, values))
                    all_results.append(row)
        except requests.RequestException as e:
            print(f"Warning: UniProt batch fetch failed: {e}")

        if i + batch_size < len(accessions):
            time.sleep(delay)

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    # Normalize column name for merging
    if "Entry" in df.columns:
        df = df.rename(columns={"Entry": "protein_id"})
    return df


def extract_accessions(protein_ids: list[str]) -> list[str]:
    """Extract UniProt accessions from FASTA-style IDs.

    Handles formats like:
      sp|P0A6Y8|DNAK_ECOLI  -> P0A6Y8
      tr|A0A0H3JQN0|...     -> A0A0H3JQN0
      P0A6Y8                 -> P0A6Y8
    """
    accessions = []
    for pid in protein_ids:
        parts = pid.split("|")
        if len(parts) >= 2:
            accessions.append(parts[1])
        else:
            accessions.append(parts[0])
    return accessions


def score_targets(
    community_df: pd.DataFrame,
    annotations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Score proteins for target potential.

    Scoring factors:
      - betweenness_centrality: network bottleneck importance
      - ppi_edges: connectivity (hubs)
      - ppi_community size: module context
      - annotation richness: functional knowledge
    """
    df = community_df.copy()

    # Normalize betweenness to 0-1
    bc_max = df["betweenness_centrality"].max()
    if bc_max > 0:
        df["centrality_score"] = df["betweenness_centrality"] / bc_max
    else:
        df["centrality_score"] = 0

    # Hub score: normalize degree
    deg_max = df["ppi_edges"].max()
    if deg_max > 0:
        df["hub_score"] = df["ppi_edges"] / deg_max
    else:
        df["hub_score"] = 0

    # Confidence score from contact scores
    df["confidence_score"] = df["ppi_max_contact_score"]

    # Combined target score (simple weighted sum)
    df["target_score"] = (
        0.4 * df["centrality_score"]
        + 0.3 * df["hub_score"]
        + 0.3 * df["confidence_score"]
    )

    # Merge annotations if available
    if annotations is not None and len(annotations) > 0:
        # Extract accessions from protein_id for merging
        df["_accession"] = extract_accessions(df["protein_id"].tolist())
        if "protein_id" in annotations.columns:
            annotations = annotations.copy()
            annotations["_accession"] = annotations["protein_id"]
        df = df.merge(annotations, left_on="_accession", right_on="_accession", how="left", suffixes=("", "_ann"))
        df = df.drop(columns=["_accession"], errors="ignore")

    df = df.sort_values("target_score", ascending=False).reset_index(drop=True)
    return df


def run(
    community_csv: str,
    output_path: str | None = None,
    fetch_annotations: bool = True,
) -> pd.DataFrame:
    """Run annotation and target scoring pipeline."""
    community_df = pd.read_csv(community_csv)
    protein_ids = community_df["protein_id"].tolist()

    annotations = None
    if fetch_annotations:
        accessions = extract_accessions(protein_ids)
        print(f"Fetching annotations for {len(accessions)} proteins from UniProt...")
        annotations = fetch_uniprot_batch(accessions)
        if len(annotations) > 0:
            print(f"Got annotations for {len(annotations)} proteins")
        else:
            print("Warning: No annotations retrieved")

    scored = score_targets(community_df, annotations)

    if output_path:
        scored.to_csv(output_path, index=False)
        print(f"Saved target scores to {output_path}")

    return scored
