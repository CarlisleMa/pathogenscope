"""
Community Interpretation Agent.

Analyzes each PPI community to determine:
  1. Functional coherence — GO term enrichment within the community
  2. Biological pathway assignment — what process does this module represent?
  3. Therapeutic relevance — is disrupting this module therapeutically interesting?
"""

from collections import Counter

import pandas as pd

from .base import BaseAgent, AgentResult


# Pathway keywords mapped to therapeutic relevance
THERAPEUTIC_PATHWAYS = {
    "dna replication": "high — essential for proliferation",
    "dna repair": "high — synthetic lethality potential",
    "cell division": "high — essential for growth",
    "cell wall": "high — antibacterial target space",
    "peptidoglycan": "high — proven drug target class",
    "ribosome": "high — translation is a validated target",
    "translation": "high — aminoglycoside/macrolide target",
    "transcription": "moderate — rifamycin target class",
    "protein folding": "moderate — chaperone inhibitors emerging",
    "secretion": "moderate — virulence factor pathway",
    "type ii secretion": "moderate — virulence",
    "type iii secretion": "high — anti-virulence target",
    "metabolism": "low — broad, may hit host",
    "atp synthesis": "moderate — bedaquiline target class",
    "electron transport": "moderate — energy metabolism target",
    "recombination": "moderate — DNA maintenance",
    "protease": "high — proven druggable family",
    "lipid": "moderate — membrane biogenesis",
}


class CommunityInterpretationAgent(BaseAgent):
    name = "community_interpretation"

    def run(
        self,
        community_stats: pd.DataFrame,
        annotations: pd.DataFrame | None = None,
        **kwargs,
    ) -> AgentResult:
        result = AgentResult(agent_name=self.name)

        communities = community_stats.groupby("ppi_community")
        interpretations = []

        for comm_id, group in communities:
            print(f"  [{self.name}] Interpreting community {comm_id} "
                  f"({len(group)} proteins)...")

            members = group["protein_id"].tolist()
            accessions = self._extract_accessions(members)

            # 1. Extract functional terms
            go_terms = self._collect_go_terms(accessions, annotations)
            gene_names = self._collect_gene_names(members, annotations, accessions)

            # 2. Identify dominant function
            function_label = self._assign_function(go_terms, gene_names)

            # 3. Assess therapeutic relevance
            relevance = self._assess_relevance(function_label, go_terms, gene_names)

            # 4. Community topology summary
            topo = {
                "size": len(group),
                "total_edges": int(group["ppi_edges"].sum() // 2),
                "avg_contact_score": round(float(group["ppi_mean_contact_score"].mean()), 3),
                "max_betweenness": round(float(group["betweenness_centrality"].max()), 4),
                "hub_protein": group.loc[group["ppi_edges"].idxmax(), "protein_id"],
            }

            interpretations.append({
                "community_id": int(comm_id),
                "function_label": function_label,
                "therapeutic_relevance": relevance,
                "topology": topo,
                "top_go_terms": go_terms[:10],
                "gene_names": gene_names[:10],
                "members": members,
            })

        result.data["interpretations"] = interpretations
        result.report = self._format_report(interpretations)
        return result

    def _extract_accessions(self, protein_ids: list[str]) -> list[str]:
        accessions = []
        for pid in protein_ids:
            parts = pid.split("|")
            accessions.append(parts[1] if len(parts) >= 2 else parts[0])
        return accessions

    def _collect_go_terms(
        self, accessions: list[str], annotations: pd.DataFrame | None
    ) -> list[tuple[str, int]]:
        """Collect and count GO terms for community members."""
        if annotations is None or len(annotations) == 0:
            return []

        go_cols = [c for c in annotations.columns if "go" in c.lower() or "GO" in c]
        if not go_cols:
            return []

        acc_col = "protein_id" if "protein_id" in annotations.columns else annotations.columns[0]
        member_ann = annotations[annotations[acc_col].isin(accessions)]

        all_terms = []
        for col in go_cols:
            for val in member_ann[col].dropna():
                # GO terms are often semicolon-separated
                terms = str(val).split(";")
                all_terms.extend(t.strip() for t in terms if t.strip())

        counts = Counter(all_terms)
        return counts.most_common(20)

    def _collect_gene_names(
        self, protein_ids: list[str], annotations: pd.DataFrame | None,
        accessions: list[str],
    ) -> list[str]:
        """Collect gene names for community members."""
        # Extract from FASTA-style IDs (e.g., sp|Q49425|RUVB_MYCGE → RUVB)
        names = []
        for pid in protein_ids:
            parts = pid.split("|")
            if len(parts) >= 3:
                name = parts[2].split("_")[0]
                names.append(name)
            else:
                names.append(pid)

        # Supplement from annotations
        if annotations is not None and len(annotations) > 0:
            for col in ["Gene Names", "gene_names", "Gene names"]:
                if col in annotations.columns:
                    acc_col = ("protein_id" if "protein_id" in annotations.columns
                               else annotations.columns[0])
                    member_ann = annotations[annotations[acc_col].isin(accessions)]
                    for val in member_ann[col].dropna():
                        for name in str(val).split():
                            if name not in names:
                                names.append(name)
                    break

        return names

    def _assign_function(
        self, go_terms: list[tuple[str, int]], gene_names: list[str]
    ) -> str:
        """Assign a consensus function label to the community."""
        # Combine all text for keyword matching
        all_text = " ".join(name.lower() for name in gene_names)
        if go_terms:
            all_text += " " + " ".join(term.lower() for term, _ in go_terms)

        # Match against known pathways
        for pathway in THERAPEUTIC_PATHWAYS:
            if pathway in all_text:
                return pathway

        # Fallback: use most common GO term or gene name pattern
        if go_terms:
            return go_terms[0][0]
        if gene_names:
            return f"{gene_names[0]}-containing module"
        return "uncharacterized module"

    def _assess_relevance(
        self, function_label: str, go_terms: list[tuple[str, int]],
        gene_names: list[str],
    ) -> str:
        """Assess therapeutic relevance of disrupting this community."""
        for pathway, relevance in THERAPEUTIC_PATHWAYS.items():
            if pathway in function_label.lower():
                return relevance

        # Check gene names for known target classes
        combined = " ".join(gene_names).lower()
        for pathway, relevance in THERAPEUTIC_PATHWAYS.items():
            if any(kw in combined for kw in pathway.split()):
                return relevance

        return "unknown — needs manual review"

    def _format_report(self, interpretations: list[dict]) -> str:
        lines = ["# Community Interpretation Report\n"]
        for interp in interpretations:
            lines.append(f"## Community {interp['community_id']}: {interp['function_label']}")
            topo = interp["topology"]
            lines.append(f"  Size: {topo['size']} proteins, {topo['total_edges']} edges")
            lines.append(f"  Hub: {topo['hub_protein']}")
            lines.append(f"  Avg contact score: {topo['avg_contact_score']}")
            lines.append(f"  Therapeutic relevance: {interp['therapeutic_relevance']}")
            genes = ", ".join(interp["gene_names"][:5])
            lines.append(f"  Key genes: {genes}")
            if interp["top_go_terms"]:
                top3 = ", ".join(f"{t} ({c})" for t, c in interp["top_go_terms"][:3])
                lines.append(f"  Top GO terms: {top3}")
            lines.append("")
        return "\n".join(lines)
