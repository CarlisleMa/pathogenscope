"""
Literature Search Agent.

Searches PubMed for perturbation experiments and inhibition strategies
already attempted on each target protein, its orthologs, or pathway
neighbors. Summarizes what has been tried, what worked, and what failed:
  1. Gene-specific knockout/knockdown studies
  2. Inhibitor/compound studies on orthologs
  3. Pathway-level perturbation experiments
  4. Resistance mechanisms reported
"""

import re
import requests
import pandas as pd

from .base import BaseAgent, AgentResult

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Search templates targeting perturbation/inhibition experiments
SEARCH_STRATEGIES = [
    {
        "name": "inhibition",
        "query_suffix": '(inhibitor OR inhibition OR "small molecule" OR compound)',
        "category": "inhibitor_studies",
    },
    {
        "name": "knockout",
        "query_suffix": "(knockout OR knockdown OR deletion OR essential OR lethal)",
        "category": "genetic_perturbation",
    },
    {
        "name": "resistance",
        "query_suffix": "(resistance OR mutation OR resistant OR susceptibility)",
        "category": "resistance",
    },
    {
        "name": "structure_drug",
        "query_suffix": '("drug target" OR "crystal structure" OR "binding site" OR druggable)',
        "category": "structural_druggability",
    },
]


class LiteratureSearchAgent(BaseAgent):
    name = "literature_search"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.top_n = self.config.get("top_n", 20)
        self.max_results_per_query = self.config.get("max_results_per_query", 5)

    def run(
        self,
        target_scores: pd.DataFrame,
        community: AgentResult | None = None,
        disruption: AgentResult | None = None,
        annotations: pd.DataFrame | None = None,
        **kwargs,
    ) -> AgentResult:
        """
        Search PubMed for perturbation strategies on top targets.

        Args:
            target_scores: Ranked targets from pipeline
            community: CommunityInterpretationAgent results (for pathway context)
            disruption: DisruptionStrategyAgent results (for modality context)
            annotations: UniProt annotations (for gene name extraction)
        """
        result = AgentResult(agent_name=self.name)

        # Build pathway context from community results
        pathway_map = {}
        if community and community.status == "success":
            for interp in community.data.get("interpretations", []):
                for member in interp.get("members", []):
                    pathway_map[member] = interp.get("function_label", "")

        # Build gene name map from annotations
        gene_map = self._build_gene_map(target_scores, annotations)

        assessments = []
        for _, row in target_scores.head(self.top_n).iterrows():
            pid = row["protein_id"]
            gene_name = gene_map.get(pid)
            pathway = pathway_map.get(pid)

            if not gene_name:
                self._warn(result, f"No gene name for {pid}, skipping literature search")
                continue

            print(f"  [{self.name}] Searching PubMed for {gene_name}...")
            assessment = self._search_target(pid, gene_name, pathway)
            assessments.append(assessment)

        result.data["assessments"] = assessments
        result.data["summary"] = self._summarize(assessments)
        result.report = self._format_report(assessments)
        return result

    def _build_gene_map(
        self, target_scores: pd.DataFrame, annotations: pd.DataFrame | None
    ) -> dict[str, str]:
        """Extract gene names from protein IDs and annotations."""
        gene_map = {}
        for pid in target_scores["protein_id"].head(self.top_n):
            # Try FASTA ID first: sp|Q49425|RUVB_MYCGE -> RUVB
            if "|" in pid:
                suffix = pid.split("|")[-1]
                gene = suffix.split("_")[0] if "_" in suffix else suffix
                gene_map[pid] = gene

        # Supplement from annotations if available
        if annotations is not None:
            gene_cols = [c for c in annotations.columns if "gene" in c.lower()]
            if gene_cols:
                for _, row in annotations.iterrows():
                    # Match by accession
                    acc = row.get("From", row.get("accession", ""))
                    matching_pids = [
                        p for p in gene_map
                        if acc and acc in p
                    ]
                    for pid in matching_pids:
                        gene_val = str(row[gene_cols[0]])
                        if gene_val and gene_val != "nan":
                            # Use the shorter/cleaner annotation gene name
                            clean = gene_val.split()[0].rstrip(";")
                            gene_map[pid] = clean

        return gene_map

    def _search_target(self, protein_id: str, gene_name: str, pathway: str | None) -> dict:
        """Run all search strategies for one target."""
        categories = {}

        for strategy in SEARCH_STRATEGIES:
            query = f"{gene_name} {strategy['query_suffix']}"
            papers = self._search_pubmed(query)
            categories[strategy["category"]] = papers

        # Pathway-level search if we have context
        pathway_papers = []
        if pathway:
            pathway_query = f'"{pathway}" (inhibitor OR target OR antimicrobial)'
            pathway_papers = self._search_pubmed(pathway_query)

        # Extract key findings across all papers
        all_papers = []
        for cat_papers in categories.values():
            all_papers.extend(cat_papers)
        all_papers.extend(pathway_papers)

        findings = self._extract_findings(all_papers, gene_name)

        return {
            "protein_id": protein_id,
            "gene_name": gene_name,
            "pathway": pathway,
            "inhibitor_studies": categories.get("inhibitor_studies", []),
            "genetic_perturbation": categories.get("genetic_perturbation", []),
            "resistance": categories.get("resistance", []),
            "structural_druggability": categories.get("structural_druggability", []),
            "pathway_studies": pathway_papers,
            "total_papers": len(set(p["pmid"] for p in all_papers)),
            "findings": findings,
        }

    def _search_pubmed(self, query: str) -> list[dict]:
        """Search PubMed and return paper summaries."""
        try:
            # Step 1: ESearch to get PMIDs
            search_url = (
                f"{EUTILS_BASE}/esearch.fcgi"
                f"?db=pubmed&term={requests.utils.quote(query)}"
                f"&retmax={self.max_results_per_query}&sort=relevance"
                f"&retmode=json"
            )
            resp = requests.get(search_url, timeout=15)
            if not resp.ok:
                return []

            data = resp.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []

            # Step 2: EFetch to get paper details
            pmid_str = ",".join(pmids)
            fetch_url = (
                f"{EUTILS_BASE}/esummary.fcgi"
                f"?db=pubmed&id={pmid_str}&retmode=json"
            )
            resp = requests.get(fetch_url, timeout=15)
            if not resp.ok:
                return []

            data = resp.json()
            papers = []
            for pmid in pmids:
                info = data.get("result", {}).get(pmid, {})
                if not info or "error" in info:
                    continue

                title = info.get("title", "")
                authors = info.get("authors", [])
                first_author = authors[0].get("name", "") if authors else ""
                pub_date = info.get("pubdate", "")
                journal = info.get("source", "")

                papers.append({
                    "pmid": pmid,
                    "title": title,
                    "first_author": first_author,
                    "year": pub_date[:4] if pub_date else "",
                    "journal": journal,
                })

            return papers
        except (requests.RequestException, ValueError, KeyError):
            return []

    def _extract_findings(self, papers: list[dict], gene_name: str) -> dict:
        """Classify findings from paper titles into actionable categories."""
        findings = {
            "known_inhibitors": [],
            "essential_evidence": [],
            "resistance_mechanisms": [],
            "structural_insights": [],
            "key_experiments": [],
        }

        gene_lower = gene_name.lower()
        seen_pmids = set()

        for paper in papers:
            if paper["pmid"] in seen_pmids:
                continue
            seen_pmids.add(paper["pmid"])

            title_lower = paper["title"].lower()
            ref = f"{paper['first_author']} {paper['year']}"

            # Classify by title content
            if any(kw in title_lower for kw in ["inhibitor", "inhibition", "compound", "drug"]):
                findings["known_inhibitors"].append({
                    "reference": ref,
                    "title": paper["title"],
                    "pmid": paper["pmid"],
                })
            if any(kw in title_lower for kw in ["essential", "lethal", "viable", "knockout"]):
                findings["essential_evidence"].append({
                    "reference": ref,
                    "title": paper["title"],
                    "pmid": paper["pmid"],
                })
            if any(kw in title_lower for kw in ["resistance", "resistant", "mutation"]):
                findings["resistance_mechanisms"].append({
                    "reference": ref,
                    "title": paper["title"],
                    "pmid": paper["pmid"],
                })
            if any(kw in title_lower for kw in ["structure", "crystal", "binding", "pocket"]):
                findings["structural_insights"].append({
                    "reference": ref,
                    "title": paper["title"],
                    "pmid": paper["pmid"],
                })

            # Gene-specific key experiments
            if gene_lower in title_lower:
                findings["key_experiments"].append({
                    "reference": ref,
                    "title": paper["title"],
                    "pmid": paper["pmid"],
                })

        return findings

    def _extract_accession(self, protein_id: str) -> str:
        parts = protein_id.split("|")
        return parts[1] if len(parts) >= 2 else parts[0]

    def _summarize(self, assessments: list[dict]) -> dict:
        total_papers = sum(a["total_papers"] for a in assessments)
        with_inhibitor_literature = sum(
            1 for a in assessments if a["inhibitor_studies"]
        )
        with_genetic_evidence = sum(
            1 for a in assessments if a["genetic_perturbation"]
        )
        with_resistance_data = sum(
            1 for a in assessments if a["resistance"]
        )

        return {
            "targets_searched": len(assessments),
            "total_papers_found": total_papers,
            "with_inhibitor_literature": with_inhibitor_literature,
            "with_genetic_evidence": with_genetic_evidence,
            "with_resistance_data": with_resistance_data,
        }

    def _format_report(self, assessments: list[dict]) -> str:
        lines = ["# Literature & Perturbation Strategy Report\n"]

        for a in assessments:
            name = a["gene_name"]
            pid_suffix = a["protein_id"].split("|")[-1] if "|" in a["protein_id"] else a["protein_id"]
            lines.append(f"## {name} ({pid_suffix})")

            if a["pathway"]:
                lines.append(f"  Pathway context: {a['pathway']}")

            lines.append(f"  Total papers found: {a['total_papers']}")

            # Inhibitor studies
            if a["inhibitor_studies"]:
                lines.append(f"\n  **Inhibitor studies** ({len(a['inhibitor_studies'])} papers):")
                for p in a["inhibitor_studies"][:3]:
                    lines.append(f"    - {p['title']}")
                    lines.append(f"      {p['first_author']} ({p['year']}) PMID:{p['pmid']}")
            else:
                lines.append("\n  **Inhibitor studies**: None found")

            # Genetic perturbation
            if a["genetic_perturbation"]:
                lines.append(f"\n  **Genetic perturbation** ({len(a['genetic_perturbation'])} papers):")
                for p in a["genetic_perturbation"][:3]:
                    lines.append(f"    - {p['title']}")
                    lines.append(f"      {p['first_author']} ({p['year']}) PMID:{p['pmid']}")

            # Resistance
            if a["resistance"]:
                lines.append(f"\n  **Resistance mechanisms** ({len(a['resistance'])} papers):")
                for p in a["resistance"][:3]:
                    lines.append(f"    - {p['title']}")
                    lines.append(f"      {p['first_author']} ({p['year']}) PMID:{p['pmid']}")

            # Structural / druggability
            if a["structural_druggability"]:
                lines.append(f"\n  **Structural/druggability** ({len(a['structural_druggability'])} papers):")
                for p in a["structural_druggability"][:3]:
                    lines.append(f"    - {p['title']}")
                    lines.append(f"      {p['first_author']} ({p['year']}) PMID:{p['pmid']}")

            # Pathway studies
            if a["pathway_studies"]:
                lines.append(f"\n  **Pathway-level studies** ({len(a['pathway_studies'])} papers):")
                for p in a["pathway_studies"][:3]:
                    lines.append(f"    - {p['title']}")
                    lines.append(f"      {p['first_author']} ({p['year']}) PMID:{p['pmid']}")

            # Key findings summary
            findings = a["findings"]
            has_findings = any(
                findings[k] for k in findings
            )
            if has_findings:
                lines.append("\n  **Summary of prior art**:")
                if findings["known_inhibitors"]:
                    lines.append(f"    Inhibitors reported: {len(findings['known_inhibitors'])} studies")
                if findings["essential_evidence"]:
                    lines.append(f"    Essentiality evidence: {len(findings['essential_evidence'])} studies")
                if findings["resistance_mechanisms"]:
                    lines.append(f"    Resistance data: {len(findings['resistance_mechanisms'])} studies — "
                                 "review before committing to this target")
                if findings["structural_insights"]:
                    lines.append(f"    Structural data: {len(findings['structural_insights'])} studies")

            lines.append("")

        return "\n".join(lines)
