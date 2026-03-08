"""
Report Generator Agent.

Compiles outputs from all analysis agents into a unified target dossier:
  - Per-protein summary cards
  - Per-community summary
  - Prioritized actionable recommendations
"""

import os
from datetime import datetime

import pandas as pd

from .base import BaseAgent, AgentResult


class ReportGeneratorAgent(BaseAgent):
    name = "report_generator"

    def run(
        self,
        target_scores: pd.DataFrame,
        validation: AgentResult | None = None,
        druggability: AgentResult | None = None,
        community: AgentResult | None = None,
        hotspot: AgentResult | None = None,
        disruption: AgentResult | None = None,
        chemical_matter: AgentResult | None = None,
        literature: AgentResult | None = None,
        output_dir: str = "results",
        **kwargs,
    ) -> AgentResult:
        result = AgentResult(agent_name=self.name)

        sections = []
        sections.append(self._header(target_scores))
        sections.append(self._executive_summary(
            target_scores, validation, druggability, community
        ))
        sections.append(self._target_cards(
            target_scores, validation, druggability
        ))

        if community and community.status == "success":
            sections.append(self._community_section(community))

        if hotspot and hotspot.status != "error":
            sections.append(self._interface_section(hotspot))

        if disruption and disruption.status != "error":
            sections.append(self._strategy_section(disruption))

        if chemical_matter and chemical_matter.status == "success":
            sections.append(self._chemical_matter_section(chemical_matter))

        if literature and literature.status == "success":
            sections.append(self._literature_section(literature))

        sections.append(self._recommendations(
            target_scores, validation, druggability, disruption,
            chemical_matter, literature
        ))
        sections.append(self._methods())

        full_report = "\n\n---\n\n".join(sections)
        result.data["report"] = full_report
        result.report = full_report

        # Save to file
        report_path = os.path.join(output_dir, "target_dossier.md")
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(full_report)
        print(f"  [{self.name}] Report saved to {report_path}")

        result.data["report_path"] = report_path
        return result

    def _header(self, target_scores: pd.DataFrame) -> str:
        n_proteins = len(target_scores)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"# PathogenScope Target Dossier\n\n"
            f"**Generated**: {timestamp}\n"
            f"**Proteins analyzed**: {n_proteins}\n"
            f"**Pipeline**: FlashPPI → Louvain communities → Target scoring → Agent analysis"
        )

    def _executive_summary(
        self, targets: pd.DataFrame,
        validation: AgentResult | None,
        druggability: AgentResult | None,
        community: AgentResult | None,
    ) -> str:
        lines = ["## Executive Summary\n"]

        # Top targets
        top5 = targets.head(5)
        lines.append(f"**Top 5 targets** by combined score (centrality + hub + confidence):\n")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            pid = row["protein_id"]
            name = pid.split("|")[-1] if "|" in pid else pid
            lines.append(f"{i}. **{name}** — score={row['target_score']:.3f}, "
                         f"edges={int(row['ppi_edges'])}, "
                         f"community={int(row['ppi_community'])}")

        # Validation highlights
        if validation and validation.status == "success":
            summary = validation.data.get("summary", {})
            lines.append(f"\n**Validation**: {summary.get('essential_genes', 0)} essential genes, "
                         f"{summary.get('high_offtarget_risk', 0)} with high off-target risk")

        # Druggability highlights
        if druggability and druggability.status == "success":
            summary = druggability.data.get("summary", {})
            tiers = summary.get("tier_counts", {})
            lines.append(f"\n**Druggability**: {tiers.get('Tier 1', 0)} tier-1, "
                         f"{tiers.get('Tier 2', 0)} tier-2, "
                         f"{tiers.get('Tier 3', 0)} tier-3 targets")

        # Communities
        if community and community.status == "success":
            interps = community.data.get("interpretations", [])
            lines.append(f"\n**Communities**: {len(interps)} functional modules identified")
            for interp in interps[:3]:
                lines.append(f"  - Community {interp['community_id']}: "
                             f"{interp['function_label']} "
                             f"({interp['topology']['size']} proteins)")

        return "\n".join(lines)

    def _target_cards(
        self, targets: pd.DataFrame,
        validation: AgentResult | None,
        druggability: AgentResult | None,
    ) -> str:
        lines = ["## Target Cards\n"]

        # Index validation/druggability by protein_id
        val_map = {}
        if validation and validation.status == "success":
            for v in validation.data.get("validations", []):
                val_map[v["protein_id"]] = v

        drug_map = {}
        if druggability and druggability.status == "success":
            for a in druggability.data.get("assessments", []):
                drug_map[a["protein_id"]] = a

        for _, row in targets.head(10).iterrows():
            pid = row["protein_id"]
            name = pid.split("|")[-1] if "|" in pid else pid
            lines.append(f"### {name}")
            lines.append(f"- **ID**: {pid}")
            lines.append(f"- **Target score**: {row['target_score']:.3f}")
            lines.append(f"- **PPI edges**: {int(row['ppi_edges'])}")
            lines.append(f"- **Community**: {int(row['ppi_community'])}")
            lines.append(f"- **Betweenness centrality**: {row['betweenness_centrality']:.4f}")
            lines.append(f"- **Max contact score**: {row['ppi_max_contact_score']:.4f}")

            if pid in val_map:
                v = val_map[pid]
                lines.append(f"- **Essentiality**: {v['is_essential']}")
                lines.append(f"- **Off-target risk**: {v['off_target_risk']}")
                if v["drug_target_signals"]:
                    lines.append(f"- **Drug signals**: {', '.join(v['drug_target_signals'])}")

            if pid in drug_map:
                d = drug_map[pid]
                lines.append(f"- **Druggability**: {d['druggability_tier']}")
                lines.append(f"- **AlphaFold**: {'yes' if d['alphafold_structure'] else 'no'}")
                if d["family_match"]:
                    lines.append(f"- **Family**: {d['family_match']}")

            lines.append("")

        return "\n".join(lines)

    def _community_section(self, community: AgentResult) -> str:
        return "## Community Analysis\n\n" + community.report

    def _interface_section(self, hotspot: AgentResult) -> str:
        return "## Interface Analysis\n\n" + hotspot.report

    def _strategy_section(self, disruption: AgentResult) -> str:
        return "## Disruption Strategies\n\n" + disruption.report

    def _chemical_matter_section(self, chemical_matter: AgentResult) -> str:
        summary = chemical_matter.data.get("summary", {})
        header_lines = [
            "## Chemical Matter\n",
            f"**{summary.get('total_direct_compounds', 0)}** compounds found across "
            f"**{summary.get('total_assessed', 0)}** targets. "
            f"**{summary.get('with_clinical_compounds', 0)}** targets have clinical-stage compounds.\n",
        ]
        breakdown = summary.get("tractability_breakdown", {})
        if breakdown:
            header_lines.append("Tractability breakdown:")
            for level, count in sorted(breakdown.items()):
                header_lines.append(f"  - {level}: {count} targets")
            header_lines.append("")
        return "\n".join(header_lines) + "\n" + chemical_matter.report

    def _literature_section(self, literature: AgentResult) -> str:
        summary = literature.data.get("summary", {})
        header_lines = [
            "## Literature & Prior Perturbation Strategies\n",
            f"Searched **{summary.get('targets_searched', 0)}** targets across PubMed. "
            f"**{summary.get('total_papers_found', 0)}** relevant papers found.\n",
            f"- {summary.get('with_inhibitor_literature', 0)} targets with inhibitor studies",
            f"- {summary.get('with_genetic_evidence', 0)} targets with genetic perturbation data",
            f"- {summary.get('with_resistance_data', 0)} targets with resistance mechanism reports\n",
        ]
        return "\n".join(header_lines) + "\n" + literature.report

    def _recommendations(
        self, targets: pd.DataFrame,
        validation: AgentResult | None,
        druggability: AgentResult | None,
        disruption: AgentResult | None,
        chemical_matter: AgentResult | None = None,
        literature: AgentResult | None = None,
    ) -> str:
        lines = ["## Prioritized Recommendations\n"]

        # Index all agent results by protein_id
        val_map = {}
        if validation and validation.status == "success":
            for v in validation.data.get("validations", []):
                val_map[v["protein_id"]] = v

        drug_map = {}
        if druggability and druggability.status == "success":
            for a in druggability.data.get("assessments", []):
                drug_map[a["protein_id"]] = a

        chem_map = {}
        if chemical_matter and chemical_matter.status == "success":
            for a in chemical_matter.data.get("assessments", []):
                chem_map[a["protein_id"]] = a

        lit_map = {}
        if literature and literature.status == "success":
            for a in literature.data.get("assessments", []):
                lit_map[a["protein_id"]] = a

        recommendations = []
        for _, row in targets.head(20).iterrows():
            pid = row["protein_id"]
            rec_score = row["target_score"]

            # Boost for essentiality
            if pid in val_map:
                v = val_map[pid]
                if "essential" in str(v["is_essential"]):
                    rec_score += 0.2
                if "HIGH" in v["off_target_risk"]:
                    rec_score -= 0.3

            # Boost for druggability
            if pid in drug_map:
                d = drug_map[pid]
                if "Tier 1" in d["druggability_tier"]:
                    rec_score += 0.2
                elif "Tier 2" in d["druggability_tier"]:
                    rec_score += 0.1

            # Boost for existing chemical matter
            if pid in chem_map:
                tract = chem_map[pid]["tractability"]["level"]
                if tract == "clinical":
                    rec_score += 0.15
                elif tract == "chemical_probe":
                    rec_score += 0.1
                elif tract == "hit_matter":
                    rec_score += 0.05

            # Boost for literature-backed essentiality/inhibitor evidence
            if pid in lit_map:
                findings = lit_map[pid]["findings"]
                if findings.get("known_inhibitors"):
                    rec_score += 0.05
                if findings.get("essential_evidence"):
                    rec_score += 0.05
                # Penalty for known resistance — harder to develop
                if len(findings.get("resistance_mechanisms", [])) >= 3:
                    rec_score -= 0.1

            recommendations.append((pid, rec_score))

        recommendations.sort(key=lambda x: x[1], reverse=True)

        lines.append("### Priority targets for follow-up\n")
        for i, (pid, score) in enumerate(recommendations[:5], 1):
            name = pid.split("|")[-1] if "|" in pid else pid
            reasons = []
            if pid in val_map and "essential" in str(val_map[pid]["is_essential"]):
                reasons.append("essential gene")
            if pid in drug_map and "Tier 1" in drug_map[pid]["druggability_tier"]:
                reasons.append("highly druggable")
            if pid in val_map and "low" in val_map[pid]["off_target_risk"]:
                reasons.append("low off-target risk")
            if pid in chem_map:
                tract = chem_map[pid]["tractability"]["level"]
                if tract in ("clinical", "chemical_probe"):
                    reasons.append(f"chemical matter: {tract}")
            if pid in lit_map and lit_map[pid]["findings"].get("known_inhibitors"):
                reasons.append("prior inhibitor studies")

            reason_str = f" ({', '.join(reasons)})" if reasons else ""
            lines.append(f"{i}. **{name}** — priority score={score:.3f}{reason_str}")

        lines.append("\n### Suggested next steps\n")
        lines.append("1. Run full proteome analysis on GPU for comprehensive coverage")
        lines.append("2. Validate top targets against SeqHub results")
        lines.append("3. Obtain AlphaFold structures for top 5 targets")
        lines.append("4. Run molecular docking on druggable interface pockets")
        lines.append("5. Retrieve and cluster ChEMBL hit compounds for SAR analysis")
        lines.append("6. Review PubMed literature for resistance liability of top picks")

        return "\n".join(lines)

    def _methods(self) -> str:
        return (
            "## Methods\n\n"
            "**PPI Prediction**: FlashPPI (tattabio/flashppi) — 3-stage pipeline: "
            "protein embedding → FAISS nearest-neighbor retrieval → residue-level contact scoring.\n\n"
            "**Community Detection**: Louvain algorithm (networkx) with weighted edges.\n\n"
            "**Target Scoring**: Weighted combination of betweenness centrality (0.4), "
            "hub connectivity (0.3), and max contact confidence (0.3).\n\n"
            "**Validation**: UniProt annotations, Database of Essential Genes (DEG), "
            "human homology check.\n\n"
            "**Druggability**: AlphaFold structure availability, protein family classification, "
            "known binding sites, subcellular accessibility.\n\n"
            "**Interface Analysis**: FlashPPI contact maps — hotspot identification, "
            "patch extraction, domain mapping via UniProt/Pfam.\n\n"
            "**Chemical Matter**: ChEMBL database search for bioactive compounds — "
            "direct target hits, protein family analogues, clinical pipeline compounds.\n\n"
            "**Literature Search**: PubMed (NCBI E-utilities) — inhibitor studies, "
            "genetic perturbation experiments, resistance mechanisms, pathway-level evidence."
        )
