"""
Druggability Assessment Agent.

For each top target, assesses druggability by checking:
  1. AlphaFold structure availability
  2. Protein properties (membrane, secreted, size)
  3. Known ligands / binding sites from UniProt
  4. Protein family druggability (kinases, proteases, etc.)
"""

import requests
import pandas as pd

from .base import BaseAgent, AgentResult


# Protein families known to be druggable
DRUGGABLE_FAMILIES = {
    "kinase": "high",
    "protease": "high",
    "transferase": "high",
    "oxidoreductase": "high",
    "hydrolase": "high",
    "synthase": "moderate",
    "synthetase": "moderate",
    "ligase": "moderate",
    "polymerase": "moderate",
    "reductase": "high",
    "dehydrogenase": "moderate",
    "topoisomerase": "high",
    "gyrase": "high",
    "ribosom": "high",  # ribosomal targets
    "transporter": "moderate",
    "channel": "moderate",
    "receptor": "high",
    "chaperone": "moderate",
}


class DruggabilityAgent(BaseAgent):
    name = "druggability"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.top_n = self.config.get("top_n", 20)

    def run(
        self,
        target_scores: pd.DataFrame,
        annotations: pd.DataFrame | None = None,
        **kwargs,
    ) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        top = target_scores.head(self.top_n).copy()
        accessions = self._extract_accessions(top["protein_id"].tolist())

        assessments = []
        for acc, pid in zip(accessions, top["protein_id"].tolist()):
            print(f"  [{self.name}] Assessing {acc}...")

            # 1. AlphaFold structure
            alphafold = self._check_alphafold(acc)

            # 2. Protein properties from annotations
            properties = self._get_protein_properties(acc, pid, annotations)

            # 3. Family druggability
            family_score = self._assess_family(pid, annotations, acc)

            # 4. Known binding sites
            binding_sites = self._check_binding_sites(acc)

            # Combine into druggability score
            druggability = self._score_druggability(
                alphafold, properties, family_score, binding_sites
            )

            assessments.append({
                "protein_id": pid,
                "accession": acc,
                "alphafold_structure": alphafold["available"],
                "alphafold_confidence": alphafold.get("avg_plddt"),
                "protein_length": properties.get("length"),
                "is_membrane": properties.get("is_membrane", False),
                "is_secreted": properties.get("is_secreted", False),
                "family_druggability": family_score["level"],
                "family_match": family_score.get("match", ""),
                "has_binding_site": binding_sites["has_sites"],
                "binding_site_count": binding_sites["count"],
                "druggability_score": druggability["score"],
                "druggability_tier": druggability["tier"],
                "druggability_notes": druggability["notes"],
            })

        result.data["assessments"] = assessments
        result.data["summary"] = self._summarize(assessments)
        result.report = self._format_report(assessments)
        return result

    def _extract_accessions(self, protein_ids: list[str]) -> list[str]:
        accessions = []
        for pid in protein_ids:
            parts = pid.split("|")
            accessions.append(parts[1] if len(parts) >= 2 else parts[0])
        return accessions

    def _check_alphafold(self, accession: str) -> dict:
        """Check AlphaFold DB for predicted structure."""
        try:
            url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
            resp = requests.get(url, timeout=10)
            if resp.ok:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    entry = data[0]
                    return {
                        "available": True,
                        "avg_plddt": entry.get("paeAvgPlddt"),
                        "model_url": entry.get("pdbUrl"),
                    }
            return {"available": False}
        except requests.RequestException:
            return {"available": False}

    def _get_protein_properties(
        self, accession: str, protein_id: str, annotations: pd.DataFrame | None
    ) -> dict:
        """Extract protein properties from annotations."""
        props = {"length": None, "is_membrane": False, "is_secreted": False}

        if annotations is None or len(annotations) == 0:
            return props

        # Find the row for this protein
        acc_col = "protein_id" if "protein_id" in annotations.columns else annotations.columns[0]
        row = annotations[annotations[acc_col] == accession]
        if len(row) == 0:
            return props

        row = row.iloc[0]

        # Length
        for col in ["Length", "length"]:
            if col in row.index:
                try:
                    props["length"] = int(row[col])
                except (ValueError, TypeError):
                    pass

        # Subcellular location
        loc_text = ""
        for col in ["Subcellular location [CC]", "cc_subcellular_location"]:
            if col in row.index:
                loc_text = str(row[col]).lower()
                break

        props["is_membrane"] = any(
            kw in loc_text for kw in ["membrane", "transmembrane", "cell membrane"]
        )
        props["is_secreted"] = any(
            kw in loc_text for kw in ["secreted", "extracellular", "exported"]
        )

        return props

    def _assess_family(
        self, protein_id: str, annotations: pd.DataFrame | None, accession: str
    ) -> dict:
        """Assess druggability based on protein family."""
        # Build search text from protein name + annotations
        search_text = protein_id.lower()

        if annotations is not None and len(annotations) > 0:
            acc_col = "protein_id" if "protein_id" in annotations.columns else annotations.columns[0]
            row = annotations[annotations[acc_col] == accession]
            if len(row) > 0:
                row = row.iloc[0]
                for col in ["Protein names", "protein_name", "Gene Names", "gene_names"]:
                    if col in row.index:
                        search_text += " " + str(row[col]).lower()

        for family, level in DRUGGABLE_FAMILIES.items():
            if family in search_text:
                return {"level": level, "match": family}

        return {"level": "unknown", "match": ""}

    def _check_binding_sites(self, accession: str) -> dict:
        """Check UniProt for annotated binding sites."""
        try:
            url = (
                f"https://rest.uniprot.org/uniprotkb/{accession}"
                f"?fields=ft_binding,ft_act_site,ft_site&format=json"
            )
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                return {"has_sites": False, "count": 0}

            data = resp.json()
            features = data.get("features", [])
            binding_features = [
                f for f in features
                if f.get("type") in ("Binding site", "Active site", "Site")
            ]
            return {
                "has_sites": len(binding_features) > 0,
                "count": len(binding_features),
            }
        except (requests.RequestException, ValueError):
            return {"has_sites": False, "count": 0}

    def _score_druggability(
        self, alphafold: dict, properties: dict, family: dict, binding: dict
    ) -> dict:
        """Compute composite druggability score (0-1)."""
        score = 0.0
        notes = []

        # Structure availability (0.25)
        if alphafold["available"]:
            plddt = alphafold.get("avg_plddt")
            if plddt and plddt > 70:
                score += 0.25
                notes.append(f"AlphaFold structure (pLDDT={plddt:.0f})")
            else:
                score += 0.15
                notes.append("AlphaFold structure (low confidence)")
        else:
            notes.append("No predicted structure")

        # Family druggability (0.30)
        family_scores = {"high": 0.30, "moderate": 0.20, "unknown": 0.05}
        score += family_scores.get(family["level"], 0.05)
        if family["match"]:
            notes.append(f"Druggable family: {family['match']}")

        # Binding sites (0.25)
        if binding["has_sites"]:
            score += 0.25
            notes.append(f"{binding['count']} known binding/active site(s)")
        else:
            notes.append("No annotated binding sites")

        # Accessibility bonus (0.20)
        if properties.get("is_membrane") or properties.get("is_secreted"):
            score += 0.20
            if properties["is_membrane"]:
                notes.append("Membrane protein (accessible)")
            if properties["is_secreted"]:
                notes.append("Secreted (accessible)")
        else:
            score += 0.10  # cytoplasmic still targetable

        # Tier assignment
        if score >= 0.7:
            tier = "Tier 1 — highly druggable"
        elif score >= 0.4:
            tier = "Tier 2 — moderately druggable"
        else:
            tier = "Tier 3 — challenging target"

        return {"score": round(score, 3), "tier": tier, "notes": "; ".join(notes)}

    def _summarize(self, assessments: list[dict]) -> dict:
        tiers = {}
        for a in assessments:
            tier = a["druggability_tier"].split("—")[0].strip()
            tiers[tier] = tiers.get(tier, 0) + 1
        with_structure = sum(1 for a in assessments if a["alphafold_structure"])
        return {
            "total_assessed": len(assessments),
            "with_alphafold": with_structure,
            "tier_counts": tiers,
        }

    def _format_report(self, assessments: list[dict]) -> str:
        lines = ["# Druggability Assessment Report\n"]
        for a in assessments:
            lines.append(f"## {a['protein_id']}")
            lines.append(f"  Druggability: {a['druggability_tier']} (score={a['druggability_score']})")
            lines.append(f"  AlphaFold: {'yes' if a['alphafold_structure'] else 'no'}"
                         f" (pLDDT={a['alphafold_confidence'] or 'N/A'})")
            lines.append(f"  Family: {a['family_match'] or 'unclassified'} ({a['family_druggability']})")
            lines.append(f"  Binding sites: {a['binding_site_count']}")
            lines.append(f"  Notes: {a['druggability_notes']}")
            lines.append("")
        return "\n".join(lines)
