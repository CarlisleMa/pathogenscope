"""
Chemical Matter Agent.

Searches ChEMBL for existing bioactive compounds against each target or
its protein family, providing chemical starting points for drug programs:
  1. Direct target search — compounds tested against this exact UniProt accession
  2. Family-level search — compounds active against the same protein family
  3. Summarizes assay types, potency ranges, and clinical status
"""

import requests
import pandas as pd

from .base import BaseAgent, AgentResult

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


class ChemicalMatterAgent(BaseAgent):
    name = "chemical_matter"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.top_n = self.config.get("top_n", 20)
        self.activity_threshold_nm = self.config.get("activity_threshold_nm", 10_000)

    def run(
        self,
        target_scores: pd.DataFrame,
        druggability: AgentResult | None = None,
        **kwargs,
    ) -> AgentResult:
        """
        Search ChEMBL for chemical matter against top targets.

        Args:
            target_scores: Ranked targets from pipeline
            druggability: DruggabilityAgent results (for family context)
        """
        result = AgentResult(agent_name=self.name)

        # Build family map from druggability results
        family_map = {}
        if druggability and druggability.status == "success":
            for a in druggability.data.get("assessments", []):
                if a.get("family_match"):
                    family_map[a["protein_id"]] = a["family_match"]

        assessments = []
        for _, row in target_scores.head(self.top_n).iterrows():
            pid = row["protein_id"]
            acc = self._extract_accession(pid)
            print(f"  [{self.name}] Searching ChEMBL for {acc}...")

            assessment = self._assess_target(pid, acc, family_map.get(pid))
            assessments.append(assessment)

        result.data["assessments"] = assessments
        result.data["summary"] = self._summarize(assessments)
        result.report = self._format_report(assessments)
        return result

    def _assess_target(self, protein_id: str, accession: str, family: str | None) -> dict:
        """Full chemical matter assessment for one target."""
        # 1. Find ChEMBL target ID for this UniProt accession
        chembl_target = self._lookup_chembl_target(accession)

        # 2. Get bioactivities for this target
        direct_compounds = []
        if chembl_target:
            direct_compounds = self._get_bioactivities(chembl_target["target_chembl_id"])

        # 3. Search by protein family if direct hits are sparse
        family_compounds = []
        if family and len(direct_compounds) < 3:
            family_compounds = self._search_by_family(family)

        # 4. Check clinical pipeline
        clinical = []
        if chembl_target:
            clinical = self._get_clinical_compounds(chembl_target["target_chembl_id"])

        # 5. Classify chemical tractability
        tractability = self._classify_tractability(
            direct_compounds, family_compounds, clinical
        )

        return {
            "protein_id": protein_id,
            "accession": accession,
            "chembl_target_id": chembl_target["target_chembl_id"] if chembl_target else None,
            "chembl_target_name": chembl_target.get("pref_name", "") if chembl_target else "",
            "direct_compounds": direct_compounds[:10],
            "direct_compound_count": len(direct_compounds),
            "family": family,
            "family_compounds": family_compounds[:5],
            "family_compound_count": len(family_compounds),
            "clinical_compounds": clinical,
            "tractability": tractability,
        }

    def _lookup_chembl_target(self, accession: str) -> dict | None:
        """Find ChEMBL target entry for a UniProt accession."""
        try:
            url = (
                f"{CHEMBL_BASE}/target.json"
                f"?target_components__accession={accession}"
                f"&limit=1"
            )
            resp = requests.get(url, timeout=15)
            if not resp.ok:
                return None
            data = resp.json()
            targets = data.get("targets", [])
            return targets[0] if targets else None
        except (requests.RequestException, ValueError, IndexError):
            return None

    def _get_bioactivities(self, target_chembl_id: str) -> list[dict]:
        """Get bioactivity data for a ChEMBL target."""
        try:
            url = (
                f"{CHEMBL_BASE}/activity.json"
                f"?target_chembl_id={target_chembl_id}"
                f"&pchembl_value__isnull=false"
                f"&limit=50"
                f"&order_by=-pchembl_value"
            )
            resp = requests.get(url, timeout=15)
            if not resp.ok:
                return []

            data = resp.json()
            compounds = []
            seen_molecules = set()
            for act in data.get("activities", []):
                mol_id = act.get("molecule_chembl_id", "")
                if mol_id in seen_molecules:
                    continue
                seen_molecules.add(mol_id)

                pchembl = act.get("pchembl_value")
                try:
                    potency_nm = 10 ** (9 - float(pchembl)) if pchembl else None
                except (ValueError, TypeError):
                    potency_nm = None

                if potency_nm and potency_nm > self.activity_threshold_nm:
                    continue

                compounds.append({
                    "molecule_chembl_id": mol_id,
                    "molecule_name": act.get("molecule_pref_name", ""),
                    "pchembl_value": float(pchembl) if pchembl else None,
                    "potency_nm": round(potency_nm, 1) if potency_nm else None,
                    "assay_type": act.get("assay_type", ""),
                    "standard_type": act.get("standard_type", ""),
                })

            return compounds
        except (requests.RequestException, ValueError):
            return []

    def _search_by_family(self, family: str) -> list[dict]:
        """Search ChEMBL for compounds active against the same protein family."""
        try:
            url = (
                f"{CHEMBL_BASE}/target.json"
                f"?target_type=SINGLE PROTEIN"
                f"&pref_name__icontains={family}"
                f"&limit=5"
            )
            resp = requests.get(url, timeout=15)
            if not resp.ok:
                return []

            data = resp.json()
            family_targets = data.get("targets", [])
            compounds = []
            for ft in family_targets[:3]:
                ft_id = ft.get("target_chembl_id", "")
                ft_compounds = self._get_bioactivities(ft_id)
                for c in ft_compounds[:3]:
                    c["source_target"] = ft.get("pref_name", ft_id)
                    c["source_organism"] = ft.get("organism", "")
                    compounds.append(c)

            return compounds
        except (requests.RequestException, ValueError):
            return []

    def _get_clinical_compounds(self, target_chembl_id: str) -> list[dict]:
        """Find compounds in clinical trials for this target."""
        try:
            url = (
                f"{CHEMBL_BASE}/mechanism.json"
                f"?target_chembl_id={target_chembl_id}"
                f"&limit=20"
            )
            resp = requests.get(url, timeout=15)
            if not resp.ok:
                return []

            data = resp.json()
            clinical = []
            seen = set()
            for mech in data.get("mechanisms", []):
                mol_id = mech.get("molecule_chembl_id", "")
                if mol_id in seen:
                    continue
                seen.add(mol_id)

                # Get molecule details for max_phase
                mol_info = self._get_molecule_info(mol_id)

                clinical.append({
                    "molecule_chembl_id": mol_id,
                    "molecule_name": mol_info.get("pref_name", ""),
                    "mechanism": mech.get("mechanism_of_action", ""),
                    "action_type": mech.get("action_type", ""),
                    "max_phase": mol_info.get("max_phase", 0),
                })

            return clinical
        except (requests.RequestException, ValueError):
            return []

    def _get_molecule_info(self, molecule_chembl_id: str) -> dict:
        """Get molecule details including clinical phase."""
        try:
            url = f"{CHEMBL_BASE}/molecule/{molecule_chembl_id}.json"
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                return {}
            return resp.json()
        except (requests.RequestException, ValueError):
            return {}

    def _classify_tractability(
        self,
        direct: list[dict],
        family: list[dict],
        clinical: list[dict],
    ) -> dict:
        """Classify chemical tractability based on available matter."""
        # Check for potent direct hits
        potent_direct = [c for c in direct if c.get("pchembl_value") and c["pchembl_value"] >= 6.0]

        if clinical:
            level = "clinical"
            assessment = "Known drug target — clinical compounds exist"
        elif potent_direct:
            best = max(potent_direct, key=lambda c: c["pchembl_value"])
            level = "chemical_probe"
            assessment = (
                f"Chemical probes available — best pChEMBL={best['pchembl_value']:.1f} "
                f"({best['potency_nm']:.0f} nM)"
            )
        elif direct:
            level = "hit_matter"
            assessment = f"Hit matter exists — {len(direct)} compounds with measurable activity"
        elif family:
            level = "family_precedent"
            assessment = (
                f"No direct hits but {len(family)} compounds active against related "
                f"family members — scaffold-hopping opportunity"
            )
        else:
            level = "novel"
            assessment = "No known chemical matter — requires de novo screening"

        return {
            "level": level,
            "assessment": assessment,
            "direct_hits": len(direct),
            "potent_hits": len(potent_direct),
            "family_hits": len(family),
            "clinical_compounds": len(clinical),
        }

    def _extract_accession(self, protein_id: str) -> str:
        parts = protein_id.split("|")
        return parts[1] if len(parts) >= 2 else parts[0]

    def _summarize(self, assessments: list[dict]) -> dict:
        tractability_counts = {}
        for a in assessments:
            level = a["tractability"]["level"]
            tractability_counts[level] = tractability_counts.get(level, 0) + 1

        total_direct = sum(a["direct_compound_count"] for a in assessments)
        with_clinical = sum(1 for a in assessments if a["clinical_compounds"])

        return {
            "total_assessed": len(assessments),
            "total_direct_compounds": total_direct,
            "with_clinical_compounds": with_clinical,
            "tractability_breakdown": tractability_counts,
        }

    def _format_report(self, assessments: list[dict]) -> str:
        lines = ["# Chemical Matter Report\n"]

        for a in assessments:
            name = a["protein_id"].split("|")[-1] if "|" in a["protein_id"] else a["protein_id"]
            tract = a["tractability"]
            lines.append(f"## {name} ({a['accession']})")
            lines.append(f"  Tractability: **{tract['level'].upper()}** — {tract['assessment']}")

            if a["chembl_target_id"]:
                lines.append(f"  ChEMBL target: {a['chembl_target_id']} ({a['chembl_target_name']})")

            if a["direct_compounds"]:
                lines.append(f"  Direct compounds ({a['direct_compound_count']} total):")
                for c in a["direct_compounds"][:5]:
                    name_str = c["molecule_name"] or c["molecule_chembl_id"]
                    potency = f"pChEMBL={c['pchembl_value']:.1f}" if c["pchembl_value"] else "no potency"
                    lines.append(f"    {name_str} — {potency} ({c['assay_type']})")

            if a["family_compounds"]:
                lines.append(f"  Family analogues ({a['family_compound_count']} total):")
                for c in a["family_compounds"][:3]:
                    source = c.get("source_target", "")
                    lines.append(f"    {c['molecule_chembl_id']} — from {source}")

            if a["clinical_compounds"]:
                lines.append("  Clinical compounds:")
                for c in a["clinical_compounds"]:
                    phase = f"Phase {c['max_phase']}" if c["max_phase"] else "preclinical"
                    name_str = c["molecule_name"] or c["molecule_chembl_id"]
                    lines.append(f"    {name_str} — {c['mechanism']} ({phase})")

            lines.append("")

        return "\n".join(lines)
