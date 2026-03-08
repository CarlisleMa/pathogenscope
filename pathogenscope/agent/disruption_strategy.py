"""
Disruption Strategy Agent.

Given interface hotspot analysis, suggests PPI disruption strategies:
  1. Small molecule — is the interface compact enough for a small molecule?
  2. Peptide mimetic — can we mimic a binding helix/loop?
  3. Existing PDB co-crystal structures — any known inhibitors?
  4. Interface classification — hot spot vs flat interface
"""

import requests
import numpy as np
import pandas as pd

from .base import BaseAgent, AgentResult


class DisruptionStrategyAgent(BaseAgent):
    name = "disruption_strategy"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.compact_threshold = self.config.get("compact_threshold", 20)

    def run(
        self,
        hotspot_results: AgentResult | None = None,
        contact_maps: dict | None = None,
        target_scores: pd.DataFrame | None = None,
        **kwargs,
    ) -> AgentResult:
        """
        Suggest disruption strategies for top PPI interactions.

        Args:
            hotspot_results: Output from InterfaceHotspotAgent
            contact_maps: Raw contact maps (if hotspot not run)
            target_scores: Target rankings for prioritization
        """
        result = AgentResult(agent_name=self.name)

        if hotspot_results is None or hotspot_results.status == "error":
            self._warn(result, "No hotspot analysis available. Run InterfaceHotspotAgent first.")
            result.status = "partial"
            # Can still do PDB lookups if we have target scores
            if target_scores is not None:
                result.data["pdb_hits"] = self._bulk_pdb_search(target_scores)
                result.report = self._format_pdb_only_report(result.data["pdb_hits"])
            return result

        analyses = hotspot_results.data.get("analyses", [])
        strategies = []

        for analysis in analyses:
            pid1 = analysis["protein_1"]
            pid2 = analysis["protein_2"]
            print(f"  [{self.name}] Strategizing {pid1} ↔ {pid2}...")

            # 1. Classify interface type
            interface_type = self._classify_interface(analysis)

            # 2. Check PDB for existing structural data
            acc1 = self._extract_accession(pid1)
            acc2 = self._extract_accession(pid2)
            pdb_data = self._search_pdb_complex(acc1, acc2)

            # 3. Suggest modalities
            modalities = self._suggest_modalities(analysis, interface_type, pdb_data)

            # 4. Rank strategies
            ranked = self._rank_strategies(modalities)

            strategies.append({
                "protein_1": pid1,
                "protein_2": pid2,
                "interface_type": interface_type,
                "pdb_structures": pdb_data,
                "suggested_modalities": ranked,
            })

        result.data["strategies"] = strategies
        result.report = self._format_report(strategies)
        return result

    def _classify_interface(self, analysis: dict) -> dict:
        """Classify the PPI interface based on hotspot analysis."""
        stats = analysis["interface_stats"]
        patches_1 = analysis["patches_protein_1"]
        patches_2 = analysis["patches_protein_2"]
        n_contacts = stats["contacts_above_threshold"]

        # Interface compactness: few residues in tight cluster vs spread out
        max_patch_len = 0
        if patches_1:
            max_patch_len = max(max_patch_len, max(p["length"] for p in patches_1))
        if patches_2:
            max_patch_len = max(max_patch_len, max(p["length"] for p in patches_2))

        # Hotspot concentration
        hotspots = analysis["hotspot_residues"]
        if len(hotspots) >= 2:
            top_prob = hotspots[0]["contact_prob"]
            median_prob = hotspots[min(len(hotspots) // 2, len(hotspots) - 1)]["contact_prob"]
            concentration = top_prob / max(median_prob, 0.01)
        else:
            concentration = 1.0

        if n_contacts <= self.compact_threshold and concentration > 2.0:
            itype = "hotspot-driven"
            druggability = "favorable — compact interface with clear hotspot"
        elif n_contacts <= self.compact_threshold:
            itype = "compact"
            druggability = "favorable — small interface area"
        elif concentration > 2.0:
            itype = "hotspot-in-large-interface"
            druggability = "moderate — large interface but with exploitable hotspot"
        else:
            itype = "flat/distributed"
            druggability = "challenging — large, flat interface"

        return {
            "type": itype,
            "druggability": druggability,
            "contact_count": n_contacts,
            "max_patch_length": max_patch_len,
            "hotspot_concentration": round(concentration, 2),
        }

    def _search_pdb_complex(self, acc1: str, acc2: str) -> list[dict]:
        """Search PDB for co-crystal structures of this protein pair."""
        structures = []

        # Search for each protein individually in PDB
        for acc in [acc1, acc2]:
            hits = self._query_pdb_for_accession(acc)
            structures.extend(hits)

        return structures

    def _query_pdb_for_accession(self, accession: str) -> list[dict]:
        """Query PDB for structures containing this UniProt accession."""
        try:
            url = (
                f"https://rest.uniprot.org/uniprotkb/{accession}"
                f"?fields=xref_pdb&format=json"
            )
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                return []

            data = resp.json()
            pdb_refs = []
            for xref in data.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "PDB":
                    props = {p["key"]: p["value"] for p in xref.get("properties", [])}
                    pdb_refs.append({
                        "pdb_id": xref.get("id", ""),
                        "method": props.get("Method", ""),
                        "resolution": props.get("Resolution", ""),
                        "chains": props.get("Chains", ""),
                        "accession": accession,
                    })
            return pdb_refs
        except (requests.RequestException, ValueError):
            return []

    def _suggest_modalities(
        self, analysis: dict, interface_type: dict, pdb_data: list[dict]
    ) -> list[dict]:
        """Suggest therapeutic modalities for disrupting this PPI."""
        modalities = []
        itype = interface_type["type"]

        # Small molecule
        if itype in ("hotspot-driven", "compact"):
            modalities.append({
                "modality": "small molecule",
                "confidence": "high",
                "rationale": (
                    f"Compact interface ({interface_type['contact_count']} contacts) "
                    f"with clear binding pocket potential"
                ),
                "next_steps": [
                    "Virtual screening against hotspot pocket",
                    "Fragment-based screening",
                ],
            })
        elif itype == "hotspot-in-large-interface":
            modalities.append({
                "modality": "small molecule",
                "confidence": "moderate",
                "rationale": "Large interface but concentrated hotspot may be targetable",
                "next_steps": [
                    "Focus on hotspot residues for pocket analysis",
                    "Allosteric site search",
                ],
            })
        else:
            modalities.append({
                "modality": "small molecule",
                "confidence": "low",
                "rationale": "Flat/distributed interface — difficult for small molecules",
                "next_steps": ["Consider allosteric approach", "Explore other modalities"],
            })

        # Peptide mimetic
        patches = analysis.get("patches_protein_1", []) + analysis.get("patches_protein_2", [])
        linear_patches = [p for p in patches if p["length"] <= 20]
        if linear_patches:
            best_patch = max(linear_patches, key=lambda p: p["max_contact"])
            modalities.append({
                "modality": "peptide mimetic",
                "confidence": "moderate" if best_patch["length"] <= 12 else "low",
                "rationale": (
                    f"Linear interface patch ({best_patch['length']} residues, "
                    f"contact={best_patch['max_contact']}) suitable for peptide design"
                ),
                "next_steps": [
                    f"Design stapled peptide spanning residues {best_patch['start']}-{best_patch['end']}",
                    "Test cyclized variants for stability",
                ],
            })

        # Antibody/nanobody (always an option for extracellular targets)
        modalities.append({
            "modality": "antibody/nanobody",
            "confidence": "moderate",
            "rationale": "Biologics can target large interfaces; requires accessible epitope",
            "next_steps": [
                "Check if target is surface-exposed",
                "Identify accessible epitope near interface",
            ],
        })

        # Existing structures boost confidence
        if pdb_data:
            for m in modalities:
                if m["modality"] == "small molecule":
                    m["next_steps"].insert(0,
                        f"Leverage existing PDB structures: {', '.join(p['pdb_id'] for p in pdb_data[:3])}")

        return modalities

    def _rank_strategies(self, modalities: list[dict]) -> list[dict]:
        """Rank strategies by confidence."""
        conf_order = {"high": 0, "moderate": 1, "low": 2}
        return sorted(modalities, key=lambda m: conf_order.get(m["confidence"], 3))

    def _bulk_pdb_search(self, target_scores: pd.DataFrame) -> list[dict]:
        """Search PDB for all top targets (fallback when no hotspot data)."""
        results = []
        for pid in target_scores["protein_id"].head(10):
            acc = pid.split("|")[1] if "|" in pid else pid
            hits = self._query_pdb_for_accession(acc)
            if hits:
                results.append({"protein_id": pid, "pdb_structures": hits})
        return results

    def _extract_accession(self, protein_id: str) -> str:
        parts = protein_id.split("|")
        return parts[1] if len(parts) >= 2 else parts[0]

    def _format_report(self, strategies: list[dict]) -> str:
        lines = ["# Disruption Strategy Report\n"]
        for s in strategies:
            lines.append(f"## {s['protein_1']} ↔ {s['protein_2']}")
            itype = s["interface_type"]
            lines.append(f"  Interface: {itype['type']} ({itype['contact_count']} contacts)")
            lines.append(f"  Assessment: {itype['druggability']}")

            if s["pdb_structures"]:
                pdbs = ", ".join(p["pdb_id"] for p in s["pdb_structures"][:5])
                lines.append(f"  PDB structures: {pdbs}")

            lines.append("  Suggested approaches:")
            for m in s["suggested_modalities"]:
                lines.append(f"    [{m['confidence'].upper()}] {m['modality']}")
                lines.append(f"      {m['rationale']}")
                for step in m["next_steps"][:2]:
                    lines.append(f"      → {step}")
            lines.append("")
        return "\n".join(lines)

    def _format_pdb_only_report(self, pdb_hits: list[dict]) -> str:
        lines = ["# PDB Structure Search (no contact maps available)\n"]
        for hit in pdb_hits:
            lines.append(f"## {hit['protein_id']}")
            for pdb in hit["pdb_structures"][:5]:
                lines.append(f"  {pdb['pdb_id']} ({pdb['method']}, {pdb['resolution']})")
            lines.append("")
        return "\n".join(lines)
