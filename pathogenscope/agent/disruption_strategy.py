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


DRUGGABLE_DOMAINS = {
    # keyword → (druggability, suggested inhibitor class)
    "kinase": ("high", "kinase inhibitor"),
    "atpase": ("high", "ATPase inhibitor"),
    "atp-binding": ("high", "ATP-competitive inhibitor"),
    "gtpase": ("high", "GTPase inhibitor"),
    "protease": ("high", "protease inhibitor"),
    "peptidase": ("high", "protease inhibitor"),
    "gyrase": ("high", "gyrase inhibitor (e.g. fluoroquinolone)"),
    "topoisomerase": ("high", "topoisomerase inhibitor"),
    "polymerase": ("moderate", "polymerase inhibitor"),
    "helicase": ("moderate", "helicase inhibitor"),
    "transferase": ("moderate", "transferase inhibitor"),
    "oxidoreductase": ("moderate", "redox-site inhibitor"),
    "dehydrogenase": ("moderate", "NAD/NADP-competitive inhibitor"),
    "synthase": ("moderate", "active-site inhibitor"),
    "ligase": ("moderate", "substrate-competitive inhibitor"),
    "chaperone": ("low", "allosteric modulator"),
    "ribosom": ("low", "translation inhibitor (biologics)"),
}


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
        """Classify the PPI interface based on hotspot analysis and domain context."""
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

        # Domain-informed druggability adjustment
        domains_1 = analysis.get("domains_protein_1", [])
        domains_2 = analysis.get("domains_protein_2", [])
        interface_domains = self._match_druggable_domains(domains_1 + domains_2)

        if interface_domains:
            best = max(interface_domains, key=lambda d: d["overlap_residues"])
            domain_druggability = best["druggability"]
            # Upgrade classification when a druggable domain sits at the interface
            if domain_druggability == "high" and itype == "flat/distributed":
                druggability = (
                    f"moderate — flat interface BUT druggable domain at interface "
                    f"({best['domain_name']}: {best['inhibitor_class']})"
                )
            elif domain_druggability == "high":
                druggability += (
                    f"; druggable domain at interface "
                    f"({best['domain_name']}: {best['inhibitor_class']})"
                )
            elif domain_druggability == "moderate":
                druggability += (
                    f"; moderately druggable domain at interface ({best['domain_name']})"
                )

        return {
            "type": itype,
            "druggability": druggability,
            "contact_count": n_contacts,
            "max_patch_length": max_patch_len,
            "hotspot_concentration": round(concentration, 2),
            "interface_domains": interface_domains,
        }

    def _match_druggable_domains(self, domain_mappings: list[dict]) -> list[dict]:
        """Match interface domain mappings against known druggable domain families."""
        matched = []
        for dm in domain_mappings:
            domain_name_lower = dm.get("domain_name", "").lower()
            domain_id_lower = dm.get("domain_id", "").lower()
            search_text = f"{domain_name_lower} {domain_id_lower}"
            for keyword, (druggability, inhibitor_class) in DRUGGABLE_DOMAINS.items():
                if keyword in search_text:
                    matched.append({
                        "domain_name": dm["domain_name"],
                        "domain_id": dm.get("domain_id", ""),
                        "patch_start": dm.get("patch_start", 0),
                        "patch_end": dm.get("patch_end", 0),
                        "overlap_residues": dm.get("overlap_residues", 0),
                        "druggability": druggability,
                        "inhibitor_class": inhibitor_class,
                        "matched_keyword": keyword,
                    })
                    break  # one match per domain
        return matched

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
        interface_domains = interface_type.get("interface_domains", [])

        # Check if a high-druggability domain sits at the interface
        high_domain = None
        for d in interface_domains:
            if d["druggability"] == "high":
                high_domain = d
                break
        if high_domain is None:
            for d in interface_domains:
                if d["druggability"] == "moderate":
                    high_domain = d
                    break

        # Small molecule
        if itype in ("hotspot-driven", "compact"):
            rationale = (
                f"Compact interface ({interface_type['contact_count']} contacts) "
                f"with clear binding pocket potential"
            )
            next_steps = [
                "Virtual screening against hotspot pocket",
                "Fragment-based screening",
            ]
            if high_domain:
                rationale += (
                    f"; hotspot overlaps {high_domain['domain_name']} — "
                    f"{high_domain['inhibitor_class']} may compete for binding"
                )
                next_steps.insert(0,
                    f"Screen {high_domain['inhibitor_class']}s targeting "
                    f"{high_domain['domain_name']} (residues {high_domain['patch_start']}-{high_domain['patch_end']})"
                )
            modalities.append({
                "modality": "small molecule",
                "confidence": "high",
                "rationale": rationale,
                "next_steps": next_steps,
            })
        elif itype == "hotspot-in-large-interface":
            rationale = "Large interface but concentrated hotspot may be targetable"
            next_steps = [
                "Focus on hotspot residues for pocket analysis",
                "Allosteric site search",
            ]
            confidence = "moderate"
            if high_domain and high_domain["druggability"] == "high":
                rationale += (
                    f"; druggable domain ({high_domain['domain_name']}) at hotspot — "
                    f"try {high_domain['inhibitor_class']}"
                )
                next_steps.insert(0,
                    f"Screen {high_domain['inhibitor_class']}s targeting "
                    f"{high_domain['domain_name']}"
                )
                confidence = "high"
            modalities.append({
                "modality": "small molecule",
                "confidence": confidence,
                "rationale": rationale,
                "next_steps": next_steps,
            })
        else:
            # flat/distributed — normally low confidence
            rationale = "Flat/distributed interface — difficult for small molecules"
            next_steps = ["Consider allosteric approach", "Explore other modalities"]
            confidence = "low"
            if high_domain and high_domain["druggability"] == "high":
                rationale = (
                    f"Flat interface BUT druggable domain ({high_domain['domain_name']}) "
                    f"at interface — {high_domain['inhibitor_class']} may disrupt binding"
                )
                next_steps = [
                    f"Screen {high_domain['inhibitor_class']}s targeting "
                    f"{high_domain['domain_name']} (residues {high_domain['patch_start']}-{high_domain['patch_end']})",
                    "Allosteric site search near domain boundary",
                ]
                confidence = "moderate"
            modalities.append({
                "modality": "small molecule",
                "confidence": confidence,
                "rationale": rationale,
                "next_steps": next_steps,
            })

        # Peptide mimetic
        patches = analysis.get("patches_protein_1", []) + analysis.get("patches_protein_2", [])
        linear_patches = [p for p in patches if p["length"] <= 20]
        if linear_patches:
            best_patch = max(linear_patches, key=lambda p: p["max_contact"])
            # Check if the best patch overlaps a known domain
            patch_domain_note = ""
            for d in interface_domains:
                if (d["patch_start"] <= best_patch["end"] and
                        d["patch_end"] >= best_patch["start"]):
                    patch_domain_note = f" (within {d['domain_name']})"
                    break
            modalities.append({
                "modality": "peptide mimetic",
                "confidence": "moderate" if best_patch["length"] <= 12 else "low",
                "rationale": (
                    f"Linear interface patch ({best_patch['length']} residues, "
                    f"contact={best_patch['max_contact']}){patch_domain_note} "
                    f"suitable for peptide design"
                ),
                "next_steps": [
                    f"Design stapled peptide spanning residues {best_patch['start']}-{best_patch['end']}{patch_domain_note}",
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

            # Surface domain context from InterfaceHotspotAgent
            interface_domains = itype.get("interface_domains", [])
            if interface_domains:
                lines.append("  Domains at interface:")
                for d in interface_domains:
                    lines.append(
                        f"    {d['domain_name']} ({d['domain_id']}) — "
                        f"{d['overlap_residues']} residues at interface, "
                        f"druggability: {d['druggability']}, "
                        f"suggested: {d['inhibitor_class']}"
                    )

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
