"""
Interface Hotspot Agent.

Analyzes residue-level contact maps from FlashPPI to identify:
  1. Top interacting residue patches (hotspots)
  2. Domain mapping — which Pfam/InterPro domains are at the interface
  3. Conservation context — conserved vs variable interface positions
"""

import numpy as np
import pandas as pd
import requests
import torch

from .base import BaseAgent, AgentResult


class InterfaceHotspotAgent(BaseAgent):
    name = "interface_hotspot"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.min_contact_prob = self.config.get("min_contact_prob", 0.3)
        self.patch_window = self.config.get("patch_window", 5)

    def run(
        self,
        contact_maps: dict[tuple[str, str], torch.Tensor],
        sequences: dict[str, str] | None = None,
        annotations: pd.DataFrame | None = None,
        **kwargs,
    ) -> AgentResult:
        """
        Analyze contact maps to find interface hotspots.

        Args:
            contact_maps: {(protein_id_1, protein_id_2): contact_map_tensor}
                          contact_map should already have sigmoid applied
            sequences: {protein_id: amino_acid_sequence}
            annotations: UniProt annotations for domain lookup
        """
        result = AgentResult(agent_name=self.name)

        if not contact_maps:
            self._warn(result, "No contact maps provided. Run pipeline with save_contact_maps=True")
            result.status = "error"
            return result

        analyses = []
        for (pid1, pid2), cmap in contact_maps.items():
            print(f"  [{self.name}] Analyzing {pid1} ↔ {pid2}...")
            analysis = self._analyze_pair(pid1, pid2, cmap, sequences, annotations)
            analyses.append(analysis)

        result.data["analyses"] = analyses
        result.data["summary"] = self._summarize(analyses)
        result.report = self._format_report(analyses)
        return result

    def _analyze_pair(
        self,
        pid1: str, pid2: str,
        cmap: torch.Tensor,
        sequences: dict[str, str] | None,
        annotations: pd.DataFrame | None,
    ) -> dict:
        """Full analysis for one protein pair."""
        if isinstance(cmap, torch.Tensor):
            cmap_np = cmap.numpy()
        else:
            cmap_np = np.array(cmap)

        # Ensure 2D
        if cmap_np.ndim > 2:
            cmap_np = cmap_np.squeeze()

        # 1. Find top contact residues
        hotspots = self._find_hotspots(cmap_np)

        # 2. Identify contiguous patches
        patches_1, patches_2 = self._find_patches(cmap_np)

        # 3. Map to domains
        acc1 = self._extract_accession(pid1)
        acc2 = self._extract_accession(pid2)
        domains_1 = self._map_to_domains(acc1, patches_1)
        domains_2 = self._map_to_domains(acc2, patches_2)

        # 4. Interface statistics
        stats = self._interface_stats(cmap_np)

        # 5. Sequence context
        seq_context_1 = self._get_sequence_context(pid1, patches_1, sequences)
        seq_context_2 = self._get_sequence_context(pid2, patches_2, sequences)

        return {
            "protein_1": pid1,
            "protein_2": pid2,
            "contact_score": float(cmap_np.max()),
            "interface_stats": stats,
            "hotspot_residues": hotspots[:20],
            "patches_protein_1": patches_1,
            "patches_protein_2": patches_2,
            "domains_protein_1": domains_1,
            "domains_protein_2": domains_2,
            "sequence_context_1": seq_context_1,
            "sequence_context_2": seq_context_2,
        }

    def _find_hotspots(self, cmap: np.ndarray) -> list[dict]:
        """Find top contacting residue pairs."""
        # Get all positions above threshold
        rows, cols = np.where(cmap >= self.min_contact_prob)
        if len(rows) == 0:
            # Fall back to top 10 regardless of threshold
            flat_idx = np.argsort(cmap.ravel())[-10:]
            rows, cols = np.unravel_index(flat_idx, cmap.shape)

        hotspots = []
        for r, c in zip(rows, cols):
            hotspots.append({
                "residue_1": int(r),
                "residue_2": int(c),
                "contact_prob": round(float(cmap[r, c]), 4),
            })

        hotspots.sort(key=lambda x: x["contact_prob"], reverse=True)
        return hotspots

    def _find_patches(
        self, cmap: np.ndarray
    ) -> tuple[list[dict], list[dict]]:
        """Find contiguous residue patches on each protein's interface."""
        # Project contact map onto each axis to find interface regions
        proj_1 = cmap.max(axis=1)  # max contact per residue of protein 1
        proj_2 = cmap.max(axis=0)  # max contact per residue of protein 2

        patches_1 = self._extract_patches(proj_1)
        patches_2 = self._extract_patches(proj_2)
        return patches_1, patches_2

    def _extract_patches(self, projection: np.ndarray) -> list[dict]:
        """Extract contiguous high-contact regions from a 1D projection."""
        above = projection >= self.min_contact_prob
        if not above.any():
            # Use top positions
            top_idx = np.argsort(projection)[-5:]
            above = np.zeros_like(projection, dtype=bool)
            above[top_idx] = True

        patches = []
        in_patch = False
        start = 0

        for i in range(len(above)):
            if above[i] and not in_patch:
                start = i
                in_patch = True
            elif not above[i] and in_patch:
                patches.append({
                    "start": int(start),
                    "end": int(i - 1),
                    "length": int(i - start),
                    "max_contact": round(float(projection[start:i].max()), 4),
                    "mean_contact": round(float(projection[start:i].mean()), 4),
                })
                in_patch = False

        if in_patch:
            patches.append({
                "start": int(start),
                "end": int(len(above) - 1),
                "length": int(len(above) - start),
                "max_contact": round(float(projection[start:].max()), 4),
                "mean_contact": round(float(projection[start:].mean()), 4),
            })

        patches.sort(key=lambda x: x["max_contact"], reverse=True)
        return patches

    def _map_to_domains(self, accession: str, patches: list[dict]) -> list[dict]:
        """Map interface patches to Pfam/InterPro domains."""
        domains = self._fetch_domains(accession)
        if not domains:
            return []

        mapped = []
        for patch in patches:
            for domain in domains:
                # Check overlap
                overlap_start = max(patch["start"], domain["start"])
                overlap_end = min(patch["end"], domain["end"])
                if overlap_start <= overlap_end:
                    mapped.append({
                        "patch_start": patch["start"],
                        "patch_end": patch["end"],
                        "domain_name": domain["name"],
                        "domain_id": domain["id"],
                        "domain_start": domain["start"],
                        "domain_end": domain["end"],
                        "overlap_residues": overlap_end - overlap_start + 1,
                    })
        return mapped

    def _fetch_domains(self, accession: str) -> list[dict]:
        """Fetch domain annotations from UniProt."""
        try:
            url = (
                f"https://rest.uniprot.org/uniprotkb/{accession}"
                f"?fields=ft_domain,xref_pfam,xref_interpro&format=json"
            )
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                return []

            data = resp.json()
            domains = []
            for feature in data.get("features", []):
                if feature.get("type") in ("Domain", "Region"):
                    loc = feature.get("location", {})
                    start = loc.get("start", {}).get("value", 0)
                    end = loc.get("end", {}).get("value", 0)
                    desc = feature.get("description", "unknown")
                    domains.append({
                        "name": desc,
                        "id": feature.get("type", ""),
                        "start": int(start) - 1,  # 0-indexed
                        "end": int(end) - 1,
                    })

            # Also check cross-references for Pfam
            xrefs = data.get("uniProtKBCrossReferences", [])
            for xref in xrefs:
                if xref.get("database") == "Pfam":
                    pfam_id = xref.get("id", "")
                    props = {p["key"]: p["value"] for p in xref.get("properties", [])}
                    domains.append({
                        "name": props.get("EntryName", pfam_id),
                        "id": pfam_id,
                        "start": 0,
                        "end": 0,  # Pfam xrefs don't always have positions
                    })

            return domains
        except (requests.RequestException, ValueError):
            return []

    def _get_sequence_context(
        self, protein_id: str, patches: list[dict],
        sequences: dict[str, str] | None,
    ) -> list[dict]:
        """Extract sequence fragments at interface patches."""
        if sequences is None:
            return []

        seq = sequences.get(protein_id)
        if seq is None:
            return []

        contexts = []
        for patch in patches[:5]:  # top 5 patches
            start = max(0, patch["start"] - 3)
            end = min(len(seq), patch["end"] + 4)
            contexts.append({
                "patch_start": patch["start"],
                "patch_end": patch["end"],
                "sequence_fragment": seq[start:end],
                "full_start": start,
                "full_end": end,
            })
        return contexts

    def _interface_stats(self, cmap: np.ndarray) -> dict:
        """Compute summary statistics for the contact map."""
        above_thresh = cmap >= self.min_contact_prob
        return {
            "map_shape": list(cmap.shape),
            "max_contact": round(float(cmap.max()), 4),
            "mean_contact": round(float(cmap.mean()), 6),
            "contacts_above_threshold": int(above_thresh.sum()),
            "interface_fraction_1": round(float(above_thresh.any(axis=1).mean()), 4),
            "interface_fraction_2": round(float(above_thresh.any(axis=0).mean()), 4),
        }

    def _extract_accession(self, protein_id: str) -> str:
        parts = protein_id.split("|")
        return parts[1] if len(parts) >= 2 else parts[0]

    def _summarize(self, analyses: list[dict]) -> dict:
        return {
            "pairs_analyzed": len(analyses),
            "avg_interface_size": round(
                np.mean([a["interface_stats"]["contacts_above_threshold"]
                         for a in analyses]), 1
            ) if analyses else 0,
            "domains_at_interface": sum(
                len(a["domains_protein_1"]) + len(a["domains_protein_2"])
                for a in analyses
            ),
        }

    def _format_report(self, analyses: list[dict]) -> str:
        lines = ["# Interface Hotspot Report\n"]
        for a in analyses:
            lines.append(f"## {a['protein_1']} ↔ {a['protein_2']}")
            stats = a["interface_stats"]
            lines.append(f"  Contact score: {a['contact_score']}")
            lines.append(f"  Interface: {stats['contacts_above_threshold']} residue contacts")
            lines.append(f"  Interface coverage: {stats['interface_fraction_1']:.1%} of protein 1, "
                         f"{stats['interface_fraction_2']:.1%} of protein 2")

            if a["hotspot_residues"]:
                top3 = a["hotspot_residues"][:3]
                for h in top3:
                    lines.append(f"  Hotspot: res {h['residue_1']}–{h['residue_2']} "
                                 f"(prob={h['contact_prob']})")

            if a["patches_protein_1"]:
                p = a["patches_protein_1"][0]
                lines.append(f"  Top patch (protein 1): residues {p['start']}-{p['end']} "
                             f"(max={p['max_contact']})")
            if a["patches_protein_2"]:
                p = a["patches_protein_2"][0]
                lines.append(f"  Top patch (protein 2): residues {p['start']}-{p['end']} "
                             f"(max={p['max_contact']})")

            if a["domains_protein_1"]:
                for d in a["domains_protein_1"][:3]:
                    lines.append(f"  Domain at interface (prot1): {d['domain_name']} "
                                 f"({d['overlap_residues']} overlapping residues)")
            if a["domains_protein_2"]:
                for d in a["domains_protein_2"][:3]:
                    lines.append(f"  Domain at interface (prot2): {d['domain_name']} "
                                 f"({d['overlap_residues']} overlapping residues)")

            if a["sequence_context_1"]:
                ctx = a["sequence_context_1"][0]
                lines.append(f"  Sequence at hotspot (prot1): ...{ctx['sequence_fragment']}...")

            lines.append("")
        return "\n".join(lines)
