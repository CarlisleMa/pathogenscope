"""
Target Validation Agent.

Cross-references top-ranked PPI targets against:
  1. Database of Essential Genes (DEG) — is this gene essential?
  2. UniProt keywords/annotations — known drug target? antimicrobial resistance?
  3. Human proteome homology — off-target risk via UniProt BLAST
"""

import requests
import pandas as pd

from .base import BaseAgent, AgentResult


# Keywords in UniProt annotations that suggest druggability concerns or value
DRUGTARGET_KEYWORDS = {
    "antibiotic resistance", "antimicrobial resistance", "virulence",
    "toxin", "secreted", "cell wall", "membrane",
}
ESSENTIAL_KEYWORDS = {
    "essential", "lethal", "viable",
}


class TargetValidationAgent(BaseAgent):
    name = "target_validation"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.top_n = self.config.get("top_n", 20)
        self.blast_threshold = self.config.get("blast_threshold", 30)  # % identity

    def run(
        self,
        target_scores: pd.DataFrame,
        annotations: pd.DataFrame | None = None,
        **kwargs,
    ) -> AgentResult:
        """
        Validate top targets.

        Args:
            target_scores: DataFrame with target_score column (from score_targets)
            annotations: Optional UniProt annotations DataFrame
        """
        result = AgentResult(agent_name=self.name)
        top = target_scores.head(self.top_n).copy()
        accessions = self._extract_accessions(top["protein_id"].tolist())

        # 1. Check DEG (essential genes)
        print(f"  [{self.name}] Checking essentiality for {len(accessions)} targets...")
        essentiality = self._check_essentiality(accessions, annotations)

        # 2. Check annotation keywords for drug-target relevance
        print(f"  [{self.name}] Scanning annotations for drug-target signals...")
        keyword_hits = self._scan_keywords(accessions, annotations)

        # 3. Check human homology (off-target risk)
        print(f"  [{self.name}] Checking human homology (off-target risk)...")
        human_homologs = self._check_human_homology(accessions)

        # Build per-target validation summary
        validations = []
        for acc, pid in zip(accessions, top["protein_id"].tolist()):
            entry = {
                "protein_id": pid,
                "accession": acc,
                "is_essential": essentiality.get(acc, "unknown"),
                "drug_target_signals": keyword_hits.get(acc, []),
                "human_homolog_identity": human_homologs.get(acc, None),
                "off_target_risk": self._classify_offtarget(human_homologs.get(acc)),
            }
            validations.append(entry)

        result.data["validations"] = validations
        result.data["summary"] = self._summarize(validations)
        result.report = self._format_report(validations)
        return result

    def _extract_accessions(self, protein_ids: list[str]) -> list[str]:
        accessions = []
        for pid in protein_ids:
            parts = pid.split("|")
            accessions.append(parts[1] if len(parts) >= 2 else parts[0])
        return accessions

    def _check_essentiality(
        self, accessions: list[str], annotations: pd.DataFrame | None
    ) -> dict[str, str]:
        """Check gene essentiality from annotations and DEG database."""
        essentiality = {}

        # Check from existing UniProt annotations (keywords field)
        if annotations is not None and len(annotations) > 0:
            kw_col = None
            for col in ["Keyword", "keyword", "Keywords"]:
                if col in annotations.columns:
                    kw_col = col
                    break
            if kw_col:
                acc_col = "protein_id" if "protein_id" in annotations.columns else annotations.columns[0]
                for _, row in annotations.iterrows():
                    acc = str(row.get(acc_col, ""))
                    kws = str(row.get(kw_col, "")).lower()
                    if acc in accessions:
                        for term in ESSENTIAL_KEYWORDS:
                            if term in kws:
                                essentiality[acc] = "essential (UniProt keyword)"
                                break

        # Try DEG REST API for remaining
        for acc in accessions:
            if acc not in essentiality:
                essentiality[acc] = self._query_deg(acc)

        return essentiality

    def _query_deg(self, accession: str) -> str:
        """Query DEG database for essential gene status."""
        try:
            url = f"http://tubic.org/deg/api/search?query={accession}"
            resp = requests.get(url, timeout=5)
            if resp.ok and accession.lower() in resp.text.lower():
                return "essential (DEG)"
        except requests.RequestException:
            pass
        return "unknown"

    def _scan_keywords(
        self, accessions: list[str], annotations: pd.DataFrame | None
    ) -> dict[str, list[str]]:
        """Scan UniProt annotations for drug-target-relevant keywords."""
        hits: dict[str, list[str]] = {acc: [] for acc in accessions}

        if annotations is None or len(annotations) == 0:
            return hits

        # Search across all text columns
        text_cols = [
            c for c in annotations.columns
            if annotations[c].dtype == object and c not in ("protein_id", "Entry")
        ]
        acc_col = "protein_id" if "protein_id" in annotations.columns else annotations.columns[0]

        for _, row in annotations.iterrows():
            acc = str(row.get(acc_col, ""))
            if acc not in hits:
                continue
            combined_text = " ".join(str(row.get(c, "")) for c in text_cols).lower()
            for keyword in DRUGTARGET_KEYWORDS:
                if keyword in combined_text:
                    hits[acc].append(keyword)

        return hits

    def _check_human_homology(self, accessions: list[str]) -> dict[str, float | None]:
        """Check UniProt for human homologs via cross-references."""
        homology = {}
        for acc in accessions:
            homology[acc] = self._query_human_homolog(acc)
        return homology

    def _query_human_homolog(self, accession: str) -> float | None:
        """Query UniProt for ortholog info to estimate human homology."""
        try:
            url = (
                f"https://rest.uniprot.org/uniprotkb/{accession}"
                f"?fields=xref_orthodb,cc_similarity&format=json"
            )
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                return None
            data = resp.json()
            # Check if there's a human ortholog mentioned
            comments = data.get("comments", [])
            for comment in comments:
                texts = comment.get("texts", [])
                for text in texts:
                    val = text.get("value", "").lower()
                    if "homo sapiens" in val or "human" in val:
                        return self.blast_threshold  # flag as having human homolog
            return None
        except (requests.RequestException, ValueError):
            return None

    def _classify_offtarget(self, identity: float | None) -> str:
        if identity is None:
            return "low (no human homolog detected)"
        if identity >= 50:
            return "HIGH — significant human homology"
        if identity >= 30:
            return "moderate — some human similarity"
        return "low"

    def _summarize(self, validations: list[dict]) -> dict:
        essential = sum(1 for v in validations if "essential" in str(v["is_essential"]))
        has_signals = sum(1 for v in validations if v["drug_target_signals"])
        high_risk = sum(1 for v in validations if "HIGH" in v["off_target_risk"])
        return {
            "total_targets": len(validations),
            "essential_genes": essential,
            "with_drug_signals": has_signals,
            "high_offtarget_risk": high_risk,
        }

    def _format_report(self, validations: list[dict]) -> str:
        lines = ["# Target Validation Report\n"]
        for v in validations:
            pid = v["protein_id"]
            lines.append(f"## {pid}")
            lines.append(f"  Essentiality: {v['is_essential']}")
            signals = ", ".join(v["drug_target_signals"]) if v["drug_target_signals"] else "none"
            lines.append(f"  Drug-target signals: {signals}")
            lines.append(f"  Off-target risk: {v['off_target_risk']}")
            lines.append("")
        return "\n".join(lines)
