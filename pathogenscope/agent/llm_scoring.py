"""
LLM Scoring Agent.

Uses Claude to synthesize all upstream agent results and produce:
  1. Per-target scores (0-1) with written reasoning
  2. Comparative ranking with rationale for ordering decisions

Unlike the fixed-weight formula in score_targets(), the LLM can reason
about tradeoffs: e.g. "high centrality but known resistance + off-target
risk → downgrade despite good druggability."
"""

import json
import os
import re

import anthropic
import pandas as pd

from .base import BaseAgent, AgentResult


SCORING_SYSTEM_PROMPT = """\
You are an expert in antimicrobial drug target discovery. You will be given
data about a candidate drug target protein from a pathogen, including PPI
network properties, druggability assessment, essentiality, off-target risk,
chemical matter availability, and literature evidence.

Score this target from 0.0 to 1.0 as a drug target candidate. Consider:
- Essential genes in the pathogen are better targets
- High off-target risk (human homology) is bad
- Druggable protein families (kinases, proteases) are preferred
- Existing chemical matter or clinical compounds is a strong positive
- Known resistance mechanisms are a concern
- High PPI centrality suggests functional importance
- Literature evidence of successful inhibition is very positive

Respond with ONLY valid JSON (no markdown fences):
{
  "score": <float 0.0-1.0>,
  "reasoning": "<2-3 sentence explanation of score>",
  "strengths": ["<strength1>", "<strength2>"],
  "concerns": ["<concern1>", "<concern2>"],
  "recommendation": "<one of: prioritize, investigate, deprioritize>"
}"""

RANKING_SYSTEM_PROMPT = """\
You are an expert in antimicrobial drug target prioritization. You will be
given a list of scored drug target candidates from a pathogen proteome.

Produce a comparative ranking. For each pair of adjacent targets in your
ranking, briefly explain why the higher-ranked target is preferred.

Respond with ONLY valid JSON (no markdown fences):
{
  "ranking": [
    {
      "rank": 1,
      "protein_id": "<id>",
      "llm_score": <float>,
      "rationale": "<why this is ranked here vs neighbors>"
    }
  ],
  "top_pick_summary": "<1-2 sentence summary of the best target and why>",
  "portfolio_assessment": "<1-2 sentence assessment of overall target portfolio quality>"
}"""


class LLMScoringAgent(BaseAgent):
    """Uses Claude to score and rank drug targets from upstream agent results."""

    name = "llm_scoring"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.top_n = self.config.get("top_n", 20)
        self.model = self.config.get("model", "claude-sonnet-4-20250514")
        self.max_tokens = self.config.get("max_tokens", 1024)
        self._client = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def run(
        self,
        target_scores: pd.DataFrame,
        validation: AgentResult | None = None,
        druggability: AgentResult | None = None,
        community: AgentResult | None = None,
        chemical_matter: AgentResult | None = None,
        literature: AgentResult | None = None,
        **kwargs,
    ) -> AgentResult:
        """
        Score and rank targets using Claude.

        Args:
            target_scores: Ranked targets from pipeline
            validation: TargetValidationAgent results
            druggability: DruggabilityAgent results
            community: CommunityInterpretationAgent results
            chemical_matter: ChemicalMatterAgent results
            literature: LiteratureSearchAgent results
        """
        result = AgentResult(agent_name=self.name)

        # Index upstream results by protein_id
        val_map = self._index_by_pid(validation, "validations")
        drug_map = self._index_by_pid(druggability, "assessments")
        chem_map = self._index_by_pid(chemical_matter, "assessments")
        lit_map = self._index_by_pid(literature, "assessments")
        comm_map = self._build_community_map(community)

        # Phase 1: Score each target individually
        print(f"  [{self.name}] Scoring {min(self.top_n, len(target_scores))} targets with Claude ({self.model})...")
        scored_targets = []
        for _, row in target_scores.head(self.top_n).iterrows():
            pid = row["protein_id"]
            target_data = self._build_target_context(
                row, val_map.get(pid), drug_map.get(pid),
                comm_map.get(pid), chem_map.get(pid), lit_map.get(pid),
            )

            print(f"    Scoring {pid.split('|')[-1] if '|' in pid else pid}...")
            score_result = self._score_target(pid, target_data)
            scored_targets.append(score_result)

        # Phase 2: Comparative ranking
        print(f"  [{self.name}] Generating comparative ranking...")
        ranking = self._comparative_ranking(scored_targets)

        result.data["scored_targets"] = scored_targets
        result.data["ranking"] = ranking
        result.data["summary"] = self._summarize(scored_targets, ranking)
        result.report = self._format_report(scored_targets, ranking)
        return result

    def _index_by_pid(
        self, agent_result: AgentResult | None, key: str
    ) -> dict[str, dict]:
        """Index agent result entries by protein_id."""
        if agent_result is None or agent_result.status == "error":
            return {}
        entries = agent_result.data.get(key, [])
        return {e["protein_id"]: e for e in entries if "protein_id" in e}

    def _build_community_map(
        self, community: AgentResult | None
    ) -> dict[str, dict]:
        """Map protein_id → community interpretation."""
        if community is None or community.status == "error":
            return {}
        comm_map = {}
        for interp in community.data.get("interpretations", []):
            for member in interp.get("members", []):
                comm_map[member] = {
                    "community_id": interp["community_id"],
                    "function_label": interp["function_label"],
                    "therapeutic_relevance": interp["therapeutic_relevance"],
                    "community_size": interp["topology"]["size"],
                    "is_hub": member == interp["topology"]["hub_protein"],
                }
        return comm_map

    def _build_target_context(
        self,
        row: pd.Series,
        validation: dict | None,
        druggability: dict | None,
        community: dict | None,
        chemical: dict | None,
        literature: dict | None,
    ) -> dict:
        """Build a structured context dict for one target."""
        ctx = {
            "network_properties": {
                "target_score": float(row["target_score"]),
                "ppi_edges": int(row["ppi_edges"]),
                "betweenness_centrality": float(row["betweenness_centrality"]),
                "max_contact_score": float(row["ppi_max_contact_score"]),
                "community": int(row["ppi_community"]),
            }
        }

        if validation:
            ctx["validation"] = {
                "is_essential": validation["is_essential"],
                "off_target_risk": validation["off_target_risk"],
                "drug_target_signals": validation["drug_target_signals"],
            }

        if druggability:
            ctx["druggability"] = {
                "tier": druggability["druggability_tier"],
                "score": druggability["druggability_score"],
                "family": druggability["family_match"] or "unclassified",
                "alphafold_structure": druggability["alphafold_structure"],
                "binding_sites": druggability["binding_site_count"],
            }

        if community:
            ctx["community"] = community

        if chemical:
            ctx["chemical_matter"] = {
                "tractability": chemical["tractability"]["level"],
                "assessment": chemical["tractability"]["assessment"],
                "direct_compounds": chemical["direct_compound_count"],
                "clinical_compounds": len(chemical["clinical_compounds"]),
            }

        if literature:
            findings = literature.get("findings", {})
            ctx["literature"] = {
                "total_papers": literature["total_papers"],
                "inhibitor_studies": len(literature.get("inhibitor_studies", [])),
                "genetic_perturbation": len(literature.get("genetic_perturbation", [])),
                "resistance_studies": len(literature.get("resistance", [])),
                "known_inhibitors": len(findings.get("known_inhibitors", [])),
                "essential_evidence": len(findings.get("essential_evidence", [])),
            }

        return ctx

    def _score_target(self, protein_id: str, target_data: dict) -> dict:
        """Call Claude to score a single target."""
        user_msg = (
            f"Score this drug target candidate:\n\n"
            f"Protein: {protein_id}\n\n"
            f"Data:\n{json.dumps(target_data, indent=2)}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SCORING_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text
            parsed = self._parse_json(text)

            return {
                "protein_id": protein_id,
                "llm_score": float(parsed.get("score", 0.5)),
                "reasoning": parsed.get("reasoning", ""),
                "strengths": parsed.get("strengths", []),
                "concerns": parsed.get("concerns", []),
                "recommendation": parsed.get("recommendation", "investigate"),
                "input_data": target_data,
            }
        except Exception as e:
            self._warn_msg = f"LLM scoring failed for {protein_id}: {e}"
            print(f"    WARNING: {self._warn_msg}")
            return {
                "protein_id": protein_id,
                "llm_score": target_data["network_properties"]["target_score"],
                "reasoning": f"LLM scoring failed ({e}), falling back to formula score",
                "strengths": [],
                "concerns": [],
                "recommendation": "investigate",
                "input_data": target_data,
            }

    def _comparative_ranking(self, scored_targets: list[dict]) -> dict:
        """Call Claude to produce a comparative ranking of all scored targets."""
        summary_for_ranking = []
        for t in scored_targets:
            summary_for_ranking.append({
                "protein_id": t["protein_id"],
                "llm_score": t["llm_score"],
                "recommendation": t["recommendation"],
                "reasoning": t["reasoning"],
                "strengths": t["strengths"],
                "concerns": t["concerns"],
            })

        # Sort by score descending for context
        summary_for_ranking.sort(key=lambda x: x["llm_score"], reverse=True)

        user_msg = (
            f"Rank these {len(summary_for_ranking)} drug target candidates "
            f"and explain the comparative reasoning:\n\n"
            f"{json.dumps(summary_for_ranking, indent=2)}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=RANKING_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text
            parsed = self._parse_json(text)
            return parsed
        except Exception as e:
            print(f"    WARNING: Comparative ranking failed ({e}), using score order")
            return {
                "ranking": [
                    {
                        "rank": i + 1,
                        "protein_id": t["protein_id"],
                        "llm_score": t["llm_score"],
                        "rationale": t["reasoning"],
                    }
                    for i, t in enumerate(summary_for_ranking)
                ],
                "top_pick_summary": f"Top target: {summary_for_ranking[0]['protein_id']}" if summary_for_ranking else "",
                "portfolio_assessment": "Ranking based on individual scores (comparative analysis unavailable)",
            }

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response, stripping markdown fences if present."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        return json.loads(text)

    def _summarize(self, scored_targets: list[dict], ranking: dict) -> dict:
        prioritize = sum(1 for t in scored_targets if t["recommendation"] == "prioritize")
        investigate = sum(1 for t in scored_targets if t["recommendation"] == "investigate")
        deprioritize = sum(1 for t in scored_targets if t["recommendation"] == "deprioritize")
        scores = [t["llm_score"] for t in scored_targets]

        return {
            "total_scored": len(scored_targets),
            "mean_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "max_score": round(max(scores), 3) if scores else 0,
            "recommendations": {
                "prioritize": prioritize,
                "investigate": investigate,
                "deprioritize": deprioritize,
            },
            "top_pick": ranking.get("top_pick_summary", ""),
            "portfolio": ranking.get("portfolio_assessment", ""),
        }

    def _format_report(self, scored_targets: list[dict], ranking: dict) -> str:
        lines = ["# LLM Target Scoring Report\n"]

        # Portfolio overview
        portfolio = ranking.get("portfolio_assessment", "")
        if portfolio:
            lines.append(f"**Portfolio assessment**: {portfolio}\n")

        top_pick = ranking.get("top_pick_summary", "")
        if top_pick:
            lines.append(f"**Top pick**: {top_pick}\n")

        # Comparative ranking
        lines.append("## Comparative Ranking\n")
        for entry in ranking.get("ranking", []):
            pid = entry["protein_id"]
            name = pid.split("|")[-1] if "|" in pid else pid
            lines.append(
                f"{entry['rank']}. **{name}** — score={entry['llm_score']:.2f}"
            )
            lines.append(f"   {entry['rationale']}")
            lines.append("")

        # Detailed per-target cards
        lines.append("## Detailed Scores\n")
        sorted_targets = sorted(scored_targets, key=lambda t: t["llm_score"], reverse=True)
        for t in sorted_targets:
            pid = t["protein_id"]
            name = pid.split("|")[-1] if "|" in pid else pid
            rec = t["recommendation"].upper()
            lines.append(f"### {name} — {t['llm_score']:.2f} [{rec}]")
            lines.append(f"  {t['reasoning']}")
            if t["strengths"]:
                lines.append(f"  Strengths: {', '.join(t['strengths'])}")
            if t["concerns"]:
                lines.append(f"  Concerns: {', '.join(t['concerns'])}")
            lines.append("")

        return "\n".join(lines)
