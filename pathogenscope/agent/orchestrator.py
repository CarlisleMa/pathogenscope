"""
Master Agent Orchestrator.

Single entry point for all downstream analysis. Holds pipeline state,
manages agent dependencies, and lets you run everything at once or
query individual analyses on demand.

Usage:
    from pathogenscope.agent import MasterAgent

    agent = MasterAgent(
        target_scores=scored_df,
        community_stats=stats_df,
        annotations=annotations_df,
        contact_maps=contact_maps,       # optional
        sequences=sequences,             # optional
        output_dir="results/my_run",
        config={"top_n": 10},
    )

    # Run everything
    results = agent.run_all()

    # Or run selectively
    agent.run("validation")
    agent.run("druggability")
    agent.run("chemical_matter")
    agent.run("report")

    # Query what's been run
    agent.status()

    # Access individual results
    agent.results["validation"].report
"""

import os
from dataclasses import dataclass, field

import pandas as pd
import torch

from .base import AgentResult
from .target_validation import TargetValidationAgent
from .druggability import DruggabilityAgent
from .community_interpretation import CommunityInterpretationAgent
from .interface_hotspot import InterfaceHotspotAgent
from .disruption_strategy import DisruptionStrategyAgent
from .chemical_matter import ChemicalMatterAgent
from .literature_search import LiteratureSearchAgent
from .report import ReportGeneratorAgent


# Agent dependency graph — each agent lists what must run before it
AGENT_DEPS = {
    "validation": [],
    "druggability": [],
    "community": [],
    "hotspot": [],
    "disruption": ["hotspot"],
    "chemical_matter": ["druggability"],
    "literature": ["community"],
    "report": [
        "validation", "druggability", "community",
        "chemical_matter", "literature",
    ],
}

# Execution order when running all agents
FULL_ORDER = [
    "validation",
    "druggability",
    "community",
    "hotspot",
    "disruption",
    "chemical_matter",
    "literature",
    "report",
]


class MasterAgent:
    """Orchestrates all downstream analysis agents."""

    def __init__(
        self,
        target_scores: pd.DataFrame,
        community_stats: pd.DataFrame | None = None,
        annotations: pd.DataFrame | None = None,
        contact_maps: dict | None = None,
        sequences: dict[str, str] | None = None,
        output_dir: str = "results",
        config: dict | None = None,
    ):
        self.target_scores = target_scores
        self.community_stats = community_stats
        self.annotations = annotations
        self.contact_maps = contact_maps or {}
        self.sequences = sequences
        self.output_dir = output_dir
        self.config = config or {}

        self.results: dict[str, AgentResult] = {}
        self._agents: dict[str, object] = {}

        os.makedirs(output_dir, exist_ok=True)

    def run_all(self, skip: list[str] | None = None) -> dict[str, AgentResult]:
        """
        Run all agents in dependency order.

        Args:
            skip: List of agent names to skip (e.g. ["hotspot", "disruption"])

        Returns:
            Dict of agent_name → AgentResult
        """
        skip = set(skip or [])

        # Auto-skip interface agents if no contact maps
        if not self.contact_maps:
            skip.update(["hotspot", "disruption"])

        print("=" * 60)
        print("DOWNSTREAM AGENT ANALYSIS")
        print("=" * 60)

        for name in FULL_ORDER:
            if name in skip:
                print(f"\n  Skipping {name}")
                continue
            self.run(name)

        self._print_summary()
        return self.results

    def run(self, agent_name: str) -> AgentResult:
        """
        Run a single agent, automatically resolving dependencies.

        Args:
            agent_name: One of: validation, druggability, community, hotspot,
                        disruption, chemical_matter, literature, report
        """
        if agent_name not in AGENT_DEPS:
            valid = ", ".join(AGENT_DEPS.keys())
            raise ValueError(f"Unknown agent '{agent_name}'. Valid agents: {valid}")

        # Run missing dependencies first
        for dep in AGENT_DEPS[agent_name]:
            if dep not in self.results:
                # Skip optional deps that can't run
                if dep in ("hotspot", "disruption") and not self.contact_maps:
                    continue
                self.run(dep)

        print(f"\n--- {agent_name.replace('_', ' ').title()} ---")
        result = self._dispatch(agent_name)
        self.results[agent_name] = result

        if result.warnings:
            for w in result.warnings:
                print(f"  WARNING: {w}")

        summary = result.data.get("summary")
        if summary:
            print(f"  -> {summary}")

        return result

    def _dispatch(self, name: str) -> AgentResult:
        """Instantiate and run the appropriate agent."""
        agent_config = {"top_n": self.config.get("top_n", 20)}

        if name == "validation":
            agent = TargetValidationAgent(agent_config)
            return agent.run(
                target_scores=self.target_scores,
                annotations=self.annotations,
            )

        elif name == "druggability":
            agent = DruggabilityAgent(agent_config)
            return agent.run(
                target_scores=self.target_scores,
                annotations=self.annotations,
            )

        elif name == "community":
            agent = CommunityInterpretationAgent()
            return agent.run(
                community_stats=self.community_stats,
                annotations=self.annotations,
            )

        elif name == "hotspot":
            if not self.contact_maps:
                result = AgentResult(agent_name="interface_hotspot")
                result.status = "error"
                result.warnings.append("No contact maps — re-run pipeline with --save_contact_maps")
                return result
            agent = InterfaceHotspotAgent()
            return agent.run(
                contact_maps=self.contact_maps,
                sequences=self.sequences,
                annotations=self.annotations,
            )

        elif name == "disruption":
            hotspot_result = self.results.get("hotspot")
            agent = DisruptionStrategyAgent()
            return agent.run(
                hotspot_results=hotspot_result,
                target_scores=self.target_scores,
            )

        elif name == "chemical_matter":
            drug_result = self.results.get("druggability")
            agent = ChemicalMatterAgent(agent_config)
            return agent.run(
                target_scores=self.target_scores,
                druggability=drug_result,
            )

        elif name == "literature":
            comm_result = self.results.get("community")
            disruption_result = self.results.get("disruption")
            agent = LiteratureSearchAgent(agent_config)
            return agent.run(
                target_scores=self.target_scores,
                community=comm_result,
                disruption=disruption_result,
                annotations=self.annotations,
            )

        elif name == "report":
            agent = ReportGeneratorAgent()
            return agent.run(
                target_scores=self.target_scores,
                validation=self.results.get("validation"),
                druggability=self.results.get("druggability"),
                community=self.results.get("community"),
                hotspot=self.results.get("hotspot"),
                disruption=self.results.get("disruption"),
                chemical_matter=self.results.get("chemical_matter"),
                literature=self.results.get("literature"),
                output_dir=self.output_dir,
            )

        raise ValueError(f"No dispatch for agent '{name}'")

    def status(self) -> dict[str, str]:
        """Show what has been run and current status of each agent."""
        table = {}
        for name in FULL_ORDER:
            if name in self.results:
                r = self.results[name]
                n_warnings = len(r.warnings)
                warn_str = f" ({n_warnings} warnings)" if n_warnings else ""
                table[name] = f"{r.status}{warn_str}"
            else:
                table[name] = "not run"

        print("\nAgent Status:")
        for name, st in table.items():
            marker = "OK" if "success" in st else ("SKIP" if "not run" in st else st.upper())
            print(f"  [{marker:>8}] {name}")
        return table

    def get_report(self) -> str:
        """Get the final dossier text, running report agent if needed."""
        if "report" not in self.results:
            self.run("report")
        return self.results["report"].report

    def get_result(self, agent_name: str) -> AgentResult | None:
        """Get a specific agent's result."""
        return self.results.get(agent_name)

    def rerun(self, agent_name: str) -> AgentResult:
        """Force re-run an agent (clears cached result first)."""
        self.results.pop(agent_name, None)
        return self.run(agent_name)

    def _print_summary(self):
        """Print final summary after run_all."""
        print("\n" + "=" * 60)
        print("AGENT ANALYSIS COMPLETE")
        print("=" * 60)
        self.status()

        report_result = self.results.get("report")
        if report_result:
            report_path = report_result.data.get("report_path", "")
            if report_path:
                print(f"\n  Target dossier: {report_path}")

        # Highlight key findings
        chem = self.results.get("chemical_matter")
        if chem and chem.status == "success":
            summary = chem.data.get("summary", {})
            print(f"  Chemical matter: {summary.get('total_direct_compounds', 0)} compounds found")

        lit = self.results.get("literature")
        if lit and lit.status == "success":
            summary = lit.data.get("summary", {})
            print(f"  Literature: {summary.get('total_papers_found', 0)} papers found")
