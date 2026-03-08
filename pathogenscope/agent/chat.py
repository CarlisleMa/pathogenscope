"""
Interactive Chat Agent — Claude-powered conversational interface.

Uses the Anthropic SDK with tool use to let users interact with the
PathogenScope pipeline through natural language. Claude decides which
agents to run, interprets results, and answers follow-up questions.

Usage:
    from pathogenscope.agent import ChatAgent

    chat = ChatAgent(
        target_scores=scored_df,
        community_stats=stats_df,
        annotations=annotations_df,
        output_dir="results/my_run",
    )
    chat.start()  # interactive chat loop
"""

import json
import os
import traceback

import anthropic
import pandas as pd

from .orchestrator import MasterAgent, AGENT_DEPS, FULL_ORDER
from .base import AgentResult


# Descriptions for each agent, used in system prompt and list_agents tool
AGENT_DESCRIPTIONS = {
    "validation": (
        "Cross-references targets against the Database of Essential Genes, "
        "scans UniProt annotations for drug-target signals, and checks "
        "human homology for off-target risk."
    ),
    "druggability": (
        "Assesses druggability by checking AlphaFold structure availability, "
        "protein family classification, known binding sites, and subcellular "
        "accessibility. Produces a tiered score (Tier 1-3)."
    ),
    "community": (
        "Interprets each PPI community's biological function using GO terms "
        "and gene names, and assesses therapeutic relevance of disrupting "
        "that module."
    ),
    "hotspot": (
        "Analyzes residue-level contact maps to find interface hotspots, "
        "contiguous patches, and maps them to Pfam/InterPro domains. "
        "Requires contact maps from the pipeline."
    ),
    "disruption": (
        "Suggests PPI disruption strategies (small molecule, peptide mimetic, "
        "antibody) based on interface topology and domain context. "
        "Depends on hotspot agent."
    ),
    "chemical_matter": (
        "Searches ChEMBL for existing bioactive compounds against each target "
        "or its protein family. Reports tractability level and clinical status."
    ),
    "literature": (
        "Searches PubMed for inhibitor studies, knockout experiments, "
        "resistance mechanisms, and structural druggability papers for each target."
    ),
    "llm_scoring": (
        "Uses Claude to score each target (0-1) with reasoning, then produces "
        "a comparative ranking with rationale. Synthesizes all upstream agent data."
    ),
    "report": (
        "Compiles all agent outputs into a unified target dossier with "
        "executive summary, per-target cards, and prioritized recommendations."
    ),
}


# Tool definitions for the Claude API
TOOLS = [
    {
        "name": "run_agent",
        "description": (
            "Run a specific analysis agent. Automatically resolves dependencies "
            "(e.g., running 'disruption' will first run 'hotspot' if needed). "
            "Use 'all' to run the full pipeline."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": (
                        "Name of the agent to run. One of: "
                        + ", ".join(FULL_ORDER)
                        + ", or 'all' to run everything."
                    ),
                },
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "get_status",
        "description": (
            "Get the current run status of all agents. Shows which have been "
            "run, their status (success/partial/error), and warning counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "list_agents",
        "description": (
            "List all available agents with their descriptions and dependency "
            "chains. Useful when the user asks what agents are available or "
            "what a specific agent does."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_result",
        "description": (
            "Get the structured data output from a completed agent. Returns "
            "the full result.data dict as JSON. Use this to inspect raw data "
            "like scores, assessments, compound lists, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent whose results to retrieve.",
                },
                "key": {
                    "type": "string",
                    "description": (
                        "Optional specific key within result.data to retrieve "
                        "(e.g., 'summary', 'assessments', 'ranking'). "
                        "If omitted, returns all data."
                    ),
                },
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "get_report",
        "description": (
            "Get the human-readable markdown report from a completed agent. "
            "Use this when the user wants to see formatted results or when "
            "you need to interpret findings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent whose report to retrieve.",
                },
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "get_target_info",
        "description": (
            "Get all available information about a specific protein target "
            "across all completed agents. Aggregates validation, druggability, "
            "chemical matter, literature, and scoring data for one protein."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Protein identifier to search for. Can be a full ID "
                        "(sp|Q49425|RUVB_MYCGE), accession (Q49425), or gene "
                        "name (RUVB). Partial matches are supported."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_top_targets",
        "description": (
            "Get the top N targets from the pipeline's target scoring. "
            "Returns protein IDs, target scores, PPI edges, community, "
            "and centrality metrics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of top targets to return. Default 10.",
                },
            },
        },
    },
    {
        "name": "rerun_agent",
        "description": (
            "Force re-run an agent, clearing its cached result first. "
            "Useful if upstream data changed or you want fresh results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent to re-run.",
                },
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "save_report",
        "description": (
            "Generate the full target dossier (running the report agent if "
            "needed) and save it to disk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


SYSTEM_PROMPT = """\
You are PathogenScope Assistant, an expert in antimicrobial drug target \
discovery. You help researchers analyze pathogen proteomes to identify and \
prioritize drug targets.

You have access to the PathogenScope analysis pipeline, which includes:
{agent_list}

## How to help the user

1. **When the user asks to run analysis**: Use the run_agent tool. You can \
run individual agents or the full pipeline.
2. **When the user asks about results**: Use get_result or get_report to \
retrieve data, then interpret and explain the findings.
3. **When the user asks about a specific protein**: Use get_target_info to \
aggregate all known data about that target.
4. **When the user asks what's available**: Use list_agents or get_status.
5. **When interpreting results**: Be specific about what the data means for \
drug discovery. Explain tradeoffs (e.g., essential but has human homolog).

## Current pipeline state
- {n_targets} proteins loaded in target scores
- Community stats: {"available" if "{has_community}" == "True" else "not loaded"}
- Annotations: {"available" if "{has_annotations}" == "True" else "not loaded"}
- Contact maps: {"available" if "{has_contacts}" == "True" else "not available"}
- Output directory: {output_dir}

Be concise but thorough. Use scientific terminology appropriate for drug \
discovery researchers. When presenting results, highlight actionable insights."""


class ChatAgent:
    """Claude-powered conversational interface to PathogenScope."""

    def __init__(
        self,
        target_scores: pd.DataFrame,
        community_stats: pd.DataFrame | None = None,
        annotations: pd.DataFrame | None = None,
        contact_maps: dict | None = None,
        sequences: dict[str, str] | None = None,
        output_dir: str = "results",
        config: dict | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.config = config or {}
        self.model = model
        self.client = anthropic.Anthropic()

        # Initialize the MasterAgent that does the actual work
        self.master = MasterAgent(
            target_scores=target_scores,
            community_stats=community_stats,
            annotations=annotations,
            contact_maps=contact_maps,
            sequences=sequences,
            output_dir=output_dir,
            config=self.config,
        )

        self.messages: list[dict] = []

    def _build_system_prompt(self) -> str:
        agent_list = "\n".join(
            f"- **{name}**: {desc}"
            for name, desc in AGENT_DESCRIPTIONS.items()
        )
        return SYSTEM_PROMPT.format(
            agent_list=agent_list,
            n_targets=len(self.master.target_scores),
            has_community=self.master.community_stats is not None,
            has_annotations=self.master.annotations is not None,
            has_contacts=bool(self.master.contact_maps),
            output_dir=self.master.output_dir,
        )

    def start(self):
        """Start the interactive chat loop."""
        print("\n" + "=" * 60)
        print("PATHOGENSCOPE CHAT AGENT")
        print("=" * 60)
        print(f"  Model    : {self.model}")
        print(f"  Targets  : {len(self.master.target_scores)} proteins")
        print(f"  Output   : {self.master.output_dir}")
        print(f"\n  Type your questions in natural language.")
        print(f"  Type 'quit' or 'exit' to end the session.\n")

        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nEnding session.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Ending session.")
                break

            response = self.chat(user_input)
            print(f"\nassistant> {response}\n")

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, executing any tool calls."""
        self.messages.append({"role": "user", "content": user_message})

        # Agentic loop — keep going until Claude produces a final text response
        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._build_system_prompt(),
                tools=TOOLS,
                messages=self.messages,
            )

            # Collect the response content
            self.messages.append({"role": "assistant", "content": response.content})

            # Check if there are tool calls to process
            tool_uses = [
                block for block in response.content
                if block.type == "tool_use"
            ]

            if not tool_uses:
                # No tool calls — extract text response
                text_parts = [
                    block.text for block in response.content
                    if hasattr(block, "text")
                ]
                return "\n".join(text_parts) if text_parts else "(no response)"

            # Process tool calls and build results
            tool_results = []
            for tool_use in tool_uses:
                result = self._execute_tool(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                })

            self.messages.append({"role": "user", "content": tool_results})

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result as a string."""
        try:
            if tool_name == "run_agent":
                return self._tool_run_agent(tool_input)
            elif tool_name == "get_status":
                return self._tool_get_status()
            elif tool_name == "list_agents":
                return self._tool_list_agents()
            elif tool_name == "get_result":
                return self._tool_get_result(tool_input)
            elif tool_name == "get_report":
                return self._tool_get_report(tool_input)
            elif tool_name == "get_target_info":
                return self._tool_get_target_info(tool_input)
            elif tool_name == "get_top_targets":
                return self._tool_get_top_targets(tool_input)
            elif tool_name == "rerun_agent":
                return self._tool_rerun_agent(tool_input)
            elif tool_name == "save_report":
                return self._tool_save_report()
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    def _tool_run_agent(self, input: dict) -> str:
        agent_name = input["agent_name"]

        if agent_name == "all":
            skip = [] if self.master.contact_maps else ["hotspot", "disruption"]
            results = self.master.run_all(skip=skip)
            summary = {}
            for name, result in results.items():
                summary[name] = {
                    "status": result.status,
                    "warnings": len(result.warnings),
                    "has_report": bool(result.report),
                }
            return json.dumps({
                "action": "run_all",
                "agents_run": list(results.keys()),
                "summary": summary,
            }, default=str)

        if agent_name not in AGENT_DEPS:
            return json.dumps({
                "error": f"Unknown agent '{agent_name}'",
                "valid_agents": list(AGENT_DEPS.keys()),
            })

        result = self.master.run(agent_name)
        return json.dumps({
            "agent": agent_name,
            "status": result.status,
            "warnings": result.warnings,
            "summary": result.data.get("summary", {}),
            "has_report": bool(result.report),
        }, default=str)

    def _tool_get_status(self) -> str:
        status = {}
        for name in FULL_ORDER:
            if name in self.master.results:
                r = self.master.results[name]
                status[name] = {
                    "status": r.status,
                    "warnings": len(r.warnings),
                }
            else:
                status[name] = {"status": "not_run"}
        return json.dumps(status)

    def _tool_list_agents(self) -> str:
        agents = []
        for name in FULL_ORDER:
            agents.append({
                "name": name,
                "description": AGENT_DESCRIPTIONS.get(name, ""),
                "dependencies": AGENT_DEPS.get(name, []),
                "status": (
                    self.master.results[name].status
                    if name in self.master.results
                    else "not_run"
                ),
            })
        return json.dumps(agents, indent=2)

    def _tool_get_result(self, input: dict) -> str:
        agent_name = input["agent_name"]
        result = self.master.results.get(agent_name)
        if result is None:
            return json.dumps({
                "error": f"Agent '{agent_name}' has not been run yet. "
                         f"Use run_agent first.",
            })

        data = result.data
        key = input.get("key")
        if key:
            if key in data:
                data = {key: data[key]}
            else:
                return json.dumps({
                    "error": f"Key '{key}' not found in {agent_name} results",
                    "available_keys": list(result.data.keys()),
                })

        return json.dumps(data, indent=2, default=str)

    def _tool_get_report(self, input: dict) -> str:
        agent_name = input["agent_name"]
        result = self.master.results.get(agent_name)
        if result is None:
            return json.dumps({
                "error": f"Agent '{agent_name}' has not been run yet.",
            })
        if not result.report:
            return json.dumps({
                "warning": f"Agent '{agent_name}' produced no report text.",
                "suggestion": "Use get_result to see raw data instead.",
            })
        return result.report

    def _tool_get_target_info(self, input: dict) -> str:
        query = input["query"].lower()
        targets = self.master.target_scores

        # Find matching protein
        matches = []
        for pid in targets["protein_id"]:
            if query in pid.lower():
                matches.append(pid)

        if not matches:
            return json.dumps({
                "error": f"No protein matching '{query}' found",
                "suggestion": "Try a shorter query or use get_top_targets to see available proteins.",
            })

        pid = matches[0]  # best match
        info = {"protein_id": pid, "matches_found": len(matches)}

        # Pipeline scores
        row = targets[targets["protein_id"] == pid].iloc[0]
        info["pipeline_scores"] = {
            col: (float(row[col]) if pd.api.types.is_float_dtype(type(row[col])) else
                  int(row[col]) if pd.api.types.is_integer_dtype(type(row[col])) else
                  str(row[col]))
            for col in targets.columns
        }

        # Aggregate from all completed agents
        for agent_name, result in self.master.results.items():
            agent_data = result.data
            # Search through assessments/validations/analyses
            for key in ["validations", "assessments", "analyses"]:
                entries = agent_data.get(key, [])
                for entry in entries:
                    if entry.get("protein_id") == pid:
                        info[agent_name] = entry
                        break

        return json.dumps(info, indent=2, default=str)

    def _tool_get_top_targets(self, input: dict) -> str:
        n = input.get("n", 10)
        top = self.master.target_scores.head(n)

        targets = []
        for _, row in top.iterrows():
            pid = row["protein_id"]
            name = pid.split("|")[-1] if "|" in pid else pid
            targets.append({
                "protein_id": pid,
                "short_name": name,
                "target_score": round(float(row["target_score"]), 4),
                "ppi_edges": int(row["ppi_edges"]),
                "community": int(row["ppi_community"]),
                "betweenness_centrality": round(float(row["betweenness_centrality"]), 4),
                "max_contact_score": round(float(row["ppi_max_contact_score"]), 4),
            })

        return json.dumps(targets, indent=2)

    def _tool_rerun_agent(self, input: dict) -> str:
        agent_name = input["agent_name"]
        if agent_name not in AGENT_DEPS:
            return json.dumps({
                "error": f"Unknown agent '{agent_name}'",
                "valid_agents": list(AGENT_DEPS.keys()),
            })

        result = self.master.rerun(agent_name)
        return json.dumps({
            "agent": agent_name,
            "status": result.status,
            "warnings": result.warnings,
            "summary": result.data.get("summary", {}),
        }, default=str)

    def _tool_save_report(self) -> str:
        report_text = self.master.get_report()
        result = self.master.results.get("report")
        path = result.data.get("report_path", "") if result else ""
        return json.dumps({
            "status": "saved",
            "path": path,
            "report_length": len(report_text),
        })
