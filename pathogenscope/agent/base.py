"""
Base agent class for PathogenScope downstream analysis agents.

Each agent takes structured pipeline outputs (target scores, contact maps,
annotations) and produces analysis results + human-readable reports.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentResult:
    """Standard result container for all agents."""
    agent_name: str
    status: str = "success"  # success, partial, error
    data: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    report: str = ""


class BaseAgent(ABC):
    """Base class for downstream analysis agents."""

    name: str = "base"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def run(self, **inputs) -> AgentResult:
        """Run analysis and return structured results."""

    def _warn(self, result: AgentResult, msg: str):
        result.warnings.append(msg)
        print(f"  [{self.name}] WARNING: {msg}")
