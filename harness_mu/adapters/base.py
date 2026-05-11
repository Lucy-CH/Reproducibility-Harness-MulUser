"""Abstract base for scenario adapters.

Each Muses-Bench scenario gets its own adapter that knows how to:
1. Parse a JSONL line into a PipelineContext.
2. Orchestrate the module pipeline for that scenario.
3. Evaluate results and compute metrics.
"""

from abc import ABC, abstractmethod

from harness_mu.core.types import PipelineContext, ScenarioRecord


class ScenarioAdapter(ABC):
    """Contract for per-scenario adapters.

    An adapter maps a raw JSONL line into the Harness-MU pipeline,
    runs the appropriate modules in sequence, and returns a
    ScenarioRecord with all results and metrics.
    """

    @abstractmethod
    def parse_scenario(self, raw_line: str) -> PipelineContext:
        """Parse a JSONL line into a populated PipelineContext."""
        ...

    @abstractmethod
    def run(self, raw_line: str) -> ScenarioRecord:
        """Execute the full pipeline for one scenario line."""
        ...
