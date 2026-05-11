"""Core dataclasses shared across the Harness-MU pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModuleResult:
    """The output produced by one module execution."""
    module_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    token_usage: Optional[Dict[str, int]] = None  # {"prompt_tokens": N, "completion_tokens": M}
    error: Optional[str] = None


@dataclass
class PipelineContext:
    """Mutable state that flows through the module pipeline for one scenario."""
    scenario_id: str
    scenario_data: Dict[str, Any] = field(default_factory=dict)
    users: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    module_results: List[ModuleResult] = field(default_factory=list)


@dataclass
class ScenarioRecord:
    """The complete result of processing one scenario line."""
    scenario_id: str
    module_results: List[ModuleResult] = field(default_factory=list)
    instruction_results: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw_outputs: Dict[str, str] = field(default_factory=dict)


@dataclass
class CallMetrics:
    """Per-call instrumentation record for an LLM invocation."""
    module_name: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
