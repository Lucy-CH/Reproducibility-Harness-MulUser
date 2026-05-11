"""Thread-safe metrics collector for per-module and per-scenario instrumentation."""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List

from harness_mu.core.types import CallMetrics


@dataclass
class ModuleSummary:
    """Aggregated metrics for one module across all scenarios."""
    module_name: str
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    model: str = ""

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens


class MetricsCollector:
    """Thread-safe accumulator for LLM call metrics.

    Aggregates per-module statistics (calls, tokens, latency) across all
    scenarios in a batch run. Thread-safe via internal lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._calls: List[CallMetrics] = []
        self._scenario_metrics: Dict[str, Dict[str, Any]] = {}

    def record_call(self, metrics: CallMetrics) -> None:
        """Record a single LLM invocation."""
        with self._lock:
            self._calls.append(metrics)

    def record_scenario_metrics(self, scenario_id: str, metrics: Dict[str, Any]) -> None:
        """Record per-scenario evaluation metrics."""
        with self._lock:
            self._scenario_metrics[scenario_id] = metrics

    def get_module_summaries(self) -> Dict[str, ModuleSummary]:
        """Aggregate LLM call metrics by module name."""
        with self._lock:
            summaries: Dict[str, ModuleSummary] = {}
            for call in self._calls:
                if call.module_name not in summaries:
                    summaries[call.module_name] = ModuleSummary(module_name=call.module_name)
                s = summaries[call.module_name]
                s.total_calls += 1
                s.total_prompt_tokens += call.prompt_tokens
                s.total_completion_tokens += call.completion_tokens
                s.total_latency_ms += call.latency_ms
                s.model = call.model
            return summaries

    def get_scenario_summary(self) -> Dict[str, Any]:
        """Return aggregated scenario metrics across all scenarios."""
        with self._lock:
            if not self._scenario_metrics:
                return {}
            # Average numeric metrics across all scenarios
            keys = set()
            for m in self._scenario_metrics.values():
                keys.update(k for k, v in m.items() if isinstance(v, (int, float)))
            aggregated = {}
            for key in keys:
                values = [m[key] for m in self._scenario_metrics.values() if key in m]
                aggregated[f"avg_{key}"] = sum(values) / len(values) if values else 0.0
            aggregated["total_scenarios"] = len(self._scenario_metrics)
            return aggregated

    @property
    def total_calls(self) -> int:
        with self._lock:
            return len(self._calls)
