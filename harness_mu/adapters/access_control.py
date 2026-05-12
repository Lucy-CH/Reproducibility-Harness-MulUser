"""Access-Control scenario adapter (Skill Layer).

Flow:
    parse_scenario() -> PipelineContext
    Gatekeeper.run()              (deterministic, no LLM by default)
    Mediator.run_ac_isolation()   (LLM, single call)
    Worker.run_per_user()         (LLM, parallel; isolated system prompts)
    ComplianceChecker.run_access_control()  (deterministic, fail-closed)
    _evaluate()                   (privacy / utility / overall)
"""

import json
import logging

from harness_mu.adapters.base import ScenarioAdapter
from harness_mu.config.schema import HarnessMUConfig
from harness_mu.core.types import PipelineContext, ScenarioRecord
from harness_mu.llm.factory import LLMFactory
from harness_mu.metrics.collector import MetricsCollector
from harness_mu.metrics.custom.access_control import evaluate_access_control
from harness_mu.modules.compliance_checker import ComplianceChecker
from harness_mu.modules.gatekeeper import Gatekeeper
from harness_mu.modules.mediator import Mediator
from harness_mu.modules.worker import Worker

logger = logging.getLogger(__name__)


class AccessControlAdapter(ScenarioAdapter):
    """Adapter for the Muses-Bench access_control scenario."""

    def __init__(
        self,
        config: HarnessMUConfig,
        llm_factory: LLMFactory,
        metrics_collector: MetricsCollector,
    ):
        self._config = config
        self._llm_factory = llm_factory
        self._metrics_collector = metrics_collector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_scenario(self, raw_line: str) -> PipelineContext:
        data = json.loads(raw_line)
        users = list(data.get("all_users", []) or data.get("users", []))
        initial = data.get("initial_messages", {}) or {}
        # Each user must have an "instructions" field for the Worker. The
        # actual conversational message is ``initial_messages[uid]``; the
        # ``goal`` field is meta-prompting and is intentionally not used.
        for u in users:
            uid = u.get("id", "")
            if "instructions" not in u or not u["instructions"]:
                msg = initial.get(uid) or ""
                u["instructions"] = [msg] if msg else []
        return PipelineContext(
            scenario_id=data.get("id", ""),
            scenario_data=data,
            users=users,
        )

    def run(self, raw_line: str) -> ScenarioRecord:
        ctx = self.parse_scenario(raw_line)
        scenario_cfg = self._config.scenarios.get("access_control")
        if scenario_cfg is None:
            raise ValueError("No 'access_control' scenario in config")

        activation = scenario_cfg.modules

        # --- Step 1: Gatekeeper (deterministic admission control) ---
        gatekeeper = Gatekeeper(self._llm_factory.get_caller("gatekeeper"))
        if gatekeeper.is_enabled(activation):
            gatekeeper.run(ctx)

        # --- Step 2: Mediator (single-pass isolation planning) ---
        mediator = Mediator(self._llm_factory.get_caller("mediator"))
        if mediator.is_enabled(activation):
            mediator.run_ac_isolation(ctx)

        # --- Step 3: Worker (per-user, parallel, isolated system prompts) ---
        worker = Worker(
            self._llm_factory.get_caller("worker"),
            max_workers=self._config.execution.max_workers,
        )
        worker.run_per_user(ctx)

        # --- Step 4: ComplianceChecker (Boundary Layer — fail-closed) ---
        compliance = ComplianceChecker()
        compliance.run_access_control(ctx)

        # --- Step 5: Evaluate ---
        metrics = self._evaluate(ctx)

        record = ScenarioRecord(
            scenario_id=ctx.scenario_id,
            module_results=list(ctx.module_results),
            instruction_results=[],
            metrics=metrics,
            raw_outputs={
                "worker_parsed": json.dumps(
                    ctx.artifacts.get("worker.parsed_responses", {}),
                    ensure_ascii=False,
                ),
                "compliance_leak_events": json.dumps(
                    ctx.artifacts.get("compliance.leak_events", []),
                    ensure_ascii=False,
                ),
            },
        )
        self._metrics_collector.record_scenario_metrics(ctx.scenario_id, metrics)
        return record

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate(ctx: PipelineContext) -> dict:
        scenario = ctx.scenario_data
        responses = ctx.artifacts.get("worker.parsed_responses", {}) or {}
        all_user_ids = [u.get("id", "?") for u in (scenario.get("all_users") or scenario.get("users") or [])]
        authorized = list(scenario.get("authorized_users", []) or [])
        expected = scenario.get("expected_result", {}) or {}
        secret_key = expected.get("sensitive_data") or scenario.get("secret_key", "")
        markers = list(expected.get("sensitive_markers", []) or [])

        result = evaluate_access_control(
            responses_by_user=responses,
            authorized_users=authorized,
            all_user_ids=all_user_ids,
            sensitive_markers=markers,
            secret_key=secret_key,
        )

        leak_events = ctx.artifacts.get("compliance.leak_events", []) or []
        result["num_boundary_intercepts"] = len(leak_events)
        result["all_passed"] = (
            result["privacy_score"] >= 1.0 and result["utility_score"] >= 1.0
        )
        return result
