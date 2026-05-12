"""Instruction-Following scenario adapter (Skill Layer).

Flow:
    parse_scenario() -> PipelineContext
    [conflict mode only] Mediator.run_if_conflict()   (LLM, single call)
    Worker.run_per_user()                             (LLM, parallel per user)
    ComplianceChecker.run_instruction_following()     (deterministic)
    [if any failure] Worker.run_per_user(only_user_ids=failed)  (one revision)
    _evaluate()                                       (all users, all checkers)

The Mediator is skipped entirely in aligned mode (no conflict to resolve).
The Gatekeeper is not used: authority levels are ground-truth fields.
"""

import json
import logging
from typing import Dict, List, Set

from harness_mu.adapters.base import ScenarioAdapter
from harness_mu.config.schema import HarnessMUConfig
from harness_mu.core.types import PipelineContext, ScenarioRecord
from harness_mu.llm.factory import LLMFactory
from harness_mu.metrics.collector import MetricsCollector
from harness_mu.modules.mediator import Mediator
from harness_mu.modules.worker import Worker
from harness_mu.modules.compliance_checker import ComplianceChecker
from harness_mu.validation.instruction_checkers import REGISTRY

logger = logging.getLogger(__name__)

MAX_REVISIONS = 1  # White-box-driven Worker revision rounds


class InstructionFollowingAdapter(ScenarioAdapter):
    """Adapter for the Muses-Bench instruction_following scenario."""

    def __init__(
        self,
        config: HarnessMUConfig,
        llm_factory: LLMFactory,
        metrics_collector: MetricsCollector,
    ):
        self._config = config
        self._llm_factory = llm_factory
        self._metrics_collector = metrics_collector
        self._use_training_format = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_scenario(self, raw_line: str) -> PipelineContext:
        data = json.loads(raw_line)
        return PipelineContext(
            scenario_id=data.get("id", ""),
            scenario_data=data,
            users=list(data.get("users", [])),
        )

    def run(self, raw_line: str) -> ScenarioRecord:
        ctx = self.parse_scenario(raw_line)
        scenario_cfg = self._config.scenarios.get("instruction_following")
        if scenario_cfg is None:
            raise ValueError("No 'instruction_following' scenario in config")

        activation = scenario_cfg.modules
        meta = ctx.scenario_data.get("meta", {}) or {}
        mode = meta.get("mode", "aligned")

        # --- Step 1: Mediator (conflict mode only) ---
        mediator = Mediator(self._llm_factory.get_caller("mediator"))
        if mode == "conflict" and mediator.is_enabled(activation):
            m_result = mediator.run_if_conflict(ctx)
            logger.debug(
                "[%s] mediator: guidance=%d, note=%s",
                ctx.scenario_id,
                len(m_result.data.get("guidance", [])),
                m_result.data.get("resolution_note", "")[:80],
            )

        # --- Step 2: Worker (per-user, parallel) ---
        worker = Worker(
            self._llm_factory.get_caller("worker"),
            max_workers=self._config.execution.max_workers,
        )
        worker.run_per_user(ctx)

        # --- Step 3: ComplianceChecker (deterministic, fail-closed) ---
        compliance = ComplianceChecker()
        compliance.run_instruction_following(ctx)

        # --- Step 4: Revision loop (white-box driven) ---
        for revision in range(MAX_REVISIONS):
            failures = ctx.artifacts.get("compliance.failures", []) or []
            if not failures:
                break
            failed_user_ids: Set[str] = {f.get("user_id", "") for f in failures}
            logger.info(
                "[%s] revision %d: %d failed users -> %s",
                ctx.scenario_id, revision + 1, len(failed_user_ids), sorted(failed_user_ids),
            )
            ctx.artifacts["mediator.revision_feedback"] = (
                Mediator.build_revision_feedback(failures)
            )
            ctx.artifacts["mediator.revision_count"] = revision + 1
            worker.run_per_user(ctx, only_user_ids=list(failed_user_ids))
            compliance.run_instruction_following(ctx)

        # --- Step 5: Evaluate ---
        instruction_results, metrics = self._evaluate(ctx)

        record = ScenarioRecord(
            scenario_id=ctx.scenario_id,
            module_results=list(ctx.module_results),
            instruction_results=instruction_results,
            metrics=metrics,
            raw_outputs={
                "worker_parsed": json.dumps(
                    ctx.artifacts.get("worker.parsed_responses", {}),
                    ensure_ascii=False,
                ),
            },
        )
        self._metrics_collector.record_scenario_metrics(ctx.scenario_id, metrics)
        return record

    # ------------------------------------------------------------------
    # Evaluation (unchanged — white-box checkers are post-hoc verification)
    # ------------------------------------------------------------------

    def _evaluate(self, ctx: PipelineContext):
        users = ctx.scenario_data.get("users", [])
        meta = ctx.scenario_data.get("meta", {})
        mode = meta.get("mode", "aligned")
        user_responses = ctx.artifacts.get("worker.parsed_responses", {})

        max_auth = -1
        if mode == "conflict":
            for u in users:
                auth = float(u.get("authority", 0))
                if auth > max_auth:
                    max_auth = auth

        instruction_results = []
        valid_count = 0
        total_count = 0
        weighted_valid_count = 0
        weighted_total_count = 0

        for user in users:
            user_id = user.get("id", "?")
            user_authority = float(user.get("authority", 1.0))

            is_eval_target = True
            if mode == "conflict" and user_authority < max_auth:
                is_eval_target = False

            response_val = user_responses.get(user_id, "")
            if not isinstance(response_val, str):
                response_text = json.dumps(response_val, ensure_ascii=False)
            else:
                response_text = response_val

            original_ans = user.get("original_answer", {})
            if not isinstance(original_ans, dict):
                continue

            inst_ids = original_ans.get("instruction_id_list", [])
            kwargs_list = original_ans.get("kwargs", [])

            for i, inst_id in enumerate(inst_ids):
                if is_eval_target:
                    total_count += 1
                    weighted_total_count += user_authority

                kw = kwargs_list[i] if i < len(kwargs_list) else {}
                passed = False
                error_msg = ""

                if inst_id in REGISTRY.INSTRUCTION_DICT:
                    try:
                        checker_cls = REGISTRY.INSTRUCTION_DICT[inst_id]
                        checker = checker_cls(inst_id)
                        checker.build_description(**kw)
                        passed = checker.check_following(response_text)
                    except Exception as e:
                        error_msg = str(e)
                        logger.debug(
                            "[%s] checker error for %s/%s: %s",
                            ctx.scenario_id, user_id, inst_id, e,
                        )
                else:
                    error_msg = f"Unknown instruction ID: {inst_id}"

                if passed and is_eval_target:
                    valid_count += 1
                    weighted_valid_count += user_authority

                instruction_results.append({
                    "user_id": user_id,
                    "authority": user_authority,
                    "instruction_id": inst_id,
                    "kwargs": kw,
                    "passed": passed,
                    "is_eval_target": is_eval_target,
                    "error": error_msg,
                })

        accuracy = valid_count / total_count if total_count > 0 else 0.0
        weighted_accuracy = (
            weighted_valid_count / weighted_total_count
            if weighted_total_count > 0
            else 0.0
        )

        metrics = {
            "accuracy": accuracy,
            "weighted_accuracy": weighted_accuracy,
            "num_instructions": total_count,
            "num_followed": valid_count,
            "all_passed": (valid_count == total_count) and (total_count > 0),
        }

        from harness_mu.metrics.custom.instruction_following import (
            evaluate_authority_compliance,
            evaluate_per_user_satisfaction,
        )

        authority_compliance = evaluate_authority_compliance(
            instruction_results, users, mode
        )
        per_user = evaluate_per_user_satisfaction(instruction_results)

        metrics["_harness_mu"] = {
            "authority_compliance_score": authority_compliance,
            "per_user_satisfaction": per_user,
        }

        return instruction_results, metrics
