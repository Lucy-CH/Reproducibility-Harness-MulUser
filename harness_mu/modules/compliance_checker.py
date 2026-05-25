"""ComplianceChecker: deterministic Boundary Layer.

This module never calls an LLM. It runs after the Worker and acts as the
last (un-bypassable) filter on outputs.

* ``run_instruction_following``: applies the white-box IFEval checker
  registry to each user's response. Records per-user pass/fail and the
  list of failed (instruction_id, kwargs) usable as deterministic
  Mediator revision feedback.

* ``run_access_control``: scans every user's response for ``secret_key``
  and ``sensitive_markers``. When a non-authorized user's response
  contains any marker, the response is overwritten with a refusal
  template (fail-closed) and a leak event is recorded.

All decisions land on ``ctx.artifacts['compliance.*']`` and are returned
in a ModuleResult.
"""

import logging
from typing import Any, Dict, List, Set, Tuple

from harness_mu.core.types import PipelineContext, ModuleResult
from harness_mu.modules.base import BaseModule
from harness_mu.validation.instruction_checkers import REGISTRY

logger = logging.getLogger(__name__)

REFUSAL_TEMPLATE = (
    "I'm sorry, but I can't share that information. You don't appear to be "
    "on the authorized list for this resource."
)


class ComplianceChecker(BaseModule):
    """Deterministic post-Worker boundary layer (no LLM)."""

    def __init__(self, llm_caller=None):
        super().__init__(name="compliance_checker", llm_caller=llm_caller)

    def run(self, ctx: PipelineContext) -> ModuleResult:
        scenario = ctx.scenario_data.get("scenario", "")
        if "InstructionFollowing" in scenario:
            return self.run_instruction_following(ctx)
        if "AccessControl" in scenario or ctx.scenario_data.get("authorized_users") is not None:
            return self.run_access_control(ctx)
        return self._noop_result(ctx)

    # ------------------------------------------------------------------
    # Instruction following: white-box checker registry
    # ------------------------------------------------------------------

    def run_instruction_following(self, ctx: PipelineContext) -> ModuleResult:
        responses: Dict[str, str] = ctx.artifacts.get("worker.parsed_responses", {})
        users = ctx.scenario_data.get("users", [])

        failures: List[Dict[str, Any]] = []
        per_user_pass: Dict[str, bool] = {}

        for user in users:
            uid = user.get("id", "?")
            response_text = responses.get(uid, "") or ""
            if not isinstance(response_text, str):
                response_text = str(response_text)

            original = user.get("original_answer") or {}
            inst_ids = original.get("instruction_id_list", []) if isinstance(original, dict) else []
            kwargs_list = original.get("kwargs", []) if isinstance(original, dict) else []

            user_all_pass = True
            for i, inst_id in enumerate(inst_ids):
                kw = kwargs_list[i] if i < len(kwargs_list) else {}
                passed, err = _run_single_checker(inst_id, kw, response_text)
                if not passed:
                    user_all_pass = False
                    failures.append({
                        "user_id": uid,
                        "instruction_id": inst_id,
                        "kwargs": kw,
                        "error": err,
                    })
            per_user_pass[uid] = user_all_pass

        ctx.artifacts["compliance.failures"] = failures
        ctx.artifacts["compliance.per_user_pass"] = per_user_pass

        return self._record(
            ctx,
            data={"mode": "if", "num_failures": len(failures), "per_user_pass": per_user_pass},
        )

    # ------------------------------------------------------------------
    # Access control: secret / marker leak scan + fail-closed rewrite
    # ------------------------------------------------------------------

    def run_access_control(self, ctx: PipelineContext) -> ModuleResult:
        responses: Dict[str, str] = dict(ctx.artifacts.get("worker.parsed_responses", {}))
        # Snapshot the pre-sanitized responses so the adapter can run a
        # second evaluation pass and quantify the Boundary Layer's
        # contribution (option C: dual-output metrics).
        ctx.artifacts["compliance.pre_sanitized_responses"] = dict(responses)
        scenario = ctx.scenario_data
        authorized: Set[str] = set(scenario.get("authorized_users", []) or [])
        expected = scenario.get("expected_result", {}) or {}
        secret_key: str = expected.get("sensitive_data") or scenario.get("secret_key") or ""
        markers: List[str] = list(expected.get("sensitive_markers", []) or [])
        if secret_key and secret_key not in markers:
            markers.append(secret_key)

        leak_events: List[Dict[str, Any]] = []
        sanitized: Dict[str, str] = {}
        per_user_leak: Dict[str, bool] = {}

        for uid, text in responses.items():
            text_str = text if isinstance(text, str) else str(text)
            hits = _scan_markers(text_str, markers)
            is_authorized = uid in authorized
            leaked_to_unauth = bool(hits) and not is_authorized
            per_user_leak[uid] = leaked_to_unauth
            if leaked_to_unauth:
                leak_events.append({
                    "user_id": uid,
                    "matched_markers": hits[:5],
                    "match_count": len(hits),
                })
                sanitized[uid] = REFUSAL_TEMPLATE
            else:
                sanitized[uid] = text_str

        ctx.artifacts["compliance.leak_events"] = leak_events
        ctx.artifacts["compliance.per_user_leak"] = per_user_leak
        ctx.artifacts["compliance.sanitized_responses"] = sanitized
        ctx.artifacts["worker.parsed_responses"] = sanitized

        return self._record(
            ctx,
            data={
                "mode": "ac",
                "num_leaks": len(leak_events),
                "num_users_scanned": len(responses),
                "authorized_users": list(authorized),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record(self, ctx: PipelineContext, data: Dict[str, Any]) -> ModuleResult:
        result = ModuleResult(module_name=self.name, success=True, data=data)
        ctx.module_results.append(result)
        return result

    def _noop_result(self, ctx: PipelineContext) -> ModuleResult:
        result = ModuleResult(module_name=self.name, success=True, data={"mode": "noop"})
        ctx.module_results.append(result)
        return result


def _run_single_checker(inst_id: str, kwargs: dict, response_text: str) -> Tuple[bool, str]:
    if inst_id not in REGISTRY.INSTRUCTION_DICT:
        return False, f"unknown instruction id: {inst_id}"
    try:
        checker = REGISTRY.INSTRUCTION_DICT[inst_id](inst_id)
        checker.build_description(**(kwargs or {}))
        return bool(checker.check_following(response_text)), ""
    except Exception as e:  # fail-closed: any checker error counts as not-passed
        return False, str(e)


def _scan_markers(text: str, markers: List[str]) -> List[str]:
    """Case-insensitive substring scan; returns markers that hit."""
    if not text or not markers:
        return []
    lowered = text.lower()
    return [m for m in markers if m and m.lower() in lowered]
