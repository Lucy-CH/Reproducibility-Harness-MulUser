"""Mediator: LLM-based protocol planner (Mediation Layer).

Single-pass planner. Pass-2 LLM-as-judge is deliberately removed —
output validation is deterministic and lives in the Boundary Layer
(ComplianceChecker). The Mediator's job is to maximize utility under
hard constraints, not to enforce them.

* ``run_if_conflict``: produces a per-user guidance plan for the IF
  conflict mode. Authority comes straight from the ground-truth
  ``user.authority`` field; the LLM is asked only to phrase the
  resolution rationale and per-user constraint summaries.

* ``run_ac_isolation``: produces an access/isolation plan for AC.
  The deterministic ``authorized_users`` list seeds the plan; the LLM
  proposes per-user acknowledgement messages so non-authorized users
  still receive a useful, polite response.

* ``build_revision_feedback``: pure function. Converts deterministic
  ComplianceChecker failures into a structured Worker revision prompt.
  No LLM call.
"""

import json
import logging
from typing import Any, Dict, List

from harness_mu.core.types import PipelineContext, ModuleResult
from harness_mu.modules.base import BaseModule
from harness_mu.prompts.prompts import (
    MEDIATOR_IF_CONFLICT_SYSTEM,
    MEDIATOR_AC_ISOLATION_SYSTEM,
)

logger = logging.getLogger(__name__)


class Mediator(BaseModule):
    """Single-pass LLM planner. Validation is delegated to Boundary Layer."""

    def __init__(self, llm_caller):
        super().__init__(name="mediator", llm_caller=llm_caller)

    def run(self, ctx: PipelineContext) -> ModuleResult:
        """Default dispatch by scenario; adapters usually call run_* directly."""
        scenario = ctx.scenario_data.get("scenario", "")
        if "AccessControl" in scenario or ctx.scenario_data.get("authorized_users") is not None:
            return self.run_ac_isolation(ctx)
        return self.run_if_conflict(ctx)

    # ------------------------------------------------------------------
    # IF conflict-mode planning (LLM)
    # ------------------------------------------------------------------

    def run_if_conflict(self, ctx: PipelineContext) -> ModuleResult:
        """Build a per-user guidance plan for IF conflict mode.

        Authority is taken verbatim from ground-truth ``user.authority``;
        the LLM only generates the resolution rationale and per-user
        constraint paraphrases. Highest-authority users always get
        ``priority='highest'``; others ``'deferred'``.
        """
        users = ctx.users or ctx.scenario_data.get("users", [])
        max_auth = max((float(u.get("authority", 0)) for u in users), default=0.0)

        seed = []
        for u in users:
            auth = float(u.get("authority", 0))
            seed.append({
                "user_id": u.get("id", "?"),
                "authority": auth,
                "priority": "highest" if auth >= max_auth else "deferred",
            })
        ctx.artifacts["mediator.priority_seed"] = seed

        user_prompt = self._build_if_conflict_prompt(users, seed)
        messages = [
            {"role": "system", "content": MEDIATOR_IF_CONFLICT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        raw_output, call_metrics = self.llm_caller.call(messages=messages, temperature=0.0)
        plan = _parse_json_strict(raw_output)

        # Re-anchor authority/priority from the deterministic seed; never
        # let the LLM downgrade a highest-authority user.
        seed_by_id = {s["user_id"]: s for s in seed}
        for g in plan.get("guidance", []):
            uid = g.get("user_id", "")
            if uid in seed_by_id:
                g["priority"] = seed_by_id[uid]["priority"]
                g["authority"] = seed_by_id[uid]["authority"]

        ctx.artifacts["mediator.guidance_plan"] = plan
        return self._record(
            ctx,
            data={
                "stage": "if_conflict",
                "guidance": plan.get("guidance", []),
                "resolution_note": plan.get("resolution_note", ""),
            },
            call_metrics=call_metrics,
        )

    # ------------------------------------------------------------------
    # AC isolation planning (LLM)
    # ------------------------------------------------------------------

    def run_ac_isolation(self, ctx: PipelineContext) -> ModuleResult:
        """Build per-user access plan for AC.

        ``isolated`` is a deterministic function of ``authorized_users``;
        the LLM proposes the per-user response posture (grant / refuse-but-help).
        """
        users = ctx.users or ctx.scenario_data.get("all_users", [])
        authorized = set(ctx.scenario_data.get("authorized_users", []) or [])
        resource_name = ctx.scenario_data.get("resource_name", "the requested resource")

        seed = [
            {
                "user_id": u.get("id", "?"),
                "role": u.get("role", ""),
                "is_authorized": u.get("id", "") in authorized,
                "isolated": u.get("id", "") not in authorized,
            }
            for u in users
        ]
        ctx.artifacts["mediator.access_seed"] = seed

        user_prompt = self._build_ac_isolation_prompt(users, seed, resource_name)
        messages = [
            {"role": "system", "content": MEDIATOR_AC_ISOLATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        raw_output, call_metrics = self.llm_caller.call(messages=messages, temperature=0.0)
        plan = _parse_json_strict(raw_output)

        # Re-anchor isolation flags from seed; never let LLM grant a non-authorized user.
        seed_by_id = {s["user_id"]: s for s in seed}
        for g in plan.get("guidance", []):
            uid = g.get("user_id", "")
            if uid in seed_by_id:
                g["isolated"] = seed_by_id[uid]["isolated"]
                g["is_authorized"] = seed_by_id[uid]["is_authorized"]
                if seed_by_id[uid]["isolated"]:
                    g["action"] = "refuse_but_help"

        ctx.artifacts["mediator.guidance_plan"] = plan
        return self._record(
            ctx,
            data={
                "stage": "ac_isolation",
                "guidance": plan.get("guidance", []),
                "authorized_users": list(authorized),
            },
            call_metrics=call_metrics,
        )

    # ------------------------------------------------------------------
    # Deterministic revision feedback (no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def build_revision_feedback(failures: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert ComplianceChecker failures into Worker-readable feedback.

        Each entry has user_id, issue, suggestion. Feedback is mechanical
        (paraphrases the failed checker + its kwargs) — no LLM judgement.
        """
        feedback: List[Dict[str, str]] = []
        for f in failures:
            inst_id = f.get("instruction_id", "?")
            kw = f.get("kwargs", {}) or {}
            issue = f"Failed white-box checker '{inst_id}' with kwargs {json.dumps(kw, ensure_ascii=False)}."
            err = f.get("error")
            if err:
                issue += f" Detail: {err}"
            feedback.append({
                "user_id": f.get("user_id", "?"),
                "issue": issue,
                "suggestion": _suggestion_for(inst_id, kw),
            })
        return feedback

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record(self, ctx: PipelineContext, data: Dict[str, Any], call_metrics) -> ModuleResult:
        result = ModuleResult(
            module_name=self.name,
            success=True,
            data=data,
            latency_ms=call_metrics.latency_ms,
            token_usage={
                "prompt_tokens": call_metrics.prompt_tokens,
                "completion_tokens": call_metrics.completion_tokens,
            } if call_metrics.prompt_tokens or call_metrics.completion_tokens else None,
        )
        ctx.module_results.append(result)
        return result

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_if_conflict_prompt(users: List[dict], seed: List[Dict[str, Any]]) -> str:
        lines = [
            "Resolve a conflict-mode multi-user instruction-following task.",
            "Authority and priority are FIXED below; do not change them.",
            "",
            "--- AUTHORITY (ground truth) ---",
        ]
        for s in seed:
            lines.append(f"  {s['user_id']}: authority={s['authority']}, priority={s['priority']}")

        lines.append("")
        lines.append("--- USER INSTRUCTIONS ---")
        for u in users:
            uid = u.get("id", "?")
            lines.append(f"\nUser: {uid}")
            for instr in u.get("instructions", []):
                lines.append(f"  Instruction: {instr[:400]}")
            oa = u.get("original_answer") or {}
            inst_ids = oa.get("instruction_id_list", []) if isinstance(oa, dict) else []
            kwargs_list = oa.get("kwargs", []) if isinstance(oa, dict) else []
            if inst_ids:
                parts = []
                for i, iid in enumerate(inst_ids):
                    kw = kwargs_list[i] if i < len(kwargs_list) else {}
                    parts.append(f"{iid}({json.dumps(kw, ensure_ascii=False)})")
                lines.append(f"  Constraints: {', '.join(parts)}")

        lines.append("")
        lines.append("--- TASK ---")
        lines.append("Produce the JSON guidance plan as specified in the system prompt.")
        return "\n".join(lines)

    @staticmethod
    def _build_ac_isolation_prompt(users: List[dict], seed: List[Dict[str, Any]], resource_name: str) -> str:
        lines = [
            "Plan per-user responses for an access-control scenario.",
            f"Resource: {resource_name}",
            "Authorization is FIXED below; do not grant access to non-authorized users.",
            "",
            "--- USERS ---",
        ]
        seed_by_id = {s["user_id"]: s for s in seed}
        for u in users:
            uid = u.get("id", "?")
            s = seed_by_id.get(uid, {})
            lines.append(
                f"  {uid}: role={u.get('role', '')}, "
                f"is_authorized={s.get('is_authorized', False)}, isolated={s.get('isolated', True)}"
            )

        lines.append("")
        lines.append("--- TASK ---")
        lines.append("Produce the JSON access plan as specified in the system prompt.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_json_strict(raw_output: str) -> dict:
    """Strip code fences and parse JSON; return {} on failure."""
    clean = (raw_output or "").strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("[mediator] failed to parse JSON; raw: %s...", (raw_output or "")[:120])
        return {}


def _suggestion_for(inst_id: str, kwargs: dict) -> str:
    """Mechanical, checker-aware paraphrase of how to fix a failed constraint."""
    kw = kwargs or {}
    if inst_id == "startend:end_checker":
        phrase = kw.get("end_phrase", "")
        return f"Make the response end with exactly: {phrase!r} and nothing after it."
    if inst_id == "startend:start_checker":
        phrase = kw.get("start_phrase", "")
        return f"Make the response start with exactly: {phrase!r}."
    if inst_id == "change_case:capital_word_frequency":
        rel = kw.get("capital_relation", "")
        n = kw.get("capital_frequency", "")
        return f"Use ALL-CAPS words {rel} {n} times in total."
    if inst_id == "keywords:frequency":
        kw_word = kw.get("keyword", "")
        rel = kw.get("relation", "")
        n = kw.get("frequency", "")
        return f"Use the word {kw_word!r} {rel} {n} times."
    if inst_id == "length_constraints:number_words":
        rel = kw.get("relation", "")
        n = kw.get("num_words", "")
        return f"Make the response {rel} {n} words in length."
    if inst_id == "length_constraints:number_sentences":
        rel = kw.get("relation", "")
        n = kw.get("num_sentences", "")
        return f"Use {rel} {n} sentences in total."
    return f"Re-read constraint {inst_id} (kwargs={json.dumps(kw, ensure_ascii=False)}) and rewrite the response to satisfy it."
