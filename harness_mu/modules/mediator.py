"""Mediator: Dual-pass task planning and quality review via LLM.

Pass 1 — Guidance Planning:
  Analyzes Gatekeeper's permission output and all user instructions,
  detects conflicts, and produces per-user guidance for the Worker.

Pass 2 — Quality Review:
  Reviews Worker output against the guidance plan, either approves
  or provides specific feedback for revision.
"""

import json
import logging
from typing import List

from harness_mu.core.types import PipelineContext, ModuleResult
from harness_mu.modules.base import BaseModule
from harness_mu.prompts.prompts import MEDIATOR_PASS1_SYSTEM, MEDIATOR_PASS2_SYSTEM

logger = logging.getLogger(__name__)


class Mediator(BaseModule):
    """Dual-pass conflict resolution and quality assurance via LLM."""

    def __init__(self, llm_caller):
        super().__init__(name="mediator", llm_caller=llm_caller)

    def run(self, ctx: PipelineContext) -> ModuleResult:
        """Convenience: run both passes. For detailed control, use run_pass1/2."""
        self.run_pass1(ctx)
        return self.run_pass2(ctx)

    # ------------------------------------------------------------------
    # Pass 1 — Guidance Planning
    # ------------------------------------------------------------------

    def run_pass1(self, ctx: PipelineContext) -> ModuleResult:
        """Analyze user requirements and produce a guidance plan.

        Input:  ctx.artifacts["gatekeeper.permissions"], ctx.users
        Output: ctx.artifacts["mediator.guidance_plan"]
        """
        users = ctx.users  # Already filtered by Gatekeeper
        permissions = ctx.artifacts.get("gatekeeper.permissions", [])
        meta = ctx.scenario_data.get("meta", {})
        scenario_type = ctx.scenario_data.get("scenario", "")

        user_prompt = self._build_pass1_prompt(users, permissions, meta, scenario_type)
        messages = [
            {"role": "system", "content": MEDIATOR_PASS1_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        raw_output, call_metrics = self.llm_caller.call(messages=messages, temperature=0.0)
        plan = self._parse_output(raw_output)

        ctx.artifacts["mediator.guidance_plan"] = plan

        result = ModuleResult(
            module_name=self.name,
            success=True,
            data={
                "pass": 1,
                "mode": plan.get("mode", "unknown"),
                "conflicts_found": len(plan.get("conflicts_found", [])),
                "guidance": plan.get("guidance", []),
                "resolution_note": plan.get("resolution_note", ""),
            },
            latency_ms=call_metrics.latency_ms,
            token_usage={
                "prompt_tokens": call_metrics.prompt_tokens,
                "completion_tokens": call_metrics.completion_tokens,
            } if call_metrics.prompt_tokens or call_metrics.completion_tokens else None,
        )
        ctx.module_results.append(result)
        return result

    # ------------------------------------------------------------------
    # Pass 2 — Quality Review
    # ------------------------------------------------------------------

    def run_pass2(self, ctx: PipelineContext) -> ModuleResult:
        """Review Worker output against guidance plan.

        Input:  ctx.artifacts["worker.parsed_responses"],
                ctx.artifacts["mediator.guidance_plan"]
        Output: ctx.artifacts["mediator.review"]
        """
        worker_output = ctx.artifacts.get("worker.parsed_responses", {})
        guidance_plan = ctx.artifacts.get("mediator.guidance_plan", {})
        users = ctx.users

        user_prompt = self._build_pass2_prompt(users, guidance_plan, worker_output)
        messages = [
            {"role": "system", "content": MEDIATOR_PASS2_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        raw_output, call_metrics = self.llm_caller.call(messages=messages, temperature=0.0)
        review = self._parse_output(raw_output)

        ctx.artifacts["mediator.review"] = review

        result = ModuleResult(
            module_name=self.name,
            success=True,
            data={
                "pass": 2,
                "approved": review.get("approved", False),
                "feedback": review.get("feedback", []),
                "summary": review.get("summary", ""),
            },
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
    def _build_pass1_prompt(
        users: List[dict],
        permissions: List[dict],
        meta: dict,
        scenario_type: str,
    ) -> str:
        lines = [
            "Plan the execution for the following multi-user scenario.",
            f"Scenario type: {scenario_type}",
            f"Mode: {meta.get('mode', 'aligned')}",
            "",
            "--- AUTHORITY LEVELS (from Gatekeeper) ---",
        ]
        perm_by_user = {p.get("user_id"): p for p in permissions}
        for user in users:
            uid = user.get("id", "?")
            perm = perm_by_user.get(uid, {})
            lines.append(
                f"  {uid}: role={perm.get('role', '?')}, "
                f"authority={perm.get('authority', '?')}"
            )

        lines.append("")
        lines.append("--- USER INSTRUCTIONS ---")
        for user in users:
            uid = user.get("id", "?")
            instructions = user.get("instructions", [])
            oa = user.get("original_answer", {})
            inst_ids = oa.get("instruction_id_list", []) if isinstance(oa, dict) else []
            kwargs_list = oa.get("kwargs", []) if isinstance(oa, dict) else []

            lines.append(f"\nUser: {uid}")
            for instr in instructions:
                lines.append(f"  Instruction: {instr[:300]}")
            if inst_ids:
                constraints_desc = []
                for i, iid in enumerate(inst_ids):
                    kw = kwargs_list[i] if i < len(kwargs_list) else {}
                    constraints_desc.append(f"{iid}({json.dumps(kw)})")
                lines.append(f"  Constraints: {', '.join(constraints_desc)}")

        lines.append("")
        lines.append("--- OUTPUT ---")
        lines.append("Produce the guidance plan JSON as specified in your system prompt.")
        return "\n".join(lines)

    @staticmethod
    def _build_pass2_prompt(
        users: List[dict],
        guidance_plan: dict,
        worker_output: dict,
    ) -> str:
        lines = [
            "Review the following Worker output against the guidance plan.",
            "",
            "--- GUIDANCE PLAN ---",
        ]
        for g in guidance_plan.get("guidance", []):
            lines.append(
                f"  {g.get('user_id', '?')}: "
                f"priority={g.get('priority')}, action={g.get('action')}"
            )
            for c in g.get("constraints", []):
                lines.append(f"    Constraint: {c}")
            if g.get("note"):
                lines.append(f"    Note: {g['note']}")

        lines.append("")
        lines.append("--- WORKER OUTPUT ---")
        for uid, response in worker_output.items():
            lines.append(f"\n  User: {uid}")
            lines.append(f"  Response: {response[:500]}")

        lines.append("")
        lines.append("--- ORIGINAL USER INSTRUCTIONS ---")
        for user in users:
            uid = user.get("id", "?")
            oa = user.get("original_answer", {})
            inst_ids = oa.get("instruction_id_list", []) if isinstance(oa, dict) else []
            kwargs_list = oa.get("kwargs", []) if isinstance(oa, dict) else []
            lines.append(f"\n  {uid}:")
            for i, iid in enumerate(inst_ids):
                kw = kwargs_list[i] if i < len(kwargs_list) else {}
                lines.append(f"    {iid}: {json.dumps(kw)}")

        lines.append("")
        lines.append("--- OUTPUT ---")
        lines.append("Review and output approved/feedback JSON as specified.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_output(raw_output: str) -> dict:
        clean = raw_output.strip()
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
            logger.warning("[mediator] failed to parse JSON; raw: %s...", raw_output[:100])
            return {}
