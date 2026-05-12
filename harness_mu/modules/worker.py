"""Worker: Skill-Layer task execution.

Two execution modes:

* ``run_per_user`` (default for the new architecture) — spawns one isolated
  LLM call per user, in parallel. Each call sees only that user's instruction
  plus its own Mediator guidance entry. This is the primary defense against
  cross-user prompt injection and shared-context leakage.

* ``run`` (legacy single-call mode, kept for baseline comparison) — issues
  one LLM call carrying every user's instruction in the same context, and
  returns a JSON dict mapping user_id → response.

Both modes write the parsed output to ``ctx.artifacts['worker.parsed_responses']``.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from harness_mu.core.types import PipelineContext, ModuleResult
from harness_mu.modules.base import BaseModule
from harness_mu.prompts.prompts import WORKER_SYSTEM_BASE, WORKER_IF_FORMAT_INSTRUCTIONS

logger = logging.getLogger(__name__)


class Worker(BaseModule):
    """Task execution module driven by Mediator guidance."""

    def __init__(self, llm_caller, max_workers: int = 8):
        super().__init__(name="worker", llm_caller=llm_caller)
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Per-user mode (default for the new architecture)
    # ------------------------------------------------------------------

    def run_per_user(
        self,
        ctx: PipelineContext,
        only_user_ids: Optional[List[str]] = None,
    ) -> ModuleResult:
        """Run one isolated LLM call per user, in parallel.

        Args:
            ctx: pipeline context. Reads ``mediator.guidance_plan`` and
                ``mediator.revision_feedback`` (optional).
            only_user_ids: if provided, only these users are (re-)executed;
                their previous responses (if any) are overwritten. This is
                used by the IF revision loop to re-run only failing users.
        """
        users = ctx.users or ctx.scenario_data.get("users", []) or ctx.scenario_data.get("all_users", [])
        guidance_plan = ctx.artifacts.get("mediator.guidance_plan", {}) or {}
        guidance_by_id: Dict[str, Dict[str, Any]] = {
            g.get("user_id", ""): g for g in guidance_plan.get("guidance", [])
        }
        revision_feedback: List[Dict[str, str]] = (
            ctx.artifacts.get("mediator.revision_feedback") or []
        )
        feedback_by_user: Dict[str, List[Dict[str, str]]] = {}
        for fb in revision_feedback:
            feedback_by_user.setdefault(fb.get("user_id", ""), []).append(fb)

        targets = [u for u in users if (only_user_ids is None or u.get("id") in only_user_ids)]

        previous = dict(ctx.artifacts.get("worker.parsed_responses", {}) or {})
        results: Dict[str, str] = dict(previous)
        per_user_meta: Dict[str, Dict[str, Any]] = {}
        total_latency = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        scenario_prompt = (
            ctx.scenario_data.get("prompt")
            or ctx.scenario_data.get("system_prompt")
            or ""
        )

        def _run_one(user: dict):
            uid = user.get("id", "?")
            sys_prompt, user_msg = self._build_per_user_messages(
                user,
                guidance_by_id.get(uid, {}),
                feedback_by_user.get(uid, []),
                scenario_system_prompt=scenario_prompt,
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ]
            text, metrics = self.llm_caller.call(messages=messages)
            return uid, text, metrics

        if not targets:
            ctx.artifacts["worker.parsed_responses"] = results
            return self._record(ctx, success=True, data={"mode": "per_user", "num_users": 0})

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(targets))) as ex:
            futures = [ex.submit(_run_one, u) for u in targets]
            for f in as_completed(futures):
                try:
                    uid, text, metrics = f.result()
                except Exception as e:  # fail-soft per user
                    logger.warning("[worker] per-user call failed: %s", e)
                    continue
                results[uid] = text or ""
                total_latency += metrics.latency_ms
                total_prompt_tokens += metrics.prompt_tokens
                total_completion_tokens += metrics.completion_tokens
                per_user_meta[uid] = {
                    "latency_ms": metrics.latency_ms,
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                }

        ctx.artifacts["worker.parsed_responses"] = results
        ctx.artifacts["worker.per_user_meta"] = per_user_meta

        return self._record(
            ctx,
            success=len(results) > 0,
            data={
                "mode": "per_user",
                "num_users": len(targets),
                "num_total": len(results),
                "per_user_meta": per_user_meta,
                "revision_targets": list(only_user_ids or []),
            },
            latency_ms=total_latency,
            token_usage={
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            } if (total_prompt_tokens or total_completion_tokens) else None,
        )

    # ------------------------------------------------------------------
    # Legacy single-call mode (kept for baseline)
    # ------------------------------------------------------------------

    def run(self, ctx: PipelineContext, use_training_format: bool = False) -> ModuleResult:
        users = ctx.users
        guidance_plan = ctx.artifacts.get("mediator.guidance_plan", {})
        original_data = ctx.scenario_data

        # Build system prompt: base + format instructions + guidance + revision feedback
        revision_feedback = ctx.artifacts.get("mediator.revision_feedback")
        system_prompt = self._build_system_prompt(
            original_data.get("system_prompt", "You are a helpful assistant."),
            guidance_plan,
            revision_feedback=revision_feedback,
            use_training_format=use_training_format,
        )

        # Build user messages with instructions
        messages = [{"role": "system", "content": system_prompt}]

        if use_training_format:
            for user in users:
                uid = user.get("id", "User")
                instructions = user.get("instructions", [])
                if instructions:
                    messages.append({
                        "role": f"user_{uid}",
                        "content": "\n".join(instructions),
                    })
        else:
            conversation_text = ""
            for user in users:
                uid = user.get("id", "User")
                for instr in user.get("instructions", []):
                    conversation_text += f"<{uid}>{instr}</{uid}>\n"
            messages.append({"role": "user", "content": conversation_text})

        # Call Worker LLM
        raw_output, call_metrics = self.llm_caller.call(messages=messages)
        ctx.artifacts["worker.raw_output"] = raw_output

        parsed_responses = self._parse_output(raw_output, use_training_format)
        ctx.artifacts["worker.parsed_responses"] = parsed_responses

        result = ModuleResult(
            module_name=self.name,
            success=len(parsed_responses) > 0,
            data={
                "raw_output": raw_output,
                "parsed_responses": parsed_responses,
                "message_count": len(messages),
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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(
        base_prompt: str,
        guidance_plan: dict,
        revision_feedback: list = None,
        use_training_format: bool = False,
    ) -> str:
        """Build the full system prompt: base role + format + guidance + revision."""
        parts = [WORKER_SYSTEM_BASE]
        parts.append(f"\n{base_prompt}")

        # Append Mediator guidance
        guidance = guidance_plan.get("guidance", [])
        resolution_note = guidance_plan.get("resolution_note", "")
        if guidance:
            parts.append("\n--- MEDIATOR GUIDANCE ---")
            parts.append("Follow this plan when generating responses:")
            for g in guidance:
                uid = g.get("user_id", "?")
                action = g.get("action", "follow")
                priority = g.get("priority", "normal")
                constraints = g.get("constraints", [])
                note = g.get("note", "")

                parts.append(f"\n  User '{uid}': priority={priority}, action={action}")
                for c in constraints:
                    parts.append(f"    - Must satisfy: {c}")
                if note:
                    parts.append(f"    - Guidance: {note}")
            if resolution_note:
                parts.append(f"\n  Overall strategy: {resolution_note}")
            parts.append("\n--- END GUIDANCE ---")

        # Append revision feedback if this is a re-run
        if revision_feedback:
            parts.append("\n--- REVISION FEEDBACK ---")
            parts.append("Your previous output was rejected. Fix these issues:")
            for fb in revision_feedback:
                parts.append(
                    f"  - User '{fb.get('user_id', '?')}': {fb.get('issue', '')}"
                )
                suggestion = fb.get("suggestion", "")
                if suggestion:
                    parts.append(f"    Fix: {suggestion}")
            parts.append("--- END REVISION FEEDBACK ---")

        # Append format instructions
        if not use_training_format:
            parts.append(f"\n{WORKER_IF_FORMAT_INSTRUCTIONS}")

        return "\n".join(parts)

    @staticmethod
    def _parse_output(raw_output: str, use_training_format: bool) -> dict:
        if use_training_format:
            from harness_mu.utils.format_utils import parse_training_format_response
            return parse_training_format_response(raw_output)

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
            logger.warning("[worker] failed to parse JSON; raw: %s...", raw_output[:100])
            return {}

    # ------------------------------------------------------------------
    # Per-user mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_per_user_messages(
        user: dict,
        guidance: Dict[str, Any],
        revision_feedback: List[Dict[str, str]],
        scenario_system_prompt: str = "",
    ) -> tuple:
        """Build (system_prompt, user_message) for a single isolated user call.

        The system prompt only mentions THIS user. The user message contains
        only THIS user's instruction text.
        """
        uid = user.get("id", "User")
        role = user.get("role", "")

        sys_parts = [WORKER_SYSTEM_BASE]
        if scenario_system_prompt:
            sys_parts.append("\n--- SCENARIO SYSTEM PROMPT ---")
            sys_parts.append(scenario_system_prompt)
            sys_parts.append("--- END SCENARIO SYSTEM PROMPT ---")
        sys_parts.append(f"\nYou are responding to user: {uid}")
        if role:
            sys_parts.append(f"User role: {role}")

        action = guidance.get("action", "follow")
        priority = guidance.get("priority", "normal")
        constraints = guidance.get("constraints", []) or []
        note = guidance.get("note", "")

        sys_parts.append(f"\n--- MEDIATOR GUIDANCE FOR {uid} ---")
        sys_parts.append(f"action: {action}")
        sys_parts.append(f"priority: {priority}")
        for c in constraints:
            sys_parts.append(f"  - Must satisfy: {c}")
        if note:
            sys_parts.append(f"  - Note: {note}")
        sys_parts.append("--- END GUIDANCE ---")

        if revision_feedback:
            sys_parts.append("\n--- REVISION FEEDBACK (your previous response failed these checks) ---")
            for fb in revision_feedback:
                sys_parts.append(f"  - Issue: {fb.get('issue', '')}")
                sug = fb.get("suggestion", "")
                if sug:
                    sys_parts.append(f"    Fix: {sug}")
            sys_parts.append("--- END REVISION FEEDBACK ---")

        instructions = user.get("instructions", []) or []
        user_msg = "\n\n".join(instructions) if instructions else ""

        return "\n".join(sys_parts), user_msg

    def _record(
        self,
        ctx: PipelineContext,
        success: bool,
        data: Dict[str, Any],
        latency_ms: float = 0.0,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> ModuleResult:
        result = ModuleResult(
            module_name=self.name,
            success=success,
            data=data,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
        ctx.module_results.append(result)
        return result
