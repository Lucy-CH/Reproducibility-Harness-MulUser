"""Worker: Pure task execution driven by Mediator guidance.

Receives a guidance plan from the Mediator and user instructions,
generates responses that satisfy all specified constraints.
"""

import json
import logging

from harness_mu.core.types import PipelineContext, ModuleResult
from harness_mu.modules.base import BaseModule
from harness_mu.prompts.prompts import WORKER_SYSTEM_BASE, WORKER_IF_FORMAT_INSTRUCTIONS

logger = logging.getLogger(__name__)


class Worker(BaseModule):
    """Task execution module driven by Mediator guidance."""

    def __init__(self, llm_caller):
        super().__init__(name="worker", llm_caller=llm_caller)

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
