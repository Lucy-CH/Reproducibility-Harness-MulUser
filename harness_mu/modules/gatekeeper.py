"""Gatekeeper: LLM-based permission analysis and malicious-user detection.

Analyzes all users' identity and intent, outputs structured authority levels
and flags potential attackers. Malicious users are excluded from downstream
processing.
"""

import json
import logging
from typing import Any, Dict, List

from harness_mu.core.types import PipelineContext, ModuleResult
from harness_mu.modules.base import BaseModule
from harness_mu.prompts.prompts import GATEKEEPER_SYSTEM

logger = logging.getLogger(__name__)


class Gatekeeper(BaseModule):
    """Pre-execution permission analysis via LLM.

    Called for every scenario. For public tasks (instruction_following),
    only extracts authority levels. For private tasks (access_control),
    also detects malicious access attempts.
    """

    def __init__(self, llm_caller):
        super().__init__(name="gatekeeper", llm_caller=llm_caller)

    def run(self, ctx: PipelineContext) -> ModuleResult:
        users = ctx.scenario_data.get("users", [])
        scenario_type = ctx.scenario_data.get("scenario", "")

        user_prompt = self._build_user_prompt(users, scenario_type)
        messages = [
            {"role": "system", "content": GATEKEEPER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        raw_output, call_metrics = self.llm_caller.call(messages=messages, temperature=0.0)
        parsed = self._parse_output(raw_output)

        ctx.artifacts["gatekeeper.permissions"] = parsed.get("permissions", [])
        ctx.artifacts["gatekeeper.malicious"] = parsed.get("malicious", [])
        ctx.artifacts["gatekeeper.summary"] = parsed.get("summary", "")

        # Exclude malicious users from downstream processing
        malicious_ids = {m.get("user_id", "") for m in parsed.get("malicious", [])}
        if malicious_ids:
            logger.info("[gatekeeper] flagged malicious users: %s", malicious_ids)
            ctx.users = [u for u in users if u.get("id") not in malicious_ids]
            ctx.artifacts["gatekeeper.excluded_users"] = list(malicious_ids)

        authority_map = {
            p.get("user_id"): p.get("authority", 1)
            for p in parsed.get("permissions", [])
        }
        ctx.artifacts["gatekeeper.authority_map"] = authority_map

        result = ModuleResult(
            module_name=self.name,
            success=True,
            data={
                "permissions": parsed.get("permissions", []),
                "malicious": parsed.get("malicious", []),
                "excluded": list(malicious_ids),
                "summary": parsed.get("summary", ""),
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
    def _build_user_prompt(users: List[dict], scenario_type: str) -> str:
        """Build the user prompt listing all users and their information."""
        lines = [
            "Analyze the following users in a multi-user scenario.",
            f"Scenario type: {scenario_type}",
            "",
            "--- USERS ---",
        ]
        for user in users:
            uid = user.get("id", "?")
            role = user.get("role", "")
            authority = user.get("authority", "unspecified")
            background = user.get("background", "")
            goal = user.get("goal", "")
            instructions = user.get("instructions", [])

            lines.append(f"\nUser ID: {uid}")
            lines.append(f"  Role: {role}")
            if authority != "unspecified":
                lines.append(f"  Stated Authority: {authority}")
            if background:
                # Truncate long backgrounds
                bg = background[:300] + "..." if len(background) > 300 else background
                lines.append(f"  Background: {bg}")
            if goal:
                lines.append(f"  Goal: {goal[:200]}")
            if instructions:
                for instr in instructions[:2]:  # First 2 instructions only
                    lines.append(f"  Instruction: {instr[:200]}")

        lines.append("")
        lines.append("--- OUTPUT ---")
        lines.append("Output the JSON with permissions and malicious flagging as specified.")
        return "\n".join(lines)

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
            logger.warning("[gatekeeper] failed to parse JSON; raw: %s...", raw_output[:100])
            return {"permissions": [], "malicious": [], "summary": "parse error"}
