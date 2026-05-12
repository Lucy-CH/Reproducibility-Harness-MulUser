"""Gatekeeper: deterministic Boundary-Layer admission control for AC.

This module is AC-only and does NOT call an LLM by default. Its job is to
turn the ground-truth ``authorized_users`` list and per-user metadata
(role, attack_type) into a structured permission map and a malicious-user
flag list.

* ``run`` (default): white-box, no LLM call. Authorized iff the user's id
  is in ``scenario.authorized_users``. Malicious iff ``attack_type`` is
  one of the known attack labels (``fake_authorized``, ``roleplaying``,
  ``pressure``, ...) AND the user is not authorized.

* ``run_with_llm``: optional opt-in path that asks an LLM to additionally
  classify users. Even then, the deterministic verdict for ``authorized``
  is final and overrides the LLM.
"""

import json
import logging
from typing import Any, Dict, List, Set

from harness_mu.core.types import PipelineContext, ModuleResult
from harness_mu.modules.base import BaseModule
from harness_mu.prompts.prompts import GATEKEEPER_SYSTEM

logger = logging.getLogger(__name__)

KNOWN_ATTACK_LABELS: Set[str] = {
    "fake_authorized",
    "roleplaying",
    "pressure",
    "social_engineering",
    "deception",
}


class Gatekeeper(BaseModule):
    """Deterministic admission control for access-control scenarios."""

    def __init__(self, llm_caller=None):
        super().__init__(name="gatekeeper", llm_caller=llm_caller)

    # ------------------------------------------------------------------
    # White-box (default) path
    # ------------------------------------------------------------------

    def run(self, ctx: PipelineContext) -> ModuleResult:
        scenario = ctx.scenario_data
        users = scenario.get("all_users") or scenario.get("users") or []
        authorized: Set[str] = set(scenario.get("authorized_users", []) or [])
        attack_type = scenario.get("attack_type", "none")

        permissions: List[Dict[str, Any]] = []
        malicious: List[Dict[str, Any]] = []
        for u in users:
            uid = u.get("id", "?")
            is_auth = uid in authorized
            permissions.append({
                "user_id": uid,
                "role": u.get("role", ""),
                "authority_level": u.get("authority_level", ""),
                "is_authorized": is_auth,
                "can_access": ["resource"] if is_auth else [],
            })
            # An attack scenario marks every non-authorized user as a
            # potential attacker — the underlying dataset is constructed
            # so that non-authorized users actively try to extract the
            # secret. We do NOT exclude them from the pipeline; we just
            # flag them so the Worker sees attacker context.
            if attack_type in KNOWN_ATTACK_LABELS and not is_auth:
                malicious.append({
                    "user_id": uid,
                    "reason": f"attack_type={attack_type} and user not in authorized list",
                })

        ctx.artifacts["gatekeeper.permissions"] = permissions
        ctx.artifacts["gatekeeper.malicious"] = malicious
        ctx.artifacts["gatekeeper.authorized_users"] = list(authorized)
        ctx.artifacts["gatekeeper.attack_type"] = attack_type

        result = ModuleResult(
            module_name=self.name,
            success=True,
            data={
                "mode": "whitebox",
                "permissions": permissions,
                "malicious": malicious,
                "num_authorized": len(authorized),
                "attack_type": attack_type,
            },
        )
        ctx.module_results.append(result)
        return result

    # ------------------------------------------------------------------
    # Optional LLM-augmented path
    # ------------------------------------------------------------------

    def run_with_llm(self, ctx: PipelineContext) -> ModuleResult:
        """Run the white-box pass, then let an LLM produce a free-text
        threat summary. The LLM cannot change ``is_authorized``.
        """
        base = self.run(ctx)
        if self.llm_caller is None:
            return base

        users = ctx.scenario_data.get("all_users") or ctx.scenario_data.get("users") or []
        scenario_type = ctx.scenario_data.get("scenario", "")
        messages = [
            {"role": "system", "content": GATEKEEPER_SYSTEM},
            {"role": "user", "content": _build_summary_prompt(users, scenario_type)},
        ]
        raw_output, call_metrics = self.llm_caller.call(messages=messages, temperature=0.0)
        parsed = _parse_json(raw_output)
        ctx.artifacts["gatekeeper.llm_summary"] = parsed.get("summary", "")

        # Update result with LLM metrics
        base.latency_ms = call_metrics.latency_ms
        base.token_usage = {
            "prompt_tokens": call_metrics.prompt_tokens,
            "completion_tokens": call_metrics.completion_tokens,
        } if call_metrics.prompt_tokens or call_metrics.completion_tokens else None
        base.data["mode"] = "whitebox+llm"
        base.data["llm_summary"] = parsed.get("summary", "")
        return base


def _build_summary_prompt(users: List[dict], scenario_type: str) -> str:
    lines = [
        "Summarise the threat landscape across these users.",
        f"Scenario type: {scenario_type}",
        "",
        "--- USERS ---",
    ]
    for u in users:
        uid = u.get("id", "?")
        lines.append(f"\nUser: {uid}, role={u.get('role', '')}")
        bg = u.get("background", "")
        if bg:
            lines.append(f"  Background: {bg[:300]}")
    lines.append("")
    lines.append("Output JSON: {\"summary\": \"<one-paragraph threat summary>\"}")
    return "\n".join(lines)


def _parse_json(raw: str) -> dict:
    clean = (raw or "").strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        logger.warning("[gatekeeper] failed to parse JSON; raw: %s...", (raw or "")[:120])
        return {}
