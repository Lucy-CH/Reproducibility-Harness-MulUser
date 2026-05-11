"""ComplianceChecker: Post-execution dual-loop compliance verification.

For instruction_following, the ComplianceChecker is a pass-through stub
because this scenario has no data-leak risks. The full implementation
(white-box keyword check + LLM fine-grained check) will be needed for
access_control and meeting_scheduling.
"""

import logging

from harness_mu.core.types import PipelineContext, ModuleResult
from harness_mu.modules.base import BaseModule

logger = logging.getLogger(__name__)


class ComplianceChecker(BaseModule):
    """Post-execution compliance check (stub for instruction_following)."""

    def __init__(self, llm_caller):
        super().__init__(name="compliance_checker", llm_caller=llm_caller)

    def run(self, ctx: PipelineContext) -> ModuleResult:
        result = ModuleResult(
            module_name=self.name,
            success=True,
            data={"passed": True, "note": "stub — no data leak risk in IF scenario"},
        )
        ctx.module_results.append(result)
        logger.debug("[compliance_checker] pass-through OK")
        return result
