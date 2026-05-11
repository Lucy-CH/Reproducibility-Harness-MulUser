"""Abstract base class for all Harness-MU pipeline modules."""

import logging
from abc import ABC, abstractmethod

from harness_mu.config.schema import ModuleActivation
from harness_mu.core.types import PipelineContext, ModuleResult
from harness_mu.llm.wrappers import InstrumentedLLMCaller

logger = logging.getLogger(__name__)


class BaseModule(ABC):
    """Contract for orchestratable modules in the Harness-MU pipeline.

    Each module (Gatekeeper, Mediator, Worker, ComplianceChecker) inherits
    from this class and implements the ``run`` method. Modules are
    independently toggleable per scenario via ModuleActivation config.
    """

    def __init__(
        self,
        name: str,
        llm_caller: InstrumentedLLMCaller,
    ):
        self.name = name
        self.llm_caller = llm_caller

    @abstractmethod
    def run(self, ctx: PipelineContext) -> ModuleResult:
        """Execute the module and append the result to ctx.module_results."""
        ...

    def is_enabled(self, activation: ModuleActivation) -> bool:
        """Check whether this module is active for a given activation config."""
        mapping = {
            "gatekeeper": activation.gatekeeper,
            "mediator": activation.mediator,
            "worker": activation.worker,
            "compliance_checker": activation.compliance_checker,
        }
        return mapping.get(self.name, False)
