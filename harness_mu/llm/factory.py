"""LLMFactory creates and caches InstrumentedLLMCaller instances per module."""

import logging
from typing import Dict

from harness_mu.config.schema import HarnessMUConfig, ModuleModelConfig
from harness_mu.llm.wrappers import InstrumentedLLMCaller
from harness_mu.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)


class LLMFactory:
    """Manages LLM callers for all modules.

    Each logical module (gatekeeper, mediator, worker, compliance_checker)
    gets its own InstrumentedLLMCaller, bound to the model configured for
    that module. The factory caches callers lazily on first access.

    Model resolution order: module_models.{module_name} → default_model.
    """

    def __init__(self, config: HarnessMUConfig, metrics_collector: MetricsCollector):
        self._config = config
        self._metrics_collector = metrics_collector
        self._callers: Dict[str, InstrumentedLLMCaller] = {}

    def get_caller(self, module_name: str) -> InstrumentedLLMCaller:
        """Get or create an InstrumentedLLMCaller for the given module.

        Args:
            module_name: One of "gatekeeper", "mediator", "worker",
                         "compliance_checker".

        Returns:
            An InstrumentedLLMCaller bound to the module's model config.
        """
        if module_name not in self._callers:
            model_config = self._resolve_model_config(module_name)
            caller = InstrumentedLLMCaller(
                module_name=module_name,
                model_config=model_config,
                metrics_collector=self._metrics_collector,
            )
            self._callers[module_name] = caller
            logger.info(
                "Created LLM caller for '%s': %s/%s (temp=%.1f)",
                module_name,
                model_config.provider,
                model_config.model,
                model_config.temperature,
            )
        return self._callers[module_name]

    def _resolve_model_config(self, module_name: str) -> ModuleModelConfig:
        module_models = self._config.module_models or {}
        return module_models.get(module_name, self._config.default_model)

    @property
    def cached_modules(self) -> list:
        return list(self._callers.keys())
