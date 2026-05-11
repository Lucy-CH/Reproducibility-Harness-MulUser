"""Instrumented LLM caller wrapping muses_bench's call_llm_with_retry."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from harness_mu.utils.llm_utils import call_llm_with_retry

from harness_mu.config.schema import ModuleModelConfig
from harness_mu.core.types import CallMetrics
from harness_mu.metrics.collector import MetricsCollector
from harness_mu.utils.timer import Timer

logger = logging.getLogger(__name__)


class InstrumentedLLMCaller:
    """Wraps muses_bench's LLM calling with latency and token instrumentation.

    Each module gets its own InstrumentedLLMCaller, bound to the model
    configured for that module. All calls are recorded to the shared
    MetricsCollector for per-module aggregation.
    """

    def __init__(
        self,
        module_name: str,
        model_config: ModuleModelConfig,
        metrics_collector: MetricsCollector,
    ):
        self.module_name = module_name
        self.model_config = model_config
        self.metrics_collector = metrics_collector

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        llm_client: Any = None,
        lora_request: Any = None,
    ) -> Tuple[str, CallMetrics]:
        """Call the LLM with instrumentation.

        Args:
            messages: OpenAI-format conversation messages.
            temperature: Override for the configured temperature.
            llm_client: Optional vLLM client instance.
            lora_request: Optional LoRA request.

        Returns:
            Tuple of (response_text, CallMetrics).
        """
        temp = temperature if temperature is not None else self.model_config.temperature

        with Timer() as timer:
            response = call_llm_with_retry(
                model=self.model_config.model,
                provider=self.model_config.provider,
                messages=messages,
                temperature=temp,
                max_retries=self.model_config.max_retries,
                max_tokens=self.model_config.max_tokens,
                llm_client=llm_client,
                lora_request=lora_request,
            )

        response_text = response.choices[0].message.content

        # Extract token usage if available
        prompt_tokens = 0
        completion_tokens = 0
        try:
            usage = response.usage
            if usage:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
        except (AttributeError, TypeError):
            pass

        call_metrics = CallMetrics(
            module_name=self.module_name,
            latency_ms=timer.elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model_config.model,
        )

        self.metrics_collector.record_call(call_metrics)

        logger.debug(
            "[%s] model=%s latency=%.0fms prompt_tokens=%d completion_tokens=%d",
            self.module_name,
            self.model_config.model,
            timer.elapsed_ms,
            prompt_tokens,
            completion_tokens,
        )

        return response_text, call_metrics
