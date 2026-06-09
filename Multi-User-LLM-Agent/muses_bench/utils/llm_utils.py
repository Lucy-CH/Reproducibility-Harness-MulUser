import json
import os
import threading
import time
import httpx
from collections import defaultdict
from litellm import completion
from openai import OpenAI

# Import vLLM helper
try:
    from muses_bench.utils.vllm_helper import generate_vllm_helper
except ImportError:
    pass # Will be imported if needed, or handled inside the function


# ---------------------------------------------------------------------------
# Per-call metrics instrumentation (latency + token tracking)
# ---------------------------------------------------------------------------

class LLMCallMetricsCollector:
    """Thread-safe collector for per-module LLM call metrics.

    One global instance is shared across all callers (agent + SimulatedUser).
    Callers tag each call with a ``module_name`` so the final summary can
    break down costs by logical component.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._records: list = []   # raw per-call dicts
        self._reset_each_scenario()  # initialise per-scenario buffer

    # -- per-scenario buffer (cleared after each scenario) -------------------

    def _reset_each_scenario(self):
        self._scenario_calls: list = []

    def record_call(self, module_name: str, model: str, messages: list,
                    prompt_tokens: int, completion_tokens: int,
                    latency_ms: float, success: bool):
        rec = {
            "module_name": module_name,
            "model": model,
            "num_messages": len(messages),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": round(latency_ms, 1),
            "success": success,
        }
        with self._lock:
            self._records.append(rec)
            self._scenario_calls.append(rec)

    def pop_scenario_calls(self) -> list:
        """Return and clear the per-scenario buffer."""
        with self._lock:
            calls = list(self._scenario_calls)
            self._scenario_calls = []
        return calls

    # -- aggregate summaries ------------------------------------------------

    def get_module_summaries(self) -> dict:
        """Return per-module aggregate stats (same schema as HMU module_metrics)."""
        with self._lock:
            by_module = defaultdict(lambda: {
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_latency_ms": 0.0,
                "model": "",
            })
            for r in self._records:
                m = r["module_name"]
                s = by_module[m]
                s["total_calls"] += 1
                s["total_prompt_tokens"] += r["prompt_tokens"]
                s["total_completion_tokens"] += r["completion_tokens"]
                s["total_latency_ms"] += r["latency_ms"]
                if not s["model"]:
                    s["model"] = r["model"]
            # compute averages
            for name, s in by_module.items():
                if s["total_calls"] > 0:
                    s["avg_latency_ms"] = round(s["total_latency_ms"] / s["total_calls"], 1)
                else:
                    s["avg_latency_ms"] = 0.0
            return dict(by_module)

    def save_module_metrics(self, path: str) -> None:
        """Write per-module aggregate metrics to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_module_summaries(), f, indent=2)
        print(f"[INFO] Module metrics written to: {path}")


# Global singleton — shared across all LLM call sites in a process.
_metrics_collector = LLMCallMetricsCollector()


def get_metrics_collector():
    return _metrics_collector


def reset_metrics_collector():
    """Reset the global collector (useful between batch runs)."""
    global _metrics_collector
    _metrics_collector = LLMCallMetricsCollector()


# ---------------------------------------------------------------------------
# Instrumented LLM call wrapper
# ---------------------------------------------------------------------------

def call_llm_with_retry(model: str, provider: str, messages: list,
                        temperature: float, max_retries: int = 3,
                        retry_delay: float = 1.0, llm_client=None,
                        lora_request=None, module_name: str = "agent"):
    """
    Call LLM API with retry mechanism and exponential backoff.

    Every call is automatically instrumented: latency + token usage are
    recorded to the global ``LLMCallMetricsCollector``.  The ``module_name``
    parameter tags the call (e.g. "agent" or "simulated_user") so costs
    can be broken down per component.

    Args:
        model: Model name
        provider: Provider name (e.g., openai, anthropic, vllm)
        messages: Conversation messages
        temperature: Temperature setting
        max_retries: Maximum number of retries
        retry_delay: Initial retry delay in seconds
        llm_client: Optional vLLM client instance
        lora_request: Optional LoRARequest instance
        module_name: Label for metrics aggregation (default "agent").

    Returns:
        API response object (Mock object for vLLM to match OpenAI structure)
    """
    collector = _metrics_collector

    # helper class to mock OpenAI response structure for vLLM
    class MockResponse:
        def __init__(self, content):
            self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})()})()]
            self.usage = None  # vLLM doesn't provide token counts

    if provider == "vllm":
        if llm_client is None:
            raise ValueError("llm_client must be provided for vllm provider")

        from muses_bench.utils.vllm_client import generate_vllm_response

        t0 = time.perf_counter()
        try:
            response_text = generate_vllm_response(
                llm_client, lora_request, messages, temperature=temperature
            )
            latency = (time.perf_counter() - t0) * 1000.0
            collector.record_call(
                module_name, model, messages,
                prompt_tokens=0, completion_tokens=0,
                latency_ms=latency, success=True,
            )
            return MockResponse(response_text)
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000.0
            collector.record_call(
                module_name, model, messages,
                prompt_tokens=0, completion_tokens=0,
                latency_ms=latency, success=False,
            )
            print(f"[ERROR] vLLM generation failed: {e}")
            raise e

    api_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("OPENAI_API_KEY")

    last_exception = None

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            if api_base:
                # Use OpenAI library directly for proxy APIs
                verify_ssl = os.environ.get("OPENAI_VERIFY_SSL", "true").lower() != "false"
                http_client = httpx.Client(verify=verify_ssl)
                client = OpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=http_client
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )

                # Check if response has error
                if hasattr(response, 'error') and response.error:
                    raise Exception(f"API returned error: {response.error}")

                # Check if response has choices
                if not response.choices or len(response.choices) == 0:
                    raise Exception("API response has no choices")

                # Check if message content exists
                if not response.choices[0].message or not response.choices[0].message.content:
                    raise Exception("API response has no message content")

                # -- instrumentation --
                latency = (time.perf_counter() - t0) * 1000.0
                pt = getattr(response.usage, 'prompt_tokens', 0) or 0
                ct = getattr(response.usage, 'completion_tokens', 0) or 0
                collector.record_call(
                    module_name, model, messages,
                    prompt_tokens=pt, completion_tokens=ct,
                    latency_ms=latency, success=True,
                )
                return response
            else:
                # Use litellm for standard providers
                model_name = f"{provider}/{model}"
                completion_kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature
                }
                response = completion(**completion_kwargs)

                if not response.choices or len(response.choices) == 0:
                    raise Exception("API response has no choices")

                if not response.choices[0].message or not response.choices[0].message.content:
                    raise Exception("API response has no message content")

                # -- instrumentation --
                latency = (time.perf_counter() - t0) * 1000.0
                pt = getattr(response, 'usage', {}).get('prompt_tokens', 0) or 0 if hasattr(response, 'usage') else 0
                ct = getattr(response, 'usage', {}).get('completion_tokens', 0) or 0 if hasattr(response, 'usage') else 0
                # litellm wraps usage slightly differently
                try:
                    pt = response.usage.prompt_tokens or 0
                except Exception:
                    pt = 0
                try:
                    ct = response.usage.completion_tokens or 0
                except Exception:
                    ct = 0
                collector.record_call(
                    module_name, model, messages,
                    prompt_tokens=pt, completion_tokens=ct,
                    latency_ms=latency, success=True,
                )
                return response

        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000.0
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"[WARNING] API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"[INFO] Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] API call failed after {max_retries} attempts: {e}")
                collector.record_call(
                    module_name, model, messages,
                    prompt_tokens=0, completion_tokens=0,
                    latency_ms=latency, success=False,
                )
                raise

    # Should not reach here, but just in case
    raise last_exception if last_exception else Exception("Unknown error in API call")

