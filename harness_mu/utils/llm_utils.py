"""LLM API calling with retry mechanism and exponential backoff.

Adapted from muses_bench to remove external dependency. Supports OpenAI-compatible
proxy APIs (Huawei MAAS, DeepSeek, etc.), LiteLLM, and optional vLLM.

Includes a process-wide token-bucket rate limiter (env: OPENAI_RATE_LIMIT_PER_MIN)
shared across all module callers and worker threads, plus throttle-aware retry
with longer backoff when the upstream returns 429 / 'ces.0429' / 'Too Many Requests'
(the Huawei MAAS throttle markers).
"""

import os
import random
import threading
import time
import httpx
from litellm import completion
from openai import OpenAI


# ---------------------------------------------------------------------------
# Global rate limiter (token bucket)
# ---------------------------------------------------------------------------

class _TokenBucketRateLimiter:
    """Thread-safe token bucket. ``acquire()`` blocks until a token is available.

    Bucket is sized at ``rate_per_sec`` tokens (1-second burst capacity) and
    refills continuously at ``rate_per_sec`` tokens/second. This lets
    concurrent workers fire normally on providers without a per-second hard
    guard. For providers that do enforce a strict per-second limit (e.g.
    MAAS), lower ``OPENAI_RATE_LIMIT_PER_MIN`` so ``rate_per_sec <= 1``,
    which collapses the bucket back to single-request spacing.
    """

    def __init__(self, rate_per_minute: float):
        self.rate_per_sec = rate_per_minute / 60.0
        self.capacity = max(1.0, self.rate_per_sec)
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_sec)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                wait = deficit / self.rate_per_sec
            time.sleep(wait)


_RATE_LIMITER_LOCK = threading.Lock()
_RATE_LIMITER: "_TokenBucketRateLimiter | None" = None
_RATE_LIMITER_RPM: float = -1.0  # sentinel: not yet resolved


def _get_rate_limiter() -> "_TokenBucketRateLimiter | None":
    """Lazy singleton. Reads OPENAI_RATE_LIMIT_PER_MIN each first access.

    Returns None when the limit is unset or <= 0 (i.e. unlimited — preserves
    the legacy DeepSeek code path).
    """
    global _RATE_LIMITER, _RATE_LIMITER_RPM
    if _RATE_LIMITER_RPM >= 0:
        return _RATE_LIMITER
    with _RATE_LIMITER_LOCK:
        if _RATE_LIMITER_RPM >= 0:
            return _RATE_LIMITER
        try:
            rpm = float(os.environ.get("OPENAI_RATE_LIMIT_PER_MIN", "0") or 0)
        except ValueError:
            rpm = 0.0
        _RATE_LIMITER_RPM = rpm
        _RATE_LIMITER = _TokenBucketRateLimiter(rpm) if rpm > 0 else None
        return _RATE_LIMITER


_THROTTLE_MARKERS = ("429", "ces.0429", "too many requests", "rate limit")


def _is_throttle_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in _THROTTLE_MARKERS)


def _content_indicates_throttle(content: str) -> bool:
    if not content:
        return False
    low = content.lower()
    return "ces.0429" in low or "too many requests" in low


def call_llm_with_retry(
    model: str,
    provider: str,
    messages: list,
    temperature: float,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    max_tokens: int = 4096,
    llm_client=None,
):
    """Call LLM API with retry and exponential backoff.

    Args:
        model: Model name.
        provider: Provider name (e.g. "openai", "vllm").
        messages: Conversation messages in OpenAI format.
        temperature: Sampling temperature.
        max_retries: Maximum retry attempts.
        retry_delay: Initial retry delay in seconds (doubles each retry).
        llm_client: Optional vLLM client instance.

    Returns:
        API response object (or MockResponse for vLLM matching OpenAI structure).
    """

    class MockResponse:
        def __init__(self, content):
            self.choices = [
                type("Choice", (), {
                    "message": type("Message", (), {"content": content})(),
                })()
            ]
            self.usage = None

    if provider == "vllm":
        if llm_client is None:
            raise ValueError("llm_client must be provided for vllm provider")
        try:
            from muses_bench.utils.vllm_client import generate_vllm_response
        except ImportError:
            try:
                from harness_mu.utils.vllm_client import generate_vllm_response
            except ImportError:
                raise ImportError(
                    "vLLM support requires muses_bench or harness_mu vllm_client"
                )
        try:
            response_text = generate_vllm_response(
                llm_client, None, messages, temperature=temperature,
            )
            return MockResponse(response_text)
        except Exception as e:
            print(f"[ERROR] vLLM generation failed: {e}")
            raise

    api_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("OPENAI_API_KEY")
    rate_limiter = _get_rate_limiter()

    last_exception = None

    for attempt in range(max_retries):
        try:
            if rate_limiter is not None:
                rate_limiter.acquire()

            if api_base:
                verify_ssl = os.environ.get("OPENAI_VERIFY_SSL", "true").lower() != "false"
                http_client = httpx.Client(verify=verify_ssl)
                # max_retries=0 disables the OpenAI SDK's *internal* retry on
                # 429 (which fires 3 sub-second retries in a burst and trips
                # MAAS's 3-req/min burst-suppression layer). We do our own
                # rate-limited, throttle-aware retry in the outer loop.
                client = OpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=http_client,
                    max_retries=0,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if hasattr(response, "error") and response.error:
                    raise Exception(f"API returned error: {response.error}")
                if not response.choices or len(response.choices) == 0:
                    raise Exception("API response has no choices")
                if (
                    not response.choices[0].message
                    or not response.choices[0].message.content
                ):
                    raise Exception("API response has no message content")
                # MAAS may return HTTP 200 with a throttle marker in the body.
                content = response.choices[0].message.content
                if _content_indicates_throttle(content):
                    raise Exception(f"API returned throttle marker in body: {content[:120]}")
                return response
            else:
                model_name = f"{provider}/{model}"
                response = completion(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not response.choices or len(response.choices) == 0:
                    raise Exception("API response has no choices")
                if (
                    not response.choices[0].message
                    or not response.choices[0].message.content
                ):
                    raise Exception("API response has no message content")
                content = response.choices[0].message.content
                if _content_indicates_throttle(content):
                    raise Exception(f"API returned throttle marker in body: {content[:120]}")
                return response

        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                if _is_throttle_error(e):
                    # MAAS deepseek-v4-pro enforces two limits: 1 req/s
                    # steady-state plus a 3-req/min burst-suppression layer.
                    # Floor backoff at 10 s + jitter so the latter clears.
                    wait_time = max(10.0, retry_delay * (2 ** attempt)) + random.uniform(0.0, 2.0)
                    print(
                        f"[WARNING] API throttled (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                else:
                    wait_time = retry_delay * (2 ** attempt)
                    print(
                        f"[WARNING] API call failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                print(f"[INFO] Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(
                    f"[ERROR] API call failed after {max_retries} attempts: {e}"
                )
                raise

    raise last_exception if last_exception else Exception("Unknown error in API call")
