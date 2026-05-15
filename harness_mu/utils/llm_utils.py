"""LLM API calling with retry mechanism and exponential backoff.

Adapted from muses_bench to remove external dependency. Supports OpenAI-compatible
proxy APIs (Huawei MAAS, DeepSeek, etc.), LiteLLM, and optional vLLM.
"""

import os
import time
import httpx
from litellm import completion
from openai import OpenAI


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

    last_exception = None

    for attempt in range(max_retries):
        try:
            if api_base:
                verify_ssl = os.environ.get("OPENAI_VERIFY_SSL", "true").lower() != "false"
                http_client = httpx.Client(verify=verify_ssl)
                client = OpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=http_client,
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
                return response

        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
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
