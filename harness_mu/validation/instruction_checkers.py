"""Thin wrapper that forwards instruction-following checks to IHEval.

This module exposes the same public surface (CheckerRegistry and the
REGISTRY singleton with an INSTRUCTION_DICT mapping) used by
harness_mu.adapters.instruction_following and
harness_mu.modules.compliance_checker, but the underlying checker
classes are imported from the official IHEval (NAACL 2025) source tree
rather than being re-implemented here. This guarantees byte-identical
behaviour with the Muses-Bench IF evaluator.

IHEval is expected to be available as a sibling repository at:

    Multi-User-LLM-Agent/IHEval/src/rule_following/evaluate/

If it cannot be located the module raises ImportError at import time
so the failure surfaces immediately instead of silently degrading to an
empty registry. The path can be overridden with the IHEVAL_SRC
environment variable.
"""

from __future__ import annotations

import os
import sys
from typing import Dict


def _resolve_iheval_src() -> str:
    env_override = os.environ.get("IHEVAL_SRC")
    if env_override:
        return os.path.abspath(env_override)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(repo_root, "Multi-User-LLM-Agent", "IHEval", "src")


IHEVAL_SRC = _resolve_iheval_src()
IHEVAL_EVAL_SRC = os.path.join(IHEVAL_SRC, "rule_following", "evaluate")

if not os.path.isdir(IHEVAL_EVAL_SRC):
    raise ImportError(
        f"IHEval source not found at {IHEVAL_EVAL_SRC}. "
        "Clone https://github.com/ytyz1307zzh/IHEval into "
        "Multi-User-LLM-Agent/IHEval, or set the IHEVAL_SRC env var."
    )

# Order matters: the inner evaluate directory must be on sys.path first
# because instructions_registry.py does "import instructions" as a
# top-level module rather than "from . import instructions".
for _p in (IHEVAL_EVAL_SRC, IHEVAL_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rule_following.evaluate import instructions_registry as _iheval_registry  # noqa: E402


class CheckerRegistry:
    """Mirrors instructions_registry.INSTRUCTION_DICT for drop-in use."""

    INSTRUCTION_DICT: Dict[str, type] = _iheval_registry.INSTRUCTION_DICT


REGISTRY = CheckerRegistry()


__all__ = ["REGISTRY", "CheckerRegistry", "IHEVAL_SRC", "IHEVAL_EVAL_SRC"]
