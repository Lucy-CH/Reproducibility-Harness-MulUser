"""Shared transcript cache for Access-Control multi-turn SimulatedUser dialogues.

Both ``muses_bench`` and ``harness_mu`` read/write the same cache so that one
fixed simulator-model run can supply the attack transcripts for any number of
downstream defender-model experiments (open-loop replay).

Cache layout::

    <repo_root>/transcripts/access_control/
        <dataset_stem>__sim-<simulator_model_safe>__t<N>.jsonl

Each line in the file is one scenario record. See ``ScenarioTranscript`` for
the schema.
"""

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_CACHE_ROOT_ENV = "AC_TRANSCRIPT_CACHE_DIR"


def _slug(value: str) -> str:
    """Make ``value`` safe for use in a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unknown"


def default_cache_root() -> Path:
    """Resolve the default cache root.

    Resolution order:
    1. Explicit env var ``AC_TRANSCRIPT_CACHE_DIR``.
    2. ``<repo_root>/transcripts`` where ``repo_root`` is the parent of the
       ``Multi-User-LLM-Agent`` directory.
    """
    env = os.environ.get(_DEFAULT_CACHE_ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    # .../Multi-User-LLM-Agent/muses_bench/utils/transcript_cache.py
    muses_root = here.parents[2]
    repo_root = muses_root.parent
    return repo_root / "transcripts"


def cache_file_path(
    cache_root: Path,
    scenario_kind: str,
    dataset_stem: str,
    simulator_model: str,
    max_turns: int,
) -> Path:
    """Compose the cache file path for a (dataset, simulator, turns) triple."""
    fname = f"{dataset_stem}__sim-{_slug(simulator_model)}__t{int(max_turns)}.jsonl"
    return Path(cache_root) / scenario_kind / fname


@dataclass
class TurnRecord:
    """One turn of SimulatedUser activity."""
    turn: int
    user_messages: Dict[str, str] = field(default_factory=dict)  # uid -> raw content
    agent_response: Optional[str] = None  # generation-time defender output (audit only)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"turn": self.turn, "user_messages": dict(self.user_messages)}
        if self.agent_response is not None:
            d["agent_response"] = self.agent_response
        return d


@dataclass
class ScenarioTranscript:
    """Complete transcript for a single scenario."""
    scenario_id: str
    dataset_stem: str
    simulator_model: str
    simulator_provider: str
    max_turns: int
    turns: List[TurnRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "dataset_stem": self.dataset_stem,
            "simulator_model": self.simulator_model,
            "simulator_provider": self.simulator_provider,
            "max_turns": self.max_turns,
            "turns": [t.to_dict() for t in self.turns],
            "metadata": dict(self.metadata),
        }


class TranscriptCache:
    """Append-or-load transcripts keyed by ``scenario_id`` within one file."""

    _file_locks: Dict[str, threading.Lock] = {}
    _index_lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._index_lock:
            self._file_locks.setdefault(str(self.path), threading.Lock())

    @property
    def _lock(self) -> threading.Lock:
        return self._file_locks[str(self.path)]

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Return ``{scenario_id: record}`` reading the file once."""
        if not self.path.exists():
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        with self._lock, self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = rec.get("scenario_id")
                if sid:
                    out[sid] = rec
        return out

    def load(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        return self.load_all().get(scenario_id)

    def save(self, transcript: ScenarioTranscript) -> None:
        """Append a transcript, replacing any prior record for the same id."""
        existing = self.load_all()
        existing[transcript.scenario_id] = transcript.to_dict()
        with self._lock, self.path.open("w", encoding="utf-8") as f:
            for sid in sorted(existing.keys()):
                f.write(json.dumps(existing[sid], ensure_ascii=False) + "\n")
