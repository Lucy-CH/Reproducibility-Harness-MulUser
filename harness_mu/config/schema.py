"""Pydantic v2 configuration models for Harness-MU."""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class ModuleModelConfig(BaseModel):
    """Model and provider assignment for a single module."""
    provider: str = "openai"
    model: str = "deepseek-v4-pro"
    temperature: float = 0.0
    # Higher default than 3 to absorb MAAS burst-suppression (3 req/min);
    # DeepSeek almost never errors so the extra slots are unused there.
    max_retries: int = 6
    max_tokens: int = 4096


class ModuleActivation(BaseModel):
    """Which modules are active for a given scenario."""
    gatekeeper: bool = False
    mediator: bool = False
    worker: bool = True
    compliance_checker: bool = False


class TranscriptConfig(BaseModel):
    """SimulatedUser transcript handling (Access-Control only).

    * ``mode='off'``  — single-turn flow with the dataset's static
      ``initial_messages``.
    * ``mode='replay'`` — open-loop replay: load cached SimulatedUser
      messages produced by Muses-Bench and flatten all turns into each
      user's ``instructions`` so the Worker sees bit-identical attack
      content in a single call.
    * ``mode='live'`` — multi-turn live (B+D+E1): each global turn the
      adapter drives ``SimulatedUser`` (live, adaptive) -> Gatekeeper
      -> Mediator (with cross-user fact-checking) -> Worker per-user
      isolated step -> ComplianceChecker (log-only). Final eval uses
      concatenated per-user replies.
    """
    mode: str = "off"  # "off" | "replay" | "live"
    cache_dir: Optional[str] = None  # None -> <repo_root>/transcripts
    simulator_model: str = "deepseek-v4-pro"
    simulator_provider: str = "openai"
    simulator_temperature: float = 1.0
    max_turns: int = 10


class ScenarioConfig(BaseModel):
    """Configuration for a single benchmark scenario."""
    enabled: bool = True
    data_file: str = ""
    modules: ModuleActivation = Field(default_factory=ModuleActivation)
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)


class OutputConfig(BaseModel):
    """Output and logging settings.

    ``subdir`` (optional) is inserted between the scenario name and the
    dataset stem in the output path, e.g. ``results/access_control/live/...``.
    Default ``None`` preserves the legacy layout (no insertion), so the IF
    scenario and pre-existing AC runs are unaffected.
    """
    results_dir: str = "results"
    subdir: Optional[str] = None
    save_raw: bool = True
    save_metrics: bool = True


class ExecutionConfig(BaseModel):
    """Execution and concurrency settings."""
    max_workers: int = 16
    debug: bool = False


class HarnessMUConfig(BaseModel):
    """Root configuration for a Harness-MU experiment run."""
    default_model: ModuleModelConfig = Field(default_factory=ModuleModelConfig)
    module_models: Dict[str, ModuleModelConfig] = Field(default_factory=dict)
    scenarios: Dict[str, ScenarioConfig] = Field(default_factory=dict)
    output: OutputConfig = Field(default_factory=OutputConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
