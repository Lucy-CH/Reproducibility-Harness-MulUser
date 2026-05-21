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
    """Open-loop SimulatedUser transcript replay (Access-Control only).

    When ``mode == 'replay'`` the AC adapter loads cached SimulatedUser
    messages produced by Muses-Bench and replaces each user's instruction
    text with the flattened multi-turn attack content, so the Worker sees
    bit-identical attack input to what the Muses-Bench defender saw.
    """
    mode: str = "off"  # "off" | "replay"
    cache_dir: Optional[str] = None  # None -> <repo_root>/transcripts
    simulator_model: str = "deepseek-v4-pro"
    max_turns: int = 10


class ScenarioConfig(BaseModel):
    """Configuration for a single benchmark scenario."""
    enabled: bool = True
    data_file: str = ""
    modules: ModuleActivation = Field(default_factory=ModuleActivation)
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)


class OutputConfig(BaseModel):
    """Output and logging settings."""
    results_dir: str = "results"
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
