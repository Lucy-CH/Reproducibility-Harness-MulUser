"""Pydantic v2 configuration models for Harness-MU."""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class ModuleModelConfig(BaseModel):
    """Model and provider assignment for a single module."""
    provider: str = "openai"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_retries: int = 3


class ModuleActivation(BaseModel):
    """Which modules are active for a given scenario."""
    gatekeeper: bool = False
    mediator: bool = False
    worker: bool = True
    compliance_checker: bool = False


class ScenarioConfig(BaseModel):
    """Configuration for a single benchmark scenario."""
    enabled: bool = True
    data_file: str = ""
    modules: ModuleActivation = Field(default_factory=ModuleActivation)


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
