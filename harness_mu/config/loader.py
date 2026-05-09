"""Configuration loader with YAML parsing and environment variable override."""

import os
from pathlib import Path
from typing import Optional

import yaml

from harness_mu.config.schema import HarnessMUConfig


def _defaults_path() -> Path:
    """Return the path to the bundled defaults.yaml."""
    return Path(__file__).resolve().parent / "defaults.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return the parsed dict. Returns {} if file missing."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. override values take precedence."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(config: HarnessMUConfig) -> HarnessMUConfig:
    """Apply HMU_* environment variable overrides to the config.

    Supported env vars:
        HMU_DEFAULT_MODEL       -> default_model.model
        HMU_DEFAULT_PROVIDER    -> default_model.provider
        HMU_MEDIATOR_MODEL      -> module_models.mediator.model
        HMU_WORKER_MODEL        -> module_models.worker.model
        HMU_GATEKEEPER_MODEL    -> module_models.gatekeeper.model
        HMU_COMPLIANCE_MODEL    -> module_models.compliance_checker.model
        HMU_DEBUG               -> execution.debug ("1"/"true")
        HMU_MAX_WORKERS         -> execution.max_workers
    """
    env_map = {
        "HMU_DEFAULT_MODEL": ("default_model", "model"),
        "HMU_DEFAULT_PROVIDER": ("default_model", "provider"),
        "HMU_MEDIATOR_MODEL": ("module_models", "mediator", "model"),
        "HMU_WORKER_MODEL": ("module_models", "worker", "model"),
        "HMU_GATEKEEPER_MODEL": ("module_models", "gatekeeper", "model"),
        "HMU_COMPLIANCE_MODEL": ("module_models", "compliance_checker", "model"),
    }

    for env_var, path_parts in env_map.items():
        value = os.environ.get(env_var)
        if value is None:
            continue
        if len(path_parts) == 2:
            section, attr = path_parts
            if section == "default_model":
                setattr(config.default_model, attr, value)
        elif len(path_parts) == 3:
            _, module_name, attr = path_parts
            if module_name not in config.module_models:
                from harness_mu.config.schema import ModuleModelConfig
                config.module_models[module_name] = ModuleModelConfig()
            setattr(config.module_models[module_name], attr, value)

    debug_val = os.environ.get("HMU_DEBUG", "").lower()
    if debug_val in ("1", "true"):
        config.execution.debug = True

    max_workers = os.environ.get("HMU_MAX_WORKERS")
    if max_workers is not None:
        try:
            config.execution.max_workers = int(max_workers)
        except ValueError:
            pass

    return config


def load_config(config_path: Optional[str] = None) -> HarnessMUConfig:
    """Load Harness-MU configuration from a YAML file, merged with defaults.

    Args:
        config_path: Path to a user-provided YAML config file. If None, only
                     defaults are used.

    Returns:
        A validated HarnessMUConfig instance.

    Merge order (later overrides earlier):
        1. Bundled defaults.yaml
        2. User-provided config file (if given)
        3. Environment variables (HMU_* prefix)
    """
    merged = _load_yaml(_defaults_path())

    if config_path:
        user_config = _load_yaml(Path(config_path))
        merged = _deep_merge(merged, user_config)

    config = HarnessMUConfig(**merged)
    config = _apply_env_overrides(config)
    return config
