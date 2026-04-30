"""
Compatibility wrapper for config imports.

This module re-exports configuration from src.config.config to maintain
backward compatibility with existing imports (from src.core.config ...).

The actual configuration implementation is in src.config.config.
"""
from src.config.config import (
    ModelSettings,
    WhisperSettings,
    LLMSettings,
    GeminiSettings,
    VLLMSettings,
    ClassificationSettings,
    PipelineSettings,
    Settings,
    load_env_file,
    get_settings,
    set_settings,
)

__all__ = [
    "ModelSettings",
    "WhisperSettings",
    "LLMSettings",
    "GeminiSettings",
    "VLLMSettings",
    "ClassificationSettings",
    "PipelineSettings",
    "Settings",
    "load_env_file",
    "get_settings",
    "set_settings",
]