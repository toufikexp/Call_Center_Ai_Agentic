"""
Compatibility wrapper for configuration.

The real configuration now lives in `src/config/config.py`. This module simply
re-exports the same API so existing imports like:

    from src.core.config import get_settings, WhisperSettings

continue to work without code changes.
"""

from src.config.config import (  # type: ignore
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
