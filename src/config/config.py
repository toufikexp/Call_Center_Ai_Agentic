"""
Application configuration.

This module contains all Pydantic settings and helpers. It is re-exported by
`src/core/config.py` to keep existing imports working (from src.core.config ...).

Configuration files:
- classification_schema.json: Category and sub-category definitions
"""
import os
import json
from typing import Optional, Dict, List, Any
from pathlib import Path
from pydantic import BaseModel, Field


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.
    """
    if env_path is None:
        # Try project root
        # Path: src/config/config.py -> src/config -> src -> project_root
        # So we need parents[2] to get to project root
        project_root = Path(__file__).resolve().parents[2]
        env_path = project_root / ".env"

    env_path = Path(env_path)

    if env_path.exists():
        try:
            loaded_count = 0
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key not in os.environ:
                            os.environ[key] = value
                            loaded_count += 1
                            # Special handling for GEMINI_API_KEY
                            if key == "GEMINI_API_KEY":
                                if value and value.strip():
                                    import logging
                                    logger = logging.getLogger("config")
                                    if not logger.handlers:
                                        handler = logging.StreamHandler()
                                        handler.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
                                        logger.addHandler(handler)
                                        logger.setLevel(logging.INFO)
                                    logger.info("✅ GEMINI_API_KEY loaded from .env file")
                                else:
                                    import logging
                                    logger = logging.getLogger("config")
                                    if not logger.handlers:
                                        handler = logging.StreamHandler()
                                        handler.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
                                        logger.addHandler(handler)
                                        logger.setLevel(logging.WARNING)
                                    logger.warning("⚠️ GEMINI_API_KEY found in .env but value is empty")
        except Exception as e:
            # Log error instead of silently failing
            import logging
            logger = logging.getLogger("config")
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
                logger.addHandler(handler)
                logger.setLevel(logging.WARNING)
            logger.warning(f"Failed to load .env file from {env_path}: {e}")
    else:
        # Log if .env file not found (helpful for debugging)
        import logging
        logger = logging.getLogger("config")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.info(f".env file not found at {env_path} (using environment variables only)")


class ModelSettings(BaseModel):
    """Base settings for ML models."""

    model_path: str
    device: str = "auto"  # "auto", "cuda", "cpu"
    dtype: str = "float16"  # "float16", "float32"
    max_length: Optional[int] = None
    temperature: float = 0.7


class WhisperSettings(ModelSettings):
    """Settings for Whisper transcription model."""

    language: str = "ar"
    task: str = "transcribe"
    chunk_length_seconds: int = 30
    chunk_overlap_seconds: float = 2.0
    max_new_tokens: int = 444


class LLMSettings(ModelSettings):
    """Settings for LLM models (Qwen, etc.)."""

    trust_remote_code: bool = True


class GeminiSettings(BaseModel):
    """Settings for Gemini API (temporary)."""

    api_key: Optional[str] = None
    model_name: str = "gemini-2.0-flash-exp"
    timeout_seconds: int = 15
    min_refinement_length_ratio: float = 0.3
    default_refinement_score: float = 0.7
    refinement_prompt_template: str = Field(
        default="""You are an expert in Algerian Darija and Telecom industry terminology (Ooredoo).

Refine this transcript of a call between a subscriber and a customer service agent.

GUIDELINES:

1. Preserve the natural Algerian mix of Arabic and French (e.g., 'flexy', 'forfait', 'réseau', 'puce', 'crédit').
2. Identify and format speakers as [Agent] and [Subscriber] based on context.
3. Fix 'Whisper hallucinations' (repetitive loops or phonetic gibberish).
4. Add punctuation (commas, question marks) to reflect the flow of a real conversation.
5. Correct technical terms misheard by the ASR (e.g., 'débit' instead of 'debi').
6. Do not translate to Modern Standard Arabic; keep it in refined Darija.

Output MUST be a valid JSON object with the following keys:
- "refined_text": The complete refined transcript.
- "score": A quality score between 0.0 and 1.0 representing transcript coherence and meaning.

Original transcript:
{transcript}

JSON Output:""",
        description="Template for refinement prompt. Use {transcript} as placeholder.",
    )


class VLLMSettings(BaseModel):
    """Settings for local vLLM API."""

    base_url: str = "http://localhost:8080/v1"
    model_name: str = "Qwen/Qwen3-4B"
    api_key: str = "none"
    temperature: float = 0.1
    timeout_seconds: int = 30


class ClassificationSettings(BaseModel):
    """Settings for classification."""

    primary_categories: List[str] = Field(
        default=[
            "Customer Service Process Related",
            "Customer Service Support Topic",
            "Pricing",
            "Network",
            "Miscellaneous",
            "Product Category",
            "Channel",
            "Mobile Device",
            "Brand",
            "Information",
            "OTHER",
        ]
    )
    category_descriptions: Dict[str, str] = Field(default_factory=dict)
    category_subcategories: Dict[str, List[str]] = Field(default_factory=dict)
    subcategory_descriptions: Dict[str, Dict[str, str]] = Field(
        default_factory=dict
    )
    default_subcategory: Dict[str, str] = Field(default_factory=dict)
    other_category_name: str = "OTHER"
    other_category_threshold: float = 0.3


class PipelineSettings(BaseModel):
    """Pipeline execution settings."""

    confidence_threshold: float = 0.90
    refinement_threshold: float = 0.5
    max_retry_attempts: int = 2
    audio_sample_rate: int = 16000
    output_dir: str = "data/results"
    logs_dir: str = "data/logs"
    enable_verbose_logging: bool = True
    enable_file_logging: bool = True


class Settings(BaseModel):
    """Main application settings."""

    whisper: WhisperSettings
    gemini: GeminiSettings
    qwen: LLMSettings
    vllm: VLLMSettings
    dziribert_classifier: ModelSettings
    dziribert_sentiment: ModelSettings
    classification: ClassificationSettings
    pipeline: PipelineSettings

    @staticmethod
    def _load_classification_schema(path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load classification schema from a JSON file.

        If `path` is None, it defaults to `<project_root>/config/classification_schema.json`.
        """
        try:
            if path:
                schema_path = Path(path)
            else:
                # Schema is in the same directory as this config file
                config_dir = Path(__file__).resolve().parent
                schema_path = config_dir / "classification_schema.json"

            if not schema_path.exists():
                return {}

            with open(schema_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @classmethod
    def create_default(cls) -> "Settings":
        """Create default settings."""
        load_env_file()

        # Gemini API key from environment
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        # Trim whitespace and validate
        if gemini_key:
            gemini_key = gemini_key.strip()
            if not gemini_key:
                gemini_key = None
                import logging
                logger = logging.getLogger("config")
                if not logger.handlers:
                    handler = logging.StreamHandler()
                    handler.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
                    logger.addHandler(handler)
                    logger.setLevel(logging.WARNING)
                logger.warning("GEMINI_API_KEY is set but empty/whitespace. Refinement will be skipped.")
        else:
            import logging
            logger = logging.getLogger("config")
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            logger.info("ℹ️ GEMINI_API_KEY not found in environment. Refinement will be skipped.")
            logger.info("   To enable refinement, set GEMINI_API_KEY in .env file at project root.")

        # vLLM configuration from .env with safe defaults
        vllm_defaults = VLLMSettings()
        vllm_base_url = os.getenv("VLLM_BASE_URL", vllm_defaults.base_url)
        vllm_model_name = os.getenv("VLLM_MODEL_NAME", vllm_defaults.model_name)
        vllm_api_key = os.getenv("VLLM_API_KEY", vllm_defaults.api_key)
        vllm_temperature_str = os.getenv("VLLM_TEMPERATURE")
        try:
            vllm_temperature = (
                float(vllm_temperature_str)
                if vllm_temperature_str is not None
                else vllm_defaults.temperature
            )
        except ValueError:
            vllm_temperature = vllm_defaults.temperature

        # Classification schema from JSON
        classification_schema_path = os.getenv("CLASSIFICATION_SCHEMA_PATH")
        classification_schema = cls._load_classification_schema(
            classification_schema_path
        )

        return cls(
            whisper=WhisperSettings(
                model_path="/home/zerrougt/Call_Center_Ai_Agentic/models/whisper_dz_asr_large_chk750",
                max_length=448,
                max_new_tokens=444,
                language="ar",
                task="transcribe",
                chunk_length_seconds=30,
                chunk_overlap_seconds=2.0,
                device="auto",
                dtype="float16",
            ),
            gemini=GeminiSettings(
                api_key=gemini_key,
                model_name="gemini-2.0-flash-exp",
            ),
            qwen=LLMSettings(
                model_path="Qwen/Qwen-7B-Chat",
                trust_remote_code=True,
                max_length=512,
                temperature=0.7,
                device="auto",
                dtype="float16",
            ),
            vllm=VLLMSettings(
                base_url=vllm_base_url,
                model_name=vllm_model_name,
                api_key=vllm_api_key,
                temperature=vllm_temperature,
            ),
            dziribert_classifier=ModelSettings(
                model_path="alger-ia/dziribert",
                max_length=512,
                device="auto",
                dtype="float32",
            ),
            dziribert_sentiment=ModelSettings(
                model_path="alger-ia/dziribert_sentiment",
                max_length=512,
                device="auto",
                dtype="float32",
            ),
            classification=ClassificationSettings(
                **classification_schema
            )
            if classification_schema
            else ClassificationSettings(),
            pipeline=PipelineSettings(),
        )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.create_default()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set global settings instance."""
    global _settings
    _settings = settings


