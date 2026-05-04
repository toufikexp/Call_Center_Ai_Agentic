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

    # PEFT / LoRA support. When `adapter_path` is set, the service loads the
    # base model from `base_model_id` and applies the adapter on top
    # (merging in memory). When unset, `model_path` is treated as a full
    # merged checkpoint and loaded directly.
    base_model_id: str = "openai/whisper-large-v3"
    adapter_path: Optional[str] = None

    # Per-segment inference. Larger batches give higher GPU throughput at the
    # cost of VRAM. Values >1 only matter once preprocessing is enabled.
    batch_size: int = 4

    # Optional 8-bit quantization via bitsandbytes (slower than fp16 on most
    # GPUs but helps fit on small VRAM). Off by default.
    use_8bit: bool = False


class PreprocessingSettings(BaseModel):
    """Settings for the audio preprocessing stage (channel split + VAD)."""

    enable: bool = True

    # Silero VAD parameters
    vad_threshold: float = 0.5
    min_segment_seconds: float = 1.0
    max_segment_seconds: float = 30.0
    padding_ms: int = 250

    # Stereo channel routing. For Ooredoo recordings the agent is on the
    # left channel and the client on the right.
    agent_channel: int = 0
    client_channel: int = 1

    # Optional debug dump of segments to disk under `data/segments/<basename>/`.
    save_segments: bool = False

    # Where to look for / cache the Silero VAD model. Set to a local path for
    # fully air-gapped deployments after one warm-up run.
    silero_cache_dir: Optional[str] = None


class LLMSettings(ModelSettings):
    """Settings for LLM models (Qwen, etc.)."""

    trust_remote_code: bool = True


class GeminiSettings(BaseModel):
    """Settings for Gemini API (temporary)."""

    api_key: Optional[str] = None
    # Stable Gemini model. The previous default `gemini-2.0-flash-exp` was the
    # experimental preview name and has been removed by Google. Override with
    # `GEMINI_MODEL_NAME` in `.env` to use a different model.
    model_name: str = "gemini-2.0-flash"
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
    preprocessing: PreprocessingSettings
    gemini: GeminiSettings
    qwen: LLMSettings
    vllm: VLLMSettings
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

        # Whisper / PEFT configuration from env with safe defaults
        whisper_defaults = WhisperSettings(model_path="")
        whisper_base_model_id = os.getenv(
            "WHISPER_BASE_MODEL_ID", whisper_defaults.base_model_id
        )
        whisper_model_path = os.getenv(
            "WHISPER_MODEL_PATH",
            "/home/zerrougt/UBC_NLP_Casablanca/whisper_large_gemini_fixed/checkpoint-250",
        )
        whisper_adapter_path = os.getenv("WHISPER_ADAPTER_PATH") or None
        try:
            whisper_batch_size = int(
                os.getenv("WHISPER_BATCH_SIZE", str(whisper_defaults.batch_size))
            )
        except ValueError:
            whisper_batch_size = whisper_defaults.batch_size
        whisper_use_8bit = os.getenv("WHISPER_USE_8BIT", "0").lower() in (
            "1",
            "true",
            "yes",
        )

        # Preprocessing configuration from env
        preprocessing_defaults = PreprocessingSettings()

        def _bool_env(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        def _float_env(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        def _int_env(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        return cls(
            whisper=WhisperSettings(
                model_path=whisper_model_path,
                base_model_id=whisper_base_model_id,
                adapter_path=whisper_adapter_path,
                batch_size=whisper_batch_size,
                use_8bit=whisper_use_8bit,
                max_length=448,
                max_new_tokens=444,
                language="ar",
                task="transcribe",
                chunk_length_seconds=30,
                chunk_overlap_seconds=2.0,
                device="auto",
                dtype="float16",
            ),
            preprocessing=PreprocessingSettings(
                enable=_bool_env("PREPROCESSING_ENABLE", preprocessing_defaults.enable),
                vad_threshold=_float_env(
                    "VAD_THRESHOLD", preprocessing_defaults.vad_threshold
                ),
                min_segment_seconds=_float_env(
                    "VAD_MIN_SEGMENT_SECONDS",
                    preprocessing_defaults.min_segment_seconds,
                ),
                max_segment_seconds=_float_env(
                    "VAD_MAX_SEGMENT_SECONDS",
                    preprocessing_defaults.max_segment_seconds,
                ),
                padding_ms=_int_env(
                    "VAD_PADDING_MS", preprocessing_defaults.padding_ms
                ),
                agent_channel=_int_env(
                    "PREPROCESSING_AGENT_CHANNEL",
                    preprocessing_defaults.agent_channel,
                ),
                client_channel=_int_env(
                    "PREPROCESSING_CLIENT_CHANNEL",
                    preprocessing_defaults.client_channel,
                ),
                save_segments=_bool_env(
                    "PREPROCESSING_SAVE_SEGMENTS",
                    preprocessing_defaults.save_segments,
                ),
                silero_cache_dir=os.getenv("SILERO_CACHE_DIR") or None,
            ),
            gemini=GeminiSettings(
                api_key=gemini_key,
                model_name=os.getenv(
                    "GEMINI_MODEL_NAME", GeminiSettings.model_fields["model_name"].default
                ),
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


