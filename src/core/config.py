"""
Configuration management using Pydantic settings.
"""
import os
from typing import Optional, Dict, List
from pathlib import Path
from pydantic import BaseModel, Field


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, looks for .env in current directory and project root.
    """
    if env_path is None:
        # Try current directory first
        current_dir = Path.cwd()
        env_path = current_dir / ".env"
        
        # If not found, try project root (parent of src/)
        if not env_path.exists():
            project_root = current_dir.parent if (current_dir / "src").exists() else current_dir
            env_path = project_root / ".env"
    
    env_path = Path(env_path)
    
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value
        except Exception as e:
            # Silently fail if .env file can't be read
            pass


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
    max_new_tokens: int = 444  # Reduced from 448 to account for decoder input IDs (4 tokens)


class LLMSettings(ModelSettings):
    """Settings for LLM models (Qwen, etc.)."""
    trust_remote_code: bool = True


class GeminiSettings(BaseModel):
    """Settings for Gemini API (temporary)."""
    api_key: Optional[str] = None
    model_name: str = "gemini-3-pro"
    timeout_seconds: int = 15


class ClassificationSettings(BaseModel):
    """Settings for classification."""
    primary_categories: List[str] = Field(
        default=[
            "BILLING & PAYMENTS",
            "ACCOUNT MANAGEMENT",
            "SALES & PRODUCT",
            "COMPLAINT",
            "OTHER"
        ]
    )
    category_descriptions: Dict[str, str] = Field(default_factory=dict)
    category_subcategories: Dict[str, List[str]] = Field(default_factory=dict)
    default_subcategory: Dict[str, str] = Field(default_factory=dict)
    other_category_name: str = "OTHER"
    other_category_threshold: float = 0.3


class PipelineSettings(BaseModel):
    """Pipeline execution settings."""
    confidence_threshold: float = 0.90
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
    dziribert_classifier: ModelSettings
    dziribert_sentiment: ModelSettings
    classification: ClassificationSettings
    pipeline: PipelineSettings

    @classmethod
    def create_default(cls) -> "Settings":
        """Create default settings."""
        # Load .env file if it exists
        load_env_file()
        
        # Load from environment or use defaults
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        return cls(
            whisper=WhisperSettings(
            model_path="/home/zerrougt/Call_Center_Ai_Agentic/models/whisper_dz_asr",
                # model_path="microsoft/Phi-4-multimodal-instruct",
                max_length=448,
                max_new_tokens=444,  # Reduced to account for decoder input IDs
                language="ar",
                task="transcribe",
                chunk_length_seconds=30,
                device="auto",
                dtype="float16"
            ),
            gemini=GeminiSettings(
                api_key=gemini_key,
                model_name="gemini-3-pro"
            ),
            qwen=LLMSettings(
                model_path="Qwen/Qwen-7B-Chat",
                trust_remote_code=True,
                max_length=512,
                temperature=0.7,
                device="auto",
                dtype="float16"
            ),
            dziribert_classifier=ModelSettings(
                model_path="alger-ia/dziribert",
                max_length=512,
                device="auto",
                dtype="float32"
            ),
            dziribert_sentiment=ModelSettings(
                model_path="alger-ia/dziribert_sentiment",
                max_length=512,
                device="auto",
                dtype="float32"
            ),
            classification=ClassificationSettings(
                primary_categories=[
                    "BILLING & PAYMENTS",
                    "ACCOUNT MANAGEMENT",
                    "SALES & PRODUCT",
                    "COMPLAINT",
                    "OTHER"
                ],
                category_descriptions={
                    "BILLING & PAYMENTS": "All inquiries related to outstanding balance, invoice breakdown, credit/debit charges, payment arrangements, or statement clarification.",
                    "ACCOUNT MANAGEMENT": "Requests to change the subscription plan, update personal details, activate/deactivate a service, line suspension/resumption, OR requesting simple technical information/settings (e.g., APN settings, Voicemail number).",
                    "SALES & PRODUCT": "Inquiries about pricing for new plans, available devices, active promotions, or intent to purchase new services/upgrades.",
                    "COMPLAINT": "Any expression of strong dissatisfaction with service quality, a product, a fee, or a previous company interaction that requires rectification.",
                    "OTHER": "Any call that does not fit into the predefined categories above."
                },
                category_subcategories={
                    "COMPLAINT": [
                        "Network Coverage",
                        "Data Speed",
                        "Activation Failure",
                        "Incorrect Charges",
                        "Billing Error",
                        "Refund/Credit Issue",
                        "Resolution Failure",
                        "Agent Interaction",
                        "Process Delay"
                    ]
                },
                default_subcategory={
                    "COMPLAINT": "Network Coverage"
                }
            ),
            pipeline=PipelineSettings()
        )


# Global settings instance
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

