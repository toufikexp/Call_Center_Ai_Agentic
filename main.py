#!/usr/bin/env python3
"""
Main entry point for Call Center AI Agentic Pipeline.

Usage:
    python main.py <audio_path>

Environment Variables:
    GEMINI_API_KEY: Optional Gemini API key for transcript refinement
    HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE: skip Hub network calls
"""
import os
import sys
from pathlib import Path


def _bootstrap_env() -> None:
    """Load `.env` *before* importing anything heavyweight.

    `transformers` and `huggingface_hub` snapshot env vars like
    `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` at import time. The regular
    loader inside `Settings.create_default()` runs too late — by then those
    libraries have already decided "we're online" and will hit the Hub on
    every `from_pretrained`. Doing the env load here, before any other
    import, keeps offline mode honest.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Defer error reporting to the regular loader; we don't want a
        # bootstrap problem to mask the real one downstream.
        pass


_bootstrap_env()

# Imports below this line are safe — env vars are populated.
from src.pipeline import CallAnalysisPipeline  # noqa: E402
from src.config.config import get_settings, set_settings, Settings  # noqa: E402


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_path>")
        print("Example: python main.py ./data/audio_files/audio_file.mp3")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Get settings (handles .env loading and configuration internally)
    settings = get_settings()
    set_settings(settings)

    print(f"\n🚀 Starting call analysis for: {audio_path}")
    print("=" * 60)

    try:
        # Create and run pipeline
        pipeline = CallAnalysisPipeline(settings)
        result = pipeline.run(audio_path)

        final_status = result['result'].status.value

        # Only print success if status is COMPLETE
        if final_status == "COMPLETE":
            print("\n✅ Pipeline execution completed successfully!")
        elif final_status == "ERROR":
            print("\n❌ Pipeline execution failed with errors!")
            if result['result'].error_message:
                print(f"Error: {result['result'].error_message}")
        else:
            print(f"\n⚠️ Pipeline execution completed with status: {final_status}")

        print(f"📊 Final status: {final_status}")
        print(f"📝 Transcript length: {len(result['result'].transcript)} characters")
        print(f"🎯 Confidence: {result['result'].confidence_score:.3f}")
        print(f"✨ Refinement Score: {result['result'].refinement_score:.3f}")
        print(f"📂 Subject: {result['result'].subject} / {result['result'].sub_subject}")
        print(f"😊 Satisfaction: {result['result'].satisfaction_score:.3f}")

        # Exit with error code if pipeline failed
        if final_status == "ERROR":
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
