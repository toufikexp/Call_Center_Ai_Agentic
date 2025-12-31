#!/usr/bin/env python3
"""
Main entry point for Call Center AI Agentic Pipeline.

Usage:
    python main.py <audio_path>

Environment Variables:
    GEMINI_API_KEY: Optional Gemini API key for transcript refinement
"""
import sys
import os
from src.pipeline import CallAnalysisPipeline
from src.core.config import get_settings, set_settings, Settings


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_path>")
        print("Example: python main.py ./data/audio_files/audio_file.mp3")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Load .env file if it exists (before getting settings)
    from src.core.config import load_env_file
    load_env_file()
    
    # Get Gemini API key from environment (now includes .env file)
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    # Configure settings
    settings = get_settings()
    if gemini_key:
        settings.gemini.api_key = gemini_key
        # Model name is set in config.py (default: gemini-2.0-flash-exp)
        print("[INFO] Gemini API key configured for transcript refinement")
    else:
        print("[WARNING] Gemini API key not set. Refinement step will be skipped.")
        print("[INFO] Set GEMINI_API_KEY environment variable to enable refinement")
    
    set_settings(settings)
    
    print(f"\n🚀 Starting call analysis for: {audio_path}")
    print("=" * 60)
    
    try:
        # Create and run pipeline
        pipeline = CallAnalysisPipeline(settings)
        result = pipeline.run(audio_path)
        
        print("\n✅ Pipeline execution completed successfully!")
        print(f"📊 Final status: {result['result'].status.value}")
        print(f"📝 Transcript length: {len(result['result'].transcript)} characters")
        print(f"🎯 Confidence: {result['result'].confidence_score:.3f}")
        print(f"📂 Subject: {result['result'].subject} / {result['result'].sub_subject}")
        print(f"😊 Satisfaction: {result['result'].satisfaction_score:.3f}")
        
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

