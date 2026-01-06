import os
import time
import threading
import torch
from transformers import pipeline

from src.core.base import BaseService, ServiceResult
from src.core.config import get_settings, WhisperSettings


class TranscriptionService(BaseService):
    """
    Transcription Service using Hugging Face Pipeline with automatic chunking.
    
    Inspired by: https://huggingface.co/blog/asr-chunking
    
    The pipeline handles long-form transcription automatically using:
    - chunk_length_s: Maximum chunk size (30s)
    - stride_length_s: Overlap between chunks (5s) for better context
    
    This approach leverages CTC/Whisper's architecture to merge overlapping
    chunks intelligently, avoiding boundary artifacts.
    """
    
    def __init__(self, settings: WhisperSettings = None):
        super().__init__("transcription")
        self.settings = settings or get_settings().whisper
        self._pipe = None
        self._device = None

    def _find_latest_checkpoint(self, base_dir: str) -> str | None:
        """Find the latest checkpoint directory."""
        if not os.path.exists(base_dir):
            return None
        checkpoints = [
            os.path.join(base_dir, d) for d in os.listdir(base_dir)
            if "checkpoint-" in d and os.path.isdir(os.path.join(base_dir, d))
        ]
        return max(checkpoints, key=os.path.getmtime) if checkpoints else None

    def initialize(self) -> None:
        """Initialize the Hugging Face Pipeline with automatic chunking."""
        if self._initialized:
            return
        
        load_path = self._find_latest_checkpoint(self.settings.model_path) or self.settings.model_path
        self._device = "cuda" if torch.cuda.is_available() and self.settings.device == "auto" else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        self.logger.info(f"Loading Whisper model from: {load_path}")
        self.logger.info(f"Device: {self._device.upper()}, Dtype: {dtype}")
        self.logger.info(f"Chunk length: {self.settings.chunk_length_seconds}s, Stride: {self.settings.chunk_overlap_seconds}s")

        # Initialize pipeline with automatic chunking support
        # The pipeline will handle long files by chunking with stride internally
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=load_path,
            chunk_length_s=self.settings.chunk_length_seconds,
            stride_length_s=self.settings.chunk_overlap_seconds,
            device=self._device,
            torch_dtype=dtype,
            return_timestamps=False,  # Disable timestamps for faster processing
        )
        self._initialized = True
        self.logger.info(f"✅ Whisper Pipeline initialized on {self._device.upper()} with automatic chunking")

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration in seconds."""
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            return duration
        except ImportError:
            # librosa not available, try soundfile
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                return info.duration
            except Exception:
                return 0.0
        except Exception:
            # Other error, try soundfile as fallback
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                return info.duration
            except Exception:
                return 0.0

    def _show_progress(self, stop_event: threading.Event, audio_duration: float):
        """Show periodic progress updates during transcription."""
        start_time = time.time()
        elapsed = 0
        update_interval = 7  # Show progress every 5 seconds
        
        # Estimate number of chunks
        chunk_size = self.settings.chunk_length_seconds
        estimated_chunks = max(1, int(audio_duration / chunk_size) + 1)
        
        self.logger.info(f"📊 Audio duration: {audio_duration:.1f}s")
        self.logger.info(f"📦 Estimated chunks: ~{estimated_chunks} (chunk size: {chunk_size}s)")
        self.logger.info("🔄 Starting transcription...")
        
        while not stop_event.is_set():
            time.sleep(update_interval)
            if stop_event.is_set():
                break
            
            elapsed = time.time() - start_time
            progress_pct = min(100, (elapsed / max(audio_duration * 0.3, 1)) * 100)  # Rough estimate
            self.logger.info(f"⏳ Processing... ({elapsed:.0f}s elapsed, ~{progress_pct:.0f}% estimated)")

    def process(self, audio_path: str, sample_rate: int = 16000) -> ServiceResult:
        """
        Transcribe audio file using pipeline with automatic chunking.
        
        The pipeline handles:
        - Long-form audio via internal chunking with stride
        - Overlapping chunks for better context
        - Automatic merging of chunk outputs
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate (pipeline handles conversion automatically)
        
        Returns:
            ServiceResult with transcript and confidence score
        """
        def _execute():
            self.ensure_initialized()
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            self.logger.info(f"📁 Loading audio: {os.path.basename(audio_path)}")
            
            # Get audio duration for progress estimation
            audio_duration = self._get_audio_duration(audio_path)
            
            # Start progress indicator in background thread
            stop_progress = threading.Event()
            progress_thread = None
            if audio_duration > 0:
                progress_thread = threading.Thread(
                    target=self._show_progress,
                    args=(stop_progress, audio_duration),
                    daemon=True
                )
                progress_thread.start()
            
            try:
                self.logger.info("🎤 Using pipeline automatic chunking with stride")
                
                # Pass the audio file directly to the pipeline
                # The pipeline will:
                # 1. Load the audio
                # 2. Handle chunking with stride internally (chunk_length_s, stride_length_s)
                # 3. Merge overlapping chunks intelligently
                # 4. Return the complete transcript
                result = self._pipe(
                    audio_path,
                    generate_kwargs={
                        "language": self.settings.language,
                        "task": self.settings.task
                    }
                )
                
                # Stop progress indicator
                stop_progress.set()
                if progress_thread:
                    progress_thread.join(timeout=1)
                
                full_text = result["text"].strip() if result.get("text") else ""
                
                self.logger.info("=" * 60)
                self.logger.info("📝 TRANSCRIPTION RESULT")
                self.logger.info("=" * 60)
                if full_text:
                    # Show preview if transcript is long
                    preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
                    self.logger.info(f"Transcript preview: {preview}")
                    self.logger.info(f"Full length: {len(full_text)} characters, {len(full_text.split())} words")
                else:
                    self.logger.info("(Empty transcript)")
                self.logger.info("=" * 60)
                
                return {
                    "transcript": full_text,
                    "confidence": self.recalculate_confidence(full_text)
                }
            except Exception as e:
                # Stop progress indicator on error
                stop_progress.set()
                if progress_thread:
                    progress_thread.join(timeout=1)
                raise

        return self._execute_with_timing(_execute)
    
    
    def recalculate_confidence(self, transcript: str) -> float:
        """Recalculate confidence for a transcript."""
        words = transcript.split()
        if not words:
            return 0.0
        return min(0.95, 0.70 + (len(words) / 100.0) * 0.25)