import os
import time
import threading
import torch
from pathlib import Path
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
        
        # Validate model_path
        if not self.settings.model_path:
            raise ValueError("model_path is not set in WhisperSettings")
        
        load_path = self._find_latest_checkpoint(self.settings.model_path) or self.settings.model_path
        
        # Final validation that load_path is not None
        if not load_path:
            raise ValueError(f"Could not determine model path from: {self.settings.model_path}")
        
        # Determine base model for tokenizer and feature_extractor (required for checkpoint models)
        base_model = "openai/whisper-large"  # Default base model for checkpoint compatibility
        
        # Determine device and dtype
        if torch.cuda.is_available() and self.settings.device == "auto":
            self._device = "cuda:0"  # Use explicit cuda:0 like test script
        elif self.settings.device == "auto":
            self._device = "cpu"
        else:
            self._device = self.settings.device
        
        dtype = torch.float16 if "cuda" in str(self._device) else torch.float32

        self.logger.info(f"Loading Whisper model from: {load_path}")
        self.logger.info(f"Base model (tokenizer/feature_extractor): {base_model}")
        self.logger.info(f"Device: {self._device.upper()}, Dtype: {dtype}")
        self.logger.info(f"Chunk length: {self.settings.chunk_length_seconds}s, Stride: {self.settings.chunk_overlap_seconds}s")

        # Initialize pipeline with automatic chunking support
        # The pipeline will handle long files by chunking with stride internally
        # Note: tokenizer and feature_extractor are required for checkpoint models
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=load_path,
            tokenizer=base_model,  # Required for checkpoint models
            feature_extractor=base_model,  # Required for checkpoint models
            chunk_length_s=self.settings.chunk_length_seconds,
            stride_length_s=self.settings.chunk_overlap_seconds,
            device=self._device,
            dtype=dtype,  # Fixed: use dtype instead of deprecated torch_dtype
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
    
    def _save_audio_chunks(self, audio_path: str, output_dir: str = "data/chunks") -> list:
        """
        Split audio file into chunks and save them to disk.
        
        This creates chunks matching the pipeline's chunking parameters so that
        the saved chunks correspond to what the pipeline processes internally.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save chunk files
        
        Returns:
            List of paths to saved chunk files
        """
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get base filename without extension
        base_name = Path(audio_path).stem
        
        # Load audio file
        try:
            import librosa
            audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
        except ImportError:
            # Fallback to soundfile if librosa is not available
            try:
                import soundfile as sf
                audio_data, sr = sf.read(audio_path)
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                # Resample if needed
                if sr != 16000:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                    sr = 16000
            except ImportError:
                raise ImportError("Neither librosa nor soundfile is available. Please install one of them.")
        
        # Calculate chunk parameters (matching pipeline settings)
        chunk_length_samples = int(self.settings.chunk_length_seconds * sr)
        overlap_samples = int(self.settings.chunk_overlap_seconds * sr)
        step_samples = chunk_length_samples - overlap_samples
        
        if step_samples <= 0:
            self.logger.warning(f"Overlap ({self.settings.chunk_overlap_seconds}s) >= chunk length ({self.settings.chunk_length_seconds}s). Using no overlap.")
            step_samples = chunk_length_samples
            overlap_samples = 0
        
        # Calculate number of chunks
        total_samples = len(audio_data)
        num_chunks = max(1, (total_samples - overlap_samples + step_samples - 1) // step_samples) if step_samples > 0 else 1
        
        self.logger.info(f"💾 Saving {num_chunks} audio chunks to {output_dir}/")
        
        # Split and save chunks
        chunk_paths = []
        for i in range(num_chunks):
            start_idx = i * step_samples
            end_idx = min(start_idx + chunk_length_samples, total_samples)
            
            # Extract chunk
            chunk_audio = audio_data[start_idx:end_idx]
            
            # Generate chunk filename
            chunk_filename = f"{base_name}_chk{i+1:02d}.wav"
            chunk_path = output_path / chunk_filename
            
            # Save chunk
            try:
                import soundfile as sf
                sf.write(str(chunk_path), chunk_audio, sr)
            except ImportError:
                # Fallback: use scipy.io.wavfile if soundfile not available
                try:
                    from scipy.io import wavfile
                    wavfile.write(str(chunk_path), sr, chunk_audio)
                except ImportError:
                    raise ImportError("soundfile or scipy is required to save audio chunks. Please install one of them.")
            
            chunk_paths.append(str(chunk_path))
            self.logger.info(f"   💾 Saved: {chunk_filename} ({len(chunk_audio)/sr:.2f}s)")
        
        self.logger.info(f"✅ Saved {len(chunk_paths)} chunks to {output_dir}/")
        return chunk_paths

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
            
            # Save audio chunks to disk if audio is long enough to be chunked
            chunk_paths = []
            if audio_duration > self.settings.chunk_length_seconds:
                try:
                    chunk_paths = self._save_audio_chunks(audio_path, output_dir="data/chunks")
                except Exception as e:
                    self.logger.warning(f"Failed to save audio chunks: {e}. Continuing with transcription...")
            
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
                self.logger.info("🎤 Loading audio with librosa (ensures 16kHz sample rate)")
                
                # Load audio with librosa first (ensures 16kHz, critical for Whisper)
                # This matches the working test script approach
                import librosa
                audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
                self.logger.info(f"✅ Audio loaded: {len(audio_array)} samples at {sampling_rate}Hz")
                
                self.logger.info("🎤 Using pipeline automatic chunking with stride")
                
                # Pass the audio array to the pipeline (not the file path)
                # The pipeline will:
                # 1. Handle chunking with stride internally (chunk_length_s, stride_length_s)
                # 2. Merge overlapping chunks intelligently
                # 3. Return the complete transcript
                result = self._pipe(
                    audio_array,  # Pass array instead of file path (matches test script)
                    chunk_length_s=self.settings.chunk_length_seconds,
                    stride_length_s=self.settings.chunk_overlap_seconds,
                    return_timestamps=True,  # Helps keep track of context for long files
                    generate_kwargs={
                        "task": self.settings.task,
                        "repetition_penalty": 1.2,
                        "no_repeat_ngram_size": 4,
                        "do_sample": False,
                        "num_beams": 1,
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