"""
Transcription service using Whisper (inspired by test script).
"""
import os
import torch
import librosa
import numpy as np
from typing import Tuple
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.core.base import BaseService, ServiceResult
from src.core.config import get_settings, WhisperSettings


class TranscriptionService(BaseService):
    """Service for audio transcription using Whisper."""
    
    def __init__(self, settings: WhisperSettings = None):
        super().__init__("transcription")
        self.settings = settings or get_settings().whisper
        self._processor = None
        self._model = None
        self._device = None
    
    def _find_latest_checkpoint(self, base_dir: str) -> str | None:
        """Find the latest checkpoint directory."""
        if not os.path.exists(base_dir):
            return None
        
        checkpoints = [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if "checkpoint-" in d and os.path.isdir(os.path.join(base_dir, d))
        ]
        
        if not checkpoints:
            return None
        
        return max(checkpoints, key=os.path.getmtime)
    
    def _get_device(self) -> str:
        """Determine device to use."""
        if self.settings.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.settings.device
    
    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype."""
        return torch.float16 if self.settings.dtype == "float16" else torch.float32
    
    def initialize(self) -> None:
        """Load Whisper model and processor."""
        if self._initialized:
            return
        
        # Determine model path (check for checkpoints first)
        model_path = self.settings.model_path
        checkpoint_path = self._find_latest_checkpoint(model_path)
        load_path = checkpoint_path if checkpoint_path else model_path
        
        if checkpoint_path:
            self.logger.info(f"🤖 Loading Model from: {checkpoint_path}")
        else:
            self.logger.info(f"🤖 Loading Model from: {load_path}")
        
        # Load processor and model
        try:
            self._processor = WhisperProcessor.from_pretrained(
                load_path,
                language=self.settings.language,
                task=self.settings.task
            )
            self._model = WhisperForConditionalGeneration.from_pretrained(load_path)
        except Exception as e:
            self.logger.error(f"❌ Error loading model from {load_path}: {e}")
            raise
        
        # Set device, dtype, and eval mode
        self._device = self._get_device()
        dtype = self._get_dtype()
        self._model.eval()
        
        # Move model to device with appropriate dtype
        if self._device == "cuda":
            self._model = self._model.to(self._device).to(dtype)
        else:
            self._model = self._model.to(self._device)
        
        self.logger.info(f"✅ Model loaded on: {self._device.upper()} ({dtype})")
        self._initialized = True
    
    def _transcribe_full(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Transcribe full audio in one go."""
        self.ensure_initialized()
        
        dtype = self._get_dtype()
        input_features = self._processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(self._device).to(dtype)
        
        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features,
                max_new_tokens=self.settings.max_new_tokens,
                num_beams=1,  # Use greedy decoding for speed (was 5, much faster)
                do_sample=False  # Deterministic decoding for speed
            )
        
        transcription = self._processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def _transcribe_chunked(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        chunk_length_s: int
    ) -> str:
        """Transcribe long audio by processing it in chunks with live output."""
        self.ensure_initialized()
        
        chunk_samples = int(chunk_length_s * sample_rate)
        total_chunks = (len(audio_array) + chunk_samples - 1) // chunk_samples
        full_text = []
        
        self.logger.info(f"✂️  Processing {total_chunks} chunks ({chunk_length_s}s each)...")
        self.logger.info("-" * 30)
        
        dtype = self._get_dtype()
        for i in range(0, len(audio_array), chunk_samples):
            chunk = audio_array[i:i + chunk_samples]
            chunk_idx = i // chunk_samples + 1
            
            input_features = self._processor(
                chunk,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features.to(self._device).to(dtype)
            
            with torch.no_grad():
                predicted_ids = self._model.generate(
                    input_features,
                    max_new_tokens=self.settings.max_new_tokens,
                    num_beams=1,  # Use greedy decoding for speed (was 5, much faster)
                    do_sample=False  # Deterministic decoding for speed
                    # return_dict_in_generate=True, # Add this*****************
                    # output_scores=True, # Add this*****************
                )
            
            chunk_text = self._processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            # Live output (matching test script)
            self.logger.info(f"[{chunk_idx}/{total_chunks}] 💬: {chunk_text}")
            full_text.append(chunk_text)
        
        full_transcript = " ".join(full_text)
        
        self.logger.info("=" * 40)
        self.logger.info("📝 FINAL FULL TRANSCRIPTION:")
        self.logger.info("-" * 40)
        self.logger.info(full_transcript)
        self.logger.info("=" * 40)
        
        return full_transcript
    
    def process(self, audio_path: str, sample_rate: int = 16000) -> ServiceResult:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate (default 16000)
        
        Returns:
            ServiceResult with transcript and confidence score
        """
        def _transcribe():
            # Load audio
            self.logger.info(f"📦 Loading audio: {os.path.basename(audio_path)}")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            audio_duration = len(audio) / sample_rate
            
            # Decide if we need chunking
            if audio_duration > self.settings.chunk_length_seconds:
                transcript = self._transcribe_chunked(
                    audio,
                    sample_rate,
                    self.settings.chunk_length_seconds
                )
            else:
                self.logger.info("🚀 Transcribing single short clip...")
                transcript = self._transcribe_full(audio, sample_rate)
                self.logger.info(f"💬 Result: {transcript}")
            
            # Calculate confidence (heuristic-based)
            confidence=self.recalculate_confidence(transcript)
            # transcript_length = len(transcript.split())
            # confidence = min(
            #     0.95,
            #     0.70 + (transcript_length / 100.0) * 0.25
            # ) if transcript_length > 0 else 0.0
            
            return {
                "transcript": transcript,
                "confidence": confidence
            }
        
        return self._execute_with_timing(_transcribe)
    
    def recalculate_confidence(self, transcript: str) -> float:
        """Recalculate confidence for a transcript."""
        transcript_length = len(transcript.split())
        confidence = min(
            0.95,
            0.70 + (transcript_length / 100.0) * 0.25
        ) if transcript_length > 0 else 0.0
        return confidence

