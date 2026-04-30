"""
Audio preprocessing service.

Takes the raw call audio and produces a chronologically-ordered list of
VAD-cut speech segments, each tagged with a channel (agent / client / unknown).
This makes inference shape match the training distribution (the LoRA adapter
was fine-tuned on per-channel VAD-cut clips ≤ 30 s) and gives downstream
stages free, deterministic speaker labels.

Pipeline per call:
    1. Load audio with `soundfile` (multi-channel, native sample rate).
    2. Resample each channel to 16 kHz mono.
    3. Run Silero VAD per channel.
    4. Filter by min/max duration, pad each segment, slice the waveform.
    5. Sort the resulting segments chronologically (across channels).

The service does NOT modify the raw input file. Optional debug WAV dumps are
written under `data/segments/<basename>/` when `save_segments=True`.
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.core.base import BaseService, ServiceResult
from src.core.config import get_settings, PreprocessingSettings


_TARGET_SR = 16000


class PreprocessingService(BaseService):
    """Stereo channel split + Silero VAD segmentation."""

    def __init__(self, settings: Optional[PreprocessingSettings] = None):
        super().__init__("preprocessing")
        self.settings = settings or get_settings().preprocessing
        self._vad_model = None
        self._get_speech_timestamps = None

    def initialize(self) -> None:
        """Load Silero VAD via torch.hub (cached after first run)."""
        if self._initialized:
            return

        import torch  # imported lazily to keep service construction cheap

        if self.settings.silero_cache_dir:
            torch.hub.set_dir(self.settings.silero_cache_dir)

        self.logger.info("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        # `utils` is (get_speech_timestamps, save_audio, read_audio,
        # VADIterator, collect_chunks). We only need the first.
        self._get_speech_timestamps = utils[0]
        self._vad_model = model
        self._initialized = True
        self.logger.info("✅ Silero VAD loaded")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, audio_path: str) -> ServiceResult:
        """
        Returns a ServiceResult whose `data["segments"]` is a list of dicts:
            {"channel": "agent"|"client"|"unknown",
             "start_ms": int, "end_ms": int,
             "audio": np.ndarray (float32, 16 kHz, mono)}
        """

        def _run():
            self.ensure_initialized()

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            channels = self._load_and_split(audio_path)
            self.logger.info(
                f"📁 {os.path.basename(audio_path)}: {len(channels)} channel(s), "
                f"{[lbl for lbl, _ in channels]}"
            )

            all_segments: List[dict] = []
            for channel_label, audio in channels:
                segs = self._vad_segments(audio)
                self.logger.info(
                    f"   🔊 channel={channel_label}: {len(segs)} segment(s) after VAD"
                )
                for start_ms, end_ms, segment_audio in segs:
                    all_segments.append({
                        "channel": channel_label,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "audio": segment_audio,
                    })

            all_segments.sort(key=lambda s: s["start_ms"])

            if self.settings.save_segments:
                self._dump_debug_segments(audio_path, all_segments)

            self.logger.info("=" * 60)
            self.logger.info("🎙️ PREPROCESSING RESULT")
            self.logger.info("=" * 60)
            self.logger.info(f"Total segments: {len(all_segments)}")
            if all_segments:
                total_speech_ms = sum(s["end_ms"] - s["start_ms"] for s in all_segments)
                self.logger.info(f"Total speech duration: {total_speech_ms / 1000:.1f}s")
            self.logger.info("=" * 60)

            return {"segments": all_segments}

        return self._execute_with_timing(_run)

    # ------------------------------------------------------------------
    # Audio loading + channel split
    # ------------------------------------------------------------------
    def _load_and_split(self, audio_path: str) -> List[Tuple[str, np.ndarray]]:
        """Load audio and return [(channel_label, mono_16k_audio), ...].

        Stereo (or more) → split into agent/client. Mono → single 'unknown'
        channel. Anything beyond stereo is reduced to the configured agent
        and client channel indices.
        """
        import soundfile as sf
        import librosa

        audio, sr = sf.read(audio_path, always_2d=True, dtype="float32")
        # `audio` shape: (num_samples, num_channels)
        num_channels = audio.shape[1]

        def _resample(mono: np.ndarray) -> np.ndarray:
            if sr == _TARGET_SR:
                return mono.astype(np.float32, copy=False)
            return librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=_TARGET_SR)

        if num_channels == 1:
            return [("unknown", _resample(audio[:, 0]))]

        agent_idx = self.settings.agent_channel
        client_idx = self.settings.client_channel
        # Defensive bounds — in case someone configures channels that don't exist
        if agent_idx >= num_channels or client_idx >= num_channels:
            self.logger.warning(
                f"Configured agent_channel={agent_idx}, client_channel={client_idx} "
                f"but audio has only {num_channels} channels. Falling back to channels 0/1."
            )
            agent_idx = 0
            client_idx = min(1, num_channels - 1)

        agent = _resample(audio[:, agent_idx])
        client = _resample(audio[:, client_idx])
        return [("agent", agent), ("client", client)]

    # ------------------------------------------------------------------
    # VAD
    # ------------------------------------------------------------------
    def _vad_segments(
        self, audio: np.ndarray
    ) -> List[Tuple[int, int, np.ndarray]]:
        """Run Silero VAD on a mono 16 kHz waveform and return
        (start_ms, end_ms, segment_audio) tuples after duration filtering
        and padding."""
        import torch

        if audio.size == 0:
            return []

        wav = torch.from_numpy(np.ascontiguousarray(audio)).float()

        timestamps = self._get_speech_timestamps(
            wav,
            self._vad_model,
            sampling_rate=_TARGET_SR,
            threshold=self.settings.vad_threshold,
        )
        if not timestamps:
            return []

        padding_samples = int(self.settings.padding_ms / 1000 * _TARGET_SR)
        min_samples = int(self.settings.min_segment_seconds * _TARGET_SR)
        max_samples = int(self.settings.max_segment_seconds * _TARGET_SR)
        total_samples = audio.shape[0]

        segments: List[Tuple[int, int, np.ndarray]] = []
        for ts in timestamps:
            start = max(0, int(ts["start"]) - padding_samples)
            end = min(total_samples, int(ts["end"]) + padding_samples)
            length = end - start
            if length < min_samples or length > max_samples:
                continue
            segment = audio[start:end].astype(np.float32, copy=False)
            start_ms = int(start / _TARGET_SR * 1000)
            end_ms = int(end / _TARGET_SR * 1000)
            segments.append((start_ms, end_ms, segment))

        return segments

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def _dump_debug_segments(self, audio_path: str, segments: List[dict]) -> None:
        """Write each segment to `data/segments/<basename>/` as 16 kHz WAV."""
        try:
            import soundfile as sf
        except ImportError:
            self.logger.warning("soundfile not available; cannot dump debug segments")
            return

        basename = os.path.splitext(os.path.basename(audio_path))[0]
        out_dir = Path("data/segments") / basename
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, seg in enumerate(segments, start=1):
            fname = f"{basename}_{seg['channel']}_{i:03d}_start_{seg['start_ms']}ms.wav"
            sf.write(str(out_dir / fname), seg["audio"], _TARGET_SR)

        self.logger.info(f"💾 Saved {len(segments)} segments to {out_dir}/")
