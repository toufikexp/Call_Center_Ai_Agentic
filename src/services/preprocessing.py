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
import threading
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
        # Silero VAD is stateful (LSTM hidden state); concurrent inference on
        # the same model instance segfaults under threads. Serialise model
        # calls; the rest of the work (audio load, post-processing, file
        # writes) still overlaps across workers.
        self._vad_lock = threading.Lock()

    def initialize(self) -> None:
        """Load Silero VAD from the bundled `silero-vad` pip package.

        We deliberately avoid `torch.hub.load(...)` — even with a populated
        cache it does a GitHub HEAD call to check for updates, which adds
        ~2 minutes when the network is unreachable. The `silero-vad` pip
        package ships the model weights inside the wheel, so this load is
        purely local and offline-safe.
        """
        if self._initialized:
            return

        if self.settings.silero_cache_dir:
            # Honoured for backwards compat — only matters if a future
            # change re-enables `torch.hub` for some reason.
            import torch  # local import to keep service construction cheap
            torch.hub.set_dir(self.settings.silero_cache_dir)

        self.logger.info("Loading Silero VAD model (silero-vad package)...")
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
        except ImportError as e:
            raise RuntimeError(
                "silero-vad package is required. Install with: "
                "pip install silero-vad"
            ) from e

        self._vad_model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps
        self._initialized = True
        self.logger.info("✅ Silero VAD loaded")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, audio_path: str) -> ServiceResult:
        """
        Returns a ServiceResult whose `data` contains:
            {
                "segments":         list of {"channel", "start_ms", "end_ms", "audio"},
                "audio_duration_s": float,   # original audio duration in seconds
                "channel_count":    int,     # 1 (mono) or 2 (stereo); >2 reduced to 2
            }
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

            # Audio duration: every channel array is the same length after
            # resampling, so the duration is len(channel) / target_sr.
            if channels:
                audio_duration_s = float(len(channels[0][1])) / float(_TARGET_SR)
            else:
                audio_duration_s = 0.0
            channel_count = len(channels)

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
            self.logger.info(f"Audio duration: {audio_duration_s:.1f}s ({channel_count} channel)")
            self.logger.info(f"Total segments: {len(all_segments)}")
            if all_segments:
                total_speech_ms = sum(s["end_ms"] - s["start_ms"] for s in all_segments)
                self.logger.info(f"Total speech duration: {total_speech_ms / 1000:.1f}s")
            self.logger.info("=" * 60)

            return {
                "segments": all_segments,
                "audio_duration_s": audio_duration_s,
                "channel_count": channel_count,
            }

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
        import librosa

        audio, sr = self._safe_load_multichannel(audio_path)
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

    def _safe_load_multichannel(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio as (num_samples, num_channels) at native sample rate.

        Tries `soundfile` first (fast, exact for clean WAV/FLAC). On any
        failure falls back to `librosa.load(..., mono=False)` which uses
        audioread / ffmpeg under the hood — much more permissive about
        malformed WAV headers, mp3, m4a, etc.
        """
        import soundfile as sf

        try:
            audio, sr = sf.read(audio_path, always_2d=True, dtype="float32")
            return audio, int(sr)
        except Exception as e:
            self.logger.warning(
                f"soundfile could not open {os.path.basename(audio_path)} "
                f"({e}); falling back to librosa+audioread"
            )

        import librosa

        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        # librosa returns (num_samples,) for mono and (num_channels, num_samples)
        # for multi-channel. Normalise to (num_samples, num_channels).
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        else:
            audio = audio.T
        return audio.astype(np.float32, copy=False), int(sr)

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

        # Serialise model inference: Silero VAD's internal LSTM state is
        # mutated during a forward pass, so concurrent calls from threads
        # corrupt it and segfault on the C++ side.
        with self._vad_lock:
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
