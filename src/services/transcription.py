"""
Transcription service.

Loads Whisper (optionally with a LoRA adapter merged in memory) and
transcribes a list of pre-segmented audio clips produced by
`PreprocessingService`. One `model.generate(...)` call per batch of
segments — no long-form chunking, because each segment is already ≤ 30 s
and aligned with how the adapter was trained.

Confidence per segment is `exp(mean token log-prob)` from
`output_scores=True`, averaged across all segments to produce the
call-level `confidence_score`.
"""
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.core.base import BaseService, ServiceResult
from src.core.config import get_settings, WhisperSettings


_TARGET_SR = 16000


class TranscriptionService(BaseService):
    """Per-segment Whisper transcription with optional PEFT/LoRA adapter."""

    def __init__(self, settings: Optional[WhisperSettings] = None):
        super().__init__("transcription")
        self.settings = settings or get_settings().whisper
        self._model: Optional[WhisperForConditionalGeneration] = None
        self._processor: Optional[WhisperProcessor] = None
        self._device: Optional[str] = None
        self._dtype: Optional[torch.dtype] = None
        self._forced_decoder_ids = None
        self._adapter_version: str = ""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        if self._initialized:
            return

        self._device = self._resolve_device()
        self._dtype = self._resolve_dtype(self._device)
        self.logger.info(
            f"Device: {self._device}, dtype: {self._dtype}, "
            f"batch_size: {self.settings.batch_size}, use_8bit: {self.settings.use_8bit}"
        )

        adapter_path = self.settings.adapter_path
        base_id_from_adapter = self._read_adapter_base_id(adapter_path) if adapter_path else None

        if adapter_path and base_id_from_adapter and base_id_from_adapter != self.settings.base_model_id:
            self.logger.warning(
                f"Adapter was trained on '{base_id_from_adapter}' but settings.base_model_id "
                f"is '{self.settings.base_model_id}'. Loading the adapter on a mismatched "
                f"base will produce garbage. Aligning to the adapter's base."
            )
            base_model_id = base_id_from_adapter
        else:
            base_model_id = self.settings.base_model_id

        # ------------------------------------------------------------------
        # Load model
        # ------------------------------------------------------------------
        if adapter_path:
            self.logger.info(f"Loading base Whisper: {base_model_id}")
            base = self._load_base(base_model_id)
            self.logger.info(f"Applying LoRA adapter from: {adapter_path}")
            try:
                from peft import PeftModel
            except ImportError as e:
                raise RuntimeError(
                    "peft is required to load a LoRA adapter. "
                    "Install with: pip install peft"
                ) from e
            model = PeftModel.from_pretrained(base, adapter_path)
            model = model.merge_and_unload()
            self._adapter_version = os.path.basename(os.path.normpath(adapter_path))
        else:
            # Path A artifact: a full merged checkpoint. `model_path` is the
            # directory; tokenizer/feature extractor come from `base_model_id`
            # so they always match the audio domain.
            load_path = self.settings.model_path or base_model_id
            self.logger.info(f"Loading merged Whisper checkpoint: {load_path}")
            model = self._load_base(load_path)
            self._adapter_version = ""

        if self._device != "cpu" and not self.settings.use_8bit:
            model = model.to(self._device, dtype=self._dtype)
        model.eval()
        self._model = model

        # ------------------------------------------------------------------
        # Processor (tokenizer + feature extractor) always from the base.
        # ------------------------------------------------------------------
        self.logger.info(f"Loading processor: {base_model_id}")
        self._processor = WhisperProcessor.from_pretrained(
            base_model_id,
            language=self.settings.language,
            task=self.settings.task,
        )
        try:
            self._forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=self.settings.language,
                task=self.settings.task,
            )
        except Exception as e:  # pragma: no cover — depends on base model
            self.logger.warning(f"Could not derive forced_decoder_ids: {e}")
            self._forced_decoder_ids = None

        self._initialized = True
        self.logger.info(
            f"✅ Transcription ready (adapter_version={self._adapter_version or '<merged>'})"
        )

    def _load_base(self, model_id_or_path: str) -> WhisperForConditionalGeneration:
        kwargs: Dict[str, Any] = {}
        if self.settings.use_8bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                kwargs["device_map"] = "auto"
            except ImportError as e:
                raise RuntimeError(
                    "use_8bit=True requires bitsandbytes. "
                    "Install with: pip install bitsandbytes"
                ) from e
        else:
            kwargs["torch_dtype"] = self._dtype
        return WhisperForConditionalGeneration.from_pretrained(
            model_id_or_path, **kwargs
        )

    @staticmethod
    def _read_adapter_base_id(adapter_path: str) -> Optional[str]:
        cfg_path = Path(adapter_path) / "adapter_config.json"
        if not cfg_path.exists():
            return None
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            value = cfg.get("base_model_name_or_path")
            return str(value) if value else None
        except Exception:
            return None

    def _resolve_device(self) -> str:
        if self.settings.use_8bit:
            # bitsandbytes manages device placement via device_map="auto"
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.settings.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.settings.device

    def _resolve_dtype(self, device: str) -> torch.dtype:
        if device == "cpu":
            return torch.float32
        if self.settings.dtype == "float16":
            return torch.float16
        if self.settings.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def process(self, segments: List[Dict[str, Any]]) -> ServiceResult:
        """Transcribe a list of preprocessing segments.

        Each input segment is a dict with keys: channel, start_ms, end_ms,
        audio (np.ndarray, float32, 16 kHz mono).

        Returns ServiceResult with data:
            {
                "transcript": str,           # reconstructed dialogue
                "segments":   list[dict],    # per-segment text + confidence
                "confidence": float,         # mean per-segment confidence
                "adapter_version": str,
            }
        """

        def _run():
            self.ensure_initialized()

            if not segments:
                self.logger.warning("No segments to transcribe")
                return {
                    "transcript": "",
                    "segments": [],
                    "confidence": 0.0,
                    "adapter_version": self._adapter_version,
                }

            transcribed: List[Dict[str, Any]] = []
            batch_size = max(1, int(self.settings.batch_size))

            for batch_start in range(0, len(segments), batch_size):
                batch = segments[batch_start : batch_start + batch_size]
                batch_results = self._transcribe_batch(batch)
                transcribed.extend(batch_results)

            transcript = self._reconstruct_dialogue(transcribed)
            confidences = [s["confidence"] for s in transcribed if s["confidence"] > 0]
            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            self.logger.info("=" * 60)
            self.logger.info("📝 TRANSCRIPTION RESULT")
            self.logger.info("=" * 60)
            self.logger.info(f"Segments transcribed: {len(transcribed)}")
            self.logger.info(f"Mean confidence: {avg_confidence:.3f}")
            preview = transcript[:200] + ("..." if len(transcript) > 200 else "")
            self.logger.info(f"Preview: {preview}")
            self.logger.info("=" * 60)

            return {
                "transcript": transcript,
                "segments": transcribed,
                "confidence": avg_confidence,
                "adapter_version": self._adapter_version,
            }

        return self._execute_with_timing(_run)

    def _transcribe_batch(
        self, batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run one batched generate over the given segments. On CUDA OOM,
        falls back to halving the batch size recursively until size 1."""
        try:
            return self._do_batch(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                self.logger.error(
                    "CUDA OOM on a single segment — segment may be too long. Skipping."
                )
                return [self._failed_segment(batch[0], "CUDA OOM on single segment")]
            mid = len(batch) // 2
            self.logger.warning(
                f"CUDA OOM on batch of {len(batch)}; retrying as 2 batches of {mid} / {len(batch) - mid}"
            )
            return self._transcribe_batch(batch[:mid]) + self._transcribe_batch(batch[mid:])

    def _do_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        audios = [seg["audio"] for seg in batch]
        features = self._processor(
            audios,
            sampling_rate=_TARGET_SR,
            return_tensors="pt",
            padding=True,
        ).input_features
        features = features.to(self._device, dtype=self._dtype)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 4,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if self._forced_decoder_ids is not None:
            gen_kwargs["forced_decoder_ids"] = self._forced_decoder_ids

        with torch.no_grad():
            output = self._model.generate(features, **gen_kwargs)

        sequences = output.sequences  # [B, full_len]
        scores = getattr(output, "scores", None)

        # Per-token log-probs of the chosen tokens (generated portion only)
        per_segment_confidence = self._batch_confidences(sequences, scores)

        # Decode every sequence in the batch
        texts = self._processor.batch_decode(sequences, skip_special_tokens=True)

        results: List[Dict[str, Any]] = []
        for seg, text, conf in zip(batch, texts, per_segment_confidence):
            results.append({
                "channel": seg["channel"],
                "start_ms": int(seg["start_ms"]),
                "end_ms": int(seg["end_ms"]),
                "text": text.strip(),
                "confidence": float(conf),
            })
        return results

    def _batch_confidences(
        self,
        sequences: torch.Tensor,
        scores: Optional[Tuple[torch.Tensor, ...]],
    ) -> List[float]:
        """Return per-row `exp(mean token log-prob)` over the generated tail."""
        if not scores:
            return [0.0] * sequences.size(0)

        stacked = torch.stack(scores, dim=1).float()  # [B, T, V]
        log_probs = torch.log_softmax(stacked, dim=-1)
        gen_len = stacked.size(1)
        gen_ids = sequences[:, -gen_len:]
        token_logprobs = log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]

        tokenizer = self._processor.tokenizer
        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)

        mask = torch.ones_like(gen_ids, dtype=torch.bool)
        if eos_id is not None:
            mask &= gen_ids != eos_id
        if pad_id is not None:
            mask &= gen_ids != pad_id

        confidences: List[float] = []
        for row_logprobs, row_mask in zip(token_logprobs, mask):
            if row_mask.any():
                avg = (row_logprobs * row_mask).sum() / row_mask.sum()
            elif row_logprobs.numel() > 0:
                avg = row_logprobs.mean()
            else:
                confidences.append(0.0)
                continue
            confidences.append(max(0.0, min(1.0, math.exp(float(avg.item())))))
        return confidences

    @staticmethod
    def _failed_segment(seg: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "channel": seg["channel"],
            "start_ms": int(seg["start_ms"]),
            "end_ms": int(seg["end_ms"]),
            "text": "",
            "confidence": 0.0,
        }

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _channel_label(channel: str) -> str:
        if channel == "agent":
            return "[Agent]"
        if channel == "client":
            return "[Subscriber]"
        return "[Speaker]"

    @classmethod
    def _reconstruct_dialogue(cls, segments: List[Dict[str, Any]]) -> str:
        """Concatenate non-empty segments in chronological order with
        speaker labels and timestamps."""
        lines: List[str] = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            mm, ss = divmod(int(seg["start_ms"]) // 1000, 60)
            timestamp = f"{mm:02d}:{ss:02d}"
            lines.append(f"{timestamp} {cls._channel_label(seg['channel'])}: {text}")
        return "\n".join(lines)
