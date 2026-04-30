# ADR 0006 — Channel split + Silero VAD before transcription, PEFT/LoRA at runtime

- Status: Accepted

## Context

The fine-tuned Whisper Large v3 model is trained on per-channel,
VAD-segmented clips of 1–30 s, exported as a LoRA adapter (~35 MB:
`adapter_config.json` + `adapter_model.safetensors`).

The previous transcription pipeline:

- Loaded a full Whisper checkpoint via `transformers.pipeline(...)`. It had
  no path for adapter-only checkpoints.
- Collapsed stereo recordings to mono via `librosa.load(...)`'s default
  downmix, putting agent and client speech into the same waveform.
- Used the HF pipeline's blind 30 s sliding window with 2 s stride for
  long-form audio.

These three choices created a strong train/inference distribution mismatch:
the model never saw mixed-speaker audio or arbitrary 30 s windows during
training. WER measured on training-shaped eval data (27 %) did not represent
production quality.

## Decision

### A. Insert a `PreprocessingService` before transcription

`src/services/preprocessing.py` performs, per call:

1. Multi-channel load with `soundfile`, resample each channel to 16 kHz mono.
2. Stereo → split into `agent` (left) and `client` (right). Mono → single
   `unknown` channel.
3. Silero VAD per channel.
4. Filter by `min_segment_seconds` / `max_segment_seconds`, pad by
   `padding_ms` on each side.
5. Sort segments chronologically across channels.

The orchestrator runs this as a new `preprocess` node before `transcribe`.
A failure or "no speech detected" routes to `save_result` as `MANUAL_REVIEW`
or `ERROR`.

### B. Rewrite `TranscriptionService` to consume segments and load PEFT adapters

- Loading: when `WhisperSettings.adapter_path` is set, the service loads
  `WhisperSettings.base_model_id` (default `openai/whisper-large-v3`), then
  applies the adapter via `PeftModel.from_pretrained(...).merge_and_unload()`
  in memory. When unset, `model_path` is treated as a full merged checkpoint.
- Inference: one `model.generate(features, output_scores=True, ...)` call per
  **batch** of segments (`WhisperSettings.batch_size`, default 4). No HF
  pipeline wrapper, no long-form chunking.
- Confidence: per-segment `exp(mean token log-prob)`; the call-level
  `confidence_score` is the mean across non-empty segments.
- Output: a reconstructed dialogue string with `[Agent]` / `[Subscriber]`
  speaker tags and `MM:SS` timestamps, plus a `segments` list with text and
  confidence for each.

### C. Persist the adapter version

`CallAnalysisResult.whisper_adapter_version` records the adapter folder name
(or `""` for merged checkpoints) so eval / debugging can attribute outputs to
a specific training run.

## Consequences

- **Pro:** inference shape now matches training shape — segments ≤ 30 s, one
  speaker per segment. The 27 % WER measurement carries over to production.
- **Pro:** speaker labels become deterministic (from channel routing). The
  Gemini refinement prompt no longer needs a "guess speaker" instruction; the
  label is provided.
- **Pro:** swapping adapters is a config change. New LoRA version → set
  `WHISPER_ADAPTER_PATH` and restart. No rebuild.
- **Pro:** no `bitsandbytes` required by default. Production runs fp16 merged.
- **Pro:** confidence is now a mean over per-segment real log-probs, not a
  sample of the first 30 s.
- **Con:** a new dependency (`peft`) is exercised at runtime, plus
  `silero-vad` weights pulled via `torch.hub`. The latter needs one
  internet-enabled warm-up unless cached locally (`SILERO_CACHE_DIR`).
- **Con:** initial GPU init does base model (~3 GB) + adapter (~35 MB) +
  merge. ~10–30 s on a typical card. Pipeline retains a service for the
  lifetime of the process.
- **Con:** the previous `data/chunks/` debug output is replaced by
  `data/segments/<basename>/`, gated behind `PREPROCESSING_SAVE_SEGMENTS`.

## Alternatives considered

- **Path A — merge LoRA at deploy time, keep the long-form pipeline.** Smaller
  diff, but doesn't fix the train/inference shape mismatch. Rejected as a
  primary path; supported as a fallback when `adapter_path` is unset.
- **Stay on the HF pipeline with `chunk_length_s=30, stride_length_s=2`.**
  Continues to feed mixed-speaker audio with arbitrary boundaries to the model.
  Rejected.
- **8-bit (bitsandbytes) inference by default.** Added cost, slower than fp16
  on most modern GPUs. Kept as an opt-in via `WHISPER_USE_8BIT=1`.

## Related runbooks

- `docs/runbooks/whisper-model-swap.md` — swapping adapters and base models.
- `docs/runbooks/threshold-tuning.md` — re-tune `confidence_threshold` after
  the scoring distribution shifts post-migration.
