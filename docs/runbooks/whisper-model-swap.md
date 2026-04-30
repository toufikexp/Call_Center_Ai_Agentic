# Runbook — Swapping the Whisper transcription model

The Whisper model used for transcription is identified by
`WhisperSettings.model_path` in `src/config/config.py`. Several historical
checkpoints are commented in that file as a quick-switch reference.

## When to swap

- A new fine-tuning run produced a better checkpoint.
- The current path doesn't exist on the deployment machine.
- You want to baseline against `openai/whisper-large-v3`.

## Procedure

1. Identify the new model location:
   - Local directory containing a Whisper checkpoint
     (`config.json`, `generation_config.json`, weights), **or**
   - A Hugging Face Hub model id (e.g. `openai/whisper-large-v3`).
2. Edit `src/config/config.py`:
   ```python
   whisper=WhisperSettings(
       model_path="<NEW PATH OR HF ID>",
       ...
   ),
   ```
3. If the new path contains numbered `checkpoint-*` subdirectories, you can
   point `model_path` at the parent — `_find_latest_checkpoint` will pick
   the most recently modified subdirectory automatically.
4. Confirm the base tokenizer is still compatible. The service hard-codes
   `openai/whisper-large` as `tokenizer` and `feature_extractor`
   (see `transcription.py`). For non-large checkpoints (e.g. `whisper-base`)
   change that base in code as part of the swap.
5. Run a smoke test with a known-good audio file:
   ```bash
   python main.py data/audio_files/<sample>.wav
   ```
   Expect:
   - log line `✅ Whisper Pipeline initialized on <DEVICE>`
   - non-empty transcript in the result JSON
   - `confidence_score >= 0.7`

## Calibration

The `confidence_score` is a length-based heuristic
(`recalculate_confidence`), not a real model score. After swapping models,
spot-check whether the same threshold (`0.9`) still makes sense for the
language and audio quality you process. If it's now systematically gating too
many calls, see `docs/runbooks/threshold-tuning.md`.

## Rollback

Editing `model_path` is the only change; revert the line and restart.
Cached chunks in `data/chunks/` are model-agnostic — no cleanup needed.
