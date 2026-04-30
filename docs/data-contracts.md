# Data contracts

Schema in/out at every stage of the pipeline. This is the contract surface —
breaking changes here ripple to consumers of `data/results/*.json`.

## End-to-end

```
audio file (mp3 / wav)
        │
        ▼
PreprocessingService.process(audio_path)
        │  ServiceResult.data = {"segments": [{"channel", "start_ms", "end_ms", "audio"}, ...]}
        ▼
TranscriptionService.process(segments)
        │  ServiceResult.data = {
        │      "transcript": str,         # reconstructed dialogue with speaker tags
        │      "segments":   [{"channel", "start_ms", "end_ms", "text", "confidence"}, ...],
        │      "confidence": float,       # mean per-segment confidence
        │      "adapter_version": str,
        │  }
        ▼
RefinementService.process(transcript)
        │  ServiceResult.data = {"refined_transcript": str, "refinement_score": float}
        ▼
ClassificationService.process(transcript_or_refined)
        │  ServiceResult.data = {"subject": str, "sub_subject": str, "confidence": float}
        ▼
SentimentService.process(transcript_or_refined)
        │  ServiceResult.data = {"satisfaction_score": float}
        ▼
data/results/<basename>_<id>_result.json   (CallAnalysisResult)
```

## ServiceResult (envelope)

Defined in `src/core/base.py`.

| Field            | Type           | Notes                                      |
|------------------|----------------|--------------------------------------------|
| `success`        | `bool`         | `False` if the wrapped function raised     |
| `data`           | `Any`          | Stage-specific dict; `None` on failure     |
| `error`          | `str \| None`  | Error message when `success=False`         |
| `execution_time` | `float`        | Seconds, set by `_execute_with_timing`     |
| `metadata`       | `dict \| None` | Reserved for future use                    |

## Stage contracts

### 0. Preprocessing

**Input:** path to audio file. Supported formats: anything `soundfile.read`
accepts (wav, flac, ogg) plus mp3/m4a (via the same install of `libsndfile`).
Stereo recordings are split into agent / client channels; mono recordings
get a single `unknown` channel.

**Output `data` dict:**

| Key        | Type        | Notes                                                                                                |
|------------|-------------|------------------------------------------------------------------------------------------------------|
| `segments` | `list[dict]` | Each entry: `{"channel": "agent"\|"client"\|"unknown", "start_ms": int, "end_ms": int, "audio": np.ndarray}` |

Segments are sorted chronologically across channels. `audio` is float32 mono
at 16 kHz. Segments shorter than `min_segment_seconds` or longer than
`max_segment_seconds` are dropped; each kept segment is padded by
`padding_ms` on each side.

**Side effect:** when `PreprocessingSettings.save_segments` is `true`, each
segment is also written to `data/segments/<basename>/<basename>_<channel>_<i>_start_<ms>ms.wav`
for inspection. Off by default.

### 1. Transcription

**Input:** the `segments` list emitted by preprocessing.

**Output `data` dict:**

| Key                | Type        | Range / domain                                                                  |
|--------------------|-------------|---------------------------------------------------------------------------------|
| `transcript`       | `str`       | Speaker-tagged dialogue, lines like `"00:05 [Agent]: ..."`                      |
| `segments`         | `list[dict]` | Per-segment `{"channel", "start_ms", "end_ms", "text", "confidence"}`           |
| `confidence`       | `float`     | 0.0 – 1.0; mean of per-segment `exp(mean token log-prob)` over non-empty rows   |
| `adapter_version`  | `str`       | LoRA adapter directory name, or `""` for full merged checkpoints                |

The service runs `model.generate()` once per **batch** of segments (size
configurable via `WhisperSettings.batch_size`), with `output_scores=True`
to compute per-segment confidence. Long-form chunking is **not** used —
each segment is already ≤ `max_segment_seconds`.

### 2. Refinement

**Input:** raw transcript string.

**Output `data` dict:**

| Key                  | Type    | Range / domain                                        |
|----------------------|---------|-------------------------------------------------------|
| `refined_transcript` | `str`   | Speaker-tagged with `[Agent]` / `[Subscriber]`        |
| `refinement_score`   | `float` | 0.0 – 1.0; `0.0` means refinement skipped or failed   |

**Failure modes that still return `success=True`:**
- No `GEMINI_API_KEY` — `refined_transcript` falls back to the input,
  score `0.0`.
- Refined text shorter than `min_refinement_length_ratio * len(input)` —
  fallback as above.

**Hard failures (`success=False`, raises in service):** network errors during
the Gemini call.

**Prompt contract:** Gemini must reply with strict JSON
`{"refined_text": "...", "score": 0.0-1.0}`. The service strips ```` ```json ```` fences
before parsing.

### 3. Classification

**Input:** transcript (preferring `refined_transcript`).

**Output `data` dict:**

| Key           | Type    | Range / domain                                                |
|---------------|---------|---------------------------------------------------------------|
| `subject`     | `str`   | One of `ClassificationSettings.primary_categories`            |
| `sub_subject` | `str`   | One of allowed sub-categories for the subject, or `"N/A"`     |
| `confidence`  | `float` | 0.0 – 1.0                                                     |

**Validation in service:**
- Unknown `subject` → coerced to `OTHER` (configurable via
  `other_category_name`), `sub_subject` → `N/A`, `confidence` → `0.0`.
- Unknown `sub_subject` → tries case-insensitive / whitespace-tolerant match
  against allowed list; falls back to `default_subcategory[subject]` or first
  non-N/A entry; else `"N/A"`.
- If `subject == OTHER`, `sub_subject` is forced to `"N/A"`.

**Prompt contract (vLLM):** model must reply with strict JSON
`{"subject": "...", "sub_subject": "...", "confidence": 0.0-1.0}`.

**Taxonomy source:** `src/config/classification_schema.json`. See that file
for the authoritative list. Top-level keys:

```
{
  "primary_categories":       [str, ...],
  "category_descriptions":    { category: description },
  "category_subcategories":   { category: [subcat, ...] },
  "subcategory_descriptions": { category: { subcat: description } },
  "default_subcategory":      { category: subcat }
}
```

### 4. Sentiment

**Input:** transcript (preferring `refined_transcript`).

**Output `data` dict:**

| Key                  | Type    | Range / domain                                              |
|----------------------|---------|-------------------------------------------------------------|
| `satisfaction_score` | `float` | 1.0 – 10.0 on success; `0.0` means "not analyzed"           |
| `sentiment_label`    | `str`   | `POSITIVE`, `NEUTRAL`, `NEGATIVE`, or `""` if not analyzed  |
| `confidence`         | `float` | 0.0 – 1.0                                                   |
| `reasoning`          | `str`   | One-sentence justification (or `""` if not analyzed)        |

**Prompt contract (vLLM):** model must reply with JSON
`{"satisfaction_score": float, "sentiment_label": "POSITIVE|NEUTRAL|NEGATIVE",
"confidence": float, "reasoning": "..."}`. `satisfaction_score`,
`sentiment_label`, and `reasoning` are persisted in the final record;
`confidence` is logged only.

## Final output JSON (`data/results/*.json`)

Schema = `CallAnalysisResult.model_dump()` plus a few injected fields.

| Field                | Type             | Notes                                                                 |
|----------------------|------------------|-----------------------------------------------------------------------|
| `call_id`                   | `str`            | `call_<basename>_<8-hex>` if not provided                             |
| `transcript`                | `str`            | Reconstructed dialogue (speaker-tagged, chronological)                 |
| `refined_transcript`        | `str`            | Possibly equal to `transcript` if refinement skipped/failed           |
| `confidence_score`          | `float [0, 1]`   | Mean per-segment `exp(mean token log-prob)`                            |
| `refinement_score`          | `float [0, 1]`   | From Gemini, or `0.0`                                                 |
| `subject`                   | `str`            | Taxonomy category or `"OTHER"` / `"UNKNOWN"`                          |
| `sub_subject`               | `str`            | Sub-category or `"N/A"`                                               |
| `classification_confidence` | `float [0, 1]`   | Classifier self-reported confidence                                   |
| `satisfaction_score`        | `float [0, 10]`  | `0.0` means not analyzed                                              |
| `sentiment_label`           | `str`            | `POSITIVE` / `NEUTRAL` / `NEGATIVE` / `""`                            |
| `sentiment_reasoning`       | `str`            | One-sentence justification, or `""`                                   |
| `segments`                  | `list[object]`   | Per-segment `{channel, start_ms, end_ms, text, confidence}`           |
| `whisper_adapter_version`   | `str`            | LoRA adapter folder name, or `""` for merged checkpoints              |
| `status`                    | `str`            | One of: `PENDING`, `IN_PROGRESS`, `COMPLETE`, `MANUAL_REVIEW`, `ERROR` |
| `error_message`             | `str \| null`    | Set when `status == ERROR`                                            |
| `audio_path`                | `str`            | Injected in `_save_node`                                              |
| `run_count`                 | `int`            | Injected in `_save_node` (always 1 today)                             |

### Example: COMPLETE

```json
{
  "call_id": "call_399001002190034_abc12345",
  "transcript": "...",
  "refined_transcript": "[Agent] ... [Subscriber] ...",
  "confidence_score": 0.92,
  "refinement_score": 0.85,
  "subject": "Network",
  "sub_subject": "Network_Reception/Coverage",
  "classification_confidence": 0.95,
  "satisfaction_score": 7.5,
  "sentiment_label": "POSITIVE",
  "sentiment_reasoning": "Customer thanked the agent and the issue was resolved.",
  "status": "COMPLETE",
  "error_message": null,
  "audio_path": "data/audio_files/399001002190034.mp3",
  "run_count": 1
}
```

### Example: MANUAL_REVIEW (low refinement score)

```json
{
  "call_id": "call_xyz_12345678",
  "transcript": "...",
  "refined_transcript": "...",
  "confidence_score": 0.95,
  "refinement_score": 0.0,
  "subject": "",
  "sub_subject": "",
  "classification_confidence": 0.0,
  "satisfaction_score": 0.0,
  "sentiment_label": "",
  "sentiment_reasoning": "",
  "status": "MANUAL_REVIEW",
  "error_message": null,
  "audio_path": "...",
  "run_count": 1
}
```

### Example: ERROR

```json
{
  "call_id": "call_xyz_12345678",
  "transcript": "Error: Audio file not found: ...",
  "refined_transcript": "",
  "confidence_score": 0.0,
  "refinement_score": 0.0,
  "subject": "",
  "sub_subject": "",
  "classification_confidence": 0.0,
  "satisfaction_score": 0.0,
  "sentiment_label": "",
  "sentiment_reasoning": "",
  "status": "ERROR",
  "error_message": "Audio file not found: ...",
  "audio_path": "...",
  "run_count": 1
}
```

## Status state machine

```
                 ┌─────────────┐
                 │   PENDING   │ (initial)
                 └──────┬──────┘
                        │
                  transcribe / refine / verify
                        │
                        ▼
                 ┌─────────────┐
                 │ IN_PROGRESS │
                 └──────┬──────┘
                        │
       ┌────────────────┼─────────────────────────────────┐
       │                │                                 │
   any error       low refinement OR low confidence    all gates pass
       │                │                                 │
       ▼                ▼                                 ▼
   ┌───────┐     ┌───────────────┐                   ┌──────────┐
   │ ERROR │     │ MANUAL_REVIEW │                   │ COMPLETE │
   └───────┘     └───────────────┘                   └──────────┘
```
