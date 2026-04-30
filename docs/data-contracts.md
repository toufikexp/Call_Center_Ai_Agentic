# Data contracts

Schema in/out at every stage of the pipeline. This is the contract surface —
breaking changes here ripple to consumers of `data/results/*.json`.

## End-to-end

```
audio file (mp3 / wav)
        │
        ▼
TranscriptionService.process(audio_path, sample_rate)
        │  ServiceResult.data = {"transcript": str, "confidence": float}
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

### 1. Transcription

**Input:** path to audio file. Supported formats: anything `librosa.load`
accepts (mp3, wav, m4a, …).

**Output `data` dict:**

| Key          | Type    | Range / domain                                                                 |
|--------------|---------|--------------------------------------------------------------------------------|
| `transcript` | `str`   | Possibly empty                                                                  |
| `confidence` | `float` | 0.0 – 1.0; `exp(mean token log-prob)` from a direct `model.generate()` pass    |

**Side effect:** for audio longer than `chunk_length_seconds` (30s), per-chunk
WAV files are written to `data/chunks/<basename>_chk<NN>.wav`.

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
| `transcript`                | `str`            | Raw Whisper output                                                    |
| `refined_transcript`        | `str`            | Possibly equal to `transcript` if refinement skipped/failed           |
| `confidence_score`          | `float [0, 1]`   | `exp(mean token log-prob)` from Whisper                               |
| `refinement_score`          | `float [0, 1]`   | From Gemini, or `0.0`                                                 |
| `subject`                   | `str`            | Taxonomy category or `"OTHER"` / `"UNKNOWN"`                          |
| `sub_subject`               | `str`            | Sub-category or `"N/A"`                                               |
| `classification_confidence` | `float [0, 1]`   | Classifier self-reported confidence                                   |
| `satisfaction_score`        | `float [0, 10]`  | `0.0` means not analyzed                                              |
| `sentiment_label`           | `str`            | `POSITIVE` / `NEUTRAL` / `NEGATIVE` / `""`                            |
| `sentiment_reasoning`       | `str`            | One-sentence justification, or `""`                                   |
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
