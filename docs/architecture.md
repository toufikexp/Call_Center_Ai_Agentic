# Architecture

This document describes the runtime architecture, data flow, and component
boundaries of the Call Center AI Agentic Pipeline.

## High-level overview

The pipeline is an audio-in / JSON-out system that transforms a single call
recording into a structured analytics record. Orchestration is performed by
LangGraph; each stage is a self-contained service.

```
            ┌─────────────────────────────────────────────────────────┐
            │                    main.py (CLI)                        │
            │   parses args, loads Settings, calls pipeline.run()     │
            └────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
            ┌─────────────────────────────────────────────────────────┐
            │           CallAnalysisPipeline (LangGraph)              │
            │              src/pipeline/orchestrator.py               │
            └────────────────────────┬────────────────────────────────┘
                                     │
 ┌────────────┬─────────────┬────────┴────┬─────────┬───────────────┬─────────┬─────────┐
 ▼            ▼             ▼             ▼         ▼               ▼         ▼         ▼
Preprocess  Transcribe   Refine        Verify    Classify       Sentiment   SaveResult
(stereo     (Whisper +   (Gemini API   (gating   (vLLM /        (vLLM /     (writes
 split,      LoRA, per-   + JSON)       node)     Qwen3-4B)      Qwen3-4B)   JSON)
 Silero      segment
 VAD)        batched)
```

## Pipeline graph

Built in `_build_graph()` in `src/pipeline/orchestrator.py`. Edges:

```
START
  │
  ▼
preprocess ──(error)──► save_result
  │
  ▼
transcribe ──► refine ──► verify
                              │
                              │  (conditional: _route_decision)
                              │
              ┌───────────────┼──────────────────┐
              │               │                  │
        proceed                          manual_review
              │                                  │
              ▼                                  │
          classify ──(error)──► save_result      │
              │                    ▲             │
              ▼                    │             │
       analyze_sentiment ──────────┘             │
              │                                  │
              └────────────► save_result ◄───────┘
                                  │
                                  ▼
                                 END
```

### Routing logic

- `_route_decision` (after `verify`):
  - `result.status == ERROR` → `manual_review` (saves with ERROR status)
  - `refinement_score < pipeline.refinement_threshold` (default `0.5`) →
    `manual_review`
  - `confidence_score < pipeline.confidence_threshold` (default `0.9`) →
    `manual_review`
  - otherwise → `proceed` (continues to `classify`)
- `_check_error` (after `classify` and `analyze_sentiment`):
  - `result.status == ERROR` → `save_result` (short-circuits remaining stages)

### Final-status assignment

Performed in `_save_node`:
1. If already `ERROR`, status is preserved.
2. Else if `refinement_score < refinement_threshold` → `MANUAL_REVIEW`.
3. Else if `confidence_score < confidence_threshold` → `MANUAL_REVIEW`.
4. Else → `COMPLETE`.

## Component boundaries

```
┌──────────────────────────────────────────────────────────────────┐
│                        src/core/                                  │
│                                                                   │
│   base.py        BaseService, ServiceResult, _execute_with_timing │
│   state.py       PipelineState (TypedDict), CallAnalysisResult,   │
│                  ProcessingStatus (Enum)                          │
│   config.py      Re-exports src/config/config.py for back-compat  │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │  imports
                              │
┌─────────────────────────────┴────────────────────────────────────┐
│                        src/services/                              │
│                                                                   │
│   preprocessing.py    PreprocessingService → soundfile + Silero   │
│   transcription.py    TranscriptionService  → Whisper + LoRA      │
│   refinement.py       RefinementService     → Gemini API (cloud)  │
│   correction.py       CorrectionService     → Qwen (DISABLED)     │
│   classification.py   ClassificationService → vLLM (local)        │
│   sentiment.py        SentimentService      → vLLM (local)        │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │  imports
                              │
┌─────────────────────────────┴────────────────────────────────────┐
│                        src/pipeline/                              │
│                                                                   │
│   orchestrator.py     CallAnalysisPipeline (LangGraph StateGraph) │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │  imports
                              │
┌─────────────────────────────┴────────────────────────────────────┐
│                            main.py                                │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        src/config/                                │
│                                                                   │
│   config.py                       Settings + Pydantic sub-models  │
│   classification_schema.json      Taxonomy (categories/sub-cats)  │
└──────────────────────────────────────────────────────────────────┘
   ▲
   │ loaded by all layers via get_settings()
```

### Layering rules

- `core` depends on no other in-repo package.
- `services` depend only on `core` and external SDKs.
- `pipeline` depends on `core` + `services`.
- `main.py` depends on `pipeline` + `config`.
- **No circular imports.** No service imports another service. Services
  communicate only via the orchestrator and the shared `PipelineState`.

## Data flow

State carried through the graph (`PipelineState`):

```python
{
    "audio_path": str,         # input file path, set once at run() time
    "call_id":    str,         # generated if not provided
    "run_count":  int,         # incremented by transcribe node
    "result":     CallAnalysisResult  # mutated by every node
}
```

Per-stage transformation of `CallAnalysisResult`:

| Stage           | Reads                          | Writes                                                    |
|-----------------|--------------------------------|-----------------------------------------------------------|
| preprocess      | `audio_path`                   | `state["segments"]` (channel-split + VAD-cut clips)                            |
| transcribe      | `state["segments"]`            | `transcript` (reconstructed dialogue), `confidence_score`, `segments`, `whisper_adapter_version` |
| refine          | `transcript`                   | `refined_transcript`, `refinement_score`                                       |
| verify          | scores                         | `status` → `IN_PROGRESS` (logging only)                                        |
| classify        | `refined_transcript or transcript` | `subject`, `sub_subject`, `classification_confidence`                      |
| analyze_sentiment | `refined_transcript or transcript` | `satisfaction_score`, `sentiment_label`, `sentiment_reasoning`           |
| save_result     | everything                     | final `status`; writes JSON to `data/results/`                                 |

## External dependencies (runtime)

| Component       | Where it runs       | Used by             | Failure mode                                    |
|-----------------|---------------------|---------------------|-------------------------------------------------|
| Whisper model   | local GPU/CPU       | TranscriptionService| `initialize()` raises if model path invalid     |
| Gemini API      | Google cloud        | RefinementService   | refinement_score=0.0; pipeline → MANUAL_REVIEW  |
| vLLM server     | local network       | Classification + Sentiment | `RuntimeError` → status=ERROR              |

## On-disk layout (runtime)

```
data/
├── audio_files/    # inputs (gitignored)
├── segments/       # optional per-segment WAVs for VAD debugging (gitignored)
├── results/        # output JSON, one per call
└── logs/           # reserved (file logging not enabled by default)

models/             # local checkpoints / adapters (gitignored)
```

## Performance / scaling notes

- Single-call, single-process design. No batching, no queue, no retries beyond
  `max_retry_attempts` (currently unused because Correction is disabled).
- Whisper handles long audio via HF pipeline's internal chunking
  (`chunk_length_s=30`, `stride_length_s=2.0`).
- Models are cached for the process lifetime (loaded on first call); re-using
  the same `CallAnalysisPipeline` instance across calls amortizes load cost.
