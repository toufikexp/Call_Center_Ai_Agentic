# ADR 0002 — On-premise execution with cloud-only refinement

- Status: Accepted

## Context

The pipeline processes Algerian telecom call recordings — subject to Algerian
data protection rules. Defaulting to cloud APIs for transcription, classification,
and sentiment is non-starter. At the same time, refining noisy Whisper output
into clean Darija/French dialogue benefits significantly from a strong frontier
model, where local LLMs underperform.

## Decision

- **Transcription** runs locally via fine-tuned Whisper (`TranscriptionService`).
- **Classification** and **Sentiment** run locally via vLLM serving Qwen3-4B
  (`ClassificationService`, `SentimentService`). The OpenAI-compatible vLLM
  endpoint allows swapping models without code changes.
- **Refinement** is the *only* cloud dependency, calling
  `gemini-2.0-flash-exp` (`RefinementService`). When unavailable, the pipeline
  routes to manual review rather than producing low-quality outputs downstream.

## Consequences

- **Pro:** raw audio and full transcripts never leave the on-prem environment
  except in the refinement call.
- **Pro:** vLLM endpoint is swappable (different model, different deployment)
  via `VLLM_*` env vars without code changes.
- **Con:** refinement is a single cloud dependency. If/when policy tightens
  further, it must be replaced with a local equivalent (e.g. a larger local
  LLM with a Darija-specific prompt).
- **Con:** the Gemini SDK and prompt are coupled into `RefinementService` —
  swapping providers means editing that service. Acceptable today.

## Alternatives considered

- **All-local refinement with Qwen3-4B** — rejected for now; quality of
  speaker labelling and Whisper-hallucination cleanup was insufficient.
- **All-cloud pipeline (e.g. AssemblyAI / Whisper API + GPT-4o)** — rejected
  for compliance.
