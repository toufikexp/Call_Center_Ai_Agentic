# ADR 0004 — Serve Qwen3-4B via vLLM, not in-process

- Status: Accepted (replaces an earlier in-process Qwen approach)

## Context

The classification and sentiment stages need a local LLM. An earlier
iteration loaded Qwen-7B-Chat directly into the Python process via
`transformers.AutoModelForCausalLM` (see `CorrectionService`). That worked
but had downsides:

- ~14 GB GPU resident even when idle, blocking other workloads.
- Cold-load on each pipeline boot, slowing batch runs.
- One service per Python process meant two stages couldn't share weights.

## Decision

Run a single vLLM server on `http://localhost:8080/v1` exposing
`Qwen/Qwen3-4B` over the OpenAI-compatible API. Both
`ClassificationService` and `SentimentService` connect via the `openai`
client. Configuration lives under `VLLMSettings` and is overridable via
`VLLM_BASE_URL`, `VLLM_MODEL_NAME`, `VLLM_API_KEY`, `VLLM_TEMPERATURE`.

`CorrectionService` retains the in-process `transformers` pattern but is
currently disabled in the orchestrator.

## Consequences

- **Pro:** model weights are loaded once for the lifetime of the vLLM server,
  shared by both stages and by any other client (eval scripts, etc.).
- **Pro:** swapping models is an ops task (restart vLLM with a different
  model), not a code change.
- **Pro:** OpenAI-compatible API means the codebase is portable to any
  OpenAI-compatible endpoint (TGI, OpenAI itself, etc.) by changing env vars.
- **Con:** introduces a network hop and an external process to keep alive.
  The pipeline depends on the vLLM server being up.
- **Con:** when vLLM is unreachable, classification/sentiment fail with
  `RuntimeError`, which the orchestrator promotes to `status = ERROR`.

## Operational note

vLLM lifecycle is out of scope for this repo. See
`docs/runbooks/vllm-server.md` for the expected operational model.
