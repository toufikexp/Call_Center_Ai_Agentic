# Runbook — vLLM server lifecycle

The pipeline depends on a running vLLM server for classification and sentiment.
This runbook describes the expected operational shape; the server itself is
deployed and managed outside this repo.

## Expected service contract

| Property      | Default                       | Env var to override   |
|---------------|-------------------------------|-----------------------|
| Endpoint      | `http://localhost:8080/v1`    | `VLLM_BASE_URL`       |
| Model name    | `Qwen/Qwen3-4B`               | `VLLM_MODEL_NAME`     |
| Auth          | none                          | `VLLM_API_KEY`        |
| Temperature   | `0.1`                         | `VLLM_TEMPERATURE`    |

The endpoint must speak the OpenAI Chat Completions API (`/v1/chat/completions`)
since the codebase uses the `openai` Python client.

## Health check

```bash
# 1. List models
curl -s "$VLLM_BASE_URL/models" | jq

# 2. Smoke test a chat completion
curl -s "$VLLM_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
        \"model\": \"$VLLM_MODEL_NAME\",
        \"messages\": [{\"role\": \"user\", \"content\": \"reply with {\\\"ok\\\": true}\"}],
        \"temperature\": 0.0
      }" | jq
```

If both calls succeed, the pipeline will be able to talk to vLLM.

## Common reference launch (vanilla vLLM)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8080 \
  --served-model-name Qwen/Qwen3-4B
```

`--served-model-name` must match `VLLM_MODEL_NAME` exactly (the pipeline
passes that string verbatim in `model=` of the chat completion call).

## Swapping models

1. Stop vLLM.
2. Restart with a new `--model` and matching `--served-model-name`.
3. Update `VLLM_MODEL_NAME` in the pipeline's `.env` to the new served name.
4. No code changes required.

## When vLLM goes down

- Classification and sentiment will raise `RuntimeError` from the service
  layer.
- Orchestrator promotes the `RuntimeError` to `status = ERROR` and writes the
  partial JSON to `data/results/`.
- See `docs/runbooks/incident-response.md` → "pipeline ends in ERROR".

## Notes

- Lower temperatures (≤ 0.2) are required for the JSON-following behavior the
  classification and sentiment prompts assume. Don't raise it without
  re-validating the prompts.
- The pipeline does not retry on vLLM failure. Repeated calls will produce
  repeated ERROR records; fix the server first, then re-run the affected
  audio files.
