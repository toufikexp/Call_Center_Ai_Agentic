# Runbook — Incident response

Symptom-driven triage for the most common failure modes. All evidence is in
the logs; if you don't see a service-prefixed log line for a stage, that stage
didn't run.

## Symptom: every call ends in `MANUAL_REVIEW`

1. Open one of the recent JSON files in `data/results/`. Check
   `refinement_score`.
2. If `refinement_score == 0.0`:
   - Check the orchestrator logs for `Refinement failed: ...` or
     `GEMINI_API_KEY not found`.
   - Possible causes:
     - `.env` missing or `GEMINI_API_KEY` empty → see
       `runbooks/configuration.md`.
     - Gemini API outage / network egress blocked → confirm with
       `curl https://generativelanguage.googleapis.com`.
     - Quota exceeded → check Google Cloud console for the key in use.
3. If `confidence_score < 0.9` but `refinement_score >= 0.5`:
   - Whisper transcripts are too short for the heuristic. Either input audio
     is very brief (silent / clipped) or the threshold needs tuning. See
     `runbooks/threshold-tuning.md`.

## Symptom: pipeline ends in `ERROR`

Check `error_message` in the result JSON.

| `error_message` substring                  | Likely cause                                                                                  |
|--------------------------------------------|-----------------------------------------------------------------------------------------------|
| `Audio file not found`                     | Bad CLI argument; verify the path.                                                            |
| `Refinement API connection error`          | Gemini network failure. Retry; if persistent, see Gemini status page.                         |
| `Classification JSON parse error` /  `Classification service error` | vLLM returned non-JSON or vLLM is unreachable. See `runbooks/vllm-server.md`.       |
| `Sentiment JSON parse error` / `Sentiment analysis service error` | Same as classification — usually a vLLM issue.                                       |

## Symptom: pipeline crashes at startup with a Whisper error

Most common: the hard-coded `WhisperSettings.model_path` in
`src/config/config.py` doesn't exist on this machine.

1. Confirm the path exists and contains a Whisper checkpoint
   (`config.json`, `model.safetensors` or sharded weights, `generation_config.json`).
2. If not, switch to one of the alternatives commented in the same block, or
   point to a public model like `openai/whisper-large-v3`.
3. After editing, re-run. The model loads on first call (lazy init).

## Symptom: classifications all come back as `OTHER`

1. Verify vLLM is up: `curl $VLLM_BASE_URL/models` (with whatever auth your
   vLLM expects).
2. Check the model name in vLLM matches `VLLM_MODEL_NAME`. The OpenAI client
   passes the name through; vLLM rejects unknown names.
3. Inspect classification logs — `vLLM returned invalid category 'X'` means
   the model is responding but not following the JSON contract. Lower
   `VLLM_TEMPERATURE` (default `0.1`); confirm the `system` prompt is intact
   in `ClassificationService._build_classification_prompt`.

## Symptom: results aren't being saved

Check that `data/results/` exists and is writeable. The save node creates it
if missing; permission failures will surface as a stack trace. The `_save_node`
runs even on `ERROR` so a missing JSON file always means the process died
earlier — search logs for the last service prefix that ran.

## Always-do checks

- Tail the latest logs as you run: pipeline + per-service prefixes are
  searchable (`pipeline.orchestrator`, `service.transcription`,
  `service.refinement`, `service.classification`, `service.sentiment`).
- Each service prints a summary block at the end of its work; if a stage's
  block is missing, it didn't run (or crashed before its summary).
