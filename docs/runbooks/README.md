# Runbooks

Operational procedures for the most common tasks and failure modes.

| Runbook                                                | When to read                                               |
|--------------------------------------------------------|------------------------------------------------------------|
| [Configuration & environment](configuration.md)        | New machine, new deployment, or "is my .env right?"        |
| [vLLM server lifecycle](vllm-server.md)                | vLLM is down, swapping models, sanity-checking the server  |
| [Whisper model swap](whisper-model-swap.md)            | New checkpoint, hard-coded path missing on a new machine   |
| [Taxonomy changes](taxonomy-changes.md)                | Adding/renaming/removing categories or sub-categories      |
| [Threshold tuning](threshold-tuning.md)                | Too many `MANUAL_REVIEW`s, or quality concerns             |
| [Incident response](incident-response.md)              | Symptom-driven triage when something is broken             |
